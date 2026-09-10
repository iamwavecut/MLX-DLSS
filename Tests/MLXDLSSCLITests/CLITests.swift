import CoreGraphics
import Foundation
import ImageIO
import DLSSMLX
import XCTest
@testable import mlxdlss

final class CLITests: XCTestCase {
  func testRunSequenceRejectsUnknownNetworkGeometry() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-temporal-reference",
      "--network-geometry", "loose",
      "--input", "/tmp/frame.f32",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("network geometry must be 'vendor-aligned' or 'match-output'"),
      result.stderr
    )
  }

  func testRunSequenceRejectsNetworkGeometryWithoutTemporalInput() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--network-geometry", "match-output",
      "--input", "/tmp/frame.f32",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("motion options require rgb-temporal-reference input"),
      result.stderr
    )
  }

  func testStreamRequiresModelWidthAndHeightAndKnownOptions() throws {
    let missing = try runCLI(["stream"])
    XCTAssertEqual(missing.status, 2)
    XCTAssertTrue(missing.stderr.contains("stream requires a model-package path"), missing.stderr)
    let extent = try runCLI(["stream", "/definitely/missing/model.dlssmodel", "--width", "0", "--height", "16"])
    XCTAssertEqual(extent.status, 2)
    XCTAssertTrue(extent.stderr.contains("positive --width and --height"), extent.stderr)
    let unknown = try runCLI(["stream", "/definitely/missing/model.dlssmodel", "--width", "16", "--height", "16", "--layout", "x"])
    XCTAssertEqual(unknown.status, 2)
    XCTAssertTrue(unknown.stderr.contains("unknown stream option"), unknown.stderr)
    let scaled = try runCLI(["stream", "/definitely/missing/model.dlssmodel", "--width", "16", "--height", "16", "--processing-scale", "2"])
    XCTAssertEqual(scaled.status, 2)
    XCTAssertTrue(scaled.stderr.contains("native scale"), scaled.stderr)
  }

  func testStreamRejectsUnknownProtocolAndNonfiniteBlendScaleBeforeOpeningModel() throws {
    for options in [
      ["--protocol-version", "5"], ["--protocol-version", "x"], ["--blend-scale", "nan"],
    ] {
      let result = try runCLI(
        ["stream", "/definitely/missing/model.dlssmodel", "--width", "1", "--height", "1"] + options
      )
      XCTAssertEqual(result.status, 2)
      XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
      XCTAssertTrue(result.stderr.contains(options[0]), result.stderr)
    }
  }

  func testStreamFrameDecoderVersionedConfidenceAndOrdering() throws {
    var payload = streamPayload(flags: 3, values: [0.25, 0.5, 0.75, -0.1, 0.2, 1, 0.375])
    payload.append(streamPayload(flags: 0, values: [0.75, 0.5, 0.25, 0, 0, 1]))
    try withInputPipe(payload) { handle in
      let first = try XCTUnwrap(
        StreamCommand.readFrame(
          from: handle, width: 1, height: 1, temporal: true, protocolVersion: 2))
      XCTAssertEqual(first.flags, 3)
      XCTAssertEqual(
        first.inputs.map { $0.descriptor.name }, ["color", "motion", "depth", "historyConfidence"])
      XCTAssertEqual(first.inputs.last?.bytes, [Float(0.375)].withUnsafeBytes { Data($0) })
      let second = try XCTUnwrap(
        StreamCommand.readFrame(
          from: handle, width: 1, height: 1, temporal: true, protocolVersion: 2))
      XCTAssertEqual(second.inputs.count, 3)
      XCTAssertNil(
        try StreamCommand.readFrame(
          from: handle, width: 1, height: 1, temporal: true, protocolVersion: 2))
    }
  }

  func testStreamFrameDecoderPreservesLegacyLayoutsInBothVersions() throws {
    for version in [1, 2, 3] {
      for temporal in [false, true] {
        let values: [Float] = temporal ? [0.25, 0.5, 0.75, 0, 0, 1] : [0.25, 0.5, 0.75]
        try withInputPipe(streamPayload(flags: 1, values: values)) { handle in
          let frame = try XCTUnwrap(
            StreamCommand.readFrame(
              from: handle, width: 1, height: 1, temporal: temporal, protocolVersion: version))
          XCTAssertEqual(frame.flags, 1)
          XCTAssertEqual(
            frame.inputs.map { $0.descriptor.name },
            temporal ? ["color", "motion", "depth"] : ["color"])
          XCTAssertNil(
            try StreamCommand.readFrame(
              from: handle, width: 1, height: 1, temporal: temporal, protocolVersion: version))
        }
      }
    }
  }

  func testStreamFrameDecoderRejectsMalformedMapsAndTruncation() throws {
    let valid: [Float] = [0.25, 0.5, 0.75, 0, 0, 1, 0.5]
    var cases: [(Data, Bool, Int)] = [
      (streamPayload(flags: 2, values: valid), true, 1),
      (streamPayload(flags: 4, values: valid), true, 2),
      (streamPayload(flags: 2, values: valid), false, 2),
      (streamPayload(flags: 0, values: valid), true, 5),
      (streamPayload(flags: 24, values: valid), true, 3),
      (streamPayload(flags: 4, values: valid), false, 3),
      (streamPayload(flags: 32, values: valid), true, 3),
      (Data([0, 0]), true, 2),
    ]
    for value: Float in [.nan, .infinity, -.infinity, -0.1, 1.1] {
      var malformed = valid
      malformed[6] = value
      cases.append((streamPayload(flags: 2, values: malformed), true, 2))
    }
    for index in [0, 3, 5] {
      var malformed = valid
      malformed[index] = .nan
      cases.append((streamPayload(flags: 2, values: malformed), true, 2))
    }
    let frame = streamPayload(flags: 2, values: valid)
    for length in 4..<frame.count {
      cases.append((Data(frame.prefix(length)), true, 2))
    }
    for (payload, temporal, version) in cases {
      try withInputPipe(payload) { handle in
        XCTAssertThrowsError(
          try StreamCommand.readFrame(
            from: handle, width: 1, height: 1, temporal: temporal, protocolVersion: version))
      }
    }
  }

  func testCompactStreamDecodesExactIntegerColorAndConstantDepth() throws {
    for bits in [8, 16] {
      let values: [UInt16] = [0, 1, 127, 128, 254, 255]
      let integers = bits == 8 ? values : [0, 1, 255, 256, 32768, 65535]
      var payload = streamPayload(flags: UInt32(7 | (bits == 8 ? 8 : 16)), values: [])
      if bits == 8 {
        payload.append(contentsOf: integers.map { UInt8($0) })
      } else {
        payload.append(integers.map { $0.littleEndian }.withUnsafeBytes { Data($0) })
      }
      let guides: [Float] = [-0.25, 0.5, 0, 0, 0.125, 1]
      payload.append(guides.withUnsafeBytes { Data($0) })
      payload.append(streamPayload(flags: 4, values: [0.25, 0.5, 0.75, 1, 0, 0, 0, 0, 0, 0]))
      try withInputPipe(payload) { handle in
        let frame = try XCTUnwrap(StreamCommand.readFrame(
          from: handle, width: 2, height: 1, temporal: true, protocolVersion: 3))
        let expected = integers.map { Float($0) / (bits == 8 ? 255 : 65535) }
        XCTAssertEqual(frame.inputs[0].bytes, expected.withUnsafeBytes { Data($0) })
        XCTAssertEqual(frame.inputs.map { $0.descriptor.name }, ["color", "motion", "depth", "historyConfidence"])
        XCTAssertEqual(frame.inputs[2].bytes, [Float(1), 1].withUnsafeBytes { Data($0) })
        XCTAssertEqual(frame.inputs[3].bytes, [Float(0.125), 1].withUnsafeBytes { Data($0) })
        XCTAssertNotNil(try StreamCommand.readFrame(from: handle, width: 2, height: 1, temporal: true, protocolVersion: 3))
        XCTAssertNil(try StreamCommand.readFrame(from: handle, width: 2, height: 1, temporal: true, protocolVersion: 3))
      }
      for length in 4..<(4 + 6 * bits / 8 + guides.count * 4) {
        try withInputPipe(Data(payload.prefix(length))) { handle in
          XCTAssertThrowsError(try StreamCommand.readFrame(
            from: handle, width: 2, height: 1, temporal: true, protocolVersion: 3))
        }
      }
    }
  }

  func testVideoOutputProtocolConsumesSourceAfterGuidesAndRejectsTruncation() throws {
    let fields: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0, 0, 0.75, 1, 0.5001, 0.6, 0.7]
    let payload = streamPayload(flags: 6, values: fields)
    try withInputPipe(payload + payload) { handle in
      for _ in 0..<2 {
        let frame = try XCTUnwrap(StreamCommand.readFrame(from: handle, width: 2, height: 1,
          temporal: true, protocolVersion: 4, outputWidth: 1, outputHeight: 1))
        XCTAssertEqual(frame.inputs.map { $0.descriptor.name }, ["color", "motion", "depth", "historyConfidence"])
        XCTAssertEqual(frame.source?.descriptor.shape, [1, 1, 1, 3])
        XCTAssertEqual(frame.source?.bytes, Array(fields.suffix(3)).withUnsafeBytes { Data($0) })
      }
      XCTAssertNil(try StreamCommand.readFrame(from: handle, width: 2, height: 1,
        temporal: true, protocolVersion: 4, outputWidth: 1, outputHeight: 1))
    }
    for length in (payload.count - 12)..<payload.count {
      try withInputPipe(Data(payload.prefix(length))) { handle in
        XCTAssertThrowsError(try StreamCommand.readFrame(from: handle, width: 2, height: 1,
          temporal: true, protocolVersion: 4, outputWidth: 1, outputHeight: 1))
      }
    }
    var malformed = fields
    malformed[12] = .nan
    try withInputPipe(streamPayload(flags: 6, values: malformed)) { handle in
      XCTAssertThrowsError(try StreamCommand.readFrame(from: handle, width: 2, height: 1,
        temporal: true, protocolVersion: 4, outputWidth: 1, outputHeight: 1))
    }
  }

  func testVideoOutputOptionsFailBeforeOpeningModel() throws {
    for options in [
      ["--output-width", "0"], ["--output-height", "3"], ["--output-format", "rgb"],
      ["--detail-radius", "0"], ["--framegen-factor", "2"],
      ["--framegen-weights", "missing", "--framegen-factor", "1"],
      ["--framegen-weights", "missing", "--framegen-batch", "0"],
      ["--framegen-weights", "missing", "--framegen-precision", "int8"],
      ["--mode", "first-frame"],
    ] {
      let result = try runCLI(["stream", "/missing/model", "--width", "2", "--height", "2", "--protocol-version", "4"] + options)
      XCTAssertEqual(result.status, 2)
      XCTAssertTrue(result.stderr.contains("stream"), result.stderr)
      XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
    }
    let legacy = try runCLI(["stream", "/missing/model", "--width", "2", "--height", "2", "--output-format", "u8"])
    XCTAssertTrue(legacy.stderr.contains("--protocol-version 4"), legacy.stderr)
  }

  func testExternalStreamRejectsBadConfidenceWithoutOutputOrHungProcess() throws {
    guard let packagePath = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"]
    else {
      throw XCTSkip(
        "set MLXDLSS_NEURAL_RENDERING_PACKAGE to run malformed stream subprocess probes")
    }
    let valid: [Float] = [0.25, 0.5, 0.75, 0, 0, 1, 0.5]
    var malformed = valid
    malformed[6] = .nan
    for payload in [
      streamPayload(flags: 2, values: malformed),
      Data(streamPayload(flags: 2, values: valid).dropLast()),
    ] {
      let result = try runCLI(
        [
          "stream", packagePath, "--width", "1", "--height", "1", "--protocol-version", "2",
        ], input: payload)
      XCTAssertEqual(result.status, 2, result.stderr)
      XCTAssertTrue(result.stdout.isEmpty)
      XCTAssertTrue(result.stderr.contains("stream:"), result.stderr)
    }
  }

  private func streamPayload(flags: UInt32, values: [Float]) -> Data {
    var word = flags.littleEndian
    var data = withUnsafeBytes(of: &word) { Data($0) }
    data.append(values.withUnsafeBytes { Data($0) })
    return data
  }

  private func withInputPipe(_ data: Data, body: (FileHandle) throws -> Void) throws {
    let pipe = Pipe()
    pipe.fileHandleForWriting.write(data)
    try pipe.fileHandleForWriting.close()
    defer { try? pipe.fileHandleForReading.close() }
    try body(pipe.fileHandleForReading)
  }

  func testRenderImageRequiresOutputOption() throws {
    let result = try runCLI([
      "render-image", "/definitely/missing/image.png", "/definitely/missing/model.dlssmodel",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(result.stderr.contains("requires --output"), result.stderr)
  }

  func testRenderImageRejectsMLXOptionsForCoreML() throws {
    let result = try runCLI([
      "render-image", "/definitely/missing/image.png", "/definitely/missing/model.mlpackage",
      "--output", "/tmp/definitely-missing-output.png",
      "--backend", "coreml",
      "--precision", "float16",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("apply only to the MLX backend"),
      result.stderr
    )
  }

  func testExternalNeuralRenderingPackageRendersStillImage() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the image CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("input.png")
    try writeGradientImage(to: inputURL, width: 96, height: 64)
    let outputURL = temporaryDirectory.appendingPathComponent("output.png")

    let result = try runCLI([
      "render-image", inputURL.path, packagePath,
      "--output", outputURL.path,
      "--profile", "natural",
      "--local-tone", "0.5",
    ])

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["shape"] as? [Int], [64, 96, 3])
    XCTAssertEqual(summary["outputShape"] as? [Int], [64, 96, 3])
    XCTAssertEqual(summary["networkShape"] as? [Int], [1, 320, 320, 16])
    XCTAssertEqual(summary["profile"] as? String, "natural")
    XCTAssertEqual(summary["styleIndex"] as? Int, 1)
    XCTAssertEqual(summary["localToneStrength"] as? Double, 0.5)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
  }

  func testRGBFirstFrameInputRequiresExplicitTensorFiles() throws {
    let result = try runCLI([
      "run", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-first-frame",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains(
        "requires --input, --output, --height, and --width"
      ),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testModelInputRejectsFirstFrameControls() throws {
    let result = try runCLI([
      "run", "/definitely/missing/model.dlssmodel",
      "--auto-mask", "enabled",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("first-frame controls require rgb-first-frame input"),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testRunRejectsUnknownControlProfileBeforeOpeningPackage() throws {
    let result = try runCLI([
      "run", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-first-frame",
      "--profile", "loud",
      "--input", "/tmp/frame.f32",
      "--output", "/tmp/output.f32",
      "--height", "128",
      "--width", "128",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(result.stderr.contains("profile must be"), result.stderr)
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testExternalNeuralRenderingPackageAcceptsRecoveredRGBFirstFrameInput() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("rgb.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    let values = (0..<(128 * 128 * 3)).map { index in
      Float(index % 257) / 256
    }
    try values.withUnsafeBytes { Data($0) }.write(to: inputURL)

    let result = try runCLI([
      "run", packagePath,
      "--input", inputURL.path,
      "--input-format", "rgb-first-frame",
      "--output", outputURL.path,
      "--height", "128",
      "--width", "128",
    ])

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["inputFormat"] as? String, "rgb-first-frame")
    XCTAssertEqual(summary["profile"] as? String, "standard")
    XCTAssertEqual(summary["checkpointModelSelection"] as? Int, 0)
    XCTAssertEqual(summary["checkpointModel"] as? String, "shipping-default")
    XCTAssertEqual(summary["styleIndex"] as? Int, 0)
    XCTAssertEqual(summary["localToneStrength"] as? Double, 1)
    XCTAssertEqual(summary["localStructureStrength"] as? Double, 1)
    XCTAssertGreaterThan(summary["preprocessingNanoseconds"] as? Int ?? 0, 0)
    XCTAssertGreaterThan(summary["postprocessingNanoseconds"] as? Int ?? 0, 0)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 128, 128, 3])
    XCTAssertEqual(try Data(contentsOf: outputURL).count, 128 * 128 * 3 * 4)
  }

  func testExternalNeuralRenderingPackageAcceptsRecoveredFirstFrameControls() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("rgb.f32")
    let maskURL = temporaryDirectory.appendingPathComponent("control-mask.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    try [Float](repeating: 0.5, count: 128 * 128 * 3).withUnsafeBytes {
      try Data($0).write(to: inputURL)
    }
    var mask = [Float](repeating: 0, count: 128 * 128 * 3)
    for pixel in 0..<(128 * 128) {
      mask[pixel * 3] = 0.5
      mask[pixel * 3 + 1] = 0.25
      mask[pixel * 3 + 2] = 0.75
    }
    try mask.withUnsafeBytes { try Data($0).write(to: maskURL) }

    let result = try runCLI([
      "run", packagePath,
      "--input", inputURL.path,
      "--input-format", "rgb-first-frame",
      "--output", outputURL.path,
      "--height", "128",
      "--width", "128",
      "--control-mask", maskURL.path,
      "--profile", "natural",
      "--intensity", "0.8",
      "--style-index", "64",
      "--local-tone", "0.75",
      "--local-structure", "0.25",
      "--auto-mask", "enabled",
      "--skin-structure", "-1",
    ])

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["controlMask"] as? String, "full-rect-rgb")
    XCTAssertEqual(summary["profile"] as? String, "natural")
    XCTAssertEqual(summary["checkpointModelSelection"] as? Int, 0)
    XCTAssertEqual(summary["effectiveAutomaticMask"] as? Bool, false)
    XCTAssertEqual(summary["styleIndex"] as? Int, 64)
    XCTAssertEqual(
      try XCTUnwrap(summary["intensity"] as? Double),
      0.8,
      accuracy: 0.000_001
    )
    XCTAssertEqual(try Data(contentsOf: outputURL).count, 128 * 128 * 3 * 4)
  }

  func testExternalNeuralRenderingPackageAcceptsPointTransformedResources() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let logicalColorURL = temporaryDirectory.appendingPathComponent("logical-rgb.f32")
    let logicalMaskURL = temporaryDirectory.appendingPathComponent("logical-mask.f32")
    let resourceColorURL = temporaryDirectory.appendingPathComponent("resource-rgb.f32")
    let resourceMaskURL = temporaryDirectory.appendingPathComponent("resource-mask.f32")
    let logicalOutputURL = temporaryDirectory.appendingPathComponent("logical-output.f32")
    let transformedOutputURL = temporaryDirectory.appendingPathComponent(
      "transformed-output.f32"
    )
    let height = 128
    let logicalWidth = 128
    let resourceWidth = 256
    var resourceColor = [Float](repeating: 0, count: height * resourceWidth * 3)
    var resourceMask = [Float](repeating: 0, count: height * resourceWidth * 3)
    for y in 0..<height {
      for x in 0..<resourceWidth {
        let offset = (y * resourceWidth + x) * 3
        resourceColor[offset] = Float((x + y * 3) % 257) / 256
        resourceColor[offset + 1] = Float((x * 5 + y) % 251) / 250
        resourceColor[offset + 2] = Float((x * 7 + y * 11) % 241) / 240
        resourceMask[offset] = Float((x % 5) + 1) / 6
        resourceMask[offset + 1] = Float((x + y) % 7) / 6
        resourceMask[offset + 2] = Float((x * 3 + y) % 9) / 8
      }
    }
    var logicalColor: [Float] = []
    var logicalMask: [Float] = []
    logicalColor.reserveCapacity(height * logicalWidth * 3)
    logicalMask.reserveCapacity(height * logicalWidth * 3)
    for y in 0..<height {
      for x in 0..<logicalWidth {
        let resourceOffset = (y * resourceWidth + x * 2 + 1) * 3
        logicalColor.append(contentsOf: resourceColor[resourceOffset..<resourceOffset + 3])
        logicalMask.append(contentsOf: resourceMask[resourceOffset..<resourceOffset + 3])
      }
    }
    try logicalColor.withUnsafeBytes { try Data($0).write(to: logicalColorURL) }
    try logicalMask.withUnsafeBytes { try Data($0).write(to: logicalMaskURL) }
    try resourceColor.withUnsafeBytes { try Data($0).write(to: resourceColorURL) }
    try resourceMask.withUnsafeBytes { try Data($0).write(to: resourceMaskURL) }

    let common = [
      "--input-format", "rgb-first-frame",
      "--height", "128",
      "--width", "128",
      "--intensity", "0.8",
      "--local-tone", "0.75",
      "--local-structure", "0.25",
    ]
    let logical = try runCLI(
      [
        "run", packagePath,
        "--input", logicalColorURL.path,
        "--control-mask", logicalMaskURL.path,
        "--output", logicalOutputURL.path,
      ] + common
    )
    let transformed = try runCLI(
      [
        "run", packagePath,
        "--input", resourceColorURL.path,
        "--input-transform", "0,0,256,128,256,128",
        "--control-mask", resourceMaskURL.path,
        "--control-mask-transform", "0,0,256,128,256,128",
        "--output", transformedOutputURL.path,
      ] + common
    )

    XCTAssertEqual(logical.status, 0, logical.stderr)
    XCTAssertEqual(transformed.status, 0, transformed.stderr)
    XCTAssertEqual(
      try Data(contentsOf: transformedOutputURL),
      try Data(contentsOf: logicalOutputURL)
    )
    let summary = try jsonObject(transformed.stdout)
    XCTAssertEqual(summary["controlMask"] as? String, "point-transformed-rgb")
    XCTAssertEqual(
      summary["inputTransform"] as? [Int],
      [0, 0, 256, 128, 256, 128]
    )
    XCTAssertEqual(
      summary["controlMaskTransform"] as? [Int],
      [0, 0, 256, 128, 256, 128]
    )
  }

  func testRunRejectsMalformedTextureTransformBeforeOpeningPackage() throws {
    let result = try runCLI([
      "run", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-first-frame",
      "--input", "/tmp/color.f32",
      "--input-transform", "0,0,128",
      "--output", "/tmp/output.f32",
      "--height", "128",
      "--width", "128",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("input transform must contain six integers"),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testExternalNeuralRenderingPackageDerivesSixteenInputChannels() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("input.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    let values = (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) }
    try values.withUnsafeBytes { Data($0) }.write(to: inputURL)

    let result = try runCLI([
      "run", packagePath,
      "--input", inputURL.path,
      "--output", outputURL.path,
      "--height", "128",
      "--width", "128",
    ])

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 128, 128, 4])
    XCTAssertEqual(try Data(contentsOf: outputURL).count, 128 * 128 * 4 * 4)
  }

  func testExternalCoreMLNeuralRenderingPackageRunsRawHead() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE to run the CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("input.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    let input = (0..<(128 * 128 * 16)).map { sin(Float($0) * 0.001) }
    try input.withUnsafeBytes { Data($0) }.write(to: inputURL)

    let result = try runCLI([
      "run", packagePath,
      "--backend", "coreml",
      "--compute-units", "cpu-gpu",
      "--channels", "16",
      "--input", inputURL.path,
      "--output", outputURL.path,
      "--height", "128",
      "--width", "128",
    ])

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 128, 128, 4])
    XCTAssertEqual(summary["elementCount"] as? Int, 128 * 128 * 4)
    XCTAssertEqual(try Data(contentsOf: outputURL).count, 128 * 128 * 4 * 4)
  }

  func testTemporalReferenceSequenceRequiresAlignedGuideFiles() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-temporal-reference",
      "--input", "/tmp/frame.f32",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(result.stderr.contains("one --motion per --input"), result.stderr)
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testRunSequenceModelInputRejectsFeatureControls() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input", "/tmp/frame.f32",
      "--style-index", "1",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains(
        "run-sequence feature controls require rgb-temporal-reference input"
      ),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testRunSequenceJitterDeltaRequiresPixelMotion() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-temporal-reference",
      "--input", "/tmp/frame.f32",
      "--motion", "/tmp/motion.f32",
      "--depth", "/tmp/depth.f32",
      "--jitter-delta-x", "0.25",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains(
        "run-sequence jitter deltas require '--motion-format pixel'"
      ),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testRunSequenceRejectsNonFiniteJitterDelta() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input-format", "rgb-temporal-reference",
      "--motion-format", "pixel",
      "--input", "/tmp/frame.f32",
      "--motion", "/tmp/motion.f32",
      "--depth", "/tmp/depth.f32",
      "--jitter-delta-x", "nan",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains("run-sequence jitter delta x must be finite"),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testExternalNeuralRenderingPackageRunsTemporalReferenceSequence() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let outputDirectory = temporaryDirectory.appendingPathComponent("outputs")
    var arguments = [
      "run-sequence", packagePath,
      "--input-format", "rgb-temporal-reference",
      "--motion-format", "pixel",
      "--motion-scale-x", "1",
      "--motion-scale-y", "1",
      "--pipeline", "device-resident",
      "--execution", "block-compiled",
      "--profile", "neutral",
      "--height", "128",
      "--width", "128",
      "--output-dir", outputDirectory.path,
    ]
    for frame in 0..<2 {
      let colorURL = temporaryDirectory.appendingPathComponent("color-(frame).f32")
      let motionURL = temporaryDirectory.appendingPathComponent("motion-(frame).f32")
      let depthURL = temporaryDirectory.appendingPathComponent("depth-(frame).f32")
      let color = (0..<(128 * 128 * 3)).map { index in
        Float((index + frame * 31) % 257) / 256
      }
      let motion = [Float](repeating: 0, count: 128 * 128 * 2)
      let depth = [Float](repeating: 1, count: 128 * 128)
      try color.withUnsafeBytes { Data($0) }.write(to: colorURL)
      try motion.withUnsafeBytes { Data($0) }.write(to: motionURL)
      try depth.withUnsafeBytes { Data($0) }.write(to: depthURL)
      arguments += [
        "--input", colorURL.path,
        "--motion", motionURL.path,
        "--depth", depthURL.path,
      ]
    }

    let result = try runCLI(arguments)

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["inputFormat"] as? String, "rgb-temporal-reference")
    XCTAssertEqual(summary["motionFormat"] as? String, "pixel")
    XCTAssertEqual(summary["pipeline"] as? String, "device-resident")
    XCTAssertEqual(summary["preprocessor"] as? String, "not-applicable")
    XCTAssertEqual(summary["executionMode"] as? String, "block-compiled")
    XCTAssertEqual(summary["cadence"] as? String, "consecutiveFrames")
    XCTAssertEqual(summary["frameCount"] as? Int, 2)
    XCTAssertEqual(summary["profile"] as? String, "neutral")
    XCTAssertEqual(summary["checkpointModelSelection"] as? Int, 0)
    XCTAssertEqual(summary["styleIndex"] as? Int, 0)
    XCTAssertEqual(summary["localToneStrength"] as? Double, 0)
    XCTAssertEqual(summary["localStructureStrength"] as? Double, 0)
    XCTAssertEqual(summary["effectiveAutomaticMask"] as? Bool, false)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 128, 128, 3])
    XCTAssertGreaterThanOrEqual(
      summary["endToEndNanoseconds"] as? Int ?? 0,
      summary["executionNanoseconds"] as? Int ?? 1
    )
    var outputs: [Data] = []
    for frame in 0..<2 {
      let output = outputDirectory.appendingPathComponent(
        String(format: "%06d.f32", frame)
      )
      let data = try Data(contentsOf: output)
      XCTAssertEqual(data.count, 128 * 128 * 3 * 4)
      XCTAssertTrue(
        data.withUnsafeBytes { bytes in
          stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).allSatisfy {
            bytes.loadUnaligned(fromByteOffset: $0, as: Float.self).isFinite
          }
        })
      outputs.append(data)
    }
    XCTAssertNotEqual(outputs[0], outputs[1])
  }

  func testExternalTemporalSequenceAppliesPerFrameJitterBeforeMotionNormalization()
    throws
  {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the jitter CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let width = 128
    let height = 128
    let depthURL = temporaryDirectory.appendingPathComponent("depth.f32")
    let jitterMotionURL = temporaryDirectory.appendingPathComponent("jitter-motion.f32")
    let depth = [Float](repeating: 1, count: width * height)
    let jitterMotion = [Float](repeating: 0, count: width * height * 2)
    var colorURLs: [URL] = []
    for frame in 0..<2 {
      let colorURL = temporaryDirectory.appendingPathComponent("color-\(frame).f32")
      let color = (0..<(width * height * 3)).map { index in
        Float((index * 37 + frame * 101 + index / 3 * frame * 13) % 1021) / 1020
      }
      try color.withUnsafeBytes { Data($0) }.write(to: colorURL)
      colorURLs.append(colorURL)
    }
    try depth.withUnsafeBytes { Data($0) }.write(to: depthURL)
    try jitterMotion.withUnsafeBytes { Data($0) }.write(to: jitterMotionURL)
    var equivalentMotionURLs: [URL] = []
    for frame in 0..<2 {
      var equivalentMotion = jitterMotion
      if frame == 1 {
        for offset in stride(from: 0, to: equivalentMotion.count, by: 2) {
          equivalentMotion[offset] = -0.5
        }
      }
      let url = temporaryDirectory.appendingPathComponent(
        "equivalent-motion-\(frame).f32"
      )
      try equivalentMotion.withUnsafeBytes { Data($0) }.write(to: url)
      equivalentMotionURLs.append(url)
    }

    func arguments(motionURLs: [URL], outputName: String) -> [String] {
      var result = [
        "run-sequence", packagePath,
        "--input-format", "rgb-temporal-reference",
        "--motion-format", "pixel",
        "--motion-scale-x", "-2",
        "--motion-scale-y", "1",
        "--height", String(height),
        "--width", String(width),
        "--output-dir", temporaryDirectory.appendingPathComponent(outputName).path,
      ]
      for (frame, colorURL) in colorURLs.enumerated() {
        result += [
          "--input", colorURL.path,
          "--motion", motionURLs[frame].path,
          "--depth", depthURL.path,
        ]
      }
      return result
    }

    let jitterOutput = temporaryDirectory.appendingPathComponent("jitter-output")
    let equivalentOutput = temporaryDirectory.appendingPathComponent(
      "equivalent-output"
    )
    let jitterRun = try runCLI(
      arguments(
        motionURLs: [jitterMotionURL, jitterMotionURL],
        outputName: "jitter-output"
      ) + [
        "--jitter-delta-x", "0",
        "--jitter-delta-y", "0",
        "--jitter-delta-x", "1",
        "--jitter-delta-y", "0",
      ]
    )
    let equivalentRun = try runCLI(
      arguments(
        motionURLs: equivalentMotionURLs,
        outputName: "equivalent-output"
      )
    )

    XCTAssertEqual(jitterRun.status, 0, jitterRun.stderr)
    XCTAssertEqual(equivalentRun.status, 0, equivalentRun.stderr)
    XCTAssertEqual(
      try Data(contentsOf: jitterOutput.appendingPathComponent("000001.f32")),
      try Data(contentsOf: equivalentOutput.appendingPathComponent("000001.f32"))
    )
    let summary = try jsonObject(jitterRun.stdout)
    XCTAssertEqual(
      summary["jitterDeltaPixelsPerFrame"] as? [[Double]],
      [[0, 0], [1, 0]]
    )
  }

  func testExternalTemporalSequenceAcceptsTransformedResources() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the package CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let height = 128
    let logicalWidth = 128
    let resourceWidth = 256
    var logicalColorURLs: [URL] = []
    var logicalMotionURLs: [URL] = []
    var logicalMaskURLs: [URL] = []
    var resourceColorURLs: [URL] = []
    var resourceMotionURLs: [URL] = []
    var resourceMaskURLs: [URL] = []
    var depthURLs: [URL] = []
    for frame in 0..<2 {
      var resourceColor = [Float](
        repeating: 0,
        count: height * resourceWidth * 3
      )
      var resourceMotion = [Float](
        repeating: 0,
        count: height * resourceWidth * 2
      )
      var resourceMask = [Float](
        repeating: 0,
        count: height * resourceWidth * 3
      )
      for y in 0..<height {
        for x in 0..<resourceWidth {
          let colorOffset = (y * resourceWidth + x) * 3
          resourceColor[colorOffset] = Float((x + y * 3 + frame * 17) % 257) / 256
          resourceColor[colorOffset + 1] = Float((x * 5 + y + frame) % 251) / 250
          resourceColor[colorOffset + 2] = Float((x * 7 + y * 11) % 241) / 240
          let motionOffset = (y * resourceWidth + x) * 2
          resourceMotion[motionOffset] = Float((x + frame) % 5 - 2) / 512
          resourceMotion[motionOffset + 1] = Float((y + frame) % 3 - 1) / 512
          if x.isMultiple(of: 2) {
            resourceMask[colorOffset] = 1
            resourceMask[colorOffset + 1] = 1
            resourceMask[colorOffset + 2] = 1
          } else {
            resourceMask[colorOffset] = 0.25
            resourceMask[colorOffset + 1] = 0.5
            resourceMask[colorOffset + 2] = 0.75
          }
        }
      }
      var logicalColor: [Float] = []
      var logicalMotion: [Float] = []
      var logicalMask: [Float] = []
      logicalColor.reserveCapacity(height * logicalWidth * 3)
      logicalMotion.reserveCapacity(height * logicalWidth * 2)
      logicalMask.reserveCapacity(height * logicalWidth * 3)
      for y in 0..<height {
        for x in 0..<logicalWidth {
          let resourcePixel = y * resourceWidth + x * 2 + 1
          let colorOffset = resourcePixel * 3
          let motionOffset = resourcePixel * 2
          logicalColor.append(
            contentsOf: resourceColor[colorOffset..<colorOffset + 3]
          )
          logicalMotion.append(
            contentsOf: resourceMotion[motionOffset..<motionOffset + 2]
          )
          logicalMask.append(
            contentsOf: resourceMask[colorOffset..<colorOffset + 3]
          )
        }
      }
      let logicalColorURL = temporaryDirectory.appendingPathComponent(
        "logical-color-\(frame).f32"
      )
      let logicalMotionURL = temporaryDirectory.appendingPathComponent(
        "logical-motion-\(frame).f32"
      )
      let resourceColorURL = temporaryDirectory.appendingPathComponent(
        "resource-color-\(frame).f32"
      )
      let resourceMotionURL = temporaryDirectory.appendingPathComponent(
        "resource-motion-\(frame).f32"
      )
      let logicalMaskURL = temporaryDirectory.appendingPathComponent(
        "logical-mask-\(frame).f32"
      )
      let resourceMaskURL = temporaryDirectory.appendingPathComponent(
        "resource-mask-\(frame).f32"
      )
      let depthURL = temporaryDirectory.appendingPathComponent(
        "depth-\(frame).f32"
      )
      try logicalColor.withUnsafeBytes { try Data($0).write(to: logicalColorURL) }
      try logicalMotion.withUnsafeBytes { try Data($0).write(to: logicalMotionURL) }
      try resourceColor.withUnsafeBytes { try Data($0).write(to: resourceColorURL) }
      try resourceMotion.withUnsafeBytes { try Data($0).write(to: resourceMotionURL) }
      try logicalMask.withUnsafeBytes { try Data($0).write(to: logicalMaskURL) }
      try resourceMask.withUnsafeBytes { try Data($0).write(to: resourceMaskURL) }
      try [Float](repeating: 1, count: height * logicalWidth).withUnsafeBytes {
        try Data($0).write(to: depthURL)
      }
      logicalColorURLs.append(logicalColorURL)
      logicalMotionURLs.append(logicalMotionURL)
      logicalMaskURLs.append(logicalMaskURL)
      resourceColorURLs.append(resourceColorURL)
      resourceMotionURLs.append(resourceMotionURL)
      resourceMaskURLs.append(resourceMaskURL)
      depthURLs.append(depthURL)
    }

    func arguments(
      colors: [URL],
      motions: [URL],
      masks: [URL],
      output: URL,
      resourceTransforms: Bool,
      historyTransform: Bool
    ) -> [String] {
      var result = [
        "run-sequence", packagePath,
        "--input-format", "rgb-temporal-reference",
        "--pipeline", "device-resident",
        "--height", "128",
        "--width", "128",
        "--intensity", "0.8",
        "--output-dir", output.path,
      ]
      if resourceTransforms {
        result += [
          "--input-transform", "0,0,256,128,256,128",
          "--motion-transform", "0,0,256,128,256,128",
          "--control-mask-transform", "0,0,256,128,256,128",
        ]
      }
      if historyTransform {
        result += ["--history-transform", "8,0,112,128,128,128"]
      }
      for frame in 0..<2 {
        result += [
          "--input", colors[frame].path,
          "--motion", motions[frame].path,
          "--depth", depthURLs[frame].path,
          "--control-mask", masks[frame].path,
        ]
      }
      return result
    }

    let logicalOutput = temporaryDirectory.appendingPathComponent("logical-output")
    let resourceOutput = temporaryDirectory.appendingPathComponent("resource-output")
    let identityOutput = temporaryDirectory.appendingPathComponent("identity-output")
    let logical = try runCLI(arguments(
      colors: logicalColorURLs,
      motions: logicalMotionURLs,
      masks: logicalMaskURLs,
      output: logicalOutput,
      resourceTransforms: false,
      historyTransform: true
    ))
    let resource = try runCLI(arguments(
      colors: resourceColorURLs,
      motions: resourceMotionURLs,
      masks: resourceMaskURLs,
      output: resourceOutput,
      resourceTransforms: true,
      historyTransform: true
    ))
    let identity = try runCLI(arguments(
      colors: logicalColorURLs,
      motions: logicalMotionURLs,
      masks: logicalMaskURLs,
      output: identityOutput,
      resourceTransforms: false,
      historyTransform: false
    ))

    XCTAssertEqual(logical.status, 0, logical.stderr)
    XCTAssertEqual(resource.status, 0, resource.stderr)
    XCTAssertEqual(identity.status, 0, identity.stderr)
    for frame in 0..<2 {
      let name = String(format: "%06d.f32", frame)
      XCTAssertEqual(
        try Data(contentsOf: resourceOutput.appendingPathComponent(name)),
        try Data(contentsOf: logicalOutput.appendingPathComponent(name))
      )
    }
    XCTAssertNotEqual(
      try Data(contentsOf: logicalOutput.appendingPathComponent("000001.f32")),
      try Data(contentsOf: identityOutput.appendingPathComponent("000001.f32"))
    )
    let summary = try jsonObject(resource.stdout)
    XCTAssertEqual(
      summary["inputTransform"] as? [Int],
      [0, 0, 256, 128, 256, 128]
    )
    XCTAssertEqual(
      summary["motionTransform"] as? [Int],
      [0, 0, 256, 128, 256, 128]
    )
    XCTAssertEqual(
      summary["historyTransform"] as? [Int],
      [8, 0, 112, 128, 128, 128]
    )
    XCTAssertEqual(
      summary["controlMaskTransform"] as? [Int],
      [0, 0, 256, 128, 256, 128]
    )
    XCTAssertEqual(
      try XCTUnwrap(summary["intensity"] as? Double),
      0.8,
      accuracy: 0.000_001
    )
  }

  func testRunSequenceRejectsTransformsForModelInput() throws {
    let result = try runCLI([
      "run-sequence", "/definitely/missing/model.dlssmodel",
      "--input", "/tmp/frame.f32",
      "--input-transform", "0,0,128,128,128,128",
      "--height", "128",
      "--width", "128",
      "--output-dir", "/tmp/output",
    ])

    XCTAssertEqual(result.status, 2)
    XCTAssertTrue(
      result.stderr.contains(
        "texture transforms require rgb-temporal-reference input"
      ),
      result.stderr
    )
    XCTAssertFalse(result.stderr.contains("packageIsNotDirectory"), result.stderr)
  }

  func testExternalCoreMLNeuralRenderingPackageRunsTemporalReferenceSequence() throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_COREML_PACKAGE to run the CLI probe")
    }
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let outputDirectory = temporaryDirectory.appendingPathComponent("outputs")
    var arguments = [
      "run-sequence", packagePath,
      "--backend", "coreml",
      "--compute-units", "cpu-gpu",
      "--input-format", "rgb-temporal-reference",
      "--preprocessor", "metal",
      "--pipeline", "portable",
      "--height", "128",
      "--width", "128",
      "--output-dir", outputDirectory.path,
    ]
    for frame in 0..<2 {
      let colorURL = temporaryDirectory.appendingPathComponent("color-\(frame).f32")
      let motionURL = temporaryDirectory.appendingPathComponent("motion-\(frame).f32")
      let depthURL = temporaryDirectory.appendingPathComponent("depth-\(frame).f32")
      let color = (0..<(128 * 128 * 3)).map { index in
        Float((index + frame * 31) % 257) / 256
      }
      let motion = [Float](repeating: 0, count: 128 * 128 * 2)
      let depth = [Float](repeating: 1, count: 128 * 128)
      try color.withUnsafeBytes { Data($0) }.write(to: colorURL)
      try motion.withUnsafeBytes { Data($0) }.write(to: motionURL)
      try depth.withUnsafeBytes { Data($0) }.write(to: depthURL)
      arguments += [
        "--input", colorURL.path,
        "--motion", motionURL.path,
        "--depth", depthURL.path,
      ]
    }

    let result = try runCLI(arguments)

    XCTAssertEqual(result.status, 0, result.stderr)
    let summary = try jsonObject(result.stdout)
    XCTAssertEqual(summary["backend"] as? String, "coreml-cpu-gpu-multi-array")
    XCTAssertEqual(summary["pipeline"] as? String, "portable")
    XCTAssertEqual(summary["preprocessor"] as? String, "metal")
    XCTAssertEqual(summary["frameCount"] as? Int, 2)
    XCTAssertEqual(summary["profile"] as? String, "standard")
    XCTAssertEqual(summary["checkpointModelSelection"] as? Int, 0)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 128, 128, 3])
    for frame in 0..<2 {
      let output = outputDirectory.appendingPathComponent(
        String(format: "%06d.f32", frame)
      )
      XCTAssertEqual(try Data(contentsOf: output).count, 128 * 128 * 3 * 4)
    }
  }

  func testMalformedPackageExitsTwoWithoutCrashDump() throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let packageURL = temporaryDirectory.appendingPathComponent("Malformed.dlssmodel")
    try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: false)

    let result = try runCLI(["inspect", packageURL.path])

    XCTAssertEqual(result.status, 2)
    XCTAssertFalse(result.stderr.isEmpty)
    XCTAssertFalse(result.stderr.contains("Fatal error"))
    XCTAssertFalse(result.stderr.localizedCaseInsensitiveContains("backtrace"))
  }

  func testRunReadsAndWritesExactRawFloat32Tensor() async throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let packageURL = temporaryDirectory.appendingPathComponent("Demo.dlssmodel")
    try await DemoModelPackageWriter().write(to: packageURL)
    let inputURL = temporaryDirectory.appendingPathComponent("input.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    let input: [Float] = [0.2, 0.4, 0.25, 0.6, 2, 2]
    try input.withUnsafeBytes { Data($0) }.write(to: inputURL)

    let run = try runCLI([
      "run", packageURL.path,
      "--input", inputURL.path,
      "--output", outputURL.path,
      "--height", "1",
      "--width", "2",
      "--execution", "compiled",
      "--precision", "float16",
    ])

    XCTAssertEqual(run.status, 0, run.stderr)
    let summary = try jsonObject(run.stdout)
    XCTAssertEqual(summary["shape"] as? [Int], [1, 1, 2, 3])
    XCTAssertEqual(summary["elementCount"] as? Int, 6)
    XCTAssertEqual(summary["executionMode"] as? String, "compiled")
    XCTAssertEqual(summary["computePrecision"] as? String, "float16")
    let outputData = try Data(contentsOf: outputURL)
    let output = outputData.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
    let expected: [Float] = [0.5, 0.4, 0.75, 1, 1, 0]
    for (actual, expected) in zip(output, expected) {
      XCTAssertEqual(actual, expected, accuracy: 0.001)
    }
  }

  func testRunDoesNotOverwriteRawOutput() async throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let packageURL = temporaryDirectory.appendingPathComponent("Demo.dlssmodel")
    try await DemoModelPackageWriter().write(to: packageURL)
    let inputURL = temporaryDirectory.appendingPathComponent("input.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    try Data(count: 12).write(to: inputURL)
    try Data("keep".utf8).write(to: outputURL)

    let run = try runCLI([
      "run", packageURL.path,
      "--input", inputURL.path,
      "--output", outputURL.path,
      "--height", "1",
      "--width", "1",
    ])

    XCTAssertEqual(run.status, 2)
    XCTAssertTrue(run.stderr.contains("destination already exists"), run.stderr)
    XCTAssertEqual(try String(contentsOf: outputURL, encoding: .utf8), "keep")
  }

  func testRunRoutesCoreMLBackendOptionsBeforeOpeningModel() throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
    let inputURL = temporaryDirectory.appendingPathComponent("input.f32")
    let outputURL = temporaryDirectory.appendingPathComponent("output.f32")
    try Data(count: 12).write(to: inputURL)

    let run = try runCLI([
      "run", temporaryDirectory.appendingPathComponent("missing.mlpackage").path,
      "--backend", "coreml",
      "--compute-units", "cpu-gpu",
      "--coreml-bridge", "multi-array",
      "--input", inputURL.path,
      "--output", outputURL.path,
      "--height", "1",
      "--width", "1",
    ])

    XCTAssertEqual(run.status, 2)
    XCTAssertTrue(run.stderr.contains("modelNotFound"), run.stderr)
    XCTAssertFalse(FileManager.default.fileExists(atPath: outputURL.path))
  }

  /// Writes a small RGB gradient PNG with ImageIO (the video fixtures were removed).
  private func writeGradientImage(to url: URL, width: Int, height: Int) throws {
    var pixels = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
      for x in 0..<width {
        let offset = (y * width + x) * 4
        pixels[offset] = UInt8(min(255, x * 255 / max(1, width - 1)))
        pixels[offset + 1] = UInt8(min(255, y * 255 / max(1, height - 1)))
        pixels[offset + 2] = 128
      }
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
      let image = CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
      ),
      let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else {
      throw NSError(domain: "CLITests", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot write gradient image"])
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw NSError(domain: "CLITests", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot finalize gradient image"])
    }
  }

  private func runCLI(_ arguments: [String], input: Data? = nil) throws -> ProcessResult {
    let process = Process()
    process.executableURL = cliExecutableURL()
    process.arguments = arguments
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    let completed = DispatchSemaphore(value: 0)
    let stdin = Pipe()
    if let input {
      process.standardInput = stdin
      stdin.fileHandleForWriting.write(input)
      try stdin.fileHandleForWriting.close()
      process.terminationHandler = { _ in completed.signal() }
    }
    try process.run()
    if input != nil, completed.wait(timeout: .now() + 30) == .timedOut {
      process.terminate()
      process.waitUntilExit()
      XCTFail("stream did not terminate after malformed input")
    } else {
      process.waitUntilExit()
    }

    return ProcessResult(
      status: process.terminationStatus,
      stdout: String(
        data: stdout.fileHandleForReading.readDataToEndOfFile(),
        encoding: .utf8
      ) ?? "",
      stderr: String(
        data: stderr.fileHandleForReading.readDataToEndOfFile(),
        encoding: .utf8
      ) ?? ""
    )
  }

  /// The binary under test: `MLXDLSS_BINARY`, else the `mlxdlss` built next to
  /// this test bundle (so `swift test -c release` exercises the release build),
  /// else the package's debug build.
  private func cliExecutableURL() -> URL {
    if let override = ProcessInfo.processInfo.environment["MLXDLSS_BINARY"], !override.isEmpty {
      return URL(fileURLWithPath: override)
    }
    let sibling = Bundle(for: CLITests.self).bundleURL.deletingLastPathComponent()
      .appendingPathComponent("mlxdlss")
    if FileManager.default.isExecutableFile(atPath: sibling.path) {
      return sibling
    }
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<3 {
      url.deleteLastPathComponent()
    }
    return url.appendingPathComponent(".build/debug/mlxdlss")
  }

  private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
  }

  private func jsonObject(_ text: String) throws -> [String: Any] {
    try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
    )
  }
}

private struct ProcessResult {
  let status: Int32
  let stdout: String
  let stderr: String
}
