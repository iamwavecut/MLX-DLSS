import CryptoKit
import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in whole-network digests for comparing builds (`MLXDLSS_PERF_SPIKE=1` and
/// `MLXDLSS_LOGICAL_WEIGHTS`): SHA-256 of the float16 output at three extents and two
/// inputs. `MLXDLSS_DIGEST_DUMP=<dir>` also writes each output as raw float16, so
/// outputs that differ between builds can be compared numerically:
/// `MLXDLSS_DIGEST_COMPARE=<reference dir>,<candidate dir>` prints the relative
/// L2 error, the largest absolute difference and the share of identical values.
final class WholeFrameDigestTests: XCTestCase {
  func testWholeFrameDigests() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["MLXDLSS_PERF_SPIKE"] == "1", let path = environment["MLXDLSS_LOGICAL_WEIGHTS"] else {
      throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS")
    }
    let dumpDirectory = environment["MLXDLSS_DIGEST_DUMP"].map { URL(fileURLWithPath: $0, isDirectory: true) }
    if let dumpDirectory { try FileManager.default.createDirectory(at: dumpDirectory, withIntermediateDirectories: true) }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let model = try NeuralRenderingTransformerModel(weights: weights, compileBlocks: true)
    let network = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
    for (height, width) in [(384, 384), (768, 1024), (1088, 1920)] {
      for variant in 0..<2 {
        let count = height * width * 16
        let values = (0..<count).map { index -> Float in
          let x = Float(index)
          return variant == 0 ? sin(x * 0.001) : 0.6 * cos(x * 0.0007 + 0.3) + 0.25 * sin(x * 0.013)
        }
        let input = MLXArray(values, [1, height, width, 16]).asType(.float16)
        let output = Device.withDefaultDevice(.gpu) { model(input) }.asType(.float16)
        eval(output)
        let data = output.asData(access: .copy).data
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        print("digest \(network) \(height)x\(width) input\(variant): \(digest) shape \(output.shape)")
        if let dumpDirectory {
          try data.write(to: dumpDirectory.appendingPathComponent("\(network)-\(height)x\(width)-input\(variant).f16"))
        }
      }
    }
  }

  func testCompareDumpedOutputs() throws {
    guard let spec = ProcessInfo.processInfo.environment["MLXDLSS_DIGEST_COMPARE"] else {
      throw XCTSkip("MLXDLSS_DIGEST_COMPARE=<reference dir>,<candidate dir>")
    }
    let directories = spec.split(separator: ",").map { URL(fileURLWithPath: String($0), isDirectory: true) }
    guard directories.count == 2 else { throw XCTSkip("MLXDLSS_DIGEST_COMPARE needs two directories") }
    let names = try FileManager.default.contentsOfDirectory(atPath: directories[0].path)
      .filter { $0.hasSuffix(".f16") }.sorted()
    XCTAssertFalse(names.isEmpty, "no .f16 outputs in \(directories[0].path)")
    for name in names {
      let candidateURL = directories[1].appendingPathComponent(name)
      guard FileManager.default.fileExists(atPath: candidateURL.path) else {
        print("digest-compare \(name): missing in \(directories[1].path)")
        continue
      }
      let reference = Self.halfValues(try Data(contentsOf: directories[0].appendingPathComponent(name)))
      let candidate = Self.halfValues(try Data(contentsOf: candidateURL))
      XCTAssertEqual(reference.count, candidate.count, name)
      var squaredError = 0.0, squaredReference = 0.0, maxAbs = 0.0, identical = 0, nonFinite = 0
      for (a, b) in zip(reference, candidate) {
        let x = Double(a), y = Double(b)
        guard x.isFinite, y.isFinite else { nonFinite += 1; continue }
        let d = abs(x - y)
        squaredError += d * d
        squaredReference += x * x
        maxAbs = max(maxAbs, d)
        if a.bitPattern == b.bitPattern { identical += 1 }
      }
      let relative = squaredReference > 0 ? (squaredError / squaredReference).squareRoot() : 0
      print(String(
        format: "digest-compare %@: relL2 %.3e maxabs %.4g identical %.2f%% nonfinite %d",
        name, relative, maxAbs, 100 * Double(identical) / Double(max(1, reference.count)), nonFinite))
    }
  }

  private static func halfValues(_ data: Data) -> [Float16] {
    data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
  }
}
