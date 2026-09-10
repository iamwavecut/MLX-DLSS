import Cmlx
import CoreVideo
import Foundation
import IOSurface
import Metal
import MLX

public struct MLXMediaError: Error, LocalizedError, Sendable {
  public let message: String
  public init(_ message: String) { self.message = message }
  public var errorDescription: String? { message }
}

/// A retained pixel buffer handed between pipeline stages. A producer must finish
/// writing before handing it off, and must not modify it while consumers retain it.
public struct MLXPixelBuffer: @unchecked Sendable {
  public let buffer: CVPixelBuffer
  public init(_ buffer: CVPixelBuffer) { self.buffer = buffer }
  public var width: Int { CVPixelBufferGetWidth(buffer) }
  public var height: Int { CVPixelBufferGetHeight(buffer) }
}

/// Immutable RGB storage. Frames are evaluated before crossing an actor boundary,
/// except frames the video pipeline schedules ahead: their graph is already
/// submitted and every consumer evaluates before touching the bytes. Only the
/// rendering actors build new graphs from this storage.
public final class MLXVideoFrame: @unchecked Sendable {
  let array: MLXArray
  public let width: Int
  public let height: Int

  init(_ array: MLXArray) {
    precondition(array.ndim == 4 && array.dim(0) == 1 && array.dim(3) == 3)
    self.array = contiguous(array.asType(.float32))
    eval(self.array)
    width = array.dim(2)
    height = array.dim(1)
  }

  /// A frame whose graph is submitted without waiting for it, so its GPU work can
  /// overlap the encoding of the previous frame.
  init(scheduling array: MLXArray) {
    precondition(array.ndim == 4 && array.dim(0) == 1 && array.dim(3) == 3)
    self.array = contiguous(array.asType(.float32))
    asyncEval(self.array)
    width = array.dim(2)
    height = array.dim(1)
  }

  public convenience init(rgb: Data, width: Int, height: Int) throws {
    guard width > 0, height > 0, width <= Int(Int32.max) / height / 3,
      rgb.count == width * height * 3 * MemoryLayout<Float>.size
    else { throw MLXMediaError("Invalid RGB frame dimensions or byte count") }
    self.init(MLXArray(rgb, [1, height, width, 3], dtype: .float32))
  }

  public convenience init(pixelBuffer: MLXPixelBuffer) throws {
    let buffer = pixelBuffer.buffer
    let format = CVPixelBufferGetPixelFormatType(buffer)
    guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_64RGBAHalf else {
      throw MLXMediaError("Native RGB import requires BGRA8 or RGBA16F")
    }
    let half = format == kCVPixelFormatType_64RGBAHalf
    let bytes = try Self.storage(buffer, dtype: half ? .float16 : .uint8)
    let dimensions = MLXArray([UInt32(pixelBuffer.width), UInt32(pixelBuffer.height),
      UInt32(CVPixelBufferGetBytesPerRow(buffer) / (half ? 2 : 1)), 0, 0, 0, 0, 0])
    let count = pixelBuffer.width * pixelBuffer.height * 3
    let rgb = Self.importRGB([bytes, dimensions], template: [("halfInput", half)],
      grid: (count, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [[1, pixelBuffer.height, pixelBuffer.width, 3]], outputDTypes: [.float32])[0]
    self.init(rgb)
  }

  public func copyRGBData() -> Data {
    eval(array)
    return array.asData(access: .copy).data
  }

  static func storage(_ buffer: CVPixelBuffer, dtype: DType) throws -> MLXArray {
    guard !CVPixelBufferIsPlanar(buffer),
      let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue()
    else { throw MLXMediaError("Native GPU import requires a non-planar IOSurface pixel buffer") }
    let address = IOSurfaceGetBaseAddress(surface)
    let count = CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer) / dtype.size
    guard count > 0, count <= Int(Int32.max) else { throw MLXMediaError("Pixel buffer is too large") }
    // Release the retained buffer in the C finalizer. MLX Swift 0.31.6's managed
    // initializer leaves its closure capture allocated, which would retain every
    // decoded IOSurface in a long video.
    let owner = Unmanaged.passRetained(buffer).toOpaque()
    var shape = [Int32(count)]
    let type = dtype == .float16 ? MLX_FLOAT16 : dtype == .float32 ? MLX_FLOAT32 : MLX_UINT8
    return MLXArray(mlx_array_new_data_managed_payload(address, &shape, 1, type, owner) { payload in
      if let payload { Unmanaged<CVPixelBuffer>.fromOpaque(payload).release() }
    })
  }

  private static let importRGB = MLXFast.metalKernel(
    name: "mlxdlss_native_import_rgb", inputNames: ["pixels", "dimensions"], outputNames: ["rgb"],
    source: #"""
      uint i = thread_position_in_grid.x, width = dimensions[0], height = dimensions[1];
      if (i >= width * height * 3) return;
      uint c = i % 3, x = (i / 3) % width, y = i / (3 * width);
      uint channel = halfInput ? c : 2 - c;
      rgb[i] = float(pixels[y * dimensions[2] + x * 4 + channel]) / (halfInput ? 1.0f : 255.0f);
      """#)
}

/// Converts evaluated MLX RGB directly into an encoder-owned IOSurface on Metal.
/// The completion handler keeps both buffer owners alive until the GPU finishes.
public actor MLXPixelBufferWriter {
  private let device: any MTLDevice
  private let queue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let pool: CVPixelBufferPool
  private let width: Int
  private let height: Int

  public init(width: Int, height: Int, halfOutput: Bool = false) throws {
    guard width > 0, height > 0, let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else { throw MLXMediaError("Metal is unavailable") }
    self.width = width
    self.height = height
    self.device = device
    self.queue = queue
    let source = """
      #include <metal_stdlib>
      using namespace metal;
      kernel void pack(device const float *rgb [[buffer(0)]],
                       device \(halfOutput ? "half" : "uchar") *pixels [[buffer(1)]],
                       constant uint3 &dimensions [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
        if (i >= dimensions.x * dimensions.y) return;
        uint base = (i / dimensions.x) * dimensions.z + (i % dimensions.x) * 4;
        for (uint c = 0; c < 3; ++c) {
          float value = clamp(rgb[i * 3 + c], 0.0f, 1.0f);
          pixels[base + \(halfOutput ? "c" : "2 - c")] = \(halfOutput ? "half(value)" : "uchar(value * 255.0f + 0.5f)");
        }
        pixels[base + 3] = \(halfOutput ? "half(1.0f)" : "uchar(255)");
      }
      """
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "pack") else { throw MLXMediaError("Missing RGB packing kernel") }
    pipeline = try device.makeComputePipelineState(function: function)
    var pool: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, [
      kCVPixelBufferWidthKey: width, kCVPixelBufferHeightKey: height,
      kCVPixelBufferPixelFormatTypeKey: halfOutput ? kCVPixelFormatType_64RGBAHalf : kCVPixelFormatType_32BGRA,
      kCVPixelBufferMetalCompatibilityKey: true,
      kCVPixelBufferIOSurfacePropertiesKey: [:],
    ] as CFDictionary, &pool)
    guard status == kCVReturnSuccess, let pool else { throw MLXMediaError("Cannot allocate output pool: \(status)") }
    self.pool = pool
  }

  public func write(_ frame: MLXVideoFrame) async throws -> MLXPixelBuffer {
    guard frame.width == width, frame.height == height else { throw MLXMediaError("Output frame size changed") }
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
    guard status == kCVReturnSuccess, let pixelBuffer,
      let surface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue(),
      let destination = device.makeBuffer(bytesNoCopy: IOSurfaceGetBaseAddress(surface), length: IOSurfaceGetAllocSize(surface), options: .storageModeShared),
      let source = frame.array.asMTLBuffer(device: device, noCopy: true),
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw MLXMediaError("Cannot bind native output buffers") }
    command.label = "DLSS native video output"
    let half = CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_64RGBAHalf
    var dimensions = SIMD3<UInt32>(UInt32(width), UInt32(height), UInt32(CVPixelBufferGetBytesPerRow(pixelBuffer) / (half ? 2 : 1)))
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    encoder.setBytes(&dimensions, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
    encoder.dispatchThreads(MTLSize(width: width * height, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    let result = MLXPixelBuffer(pixelBuffer)
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
      command.addCompletedHandler { [frame, result] completed in
        withExtendedLifetime((frame, result)) {
          if let error = completed.error { continuation.resume(throwing: error) }
          else { continuation.resume() }
        }
      }
      command.commit()
    }
    return result
  }
}

public actor MLXNativeFrameGenerator {
  private let generator: FrameGenerator
  public init(weightsURL: URL, precision: FrameGenerator.Precision = .float16) throws {
    generator = try FrameGenerator(weightsURL: weightsURL, precision: precision)
  }

  public func interpolate(_ previous: MLXVideoFrame, _ current: MLXVideoFrame, factor: Int) throws -> [MLXVideoFrame] {
    guard factor >= 2, factor <= 16 else { throw MLXMediaError("Frame generation factor must be within 2...16") }
    let generated = try generator.interpolationGraph(previous.array, current.array,
      phases: (1..<factor).map { Float($0) / Float(factor) })
    eval(generated)
    return (0..<(factor - 1)).map { MLXVideoFrame(generated[$0..<($0 + 1)]) }
  }
}
