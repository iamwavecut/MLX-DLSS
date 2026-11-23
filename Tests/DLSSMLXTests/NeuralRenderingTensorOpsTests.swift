import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// The Metal 4 tensor-operation feed-forward (`matmul2d`, which runs on the GPU
/// Neural Accelerators of M5 and later) against the simdgroup kernel. On shader
/// cores (M2 Max) the two agree bit for bit; the Neural Accelerators may sum the
/// products in another order, and E4M3 publication turns a last-bit difference
/// before a rounding boundary into one E4M3 step, so the test bounds the
/// difference instead of requiring identity.
final class NeuralRenderingTensorOpsTests: XCTestCase {
  private typealias Ops = NeuralRenderingTransformerOperations

  private func normal(_ shape: [Int], _ scale: Float) -> MLXArray {
    (MLXRandom.normal(shape) * scale).asType(.float16)
  }

  private func requireTensorOperations() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else { throw XCTSkip("Metal 4 tensor operations need macOS 26") }
  }

  func testTensorOpsFeedForwardMatchesTheSimdgroupKernel() throws {
    try requireTensorOperations()
    MLXRandom.seed(3)
    for (groups, height, width) in [(2, 24, 40), (4, 9, 13), (2, 5, 3), (4, 64, 96)] {
      let channels = groups * 32
      let input = Ops.e4m3RoundTrip(normal([1, height, width, channels], 1.5))
      let dense = Ops.denseExpansionWeight(normal([groups, 4, groups, 32, 32], 0.08))
      let grouped = Ops.groupedProjectionWeight(normal([groups, 4, 32, 32], 0.08))
      let output = normal([channels, channels], 0.08)
      let scale = MLXRandom.uniform(low: 0.5, high: 1.0, [channels]).asType(.float16)
      let expected = Device.withDefaultDevice(.gpu) {
        NeuralRenderingFusedBranchedFeedForward.apply(
          input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
          cosine: scale, kernel: nil, simdgroups: nil)
      }
      for rows in [16, 32] {
        let actual = Device.withDefaultDevice(.gpu) {
          NeuralRenderingFusedBranchedFeedForward.applyTensorOps(
            input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
            cosine: scale, rowsPerSimdgroup: rows)
        }
        eval(expected, actual)
        let a = expected.asType(.float32), b = actual.asType(.float32)
        let identical = (expected.view(dtype: .uint16) .== actual.view(dtype: .uint16)).asType(.float32).mean()
          .item(Float.self)
        let relative = (sqrt(((a - b) * (a - b)).sum()) / sqrt((a * a).sum())).item(Float.self)
        let maxAbs = abs(a - b).max().item(Float.self)
        let label = "\(channels)ch \(height)x\(width) rows \(rows)"
        print("tensor-ops ffn \(label): identical \(identical * 100)% relL2 \(relative) max |Δ| \(maxAbs)")
        XCTAssertFalse(MLX.isNaN(actual.asType(.float32)).any().item(Bool.self), label)
        XCTAssertLessThan(relative, 0.02, label)
        XCTAssertGreaterThan(identical, 0.9, label)
      }
    }
  }

  /// Opt-in timing (`MLXDLSS_PERF_SPIKE=1`) at the small network's 1088x1920 shapes:
  /// alternating samples of the simdgroup kernel and the tensor-operation variant.
  func testTensorOpsFeedForwardTiming() throws {
    guard ProcessInfo.processInfo.environment["MLXDLSS_PERF_SPIKE"] == "1" else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    try requireTensorOperations()
    MLXRandom.seed(4)
    for (groups, height, width) in [(2, 272, 480), (4, 136, 240)] {
      let channels = groups * 32
      let input = Ops.e4m3RoundTrip(normal([1, height, width, channels], 1.5))
      let dense = Ops.denseExpansionWeight(normal([groups, 4, groups, 32, 32], 0.08))
      let grouped = Ops.groupedProjectionWeight(normal([groups, 4, 32, 32], 0.08))
      let output = normal([channels, channels], 0.08)
      let scale = MLXRandom.uniform(low: 0.5, high: 1.0, [channels]).asType(.float16)
      eval(input, dense, grouped, output, scale)
      let variants: [(String, () -> MLXArray)] = [
        ("simdgroup", {
          NeuralRenderingFusedBranchedFeedForward.apply(
            input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
            cosine: scale, kernel: nil, simdgroups: nil)
        }),
        ("tensor-ops rows16", {
          NeuralRenderingFusedBranchedFeedForward.applyTensorOps(
            input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
            cosine: scale, rowsPerSimdgroup: 16)
        }),
        ("tensor-ops rows32", {
          NeuralRenderingFusedBranchedFeedForward.applyTensorOps(
            input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
            cosine: scale, rowsPerSimdgroup: 32)
        }),
      ]
      var samples = [[Double]](repeating: [], count: variants.count)
      for _ in 0..<3 { for (_, body) in variants { eval(Device.withDefaultDevice(.gpu) { body() }) } }
      for _ in 0..<15 {
        for (index, (_, body)) in variants.enumerated() {
          let clock = ContinuousClock()
          let start = clock.now
          // Four chained launches per sample amortize the submission overhead.
          var result = Device.withDefaultDevice(.gpu) { body() }
          for _ in 0..<3 { result = Device.withDefaultDevice(.gpu) { body() } + result * 0 }
          eval(result)
          let duration = start.duration(to: clock.now)
          samples[index].append((Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15) / 4)
        }
      }
      for (index, (label, _)) in variants.enumerated() {
        let sorted = samples[index].sorted()
        print(String(format: "tensor-ops timing %dch %dx%d %@: min %.3f ms median %.3f ms", channels, height, width, label, sorted[0], sorted[sorted.count / 2]))
      }
    }
  }
}
