import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class NeuralRenderingStreamedGlobalAttentionTests: XCTestCase {
  func testStreamedAttentionMatchesFullGlobalReferenceIncludingRaggedTiles() {
    for tokens in [2, 6, 32, 34, 128, 510, 2040] {
      let tensors = inputs(tokens: tokens, heads: 2)
      let expected = reference(query: tensors.0, key: tensors.1, value: tensors.2)
      let actual = NeuralRenderingStreamedGlobalAttention.apply(query: tensors.0, key: tensors.1, value: tensors.2)
      eval(expected, actual)
      let delta = abs(actual.asType(.float32) - expected.asType(.float32))
      let maximum = delta.max().item(Float.self), mean = delta.mean().item(Float.self)
      print("streamed-global \(tokens) tokens: max \(maximum), mean \(mean)")
      XCTAssertEqual(maximum, 0, "Global attention must retain the reference rounding points at \(tokens) tokens")
    }
  }

  func testDifferentExtentsCanRemainPendingInTheSameGraph() {
    var expected: [MLXArray] = [], actual: [MLXArray] = []
    for tokens in [6, 34, 128, 510, 6] {
      let tensors = inputs(tokens: tokens, heads: 3)
      expected.append(reference(query: tensors.0, key: tensors.1, value: tensors.2))
      actual.append(NeuralRenderingStreamedGlobalAttention.apply(query: tensors.0, key: tensors.1, value: tensors.2))
    }
    eval(expected + actual)
    for (a, b) in zip(expected, actual) { XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self)) }
  }

  func testStreamedAttentionTimingWhenConfigured() throws {
    guard ProcessInfo.processInfo.environment["MLXDLSS_STREAMED_ATTENTION_TIMING"] == "1" else {
      throw XCTSkip("Set MLXDLSS_STREAMED_ATTENTION_TIMING=1 to measure both global attention paths")
    }
    for tokens in [48, 192, 510, 768, 2040] {
      let tensors = inputs(tokens: tokens, heads: 32, batches: 1)
      func run(_ streamed: Bool) -> MLXArray {
        streamed ? NeuralRenderingStreamedGlobalAttention.apply(query: tensors.0, key: tensors.1, value: tensors.2)
          : reference(query: tensors.0, key: tensors.1, value: tensors.2)
      }
      for streamed in [false, true, false] {
        for _ in 0..<4 { eval(run(streamed)) }
        var samples: [Double] = []
        for _ in 0..<20 {
          let started = ContinuousClock.now
          eval(run(streamed))
          let elapsed = started.duration(to: .now).components
          samples.append(Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15)
        }
        print("global-attention \(tokens) tokens 32h \(streamed ? "streamed" : "materialized"): \(samples.sorted()[samples.count / 2]) ms")
      }
      Memory.clearCache()
    }
  }

  func testRealGlobalStageParityAndTimingWhenConfigured() throws {
    let path = try htLogicalWeightsPath("Set MLXDLSS_LOGICAL_WEIGHTS for real global-stage parity")
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let reference = try NeuralRenderingGlobalStage(weights: weights)
    let fused = try NeuralRenderingGlobalStage(weights: weights, fusedOperations: true, compileGraph: true)
    let timing = ProcessInfo.processInfo.environment["MLXDLSS_STREAMED_ATTENTION_TIMING"] == "1"
    for tokens in [48, 192, 510, 768] {
      MLXRandom.seed(UInt64(tokens))
      let input = NeuralRenderingTransformerOperations.e4m3RoundTrip((MLXRandom.normal([1, 1, tokens, 1024]) * 0.2).asType(.float16))
      let a = reference(input), b = fused(input)
      eval(a, b)
      let delta = abs(a.asType(.float32) - b.asType(.float32))
      print("global-stage parity \(tokens): max \(delta.max().item(Float.self)), mean \(delta.mean().item(Float.self))")
      XCTAssertEqual(delta.max().item(Float.self), 0)
      if timing {
        for useFused in [false, true, false] {
          var samples: [Double] = []
          for iteration in 0..<24 {
            let started = ContinuousClock.now
            eval(useFused ? fused(input) : reference(input))
            let d = started.duration(to: .now).components
            if iteration >= 4 { samples.append(Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15) }
          }
          print("global-stage \(tokens) fused=\(useFused): \(samples.sorted()[samples.count / 2]) ms")
        }
      }
    }
  }

  func testCompiledGlobalBlockWhenConfigured() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXDLSS_GLOBAL_COMPILE_TIMING"] == "1", let path = env["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("Set MLXDLSS_GLOBAL_COMPILE_TIMING=1 and real weights") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let block = try NeuralRenderingGlobalBlock(weights: weights, blockIndex: 31, fusedOperations: true)
    let compiled = compile(inputs: []) { block($0) }
    for rows in [48, 192, 510, 768] {
      let x = NeuralRenderingTransformerOperations.e4m3RoundTrip((MLXRandom.normal([1, 1, rows, 1024]) * 0.2).asType(.float16))
      let a = block(x), b = compiled(x)
      eval(a, b)
      XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
      for useCompiled in [false, true, false] {
        var samples: [Double] = []
        for i in 0..<34 {
          let start = ContinuousClock.now
          eval(useCompiled ? compiled(x) : block(x))
          let t = start.duration(to: .now).components
          if i >= 4 { samples.append(Double(t.seconds) * 1000 + Double(t.attoseconds) / 1e15) }
        }
        print("global-compile \(rows) compiled=\(useCompiled): \(samples.sorted()[samples.count / 2]) ms")
      }
    }
  }

  private func inputs(tokens: Int, heads: Int, batches: Int = 2) -> (MLXArray, MLXArray, MLXArray) {
    MLXRandom.seed(UInt64(tokens))
    // Transposed batches exercise non-contiguous inputs to both implementations.
    let shape = [heads, batches, tokens, 32]
    let query = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      (MLXRandom.normal(shape) * 0.7).asType(.float16)).transposed(1, 0, 2, 3)
    let key = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      (MLXRandom.normal(shape) * 0.2).asType(.float16)).transposed(1, 0, 2, 3)
    let value = NeuralRenderingTransformerOperations.e4m3RoundTrip(
      (MLXRandom.normal(shape) * 0.8).asType(.float16)).transposed(1, 0, 2, 3)
    eval(query, key, value)
    return (query, key, value)
  }

  private func reference(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    let scores = clip(matmul(query, key.transposed(0, 1, 3, 2)), min: -3, max: 3)
    let probabilities = NeuralRenderingTransformerOperations.vendorApproximateSoftmax(scores)
    return NeuralRenderingTransformerOperations.e4m3RoundTrip(matmul(probabilities, value))
  }
}
