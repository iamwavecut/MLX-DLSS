import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// The global vit_1d block against the pipeline it replaced on 2026-10-17:
/// separate query/key/value publishes over transposed heads, the previous
/// resident attention kernel (below, verbatim), the transpose back, and the
/// residual followed by a separate E4M3 round trip. Outputs must agree bit
/// for bit; `MLXDLSS_GLOBAL_BLOCK_AB=1` also times both, alternating.
final class NeuralRenderingGlobalBlockABTests: XCTestCase {
  typealias Ops = NeuralRenderingTransformerOperations

  private static let previousHeader = Ops.e4m3MetalHeaderText + "\n" + #"""
    #include <metal_simdgroup_matrix>
    using namespace metal;
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)

    METAL_FUNC half2 vendor_weights(half2 score) {
      half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
      affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
      ushort2 bits = as_type<ushort2>(affine);
      uint packed = uint(bits.x) | (uint(bits.y) << 16);
      uint transformed = (packed << 5) + 0x7FF88000;
      return as_type<half2>(ushort2(ushort(transformed), ushort(transformed >> 16)));
    }
    """#

  private static let previousResident = MLXFast.metalKernel(
    name: "test_previous_global_attention_resident_v2", inputNames: ["query", "key", "value"], outputNames: ["output"],
    source: #"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint headBase = threadgroup_position_in_grid.z * pitch * 32;
      const uint row0 = threadgroup_position_in_grid.y * 8;
      threadgroup half S[8 * pitch];
      threadgroup half R[8];
      for (uint token0 = simd * 8; token0 < pitch; token0 += 32) {
        simdgroup_matrix<float, 8, 8> acc;
        acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> q, k;
          simdgroup_load(q, query + headBase + row0 * 32 + c, 32, ulong2(0), false);
          simdgroup_load(k, key + headBase + token0 * 32 + c, 32, ulong2(0), true);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(q.thread_elements()[0]);
          left.thread_elements()[1] = float(q.thread_elements()[1]);
          right.thread_elements()[0] = float(k.thread_elements()[0]);
          right.thread_elements()[1] = float(k.thread_elements()[1]);
          simdgroup_multiply_accumulate(acc, left, right, acc);
        }
        simdgroup_matrix<half, 8, 8> s;
        s.thread_elements()[0] = clamp(half(acc.thread_elements()[0]), -3.0h, 3.0h);
        s.thread_elements()[1] = clamp(half(acc.thread_elements()[1]), -3.0h, 3.0h);
        simdgroup_store(s, S + token0, pitch, ulong2(0), false);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Vendor weights for every valid score pair, all threads.
      const uint pairs = uint(tokens) / 2;
      for (uint i = tid; i < 8 * pairs; i += 128) {
        const uint row = i / pairs;
        const uint j = (i % pairs) * 2;
        half2 weight = vendor_weights(half2(S[row * pitch + j], S[row * pitch + j + 1]));
        S[row * pitch + j] = weight.x;
        S[row * pitch + j + 1] = weight.y;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // The sequential half denominator, one lane per query row.
      if (tid < 8) {
        half total = 0.0h;
        for (uint j = 0; j < uint(tokens); ++j) { total += S[tid * pitch + j]; }
        R[tid] = half(1.0f / float(total));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < 8 * pitch; i += 128) {
        const uint row = i / pitch;
        const uint j = i % pitch;
        S[i] = j < uint(tokens) ? half(float(mlxdlss_e4m3(S[i] * R[row]))) : 0.0h;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
      for (uint c = 0; c < pitch; c += 8) {
        simdgroup_matrix<half, 8, 8> p, v;
        simdgroup_load(p, S + c, pitch, ulong2(0), false);
        simdgroup_load(v, value + headBase + c * 32 + simd * 8, 32, ulong2(0), false);
        simdgroup_matrix<float, 8, 8> left, right;
        left.thread_elements()[0] = float(p.thread_elements()[0]);
        left.thread_elements()[1] = float(p.thread_elements()[1]);
        right.thread_elements()[0] = float(v.thread_elements()[0]);
        right.thread_elements()[1] = float(v.thread_elements()[1]);
        simdgroup_multiply_accumulate(acc, left, right, acc);
      }
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[1]))));
      simdgroup_store(result, output + headBase + row0 * 32 + simd * 8, 32, ulong2(0), false);
      """#, header: previousHeader)

  private struct Weights {
    let expansion, projection, feedForwardCosine, qkv, scale, attentionProjection, attentionCosine: MLXArray

    init(seed: UInt64) {
      let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 7)
      expansion = (MLXRandom.normal([1024, 4096], key: keys[0]) * 0.03).asType(.float16)
      projection = (MLXRandom.normal([4096, 1024], key: keys[1]) * 0.015).asType(.float16)
      feedForwardCosine = MLXRandom.uniform(low: 0.5, high: 1.0, [1024], key: keys[2]).asType(.float16)
      qkv = (MLXRandom.normal([1024, 3072], key: keys[3]) * 0.03).asType(.float16)
      scale = MLXRandom.uniform(low: 0.4, high: 2.0, [32], key: keys[4]).asType(.float16)
      attentionProjection = (MLXRandom.normal([1024, 1024], key: keys[5]) * 0.03).asType(.float16)
      attentionCosine = MLXRandom.uniform(low: 0.5, high: 1.0, [1024], key: keys[6]).asType(.float16)
    }

    var validated: ValidatedWeights {
      ValidatedWeights(arrays: [
        "block0.layer0.weight": expansion, "block0.layer1.weight": projection,
        "block0.layer1.ffn_cos_skip": feedForwardCosine, "block0.layer2.qkv_weight": qkv,
        "block0.layer2.attn_scale": scale, "block0.layer4.projection_weight": attentionProjection,
        "block0.layer4.attn_cos_skip": attentionCosine,
      ])
    }
  }

  private static func previousBlock(_ input: MLXArray, _ w: Weights) -> MLXArray {
    let shape = input.shape
    let tokenCount = shape[1] * shape[2]
    let tokens = input.reshaped([1, tokenCount, 1024])
    let published = Ops.quadraticGatePublish(matmul(tokens, w.expansion))
    let feedForwardOutput = Ops.cosineResidual(skip: tokens, branch: matmul(published, w.projection), cosine: w.feedForwardCosine)
    let qkv = split(matmul(feedForwardOutput, w.qkv), parts: 3, axis: -1)
    let heads = qkv.map { $0.reshaped([1, tokenCount, 32, 32]).transposed(0, 2, 1, 3) }
    let pitch = (tokenCount + 7) / 8 * 8
    func pad(_ x: MLXArray) -> MLXArray {
      pitch == tokenCount ? x : padded(x, widths: [[0, 0], [0, 0], [0, pitch - tokenCount], [0, 0]])
    }
    let attended = previousResident(
      [pad(Ops.vendorCosinePublish(heads[0], scale: w.scale * Float(32).squareRoot())),
       pad(Ops.vendorCosinePublish(heads[1])), pad(Ops.e4m3RoundTrip(heads[2]))],
      template: [("tokens", tokenCount), ("pitch", pitch)],
      grid: (128, pitch / 8, 32), threadGroup: (128, 1, 1),
      outputShapes: [[1, 32, pitch, 32]], outputDTypes: [.float16])[0][0..., 0..., 0..<tokenCount, 0...]
    let branch = matmul(attended.transposed(0, 2, 1, 3).reshaped([1, tokenCount, 1024]), w.attentionProjection)
    return Ops.e4m3RoundTrip(Ops.cosineResidual(skip: feedForwardOutput, branch: branch, cosine: w.attentionCosine))
      .reshaped(shape)
  }

  private func blocks(seed: UInt64) throws -> (previous: (MLXArray) -> MLXArray, current: NeuralRenderingGlobalBlock) {
    let w = Weights(seed: seed)
    let current = try NeuralRenderingGlobalBlock(
      weights: w.validated, blockIndex: 0, preciseAttention: true, fusedOperations: true, compileGraph: true)
    return (compile(inputs: []) { (input: MLXArray) -> MLXArray in Self.previousBlock(input, w) }, current)
  }

  func testGlobalBlockMatchesPreviousPipelineBitForBit() throws {
    let (previous, current) = try blocks(seed: 11)
    for (height, width) in [(1, 2), (2, 17), (8, 8), (12, 16), (17, 30), (20, 32), (32, 32)] {
      let input = Ops.e4m3RoundTrip(
        (MLXRandom.normal([1, height, width, 1024], key: MLXRandom.key(UInt64(height * 100 + width))) * 0.6).asType(.float16))
      let (expected, actual) = Device.withDefaultDevice(.gpu) { (previous(input), current(input)) }
      eval(expected, actual)
      let differing = (expected.view(dtype: .uint16) .!= actual.view(dtype: .uint16)).sum().item(Int32.self)
      XCTAssertEqual(differing, 0, "\(height)x\(width) tokens")
    }
  }

  /// GPU time this process accumulated (IOKit's per-client `accumulatedGPUTime`),
  /// which other GPU clients inflate far less than wall time.
  private func processGPUNanoseconds() -> UInt64 {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    process.arguments = ["-r", "-c", "AGXDeviceUserClient", "-l", "-w0"]
    let pipe = Pipe()
    process.standardOutput = pipe
    guard (try? process.run()) != nil else { return 0 }
    let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    process.waitUntilExit()
    let pattern = try! NSRegularExpression(pattern: #""accumulatedGPUTime"=(\d+)"#)
    var total: UInt64 = 0
    for client in text.components(separatedBy: "+-o AGXDeviceUserClient").dropFirst()
    where client.contains("\"IOUserClientCreator\" = \"pid \(getpid()),") {
      for match in pattern.matches(in: client, range: NSRange(client.startIndex..., in: client)) {
        total += UInt64(client[Range(match.range(at: 1), in: client)!]) ?? 0
      }
    }
    return total
  }

  func testGlobalBlockTimingAgainstPreviousPipeline() throws {
    guard ProcessInfo.processInfo.environment["MLXDLSS_GLOBAL_BLOCK_AB"] == "1" else { throw XCTSkip("MLXDLSS_GLOBAL_BLOCK_AB=1") }
    let (previous, current) = try blocks(seed: 5)
    let rounds = Int(ProcessInfo.processInfo.environment["MLXDLSS_GLOBAL_BLOCK_AB_ROUNDS"] ?? "30") ?? 30
    // One sample chains the eight blocks of a global stage (the same weights)
    // twice; wall and GPU milliseconds are per eight-block stage.
    let calls = 2
    func sample(_ body: (MLXArray) -> MLXArray, _ input: MLXArray) -> (Double, Double) {
      let gpuStart = processGPUNanoseconds()
      let start = ContinuousClock.now
      var outputs: [MLXArray] = []
      Device.withDefaultDevice(.gpu) {
        for _ in 0..<calls { outputs.append((0..<8).reduce(input) { value, _ in body(value) }) }
      }
      eval(outputs)
      let elapsed = start.duration(to: .now)
      let gpu = Double(processGPUNanoseconds() - gpuStart) / 1e6 / Double(calls)
      return ((Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15) / Double(calls), gpu)
    }
    for (height, width) in [(20, 32), (12, 16)] {
      let input = Ops.e4m3RoundTrip((MLXRandom.normal([1, height, width, 1024]) * 0.6).asType(.float16))
      eval(input)
      for _ in 0..<3 { _ = sample(previous, input); _ = sample(current.callAsFunction, input) }
      var before: [(Double, Double)] = [], after: [(Double, Double)] = []
      for _ in 0..<rounds {
        before.append(sample(previous, input))
        after.append(sample(current.callAsFunction, input))
      }
      // Other GPU clients only add time, so the low quantiles are the cleanest.
      func quantile(_ values: [Double], _ q: Double) -> Double { values.sorted()[Int(Double(values.count - 1) * q)] }
      func line(_ values: [Double]) -> String {
        String(format: "min %.3f p25 %.3f median %.3f", quantile(values, 0), quantile(values, 0.25), quantile(values, 0.5))
      }
      let wall = (before.map(\.0), after.map(\.0)), gpu = (before.map(\.1), after.map(\.1))
      print("global-stage-ab \(height * width) tokens wall ms: previous \(line(wall.0)); current \(line(wall.1)); "
        + String(format: "p25 %+.1f%%", 100 * (quantile(wall.1, 0.25) / quantile(wall.0, 0.25) - 1)))
      print("global-stage-ab \(height * width) tokens GPU ms: previous \(line(gpu.0)); current \(line(gpu.1)); "
        + String(format: "median %+.1f%%", 100 * (quantile(gpu.1, 0.5) / quantile(gpu.0, 0.5) - 1)))
    }
  }
}
