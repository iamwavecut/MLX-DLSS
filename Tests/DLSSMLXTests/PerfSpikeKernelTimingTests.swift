import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in performance diagnostics of the fused float16 path (`MLXDLSS_PERF_SPIKE=1`,
/// most cases also need `MLXDLSS_LOGICAL_WEIGHTS`): kernel timings, exactness of the
/// optional paths against the per-operation chains, and whole-frame A/B of the
/// runtime toggles. The exactness cases assert; the timings are diagnostics.
final class PerfSpikeKernelTimingTests: XCTestCase {
  private var enabled: Bool { ProcessInfo.processInfo.environment["MLXDLSS_PERF_SPIKE"] == "1" }

  private func ms(_ body: () -> [MLXArray], warm: Int = 3, runs: Int = 7) -> Double {
    for _ in 0..<warm { eval(Device.withDefaultDevice(.gpu) { body() }) }
    var best = Double.infinity
    for _ in 0..<runs {
      let clock = ContinuousClock(); let start = clock.now
      let out = Device.withDefaultDevice(.gpu) { body() }
      eval(out)
      let d = start.duration(to: clock.now)
      best = min(best, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
    }
    return best
  }

  private func report(_ label: String, _ value: Double, _ extra: String = "") {
    print("perf-spike \(label): \(String(format: "%.3f", value)) ms \(extra)")
  }

  // MARK: 1. simdgroup MMA throughput: half vs float vs half-loaded-converted-to-float

  private static let mmaKernel = MLXFast.metalKernel(
    name: "perf_spike_mma",
    inputNames: ["seed"],
    outputNames: ["output"],
    source: #"""
      threadgroup half tgHalf[4 * 64];
      threadgroup float tgFloat[4 * 64];
      const uint tid = thread_position_in_threadgroup.x;
      for (uint i = tid; i < 256; i += 128) { tgHalf[i] = half(i % 13) * 0.01h; tgFloat[i] = float(i % 13) * 0.01f; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint gid = thread_position_in_grid.x;
      simdgroup_matrix<half, 8, 8> ah;
      ah.thread_elements()[0] = half(gid % 7) * 0.01h;
      ah.thread_elements()[1] = half(seed[0]);
      simdgroup_matrix<float, 8, 8> af;
      af.thread_elements()[0] = float(gid % 7) * 0.01f;
      af.thread_elements()[1] = float(seed[0]);
      simdgroup_matrix<half, 8, 8> bhReg[4];
      simdgroup_matrix<float, 8, 8> bfReg[4];
      for (int i = 0; i < 4; ++i) {
        simdgroup_load(bhReg[i], tgHalf + i * 64, 8, ulong2(0), false);
        simdgroup_load(bfReg[i], tgFloat + i * 64, 8, ulong2(0), false);
      }
      simdgroup_matrix<half, 8, 8> acch[8];
      simdgroup_matrix<float, 8, 8> accf[8];
      for (int i = 0; i < 8; ++i) {
        acch[i].thread_elements()[0] = 0.0h; acch[i].thread_elements()[1] = 0.0h;
        accf[i].thread_elements()[0] = 0.0f; accf[i].thread_elements()[1] = 0.0f;
      }
      for (uint it = 0; it < uint(iterations); ++it) {
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 8; ++i) {
          if (mode == 0) {
            simdgroup_matrix<half, 8, 8> b;
            simdgroup_load(b, tgHalf + (i % 4) * 64, 8, ulong2(0), false);
            simdgroup_multiply_accumulate(acch[i], ah, b, acch[i]);
          } else if (mode == 1) {
            simdgroup_matrix<float, 8, 8> b;
            simdgroup_load(b, tgFloat + (i % 4) * 64, 8, ulong2(0), false);
            simdgroup_multiply_accumulate(accf[i], af, b, accf[i]);
          } else if (mode == 2) {
            simdgroup_matrix<half, 8, 8> bh;
            simdgroup_load(bh, tgHalf + (i % 4) * 64, 8, ulong2(0), false);
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = float(bh.thread_elements()[0]);
            b.thread_elements()[1] = float(bh.thread_elements()[1]);
            simdgroup_multiply_accumulate(accf[i], af, b, accf[i]);
          } else if (mode == 3) {
            simdgroup_multiply_accumulate(acch[i], ah, bhReg[i % 4], acch[i]);
          } else if (mode == 4) {
            simdgroup_multiply_accumulate(accf[i], af, bfReg[i % 4], accf[i]);
          } else if (mode == 6) {
            simdgroup_multiply_accumulate(accf[i], ah, bhReg[i % 4], accf[i]);
          } else if (mode == 7) {
            simdgroup_matrix<half, 8, 8> bh;
            simdgroup_load(bh, tgHalf + (i % 4) * 64, 8, ulong2(0), false);
            simdgroup_multiply_accumulate(accf[i], ah, bh, accf[i]);
          } else {
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = float(bhReg[i % 4].thread_elements()[0]);
            b.thread_elements()[1] = float(bhReg[i % 4].thread_elements()[1]);
            simdgroup_multiply_accumulate(accf[i], af, b, accf[i]);
          }
        }
      }
      float total = 0.0f;
      for (int i = 0; i < 8; ++i) {
        total += float(acch[i].thread_elements()[0]) + float(acch[i].thread_elements()[1]);
        total += accf[i].thread_elements()[0] + accf[i].thread_elements()[1];
      }
      output[gid] = total;
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )

  func testMMAThroughput() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    let threadgroups = 38 * 32
    let iterations = 400
    let seed = MLXArray([Float(0.001)])
    for (mode, name) in [(0, "half tg-load per MMA [unrolled]"), (1, "float tg-load per MMA [unrolled]"), (2, "half tg-load + convert, float MMA [unrolled]"), (3, "half register operands [unrolled]"), (4, "float register operands [unrolled]"), (5, "half registers converted per MMA, float MMA [unrolled]"), (6, "mixed: half operands, float accumulator [unrolled]"), (7, "mixed: half tg-load, float accumulator [unrolled]")] {
      let t = ms({
        [Self.mmaKernel([seed], template: [("iterations", iterations), ("mode", mode)],
          grid: (threadgroups * 128, 1, 1), threadGroup: (128, 1, 1),
          outputShapes: [[threadgroups * 128]], outputDTypes: [.float32])[0]]
      })
      let flops = Double(threadgroups) * 4 * 8 * Double(iterations) * 1024
      report("mma \(name)", t, String(format: "= %.2f TFLOPS", flops / t / 1e9))
    }
  }

  // MARK: 2. elementwise publish kernels as bandwidth probes

  func testElementwiseBandwidth() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    let x = (MLXRandom.normal([1, 1088, 1920, 32]) * 0.5).asType(.float16)
    eval(x)
    let bytes = Double(x.size * 2 * 2)
    let t1 = ms({ [NeuralRenderingTransformerOperations.e4m3RoundTrip(x)] })
    report("e4m3RoundTrip [1,1088,1920,32] half (1 elem/thread)", t1, String(format: "= %.0f GB/s", bytes / t1 / 1e6))
    let t2 = ms({ [NeuralRenderingTransformerOperations.quadraticGatePublish(x)] })
    report("quadraticGatePublish same tensor (half4/thread)", t2, String(format: "= %.0f GB/s", bytes / t2 / 1e6))
    let cosine = MLXArray.ones([32], dtype: .float16)
    let t3 = ms({ [NeuralRenderingFusedWindowAttention.cosineResidual(skip: x, branch: x, cosine: cosine, publish: true)] })
    report("fused cosineResidual+publish (half4/thread, 2 reads)", t3, String(format: "= %.0f GB/s", bytes * 1.5 / t3 / 1e6))
    let t4 = ms({ [x + x * cosine] })
    report("MLX branch + skip*cosine (2 kernels)", t4)
    let t5 = ms({ [NeuralRenderingTransformerOperations.e4m3RoundTrip(x + x * cosine)] })
    report("MLX residual + e4m3 (3 kernels, the unfused block tail)", t5)
  }

  // MARK: 3. fused multi-head window attention core per 1080p level

  func testFusedWindowAttentionCore() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    for (heads, height, width) in [(2, 272, 480), (4, 136, 240), (8, 68, 120), (16, 34, 60)] {
      let channels = heads * 32
      let qkv = (MLXRandom.normal([1, height, width, channels * 3]) * 0.5).asType(.float16)
      let scale = MLXRandom.uniform(low: 0.5, high: 2.0, [heads]).asType(.float16)
      let bias = (MLXRandom.normal([heads, 64, 64]) * 2.0).asType(.float16)
      let x = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
      let qkvWeight = (MLXRandom.normal([channels, channels * 3]) * 0.05).asType(.float16)
      let projection = (MLXRandom.normal([channels, channels]) * 0.05).asType(.float16)
      eval(qkv, scale, bias, x, qkvWeight, projection)
      let windows = Double((height + 7) / 8 * ((width + 7) / 8))
      let macs = windows * Double(heads) * (64.0 * 64 * 32 * 2)
      let t = ms({ [NeuralRenderingFusedWindowAttention.apply(qkv: qkv, attentionScale: scale, attentionBias: bias, headCount: heads, windowOrigin: .zero)] })
      report("window attention core \(heads)h \(height)x\(width)", t,
        String(format: "windows*heads=%.0f, %.2f GMAC, %.2f TMAC/s", windows * Double(heads), macs / 1e9, macs / t / 1e9))
      let tq = ms({ [matmul(x, qkvWeight)] })
      report("  qkv matmul [rows,\(channels)]x[\(channels),\(channels * 3)]", tq,
        String(format: "%.2f TMAC/s", Double(height * width) * Double(channels * channels * 3) / tq / 1e9))
      let tp = ms({ [matmul(x, projection)] })
      report("  projection matmul [rows,\(channels)]x[\(channels),\(channels)]", tp)
    }
  }

  // MARK: 4. branched FFN pieces (dense arrangement) per level

  func testBranchedFeedForwardPieces() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    typealias Ops = NeuralRenderingTransformerOperations
    for (groups, height, width) in [(2, 272, 480), (4, 136, 240), (8, 68, 120)] {
      let channels = groups * 32
      let rows = height * width
      let input = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
      let expansion = (MLXRandom.normal([groups, 4, groups, 32, 32]) * 0.05).asType(.float16)
      let branch = (MLXRandom.normal([groups, 4, 32, 32]) * 0.05).asType(.float16)
      let output = (MLXRandom.normal([channels, channels]) * 0.05).asType(.float16)
      eval(input, expansion, branch, output)
      let dense = Ops.denseExpansionWeight(expansion)
      let grouped = Ops.groupedProjectionWeight(branch)
      let flat = input.reshaped([rows, channels])
      let expanded = matmul(flat, dense); eval(expanded)
      let gated = Ops.quadraticGatePublish(expanded); eval(gated)
      let projected = matmul(gated.reshaped([rows, groups, 128]).transposed(1, 0, 2), grouped).transposed(1, 0, 2).reshaped([rows, channels])
      eval(projected)
      let published = Ops.e4m3RoundTrip(projected); eval(published)
      let tAll = ms({ [Ops.denseBranchedFeedForward(input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output)] })
      report("branched FFN \(channels)ch \(height)x\(width) whole", tAll)
      let t1 = ms({ [matmul(flat, dense)] })
      report("  expansion matmul [\(rows),\(channels)]x[\(channels),\(channels * 4)]", t1,
        String(format: "%.2f TMAC/s", Double(rows) * Double(channels * channels * 4) / t1 / 1e9))
      let t2 = ms({ [Ops.quadraticGatePublish(expanded)] })
      report("  gate+publish on [\(rows),\(channels * 4)]", t2)
      let t3 = ms({ [matmul(gated.reshaped([rows, groups, 128]).transposed(1, 0, 2), grouped)] })
      report("  grouped matmul incl. transpose copy", t3)
      let t3b = ms({ [matmul(gated.reshaped([rows, groups, 128]).transposed(1, 0, 2), grouped).transposed(1, 0, 2).reshaped([rows, channels])] })
      report("  grouped matmul + transpose back (lazy, no copy yet)", t3b)
      let t4 = ms({ [Ops.e4m3RoundTrip(projected)] })
      report("  e4m3 on [\(rows),\(channels)] contiguous", t4)
      let t4b = ms({ [Ops.e4m3RoundTrip(matmul(gated.reshaped([rows, groups, 128]).transposed(1, 0, 2), grouped).transposed(1, 0, 2).reshaped([rows, channels]))] })
      report("  grouped matmul + transpose back + e4m3 (forces copy)", t4b)
      let t5 = ms({ [matmul(published, output)] })
      report("  output projection matmul", t5)
    }
  }

  // MARK: 5. fused single-head window block

  func testFusedWindowBlock1h() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    for (height, width) in [(544, 960), (1088, 1920)] {
      let x = (MLXRandom.normal([1, height, width, 32]) * 0.5).asType(.float16)
      let w1 = (MLXRandom.normal([32, 128]) * 0.1).asType(.float16)
      let w2 = (MLXRandom.normal([128, 32]) * 0.1).asType(.float16)
      let cos1 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
      let qkv = (MLXRandom.normal([32, 96]) * 0.1).asType(.float16)
      let scale = MLXArray([Float(1.2)]).asType(.float16)
      let bias = (MLXRandom.normal([1, 64, 64]) * 2.0).asType(.float16)
      let proj = (MLXRandom.normal([32, 32]) * 0.1).asType(.float16)
      let cos2 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
      eval(x, w1, w2, cos1, qkv, scale, bias, proj, cos2)
      let t = ms({ [NeuralRenderingFusedWindowBlock.apply(x, expansionWeight: w1, feedForwardProjectionWeight: w2, feedForwardCosine: cos1, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, attentionProjectionWeight: proj, attentionCosine: cos2, windowOrigin: .zero, publish: true)] })
      let tokens = Double(height * width)
      let macs = tokens * (32.0 * 128 * 2 + 32 * 96 + 64 * 32 * 2 + 32 * 32)
      report("fused 1h window block \(height)x\(width)", t, String(format: "%.2f GMAC, %.2f TMAC/s, %.0f GB/s in+out", macs / 1e9, macs / t / 1e9, tokens * 64 * 2 / t / 1e6))
    }
  }

  // MARK: 6. whole frame on real weights

  func testWholeFrame() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let model = try NeuralRenderingTransformerModel(weights: weights, compileBlocks: true)
    let shapes = (ProcessInfo.processInfo.environment["MLXDLSS_PERF_SHAPES"] ?? "768x1024,1088x1920").split(separator: ",")
    for shape in shapes {
      let parts = shape.split(separator: "x").compactMap { Int($0) }
      let (height, width) = (parts[0], parts[1])
      let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
      eval(input)
      let t = ms({ [model(input)] }, warm: 2, runs: 5)
      let mpx = Double(height * width) / 1e6
      report("whole frame \(height)x\(width)", t, String(format: "%.2f Mpx, %.1f ms/Mpx", mpx, t / mpx))
    }
  }

  // MARK: 7. exhaustive E4M3 check on the GPU: fast 16-bit path vs the float reference

  func testFastE4M3IsBitExactOnGPU() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    typealias Ops = NeuralRenderingTransformerOperations
    let header = Ops.e4m3ReferenceMetalHeaderText.replacingOccurrences(of: "mlxdlss_e4m3", with: "mlxdlss_e4m3_ref")
      + "\n" + Ops.e4m3FastMetalHeaderText.replacingOccurrences(of: "mlxdlss_e4m3", with: "mlxdlss_e4m3_fast")
    let kernel = MLXFast.metalKernel(
      name: "perf_spike_e4m3_exhaustive", inputNames: ["input"], outputNames: ["mismatch"],
      source: #"""
        const uint i = thread_position_in_grid.x;
        if (i >= 65536) { return; }
        const half h = as_type<half>(ushort(input[i]));
        const ushort ref = as_type<ushort>(half(float(mlxdlss_e4m3_ref(h))));
        const ushort fastHalf = as_type<ushort>(half(float(mlxdlss_e4m3_fast(h))));
        const ushort fastFloat = as_type<ushort>(half(float(mlxdlss_e4m3_fast(float(h)))));
        const ushort refFloat = as_type<ushort>(half(float(mlxdlss_e4m3_ref(float(h)))));
        mismatch[i] = uint32_t(ref != fastHalf) + 1000u * uint32_t(ref != fastFloat) + 1000000u * uint32_t(ref != refFloat);
        """#,
      header: header)
    let input = MLXArray((0..<65536).map { UInt32($0) })
    let out = kernel([input], grid: (65536, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[65536]], outputDTypes: [.uint32])[0]
    eval(out)
    let values = out.asArray(UInt32.self)
    let halfMismatch = values.filter { $0 % 1000 != 0 }.count
    let floatMismatch = values.filter { ($0 / 1000) % 1000 != 0 }.count
    let refFloatMismatch = values.filter { $0 / 1000000 != 0 }.count
    let bad = values.enumerated().filter { $0.element % 1000 != 0 }.map { $0.offset }
    let nan = bad.filter { ($0 & 0x7C00) == 0x7C00 && ($0 & 0x3FF) != 0 }.count
    let negative = bad.filter { $0 & 0x8000 != 0 }.count
    print("perf-spike e4m3 exhaustive: half-path mismatches \(halfMismatch) (nan patterns \(nan), negative \(negative), first \(bad.prefix(6).map { String($0, radix: 16) })), float-ctor mismatches \(floatMismatch), ref float-vs-half \(refFloatMismatch)")
    XCTAssertEqual(halfMismatch, 0)
    XCTAssertEqual(floatMismatch, 0)
  }

  // MARK: 8. post head, fused output residual, global attention v2: exactness + timing

  func testFusedHeadIsBitExact() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    for (height, width) in [(64, 64), (544, 960), (1088, 1920)] {
      let features = (MLXRandom.normal([1, height, width, 32]) * 2).asType(.float16)
      let gain = (MLXRandom.normal([16, 4]) * 0.3).asType(.float16)
      let convolution = (MLXRandom.normal([16, 4]) * 0.3).asType(.float16)
      let zeros = MLXArray.zeros([16, 4], dtype: .float16)
      let headWeight = concatenated([concatenated([gain, zeros], axis: 0), concatenated([zeros, convolution], axis: 0)], axis: 1)
      eval(features, gain, convolution, headWeight)
      let reference = NeuralRenderingPostBlock.head(features, gain: gain, convolution: convolution, headWeight: headWeight, fused: false)
      let fused = NeuralRenderingPostBlock.head(features, gain: gain, convolution: convolution, headWeight: headWeight, fused: true)
      eval(reference, fused)
      let delta = abs(reference.asType(.float32) - fused.asType(.float32)).max().item(Float.self)
      let tr = ms({ [NeuralRenderingPostBlock.head(features, gain: gain, convolution: convolution, headWeight: headWeight, fused: false)] })
      let tf = ms({ [NeuralRenderingPostBlock.head(features, gain: gain, convolution: convolution, headWeight: headWeight, fused: true)] })
      print("perf-spike post head \(height)x\(width): max |Δ| \(delta); reference \(String(format: "%.3f", tr)) ms, fused \(String(format: "%.3f", tf)) ms")
      XCTAssertEqual(delta, 0)
    }
  }

  func testFusedOutputResidualIsBitExact() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    typealias Ops = NeuralRenderingTransformerOperations
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let saved = Ops.fusedOutputResidualEnabled
    defer { Ops.fusedOutputResidualEnabled = saved }
    for (block, channels, hidden, heads, height, width) in [(5, 64, 224, 2, 68, 120), (15, 256, 704, 8, 17, 30), (57, 128, 384, 4, 34, 60)] {
      let input = (MLXRandom.normal([1, height, width, channels]) * 0.5).asType(.float16)
      eval(input)
      Ops.fusedOutputResidualEnabled = false
      let reference = try NeuralRenderingWindowBlock(weights: weights, blockIndex: block, channels: channels, hiddenChannels: hidden, headCount: heads, compileBlock: true)
      let referenceOut = reference(input); eval(referenceOut)
      let referenceUnpublished = reference.unpublished(input); eval(referenceUnpublished)
      Ops.fusedOutputResidualEnabled = true
      let fused = try NeuralRenderingWindowBlock(weights: weights, blockIndex: block, channels: channels, hiddenChannels: hidden, headCount: heads, compileBlock: true)
      let fusedOut = fused(input); eval(fusedOut)
      let fusedUnpublished = fused.unpublished(input); eval(fusedUnpublished)
      let d1 = abs(referenceOut.asType(.float32) - fusedOut.asType(.float32)).max().item(Float.self)
      let d2 = abs(referenceUnpublished.asType(.float32) - fusedUnpublished.asType(.float32)).max().item(Float.self)
      Ops.fusedOutputResidualEnabled = false
      let tr = ms({ [reference(input)] })
      Ops.fusedOutputResidualEnabled = true
      let tf = ms({ [fused(input)] })
      print("perf-spike fused output residual block \(block) \(heads)h \(height)x\(width): max |Δ| published \(d1) unpublished \(d2); reference \(String(format: "%.3f", tr)) ms, fused \(String(format: "%.3f", tf)) ms")
      XCTAssertEqual(d1, 0); XCTAssertEqual(d2, 0)
    }
    // split family
    let input = (MLXRandom.normal([1, 36, 60, 512]) * 0.5).asType(.float16)
    eval(input)
    Ops.fusedOutputResidualEnabled = false
    let reference = try NeuralRenderingSplitWindowBlock(weights: weights, blockIndex: 41, preciseAttention: true, fusedFeedForward: true)
    let referenceOut = reference(input); eval(referenceOut)
    Ops.fusedOutputResidualEnabled = true
    let fusedOut = reference(input); eval(fusedOut)
    let d = abs(referenceOut.asType(.float32) - fusedOut.asType(.float32)).max().item(Float.self)
    Ops.fusedOutputResidualEnabled = false
    let tr = ms({ [reference(input)] })
    Ops.fusedOutputResidualEnabled = true
    let tf = ms({ [reference(input)] })
    print("perf-spike fused output residual split block 41 36x60: max |Δ| \(d); reference \(String(format: "%.3f", tr)) ms, fused \(String(format: "%.3f", tf)) ms")
    XCTAssertEqual(d, 0)
  }

  private func globalInputs(tokens: Int, heads: Int) -> (MLXArray, MLXArray, MLXArray) {
    typealias Ops = NeuralRenderingTransformerOperations
    let q = (MLXRandom.normal([1, heads, tokens, 32]) * 0.4).asType(.float16)
    let k = (MLXRandom.normal([1, heads, tokens, 32]) * 0.4).asType(.float16)
    let v = (MLXRandom.normal([1, heads, tokens, 32]) * 0.4).asType(.float16)
    let query = Ops.vendorCosinePublish(q, scale: MLXArray.ones([heads], dtype: .float16) * 5.6)
    let key = Ops.vendorCosinePublish(k)
    let value = Ops.e4m3RoundTrip(v)
    eval(query, key, value)
    return (query, key, value)
  }

  private func globalReference(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    typealias Ops = NeuralRenderingTransformerOperations
    let scores = matmul(query, key.transposed(0, 1, 3, 2))
    let clipped = maximum(minimum(scores, MLXArray(Float(3)).asType(.float16)), MLXArray(Float(-3)).asType(.float16))
    return Ops.e4m3RoundTrip(matmul(Ops.vendorApproximateSoftmax(clipped), value))
  }

  func testGlobalAttentionV2IsBitExactAndTiming() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    let saved = NeuralRenderingStreamedGlobalAttention.v2Enabled
    defer { NeuralRenderingStreamedGlobalAttention.v2Enabled = saved }
    for tokens in [2, 6, 34, 48, 128, 192, 510, 640, 768, 1024, 1030, 2040] {
      let heads = tokens > 1000 ? 8 : 32
      let (query, key, value) = globalInputs(tokens: tokens, heads: heads)
      let reference = globalReference(query: query, key: key, value: value)
      NeuralRenderingStreamedGlobalAttention.v2Enabled = false
      let v1 = NeuralRenderingStreamedGlobalAttention.apply(query: query, key: key, value: value)
      NeuralRenderingStreamedGlobalAttention.v2Enabled = true
      let v2 = NeuralRenderingStreamedGlobalAttention.apply(query: query, key: key, value: value)
      eval(reference, v1, v2)
      let d1 = abs(reference.asType(.float32) - v1.asType(.float32)).max().item(Float.self)
      let d2 = abs(reference.asType(.float32) - v2.asType(.float32)).max().item(Float.self)
      let tm = ms({ [globalReference(query: query, key: key, value: value)] })
      NeuralRenderingStreamedGlobalAttention.v2Enabled = false
      let t1 = ms({ [NeuralRenderingStreamedGlobalAttention.apply(query: query, key: key, value: value)] })
      NeuralRenderingStreamedGlobalAttention.v2Enabled = true
      let t2 = ms({ [NeuralRenderingStreamedGlobalAttention.apply(query: query, key: key, value: value)] })
      print("perf-spike global attention \(tokens) tokens \(heads)h: max |Δ| v1 \(d1) v2 \(d2); materialized \(String(format: "%.3f", tm)) ms, v1 \(String(format: "%.3f", t1)) ms, v2 \(String(format: "%.3f", t2)) ms")
      XCTAssertEqual(d2, 0, "v2 must match the reference at \(tokens) tokens")
    }
  }

  // MARK: 9. whole frame checksum + peak memory (compare across env toggles)

  func testWholeFrameChecksum() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let model = try NeuralRenderingTransformerModel(weights: weights, compileBlocks: true)
    let shapes = (ProcessInfo.processInfo.environment["MLXDLSS_PERF_SHAPES"] ?? "768x1024,1088x1920").split(separator: ",")
    for shape in shapes {
      let parts = shape.split(separator: "x").compactMap { Int($0) }
      let (height, width) = (parts[0], parts[1])
      let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
      eval(input)
      Memory.clearCache()
      Memory.peakMemory = 0
      let output = Device.withDefaultDevice(.gpu) { model(input) }
      eval(output)
      let f32 = output.asType(.float32)
      let checksum = abs(f32).sum().item(Float.self)
      let mean = f32.mean().item(Float.self)
      let peak = Double(Memory.peakMemory) / 1_073_741_824
      let t = ms({ [model(input)] }, warm: 1, runs: 5)
      print("perf-spike checksum \(height)x\(width): sum|y| \(checksum) mean \(mean) peak \(String(format: "%.2f", peak)) GB time \(String(format: "%.1f", t)) ms")
    }
  }

  // MARK: 10. in-process A/B of runtime toggles (alternating rounds, same process state)

  func testWholeFrameAB() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    typealias Ops = NeuralRenderingTransformerOperations
    let env = ProcessInfo.processInfo.environment
    // "baseline" is the pre-toggle behaviour (per-block global eval, float E4M3 path
    // aside, no fused head/residual, v1 global attention); list the toggles to enable.
    let configs = (env["MLXDLSS_PERF_AB"] ?? "baseline;GLOBAL_EVAL=0,GLOBAL_ATTENTION_V2=1,FUSED_HEAD=1,FUSED_OUTPUT_RESIDUAL=1,E4M3_VEC=1").split(separator: ";").map(String.init)
    let rounds = Int(env["MLXDLSS_PERF_ROUNDS"] ?? "3") ?? 3
    let shapes = (env["MLXDLSS_PERF_SHAPES"] ?? "768x1024,1088x1920").split(separator: ",").map { $0.split(separator: "x").compactMap { Int($0) } }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    func apply(_ config: String) {
      NeuralRenderingGlobalStage.perBlockEvalEnabled = true
      NeuralRenderingPostBlock.fusedHeadEnabled = false
      Ops.fusedOutputResidualEnabled = false
      NeuralRenderingStreamedGlobalAttention.v2Enabled = false
      Ops.vectorizedE4M3Enabled = false
      NeuralRenderingFusedWindowBlock.shuffleReciprocals = false
      for item in config.split(separator: ",") {
        let kv = item.split(separator: "=").map(String.init)
        guard kv.count == 2 else { continue }
        switch kv[0] {
        case "GLOBAL_EVAL": NeuralRenderingGlobalStage.perBlockEvalEnabled = kv[1] != "0"
        case "FUSED_HEAD": NeuralRenderingPostBlock.fusedHeadEnabled = kv[1] == "1"
        case "FUSED_OUTPUT_RESIDUAL": Ops.fusedOutputResidualEnabled = kv[1] == "1"
        case "GLOBAL_ATTENTION_V2": NeuralRenderingStreamedGlobalAttention.v2Enabled = kv[1] == "1"
        case "E4M3_VEC": Ops.vectorizedE4M3Enabled = kv[1] == "1"
        case "WINDOW_SHUFFLE": NeuralRenderingFusedWindowBlock.shuffleReciprocals = kv[1] == "1"
        default: print("perf-spike AB: unknown toggle \(kv[0])")
        }
      }
    }
    var models: [NeuralRenderingTransformerModel] = []
    for config in configs {
      apply(config)
      models.append(try NeuralRenderingTransformerModel(weights: weights, compileBlocks: true))
    }
    print("perf-spike AB: fast e4m3 header \(Ops.fastE4M3Enabled ? "ON" : "off"), configs \(configs)")
    var results: [String: [Double]] = [:]
    var checksums: [String: Float] = [:]
    for (height, width) in shapes.map({ ($0[0], $0[1]) }) {
      let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
      eval(input)
      // One timed run per config per round, interleaved, so that drift (thermal,
      // other GPU clients) affects every config alike; the minimum is the estimate.
      for (config, model) in zip(configs, models) {
        apply(config)
        let key = "\(height)x\(width) \(config)"
        let output = Device.withDefaultDevice(.gpu) { model(input) }
        eval(output)
        checksums[key] = abs(output.asType(.float32)).sum().item(Float.self)
        eval(Device.withDefaultDevice(.gpu) { model(input) })
      }
      for _ in 0..<rounds {
        for (config, model) in zip(configs, models) {
          apply(config)
          let key = "\(height)x\(width) \(config)"
          let t = ms({ [model(input)] }, warm: 0, runs: 1)
          results[key, default: []].append(t)
        }
      }
      for config in configs {
        let key = "\(height)x\(width) \(config)"
        let ordered = results[key]!
        let samples = ordered.sorted()
        print("perf-spike AB \(key): min \(String(format: "%.1f", samples[0])) ms, median \(String(format: "%.1f", samples[samples.count / 2])) ms, in order \(ordered.map { String(format: "%.0f", $0) }), checksum \(checksums[key]!)")
      }
    }
  }

  // MARK: 12. mixed-precision MMA (half operands, float accumulator) vs converted float MMA: bit-exact?

  private static let mixedMMAKernel = MLXFast.metalKernel(
    name: "perf_spike_mixed_mma_check",
    inputNames: ["a", "b"],
    outputNames: ["converted", "mixed"],
    source: #"""
      // a: [tiles, 8, 8] half, b: [tiles, 8, 8] half; each simdgroup multiplies one pair
      // over `depth` k-steps (reusing the same tiles) both ways.
      const uint tile = thread_position_in_grid.x / 32;
      simdgroup_matrix<half, 8, 8> ah, bh;
      simdgroup_load(ah, a + tile * 64, 8, ulong2(0), false);
      simdgroup_load(bh, b + tile * 64, 8, ulong2(0), false);
      simdgroup_matrix<float, 8, 8> af, bf, accConverted, accMixed;
      af.thread_elements()[0] = float(ah.thread_elements()[0]);
      af.thread_elements()[1] = float(ah.thread_elements()[1]);
      bf.thread_elements()[0] = float(bh.thread_elements()[0]);
      bf.thread_elements()[1] = float(bh.thread_elements()[1]);
      accConverted.thread_elements()[0] = 0.0f; accConverted.thread_elements()[1] = 0.0f;
      accMixed.thread_elements()[0] = 0.0f; accMixed.thread_elements()[1] = 0.0f;
      for (uint k = 0; k < uint(depth); ++k) {
        simdgroup_multiply_accumulate(accConverted, af, bf, accConverted);
        simdgroup_multiply_accumulate(accMixed, ah, bh, accMixed);
      }
      simdgroup_store(accConverted, converted + tile * 64, 8, ulong2(0), false);
      simdgroup_store(accMixed, mixed + tile * 64, 8, ulong2(0), false);
      """#,
    header: "#include <metal_simdgroup_matrix>\n"
  )

  func testMixedPrecisionMMAMatchesConvertedFloat() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    let tiles = 4096
    for (scale, depth) in [(Float(1), 1), (Float(3), 4), (Float(0.05), 16), (Float(20), 8)] {
      let a = (MLXRandom.normal([tiles, 8, 8]) * scale).asType(.float16)
      let b = (MLXRandom.normal([tiles, 8, 8]) * scale).asType(.float16)
      eval(a, b)
      let outputs = Self.mixedMMAKernel([a, b], template: [("depth", depth)], grid: (tiles * 32, 1, 1), threadGroup: (128, 1, 1),
        outputShapes: [[tiles, 8, 8], [tiles, 8, 8]], outputDTypes: [.float32, .float32])
      eval(outputs)
      let delta = abs(outputs[0] - outputs[1])
      let maxDelta = delta.max().item(Float.self)
      let count = (delta .> 0).sum().item(Int32.self)
      let bitsEqual = (outputs[0].view(dtype: .uint32) .== outputs[1].view(dtype: .uint32)).all().item(Bool.self)
      print("perf-spike mixed MMA scale \(scale) depth \(depth): max |Δ| \(maxDelta), \(count) elements differ, bit-identical \(bitsEqual)")
    }
  }

  // MARK: 14. post block pieces on real weights (why does block 70 cost 33 ms?)

  func testPostBlockPieces() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    typealias Ops = NeuralRenderingTransformerOperations
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let post = try NeuralRenderingPostBlock(weights: weights, compileBlocks: true)
    let block = try NeuralRenderingWindowBlock(weights: weights, blockIndex: 70, channels: 32, hiddenChannels: 128, headCount: 1, compileBlock: true)
    let sine = try weights.required("block70.layer0.inp_merge_sin")
    let cosine = try weights.required("block70.layer0.inp_merge_cos")
    for (height, width) in [(768, 1024), (1088, 1920)] {
      let low = (MLXRandom.normal([1, height / 2, width / 2, 32]) * 0.5).asType(.float16)
      let skip = (MLXRandom.normal([1, height, width, 32]) * 0.5).asType(.float16)
      eval(low, skip)
      let merged = NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: skip, lowScale: sine, skipScale: cosine, publish: false)
      eval(merged)
      let features = block(merged); eval(features)
      let tMerge = ms({ [NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: skip, lowScale: sine, skipScale: cosine, publish: false)] })
      let tBlock = ms({ [block(merged)] })
      let tHead = ms({ [NeuralRenderingPostBlock.head(features, gain: post.outputGain, convolution: post.outputConvolution, headWeight: post.headWeight, fused: false)] })
      let tHeadFused = ms({ [NeuralRenderingPostBlock.head(features, gain: post.outputGain, convolution: post.outputConvolution, headWeight: post.headWeight, fused: true)] })
      let tAll = ms({ [post(low, skip: skip)] })
      Memory.clearCache()
      let tAllCold = ms({ [post(low, skip: skip)] }, warm: 0, runs: 1)
      print("perf-spike post block \(height)x\(width): merge \(String(format: "%.2f", tMerge)) ms, 1h block \(String(format: "%.2f", tBlock)) ms, head \(String(format: "%.2f", tHead)) ms (fused \(String(format: "%.2f", tHeadFused)) ms), whole \(String(format: "%.2f", tAll)) ms, whole after clearCache \(String(format: "%.2f", tAllCold)) ms")
    }
  }

  // MARK: 15. real global stage at the 1080p token count: eval/v2/materialized in one process

  func testGlobalStageAB() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let stage = try NeuralRenderingGlobalStage(weights: weights, quantizeFFN: false, preciseAttention: true, fusedOperations: true, compileGraph: true)
    let savedEval = NeuralRenderingGlobalStage.perBlockEvalEnabled
    let savedV2 = NeuralRenderingStreamedGlobalAttention.v2Enabled
    defer { NeuralRenderingGlobalStage.perBlockEvalEnabled = savedEval; NeuralRenderingStreamedGlobalAttention.v2Enabled = savedV2 }
    for (height, width) in [(12, 16), (20, 32), (34, 60)] {
      let input = (MLXRandom.normal([1, height, width, 1024]) * 0.5).asType(.float16)
      eval(input)
      var reference: MLXArray?
      for (label, evalOn, v2) in [("eval+v1/materialized", true, false), ("noeval+v1/materialized", false, false), ("eval+v2", true, true), ("noeval+v2", false, true), ("noeval+v1/materialized (again)", false, false)] {
        NeuralRenderingGlobalStage.perBlockEvalEnabled = evalOn
        NeuralRenderingStreamedGlobalAttention.v2Enabled = v2
        let output = Device.withDefaultDevice(.gpu) { stage(input) }
        eval(output)
        let delta = reference.map { abs($0.asType(.float32) - output.asType(.float32)).max().item(Float.self) } ?? 0
        if reference == nil { reference = output }
        let t = ms({ [stage(input)] }, warm: 2, runs: 7)
        print("perf-spike global stage \(height * width) tokens \(label): \(String(format: "%.2f", t)) ms, max |Δ| vs first \(delta)")
      }
    }
  }

  // MARK: 16. CPU graph construction vs GPU evaluation of a whole frame

  func testWholeFrameBuildVsEval() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let model = try NeuralRenderingTransformerModel(weights: weights, compileBlocks: true)
    let shapes = (ProcessInfo.processInfo.environment["MLXDLSS_PERF_SHAPES"] ?? "768x1024,1088x1920").split(separator: ",")
    func elapsed(_ start: ContinuousClock.Instant) -> Double {
      let d = start.duration(to: .now); return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
    }
    for shape in shapes {
      let parts = shape.split(separator: "x").compactMap { Int($0) }
      let (height, width) = (parts[0], parts[1])
      let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
      eval(input)
      for _ in 0..<2 { eval(Device.withDefaultDevice(.gpu) { model(input) }) }
      var builds: [Double] = [], evals: [Double] = [], asyncs: [Double] = []
      for _ in 0..<5 {
        let t0 = ContinuousClock.now
        let output = Device.withDefaultDevice(.gpu) { model(input) }
        builds.append(elapsed(t0))
        let t1 = ContinuousClock.now
        eval(output)
        evals.append(elapsed(t1))
        // asyncEval returns once the graph is scheduled: the remaining wait is GPU time not hidden by encoding.
        let output2 = Device.withDefaultDevice(.gpu) { model(input) }
        let t2 = ContinuousClock.now
        asyncEval(output2)
        let scheduled = elapsed(t2)
        eval(output2)
        asyncs.append(scheduled)
      }
      print("perf-spike build-vs-eval \(height)x\(width): graph build \(builds.sorted().map { String(format: "%.1f", $0) }) ms, eval \(evals.sorted().map { String(format: "%.1f", $0) }) ms, asyncEval scheduling \(asyncs.sorted().map { String(format: "%.1f", $0) }) ms")
    }
  }

  // MARK: 17. MLX per-op CPU overhead: custom kernel call cost vs plain op, build and eval

  func testPerOpOverhead() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    typealias Ops = NeuralRenderingTransformerOperations
    func elapsed(_ start: ContinuousClock.Instant) -> Double {
      let d = start.duration(to: .now); return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
    }
    let tiny = MLXArray.ones([1, 8, 8, 32], dtype: .float16); eval(tiny)
    let n = 400
    for round in 0..<3 {
      // custom kernel (templated: e4m3RoundTrip uses elementCount template), chained
      var x = tiny
      let t0 = ContinuousClock.now
      for _ in 0..<n { x = Ops.e4m3RoundTrip(x) }
      let buildCustom = elapsed(t0)
      let t1 = ContinuousClock.now
      eval(x)
      let evalCustom = elapsed(t1)
      // custom kernel without template args (params buffer): quadraticGatePublish
      var y = tiny
      let t2 = ContinuousClock.now
      for _ in 0..<n { y = Ops.quadraticGatePublish(y) }
      let buildParams = elapsed(t2)
      let t3 = ContinuousClock.now
      eval(y)
      let evalParams = elapsed(t3)
      // the big fused 1h window block kernel (15 KB source), chained on a small image
      let w1 = MLXArray.ones([32, 128], dtype: .float16) * 0.01, w2 = MLXArray.ones([128, 32], dtype: .float16) * 0.01
      let cos1 = MLXArray.ones([32], dtype: .float16), qkv = MLXArray.ones([32, 96], dtype: .float16) * 0.01
      let scale = MLXArray([Float(1)]).asType(.float16), bias = MLXArray.zeros([1, 64, 64], dtype: .float16)
      let proj = MLXArray.ones([32, 32], dtype: .float16) * 0.01, cos2 = MLXArray.ones([32], dtype: .float16)
      eval(w1, w2, cos1, qkv, scale, bias, proj, cos2)
      var z = tiny
      let t4 = ContinuousClock.now
      for _ in 0..<n { z = NeuralRenderingFusedWindowBlock.apply(z, expansionWeight: w1, feedForwardProjectionWeight: w2, feedForwardCosine: cos1, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, attentionProjectionWeight: proj, attentionCosine: cos2, windowOrigin: .zero, publish: true) }
      let buildBig = elapsed(t4)
      let t5 = ContinuousClock.now
      eval(z)
      let evalBig = elapsed(t5)
      // plain MLX elementwise op chain
      var p = tiny
      let t6 = ContinuousClock.now
      for _ in 0..<n { p = p * cos1 }
      let buildPlain = elapsed(t6)
      let t7 = ContinuousClock.now
      eval(p)
      let evalPlain = elapsed(t7)
      // plain matmul chain
      var m = tiny.reshaped([64, 32])
      let t8 = ContinuousClock.now
      for _ in 0..<n { m = matmul(m, proj) }
      let buildMM = elapsed(t8)
      let t9 = ContinuousClock.now
      eval(m)
      let evalMM = elapsed(t9)
      if round == 2 {
        print(String(format: "perf-spike per-op (µs/op, %d chained tiny ops): custom templated build %.1f eval %.1f | custom params build %.1f eval %.1f | fused 1h block (15 KB source) build %.1f eval %.1f | plain elementwise build %.1f eval %.1f | matmul build %.1f eval %.1f", n, buildCustom / Double(n) * 1000, evalCustom / Double(n) * 1000, buildParams / Double(n) * 1000, evalParams / Double(n) * 1000, buildBig / Double(n) * 1000, evalBig / Double(n) * 1000, buildPlain / Double(n) * 1000, evalPlain / Double(n) * 1000, buildMM / Double(n) * 1000, evalMM / Double(n) * 1000))
      }
    }
  }

  // MARK: 18. float32 probes for the compiled (tile-kernel) feed-forward path, one stage per process

  func testFloat32CompiledProbe() throws {
    guard let probe = ProcessInfo.processInfo.environment["MLXDLSS_PROBE"] else { throw XCTSkip("MLXDLSS_PROBE=ffn:ROWS | block:H:W (needs MLXDLSS_LOGICAL_WEIGHTS for block)") }
    typealias Ops = NeuralRenderingTransformerOperations
    let fields = probe.split(separator: ":").map(String.init)
    if fields[0] == "ffn" {
      let rows = Int(fields[1])!
      let dtype: DType = fields.count > 2 && fields[2] == "half" ? .float16 : .float32
      let input = MLXArray((0..<(rows * 32)).map { sin(Float($0) * 0.001) }, [1, rows / 8, 8, 32]).asType(dtype)
      let expansion = MLXArray((0..<(32 * 128)).map { cos(Float($0) * 0.003) * 0.05 }, [32, 128]).asType(dtype)
      let projection = MLXArray((0..<(128 * 32)).map { sin(Float($0) * 0.005) * 0.05 }, [128, 32]).asType(dtype)
      eval(input, expansion, projection)
      let expected = matmul(Ops.e4m3RoundTrip(Ops.quadraticGateActivation(matmul(input, expansion))), projection)
      eval(expected)
      let actual = Ops.fusedSimpleFeedForward(input, expansionWeight: expansion, projectionWeight: projection)
      eval(actual)
      let delta = abs(actual.asType(.float32) - expected.asType(.float32))
      let maxDelta = delta.max().item(Float.self)
      let count = (delta .> 0.001).sum().item(Int32.self)
      let firstBad = argMax(delta.flattened()).item(Int32.self)
      print("perf-spike probe ffn rows=\(rows) \(dtype): max |Δ| \(maxDelta), \(count) elements > 1e-3, first at flat index \(firstBad) (row \(Int(firstBad) / 32))")
    } else {
      guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("needs weights") }
      let height = Int(fields[1])!, width = Int(fields[2])!
      let mode = fields.count > 3 ? fields[3] : "compare"
      let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu))
      let input = MLXArray((0..<(height * width * 32)).map { sin(Float($0) * 0.001) }, [1, height, width, 32])
      eval(input)
      func report(_ label: String, _ a: MLXArray, _ b: MLXArray) {
        eval(a, b)
        let delta = abs(a - b)
        let maxDelta = delta.max().item(Float.self)
        let count = (delta .> 0.001).sum().item(Int32.self)
        let where_ = argMax(delta.flattened()).item(Int32.self)
        print("perf-spike probe \(label) \(height)x\(width) float32: max |Δ| \(maxDelta), \(count) elements > 1e-3, worst flat index \(where_) (pixel \(Int(where_) / 32), channel \(Int(where_) % 32)); |a| max \(abs(a).max().item(Float.self))")
      }
      if fields[0] == "seq" || fields[0] == "seqsep" {
        // Replicates testExternalRecoveredCompiledWindowSequenceMatchesEagerBlocks exactly.
        let eagerBlocks = try (1...3).map { try NeuralRenderingWindowBlock(weights: weights, blockIndex: $0, channels: 32, hiddenChannels: 128, headCount: 1) }
        let sequence = try NeuralRenderingWindowSequence(weights: weights, blockIndices: 1...3, channels: 32, hiddenChannels: 128, headCount: 1, compileSequence: true)
        let reference = eagerBlocks.reduce(input) { value, block in block(value) }
        let candidate = sequence(input)
        if fields[0] == "seqsep" { eval(reference); eval(candidate) } else { eval(reference, candidate) }
        report("sequence 1-3 compiled vs eager (\(fields[0]))", candidate, reference)
        return
      }
      if fields[0] == "ffnreal" {
        let w1 = try weights.required("block1.layer0.weight1"), w2 = try weights.required("block1.layer0.weight2")
        typealias Ops = NeuralRenderingTransformerOperations
        let literal = matmul(Ops.e4m3RoundTrip(Ops.quadraticGateActivation(matmul(input, w1))), w2)
        let fused = Ops.fusedSimpleFeedForward(input, expansionWeight: w1, projectionWeight: w2)
        report("ffn real weights fused vs literal", fused, literal)
        return
      }
      let eager = try NeuralRenderingWindowBlock(weights: weights, blockIndex: 1, channels: 32, hiddenChannels: 128, headCount: 1)
      let compiled = try NeuralRenderingWindowBlock(weights: weights, blockIndex: 1, channels: 32, hiddenChannels: 128, headCount: 1, compileBlock: true)
      switch mode {
      case "eager2": report("block1 eager vs eager", eager(input), eager(input))
      case "compiled2": report("block1 compiled vs compiled", compiled(input), compiled(input))
      case "ffnonly":
        typealias Ops = NeuralRenderingTransformerOperations
        let w1 = try weights.required("block1.layer0.weight1"), w2 = try weights.required("block1.layer0.weight2")
        let cos = try weights.required("block1.layer0.ffn_cos_skip")
        let a = Ops.cosineResidual(skip: input, branch: matmul(Ops.e4m3RoundTrip(Ops.quadraticGateActivation(matmul(input, w1))), w2), cosine: cos)
        let b = Ops.cosineResidual(skip: input, branch: Ops.fusedSimpleFeedForward(input, expansionWeight: w1, projectionWeight: w2), cosine: cos)
        report("block1 ffn+residual literal vs fused", b, a)
      default: report("block1 compiled vs eager", compiled(input), eager(input))
      }
    }
  }

  // MARK: 19. reciprocal broadcast via simd_shuffle and two-lane norms: exactness + timing

  func testShuffleVariantsAreBitExact() throws {
    guard enabled else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1") }
    let saved = NeuralRenderingFusedWindowBlock.shuffleReciprocals
    defer { NeuralRenderingFusedWindowBlock.shuffleReciprocals = saved }
    // single-head block
    for (height, width) in [(37, 53), (544, 960), (1088, 1920)] {
      let x = (MLXRandom.normal([1, height, width, 32]) * 0.5).asType(.float16)
      let w1 = (MLXRandom.normal([32, 128]) * 0.1).asType(.float16)
      let w2 = (MLXRandom.normal([128, 32]) * 0.1).asType(.float16)
      let cos1 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
      let qkv = (MLXRandom.normal([32, 96]) * 0.1).asType(.float16)
      let scale = MLXArray([Float(1.2)]).asType(.float16)
      let bias = (MLXRandom.normal([1, 64, 64]) * 2.0).asType(.float16)
      let proj = (MLXRandom.normal([32, 32]) * 0.1).asType(.float16)
      let cos2 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
      eval(x, w1, w2, cos1, qkv, scale, bias, proj, cos2)
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4)] {
        func run() -> MLXArray {
          NeuralRenderingFusedWindowBlock.apply(x, expansionWeight: w1, feedForwardProjectionWeight: w2, feedForwardCosine: cos1, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, attentionProjectionWeight: proj, attentionCosine: cos2, windowOrigin: origin, publish: true)
        }
        NeuralRenderingFusedWindowBlock.shuffleReciprocals = false
        let reference = run(); eval(reference)
        var line = "perf-spike shuffle 1h block \(height)x\(width) origin (\(origin.x),\(origin.y)):"
        line += String(format: " scratch %.3f ms", ms({ [run()] }))
        for bits in [1] {
          NeuralRenderingFusedWindowBlock.shuffleReciprocals = true
          let candidate = run(); eval(candidate)
          let delta = abs(reference.asType(.float32) - candidate.asType(.float32)).max().item(Float.self)
          line += String(format: " | shuffle %.3f ms Δ %g", ms({ [run()] }), delta); _ = bits
          XCTAssertEqual(delta, 0, "1h block bits \(bits) at \(height)x\(width)")
        }
        print(line)
      }
    }
    // multi-head core
    for (heads, height, width) in [(2, 272, 480), (4, 136, 240), (8, 68, 120), (16, 34, 60), (2, 19, 37)] {
      let channels = heads * 32
      let qkv = (MLXRandom.normal([1, height, width, channels * 3]) * 0.5).asType(.float16)
      let scale = MLXRandom.uniform(low: 0.5, high: 2.0, [heads]).asType(.float16)
      let bias = (MLXRandom.normal([heads, 64, 64]) * 2.0).asType(.float16)
      eval(qkv, scale, bias)
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: 0, x: -4)] {
        func run() -> MLXArray {
          NeuralRenderingFusedWindowAttention.apply(qkv: qkv, attentionScale: scale, attentionBias: bias, headCount: heads, windowOrigin: origin)
        }
        NeuralRenderingFusedWindowBlock.shuffleReciprocals = false
        let reference = run(); eval(reference)
        var line = "perf-spike shuffle core \(heads)h \(height)x\(width) origin (\(origin.x),\(origin.y)):"
        line += String(format: " scratch %.3f ms", ms({ [run()] }))
        for bits in [1] {
          NeuralRenderingFusedWindowBlock.shuffleReciprocals = true
          let candidate = run(); eval(candidate)
          let delta = abs(reference.asType(.float32) - candidate.asType(.float32)).max().item(Float.self)
          line += String(format: " | shuffle %.3f ms Δ %g", ms({ [run()] }), delta); _ = bits
          XCTAssertEqual(delta, 0, "core bits \(bits) \(heads)h at \(height)x\(width)")
        }
        print(line)
      }
    }
  }

  // MARK: 20. compiled vs eager global stage at the 1080p token count, one process

  func testGlobalStageCompileAB() throws {
    guard enabled, let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip("MLXDLSS_PERF_SPIKE=1 and MLXDLSS_LOGICAL_WEIGHTS") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)).cast(to: .float16)
    let compiled = try NeuralRenderingGlobalStage(weights: weights, quantizeFFN: false, preciseAttention: true, fusedOperations: true, compileGraph: true)
    let eager = try NeuralRenderingGlobalStage(weights: weights, quantizeFFN: false, preciseAttention: true, fusedOperations: true, compileGraph: false)
    for (height, width) in [(12, 16), (17, 30), (20, 32)] {
      let input = (MLXRandom.normal([1, height, width, 1024]) * 0.5).asType(.float16)
      eval(input)
      let a = Device.withDefaultDevice(.gpu) { compiled(input) }
      let b = Device.withDefaultDevice(.gpu) { eager(input) }
      eval(a, b)
      let delta = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
      var samples: [(Double, Double)] = []
      for _ in 0..<4 {
        samples.append((ms({ [compiled(input)] }, warm: 1, runs: 3), ms({ [eager(input)] }, warm: 1, runs: 3)))
      }
      let bestCompiled = samples.map(\.0).min()!, bestEager = samples.map(\.1).min()!
      print("perf-spike global stage compile \(height * width) tokens: compiled \(String(format: "%.2f", bestCompiled)) ms, eager \(String(format: "%.2f", bestEager)) ms, max |Δ| \(delta)")
    }
  }
}
