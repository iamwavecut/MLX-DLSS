import Foundation
import MLX

/// The multi-head window blocks' branched feed-forward and its published
/// cosine residual in one Metal kernel:
/// `E4M3(FFN(x) + x * cos)`, where `FFN` is the dense arrangement of
/// `NeuralRenderingTransformerOperations.denseBranchedFeedForward`.
///
/// Each simdgroup owns eight rows and keeps them in simdgroup-matrix registers.
/// For every output group it produces the gated hidden tiles one 8×8 tile at a
/// time and feeds each straight into the group's branch projection, so the
/// `[rows, 4C]` hidden activation never leaves the registers. The published
/// group projections wait in the thread's own output slots (each thread reads
/// back only what it wrote) until the output projection consumes them in
/// chunks of eight column tiles, which bounds the live float accumulators. The roundings are
/// those of the GEMM chain it replaces: every product accumulates in float in
/// the GEMMs' sequential k order (half operands into float accumulators,
/// bit-identical to MLX's GEMMs), the GEMM results round to half, the gate and
/// the E4M3 publications sit at the same points, and the residual rounds twice
/// in half like `NeuralRenderingFusedWindowAttention.cosineResidual`.
enum NeuralRenderingFusedBranchedFeedForward {
  static let supportedChannels: Set<Int> = [64, 128, 256]
  /// Widths that use the kernel inside the window blocks. At 256 channels the
  /// expansion weight (512 KB) no longer stays cached for eight-row tiles and
  /// the GEMM chain is faster (2026-10-17, M2 Max, 68x120: 0.81 ms vs 0.94 ms).
  /// Diagnostics override: `MLXDLSS_FUSED_FFN_CHANNELS=64,128,256`.
  nonisolated(unsafe) static var blockChannels: Set<Int> =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN_CHANNELS"].map {
      Set($0.split(separator: ",").compactMap { Int($0) }).intersection(supportedChannels)
    } ?? [64, 128]
  /// Simdgroups (eight rows each) per threadgroup.
  nonisolated(unsafe) static var simdgroupsPerThreadgroup: Int =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN_SIMDGROUPS"].flatMap(Int.init) ?? 4
  /// `MLXDLSS_FUSED_FFN=0` restores the GEMM chain (diagnostics).
  nonisolated(unsafe) static var isEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN"] != "0"

  /// The Metal 4 tensor-operation variant (`tensorOpsSource`): `matmul2d` runs on
  /// the GPU Neural Accelerators of M5, M6 and later and on shader cores before
  /// them. Off unless `MLXDLSS_TENSOR_OPS_FFN=1` (macOS 26 and later, where MLX
  /// compiles custom kernels as Metal 4). On an M2 Max it matches the simdgroup
  /// kernel bit for bit and is not faster; it has not been timed on Neural
  /// Accelerator hardware, whose accumulation order can differ.
  /// `scripts/bench-neural-accelerators.sh` compares both paths there. When Metal
  /// cannot compile the kernel on this GPU (`tensorOpsCompile`), the simdgroup
  /// kernel stays in use.
  nonisolated(unsafe) static var tensorOpsEnabled: Bool = {
    guard ProcessInfo.processInfo.environment["MLXDLSS_TENSOR_OPS_FFN"] == "1" else { return false }
    if tensorOpsCompile(rowsPerSimdgroup: tensorOpsRowsPerSimdgroup) { return true }
    FileHandle.standardError.write(
      Data("MLXDLSS_TENSOR_OPS_FFN=1: Metal cannot compile the tensor-operation kernel on this GPU; using the simdgroup kernel\n".utf8))
    return false
  }()

  /// True when Metal compiles and runs the tensor-operation kernel at 64 and 128
  /// channels with `rows` rows per simdgroup. MLX reports a failed compile as an
  /// error, which `checkedEval` throws instead of stopping the process. The
  /// virtual GPU of GitHub's macOS 26.7 runners fails to compile it.
  static func tensorOpsCompile(rowsPerSimdgroup rows: Int) -> Bool {
    guard #available(macOS 26.0, iOS 26.0, *) else { return false }
    let zeros = { (shape: [Int]) in MLXArray.zeros(shape, dtype: .float16) }
    do {
      for channels in [64, 128] {
        try checkedEval(
          applyTensorOps(
            zeros([1, 2, 2, channels]), denseExpansionWeight: zeros([channels, channels * 4]),
            groupedProjectionWeight: zeros([channels / 32, 128, 32]), outputProjectionWeight: zeros([channels, channels]),
            cosine: zeros([channels]), rowsPerSimdgroup: rows))
      }
      return true
    } catch {
      return false
    }
  }
  /// Rows per simdgroup of the tensor-operation variant (16 or 32).
  nonisolated(unsafe) static var tensorOpsRowsPerSimdgroup: Int =
    ProcessInfo.processInfo.environment["MLXDLSS_TENSOR_OPS_FFN_ROWS"].flatMap(Int.init).flatMap { [16, 32].contains($0) ? $0 : nil } ?? 16

  private static let kernel = makeKernel(name: "mlxdlss_fused_branched_feed_forward", source: source)
  private static let tensorOpsKernel = MLXFast.metalKernel(
    name: "mlxdlss_tensor_ops_branched_feed_forward",
    inputNames: ["input", "expansion", "projection", "outputProjection", "cosine", "params"],
    outputNames: ["output"],
    source: tensorOpsSource,
    header: "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
      + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  static func makeKernel(name: String, source: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
      name: name,
      inputNames: ["input", "expansion", "projection", "outputProjection", "cosine", "params"],
      outputNames: ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  static let source = #"""
      constexpr uint tiles = channels / 8;
      constexpr uint groups = channels / 32;
      constexpr uint hiddenStride = channels * 4;
      const uint simd = simdgroup_index_in_threadgroup;
      const ushort lane = thread_index_in_simdgroup;
      const ushort fragmentRow = ((lane / 4) & 4) + ((lane / 2) % 4);
      const ushort fragmentColumn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
      const uint rowCount = params[0];
      const uint row = (threadgroup_position_in_grid.x * simdgroups + simd) * 8 + fragmentRow;
      const bool inside = row < rowCount;
      const device half2* input2 = (const device half2*)input;
      device half2* output2 = (device half2*)output;
      const device half2* cosine2 = (const device half2*)cosine;

      simdgroup_matrix<half, 8, 8> xt[tiles];
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < tiles; ++k) {
        const half2 value = inside ? input2[(row * channels + k * 8 + fragmentColumn) / 2] : half2(0.0h);
        xt[k].thread_elements()[0] = value.x;
        xt[k].thread_elements()[1] = value.y;
      }
      for (uint g = 0; g < groups; ++g) {
        simdgroup_matrix<float, 8, 8> branch[4];
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          branch[o].thread_elements()[0] = 0.0f;
          branch[o].thread_elements()[1] = 0.0f;
        }
        for (uint b = 0; b < 4; ++b) {
          _Pragma("clang loop unroll(full)")
          for (uint n = 0; n < 4; ++n) {
            // Hidden tile (columns g, b, n*8..n*8+7) = x · W1, gate, E4M3.
            simdgroup_matrix<float, 8, 8> acc;
            acc.thread_elements()[0] = 0.0f;
            acc.thread_elements()[1] = 0.0f;
            const device half* column = expansion + (g * 4 + b) * 32 + n * 8;
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < tiles; ++k) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, column + k * 8 * hiddenStride, hiddenStride, ulong2(0), false);
              simdgroup_multiply_accumulate(acc, xt[k], right, acc);
            }
            simdgroup_matrix<half, 8, 8> hidden;
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              half value = half(acc.thread_elements()[e]);
              half clamped = clamp(value, half(-4.0), half(4.0));
              half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
              half gate = fma(clamped, linear, half(0.89453125));
              hidden.thread_elements()[e] = half(float(mlxdlss_e4m3(value * gate)));
            }
            // Next k-step (b*32 + n*8) of the group's branch projection.
            const device half* rows = projection + g * 128 * 32 + (b * 32 + n * 8) * 32;
            _Pragma("clang loop unroll(full)")
            for (uint o = 0; o < 4; ++o) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, rows + o * 8, 32, ulong2(0), false);
              simdgroup_multiply_accumulate(branch[o], hidden, right, branch[o]);
            }
          }
        }
        // Published group projection, parked in this thread's own output slots.
        _Pragma("clang loop unroll(full)")
        for (uint kk = 0; kk < 4; ++kk) {
          const half2 published = half2(
            half(float(mlxdlss_e4m3(half(branch[kk].thread_elements()[0])))),
            half(float(mlxdlss_e4m3(half(branch[kk].thread_elements()[1])))));
          if (inside) { output2[(row * channels + g * 32 + kk * 8 + fragmentColumn) / 2] = published; }
        }
      }

      // Output projection in chunks of eight column tiles; each residual result
      // replaces the input fragment it consumed until the final store.
      constexpr uint chunkTiles = 8;
      _Pragma("clang loop unroll(full)")
      for (uint chunk = 0; chunk < tiles / chunkTiles; ++chunk) {
        simdgroup_matrix<float, 8, 8> out[chunkTiles];
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          out[o].thread_elements()[0] = 0.0f;
          out[o].thread_elements()[1] = 0.0f;
        }
        for (uint kk = 0; kk < tiles; ++kk) {
          const half2 parked = inside ? output2[(row * channels + kk * 8 + fragmentColumn) / 2] : half2(0.0h);
          simdgroup_matrix<half, 8, 8> published;
          published.thread_elements()[0] = parked.x;
          published.thread_elements()[1] = parked.y;
          const device half* rows = outputProjection + kk * 8 * channels + chunk * chunkTiles * 8;
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < chunkTiles; ++o) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, rows + o * 8, channels, ulong2(0), false);
            simdgroup_multiply_accumulate(out[o], published, right, out[o]);
          }
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          const uint tile = chunk * chunkTiles + o;
          const half2 scale = cosine2[(tile * 8 + fragmentColumn) / 2];
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half branchValue = half(out[o].thread_elements()[e]);
            const half skip = xt[tile].thread_elements()[e] * scale[e];
            xt[tile].thread_elements()[e] = half(float(mlxdlss_e4m3(half(branchValue + skip))));
          }
        }
      }

      if (inside) {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < tiles; ++o) {
          output2[(row * channels + o * 8 + fragmentColumn) / 2] = half2(xt[o].thread_elements()[0], xt[o].thread_elements()[1]);
        }
      }
      """#

  /// The same feed-forward with Metal 4 tensor operations. Each simdgroup owns
  /// `rows` rows (16 or 32). Per output group it multiplies the rows by the
  /// group's four 32-wide expansion slices (`matmul2d`, float accumulation),
  /// gates and publishes each hidden slice into the simdgroup's threadgroup tile
  /// and accumulates the branch projection from that tile; the published group
  /// projections collect in a second threadgroup tile that feeds the output
  /// projection, 32 columns at a time, and the residual. Element coordinates come
  /// from `get_multidimensional_index`, so the code makes no assumption about how
  /// a GPU distributes cooperative-tensor elements over the lanes. The roundings
  /// are the simdgroup kernel's: GEMM results round to half, the gate and the
  /// E4M3 publications sit at the same points, the residual rounds twice in half.
  static let tensorOpsSource = #"""
      using namespace mpp::tensor_ops;
      constexpr int C = channels;
      constexpr int groups = channels / 32;
      constexpr int hiddenStride = channels * 4;
      constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate;
      constexpr auto expandDescriptor = matmul2d_descriptor(rows, 32, C, false, false, false, mode);
      constexpr auto branchDescriptor = matmul2d_descriptor(rows, 32, 32, false, false, false, mode);
      constexpr auto outputDescriptor = matmul2d_descriptor(rows, 32, C, false, false, false, mode);
      threadgroup half scratch[simdgroups * rows * (32 + C)];
      const uint simd = simdgroup_index_in_threadgroup;
      const uint rowCount = params[0];
      const uint row0 = (threadgroup_position_in_grid.x * simdgroups + simd) * rows;
      if (row0 >= rowCount) { return; }
      const int valid = int(min(uint(rows), rowCount - row0));
      threadgroup half* hiddenTile = scratch + simd * rows * (32 + C);
      threadgroup half* publishedTile = hiddenTile + rows * 32;

      using DeviceMatrix = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
      using GroupMatrix = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
      auto x = DeviceMatrix((device half*)(input + row0 * C), dextents<int32_t, 2>(C, valid), array<int32_t, 2>({1, C}));
      auto hidden = GroupMatrix(hiddenTile, dextents<int32_t, 2>(32, rows), array<int32_t, 2>({1, 32}));
      auto published = GroupMatrix(publishedTile, dextents<int32_t, 2>(C, rows), array<int32_t, 2>({1, C}));

      matmul2d<expandDescriptor, execution_simdgroup> expand;
      matmul2d<branchDescriptor, execution_simdgroup> branchProjection;
      matmul2d<outputDescriptor, execution_simdgroup> outputProjectionOp;

      for (int g = 0; g < groups; ++g) {
        auto weight0 = DeviceMatrix((device half*)expansion, dextents<int32_t, 2>(32, C), array<int32_t, 2>({1, hiddenStride}));
        auto rightP0 = DeviceMatrix((device half*)projection, dextents<int32_t, 2>(32, 32), array<int32_t, 2>({1, 32}));
        auto branch = branchProjection.template get_destination_cooperative_tensor<decltype(hidden), decltype(rightP0), float>();
        _Pragma("clang loop unroll(full)")
        for (uint16_t i = 0; i < branch.get_capacity(); ++i) { branch[i] = 0.0f; }
        for (int b = 0; b < 4; ++b) {
          auto weight = DeviceMatrix((device half*)(expansion + (g * 4 + b) * 32), dextents<int32_t, 2>(32, C), array<int32_t, 2>({1, hiddenStride}));
          auto acc = expand.template get_destination_cooperative_tensor<decltype(x), decltype(weight0), float>();
          _Pragma("clang loop unroll(full)")
          for (uint16_t i = 0; i < acc.get_capacity(); ++i) { acc[i] = 0.0f; }
          expand.run(x, weight, acc);
          _Pragma("clang loop unroll(full)")
          for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            const auto index = acc.get_multidimensional_index(i);
            const half value = half(acc[i]);
            const half clamped = clamp(value, half(-4.0), half(4.0));
            const half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
            const half gate = fma(clamped, linear, half(0.89453125));
            hiddenTile[index[1] * 32 + index[0]] = half(float(mlxdlss_e4m3(value * gate)));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          auto right = DeviceMatrix((device half*)(projection + (g * 128 + b * 32) * 32), dextents<int32_t, 2>(32, 32), array<int32_t, 2>({1, 32}));
          branchProjection.run(hidden, right, branch);
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        _Pragma("clang loop unroll(full)")
        for (uint16_t i = 0; i < branch.get_capacity(); ++i) {
          const auto index = branch.get_multidimensional_index(i);
          publishedTile[index[1] * C + g * 32 + index[0]] = half(float(mlxdlss_e4m3(half(branch[i]))));
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);

      for (int chunk = 0; chunk < C / 32; ++chunk) {
        auto weight = DeviceMatrix((device half*)(outputProjection + chunk * 32), dextents<int32_t, 2>(32, C), array<int32_t, 2>({1, C}));
        auto out = outputProjectionOp.template get_destination_cooperative_tensor<decltype(published), decltype(weight), float>();
        _Pragma("clang loop unroll(full)")
        for (uint16_t i = 0; i < out.get_capacity(); ++i) { out[i] = 0.0f; }
        outputProjectionOp.run(published, weight, out);
        _Pragma("clang loop unroll(full)")
        for (uint16_t i = 0; i < out.get_capacity(); ++i) {
          const auto index = out.get_multidimensional_index(i);
          const int r = index[1];
          if (r < valid) {
            const uint column = uint(chunk * 32 + index[0]);
            const uint slot = (row0 + uint(r)) * uint(C) + column;
            const half branchValue = half(out[i]);
            const half skip = input[slot] * cosine[column];
            output[slot] = half(float(mlxdlss_e4m3(half(branchValue + skip))));
          }
        }
      }
      """#

  /// `E4M3(denseBranchedFeedForward(input) + input * cosine)` for a half input
  /// with 64, 128 or 256 channels.
  static func apply(
    _ input: MLXArray,
    denseExpansionWeight: MLXArray,
    groupedProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray,
    cosine: MLXArray,
    kernel: MLXFast.MLXFastKernel? = nil,
    simdgroups: Int? = nil
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition(supportedChannels.contains(channels) && input.dtype == .float16)
    precondition(denseExpansionWeight.shape == [channels, channels * 4])
    precondition(groupedProjectionWeight.shape == [channels / 32, 128, 32])
    precondition(outputProjectionWeight.shape == [channels, channels] && cosine.shape == [channels])
    let rowCount = input.size / channels
    if kernel == nil, tensorOpsEnabled, channels <= 128 {
      return applyTensorOps(
        input, denseExpansionWeight: denseExpansionWeight, groupedProjectionWeight: groupedProjectionWeight,
        outputProjectionWeight: outputProjectionWeight, cosine: cosine)
    }
    let simdgroups = simdgroups ?? simdgroupsPerThreadgroup
    let threadgroups = (rowCount + 8 * simdgroups - 1) / (8 * simdgroups)
    let half = { (array: MLXArray) in array.asType(.float16) }
    return (kernel ?? Self.kernel)(
      [
        input, half(denseExpansionWeight), half(groupedProjectionWeight), half(outputProjectionWeight),
        half(cosine), NeuralRenderingKernelParameters.array([UInt32(rowCount)]),
      ],
      template: [("channels", channels), ("simdgroups", simdgroups)],
      grid: (threadgroups * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }

  /// The tensor-operation variant of `apply` (64 or 128 channels).
  static func applyTensorOps(
    _ input: MLXArray,
    denseExpansionWeight: MLXArray,
    groupedProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray,
    cosine: MLXArray,
    rowsPerSimdgroup: Int? = nil,
    simdgroups: Int? = nil
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition([64, 128].contains(channels) && input.dtype == .float16)
    let rows = rowsPerSimdgroup ?? tensorOpsRowsPerSimdgroup
    // Each simdgroup holds rows x (32 + channels) halfs of threadgroup memory;
    // up to four simdgroups within the 32 KB a threadgroup may use.
    let simdgroups = simdgroups ?? min(4, 32_768 / (rows * (32 + channels) * 2))
    let rowCount = input.size / channels
    let threadgroups = (rowCount + rows * simdgroups - 1) / (rows * simdgroups)
    let half = { (array: MLXArray) in array.asType(.float16) }
    return tensorOpsKernel(
      [
        input, half(denseExpansionWeight), half(groupedProjectionWeight), half(outputProjectionWeight),
        half(cosine), NeuralRenderingKernelParameters.array([UInt32(rowCount)]),
      ],
      template: [("channels", channels), ("simdgroups", simdgroups), ("rows", rows)],
      grid: (threadgroups * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
