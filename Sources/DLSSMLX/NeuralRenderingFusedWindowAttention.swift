import Foundation
import MLX

/// Window attention core for the multi-head window families: one Metal
/// threadgroup per (8×8 window, head) that takes the projected q/k/v rows of
/// the window, applies the vendor cosine normalisation (query scaled per head)
/// and E4M3 publication, the scores with the recovered attention bias, the
/// bit-affine softmax with E4M3 probabilities and the attended values, and
/// writes E4M3-published `[1, H, W, channels]` output rows in image layout.
///
/// It replaces the padded window partition, the per-head reshapes, the two
/// publish kernels, the scores/probabilities matmuls, the softmax kernel, the
/// value publication, the window reverse and the final slice of the
/// per-operation path with one dispatch. Every rounding point is copied from
/// those kernels: float accumulation in the matmuls, sequential per-token norm
/// and per-row softmax totals, half arithmetic everywhere else. Tokens outside
/// the image (shifted origins, ragged edges) behave like the zero padding of
/// the per-operation path: their q/k/v are zero and their outputs are dropped.
/// Small integer parameter buffers for the custom kernels, kept alive for
/// the process. A parameter array created per call is released as soon as
/// its consumer is encoded, and MLX's concurrent command encoder only
/// orders dispatches by read-after-write and write-after-write on buffers:
/// a recycled parameter buffer could be overwritten while the kernel that
/// reads it is still running. Caching by value keeps every buffer alive.
enum NeuralRenderingKernelParameters {
  nonisolated(unsafe) private static var cache: [[UInt32]: MLXArray] = [:]
  static func array(_ values: [UInt32]) -> MLXArray {
    if let cached = cache[values] { return cached }
    let created = MLXArray(values)
    eval(created)
    cache[values] = created
    return created
  }
}

enum NeuralRenderingFusedWindowAttention {
  static let windowSize = 8
  static let simdgroupsPerWindow = 4
  private static let kernel = MLXFast.metalKernel(
    name: "mlxdlss_window_attention_core",
    inputNames: ["qkv", "attentionScale", "attentionBias", "params"],
    outputNames: ["output"],
    source: #"""
      // Threadgroup memory (halfs): K 0..2048, V 2048..4096, 640 of scratch per simdgroup
      // (512 for the query rows / scores, then 16 reciprocals).
      threadgroup half4 arena4[(4096 + 4 * (shuffleReciprocals ? 512 : 640)) / 4];
      threadgroup half* arena = (threadgroup half*)arena4;
      threadgroup half* K = arena;
      threadgroup half* V = arena + 2048;
      threadgroup half4* K4 = (threadgroup half4*)K;
      threadgroup half4* V4 = (threadgroup half4*)V;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const uint tid = thread_position_in_threadgroup.x;
      threadgroup half* T = arena + 4096 + simd * (shuffleReciprocals ? 512 : 640);
      threadgroup half4* T4 = (threadgroup half4*)T;
      threadgroup half* R = T + 512;   // reused: 16 reciprocals (query norm), 8 (softmax)

      const uint height = params[0];
      const uint width = params[1];
      const uint padTop = params[2];
      const uint padLeft = params[3];
      const uint headCount = params[4];
      const uint channels = headCount * 32;
      const uint rowStride = channels * 3;
      const uint windowsX = (width + padLeft + 7) / 8;
      const uint head = threadgroup_position_in_grid.x % headCount;
      const uint windowIndex = threadgroup_position_in_grid.x / headCount;
      const uint windowY = windowIndex / windowsX;
      const uint windowX = windowIndex % windowsX;
      const int rowY0 = int(windowY * 8) - int(padTop);
      const int colX0 = int(windowX * 8) - int(padLeft);
      const device half4* qkv4 = (const device half4*)qkv;
      device half4* output4 = (device half4*)output;
      const device half4* attentionBias4 = (const device half4*)attentionBias;
      const half scale = attentionScale[head];

      // Phase 0: this simdgroup's 16 tokens. Keys: stage, cosine-normalise (one
      // lane per token, sequential partial sums), publish into the shared K rows.
      // Values: stage, publish into the shared V rows.
      for (uint part = 1; part < 3; ++part) {
        for (uint i = lane; i < 128; i += 32) {
          const uint token = simd * 16 + i / 8;
          const int y = rowY0 + int(token / 8);
          const int x = colX0 + int(token % 8);
          const bool in = y >= 0 && y < int(height) && x >= 0 && x < int(width);
          T4[i] = in ? qkv4[((uint(y) * width + uint(x)) * rowStride + part * channels + head * 32) / 4 + i % 8] : half4(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 1) {
          half reciprocalNorm = 0.0h;
          if (lane < 16) {
            threadgroup half* vec = T + lane * 32;
            half value[32];
            for (uint c = 0; c < 32; ++c) { value[c] = vec[c]; }
            half partial[4][2];
            for (uint l = 0; l < 4; ++l) {
              for (uint parity = 0; parity < 2; ++parity) {
                uint c = l * 2 + parity;
                half first = fma(value[c + 8], value[c + 8], value[c] * value[c]);
                half second = fma(value[c + 24], value[c + 24], value[c + 16] * value[c + 16]);
                partial[l][parity] = first + second;
              }
            }
            half xorTwo[4][2];
            for (uint l = 0; l < 4; ++l) {
              for (uint parity = 0; parity < 2; ++parity) {
                xorTwo[l][parity] = partial[l][parity] + partial[l ^ 2][parity];
              }
            }
            half xorOne[4][2];
            for (uint l = 0; l < 4; ++l) {
              for (uint parity = 0; parity < 2; ++parity) {
                xorOne[l][parity] = xorTwo[l][parity] + xorTwo[l ^ 1][parity];
              }
            }
            half norm = max(xorOne[0][0] + xorOne[0][1], half(0.00006198883056640625));
            reciprocalNorm = half(metal::fast::rsqrt(float(norm)));
          }
          if (!shuffleReciprocals) {
            if (lane < 16) { R[lane] = reciprocalNorm; }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint i = lane; i < 128; i += 32) {
            const half tokenReciprocal = shuffleReciprocals
              ? simd_shuffle(reciprocalNorm, i / 8) : R[i / 8];
            half4 normalized = T4[i] * tokenReciprocal;
            K4[(simd * 16 + i / 8) * 8 + i % 8] = half4(
              half(float(mlxdlss_e4m3(normalized.x))), half(float(mlxdlss_e4m3(normalized.y))),
              half(float(mlxdlss_e4m3(normalized.z))), half(float(mlxdlss_e4m3(normalized.w))));
          }
        } else {
          for (uint i = lane; i < 128; i += 32) {
            half4 value = T4[i];
            V4[(simd * 16 + i / 8) * 8 + i % 8] = half4(
              half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
              half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Queries of this simdgroup's 16 tokens: stage, normalise with the head
      // scale (one lane per token for the norm), publish, load as tiles.
      for (uint i = lane; i < 128; i += 32) {
        const uint token = simd * 16 + i / 8;
        const int y = rowY0 + int(token / 8);
        const int x = colX0 + int(token % 8);
        const bool in = y >= 0 && y < int(height) && x >= 0 && x < int(width);
        T4[i] = in ? qkv4[((uint(y) * width + uint(x)) * rowStride + head * 32) / 4 + i % 8] : half4(0.0h);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      half reciprocalNorm = 0.0h;
      if (lane < 16) {
        threadgroup half* vec = T + lane * 32;
        half value[32];
        for (uint c = 0; c < 32; ++c) { value[c] = vec[c]; }
        half partial[4][2];
        for (uint l = 0; l < 4; ++l) {
          for (uint parity = 0; parity < 2; ++parity) {
            uint c = l * 2 + parity;
            half first = fma(value[c + 8], value[c + 8], value[c] * value[c]);
            half second = fma(value[c + 24], value[c + 24], value[c + 16] * value[c + 16]);
            partial[l][parity] = first + second;
          }
        }
        half xorTwo[4][2];
        for (uint l = 0; l < 4; ++l) {
          for (uint parity = 0; parity < 2; ++parity) {
            xorTwo[l][parity] = partial[l][parity] + partial[l ^ 2][parity];
          }
        }
        half xorOne[4][2];
        for (uint l = 0; l < 4; ++l) {
          for (uint parity = 0; parity < 2; ++parity) {
            xorOne[l][parity] = xorTwo[l][parity] + xorTwo[l ^ 1][parity];
          }
        }
        half norm = max(xorOne[0][0] + xorOne[0][1], half(0.00006198883056640625));
        reciprocalNorm = half(metal::fast::rsqrt(float(norm)));
      }
      if (!shuffleReciprocals) {
        if (lane < 16) { R[lane] = reciprocalNorm; }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = lane; i < 128; i += 32) {
        const half tokenReciprocal = shuffleReciprocals
          ? simd_shuffle(reciprocalNorm, i / 8) : R[i / 8];
        half4 normalized = T4[i] * tokenReciprocal;
        normalized *= scale;
        T4[i] = half4(
          half(float(mlxdlss_e4m3(normalized.x))), half(float(mlxdlss_e4m3(normalized.y))),
          half(float(mlxdlss_e4m3(normalized.z))), half(float(mlxdlss_e4m3(normalized.w))));
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<half, 8, 8> qt[2][4];
      for (uint rr = 0; rr < 2; ++rr) {
        for (uint k = 0; k < 4; ++k) {
          simdgroup_load(qt[rr][k], T + rr * 256 + k * 8, 32, ulong2(0), false);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Per window row: scores (float accumulation), bias, bit-affine softmax
      // with E4M3 probabilities, attended = E4M3(P · v), store.
      for (uint rr = 0; rr < 2; ++rr) {
        const uint token0 = (simd * 2 + rr) * 8;
        threadgroup half* S = T;
        for (uint j = 0; j < 8; ++j) {
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f;
          acc.thread_elements()[1] = 0.0f;
          for (uint k = 0; k < 4; ++k) {
            simdgroup_matrix<half, 8, 8> rightHalf;
            simdgroup_load(rightHalf, K + j * 8 * 32 + k * 8, 32, ulong2(0), true);
            simdgroup_matrix<float, 8, 8> left;
            simdgroup_matrix<float, 8, 8> right;
            left.thread_elements()[0] = float(qt[rr][k].thread_elements()[0]);
            left.thread_elements()[1] = float(qt[rr][k].thread_elements()[1]);
            right.thread_elements()[0] = float(rightHalf.thread_elements()[0]);
            right.thread_elements()[1] = float(rightHalf.thread_elements()[1]);
            simdgroup_multiply_accumulate(acc, left, right, acc);
          }
          simdgroup_matrix<half, 8, 8> result;
          result.thread_elements()[0] = half(acc.thread_elements()[0]);
          result.thread_elements()[1] = half(acc.thread_elements()[1]);
          simdgroup_store(result, S + j * 8, 64, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 128; i += 32) {
          T4[i] = T4[i] + attentionBias4[(head * 64 + token0 + i / 16) * 16 + i % 16];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        half reciprocalTotal = 0.0h;
        if (lane < 8) {
          threadgroup half* row = S + lane * 64;
          half total = 0.0h;
          for (uint j = 0; j < 64; j += 2) {
            half2 score = half2(row[j], row[j + 1]);
            half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
            affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
            ushort2 bits = as_type<ushort2>(affine);
            uint packed = uint(bits.x) | (uint(bits.y) << 16);
            uint transformed = (packed << 5) + 0x7FF88000;
            half2 weight = as_type<half2>(ushort2(ushort(transformed), ushort(transformed >> 16)));
            total += weight.x;
            total += weight.y;
          }
          reciprocalTotal = half(1.0f / float(total));
          if (!shuffleReciprocals) { R[lane] = reciprocalTotal; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 128; i += 32) {
          const half reciprocal = shuffleReciprocals ? simd_shuffle(reciprocalTotal, i / 16) : R[i / 16];
          half4 scores = T4[i];
          half4 probabilities;
          for (uint p = 0; p < 2; ++p) {
            half2 score = p == 0 ? scores.xy : scores.zw;
            half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
            affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
            ushort2 bits = as_type<ushort2>(affine);
            uint packed = uint(bits.x) | (uint(bits.y) << 16);
            uint transformed = (packed << 5) + 0x7FF88000;
            half2 weight = as_type<half2>(ushort2(ushort(transformed), ushort(transformed >> 16)));
            half2 probability = half2(
              half(float(mlxdlss_e4m3(weight.x * reciprocal))), half(float(mlxdlss_e4m3(weight.y * reciprocal))));
            if (p == 0) { probabilities.xy = probability; } else { probabilities.zw = probability; }
          }
          T4[i] = probabilities;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<half, 8, 8> at[4];
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f;
          acc.thread_elements()[1] = 0.0f;
          for (uint k = 0; k < 8; ++k) {
            simdgroup_matrix<half, 8, 8> leftHalf;
            simdgroup_matrix<half, 8, 8> rightHalf;
            simdgroup_load(leftHalf, S + k * 8, 64, ulong2(0), false);
            simdgroup_load(rightHalf, V + k * 8 * 32 + o * 8, 32, ulong2(0), false);
            simdgroup_matrix<float, 8, 8> left;
            simdgroup_matrix<float, 8, 8> right;
            left.thread_elements()[0] = float(leftHalf.thread_elements()[0]);
            left.thread_elements()[1] = float(leftHalf.thread_elements()[1]);
            right.thread_elements()[0] = float(rightHalf.thread_elements()[0]);
            right.thread_elements()[1] = float(rightHalf.thread_elements()[1]);
            simdgroup_multiply_accumulate(acc, left, right, acc);
          }
          at[o].thread_elements()[0] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[0]))));
          at[o].thread_elements()[1] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[1]))));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = 0; o < 4; ++o) {
          simdgroup_store(at[o], T + o * 8, 32, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int y = rowY0 + int(simd * 2 + rr);
        for (uint i = lane; i < 64; i += 32) {
          const int x = colX0 + int(i / 8);
          if (y < 0 || y >= int(height) || x < 0 || x >= int(width)) { continue; }
          output4[((uint(y) * width + uint(x)) * channels + head * 32) / 4 + i % 8] = T4[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      """#,
    header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// E4M3-published attended values `[1, H, W, channels]` for projected rows
  /// `qkv` of shape `[1, H, W, 3 * channels]` (half precision).
  static func apply(
    qkv: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    headCount: Int,
    windowOrigin: NeuralRenderingWindowOrigin
  ) -> MLXArray {
    precondition(qkv.ndim == 4 && qkv.shape[0] == 1 && qkv.dtype == .float16)
    let channels = qkv.shape[3] / 3
    precondition(channels == headCount * 32 && qkv.shape[3] == channels * 3)
    precondition(attentionScale.shape == [headCount])
    precondition(attentionBias.shape == [headCount, 64, 64])
    let height = qkv.shape[1]
    let width = qkv.shape[2]
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    let windowsY = (height + padTop + windowSize - 1) / windowSize
    let windowsX = (width + padLeft + windowSize - 1) / windowSize
    let threadgroups = windowsY * windowsX * headCount
    let params = NeuralRenderingKernelParameters.array([UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), UInt32(headCount)])
    return kernel(
      [
        qkv, attentionScale.asType(.float16), attentionBias.asType(.float16).reshaped([headCount * 64 * 64]),
        params,
      ],
      // MLX 0.31 uses constant pointers for inputs smaller than eight elements.
      // Separate the two scale signatures so its cache never evicts a pipeline
      // still referenced by an encoded dispatch with another head count.
      template: [
        ("constantAttentionScale", headCount < 8),
        ("shuffleReciprocals", NeuralRenderingFusedWindowBlock.shuffleReciprocals),
      ],
      grid: (threadgroups * 32 * simdgroupsPerWindow, 1, 1),
      threadGroup: (32 * simdgroupsPerWindow, 1, 1),
      outputShapes: [[1, height, width, channels]],
      outputDTypes: [.float16]
    )[0]
  }

  private static let residualKernel = MLXFast.metalKernel(
    name: "mlxdlss_cosine_residual_publish",
    inputNames: ["branch", "skip", "cosine", "params"],
    outputNames: ["output"],
    source: #"""
      const uint index = thread_position_in_grid.x;
      const uint count = params[0];
      if (index >= count) { return; }
      const uint channels = params[1];
      const bool publish = params[2] != 0u;
      const device half4* branch4 = (const device half4*)branch;
      const device half4* skip4 = (const device half4*)skip;
      const device half4* cosine4 = (const device half4*)cosine;
      device half4* output4 = (device half4*)output;
      half4 scaled = skip4[index] * cosine4[index % (channels / 4)];
      half4 value = branch4[index] + scaled;
      if (publish) {
        value = half4(
          half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
          half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
      }
      output4[index] = value;
      """#,
    header: NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// `branch + skip * cosine` in half precision, optionally E4M3-published,
  /// in one pass (same roundings as `cosineResidual` followed by `e4m3RoundTrip`).
  static func cosineResidual(skip: MLXArray, branch: MLXArray, cosine: MLXArray, publish: Bool) -> MLXArray {
    precondition(skip.shape == branch.shape && skip.dtype == .float16 && branch.dtype == .float16)
    let channels = skip.shape[skip.ndim - 1]
    precondition(cosine.shape == [channels] && channels.isMultiple(of: 4))
    let count = skip.size / 4
    let params = NeuralRenderingKernelParameters.array([UInt32(count), UInt32(channels), publish ? UInt32(1) : UInt32(0)])
    return residualKernel(
      [branch, skip, cosine.asType(.float16), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [skip.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
