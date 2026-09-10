import Foundation
import MLX

/// One Metal kernel per 8×8 window for the single-head 32-channel window
/// blocks (blocks 0–4, 67–70 and the recovered head): FFN, cosine residual,
/// QKV, vendor cosine normalisation, attention with bias and the recovered
/// bit-affine softmax, projection, residual and E4M3 publication.
///
/// Four simdgroups share one window; each owns two window rows (16 tokens).
/// Threadgroup memory holds 14 KB: an 8 KB shared region that first stages the
/// feed-forward weight chunks and then the normalised keys and published
/// values, plus 1.5 KB of scratch per simdgroup. The feed-forward output, the
/// normalised queries and the attended values stay in simdgroup-matrix
/// registers. The arithmetic is copied from the per-operation kernels so the
/// rounding points stay identical: the FFN accumulates in half like the fused
/// feed-forward kernels (the hidden activation is produced in two 64-wide
/// chunks feeding one accumulator, which keeps the accumulation order), the
/// attention matmuls accumulate in float like the MLX GEMMs they replace, the
/// per-token norms and the bit-affine softmax totals are summed in the
/// recovered sequential order by one lane per token, and only element-wise
/// steps are spread over all lanes.
enum NeuralRenderingFusedWindowBlock {
  static let channels = 32
  static let hiddenChannels = 128
  static let windowSize = 8
  static let simdgroupsPerWindow = 4

  private static let kernel = MLXFast.metalKernel(
    name: "mlxdlss_fused_window_block_1h32",
    inputNames: [
      "input", "expansion", "projection", "feedForwardCosine", "qkvWeight",
      "attentionScale", "attentionBias", "attentionProjection", "attentionCosine", "params",
    ],
    outputNames: ["output"],
    source: #"""
      // Threadgroup memory (halfs): shared region 0..4096 (weight chunks, then
      // K 0..2048 and V 2048..4096), then 768 halfs of scratch per simdgroup.
      threadgroup half4 arena4[(4096 + 4 * 768) / 4];
      threadgroup half* arena = (threadgroup half*)arena4;
      threadgroup half* W = arena;
      threadgroup half4* W4 = (threadgroup half4*)W;
      threadgroup half* K = arena;
      threadgroup half* V = arena + 2048;
      threadgroup half4* K4 = (threadgroup half4*)K;
      threadgroup half4* V4 = (threadgroup half4*)V;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const uint tid = thread_position_in_threadgroup.x;
      threadgroup half* T = arena + 4096 + simd * 768;
      threadgroup half4* T4 = (threadgroup half4*)T;
      threadgroup half* R = T + 512;   // 16 per-token reciprocals

      const uint height = params[0];
      const uint width = params[1];
      const uint padTop = params[2];
      const uint padLeft = params[3];
      const bool publish = params[4] != 0u;
      const uint windowsX = (width + padLeft + 7) / 8;
      const uint windowIndex = threadgroup_position_in_grid.x;
      const uint windowY = windowIndex / windowsX;
      const uint windowX = windowIndex % windowsX;
      const int rowY0 = int(windowY * 8) - int(padTop);
      const int colX0 = int(windowX * 8) - int(padLeft);
      const device half4* input4 = (const device half4*)input;
      device half4* output4 = (device half4*)output;
      const device half4* expansion4 = (const device half4*)expansion;
      const device half4* projection4 = (const device half4*)projection;
      const device half4* feedForwardCosine4 = (const device half4*)feedForwardCosine;
      const device half4* attentionCosine4 = (const device half4*)attentionCosine;
      const device half4* attentionBias4 = (const device half4*)attentionBias;
      const half scale = attentionScale[0];

      simdgroup_matrix<half, 8, 8> xt[2][4];
      simdgroup_matrix<half, 8, 8> ft[2][4];
      simdgroup_matrix<half, 8, 8> qt[2][4];
      simdgroup_matrix<half, 8, 8> at[2][4];

      // Phase 0: window rows into registers (zeros outside the image, like the padded path).
      for (uint rr = 0; rr < 2; ++rr) {
        const int y = rowY0 + int(simd * 2 + rr);
        for (uint i = lane; i < 64; i += 32) {
          const int x = colX0 + int(i / 8);
          const bool in = y >= 0 && y < int(height) && x >= 0 && x < int(width);
          T4[i] = in ? input4[(uint(y) * width + uint(x)) * 8 + i % 8] : half4(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 4; ++k) {
          simdgroup_load(xt[rr][k], T + k * 8, 32, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Phase 1: feed-forward. Expansion (half accumulation, gate, E4M3) and
      // projection over two 64-wide hidden chunks into one accumulator per tile;
      // each chunk's weights are staged in the shared region.
      simdgroup_matrix<half, 8, 8> pacc[2][4];
      for (uint rr = 0; rr < 2; ++rr) {
        for (uint o = 0; o < 4; ++o) {
          pacc[rr][o].thread_elements()[0] = 0.0h;
          pacc[rr][o].thread_elements()[1] = 0.0h;
        }
      }
      for (uint chunk = 0; chunk < 2; ++chunk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 512; i += 128) {
          W4[i] = expansion4[(i / 16) * 32 + chunk * 16 + i % 16];
          W4[512 + i] = projection4[chunk * 512 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half* H = T;
        for (uint rr = 0; rr < 2; ++rr) {
          for (uint c = 0; c < 8; ++c) {
            simdgroup_matrix<half, 8, 8> acc;
            acc.thread_elements()[0] = 0.0h;
            acc.thread_elements()[1] = 0.0h;
            for (uint k = 0; k < 4; ++k) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, W + k * 8 * 64 + c * 8, 64, ulong2(0), false);
              simdgroup_multiply_accumulate(acc, xt[rr][k], right, acc);
            }
            for (uint e = 0; e < 2; ++e) {
              half value = acc.thread_elements()[e];
              half clamped = clamp(value, half(-4.0), half(4.0));
              half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
              half gate = fma(clamped, linear, half(0.89453125));
              acc.thread_elements()[e] = half(float(mlxdlss_e4m3(value * gate)));
            }
            simdgroup_store(acc, H + c * 8, 64, ulong2(0), false);
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint o = 0; o < 4; ++o) {
            for (uint k = 0; k < 8; ++k) {
              simdgroup_matrix<half, 8, 8> left;
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(left, H + k * 8, 64, ulong2(0), false);
              simdgroup_load(right, W + 2048 + k * 8 * 32 + o * 8, 32, ulong2(0), false);
              simdgroup_multiply_accumulate(pacc[rr][o], left, right, pacc[rr][o]);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
      // residual: F = branch + X * cos (two half roundings, like the MLX path)
      for (uint rr = 0; rr < 2; ++rr) {
        for (uint o = 0; o < 4; ++o) {
          simdgroup_store(pacc[rr][o], T + o * 8, 32, ulong2(0), false);
          simdgroup_store(xt[rr][o], T + 256 + o * 8, 32, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64; i += 32) {
          half4 skip = T4[64 + i] * feedForwardCosine4[i % 8];
          T4[i] = T4[i] + skip;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 4; ++k) {
          simdgroup_load(ft[rr][k], T + k * 8, 32, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Phase 2: QKV projection (float accumulation, half result), one 32-column
      // part at a time for both window rows; vendor cosine normalisation of q
      // (scaled) and k with one lane per token for the norm, E4M3 of v.
      threadgroup_barrier(mem_flags::mem_threadgroup);   // weight staging is over: K/V region is free
      for (uint part = 0; part < 3; ++part) {
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<float, 8, 8> acc[2];
          for (uint rr = 0; rr < 2; ++rr) {
            acc[rr].thread_elements()[0] = 0.0f;
            acc[rr].thread_elements()[1] = 0.0f;
          }
          for (uint k = 0; k < 4; ++k) {
            simdgroup_matrix<half, 8, 8> rightHalf;
            simdgroup_load(rightHalf, qkvWeight + k * 8 * 96 + (part * 4 + o) * 8, 96, ulong2(0), false);
            simdgroup_matrix<float, 8, 8> right;
            right.thread_elements()[0] = float(rightHalf.thread_elements()[0]);
            right.thread_elements()[1] = float(rightHalf.thread_elements()[1]);
            for (uint rr = 0; rr < 2; ++rr) {
              simdgroup_matrix<float, 8, 8> left;
              left.thread_elements()[0] = float(ft[rr][k].thread_elements()[0]);
              left.thread_elements()[1] = float(ft[rr][k].thread_elements()[1]);
              simdgroup_multiply_accumulate(acc[rr], left, right, acc[rr]);
            }
          }
          for (uint rr = 0; rr < 2; ++rr) {
            simdgroup_matrix<half, 8, 8> result;
            result.thread_elements()[0] = half(acc[rr].thread_elements()[0]);
            result.thread_elements()[1] = half(acc[rr].thread_elements()[1]);
            simdgroup_store(result, T + rr * 256 + o * 8, 32, ulong2(0), false);
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (part < 2) {
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
            R[lane] = half(metal::fast::rsqrt(float(norm)));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          for (uint i = lane; i < 128; i += 32) {
            const uint tokenInSimd = i / 8;
            half4 normalized = T4[i] * R[tokenInSimd];
            if (part == 0) { normalized *= scale; }
            half4 value = half4(
              half(float(mlxdlss_e4m3(normalized.x))), half(float(mlxdlss_e4m3(normalized.y))),
              half(float(mlxdlss_e4m3(normalized.z))), half(float(mlxdlss_e4m3(normalized.w))));
            if (part == 0) {
              T4[i] = value;
            } else {
              K4[(simd * 16 + tokenInSimd) * 8 + i % 8] = value;
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (part == 0) {
            for (uint rr = 0; rr < 2; ++rr) {
              for (uint k = 0; k < 4; ++k) {
                simdgroup_load(qt[rr][k], T + rr * 256 + k * 8, 32, ulong2(0), false);
              }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        } else {
          for (uint i = lane; i < 128; i += 32) {
            half4 value = T4[i];
            V4[(simd * 16 + i / 8) * 8 + i % 8] = half4(
              half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
              half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Phase 3: per window row, scores = q · kᵀ (float accumulation), bias,
      // the bit-affine softmax (sequential total per row, element-wise second
      // pass) with E4M3 probabilities, attended = E4M3(P · v).
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
          T4[i] = T4[i] + attentionBias4[(token0 + i / 16) * 16 + i % 16];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
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
          R[lane] = half(1.0f / float(total));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 128; i += 32) {
          const half reciprocal = R[i / 16];
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
          at[rr][o].thread_elements()[0] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[0]))));
          at[rr][o].thread_elements()[1] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[1]))));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Phase 4: output projection (float accumulation) for both window rows,
      // residual with the feed-forward output, publication, store.
      simdgroup_matrix<half, 8, 8> ot[2][4];
      for (uint o = 0; o < 4; ++o) {
        simdgroup_matrix<float, 8, 8> acc[2];
        for (uint rr = 0; rr < 2; ++rr) {
          acc[rr].thread_elements()[0] = 0.0f;
          acc[rr].thread_elements()[1] = 0.0f;
        }
        for (uint k = 0; k < 4; ++k) {
          simdgroup_matrix<half, 8, 8> rightHalf;
          simdgroup_load(rightHalf, attentionProjection + k * 8 * 32 + o * 8, 32, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> right;
          right.thread_elements()[0] = float(rightHalf.thread_elements()[0]);
          right.thread_elements()[1] = float(rightHalf.thread_elements()[1]);
          for (uint rr = 0; rr < 2; ++rr) {
            simdgroup_matrix<float, 8, 8> left;
            left.thread_elements()[0] = float(at[rr][k].thread_elements()[0]);
            left.thread_elements()[1] = float(at[rr][k].thread_elements()[1]);
            simdgroup_multiply_accumulate(acc[rr], left, right, acc[rr]);
          }
        }
        for (uint rr = 0; rr < 2; ++rr) {
          ot[rr][o].thread_elements()[0] = half(acc[rr].thread_elements()[0]);
          ot[rr][o].thread_elements()[1] = half(acc[rr].thread_elements()[1]);
        }
      }
      for (uint rr = 0; rr < 2; ++rr) {
        for (uint o = 0; o < 4; ++o) {
          simdgroup_store(ot[rr][o], T + o * 8, 32, ulong2(0), false);
          simdgroup_store(ft[rr][o], T + 256 + o * 8, 32, ulong2(0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int y = rowY0 + int(simd * 2 + rr);
        for (uint i = lane; i < 64; i += 32) {
          const int x = colX0 + int(i / 8);
          if (y < 0 || y >= int(height) || x < 0 || x >= int(width)) { continue; }
          half4 skip = T4[64 + i] * attentionCosine4[i % 8];
          half4 value = T4[i] + skip;
          if (publish) {
            value = half4(
              half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
              half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
          }
          output4[(uint(y) * width + uint(x)) * 8 + i % 8] = value;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      """#,
    header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// Whole window block for a `[1, H, W, 32]` half-precision input.
  static func apply(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    feedForwardProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    windowOrigin: NeuralRenderingWindowOrigin,
    publish: Bool
  ) -> MLXArray {
    precondition(input.ndim == 4 && input.shape[0] == 1 && input.shape[3] == channels)
    precondition(input.dtype == .float16)
    let height = input.shape[1]
    let width = input.shape[2]
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    let windowsY = (height + padTop + windowSize - 1) / windowSize
    let windowsX = (width + padLeft + windowSize - 1) / windowSize
    let windowCount = windowsY * windowsX
    let half = { (array: MLXArray) in array.asType(.float16) }
    // Geometry travels in a buffer rather than template values so the kernel is
    // compiled once per process instead of once per shape and window phase.
    let params = NeuralRenderingKernelParameters.array([UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), publish ? UInt32(1) : UInt32(0)])
    return kernel(
      [
        input, half(expansionWeight), half(feedForwardProjectionWeight), half(feedForwardCosine),
        half(qkvWeight), half(attentionScale), half(attentionBias.reshaped([64 * 64])),
        half(attentionProjectionWeight), half(attentionCosine), params,
      ],
      grid: (windowCount * 32 * simdgroupsPerWindow, 1, 1),
      threadGroup: (32 * simdgroupsPerWindow, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
