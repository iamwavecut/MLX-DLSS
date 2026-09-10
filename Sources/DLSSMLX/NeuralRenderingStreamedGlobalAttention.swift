import Foundation
import MLX

/// Keeps an eight-query tile on-chip for short sequences. Larger sequences use
/// two passes over K, retaining the sequential half denominator and E4M3
/// probabilities without allocating quadratic score/probability arrays.
enum NeuralRenderingStreamedGlobalAttention {
  private static let mode = ProcessInfo.processInfo.environment["MLXDLSS_STREAMED_GLOBAL_ATTENTION"]
  /// The v2 kernels convert scores to vendor weights and probabilities with
  /// every thread and keep only the half denominator sum sequential, so the
  /// resident tile serves up to 1024 tokens; `MLXDLSS_GLOBAL_ATTENTION_V2=0`
  /// restores the earlier kernels and their 512-token limit (diagnostics).
  nonisolated(unsafe) static var v2Enabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_GLOBAL_ATTENTION_V2"] != "0"
  static var residentMaxTokens: Int { v2Enabled ? 1024 : 512 }

  /// Longer sequences keep the materialized attention unless streaming is forced.
  static func isEnabled(tokens: Int) -> Bool {
    mode == "1" || (mode != "0" && tokens <= residentMaxTokens)
  }

  static func apply(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    precondition(query.ndim == 4 && query.shape.last == 32)
    precondition(query.shape == key.shape && query.shape == value.shape)
    precondition(query.dtype == .float16 && key.dtype == .float16 && value.dtype == .float16)
    let tokens = query.dim(2)
    precondition(tokens > 0 && tokens.isMultiple(of: 2))
    if v2Enabled { return applyV2(query: query, key: key, value: value) }
    if tokens <= 512 {
      let paddedTokens = (tokens + 7) / 8 * 8
      func pad(_ x: MLXArray) -> MLXArray {
        paddedTokens == tokens ? x : padded(x, widths: [[0, 0], [0, 0], [0, paddedTokens - tokens], [0, 0]])
      }
      let output = resident([pad(query), pad(key), pad(value)],
        template: [("tokens", tokens), ("pitch", paddedTokens)],
        grid: (128, paddedTokens / 8, query.dim(0) * query.dim(1)), threadGroup: (128, 1, 1),
        outputShapes: [[query.dim(0), query.dim(1), paddedTokens, 32]], outputDTypes: [.float16])[0]
      return output[0..., 0..., 0..<tokens, 0...]
    }
    let grid = (128, (tokens + 7) / 8, query.dim(0) * query.dim(1))
    let reciprocal = denominator([query, key], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [Array(query.shape.dropLast())], outputDTypes: [.float16])[0]
    return attention([query, key, value, reciprocal], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [query.shape], outputDTypes: [.float16])[0]
  }

  private static let header = NeuralRenderingTransformerOperations.e4m3MetalHeaderText + "\n" + #"""
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

    METAL_FUNC void score_tile(
      threadgroup half* Q, threadgroup half* K, threadgroup half* S,
      const device half* key, uint headBase, uint token0, uint tokens, uint tid, uint simd
    ) {
      for (uint i = tid; i < 32 * 32; i += 128) {
        uint token = token0 + i / 32;
        K[i] = token < tokens ? key[headBase + token * 32 + i % 32] : 0.0h;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      for (uint c = 0; c < 32; c += 8) {
        simdgroup_matrix<half, 8, 8> q, k;
        simdgroup_load(q, Q + c, 32, ulong2(0), false);
        simdgroup_load(k, K + simd * 8 * 32 + c, 32, ulong2(0), true);
        simdgroup_matrix<float, 8, 8> left, right;
        left.thread_elements()[0] = float(q.thread_elements()[0]);
        left.thread_elements()[1] = float(q.thread_elements()[1]);
        right.thread_elements()[0] = float(k.thread_elements()[0]);
        right.thread_elements()[1] = float(k.thread_elements()[1]);
        simdgroup_multiply_accumulate(acc, left, right, acc);
      }
      simdgroup_matrix<half, 8, 8> scores;
      scores.thread_elements()[0] = clamp(half(acc.thread_elements()[0]), -3.0h, 3.0h);
      scores.thread_elements()[1] = clamp(half(acc.thread_elements()[1]), -3.0h, 3.0h);
      simdgroup_store(scores, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """#

  private static let setup = #"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint tokens = uint(query_shape[2]);
    const uint head = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * 8;
    const uint headBase = head * tokens * 32;
    threadgroup half Q[8 * 32];
    threadgroup half K[32 * 32];
    threadgroup half S[8 * 32];
    for (uint i = tid; i < 8 * 32; i += 128) {
      uint row = row0 + i / 32;
      Q[i] = row < tokens ? query[headBase + row * 32 + i % 32] : 0.0h;
    }
    """#

  private static let resident = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_resident", inputNames: ["query", "key", "value"], outputNames: ["output"],
    source: #"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint headBase = threadgroup_position_in_grid.z * pitch * 32;
      const uint row0 = threadgroup_position_in_grid.y * 8;
      threadgroup half S[8 * pitch];
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
      if (tid < 8) {
        half total = 0.0h;
        for (uint j = 0; j < tokens; j += 2) {
          half2 weight = vendor_weights(half2(S[tid * pitch + j], S[tid * pitch + j + 1]));
          total += weight.x; total += weight.y;
          S[tid * pitch + j] = weight.x; S[tid * pitch + j + 1] = weight.y;
        }
        half reciprocal = half(1.0f / float(total));
        for (uint j = 0; j < pitch; ++j) {
          S[tid * pitch + j] = j < tokens ? half(float(mlxdlss_e4m3(S[tid * pitch + j] * reciprocal))) : 0.0h;
        }
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
      """#, header: header)

  private static let denominator = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_denominator", inputNames: ["query", "key"], outputNames: ["reciprocal"],
    source: setup + #"""
      half total = 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        if (tid < 8) {
          for (uint j = 0; j < min(32u, tokens - token0); j += 2) {
            half2 weight = vendor_weights(half2(S[tid * 32 + j], S[tid * 32 + j + 1]));
            total += weight.x;
            total += weight.y;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid < 8 && row0 + tid < tokens) {
        reciprocal[head * tokens + row0 + tid] = half(1.0f / float(total));
      }
      """#, header: header)

  private static let attention = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_values", inputNames: ["query", "key", "value", "reciprocal"], outputNames: ["output"],
    source: setup + #"""
      simdgroup_matrix<float, 8, 8> attended;
      attended.thread_elements()[0] = 0.0f;
      attended.thread_elements()[1] = 0.0f;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        if (tid < 8) {
          half r = row0 + tid < tokens ? reciprocal[head * tokens + row0 + tid] : 0.0h;
          for (uint j = 0; j < 32; j += 2) {
            half2 weight = vendor_weights(half2(S[tid * 32 + j], S[tid * 32 + j + 1]));
            S[tid * 32 + j] = token0 + j < tokens ? half(float(mlxdlss_e4m3(weight.x * r))) : 0.0h;
            S[tid * 32 + j + 1] = token0 + j + 1 < tokens ? half(float(mlxdlss_e4m3(weight.y * r))) : 0.0h;
          }
        }
        // K's storage is now free for the matching value rows.
        for (uint i = tid; i < 32 * 32; i += 128) {
          uint token = token0 + i / 32;
          K[i] = token < tokens ? value[headBase + token * 32 + i % 32] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> p, v;
          simdgroup_load(p, S + c, 32, ulong2(0), false);
          simdgroup_load(v, K + c * 32 + simd * 8, 32, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(p.thread_elements()[0]);
          left.thread_elements()[1] = float(p.thread_elements()[1]);
          right.thread_elements()[0] = float(v.thread_elements()[0]);
          right.thread_elements()[1] = float(v.thread_elements()[1]);
          simdgroup_multiply_accumulate(attended, left, right, attended);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[1]))));
      simdgroup_store(result, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < 8 * 32; i += 128) {
        uint row = row0 + i / 32;
        if (row < tokens) output[headBase + row * 32 + i % 32] = S[i];
      }
      """#, header: header)

  // MARK: - v2

  static func applyV2(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    let tokens = query.dim(2)
    if tokens <= residentMaxTokens {
      let paddedTokens = (tokens + 7) / 8 * 8
      func pad(_ x: MLXArray) -> MLXArray {
        paddedTokens == tokens ? x : padded(x, widths: [[0, 0], [0, 0], [0, paddedTokens - tokens], [0, 0]])
      }
      let output = residentV2([pad(query), pad(key), pad(value)],
        template: [("tokens", tokens), ("pitch", paddedTokens)],
        grid: (128, paddedTokens / 8, query.dim(0) * query.dim(1)), threadGroup: (128, 1, 1),
        outputShapes: [[query.dim(0), query.dim(1), paddedTokens, 32]], outputDTypes: [.float16])[0]
      return output[0..., 0..., 0..<tokens, 0...]
    }
    let grid = (128, (tokens + 7) / 8, query.dim(0) * query.dim(1))
    let reciprocal = denominatorV2([query, key], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [Array(query.shape.dropLast())], outputDTypes: [.float16])[0]
    return attentionV2([query, key, value, reciprocal], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [query.shape], outputDTypes: [.float16])[0]
  }

  private static let residentV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_resident_v2", inputNames: ["query", "key", "value"], outputNames: ["output"],
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
      """#, header: header)

  /// Score tile followed by the parallel weight conversion of its valid columns.
  private static let weightTileV2 = #"""
    METAL_FUNC void weight_tile_v2(threadgroup half* S, uint token0, uint tokens, uint tid) {
      const uint valid = min(32u, tokens - token0);
      for (uint i = tid; i < 8 * 16; i += 128) {
        const uint row = i / 16;
        const uint j = (i % 16) * 2;
        if (j < valid) {
          half2 weight = vendor_weights(half2(S[row * 32 + j], S[row * 32 + j + 1]));
          S[row * 32 + j] = weight.x;
          S[row * 32 + j + 1] = weight.y;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """#

  private static let denominatorV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_denominator_v2", inputNames: ["query", "key"], outputNames: ["reciprocal"],
    source: setup + #"""
      half total = 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        weight_tile_v2(S, token0, tokens, tid);
        if (tid < 8) {
          const uint valid = min(32u, tokens - token0);
          for (uint j = 0; j < valid; ++j) { total += S[tid * 32 + j]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid < 8 && row0 + tid < tokens) {
        reciprocal[head * tokens + row0 + tid] = half(1.0f / float(total));
      }
      """#, header: header + weightTileV2)

  private static let attentionV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_values_v2", inputNames: ["query", "key", "value", "reciprocal"], outputNames: ["output"],
    source: setup + #"""
      simdgroup_matrix<float, 8, 8> attended;
      attended.thread_elements()[0] = 0.0f;
      attended.thread_elements()[1] = 0.0f;
      const half r = row0 + (tid % 8) < tokens ? reciprocal[head * tokens + row0 + (tid % 8)] : 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        weight_tile_v2(S, token0, tokens, tid);
        for (uint i = tid; i < 8 * 32; i += 128) {
          const uint row = i / 32;
          const uint j = i % 32;
          const half rr = row0 + row < tokens ? reciprocal[head * tokens + row0 + row] : 0.0h;
          S[i] = token0 + j < tokens ? half(float(mlxdlss_e4m3(S[i] * rr))) : 0.0h;
        }
        // K's storage is now free for the matching value rows.
        for (uint i = tid; i < 32 * 32; i += 128) {
          uint token = token0 + i / 32;
          K[i] = token < tokens ? value[headBase + token * 32 + i % 32] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> p, v;
          simdgroup_load(p, S + c, 32, ulong2(0), false);
          simdgroup_load(v, K + c * 32 + simd * 8, 32, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(p.thread_elements()[0]);
          left.thread_elements()[1] = float(p.thread_elements()[1]);
          right.thread_elements()[0] = float(v.thread_elements()[0]);
          right.thread_elements()[1] = float(v.thread_elements()[1]);
          simdgroup_multiply_accumulate(attended, left, right, attended);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      (void)r;
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[1]))));
      simdgroup_store(result, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < 8 * 32; i += 128) {
        uint row = row0 + i / 32;
        if (row < tokens) output[headBase + row * 32 + i % 32] = S[i];
      }
      """#, header: header + weightTileV2)
}
