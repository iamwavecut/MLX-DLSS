import Foundation
import MLX
import MLXNN

/// DLSS frame generation for plain video on MLX/Metal — the port of the
/// library's video path (two colour frames in, one interpolated frame out).
///
/// The graph, recovered from the library's kernels and verified against its
/// memory snapshots (see the Python reference `mlxdlss.framegen`):
///
///   a, b   = 2x2 box means of the frames, zero-padded to a multiple of 16
///   err    = [1/8, 3/4, 1/8]-blurred mean_rgb |a - b|
///   x      = [a, err, b, err, 0, t]                          (10 channels)
///   block0 : 3 stem convs (leaky-clamp, 2x2 mean pool), 8 residual convs,
///            3 activated heads on the x2-upsampled trunk, one linear head -> flow(4) mask(1) res(3)
///   block1 : the same on [warp(a, 2 f0A), warp(b, 2 f0B), f0, m0, 0, r0, t]  (18 channels)
///   export = [2 f0 + f1, m0 + m1, r0 + r1]                   (half resolution)
///   out    = m * warp(A, 2 flowA) + (1 - m) * warp(B, 2 flowB), m = sigmoid(mask) upsampled
///
/// Every stage takes a batch: `[N, H, W, C]` frames with one phase per sample,
/// so the phases of a multi-frame factor and consecutive pairs of a video run
/// in one pass (the per-frame cost is the latency of ~30 small dispatches;
/// batching amortises it).
///
/// Weights: the dense `[Cout, Cin, 3, 3]` fp16 tensors written by
/// `mlxdlss-weights extract-fg` from the user's own `libnvidia-ngx-dlssg.so`.
public final class FrameGenerator {
  public enum Precision: String, Sendable { case float16, float32 }

  public struct Error: Swift.Error, CustomStringConvertible {
    public let description: String
  }

  private struct Block {
    var stem: [(MLXArray, MLXArray)]
    var residual: [(MLXArray, MLXArray)]
    var heads: [(MLXArray, MLXArray)]
    var outWeight: MLXArray
    var outBias: MLXArray
    // fused-kernel form (float16): one launch per layer, the three heads as one
    // convolution and the linear head as one block-diagonal convolution
    var fusedStem: [FrameGenerationFusedConv.Layer] = []
    var fusedResidual: [FrameGenerationFusedConv.Layer] = []
    var fusedHeads: FrameGenerationFusedConv.Layer? = nil
    var fusedOut: FrameGenerationFusedConv.Layer? = nil
  }

  public let precision: Precision
  private let dtype: DType
  private let block0: Block
  private let block1: Block
  /// `MLXDLSS_FG_COMPILE=0` runs the eager graph (diagnostics).
  nonisolated(unsafe) static var compileEnabled: Bool = ProcessInfo.processInfo.environment["MLXDLSS_FG_COMPILE"] != "0"
  /// `MLXDLSS_FG_PROFILE=1` prints per-stage timings (eager path only) to stderr.
  nonisolated(unsafe) static var profileEnabled: Bool = ProcessInfo.processInfo.environment["MLXDLSS_FG_PROFILE"] == "1"
  /// `MLXDLSS_FG_FUSED_INPUTS=0` assembles the block inputs with MLX operations (diagnostics).
  nonisolated(unsafe) static var fusedInputsEnabled: Bool = ProcessInfo.processInfo.environment["MLXDLSS_FG_FUSED_INPUTS"] != "0"
  private lazy var compiledSynthesize: ([MLXArray]) -> [MLXArray] = compile(inputs: [], shapeless: false) { [self] inputs in
    [self.synthesizeGraph(inputs[0], inputs[1], phases: inputs[2])]
  }

  /// Tensor names the dense weight file must contain.
  public static let tensorNames: [String] = {
    var names: [String] = []
    func pair(_ base: String) { names += ["\(base).weight", "\(base).bias"] }
    for i in 0..<3 { pair("block0.stem\(i)") }
    for i in 0..<8 { pair("block0.res\(i)") }
    for i in 0..<3 { pair("block0.bot0.head\(i)") }
    pair("block0.bot1")
    for i in 0..<2 { pair("block1.stem\(i)") }
    for i in 0..<8 { pair("block1.res\(i)") }
    for i in 0..<3 { pair("block1.bot0.head\(i)") }
    pair("block1.bot1")
    return names
  }()

  public convenience init(weightsURL: URL, precision: Precision = .float16) throws {
    let arrays = try loadArrays(url: weightsURL, stream: .cpu)
    try self.init(weights: arrays, precision: precision)
  }

  /// `weights` holds dense `[Cout, Cin, kh, kw]` tensors (any float dtype).
  public init(weights: [String: MLXArray], precision: Precision = .float16) throws {
    let missing = Self.tensorNames.filter { weights[$0] == nil }
    if !missing.isEmpty {
      throw Error(description: "frame generation weights are missing \(missing.count) tensors, e.g. \(missing.prefix(3).joined(separator: ", "))")
    }
    self.precision = precision
    let dtype: DType = precision == .float16 ? .float16 : .float32
    self.dtype = dtype
    func conv(_ name: String) throws -> (MLXArray, MLXArray) {
      let w = weights["\(name).weight"]!
      let b = weights["\(name).bias"]!
      guard w.ndim == 4, w.dim(2) == 3, w.dim(3) == 3, b.ndim == 1, b.dim(0) == w.dim(0) else {
        throw Error(description: "\(name): expected a [Cout, Cin, 3, 3] weight and a [Cout] bias, got \(w.shape) and \(b.shape)")
      }
      // MLX convolutions take [Cout, kh, kw, Cin] weights over NHWC inputs.
      return (w.transposed(0, 2, 3, 1).asType(dtype), b.asType(dtype))
    }
    func dense(_ name: String) -> (MLXArray, MLXArray) { (weights["\(name).weight"]!, weights["\(name).bias"]!) }
    func block(_ prefix: String, stems: Int) throws -> Block {
      var block = Block(
        stem: try (0..<stems).map { try conv("\(prefix).stem\($0)") },
        residual: try (0..<8).map { try conv("\(prefix).res\($0)") },
        heads: try (0..<3).map { try conv("\(prefix).bot0.head\($0)") },
        outWeight: try conv("\(prefix).bot1").0,
        outBias: try conv("\(prefix).bot1").1
      )
      if dtype == .float16 {
        block.fusedStem = (0..<stems).map { let (w, b) = dense("\(prefix).stem\($0)"); return FrameGenerationFusedConv.Layer(weight: w, bias: b) }
        block.fusedResidual = (0..<8).map { let (w, b) = dense("\(prefix).res\($0)"); return FrameGenerationFusedConv.Layer(weight: w, bias: b) }
        let heads = (0..<3).map { dense("\(prefix).bot0.head\($0)") }
        block.fusedHeads = FrameGenerationFusedConv.Layer(
          weight: concatenated(heads.map { $0.0.asType(.float32) }, axis: 0), bias: concatenated(heads.map { $0.1.asType(.float32) }, axis: 0))
        // linear head: rows 0..<4 read head 0, row 4 head 1, rows 5..<8 head 2 -> block-diagonal over 3*Ch inputs
        let (ow, ob) = dense("\(prefix).bot1")
        let ch = ow.dim(1)
        let spans = [(0, 4), (4, 5), (5, 8)]
        let rows = spans.enumerated().map { (i, span) -> MLXArray in
          let part = ow[span.0..<span.1].asType(.float32)   // [n, ch, 3, 3]
          return padded(part, widths: [[0, 0], [i * ch, (2 - i) * ch], [0, 0], [0, 0]])
        }
        block.fusedOut = FrameGenerationFusedConv.Layer(weight: concatenated(rows, axis: 0), bias: ob)
      }
      return block
    }
    block0 = try block("block0", stems: 3)
    block1 = try block("block1", stems: 2)
    for block in [block0, block1] {
      for (w, b) in block.stem + block.residual + block.heads { eval(w, b) }
      eval(block.outWeight, block.outBias)
    }
  }

  // MARK: - Building blocks

  private static func act(_ x: MLXArray) -> MLXArray {
    clip(leakyRelu(x, negativeSlope: 0.01), min: -6, max: 6)
  }

  private static func conv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    conv2d(x, w, stride: 1, padding: 1) + b
  }

  private static func padChannels(_ x: MLXArray, to channels: Int) -> MLXArray {
    let c = x.dim(3)
    if c == channels { return x }
    let zeros = MLXArray.zeros([x.dim(0), x.dim(1), x.dim(2), channels - c], dtype: x.dtype)
    return concatenated([x, zeros], axis: 3)
  }

  /// 2x2 mean pool of an NHWC array with even height and width.
  public static func meanPool2(_ x: MLXArray) -> MLXArray {
    let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return x.reshaped([n, h / 2, 2, w / 2, 2, c]).mean(axes: [2, 4])
  }

  /// 2x2 box mean of full-resolution frames, zero-padded to a multiple of 16 (the library's tile).
  static func box2(_ x: MLXArray) -> MLXArray {
    let (h, w) = (x.dim(1), x.dim(2))
    let pooled = meanPool2(x[0..., 0..<(h / 2 * 2), 0..<(w / 2 * 2), 0...])
    let hp = (pooled.dim(1) + 15) / 16 * 16
    let wp = (pooled.dim(2) + 15) / 16 * 16
    if hp == pooled.dim(1), wp == pooled.dim(2) { return pooled }
    return padded(pooled, widths: [[0, 0], [0, hp - pooled.dim(1)], [0, wp - pooled.dim(2)], [0, 0]])
  }

  /// mean_rgb |a - b| blurred with the separable [1/8, 3/4, 1/8] kernel, borders replicated.
  static func photometricError(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let e = abs(a - b).mean(axis: 3, keepDims: true)
    let h = e.dim(1), w = e.dim(2)
    let ep = padded(e, widths: [[0, 0], [1, 1], [1, 1], [0, 0]], mode: .edge)
    func tap(_ dy: Int, _ dx: Int) -> MLXArray { ep[0..., (1 + dy)..<(1 + dy + h), (1 + dx)..<(1 + dx + w), 0...] }
    let k: [Float] = [0.125, 0.75, 0.125]
    var acc = MLXArray.zeros(e.shape, dtype: e.dtype)
    for (i, dy) in [-1, 0, 1].enumerated() {
      for (j, dx) in [-1, 0, 1].enumerated() {
        acc = acc + tap(dy, dx) * (k[i] * k[j])
      }
    }
    return acc
  }

  private func run(_ block: Block, _ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let clock = ContinuousClock()
    var mark = clock.now
    func lap(_ name: String, _ x: MLXArray) {
      guard Self.profileEnabled else { return }
      eval(x)
      let d = mark.duration(to: clock.now)
      let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
      FileHandle.standardError.write(Data("    \(name.padding(toLength: 20, withPad: " ", startingAt: 0)) \(String(format: "%7.2f", ms)) ms  \(x.shape)\n".utf8))
      mark = clock.now
    }
    var x = input
    if FrameGenerationFusedConv.enabled, let fusedHeads = block.fusedHeads, let fusedOut = block.fusedOut, x.dtype == .float16 {
      for (i, layer) in block.fusedStem.enumerated() {
        x = FrameGenerationFusedConv.apply(Self.padChannels(x, to: layer.cin), layer, activation: true, pool: true)
        lap("stem\(i)", x)
      }
      var skip = x
      for (i, layer) in block.fusedResidual.enumerated() {
        if i % 2 == 0 {
          let y = FrameGenerationFusedConv.apply(x, layer, activation: true)
          skip = x
          x = y
        } else {
          x = FrameGenerationFusedConv.apply(x, layer, activation: true, residual: skip)
        }
        lap("res\(i)", x)
      }
      let heads = FrameGenerationFusedConv.apply(Self.upsample2(x), fusedHeads, activation: true)
      lap("heads", heads)
      let fuseUpsample = FrameGenerationFusedConv.fusedUpsampleEnabled
      let out = FrameGenerationFusedConv.apply(fuseUpsample ? heads : Self.upsample2(heads), fusedOut,
        activation: false, upsample: fuseUpsample)
      lap("out", out)
      return (out[0..., 0..., 0..., 0..<4], out[0..., 0..., 0..., 4..<5], out[0..., 0..., 0..., 5..<8])
    }
    for (i, (w, b)) in block.stem.enumerated() {
      x = Self.meanPool2(Self.act(Self.conv(Self.padChannels(x, to: w.dim(3)), w, b)))
      lap("stem\(i)", x)
    }
    var skip = x
    for (i, (w, b)) in block.residual.enumerated() {
      let y = Self.act(Self.conv(x, w, b))
      if i % 2 == 0 {
        skip = x
        x = y
      } else {
        x = y + skip
      }
      lap("res\(i)", x)
    }
    let trunk = Self.upsample2(x)
    let heads = block.heads.map { Self.act(Self.conv(trunk, $0.0, $0.1)) }
    lap("heads", heads[2])
    let spans = [(0, 4), (4, 5), (5, 8)]
    let outs = spans.enumerated().map { (i, span) -> MLXArray in
      let w = block.outWeight[span.0..<span.1]
      let b = block.outBias[span.0..<span.1]
      return Self.conv(Self.upsample2(heads[i]), w, b)
    }
    return (outs[0], outs[1], outs[2])
  }

  // MARK: - Metal kernels (all batch-aware: sample n = index / (pixels per sample))

  /// Bilinear x2 upsample with the half-pixel convention (PyTorch align_corners=False), NHWC.
  private static let up2Kernel = MLXFast.metalKernel(
    name: "mlxdlss_fg_up2",
    inputNames: ["input", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel of every sample: params = [inHeight, inWidth, channels, batch].
      const uint index = thread_position_in_grid.x;
      const int ih = int(params[0]);
      const int iw = int(params[1]);
      const uint channels = params[2];
      const uint batch = params[3];
      const uint oh = uint(ih) * 2;
      const uint ow = uint(iw) * 2;
      if (index >= oh * ow * batch) { return; }
      const uint n = index / (oh * ow);
      const uint p = index % (oh * ow);
      const uint y = p / ow;
      const uint x = p % ow;
      auto src = input + n * uint(ih) * uint(iw) * channels;   // the input dtype is a template parameter
      const float sx = max((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const float sy = max((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const int x0 = min(int(sx), iw - 1);
      const int y0 = min(int(sy), ih - 1);
      const int x1 = min(x0 + 1, iw - 1);
      const int y1 = min(y0 + 1, ih - 1);
      const float tx = sx - float(x0);
      const float ty = sy - float(y0);
      for (uint c = 0; c < channels; ++c) {
        const float v00 = float(src[(y0 * iw + x0) * channels + c]);
        const float v01 = float(src[(y0 * iw + x1) * channels + c]);
        const float v10 = float(src[(y1 * iw + x0) * channels + c]);
        const float v11 = float(src[(y1 * iw + x1) * channels + c]);
        out[index * channels + c] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11;
      }
      """#
  )

  /// block0 input in one launch: [box2(A) rgb, err, box2(B) rgb, err, 0, t, 0 x6] at half
  /// resolution (zero-padded to the 16-multiple), err = [1/8,3/4,1/8]-blurred mean |box2(A)-box2(B)|.
  private static let block0InputKernel = MLXFast.metalKernel(
    name: "mlxdlss_fg_block0_input",
    inputNames: ["a", "b", "phase", "params"],
    outputNames: ["out"],
    source: #"""
      // params = [H, W, hp, wp, batch]; phase: [batch]; a, b: [batch,H,W,3] float; out: [batch,hp,wp,16] half
      const uint index = thread_position_in_grid.x;
      const int H = int(params[0]);
      const int W = int(params[1]);
      const int hp = int(params[2]);
      const int wp = int(params[3]);
      const uint batch = params[4];
      if (index >= uint(hp * wp) * batch) { return; }
      const uint n = index / uint(hp * wp);
      const uint p = index % uint(hp * wp);
      const int y = int(p) / wp;
      const int x = int(p) % wp;
      const float phaseValue = float(phase[n]);
      const device float* fa = a + n * uint(H * W * 3);
      const device float* fb = b + n * uint(H * W * 3);
      const int h2 = H / 2, w2 = W / 2;
      auto box = [&](const device float* img, int yy, int xx, thread float* rgb) {
        if (yy < 0 || yy >= h2 || xx < 0 || xx >= w2) { rgb[0] = rgb[1] = rgb[2] = 0.0f; return; }
        for (int c = 0; c < 3; ++c) {
          rgb[c] = 0.25f * (img[((2 * yy) * W + 2 * xx) * 3 + c] + img[((2 * yy) * W + 2 * xx + 1) * 3 + c]
                          + img[((2 * yy + 1) * W + 2 * xx) * 3 + c] + img[((2 * yy + 1) * W + 2 * xx + 1) * 3 + c]);
        }
      };
      float ra[3], rb[3];
      box(fa, y, x, ra); box(fb, y, x, rb);
      const float k[3] = {0.125f, 0.75f, 0.125f};
      float err = 0.0f;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = clamp(y + dy, 0, hp - 1), xx = clamp(x + dx, 0, wp - 1);
          float ta[3], tb[3];
          box(fa, yy, xx, ta); box(fb, yy, xx, tb);
          const float d = (fabs(ta[0] - tb[0]) + fabs(ta[1] - tb[1]) + fabs(ta[2] - tb[2])) / 3.0f;
          err += k[dy + 1] * k[dx + 1] * d;
        }
      }
      device half* o = out + index * 16;
      o[0] = ra[0]; o[1] = ra[1]; o[2] = ra[2]; o[3] = err;
      o[4] = rb[0]; o[5] = rb[1]; o[6] = rb[2]; o[7] = err;
      o[8] = 0.0h; o[9] = phaseValue;
      for (int c = 10; c < 16; ++c) { o[c] = 0.0h; }
      """#
  )

  /// `[N,hp,wp,16]` block0 input from full-res float32 frames `[N,H,W,3]` and `phases` `[N]`.
  static func block0Input(_ a: MLXArray, _ b: MLXArray, phases: MLXArray) -> MLXArray {
    let (n, h, w) = (a.dim(0), a.dim(1), a.dim(2))
    let hp = (h / 2 + 15) / 16 * 16, wp = (w / 2 + 15) / 16 * 16
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(hp), UInt32(wp), UInt32(n)])
    let count = n * hp * wp
    return block0InputKernel(
      [contiguous(a.asType(.float32)), contiguous(b.asType(.float32)), contiguous(phases.asType(.float32).reshaped([n])), params],
      grid: (count, 1, 1), threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[n, hp, wp, 16]], outputDTypes: [.float16]
    )[0]
  }

  /// block1 input in one launch: [warp(candA, 2 f0), warp(candB, 2 f0B), f0(4), m0, 0, r0(3), t, 0 x6]
  /// where f0/m0/r0 are block0's coarse outputs upsampled x2 (half-pixel bilinear) in-kernel.
  private static let block1InputKernel = MLXFast.metalKernel(
    name: "mlxdlss_fg_block1_input",
    inputNames: ["cand", "coarse", "phase", "params"],
    outputNames: ["out"],
    source: #"""
      // cand: [batch,hp,wp,16] (channels 0-3 = candA, 4-7 = candB); coarse: [batch,hp/2,wp/2,8]
      // params = [hp, wp, batch]; phase: [batch]; out: [batch,hp,wp,24] half
      const uint index = thread_position_in_grid.x;
      const int hp = int(params[0]);
      const int wp = int(params[1]);
      const uint batch = params[2];
      if (index >= uint(hp * wp) * batch) { return; }
      const uint n = index / uint(hp * wp);
      const uint p = index % uint(hp * wp);
      const int y = int(p) / wp;
      const int x = int(p) % wp;
      const float phaseValue = float(phase[n]);
      const int ch = hp / 2, cw = wp / 2;
      const device half* cn = cand + n * uint(hp * wp * 16);
      const device half* co = coarse + n * uint(ch * cw * 8);
      const float sx = max((float(x) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const float sy = max((float(y) + 0.5f) * 0.5f - 0.5f, 0.0f);
      const int cx0 = min(int(sx), cw - 1), cy0 = min(int(sy), ch - 1);
      const int cx1 = min(cx0 + 1, cw - 1), cy1 = min(cy0 + 1, ch - 1);
      const float tx = sx - float(cx0), ty = sy - float(cy0);
      float up[8];
      for (int c = 0; c < 8; ++c) {
        up[c] = (1 - tx) * (1 - ty) * float(co[(cy0 * cw + cx0) * 8 + c]) + tx * (1 - ty) * float(co[(cy0 * cw + cx1) * 8 + c])
              + (1 - tx) * ty * float(co[(cy1 * cw + cx0) * 8 + c]) + tx * ty * float(co[(cy1 * cw + cx1) * 8 + c]);
      }
      device half* o = out + index * 24;
      for (int which = 0; which < 2; ++which) {
        const float px = float(x) + 2.0f * up[which * 2];
        const float py = float(y) + 2.0f * up[which * 2 + 1];
        const float fx0 = floor(px), fy0 = floor(py);
        const float wx = px - fx0, wy = py - fy0;
        const int x0 = clamp(int(fx0), 0, wp - 1), x1 = clamp(int(fx0) + 1, 0, wp - 1);
        const int y0 = clamp(int(fy0), 0, hp - 1), y1 = clamp(int(fy0) + 1, 0, hp - 1);
        for (int c = 0; c < 4; ++c) {
          const int cc = which * 4 + c;
          const float v = (1 - wx) * (1 - wy) * float(cn[(y0 * wp + x0) * 16 + cc]) + wx * (1 - wy) * float(cn[(y0 * wp + x1) * 16 + cc])
                        + (1 - wx) * wy * float(cn[(y1 * wp + x0) * 16 + cc]) + wx * wy * float(cn[(y1 * wp + x1) * 16 + cc]);
          o[which * 4 + c] = v;
        }
      }
      o[8] = up[0]; o[9] = up[1]; o[10] = up[2]; o[11] = up[3];
      o[12] = up[4];
      o[13] = 0.0h;
      o[14] = up[5]; o[15] = up[6]; o[16] = up[7];
      o[17] = phaseValue;
      for (int c = 18; c < 24; ++c) { o[c] = 0.0h; }
      """#
  )

  /// `[N,hp,wp,24]` block1 input from the block0 input and block0's coarse `[N,hp/2,wp/2,8]` output.
  static func block1Input(_ cand: MLXArray, coarse: MLXArray, phases: MLXArray) -> MLXArray {
    let (n, hp, wp) = (cand.dim(0), cand.dim(1), cand.dim(2))
    let params = MLXArray([UInt32(hp), UInt32(wp), UInt32(n)])
    let count = n * hp * wp
    return block1InputKernel(
      [contiguous(cand.asType(.float16)), contiguous(coarse.asType(.float16)), contiguous(phases.asType(.float32).reshaped([n])), params],
      grid: (count, 1, 1), threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[n, hp, wp, 24]], outputDTypes: [.float16]
    )[0]
  }

  /// `[N,h,w,C]` -> `[N,2h,2w,C]`, bilinear, half-pixel centres.
  public static func upsample2(_ x: MLXArray) -> MLXArray {
    let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(c), UInt32(n)])
    let count = n * 4 * h * w
    return up2Kernel(
      [contiguous(x), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[n, 2 * h, 2 * w, c]],
      outputDTypes: [x.dtype]
    )[0]
  }

  /// Backward bilinear warp of NHWC images by per-pixel offsets (in pixels), borders clamped.
  private static let warpKernel = MLXFast.metalKernel(
    name: "mlxdlss_fg_warp",
    inputNames: ["image", "flow", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel: params = [height, width, channels, batch]; flow [batch,H,W,2].
      const uint index = thread_position_in_grid.x;
      const uint height = params[0];
      const uint width = params[1];
      const uint channels = params[2];
      const uint batch = params[3];
      if (index >= height * width * batch) { return; }
      const uint n = index / (height * width);
      const uint p = index % (height * width);
      const uint y = p / width;
      const uint x = p % width;
      auto img = image + n * height * width * channels;
      const float px = float(x) + float(flow[index * 2]);
      const float py = float(y) + float(flow[index * 2 + 1]);
      const float fx0 = floor(px);
      const float fy0 = floor(py);
      const float tx = px - fx0;
      const float ty = py - fy0;
      const int x0 = clamp(int(fx0), 0, int(width) - 1);
      const int x1 = clamp(int(fx0) + 1, 0, int(width) - 1);
      const int y0 = clamp(int(fy0), 0, int(height) - 1);
      const int y1 = clamp(int(fy0) + 1, 0, int(height) - 1);
      for (uint c = 0; c < channels; ++c) {
        const float v00 = float(img[(y0 * width + x0) * channels + c]);
        const float v01 = float(img[(y0 * width + x1) * channels + c]);
        const float v10 = float(img[(y1 * width + x0) * channels + c]);
        const float v11 = float(img[(y1 * width + x1) * channels + c]);
        out[index * channels + c] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11;
      }
      """#
  )

  /// The output kernel: upsample the half-resolution export (plain x/scale
  /// sampling, index-clamped), warp both frames by the scaled flows, blend by the mask.
  private static let composeKernel = MLXFast.metalKernel(
    name: "mlxdlss_fg_compose",
    inputNames: ["a", "b", "coarse", "params"],
    outputNames: ["out"],
    source: #"""
      // One thread per output pixel: params = [height, width, coarseHeight, coarseWidth, scale(bits), channels, sigmoidMask, batch].
      const uint index = thread_position_in_grid.x;
      const uint height = params[0];
      const uint width = params[1];
      const int ch = int(params[2]);
      const int cw = int(params[3]);
      const float scale = as_type<float>(params[4]);
      const int cc = int(params[5]);
      const bool sigmoidMask = params[6] != 0u;
      const uint batch = params[7];
      if (index >= height * width * batch) { return; }
      const uint n = index / (height * width);
      const uint p = index % (height * width);
      const uint y = p / width;
      const uint x = p % width;
      auto fa = a + n * height * width * 3;          // dtypes are template parameters
      auto fb = b + n * height * width * 3;
      auto co = coarse + n * uint(ch * cw * cc);
      const float u = min(float(x) / scale, float(cw - 1));
      const float v = min(float(y) / scale, float(ch - 1));
      const float u0 = floor(u);
      const float v0 = floor(v);
      const float tu = u - u0;
      const float tv = v - v0;
      const int cx0 = clamp(int(u0), 0, cw - 1);
      const int cx1 = clamp(int(u0) + 1, 0, cw - 1);
      const int cy0 = clamp(int(v0), 0, ch - 1);
      const int cy1 = clamp(int(v0) + 1, 0, ch - 1);
      float up[5];
      for (uint c = 0; c < 5; ++c) {
        float s00 = float(co[(cy0 * cw + cx0) * cc + c]);
        float s01 = float(co[(cy0 * cw + cx1) * cc + c]);
        float s10 = float(co[(cy1 * cw + cx0) * cc + c]);
        float s11 = float(co[(cy1 * cw + cx1) * cc + c]);
        if (c == 4 && sigmoidMask) {
          s00 = 1.0f / (1.0f + exp(-s00)); s01 = 1.0f / (1.0f + exp(-s01)); s10 = 1.0f / (1.0f + exp(-s10)); s11 = 1.0f / (1.0f + exp(-s11));
        }
        up[c] = (1 - tu) * (1 - tv) * s00 + tu * (1 - tv) * s01 + (1 - tu) * tv * s10 + tu * tv * s11;
      }
      const float m = up[4];
      float rgb[3] = {0, 0, 0};
      for (uint which = 0; which < 2; ++which) {
        const float px = float(x) + up[which * 2] * scale;
        const float py = float(y) + up[which * 2 + 1] * scale;
        const float fx0 = floor(px);
        const float fy0 = floor(py);
        const float tx = px - fx0;
        const float ty = py - fy0;
        const int x0 = clamp(int(fx0), 0, int(width) - 1);
        const int x1 = clamp(int(fx0) + 1, 0, int(width) - 1);
        const int y0 = clamp(int(fy0), 0, int(height) - 1);
        const int y1 = clamp(int(fy0) + 1, 0, int(height) - 1);
        const float weight = which == 0 ? m : (1 - m);
        auto img = which == 0 ? fa : fb;
        for (uint c = 0; c < 3; ++c) {
          const float v00 = float(img[(y0 * width + x0) * 3 + c]), v01 = float(img[(y0 * width + x1) * 3 + c]);
          const float v10 = float(img[(y1 * width + x0) * 3 + c]), v11 = float(img[(y1 * width + x1) * 3 + c]);
          rgb[c] += weight * ((1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v01 + (1 - tx) * ty * v10 + tx * ty * v11);
        }
      }
      for (uint c = 0; c < 3; ++c) { out[index * 3 + c] = clamp(rgb[c], 0.0f, 1.0f); }
      """#
  )

  /// `image` [N,H,W,C], `flow` [N,H,W,2] pixel offsets -> [N,H,W,C] in the image's dtype.
  static func warp(_ image: MLXArray, flow: MLXArray) -> MLXArray {
    let (n, h, w, c) = (image.dim(0), image.dim(1), image.dim(2), image.dim(3))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(c), UInt32(n)])
    let count = n * h * w
    return warpKernel(
      [contiguous(image), contiguous(flow.asType(image.dtype)), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [image.shape],
      outputDTypes: [image.dtype]
    )[0]
  }

  /// `a`, `b` [N,H,W,3] frames; `coarse` [N,h,w,C] (flowA, flowB, mask[, ...]) -> [N,H,W,3] in the frames' dtype.
  public static func compose(_ a: MLXArray, _ b: MLXArray, coarse: MLXArray, scale: Float = 2, sigmoidMask: Bool = false) -> MLXArray {
    let (n, h, w) = (a.dim(0), a.dim(1), a.dim(2))
    let params = MLXArray([UInt32(h), UInt32(w), UInt32(coarse.dim(1)), UInt32(coarse.dim(2)), scale.bitPattern, UInt32(coarse.dim(3)), sigmoidMask ? 1 : 0, UInt32(n)])
    let count = n * h * w
    return composeKernel(
      [contiguous(a), contiguous(b.asType(a.dtype)), contiguous(coarse), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [[n, h, w, 3]],
      outputDTypes: [a.dtype]
    )[0]
  }

  // MARK: - Public API

  /// Half-resolution export `[N, h, w, 8]`: flowA.xy, flowB.xy, mask logit, residual.rgb, for `[N,H,W,3]` frames and `[N]` phases.
  public func synthesize(_ aFull: MLXArray, _ bFull: MLXArray, phases: MLXArray) -> MLXArray {
    if Self.compileEnabled {
      return compiledSynthesize([aFull, bFull, phases])[0]
    }
    return synthesizeGraph(aFull, bFull, phases: phases)
  }

  /// One phase for the whole batch.
  public func synthesize(_ aFull: MLXArray, _ bFull: MLXArray, phase: Float) -> MLXArray {
    synthesize(aFull, bFull, phases: MLXArray(Array(repeating: phase, count: aFull.dim(0))))
  }

  private func synthesizeGraph(_ aFull: MLXArray, _ bFull: MLXArray, phases: MLXArray) -> MLXArray {
    if Self.fusedInputsEnabled, dtype == .float16, FrameGenerationFusedConv.enabled, block0.fusedHeads != nil {
      return synthesizeFused(aFull, bFull, phases: phases)
    }
    let clock = ContinuousClock()
    var mark = clock.now
    func lap(_ name: String, _ arrays: MLXArray...) {
      guard Self.profileEnabled else { return }
      eval(arrays)
      let now = clock.now
      let ms = Double((mark.duration(to: now)).components.attoseconds) / 1e15 + Double((mark.duration(to: now)).components.seconds) * 1e3
      FileHandle.standardError.write(Data("  \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) \(String(format: "%7.2f", ms)) ms\n".utf8))
      mark = now
    }
    let n = aFull.dim(0)
    let a = Self.box2(aFull.asType(dtype))
    let b = Self.box2(bFull.asType(dtype))
    let err = Self.photometricError(a, b)
    let zero = MLXArray.zeros(err.shape, dtype: dtype)
    let t = broadcast(phases.asType(dtype).reshaped([n, 1, 1, 1]), to: err.shape)
    let candA = concatenated([a, err], axis: 3)
    let candB = concatenated([b, err], axis: 3)
    lap("candidates", candA, candB)
    let (f0c, m0c, r0c) = run(block0, concatenated([candA, candB, zero, t], axis: 3))
    lap("block0", f0c, m0c, r0c)
    let f0 = Self.upsample2(f0c), m0 = Self.upsample2(m0c), r0 = Self.upsample2(r0c)
    let warpedA = Self.warp(candA, flow: f0[0..., 0..., 0..., 0..<2] * 2)
    let warpedB = Self.warp(candB, flow: f0[0..., 0..., 0..., 2..<4] * 2)
    lap("upsample + warps", warpedA, warpedB, m0, r0)
    let (f1, m1, r1) = run(block1, concatenated([warpedA, warpedB, f0, m0, zero, r0, t], axis: 3))
    lap("block1", f1, m1, r1)
    let flow: MLXArray = f0 * 2 + f1
    let mask: MLXArray = m0 + m1
    let residual: MLXArray = r0 + r1
    return concatenated([flow, mask, residual], axis: 3)
  }

  /// The float16 graph with the block inputs assembled by single kernels.
  private func synthesizeFused(_ aFull: MLXArray, _ bFull: MLXArray, phases: MLXArray) -> MLXArray {
    let input0 = Self.block0Input(aFull, bFull, phases: phases)             // [N,hp,wp,16]
    let (f0c, m0c, r0c) = run(block0, input0)                               // coarse [N,hp/2,wp/2,·]
    let coarse = concatenated([f0c, m0c, r0c], axis: 3)                     // [.., 8]
    let input1 = Self.block1Input(input0, coarse: coarse, phases: phases)   // [N,hp,wp,24]
    let (f1, m1, r1) = run(block1, input1)
    let up = Self.upsample2(coarse)
    let flow: MLXArray = up[0..., 0..., 0..., 0..<4] * 2 + f1
    let rest: MLXArray = up[0..., 0..., 0..., 4..<8] + concatenated([m1, r1], axis: 3)
    return concatenated([flow, rest], axis: 3)
  }

  /// Diagnostics: the graph up to a stage (0 candidates, 1 block0, 2 warps, 3 block1), eager.
  public func synthesizeUpTo(_ stage: Int, _ aFull: MLXArray, _ bFull: MLXArray, phase: Float) -> MLXArray {
    let n = aFull.dim(0)
    let a = Self.box2(aFull.asType(dtype))
    let b = Self.box2(bFull.asType(dtype))
    let err = Self.photometricError(a, b)
    let zero = MLXArray.zeros(err.shape, dtype: dtype)
    let t = broadcast(MLXArray(Array(repeating: phase, count: n)).asType(dtype).reshaped([n, 1, 1, 1]), to: err.shape)
    let candA = concatenated([a, err], axis: 3)
    let candB = concatenated([b, err], axis: 3)
    if stage == 0 { return concatenated([candA, candB], axis: 3) }
    let (f0c, m0c, r0c) = run(block0, concatenated([candA, candB, zero, t], axis: 3))
    if stage == 1 { return concatenated([f0c, m0c, r0c], axis: 3) }
    let f0 = Self.upsample2(f0c), m0 = Self.upsample2(m0c), r0 = Self.upsample2(r0c)
    let warpedA = Self.warp(candA, flow: f0[0..., 0..., 0..., 0..<2] * 2)
    let warpedB = Self.warp(candB, flow: f0[0..., 0..., 0..., 2..<4] * 2)
    let input1 = concatenated([warpedA, warpedB, f0, m0, zero, r0, t], axis: 3)
    if stage == 2 { return input1 }
    let (f1, m1, r1) = run(block1, input1)
    return concatenated([f0 * 2 + f1, m0 + m1, r0 + r1], axis: 3)
  }

  private func check(_ a: MLXArray, _ b: MLXArray) throws {
    guard a.ndim == 4, a.dim(3) == 3 else { throw Error(description: "frames must be [N, H, W, 3], got \(a.shape)") }
    guard a.shape == b.shape else { throw Error(description: "frames differ in shape: \(a.shape) vs \(b.shape)") }
  }

  /// Interpolated frames `[N, H, W, 3]` in [0, 1]: sample i is the frame between `a[i]` and `b[i]` at `phases[i]`.
  /// `a` and `b` may hold one frame each (`[1, H, W, 3]`) for several phases.
  public func interpolate(_ a: MLXArray, _ b: MLXArray, phases: [Float]) throws -> MLXArray {
    try interpolate(a, b, phases: phases, rgb8: false)
  }

  /// Interpolated RGB8 frames, rounded after composition, with one GPU evaluation through quantization.
  /// Accepts the same normalized inputs and phase batches as `interpolate(_:_:phases:)`.
  public func interpolateRGB8(_ a: MLXArray, _ b: MLXArray, phases: [Float]) throws -> MLXArray {
    try interpolate(a, b, phases: phases, rgb8: true)
  }

  private func interpolate(_ a: MLXArray, _ b: MLXArray, phases: [Float], rgb8: Bool) throws -> MLXArray {
    var out = try interpolationGraph(a, b, phases: phases)
    if rgb8 { out = (out * 255 + 0.5).asType(.uint8) }
    eval(out)
    return out
  }

  /// Internal device composition can append output packing before the evaluation boundary.
  func interpolationGraph(_ a: MLXArray, _ b: MLXArray, phases: [Float]) throws -> MLXArray {
    try check(a, b)
    guard !phases.isEmpty else { throw Error(description: "at least one phase is needed") }
    var aa = a, bb = b
    if a.dim(0) == 1, phases.count > 1 {
      aa = broadcast(a, to: [phases.count, a.dim(1), a.dim(2), 3])
      bb = broadcast(b, to: [phases.count, b.dim(1), b.dim(2), 3])
    }
    guard aa.dim(0) == phases.count else { throw Error(description: "\(phases.count) phases for \(aa.dim(0)) frame pairs") }
    let export = synthesize(aa, bb, phases: MLXArray(phases))
    return Self.compose(aa.asType(.float32), bb.asType(.float32), coarse: export, sigmoidMask: true)
  }

  /// The single frame `[1, H, W, 3]` between `a` and `b` at `phase`.
  public func interpolate(_ a: MLXArray, _ b: MLXArray, phase: Float = 0.5) throws -> MLXArray {
    try interpolate(a, b, phases: [phase])
  }

  /// `factor - 1` intermediate frames at phases k/factor, computed as one batch.
  public func generate(_ a: MLXArray, _ b: MLXArray, factor: Int) throws -> [MLXArray] {
    guard factor >= 2 else { throw Error(description: "factor must be at least 2") }
    let out = try interpolate(a, b, phases: (1..<factor).map { Float($0) / Float(factor) })
    return (0..<(factor - 1)).map { out[$0..<($0 + 1)] }
  }

  /// For consecutive `frames` f0…fk (each `[1, H, W, 3]`), the `factor - 1` frames of every
  /// pair (f_i, f_{i+1}) in one batch: returns `[k, factor - 1, H, W, 3]`-shaped nested lists.
  public func generatePairs(_ frames: [MLXArray], factor: Int) throws -> [[MLXArray]] {
    guard factor >= 2 else { throw Error(description: "factor must be at least 2") }
    guard frames.count >= 2 else { return [] }
    for f in frames { try check(f, frames[0]) }
    let pairs = frames.count - 1
    let n = factor - 1
    let phases = (0..<pairs).flatMap { _ in (1..<factor).map { Float($0) / Float(factor) } }
    let a = concatenated(frames.dropLast().flatMap { f in Array(repeating: f, count: n) }, axis: 0)
    let b = concatenated(frames.dropFirst().flatMap { f in Array(repeating: f, count: n) }, axis: 0)
    let out = try interpolate(a, b, phases: phases)
    return (0..<pairs).map { p in (0..<n).map { i in out[(p * n + i)..<(p * n + i + 1)] } }
  }
}
