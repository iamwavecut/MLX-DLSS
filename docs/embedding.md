# Embedding the neural-rendering component

The `mlxdlss` commands are thin wrappers around these types; an application, a
media pipeline or a server drives the recovered checkpoint through the
library alone. Every example assumes a user-supplied `MODEL.dlssmodel` produced
by `mlxdlss-weights`; MLX-DLSS never bundles or redistributes weights.

| Use case | Type | Cadence |
| --- | --- | --- |
| Still image, screenshot, video frames without history | `NeuralRenderingFirstFrameBackend` | frame independent |
| Frames with motion vectors, depth and display history | `NeuralRenderingTemporalReferenceBackend` | consecutive frames |
| Raw float32 tensors from any host | either backend around a `NeuralRenderBackend` head | as above |

Both backends conform to `NeuralRenderBackend`, so a host swaps the head
(`MLXNeuralRenderer` or `CoreMLNeuralRenderer`) or the temporal path without
touching its own frame loop.

## Network geometry

The checkpoint runs on a network extent of at least `320×320` and multiples of
`64`. `NeuralRenderingNetworkGeometryPolicy` maps a logical frame onto it:
`.vendorAligned` (default) mirrors the frame to the right and bottom, runs the
head on the aligned extent and crops the result, so `1920×1080` runs at
`1920×1088`; `.matchOutput` feeds the frame unchanged and is only for sizes the
head accepts directly.

## Still frame

```swift
import DLSSCore
import DLSSMLX

let head = try MLXNeuralRenderer(
  packageURL: modelPackageURL, executionMode: .metalFused, computePrecision: .float16
)
var configuration = NeuralRenderingFirstFrameConfiguration(profile: .standard)
configuration.intensity = 1
configuration.featureControls = NeuralRenderingFeatureControls(
  normalizedStyle: 0, localToneStrength: 1, localStructureStrength: 1
)
let backend = NeuralRenderingFirstFrameBackend(head: head, configuration: configuration)

let color = try HostTensor(
  descriptor: TensorDescriptor(name: "color", shape: [1, height, width, 3], dataType: .float32, layout: .nhwc),
  bytes: rgbFloat32Data
)
let result = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [color]))
let rgb = result.output(named: "color")!.bytes   // [1, height, width, 3] float32 in [0, 1]
```

An optional `controlMask` tensor (`[1, H, W, 3]`, red = blend, green = tone,
blue = structure) travels next to `color` in the same request. The photoreal
recipe (`--processing-scale`, `--detail-strength`, `--colour-strength`) is
`NeuralRenderingDetailComposition.resample` and `.compose` applied around the
backend exactly as `mlxdlss run` and `mlxdlss render-image` do.

## Temporal reference

```swift
let temporal = NeuralRenderingTemporalReferenceBackend(
  backend: head, depthInverted: false, controlMaskIntensity: 1,
  temporalPreprocessor: try MetalNeuralRenderingTemporalFeaturePreprocessor(depthGuideMode: .flat)
)
let request = try NeuralRenderRequest(
  sequenceID: 1,
  inputs: [color, motion, depth],   // motion: [1, H, W, 2] normalized history-UV offsets, depth: [1, H, W, 1]
  temporalContext: NeuralRenderFrameContext(streamID: 1, frameIndex: frameIndex)
)
let output = try await temporal.render(request).output(named: "color")
```

Consecutive `frameIndex` values keep the display history; a gap, a stream
change or a reset request clears it. `NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion`
converts engine pixel motion with its scale and jitter.

## Metal video output and frame generation

`MLXNeuralRenderingDeviceTemporalBackend` can compose display RGB and generate
intermediate frames inside its actor, keeping the float32 NR → FG arrays on
the GPU. The portable `render()` method still returns the original full-size
`HostTensor`; opt into the streaming output separately:

```swift
import Foundation

let video = try MLXNeuralRenderingDeviceTemporalBackend(
  packageURL: modelPackageURL, executionMode: .metalFused, computePrecision: .float16,
  videoOutput: MLXVideoOutputOptions(width: outputWidth, height: outputHeight,
                                   detailStrength: 2, format: .u8),
  frameGeneration: MLXVideoFrameGenerationOptions(weightsURL: frameGenerationWeightsURL,
                                                precision: .float16, factor: 2, batch: 4)
)
// Call sequentially for each input frame; request contains processing-size RGB and guides.
let written = try await video.renderToStream(request, source: originalColor, to: destination)
// Once, at end of input, emit any remaining generated/original pairs.
let finalWritten = try await video.finishStream(to: destination)
```

`originalColor` is a float32 NHWC `HostTensor` at the configured output extent;
`destination` is a writable `FileHandle`. The output is raw RGB in the selected
`.f32`, `.u8`, or `.u16` format. Omit `frameGeneration` for one display frame per
input. With FG, calls can write zero or several frames: one first original,
then batches of generated/original frames in order. Keep one actor per video,
and drain `finishStream` before closing the output. Temporal resets affect NR
history while retaining the consecutive FG pairs, matching the video adapter.

## Fixed-shape Core ML head

```swift
import DLSSCoreML

let head = try await CoreMLNeuralRenderer(
  modelURL: coreMLPackageURL,    // converted at the network extent, e.g. 320×320
  configuration: CoreMLBackendConfiguration(computeUnits: .cpuAndGPU)
)
```

The package is compiled for one network extent: a `256×256` frame needs a
`320×320` package, `1080p` needs `1920×1088`. A Core ML package freezes the
graph at conversion time, so rebuild it with `mlxdlss-weights coreml` whenever the
recovered graph changes; the MLX path applies such fixes at load time.

## Native media and live preview

On macOS 26+, `DLSSMedia` owns AVFoundation decode/encode, VideoToolbox or Vision
motion, temporal NR, FG and optional DLSS SR / RTX VSR 2×. It requires no Python process
or raw-frame pipe:

```swift
import DLSSMedia

var options = MediaProcessingOptions(renderingModel: modelPackageURL,
                                     frameGenerationWeights: frameGenerationWeightsURL)
options.detailStrength = 2
let processor = NativeMediaProcessor()
let result = try await processor.processVideo(input: inputURL, output: outputURL,
                                              options: options) { progress in
  // Update UI on its actor; keep this callback short.
  print(progress.inputFrames, progress.outputFrames)
}
```

For images, omit FG weights and call `processImage`. For a custom frame loop,
`MLXVideoFrame` imports IOSurface-backed pixel buffers and
`MLXPixelBufferWriter` exports them; `renderVideoFrame` retains temporal history
and applies display settings without a host float32 copy. Submit one frame at
a time per renderer. Cancellation prevents publication of a partial output;
existing files are never replaced.

Set `options.superResolutionWeights` to enable [VSR 2×](super-resolution.md),
alone or as the last effect. It doubles the output dimensions and also applies
to live preview. `MLXNativeSuperResolver.upscale` accepts individual native
frames; VSR retains the reference model's RGB8 quantization.

For temporal video upscaling, set `options.dlssSuperResolutionModel` to a
prepared `.srmodel` instead. DLSS SR uses motion and recurrent history and
runs after NR/FG. Choose one upscaler per job.

`NativeMediaPreview.render(MediaPreviewRequest(input:isVideo:time:options:))`
returns original/processed `CGImage`s, the actual selected timestamp and timing.
Retain one preview actor to reuse weights and decoded frames. Submit requests
sequentially, coalesce changes and discard superseded responses, as the native
app does. Video preview uses up to three preceding frames and a fresh temporal
history per request; FG is reserved for export. The actor keeps its
VideoToolbox optical-flow session between requests of the same size and motion
mode, so a seek does not pay for a new session. Preview and export should share
one scheduling lane so they do not compete for the GPU.

`mlxdlss preview-stream` exposes that actor to the web UI on macOS 26+. Each
JSON line contains `input`, `video`, `time` and an `options` array of CLI flags.
Each response contains base64 PNGs (`original`, `processed`), dimensions,
`time`, `duration`, `frameInterval`, `historyFrames` and `elapsedSeconds`.
Errors return an `error` field; the process remains available for the next request.

The native `process-video` command exposes these options in addition to the
[rendering controls](../README.md#cli):

| Options | Behavior |
| --- | --- |
| `--temporal on\|off`, `--motion automatic\|vision\|videotoolbox\|zero` | History and motion backend; Automatic falls back to Vision if VT cannot start |
| `--scene-cut-threshold 0.3` | Reset unreliable history; 0 disables automatic resets |
| `--start-frame N`, `--frames N` | Select an input range |
| `--codec h264\|hevc\|prores`, `--bitrate BPS` | Encoder; ProRes requires MOV |
| `--factor 2`, `--order nr-fg\|fg-nr`, `--slow-motion on` | FG cadence/order; slow motion preserves audio pitch |
| `--audio off` | Omit audio |
| `--vsr-weights PATH` | RTX VSR 2× after NR and FG; also supported by `process-image` |
| `--sr-model PATH` | Temporal DLSS SR 2× after NR and FG; video only |

Original timestamps and variable frame intervals are retained. FG emits
`(N-1)×factor+1` frames. JSON reports stage timings; stderr reports progress.
Native video is SDR 8-bit sRGB; PNG/TIFF exports retain 16 bits. The Python
adapter supports custom FFmpeg arguments and RGB16 video.

## Performance

M2 Max, 38-core GPU, real weights. Processing scale increases network pixel
count quadratically. Compare complete frame loops at the same scale and output
format; warm kernel timings exclude media I/O and startup.

### 1080p, HT and small networks (2026-10-17)

Network alone, warm, best of five (`PerfSpikeKernelTimingTests.testWholeFrame`):

| Network extent | HT at `0ca2dea` | HT now | Small now |
| --- | --- | --- | --- |
| 768×1024 | 70.8 ms | 52.6 ms | 27.5 ms |
| 1088×1920 | 173.8 ms | 134.0 ms | 72.8 ms |

`mlxdlss process-video` over 60 frames of a 1920×1080 H.264 clip, temporal,
automatic motion, no audio, whole command after a warm-up run. The runs
alternated between the two builds:

| Build | HT | Small |
| --- | --- | --- |
| `0ca2dea` | 12.18–12.19 s | not supported |
| now | 9.59–9.69 s | 5.91–6.05 s |

Peak footprint fell from 5.7–5.8 GB to 3.9 GB for HT; small takes 2.6 GB.
Later runs of the current build on the same day were 23–30% slower while
another process kept the GPU 92% busy; compare only runs taken together.

Every change reproduces the previous output bit for bit. With `--motion zero`
both builds write the same video (SHA-256 of the decoded RGB). With automatic
motion, VideoToolbox optical flow can differ between runs of the same build,
and the recurrent history carries such a difference into later frames.

The changes, each with a diagnostics switch:

- The multi-head window attention keeps one window row per simdgroup in
  registers and only the keys and values in threadgroup memory.
- The multi-head blocks' branched feed-forward with its published residual is
  one kernel at 64 and 128 channels (`MLXDLSS_FUSED_FFN=0`,
  `MLXDLSS_FUSED_FFN_CHANNELS`); at 256 channels the GEMM chain stays faster.
- The 64-channel attention block (QKV, attention, projection, residual) is one
  kernel (`MLXDLSS_FUSED_ATTENTION_BLOCK=0`,
  `MLXDLSS_FUSED_ATTENTION_BLOCK_CHANNELS`).
- The single-head kernel absorbs its neighbours: the stem adapter and 2×2 pool,
  the downsample pool, the upsample merge, and the post block's merge and
  output head (`MLXDLSS_FUSED_1H_GLUE=0`).
- Frames whose extent is not a multiple of 64, such as 1080p, build their
  features on the padded network extent in one kernel instead of gathering
  logical features onto it: 82.8 → 75.7 ms per small-network 1080p frame
  (`MLXDLSS_NR_EXTENDED_FEATURES=0`).

The 1088×1920 small network spent 26 ms in the two full-resolution
single-head blocks (stem and post), 18 ms in the global stage, whose GEMMs run
at 5.6 TMAC/s against MLX's 6.1 TMAC/s peak on this GPU, and the rest in 26
blocks of 1.3–3.4 ms (`NeuralRenderingBlockTimingTests`).

### Window kernels without lane shuffles (2026-10-17, later)

The window kernels sum each softmax row in the recovered sequential order.
The four lanes that hold a row used to pass its 64 weights around with 32
`simd_shuffle` calls, and those shuffles took about 12% of the single-head
kernel. Each simdgroup now stores its weight tiles in 512 bytes of
threadgroup scratch and every lane reads its row back, 32 keys at a time. The
per-token norm partials go through the simdgroup's own value rows the same
way. Threadgroup memory per window grows from 8 KB to 12 KB. Three smaller
changes ride along:

- The half path of E4M3 rounding clamps the magnitude at 448 and rounds it
  with one magic-number add whose exponent follows the input (11 integer and
  half operations instead of about 16). Probabilities, which are non-negative
  and below 448, and attended values, which cannot be NaN, use shorter forms.
  `PerfSpikeKernelTimingTests.testFastE4M3IsBitExactOnGPU` checks every half
  value against the float path.
- The bit-affine softmax transform works on each half on its own: within the
  clamp range, `(packed << 5) + 0x7FF88000` equals `(bits << 5) ^ 0x8000` per
  half.
- The stem kernel pools the window from threadgroup memory instead of lane
  shuffles.

Kernel timings in one process, alternating rounds against the earlier
sources, minimum at 1088×1920 unless noted (`NeuralRenderingKernelVariantTests`):

| Kernel | Before | After |
| --- | --- | --- |
| Single-head block | 10.07 ms | 8.81 ms |
| Stem: adapter, block 0, pool | 10.63 ms | 9.11 ms |
| Post: merge, block, head | 10.38 ms | 8.97 ms |
| Attention core, 2 heads, 272×480 | 0.74 ms | 0.55 ms |
| Attention core, 8 heads, 68×120 | 0.32 ms | 0.27 ms |

The whole network (`PerfSpikeKernelTimingTests.testWholeFrame`, best of five
per run, four alternating runs of each build) went from 81.7–83.9 ms to
75.6–78.0 ms for small and from 151.0–156.0 ms to 139.5–142.0 ms for HT at
1088×1920. A browser and the window server kept the GPU 10–60% busy during
these runs, so their absolute times sit above the table above; compare only
within each pair.

All outputs stay bit-identical: the stem and post kernels at every window
phase against the earlier sources, the single-head block and attention core
likewise, and whole frames of both networks at 384×384, 768×1024 and
1088×1920 by SHA-256 of the output.

Tried on the single-head kernel without a gain (same measurement, changes
within noise unless noted):

- Feed-forward weight tiles read as vector loads in fragment order, or
  prefetched one hidden tile ahead: no change, or up to 19% slower with full
  unrolling.
- Feed-forward weights staged in threadgroup memory: 5–10% slower.
- Two window rows per simdgroup (`MLXDLSS_1H_ROWS=2`): 16–23% slower.
- Rolled hidden-tile loops: 5–11% slower.
- More threadgroup memory per window costs occupancy: 24 KB is 2–9% slower
  and 32 KB 14% slower than 16 KB; 8 KB with one more barrier showed no
  difference against 12 KB.

The feed-forward half of the kernel (128 of its 256 simdgroup MMAs per window
row) and the E4M3 roundings remain the main costs.

### Global stage (2026-10-17, later)

The eight `vit_1d` global blocks (1024 channels, 32 heads of 32) see 640
tokens at 1088×1920 and 192 at 768×1024. Two changes:

- One kernel publishes the QKV projection: cosine queries with the head
  scale, cosine keys and E4M3 values, written straight into the attention
  kernel's layouts. It replaces three publishes over transposed heads, their
  copies and the padding. The attention kernel writes its rows in token order,
  so the transpose back is gone, and the attention residual publishes in the
  same pass instead of a separate E4M3 round trip.
- The resident attention kernel reads keys and values lane-major: a lane
  fetches its fragment pairs for four 8-token tiles (for keys, the four
  8-channel chunks of a tile) with two 8-byte loads instead of one
  `simdgroup_load` per fragment. Eight simdgroups score whole 32-token groups
  with four independent accumulators and turn the scores into vendor weights
  in registers; one lane per row sums the weights in key order, 32 per batch of
  loads; four simdgroups run the value chains. Half operands feed the float
  accumulators directly, which is bit-identical to converting them first.

In one process, alternating rounds, quiet GPU:

| Measurement | Before | After |
| --- | --- | --- |
| Attention kernel and transpose back, 640 tokens | 0.50 ms | 0.27 ms |
| Same, 192 tokens | 0.081 ms | 0.068 ms |
| Same, 1024 tokens | 1.52 ms | 0.73 ms |
| Eight blocks, 640 tokens, median | 22.45 ms | 19.76 ms |
| Eight blocks, 192 tokens, median | 9.85 ms | 9.47 ms |

The block rows come from `NeuralRenderingGlobalBlockABTests`
(`MLXDLSS_GLOBAL_BLOCK_AB=1`), which times the block against the previous
pipeline with the previous kernel source. The whole network
(`PerfSpikeKernelTimingTests.testWholeFrame`, best of five per run, four
alternating runs of each build):

| Network extent | Small before | Small after | HT before | HT after |
| --- | --- | --- | --- | --- |
| 768×1024 | 25.3–26.4 ms | 24.7–25.5 ms | 49.7–55.0 ms | 48.3–53.4 ms |
| 1088×1920 | 66.8–67.6 ms | 64.6–65.6 ms | 122.0–138.3 ms | 119.8–127.6 ms |

Outputs stay bit-identical. The resident kernel equals the materialized
attention at 2–1024 tokens (`testGlobalAttentionV2IsBitExactAndTiming`), the
block equals the previous pipeline at 2–1024 tokens
(`NeuralRenderingGlobalBlockABTests`), and whole frames of both networks at
384×384, 768×1024 and 1088×1920 keep their SHA-256 for two inputs each.
`MLXDLSS_GLOBAL_ATTENTION_V2=0` restores the earlier kernels and the separate
publishes.

Tried without a gain:

- 16 query rows per threadgroup: 13–25% slower; the doubled threadgroup
  memory costs more occupancy than the halved key and value traffic saves.
- One score tile per simdgroup instead of whole 32-token groups: 6–10%
  slower.
- Conflict-free score row stride alone, prefetching value tiles one step
  ahead, keys and values in 8×8 tile-major order: within noise.
- A custom GEMM that keeps MLX's K order reproduces MLX's bits (no differing
  element at the four global shapes), but the best of six tile configurations
  ran 10–15% slower than MLX's kernel. Fusing the gate, residual and publish
  epilogues into it would not pay.

The four GEMMs, about 1.5 ms of each block at 640 tokens, remain the main
cost.

### Export and preview timeline (2026-10-17, later)

A Metal System Trace of a 1080p small-network export shows the GPU busy for
65–70 ms of each frame, with about 0.02 ms idle between command buffers.
Scheduling has no gaps left to recover inside a frame.

VideoToolbox optical flow takes 22–37 ms per frame and bounds the export of
SD clips; the pipeline adds 4–6 ms around it. Each flow takes the previous
one as a hint, so running flows in parallel would change the output.

Live preview keeps its VideoToolbox session across requests of the same frame
size and motion mode, and `NativeOpticalFlow.reset()` starts each new window
as a fresh session would. Starting a session costs about 0.4 s. A seek with
temporal history fell from 0.75–1.1 s to 0.16–0.17 s on a 654×480 clip and
takes 0.54–0.58 s at 1216×1664. The reused session returns the same preview
as a new one. The first preview after launch takes 1.3–1.9 s: 0.34 s to load
the renderer, 0.43 s for the first network frames and 0.4 s for the session.

### MLX 0.33 and the GPU Neural Accelerators (2026-11-23)

mlx-swift 0.33.0 (MLX 0.32.4) replaces 0.31.6. On the M2 Max both networks
return the same output bit for bit (SHA-256 at 384×384, 768×1024 and
1088×1920, two inputs each), and frame times stay within the run-to-run noise.

mlx-swift builds MLX in JIT mode: GEMM, convolution, attention and elementwise
kernels compile from source at first use. `scripts/prepare-mlx-metallib.sh`
now copies the 4.7 MB library that Swift Build compiles into
`mlx-swift_Cmlx.bundle` instead of building MLX's full library with CMake
(147 MB for MLX 0.32.4, 108 MB for 0.31.6). Outputs are unchanged, and
`.build/MLX DLSS.app` shrank from 155 MB to 68 MB. CMake and Ninja remain the
fallback when the bundle has no library.

M5, M6 and later have a Neural Accelerator in every GPU core. On macOS 26.2
and later MLX routes float16 and bfloat16 matrix multiplications there through
JIT-compiled Metal 4 tensor kernels; in this network these are the global
stage's GEMMs. The custom window, single-head and feed-forward kernels use
simdgroup matrix instructions on the shader cores.

- `mlxdlss device` prints the GPU architecture and whether MLX uses the Neural
  Accelerators in this process (`MLXGPUProfile`).
- `MLX_METAL_GPU_ARCH=applegpu_g16s` on an M5 Pro or Max makes MLX take the
  shader path; forcing a Neural Accelerator architecture on an older GPU
  returns wrong output.
- `MLXDLSS_TENSOR_OPS_FFN=1` (macOS 26 and later) runs the fused feed-forward
  at 64 and 128 channels as `matmul2d` tensor operations, which the Neural
  Accelerators execute; `MLXDLSS_TENSOR_OPS_FFN_ROWS` selects 16 or 32 rows per
  simdgroup. On the M2 Max it matches the simdgroup kernel bit for bit, in the
  whole network too, and is not faster (minimum of 15 samples, 1088×1920
  shapes):

  | Shape | Simdgroup kernel | Tensor ops, 16 rows | Tensor ops, 32 rows |
  | --- | --- | --- | --- |
  | 64 ch, 272×480 | 0.98 ms | 1.00 ms | 1.39 ms |
  | 128 ch, 136×240 | 0.83 ms | 0.88 ms | 1.25 ms |

- `scripts/bench-neural-accelerators.sh` compares, on M5 or M6, the Neural
  Accelerator path, the shader path and the tensor-operation feed-forward:
  output difference from the first configuration, then alternating whole-frame
  timings. With arguments it compares any environment configurations, for
  example `'fused=' 'chain=MLXDLSS_FUSED_FFN=0'`.

No M5 or M6 measurement exists yet; the tensor-operation feed-forward stays
off until one shows a gain.

### Earlier measurements

Same 228-frame 512×384, 60 fps clip, temporal rendering, detail 2:

| Path | Whole clip | Input frames/s |
| --- | --- | --- |
| Before the earlier optimizations | 18.4–22.8 s | 10.0–12.4 |
| Earlier optimized Python/Metal pipeline | 10.1–16.1 s | 14.2–22.6 |
| Native pipeline | 11.33–12.36 s | 18.4–20.1 |

Desktop load varied across days; these ranges are not a paired speedup claim.
Earlier alternating runs showed about 1.4× throughput. The latest four native
A/B runs showed another 4–9% with global attention/graph caching enabled, versus
11.77–13.43 s disabled. All four decoded RGB streams had the same SHA-256.
They ran without an executable search path. Decode/encode took 0.37–0.48 s,
motion 3.19–3.32 s, NR 7.06–7.73 s; peak process footprint was 1.54–1.81 GB.

Short global attention sequences use on-chip tiles and cached MLX graphs while
preserving full spatial context, half denominators and E4M3 publication.
Before graph caching, eight blocks took 3.79 vs 4.85–5.51 ms at 48 tokens, and
21.59 vs 24.26–24.92 ms at 510 tokens, with exact output matches. Larger
sequences retain materialized attention. `MLXDLSS_STREAMED_GLOBAL_ATTENTION=0`
and `MLXDLSS_COMPILE_GLOBAL=0` disable the optimizations; forcing streamed
attention with `=1` above 512 tokens saves memory but was slower on this GPU.
See [NR/FG chain](frame-generation.md#gpu-video-chain) and
[FG output-head fusion](frame-generation.md#native-output-head-fusion) for
warm timings, including the additional 18–22% FG throughput gain.

Memory: the earlier Metal still-image path used 2.2 GB resident at 4K.
PyTorch float32 used about 1 GB per megapixel of network input; `fast` precision
roughly halves it. Bounded chunks prevent window count from growing the peak
(`MLXDLSS_TORCH_CHUNK_TOKENS`, 0 disables). PyTorch was within 0.002 MAE of Metal;
Core ML was within 0.008–0.014 MAE of the vendor captures.

Experiments excluded from the default path:

- **FP8 packing:** exact weight decode/GEMM took 0.54–2.90 ms vs MLX's
  0.25–0.84 ms. Packing window intermediates halved storage but gave no speed
  gain: three 1088×1920 blocks took 50.97 vs 50.46–50.61 ms.
- **ANE:** direct [ANEForge](https://github.com/sbryngelson/ANEForge) dispatch
  of the 1024→4096 projection took 0.30/0.51/1.04 ms at 48/192/768 tokens vs
  Metal's 0.27/0.35/0.73 ms, excluding copies; max half-output error was 0.001.
  Core ML chose ANE at 192 tokens (1.05 ms including Python prediction) and CPU
  at 48. These layer probes do not validate the whole network on ANE.
- **Temporal feature reuse:** updating the global latent every other frame
  added 0.0037–0.0072 mean RGB error, max 0.064, on 16 frames. A linear correction
  did not help. This used fresh encoder inputs and independent-frame rendering;
  a learned replacement needs a separate training corpus and temporal quality gate.
