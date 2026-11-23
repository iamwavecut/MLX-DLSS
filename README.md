# MLX-DLSS

> [!TIP]
> **A note for the NVIDIA reader.** This port was worked out on a laptop and on
> GPU instances rented by the hour, some of which even booted. A pair of DGX
> Sparks would have replaced the rentals and would have a steady job here:
> experiments like this one, and the pet projects queued behind it. Hit me up on
> X: [@WaveCut](https://x.com/WaveCut).

Run NVIDIA's DLSS neural rendering and frame generation on images and video.
Apple Silicon uses MLX and Metal; PyTorch supports other GPUs and CPUs.

This is an experimental port, not a game integration or an NVIDIA product.
Bring your own NVIDIA libraries to extract weights. No vendor binaries or
weights are included or downloaded.

![Neural rendering: input, defaults, scale 2 with detail 2](docs/assets/neural-rendering-control.png)

[Frame-generation comparison](https://github.com/user-attachments/assets/ce94f426-910b-4556-bdf9-662cbdd5933a)

## Changelog

- **2026-11-23**: MLX 0.33 with the same output; the app ships MLX's 4.7 MB
  runtime-compiled kernel library instead of the full one (155 → 68 MB).
  `mlxdlss device` shows whether MLX uses the GPU Neural Accelerators of M5
  and M6, and `scripts/bench-neural-accelerators.sh` compares the paths there
  ([details](docs/embedding.md#mlx-033-and-the-gpu-neural-accelerators-2026-11-23)).
- **2026-10-17**: supports the 2× faster DLSS 5 neural rendering model from
  [NVIDIA's October update](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/)
  (the small network; extract it from `nvngx_dlssnr.dll` 310.9.16.0).

## Choose a mode

| Mode | Runtime | Preview |
| --- | --- | --- |
| macOS app | Swift, Metal, AVFoundation | Live settings; split comparison; frame stepping on a filmstrip |
| Web UI | Python, NiceGUI; native Metal media or PyTorch/FFmpeg | Live settings and video timeline |
| CLI / Python API | Native Metal or PyTorch | File output |

On macOS 26+, the web UI uses the same native preview and media pipeline as
the app. Other platforms use the portable Python pipeline.
`mlxdlss-web --native` is a separate pywebview wrapper and still needs Python.

## macOS quick start

Requires Apple Silicon, macOS 26+ and Xcode with Swift 6.2+. CMake and Ninja are
needed only when SwiftPM does not compile MLX's Metal library itself (see
`scripts/prepare-mlx-metallib.sh`).

```sh
scripts/build-native-app.sh
open '.build/MLX DLSS.app'
```

Set your [prepared weights](#weights) once in **Settings › Models** (⌘,).
Drop images or video on the window, compare the rendered frame with the
original and process the queue (⌘↩). The app includes its CLI and Metal library.
It needs no Python or FFmpeg at runtime.

The canvas has four compare modes: Original, Split, Rendered and Side by Side
(⌘1–⌘4). Drag the split handle to move the divider; hold M to see the original.
Pinch, ⌘-scroll or double-click to zoom; ⌘0 shows actual pixels and ⌘9 fits
the frame. On video, drag along the filmstrip or press ←/→ to step one frame:
the decoded frame shows at once and the rendered frame follows. The inspector
lists the stages in processing order; double-click a slider's name to reset it.

Temporal rendering is on by default. Live preview uses up to three preceding
frames; export uses the full sequence and applies frame generation. Native
video output is SDR 8-bit; PNG/TIFF stills retain 16-bit output.

Experimental [super resolution 2×](docs/super-resolution.md): DLSS SR for video,
RTX VSR for images. Both support live preview and require separate model files.

![Native macOS app: split comparison of a video frame, filmstrip and inspector](docs/assets/native-app.png)

## Weights

Initial extraction uses Python 3.10+:

```sh
python3 -m pip install ./python
mlxdlss-weights all nvngx_dlssnr.dll weights/
mlxdlss-weights extract-fg nvngx_dlssg.dll weights/framegen.safetensors
```

| Library | Supported builds | Networks |
| --- | --- | --- |
| `nvngx_dlssnr.dll` | **310.8.0.0** | HT (71 blocks) |
| `nvngx_dlssnr.dll` | **310.9.16.0** | HT and the small FP8 network (31 blocks), NVIDIA's new default |
| `nvngx_dlssg.dll` | **310.8.0**, **310.9.16** | frame generation |
| `libnvidia-ngx-dlssg.so` | **310.7.0** (DLSS SDK) | frame generation, the same weights |

`mlxdlss-weights sha256 FILE` checks the DLL build.

Outputs: `NeuralRendering.dlssmodel` (HT) and, from 310.9.16,
`NeuralRendering-Small.dlssmodel` for Metal; the matching
`dlssnr-weights-logical.safetensors` and `dlssnr-small-weights-logical.safetensors`
for PyTorch; `framegen.safetensors` for either backend.

The small network, the newest and about twice as fast, is the default:
`mlxdlss-weights extract` takes it when the DLL carries it, the app selects it
and shows an HT/Small switch (a saved HT package moves to the small one beside
it until you pick a network yourself), and `scripts/bench.sh` and
`scripts/verify-local.sh` use it first. The
[small network](docs/recovery-notes.md#the-small-fp8-network) has no NVIDIA
parity measurement yet.

## CLI

The app build also creates `.build/release/mlxdlss`:

```sh
.build/release/mlxdlss process-image in.png --output out.png --model weights/NeuralRendering-Small.dlssmodel
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --model weights/NeuralRendering-Small.dlssmodel
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --framegen-weights weights/framegen.safetensors --factor 2
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --sr-model weights/dlss-sr.srmodel
```

Combine `--model` and `--framegen-weights`; `--order nr-fg|fg-nr` selects their
order. Video defaults to temporal rendering, automatic optical flow, H.264 and
audio. Existing output files are preserved. [More options and APIs](docs/embedding.md#native-media-and-live-preview).

| Rendering control | Default | Effect |
| --- | --- | --- |
| `--profile` | `standard` | `standard`, `natural`, `cinematic`, `neutral` |
| `--processing-scale` | `1` | More detail at higher cost; range 1–4 |
| `--detail-strength` | `1` | Strength of fine changes; range 0–8 |
| `--colour-strength` | `1` | Strength of broad colour changes; range 0–4 |
| `--intensity` | `1` | Overall effect strength; 0 bypasses it |

## Web and Python

Requires Python 3.10+. FFmpeg/ffprobe serve portable video and comparisons:

```sh
python3 -m pip install './python[web,video]'
mlxdlss-web                       # http://127.0.0.1:8181
```

Set weights and backend in **Settings**. Preview images or a selected video
frame while adjusting controls. Export batches with NR/FG in either order,
FG ×2–16, slow motion, H.264/HEVC/ProRes, audio and frame ranges. Jobs support
cancel, retry and downloads; the output folder is configurable.

On macOS 26+, **Upscale 2×** uses RTX VSR for images and DLSS SR for video.
Configure their model files in Settings; preview and export share the Metal path.

Temporal is on for new videos. Preview uses up to three preceding frames;
frame generation runs on export. Saved jobs retain their settings.

![Web panel with temporal preview and export controls](docs/assets/web-video.png)

For the web's Metal backend (image/tensor APIs support macOS 14+):

```sh
swift build -c release
MLXDLSS_BUILD_CONFIGURATION=release scripts/prepare-mlx-metallib.sh "$(swift build -c release --show-bin-path)"
```

Repeat after a clean build or an MLX update to place a matching `mlx.metallib`
beside the binary.
For custom FFmpeg filters, RGB16 video, PyTorch, Core ML conversion and the
Python API, see the [Python guide](python/README.md).

## Accuracy and speed

| Measurement | Result |
| --- | --- |
| Neural rendering vs NVIDIA | 0.004–0.005 MAE on 1152–1408 px game renders |
| Frame generation vs NVIDIA | 59.9 dB PSNR; maximum difference 3/255 |
| Native temporal video, M2 Max | 18.4–20.1 input frames/s |
| 1080p temporal video, M2 Max | HT 6.2–6.3 input frames/s, small 9.9–10.2 |

The video sample has 228 frames at 512×384, detail strength 2: **11.33–12.36 s**
including startup, motion, decode and encode. The 1080p clip has 60 frames:
HT 9.59–9.69 s (12.18 s at `0ca2dea`), small 5.91–6.05 s.
Desktop load affects timings.
See [measurements and experiments](docs/embedding.md#performance),
[FG benchmarks](docs/frame-generation.md#speed) and [parity limits](docs/recovery-notes.md).

## Docs and development

- [Swift API and native media](docs/embedding.md)
- [Frame generation](docs/frame-generation.md)
- [Super resolution: RTX VSR and DLSS SR](docs/super-resolution.md)
- [Model recovery](docs/recovery-notes.md) and [research](docs/research/)

```sh
scripts/verify.sh                 # tests, builds and public-tree audit
```

CI tests Swift/Metal on Apple Silicon and Python on macOS, Linux and Windows.
Keep weights and captures outside Git. See [Contributing](CONTRIBUTING.md),
[Security](SECURITY.md), [Publication](PUBLICATION.md) and [Notice](NOTICE).

Source: [Apache 2.0](LICENSE). Extracted models retain the vendor's terms and
must not be redistributed.
