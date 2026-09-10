"""Shared Swift arguments for live previews and native video exports."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

from .effects import DLSSSuperResolution, FrameGen, NeuralRender, OutputOptions, SuperResolution


def available(settings, effects, source: Path) -> bool:
    version = platform.mac_ver()[0].split(".")[0]
    return (platform.system() == "Darwin" and version.isdigit() and int(version) >= 26
            and source.suffix.lower() not in {".mkv", ".webm", ".avi"}
            and all(settings.resolved_backend(e.kind) == "mlxdlss" for e in effects)
            and (bool(effects) or settings.resolved_backend("fg") == "mlxdlss")
            and not any(isinstance(e, NeuralRender) and e.motion == "flow" for e in effects))


def rendering_arguments(nr: NeuralRender | None, settings, *, video: bool) -> list[str]:
    if nr is None:
        return []
    if not settings.nr_model:
        raise ValueError("Choose a Metal model in Settings")
    args = ["--model", str(Path(settings.nr_model).expanduser()), "--profile", nr.profile]
    for name in ("processing_scale", "detail_strength", "colour_strength", "detail_radius", "intensity"):
        args += ["--" + name.replace("_", "-"), str(getattr(nr, name))]
    if video:
        args += ["--temporal", "on" if nr.temporal else "off", "--motion", nr.motion,
                 "--scene-cut-threshold", str(nr.scene_cut_threshold)]
    return args


def super_resolution_arguments(vsr: SuperResolution | None, settings) -> list[str]:
    if vsr is None:
        return []
    if not settings.has_vsr_weights():
        raise ValueError("Choose RTX VSR weights in Settings")
    return ["--vsr-weights", str(Path(settings.vsr_weights).expanduser())]


def dlss_sr_arguments(sr: DLSSSuperResolution | None, settings) -> list[str]:
    if sr is None:
        return []
    if not settings.has_sr_model():
        raise ValueError("Choose a DLSS SR model in Settings")
    return ["--sr-model", str(Path(settings.sr_model).expanduser())]


def video_arguments(source, target, effects, settings, output: OutputOptions) -> list[str]:
    from ..mlxdlss_stream import find_mlxdlss

    nr = next((e for e in effects if isinstance(e, NeuralRender)), None)
    fg = next((e for e in effects if isinstance(e, FrameGen)), None)
    sr = next((e for e in effects if isinstance(e, DLSSSuperResolution)), None)
    args = [find_mlxdlss(settings.mlxdlss_binary or None), "process-video", str(source), "--output", str(target)]
    args += rendering_arguments(nr, settings, video=True)
    args += dlss_sr_arguments(sr, settings)
    audio = output.include_audio and (fg is None or fg.audio != "none")
    args += ["--codec", output.codec, "--audio", "on" if audio else "off", "--start-frame", str(output.start_frame)]
    if output.frame_limit is not None:
        args += ["--frames", str(output.frame_limit)]
    if fg is not None:
        if not settings.fg_weights:
            raise ValueError("Choose frame-generation weights in Settings")
        args += ["--framegen-weights", str(Path(settings.fg_weights).expanduser()), "--factor", str(fg.factor),
                 "--slow-motion", "on" if fg.mode == "slowmo" else "off",
                 "--order", "fg-nr" if effects[0].kind == "fg" else "nr-fg"]
    return args


# macOS's GPU watchdog kills command buffers that block display compositing while
# the display is active (IOGPU "Impacting Interactivity"); MLX cannot catch that,
# so the child process dies. The driver hint below is the workaround recommended
# in ml-explore/mlx#3267, and one retry covers the remaining kills.
WATCHDOG_ENVIRONMENT = {"AGX_RELAX_CDM_CTXSTORE_TIMEOUT": "1"}
WATCHDOG_SIGNATURE = "kIOGPUCommandBufferCallbackErrorImpactingInteractivity"


def child_environment(**overrides) -> dict:
    return {**WATCHDOG_ENVIRONMENT, **os.environ, **overrides}


def run_media(command, report, should_stop) -> dict:
    from .runners import Cancelled

    output_index = command.index("--output") + 1
    target = Path(command[output_index])
    if target.exists():
        raise FileExistsError(f"Output already exists: {target.name}")
    # Keep encoder staging and temporary audio inside a single owned directory,
    # including when cancellation kills Swift before its defers can run.
    with tempfile.TemporaryDirectory(prefix=".native-", dir=target.parent) as directory:
        pending = Path(directory) / target.name
        args = list(command)
        args[output_index] = str(pending)
        environment = child_environment(TMPDIR=directory)
        try:
            result = _run(args, report, should_stop, environment)
        except RuntimeError as error:
            if WATCHDOG_SIGNATURE not in str(error) or should_stop():
                raise
            report("GPU watchdog interrupted the job, retrying once", 0.0, 0, 0)
            result = _run(args, report, should_stop, environment)
        if should_stop():
            raise Cancelled()
        os.link(pending, target)
        result["output"] = target.resolve().as_uri()
        return result


def _run(command, report, should_stop, environment) -> dict:
    """Drain progress without filling a pipe; cancellation also reaps the child."""
    from .runners import Cancelled

    # Separate handles let us reread progress without moving the child's write
    # position. This also works on Windows, where os.pread is unavailable.
    with tempfile.TemporaryDirectory(prefix="mlxdlss-progress-") as directory, \
            (Path(directory) / "stderr").open("w+b") as errors, \
            (Path(directory) / "stderr").open("rb") as progress:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True, env=environment)
        try:
            while True:
                if should_stop():
                    raise Cancelled()
                try:
                    stdout, _ = process.communicate(timeout=0.15)
                    break
                except subprocess.TimeoutExpired:
                    progress.seek(0)
                    lines = progress.read(1_000_000).decode(errors="replace")
                    matches = re.findall(r"(\d+)/(\d+) input frames, (\d+) output", lines)
                    if matches:
                        done, total, out = map(int, matches[-1])
                        report(f"{done} input → {out} output frames", min(0.97, done / max(1, total)), done, total)
            if process.returncode:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace")[-2000:] or "Native video processing failed")
            return json.loads(stdout)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout:
                process.stdout.close()
