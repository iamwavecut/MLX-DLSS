"""One cached preview per page; inference shares the export worker's GPU lock."""
from __future__ import annotations

import base64
import asyncio
import io
import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import native
from .effects import DLSSSuperResolution, NeuralRender, SuperResolution, parse_effects, validate_chain


class LatestPreview:
    """One active render and one replaceable pending request, with a debounce."""
    def __init__(self, render, publish, error, *, busy=lambda: False, delay=0.18):
        self.render, self.publish, self.error = render, publish, error
        self.busy, self.delay = busy, delay
        self.pending = None
        self.revision = 0
        self.task = None
        self.closed = False

    def submit(self, request):
        if self.closed:
            return
        self.revision += 1
        self.pending = (self.revision, request)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._work())

    async def _work(self):
        while self.pending is not None and not self.closed:
            await asyncio.sleep(self.delay)
            if self.closed:
                break
            if self.busy():
                await asyncio.sleep(0.2)
                continue
            revision, request = self.pending
            self.pending = None
            try:
                result = await asyncio.to_thread(self.render, request)
                if revision == self.revision and not self.closed:
                    self.publish(result)
            except Exception as error:
                if revision == self.revision and not self.closed:
                    self.error(str(error))

    def close(self):
        self.closed = True
        self.revision += 1
        self.pending = None


class PreviewSession:
    def __init__(self, runner):
        self.runner = runner
        self.process = None
        self.process_key = None
        self.errors = None
        self.source_key = None
        self.frames = []
        self.metadata = {}
        self.closed = threading.Event()

    def render(self, source: Path, kind: str, effects_raw, seconds: float = 0) -> dict:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Preview time must be finite and non-negative")
        effects = parse_effects(effects_raw, kind=kind)
        if effects:
            validate_chain(effects, kind)
        nr = next((e for e in effects if isinstance(e, NeuralRender)), None)
        vsr = next((e for e in effects if isinstance(e, SuperResolution)), None)
        sr = next((e for e in effects if isinstance(e, DLSSSuperResolution)), None)
        with self.runner.cache.execution_lock:
            if self.closed.is_set():
                raise RuntimeError("Preview closed")
            settings = self.runner.settings_provider()
            if native.available(settings, [e for e in (nr, vsr, sr) if e is not None], source):
                return self._native(source, kind, nr, vsr, settings, seconds, sr)
            self._stop_process()
            if sr is not None:
                raise ValueError("DLSS SR needs native Metal on macOS 26 and MP4/MOV input")
            if vsr is not None:
                raise ValueError("RTX VSR needs native Metal on macOS 26; select Auto or Metal in Settings")
            return self._portable(source, kind, nr, settings, seconds)

    def _native(self, source, kind, nr, vsr, settings, seconds, sr=None):
        from ..mlxdlss_stream import find_mlxdlss

        binary = find_mlxdlss(settings.mlxdlss_binary or None)
        options = native.rendering_arguments(nr, settings, video=kind == "video") + native.super_resolution_arguments(vsr, settings)
        options += native.dlss_sr_arguments(sr, settings)
        if self.process is None or self.process.poll() is not None or self.process_key != binary:
            self._stop_process()
            self.errors = tempfile.TemporaryFile()
            self.process = subprocess.Popen([binary, "preview-stream"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.errors, text=True, bufsize=1,
                                            env=native.child_environment())
            self.process_key = binary
        process = self.process
        if self.closed.is_set():
            self._stop_process()
            raise RuntimeError("Preview closed")
        request = {"input": str(source), "video": kind == "video", "time": seconds,
                   "options": options}
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                self.errors.seek(0)
                raise RuntimeError(self.errors.read().decode(errors="replace")[-1500:] or "Native preview stopped; rebuild mlxdlss")
            result = json.loads(line)
            if "error" in result:
                raise ValueError(result["error"])
            return result
        except (BrokenPipeError, OSError):
            self._stop_process()
            raise RuntimeError("Native preview stopped; rebuild mlxdlss") from None

    def _portable(self, source, kind, nr, settings, seconds):
        started = time.perf_counter()
        key = (str(source), kind, seconds if kind == "video" else 0)
        if self.source_key != key:
            self._load_frames(source, kind, seconds)
            self.source_key = key
        original = self.frames[-1]
        frames = self.frames if nr and nr.temporal and kind == "video" else [original]
        processed = original
        if nr is not None:
            stage = self.runner._neural_rendering_stage(nr, settings)
            height, width = original.shape[:2]
            for processed in stage.apply(iter(frames), width, height):
                if self.closed.is_set():
                    raise RuntimeError("Preview closed")
        return {**self.metadata, "original": _png(original), "processed": _png(processed),
                "width": original.shape[1], "height": original.shape[0],
                "historyFrames": len(frames) - 1, "elapsedSeconds": time.perf_counter() - started}

    def _load_frames(self, source, kind, seconds):
        if kind == "image":
            from PIL import ImageOps
            with Image.open(source) as image:
                self.frames = [np.asarray(ImageOps.exif_transpose(image).convert("RGB"), np.float32) / 255]
            self.metadata = {"time": 0, "duration": 0, "frameInterval": 1 / 30}
            return
        import cv2

        capture = cv2.VideoCapture(str(source))
        try:
            rate = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not capture.isOpened() or rate <= 0 or count <= 0:
                raise ValueError("Cannot read video preview")
            index = min(count - 1, max(0, int(math.ceil(seconds * rate - 1e-6))))
            first = max(0, index - 3)
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            frames = []
            for _ in range(first, index + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Cannot decode the selected frame")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
            self.frames = frames
            self.metadata = {"time": index / rate, "duration": count / rate, "frameInterval": 1 / rate}
        finally:
            capture.release()

    def _stop_process(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        if self.errors is not None:
            self.errors.close()
            self.errors = None

    def close(self):
        self.closed.set()
        # Interrupt a blocked pipe read before waiting for the inference lock.
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()
        with self.runner.cache.execution_lock:
            self._stop_process()
            self.frames = []


def _png(frame):
    data = io.BytesIO()
    Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)).save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()
