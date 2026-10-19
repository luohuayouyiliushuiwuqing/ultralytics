# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal shared helpers for the sky_workflow stages.

NOTE: this is a reconstructed stub. The original `sky_only` module lives in the
v3 project and was not present in this checkout. Only the symbols imported by
stages/01_label_teacher.py, 04_masks_to_yolo.py and 05_build_dataset.py are
provided here. Replace with the v3 original if/when it becomes available.
"""

from __future__ import annotations

import sys
import numpy as np
import urllib.request
from pathlib import Path

# Image extensions accepted as sources across the pipeline.
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# JianyuanWang/skyseg teacher, trimmed to a single output. ~168 MB, downloaded once
# into sky_workflow/models/ the first time it is needed. Mirrors are tried in order:
# huggingface.co is unreachable from some networks while the mirror still resolves.
TEACHER_URLS = (
    "https://hf-mirror.com/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx?download=true",
)
TEACHER_URL = TEACHER_URLS[0]
# A real skyseg.onnx is ~168 MB; anything tiny is an error page served with HTTP 200.
TEACHER_MIN_BYTES = 1024 * 1024


class Progress:
    """Minimal terminal progress bar (stdlib only -- no tqdm dependency).

    Redraws one line in place with \\r. When stdout is not a terminal (piped to a
    file or captured by CI) the bar is suppressed and only a closing summary line
    is printed, so logs stay readable.
    """

    def __init__(self, total: float, label: str = "", unit: str = "", scale: float = 1.0, width: int = 30):
        self.total = max(float(total), 1.0)
        self.label = label
        self.unit = unit
        self.scale = scale
        self.width = width
        self.done = 0.0
        self.live = sys.stdout.isatty()

    def _render(self) -> str:
        ratio = min(self.done / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = "=" * filled + (">" if filled < self.width else "")
        shown = f"{self.done / self.scale:.0f}/{self.total / self.scale:.0f}{self.unit}"
        return f"{self.label} [{bar:<{self.width}}] {ratio:6.1%} {shown}"

    def update(self, done: float) -> None:
        self.done = min(float(done), self.total)
        if not self.live:
            return
        sys.stdout.write("\r" + self._render())
        sys.stdout.flush()

    def close(self) -> None:
        if self.live:
            self.update(self.total)
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(f"{self.label} {self.total / self.scale:.0f}{self.unit}")

# JianyuanWang/skyseg uses ImageNet normalization at a fixed 320x320 input.
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NET_SIZE = 320


def ensure_teacher(path: str | Path, urls: tuple = TEACHER_URLS) -> Path:
    """Return the teacher ONNX path, downloading it if it is missing.

    Downloads to a `.part` sibling first so an interrupted run never leaves a
    truncated file that looks complete to the next run. Each mirror in `urls` is
    tried in order; a response smaller than TEACHER_MIN_BYTES is treated as an
    error page rather than a model.
    """
    path = Path(path)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    failures = []
    for url in urls:
        print(f"teacher {path.name} missing -- downloading {url}")
        progress = None

        def _report(blocks, block_size, total):
            nonlocal progress
            if progress is None and total:  # total is 0 when the server omits a length
                progress = Progress(total, label=f"  {path.name}", unit=" MB", scale=1e6)
            if progress:
                progress.update(blocks * block_size)

        try:
            urllib.request.urlretrieve(url, part, reporthook=_report)
            size = part.stat().st_size
            if size < TEACHER_MIN_BYTES:
                raise OSError(f"only {size} bytes -- not a ~168 MB model")
            part.replace(path)
            if progress:
                progress.close()
            else:
                print(f"  downloaded -> {path} ({size / 1e6:.0f} MB)")
            return path
        except Exception as error:
            failures.append(f"  {url}: {type(error).__name__}: {error}")
            if progress:
                progress.close()
            try:
                part.unlink()
            except OSError:
                pass
    raise RuntimeError("could not download the teacher ONNX:\n" + "\n".join(failures))


def make_session(model: str | Path, device: str = "auto"):
    """Return an inference session over the teacher ONNX.

    device == "cpu" -> onnxruntime CPU (always available, deterministic).
    device in {auto, GPU, NPU} -> OpenVINO if importable, else onnxruntime CPU.
    The returned object exposes ``get_inputs()[0].name`` and ``run(None, feed)[0]``
    so the calling stages need no changes.
    """
    model = str(model)

    def _ov_session():
        from openvino.runtime import Core

        core = Core()
        compiled = core.compile_model(model, "AUTO")
        in_name = compiled.inputs[0].any_name
        out_name = compiled.outputs[0].any_name

        class _In:
            name = in_name

        class _OVSession:
            device = "OpenVINO AUTO"

            def get_inputs(self):
                return [_In()]

            def run(self, _, feed):
                return [compiled({in_name: feed[in_name]})[out_name]]

        return _OVSession()

    if device == "cpu":
        import onnxruntime as ort

        return ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    try:
        return _ov_session()
    except Exception:
        import onnxruntime as ort

        return ort.InferenceSession(model, providers=["CPUExecutionProvider"])
