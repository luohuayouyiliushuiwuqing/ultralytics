# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal shared helpers for the sky_workflow stages.

NOTE: this is a reconstructed stub. The original `sky_only` module lives in the
v3 project and was not present in this checkout. Only the symbols imported by
stages/01_label_teacher.py, 04_masks_to_yolo.py and 05_build_dataset.py are
provided here. Replace with the v3 original if/when it becomes available.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

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

# Default teacher location and the tint used when drawing the sky region.
DEFAULT_TEACHER = Path(__file__).resolve().parent.parent / "models" / "skyseg_teacher.onnx"
# OpenCV is BGR, so this is magenta. Deliberately a colour the scene itself does not
# contain: a blue tint would melt into the sky it is marking, and green into the grass.
SKY_TINT = (255, 60, 255)


def probability(session, image: np.ndarray) -> np.ndarray:
    """Sky probability in [0, 1], resampled back to the frame's own resolution.

    Two things fail silently if you get them wrong: the teacher needs ImageNet
    normalization (plain /255 returns garbage that still looks like a plausible
    mask), and it is fixed at NET_SIZE, so a non-square frame is squashed before
    inference and stretched back afterwards.
    """
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (NET_SIZE, NET_SIZE)), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)
    raw = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(blob[None])})[0]
    return cv2.resize(raw[0, 0], (width, height), interpolation=cv2.INTER_LINEAR)


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tint the sky region so its boundary is visible at thumbnail size."""
    vis = image.copy()
    sky = mask > 127
    tint = np.array(SKY_TINT, dtype=np.float32)
    vis[sky] = (vis[sky].astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    return vis


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image, tolerating non-ASCII paths (cv2.imwrite fails on some)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(path.suffix or ".png", image)[1].tobytes())


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


def main() -> int:
    """Run the teacher on one frame and write the original, its mask, and an overlay.

        python sky_workflow/stages/sky_only.py datasets/images/d3_1015_001.jpg

    Writes three files into --out (default: the current directory, deliberately NOT
    next to the source frame, which must stay untouched):
        <stem>_orig.<ext>    the source frame, byte for byte
        <stem>_mask.png      binary sky mask (255 = sky, 0 = ground)
        <stem>_overlay.png   the frame with the sky region tinted

    Useful for checking the teacher on a frame before committing a whole labeling
    pass to it.
    """
    parser = argparse.ArgumentParser(
        description="Run the sky teacher on one frame: original + mask + overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="frame to inspect")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output directory (default: current directory)")
    parser.add_argument("-m", "--model", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="probability -> sky")
    parser.add_argument("--device", default="cpu", help="cpu / auto / GPU / NPU")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay tint strength 0..1")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"no image at {args.image}", file=sys.stderr)
        return 1
    image = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"unreadable image: {args.image}", file=sys.stderr)
        return 1

    try:
        model = ensure_teacher(args.model)
    except Exception as error:  # offline or blocked: say so plainly, no stack trace
        print(f"could not obtain teacher {args.model}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    session = make_session(model, args.device)
    prob = probability(session, image)
    mask = (prob >= args.threshold).astype(np.uint8) * 255

    out = args.out or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    original = out / f"{stem}_orig{args.image.suffix.lower()}"
    original.write_bytes(args.image.read_bytes())  # exact copy, no re-encode
    mask_path = out / f"{stem}_mask.png"
    overlay_path = out / f"{stem}_overlay.png"
    write_image(mask_path, mask)
    write_image(overlay_path, overlay(image, mask, args.alpha))

    print(f"{args.image.name}  {image.shape[1]}x{image.shape[0]}  "
          f"sky={float(prob.mean()):.1%}  (threshold {args.threshold}, "
          f"{getattr(session, 'device', 'onnxruntime CPU')})")
    print(f"  original -> {original}")
    print(f"  mask     -> {mask_path}")
    print(f"  overlay  -> {overlay_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
