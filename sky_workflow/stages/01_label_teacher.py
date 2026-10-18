"""Stage 1 -- teacher labeling: frames -> sky probability + binary mask.

    python sky_workflow/stages/01_label_teacher.py input -o sky_workflow/work/mask

Writes a mirrored tree of lossless .png masks (255 = sky, 0 = ground) and, with
--prob, a matching tree of .npy probability maps. The probability maps are what a
soft-label distillation needs later; the binary masks are what the YOLO polygon
export consumes. Generate both while the teacher is loaded -- re-running it costs
a full pass.

The teacher is JianyuanWang/skyseg trimmed to a single output. Two things about it
are easy to get wrong and both fail silently:

* It needs ImageNet normalization. Feeding plain /255 returns garbage that still
  looks like a plausible mask.
* It is fixed at 320x320, so a 640x480 frame is squashed to square before
  inference and the result is stretched back. The boundary lands in the right
  place, but the teacher *judged* a distorted image.

--device GPU runs the same graph through OpenVINO on the Arc iGPU at ~22 ms/frame
instead of ~174 ms on CPU, with masks differing by 0-2 pixels.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sky_only import EXTENSIONS, IMG_MEAN, IMG_STD, NET_SIZE, make_session  # noqa: E402


def probability(session, image: np.ndarray) -> np.ndarray:
    """Sky probability in [0, 1], resampled back to the frame's own resolution."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (NET_SIZE, NET_SIZE)), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)
    raw = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(blob[None])})[0]
    return cv2.resize(raw[0, 0], (width, height), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=here / "work/mask")
    parser.add_argument("--prob", type=Path, default=None, help="also write float32 .npy maps here")
    parser.add_argument("-m", "--model", type=Path, default=here / "models/skyseg_teacher.onnx")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto/GPU/NPU use OpenVINO; cpu uses onnxruntime")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"missing teacher {args.model} -- see sky_workflow/README.md", file=sys.stderr)
        return 1

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images under {args.input}", file=sys.stderr)
        return 1

    session = make_session(args.model, args.device)
    root = args.input
    started = time.perf_counter()
    for count, path in enumerate(files, 1):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip unreadable {path}")
            continue
        prob = probability(session, image)
        rel = path.relative_to(root)
        mask_target = args.out / rel.with_suffix(".png")
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        mask_target.write_bytes(cv2.imencode(".png", (prob >= args.threshold).astype(np.uint8) * 255)[1].tobytes())
        if args.prob:
            prob_target = args.prob / rel.with_suffix(".npy")
            prob_target.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_target, prob.astype(np.float32))
        if count % 200 == 0 or count == len(files):
            rate = count / (time.perf_counter() - started)
            print(f"  {count}/{len(files)}  {rate:.1f} img/s  sky={prob.mean():.1%}")

    print(f"{args.out}  ({len(files)} masks, device={getattr(session, 'device', 'onnxruntime CPU')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
