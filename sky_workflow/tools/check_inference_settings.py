"""Diagnostic -- does max_det=1 lose sky coverage, and what does it cost?

    python sky_workflow/tools/check_inference_settings.py --count 24

Sky is one region, so the instance machinery in the seg head is overhead at
inference time: ultralytics scores up to 8400 boxes, runs class-agnostic NMS over
the survivors, then interpolates a mask for each of up to 300 kept detections --
and all of that is CPU code, which is exactly where it lands on the board. Capping
the head prices that out; on this dataset post-processing costs several times more
than the network itself.

The question this answers is whether capping is *safe*: if the model paints the sky
as several instances, keeping only the top one drops part of it. Each cap re-runs
the same frames, and the resulting mask is IoU'd both against the teacher polygon
label and against the loosest cap's mask. IoU(label) holding flat while vs-union
stays near 1.0 means the cap is free.

Read the deltas, not the absolute numbers: they come from whatever weights you point
at, and a one-epoch smoke model has no accuracy to lose.

Everything here runs at predict's default conf=0.25, deliberately -- that is the
regime the board will use. Do not compare these figures to the "postprocess per
image" line printed during training: the validator scores at conf=0.001 to build the
full PR curve, which multiplies the number of boxes NMS has to sort through.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]


def render_polygons(label_path: Path, height: int, width: int) -> np.ndarray:
    """Rebuild a label file the way stage 08 does -- normalized points back to pixels."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(v) for v in line.split()[1:]]
        points = np.array(values, dtype=np.float64).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def union_mask(result) -> np.ndarray:
    """Fold every kept instance into one full-resolution mask."""
    mask = np.zeros(result.orig_shape, dtype=np.uint8)
    if result.masks is not None:
        for polygon in result.masks.xy:
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=HERE / "runs/smoke/weights/best.pt")
    parser.add_argument("--images", type=Path, default=HERE / "dataset/images/val")
    parser.add_argument("--labels", type=Path, default=HERE / "dataset/labels/val")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--max-det", default="300,32,16,8,4,1",
                        help="comma-separated caps to price out; 300 is the ultralytics default")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"missing {args.weights} -- run stages/06_train_yolo.py first", file=sys.stderr)
        return 1
    frames = sorted(args.images.glob("*.png"))[: args.count]
    if not frames:
        print(f"no images under {args.images}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    caps = [int(v) for v in args.max_det.split(",")]
    columns = {}
    for cap in caps:
        session = YOLO(str(args.weights))
        kwargs = {"max_det": cap}
        session.predict([str(frames[0])], imgsz=args.imgsz, device=args.device, verbose=False, **kwargs)
        start = time.perf_counter()
        results = session.predict([str(p) for p in frames], imgsz=args.imgsz, device=args.device,
                                  verbose=False, **kwargs)
        masks = [union_mask(r) for r in results]
        kept = [0 if r.boxes is None else len(r.boxes) for r in results]
        columns[cap] = ((time.perf_counter() - start) / len(frames) * 1000, masks, int(np.mean(kept)))

    loosest = max(caps)
    reference = columns[loosest][1]  # the loosest cap is the coverage the head can reach
    print(f"{len(frames)} frames | weights={args.weights.name} | imgsz={args.imgsz} device={args.device}")
    print(f"max_det={loosest} keeps a mean of {columns[loosest][2]} instances per sky frame\n")
    print(f"{'max_det':>8} {'wall':>9} {'inst':>9} {'IoU mean/min':>13}  {'vs-union mean/min':>11}")
    for cap, (ms, masks, n_instances) in columns.items():
        rows = []
        for frame, mask, full in zip(frames, masks, reference):
            label = args.labels / f"{frame.stem}.txt"
            ground = render_polygons(label, *mask.shape) if label.is_file() else np.zeros_like(mask)
            rows.append((iou(mask, ground), iou(mask, full)))
        scores, agree = np.array(rows).T
        tail = "" if cap == loosest else f"  {agree.mean():.4f}/{agree.min():.4f}"
        print(f"{cap:>8} {ms:>7.1f}ms {n_instances:>9} "
              f"{scores.mean():.4f}/{scores.min():.4f}   {tail}")

    print("\nHow to read this:")
    print("  wall/img   one frame end to end after a warmup call, so the first-call graph build")
    print("             is not counted; it contains preprocess, NPU/iGPU inference and all CPU post.")
    print("  inst/img   how many instances the head keeps -- the instance machinery is being used")
    print("             to paint one region, which is why post costs more than inference.")
    print("  IoU(label) mean/min agreement with the teacher-derived polygon. The ceiling is the teacher.")
    print("  vs-union   this cap's mask against the loosest cap's mask. ~1.0 means the cap is free;")
    print("             a drop means the sky really is split across instances and top-1 loses it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
