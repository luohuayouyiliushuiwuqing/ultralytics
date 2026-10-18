"""Stage 8 -- prove the polygon labels still say what the masks said.

    python sky_workflow/stages/08_verify_dataset.py

Every conversion in stage 4 is lossy in a way that looks fine until you measure
it: holes vanish, approxPolyDP shaves the grass blades, a clipped coordinate can
push a boundary inside by a pixel. A round-trip is the only check that catches all
of them at once -- re-render each .txt with fillPoly and IoU it against the mask
the label was generated from.

Reads dataset/manifest.csv so the flat dataset stem maps back to the source
mask path. Reports the IoU distribution per split, the worst frames, and writes a
sheet of them: frame | source mask | polygon render | disagreement.

The number that matters is the *tail*, not the mean. A mean of 0.99 with 30 frames
under 0.90 means those frames were rewritten, and they will quietly dominate the
loss because they are the hardest ones.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sky_only import EXTENSIONS  # noqa: E402


def render_polygons(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    text = label_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        values = [float(v) for v in line.split()[1:]]
        if len(values) < 6:
            continue
        points = np.array(values, dtype=np.float64).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=here / "dataset")
    parser.add_argument("--images", type=Path, default=Path("input"))
    parser.add_argument("--masks", type=Path, default=here / "work/mask_clean")
    parser.add_argument("--sheet", type=Path, default=here / "work/verify_worst.png")
    parser.add_argument("--worst", type=int, default=12)
    parser.add_argument("--accept", type=float, default=0.97, help="round-trip IoU floor for a usable label")
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.csv"
    if not manifest_path.is_file():
        print(f"{manifest_path} missing -- run stages/05_build_dataset.py first", file=sys.stderr)
        return 1
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    results, missing = [], 0
    for row in rows:
        image_path = args.dataset / "images" / row["split"] / f"{row['stem']}{Path(row['frame']).suffix}"
        label_path = args.dataset / "labels" / row["split"] / f"{row['stem']}.txt"
        mask_path = args.masks / row["frame"]
        if not image_path.is_file() or not label_path.is_file():
            missing += 1
            continue
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if mask is None or image is None:
            missing += 1
            continue
        height, width = image.shape[:2]
        source = mask > 127
        rendered = render_polygons(label_path, height, width) > 127
        union = np.count_nonzero(source | rendered)
        iou = float(np.count_nonzero(source & rendered) / union) if union else 1.0
        results.append((iou, row["split"], row["frame"], source, rendered, image))

    if not results:
        print("nothing verified", file=sys.stderr)
        return 1

    print(f"{len(results)} labels round-tripped against their source masks ({missing} unresolved)")
    for split in ("train", "val"):
        array = np.array([r[0] for r in results if r[1] == split])
        if not array.size:
            continue
        print(f"  {split:<5} n={array.size:<5} IoU mean={array.mean():.4f} median={np.median(array):.4f} "
              f"p05={np.percentile(array,5):.4f} min={array.min():.4f}   below {args.accept}: "
              f"{(array < args.accept).sum()} ({(array < args.accept).mean():.1%})")

    all_array = np.array([r[0] for r in results])
    verdict = "OK" if all_array.min() >= args.accept else "REVIEW"
    print(f"\n  overall min={all_array.min():.4f}  -> {verdict} "
          f"(floor {args.accept}; lower it or raise --epsilon/--max-points in stage 4)")

    results.sort(key=lambda r: r[0])
    tiles = []
    for iou, split, rel, source, rendered, image in results[: args.worst]:
        diff = (source ^ rendered).astype(np.uint8) * 255
        quad = np.hstack([image,
                          cv2.cvtColor(source.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                          cv2.cvtColor(rendered.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                          cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)])
        quad = cv2.resize(quad, (int(quad.shape[1] * 0.5), int(quad.shape[0] * 0.5)))
        cv2.putText(quad, f"{Path(rel).stem} IoU={iou:.4f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(quad)
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    args.sheet.write_bytes(cv2.imencode(".png", np.vstack(tiles))[1].tobytes())
    print(f"  worst {len(tiles)} frames (frame | source mask | polygon render | xor) -> {args.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
