"""Stage 4 -- binary masks to YOLO segmentation polygons.

    python sky_workflow/stages/04_masks_to_yolo.py --images input --masks sky_workflow/work/mask_clean -o sky_workflow/work/labels

YOLO seg format: one .txt per image, same stem, one object per line as
`class x1 y1 x2 y2 ... xn yn` with every coordinate normalised by the frame's own
width and height. Class 0 = sky. An empty .txt is a valid "no sky here" label, and
a frame whose sky covers everything degenerates to a four-point border polygon.

Three conversions lose information and all three are measured rather than assumed:

* **Holes.** YOLO polygons have no even-odd fill, so a ground island surrounded by
  sky cannot be expressed. Only outer contours are emitted and `hole_px` records
  how much area that silently claims as sky.
* **Point count.** approxPolyDP starts at --epsilon of the arc length and coarsens
  in 1.5x steps until the polygon fits --max-points. Long horizons are cheap; the
  grass-blade close-ups in domain 3 are where this bites.
* **Quantisation.** Coordinates are written to 6 decimals, well under a pixel at
  640x480, so this one is not worth worrying about.

If work/excluded.txt exists its paths are skipped -- that is where the stage 3
review records genuine teacher errors.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sky_only import EXTENSIONS  # noqa: E402


def load_excluded(path: Path) -> set:
    """Read the exclusion list as stems, so entries match whatever suffix they carry."""
    if not path.is_file():
        return set()
    names = {Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"excluded by {path}: {len(names)} frames")
    return names


def simplify(contour: np.ndarray, epsilon_ratio: float, max_points: int) -> np.ndarray:
    """Coarsen until the polygon fits the point budget, never below a triangle."""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon, True)
    while len(poly) > max_points and len(poly) > 3:
        epsilon *= 1.5
        poly = cv2.approxPolyDP(contour, epsilon, True)
    return poly.reshape(-1, 2).astype(np.float64)


def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float, max_points: int, min_area: int):
    """Returns (polygons in pixel coords, hole pixels dropped for lack of even-odd fill)."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return [], 0
    outer, holes_px, polygons = [], 0, []
    for index, contour in enumerate(contours):
        parent = hierarchy[0][index][3]
        if parent != -1:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area:
            continue
        outer.append(index)
        polygons.append(simplify(contour, epsilon_ratio, max_points))
        # Children of a retained outer are holes we cannot represent.
        child = hierarchy[0][index][2]
        while child != -1:
            if 0 <= child < len(contours):
                holes_px += int(round(abs(float(cv2.contourArea(contours[child])))))
            child = hierarchy[0][child][0]
    return polygons, holes_px


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("input"), help="only for reading frame sizes")
    parser.add_argument("--masks", type=Path, default=here / "work/mask_clean")
    parser.add_argument("-o", "--out", type=Path, default=here / "work/labels")
    parser.add_argument("--epsilon", type=float, default=0.0015, help="approxPolyDP start, as a fraction of arc length")
    parser.add_argument("--max-points", type=int, default=120)
    parser.add_argument("--min-area", type=int, default=200, help="ignore sky components smaller than this")
    parser.add_argument("--stats", type=Path, default=here / "work/polygon_stats.csv")
    parser.add_argument("--excluded", type=Path, default=None,
                        help="exclusion list written by stage 1 --review (default: sibling of --masks)")
    args = parser.parse_args()

    excluded = load_excluded(args.excluded or (args.masks.parent / "excluded.txt"))
    masks = sorted(p for p in args.masks.rglob("*.png") if p.is_file())
    if not masks:
        print(f"no masks under {args.masks}", file=sys.stderr)
        return 1

    rows, skipped, empty = [], 0, 0
    for mask_path in masks:
        rel = mask_path.relative_to(args.masks)
        # as_posix, not str(): excluded.txt is written with forward slashes and
        # str(Path) yields backslashes on Windows, which silently matches nothing.
        key = Path(rel).stem
        image_path = args.images / rel.with_suffix(".png")
        if not image_path.is_file():
            matches = [args.images / rel.parent / f"{rel.stem}{ext}" for ext in EXTENSIONS]
            image_path = next((m for m in matches if m.is_file()), image_path)
        if key in excluded:
            skipped += 1
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"  skip unreadable pair {rel}")
            continue
        height, width = image.shape[:2]

        polygons, holes_px = mask_to_polygons((mask > 127).astype(np.uint8), args.epsilon, args.max_points, args.min_area)
        lines = []
        for poly in polygons:
            clipped = np.clip(poly, 0.0, None)
            clipped[:, 0] = np.minimum(clipped[:, 0], width)
            clipped[:, 1] = np.minimum(clipped[:, 1], height)
            normalised = clipped / np.array([width, height], dtype=np.float64)
            flat = " ".join(f"{v:.6f}" for v in normalised.reshape(-1))
            lines.append(f"0 {flat}")
        target = args.out / rel.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        if not lines:
            empty += 1

        points = sum(len(p) for p in polygons)
        rows.append((key, len(polygons), points, holes_px, float((mask > 127).mean())))

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text("mask,polygons,points,hole_px,sky_ratio\n" +
                          "".join(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n" for r in rows), encoding="utf-8")

    if not rows:
        print("nothing converted", file=sys.stderr)
        return 1
    counts = np.array([r[1] for r in rows])
    points = np.array([r[2] for r in rows])
    holes = np.array([r[3] for r in rows])
    print(f"{len(rows)} label files -> {args.out}   skipped(excluded)={skipped}   empty={empty}")
    print(f"  polygons/frame: median={np.median(counts):.0f} max={counts.max()}  "
          f"(multi-component frames: {(counts > 1).sum()})")
    print(f"  points/frame:   median={np.median(points):.0f} p95={np.percentile(points,95):.0f} max={points.max()}")
    print(f"  hole pixels unrepresentable: total={holes.sum()}  frames affected={(holes > 0).sum()}  "
          f"worst={holes.max()} ({holes.max() / (640 * 480):.2%} of a frame)")
    print(f"  stats -> {args.stats}")
    print("\nRun 08_verify_dataset.py after 05: a polygon conversion can be quietly wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
