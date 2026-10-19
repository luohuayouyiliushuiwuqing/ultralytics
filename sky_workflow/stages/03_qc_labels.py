"""Stage 3 -- QC the labels without ground truth.

    python sky_workflow/stages/03_qc_labels.py sky_workflow/work/mask_clean --images input

There is no manual ground truth, so label quality has to be argued from two
independent properties the data happens to have:

* **Temporal smoothness.** The frames are handheld sweeps, so the sky fraction
  moves gradually. A neighbour-to-neighbour jump far outside the normal range
  means one of the two masks disagrees with the scene, not that the camera moved.
* **Boundary plausibility.** Rendered as a montage so a human can look at the
  suspects directly. This is the check that actually decides anything.

Writes a per-frame CSV (ratio plus the jump into and out of each frame) and a
sheet of the worst offenders. Exclusions are then a *human* decision recorded in
work/excluded.txt -- one path per line, relative to the image root -- which
stages 4 and 5 read if it exists. Auto-excluding on jump size would throw away
fast pans along with real errors.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sky_only import EXTENSIONS  # noqa: E402


def render(mask: np.ndarray, colour) -> np.ndarray:
    """Tint the sky region so boundary errors are visible at thumbnail size."""
    rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rgb[mask > 127] = (rgb[mask > 127] * 0.45 + np.array(colour) * 0.55).astype(np.uint8)
    return rgb


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mask_root", type=Path, nargs="?", default=here / "work/mask_clean")
    parser.add_argument("--images", type=Path, default=Path("input"), help="for the montage only")
    parser.add_argument("--report", type=Path, default=here / "work/qc_frames.csv")
    parser.add_argument("--sheet", type=Path, default=here / "work/qc_suspects.png")
    parser.add_argument("--worst", type=int, default=12, help="how many suspects to montage")
    parser.add_argument("--jump-p99", type=float, default=None,
                        help="flag frames whose neighbour jump exceeds this absolute ratio delta")
    args = parser.parse_args()

    # Group by sequence directory: jumps only mean something between neighbours.
    groups = {}
    for path in sorted(args.mask_root.rglob("*.png")):
        groups.setdefault(str(path.parent.name), []).append(path)

    rows, suspect = [], []
    for name, paths in sorted(groups.items()):
        ratios = []
        for path in paths:
            mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            ratios.append(None if mask is None else float((mask > 127).mean()))
        for index, path in enumerate(paths):
            ratio = ratios[index]
            if ratio is None:
                continue
            before = abs(ratio - ratios[index - 1]) if index and ratios[index - 1] is not None else 0.0
            after = abs(ratio - ratios[index + 1]) if index + 1 < len(ratios) and ratios[index + 1] is not None else 0.0
            jump = max(before, after)
            # as_posix so these lines can be pasted straight into work/excluded.txt.
            rows.append((path.relative_to(args.mask_root).as_posix(), ratio, jump))
            suspect.append((jump, path.relative_to(args.mask_root).as_posix()))

    if not rows:
        print("no masks found", file=sys.stderr)
        return 1

    jumps = np.array([r[2] for r in rows])
    cutoff = args.jump_p99 if args.jump_p99 is not None else float(np.percentile(jumps, 99))
    flagged = [rel for jump, rel in sorted(suspect, reverse=True) if jump >= cutoff]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("mask,sky_ratio,max_neighbour_jump\n" +
                           "".join(f"{r[0]},{r[1]:.6f},{r[2]:.6f}\n" for r in rows), encoding="utf-8")

    print(f"{len(rows)} masks in {len(groups)} sequences -> {args.report}")
    print(f"  neighbour jump: median={np.median(jumps):.3%} p90={np.percentile(jumps,90):.3%} "
          f"p99={np.percentile(jumps,99):.3%} max={jumps.max():.3%}")
    print(f"  flagged at jump >= {cutoff:.3%}: {len(flagged)} frames")
    for rel in flagged[: args.worst]:
        print(f"    {rel}")

    # Montage the suspects so the decision is made by looking, not by the number.
    tiles = []
    for rel in flagged[: args.worst]:
        image = cv2.imdecode(np.fromfile(str(args.images / rel), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(args.mask_root / rel), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        tile = cv2.addWeighted(image, 0.55, render(mask, (0, 255, 0)), 0.45, 0)
        cv2.putText(tile, Path(rel).stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, Path(rel).stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (320, 240)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((240, 320, 3), 40, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        args.sheet.write_bytes(cv2.imencode(".png", sheet)[1].tobytes())
        print(f"  suspects montaged -> {args.sheet}")
    print("\nReview the sheet, then record genuine label errors one path per line in work/excluded.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
