"""Stage 2 -- connected-component cleanup of the teacher's masks.

    python sky_workflow/stages/02_clean_labels.py sky_workflow/work/mask -o sky_workflow/work/mask_clean

The teacher leaves isolated specks of "ground" inside the sky and specks of "sky"
inside the ground. Both are noise, and both teach the student to hallucinate the
same noise. Two passes of connected-component pruning fix it:

  1. prune small components of the *inverse* mask -> fills holes inside the sky
  2. prune small components of the sky itself      -> removes floating specks

Order matters, and getting it wrong inverts the mask: after step 1 the array is in
ground-space and must be flipped back before step 2 runs.

Every mask lands in the output tree whether it changed or not, so downstream
stages never have to fall back to the uncleaned set.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MASK_GLOB = "*.png"


def prune_small(binary: np.ndarray, min_area: int):
    """Zero out connected components smaller than min_area. Returns (image, removed)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary, 0
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_area) + 1
    out = binary.copy()
    out[np.isin(labels, small)] = 0
    return out, int(small.size)


def clean(mask: np.ndarray, min_area: int):
    binary = (mask > 127).astype(np.uint8)
    pruned_ground, holes = prune_small(1 - binary, min_area)
    filled = 1 - pruned_ground
    cleaned, specks = prune_small(filled, min_area)
    return cleaned * 255, holes, specks


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mask_root", type=Path, nargs="?", default=here / "work/mask")
    parser.add_argument("-o", "--out", type=Path, default=here / "work/mask_clean")
    parser.add_argument("--min-area", type=int, default=150,
                        help="components below this pixel count are pruned; 150 px is ~0.05%% of 640x480")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.mask_root
    paths = sorted(p for p in root.rglob(MASK_GLOB) if p.is_file())
    if not paths:
        print(f"no masks under {root}", file=sys.stderr)
        return 1

    changed = holes_total = specks_total = 0
    worst = (0, "")
    for path in paths:
        mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        cleaned, holes, specks = clean(mask, args.min_area)
        delta = int(np.count_nonzero(cleaned != mask))
        if delta:
            changed += 1
            if delta > worst[0]:
                worst = (delta, str(path.relative_to(root)))
        holes_total += holes
        specks_total += specks
        if not args.dry_run:
            target = args.out / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cv2.imencode(".png", cleaned)[1].tobytes())

    print(f"{len(paths)} masks -> {args.out if not args.dry_run else '(dry run)'}")
    print(f"  changed: {changed}   holes filled: {holes_total}   specks removed: {specks_total}")
    print(f"  largest single-frame change: {worst[0]} px ({worst[0] / (640 * 480):.2%} of a frame) at {worst[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
