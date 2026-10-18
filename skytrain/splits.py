"""Block-based dataset split.

CRITICAL: split by contiguous 40-frame BLOCKS, never by random frame. Handheld
sweep frames are near-duplicates; random split puts validation frames next to
training frames and inflates every metric into meaninglessness.

Rule: per domain, sort by frame number, 40 frames per block, block index
i % 10 == 9 goes to val. Three domains split independently so all three appear
in val.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import SKY_ROOT, EXCLUDED

_BLOCK = 40
_VAL_EVERY = 10


def parse_stem(stem: str):
    """Parse '1_000001' -> (domain=1, frame_no=1)."""
    m = re.match(r"(\d+)_(\d+)", stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_splits(input_root: Path, out_csv: Path,
                 block: int = _BLOCK, val_every: int = _VAL_EVERY) -> dict:
    input_root = Path(input_root)
    by_domain: dict[int, list[tuple[int, str]]] = {}
    for subset in ("train", "val"):
        img_dir = input_root / "dataset" / "images" / subset
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.png")):
            if p.stem in EXCLUDED:
                continue
            parsed = parse_stem(p.stem)
            if parsed is None:
                continue
            by_domain.setdefault(parsed[0], []).append((parsed[1], p.stem))

    rows = []
    summary = {"train": 0, "val": 0}
    for dom, items in by_domain.items():
        items.sort()
        for i, (fno, stem) in enumerate(items):
            block_idx = i // block
            split = "val" if (block_idx % val_every == val_every - 1) else "train"
            rows.append((dom, block_idx, stem, split))
            summary[split] += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "block_idx", "frame_stem", "split"])
        w.writerows(rows)
    return summary


def read_splits(csv_path: Path):
    """Return (train_frames, val_frames) lists of frame stems."""
    csv_path = Path(csv_path)
    train, val = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            (train if row["split"] == "train" else val).append(row["frame_stem"])
    return train, val


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=SKY_ROOT)
    ap.add_argument("--out", type=Path,
                    default=SKY_ROOT.parent / "output" / "split" / "blocks.csv")
    args = ap.parse_args()
    summary = build_splits(args.input, args.out)
    print(f"splits written -> {args.out}")
    print(f"  train blocks/frames: {summary['train']}")
    print(f"  val   blocks/frames: {summary['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
