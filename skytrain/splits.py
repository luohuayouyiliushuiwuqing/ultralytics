"""Block-based dataset split.

CRITICAL: split by contiguous BLOCKS of frames, never by random frame. Handheld
sweep frames are near-duplicates; a random split puts validation frames next to
training frames and inflates every metric into meaninglessness.

Rule: per domain, sort by frame number, `block` frames per block, blocks where
`block_idx % val_every == val_every - 1` go to val. Domains are split
independently so every domain appears in val.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import SKY_ROOT, BLOCKS_CSV, EXCLUDED, BLOCK, VAL_EVERY, IMAGE_EXTS


def parse_stem(stem: str):
    """Parse a frame stem into (domain, sortable frame number).

    Accepts both the legacy numeric form and prefixed real-world names:
        '1_000001'    -> ('1', 1)
        'd3_1015_001' -> ('d3', 1015001)
        'demo_960'    -> ('demo', 960)

    Returns None when the stem carries no usable frame number.
    """
    match = re.match(r"([A-Za-z0-9]+)_(.+)$", stem)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(2))
    if not digits:
        return None
    return match.group(1), int(digits)


def iter_frames(images_root: Path):
    """Yield frame paths under `images_root`.

    Handles the prepared layout `images_root/dataset/images/{train,val}` as well as
    a plain flat directory of frames (`images_root/images` or `images_root` itself).
    """
    root = Path(images_root)
    nested = root / "dataset" / "images"
    if (nested / "train").is_dir() or (nested / "val").is_dir():
        for subset in ("train", "val"):
            for path in sorted((nested / subset).glob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    yield path
        return
    flat = root / "images" if (root / "images").is_dir() else root
    for path in sorted(flat.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_splits(input_root: Path, out_csv: Path,
                 block: int = BLOCK, val_every: int = VAL_EVERY) -> dict:
    """Write a blocks.csv mapping every frame stem to train/val."""
    by_domain: dict[str, list[tuple[int, str]]] = {}
    for path in iter_frames(input_root):
        if path.stem in EXCLUDED:
            continue
        parsed = parse_stem(path.stem)
        if parsed is None:
            continue
        by_domain.setdefault(parsed[0], []).append((parsed[1], path.stem))

    rows = []
    summary = {"train": 0, "val": 0}
    for dom, items in by_domain.items():
        items.sort()
        for i, (_, stem) in enumerate(items):
            block_idx = i // block
            split = "val" if (block_idx % val_every == val_every - 1) else "train"
            rows.append((dom, block_idx, stem, split))
            summary[split] += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "block_idx", "frame_stem", "split"])
        writer.writerows(rows)
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
    ap.add_argument("--out", type=Path, default=BLOCKS_CSV)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    args = ap.parse_args()
    summary = build_splits(args.input, args.out, args.block, args.val_every)
    total = summary["train"] + summary["val"]
    if not total:
        print(f"no usable frames under {args.input} -- check --input", file=sys.stderr)
        return 1
    print(f"splits written -> {args.out}")
    print(f"  train: {summary['train']}   val: {summary['val']}   "
          f"({summary['val'] / total:.1%} validation)")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
