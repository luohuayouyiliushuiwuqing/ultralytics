"""Materialise the dataset layout the student trainer expects -- one command.

    python -m skytrain.prepare

Starting from a flat directory of frames (`datasets/images`) plus the teacher
probability maps, it produces everything under `datasets/`:

    datasets/teacher_prob16/<domain>/<stem>.png       16-bit soft labels
    datasets/skytrain_data/blocks.csv                 block-based train/val split
    datasets/skytrain_data/dataset/images/{train,val}/  frames, linked not copied

Frames are symlinked rather than copied, so re-running is cheap and the dataset can
never drift from the source tree. Where symlinks are unavailable the frame is copied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import (IMAGE_DIR, PROB_NPY_DIR, PROB_DIR, SKY_ROOT, BLOCKS_CSV, BLOCK, VAL_EVERY)
from .labels import convert_tree
from .splits import build_splits, iter_frames, read_splits


def link_or_copy(source: Path, target: Path) -> None:
    """Point `target` at `source`, preferring a symlink and falling back to a copy.

    Idempotent: an existing link that already resolves to `source` is left alone, so
    re-running never deletes anything and only stale entries are rebuilt.
    """
    source = source.resolve()
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source:
                return
        except OSError:  # broken link
            pass
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:  # no symlink privilege (some Windows setups, exotic filesystems)
        shutil.copy2(source, target)


def ensure_probs(npy_dir: Path, out_dir: Path) -> int:
    """Convert teacher .npy probability maps to 16-bit PNGs; returns the count."""
    npy_dir = Path(npy_dir)
    if not npy_dir.is_dir():
        print(f"no probability dir {npy_dir} -- run the teacher first", file=sys.stderr)
        return 0
    count = convert_tree(npy_dir, Path(out_dir))
    print(f"[prepare] {count} probability maps -> {out_dir}")
    return count


def materialise(image_dir: Path, data_root: Path, blocks_csv: Path,
                block: int = BLOCK, val_every: int = VAL_EVERY) -> int:
    """Write blocks.csv and link every frame into dataset/images/{train,val}."""
    summary = build_splits(image_dir, blocks_csv, block, val_every)
    total = summary["train"] + summary["val"]
    if not total:
        print(f"no usable frames under {image_dir}", file=sys.stderr)
        return 0

    sources = {path.stem: path for path in iter_frames(image_dir)}
    train, val = read_splits(blocks_csv)
    images_root = Path(data_root) / "dataset" / "images"
    made = 0
    for split, stems in (("train", train), ("val", val)):
        target_dir = images_root / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            source = sources.get(stem)
            if source is None:
                print(f"  no source frame for {stem}")
                continue
            link_or_copy(source, target_dir / (stem + source.suffix.lower()))
            made += 1
    print(f"[prepare] {made} frames -> {images_root} "
          f"(train={summary['train']} val={summary['val']}, "
          f"{summary['val'] / total:.1%} validation)")
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=IMAGE_DIR, help="flat source frames")
    ap.add_argument("--prob-npy", type=Path, default=PROB_NPY_DIR, help="teacher .npy maps")
    ap.add_argument("--prob-out", type=Path, default=PROB_DIR, help="16-bit PNG output")
    ap.add_argument("--data-root", type=Path, default=SKY_ROOT, help="student dataset root")
    ap.add_argument("--blocks-csv", type=Path, default=BLOCKS_CSV)
    ap.add_argument("--block", type=int, default=BLOCK, help="contiguous frames per split atom")
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    ap.add_argument("--skip-probs", action="store_true", help="reuse existing 16-bit labels")
    args = ap.parse_args()

    if not args.skip_probs:
        if not ensure_probs(args.prob_npy, args.prob_out):
            return 1
    made = materialise(args.images, args.data_root, args.blocks_csv, args.block, args.val_every)
    if not made:
        return 1
    print(f"[prepare] ready: --data-root {args.data_root} "
          f"--blocks-csv {args.blocks_csv} --prob-dir {args.prob_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
