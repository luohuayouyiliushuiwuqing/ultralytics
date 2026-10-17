"""One-command reproduction entry (EXECUTION_PLAN.md §4 Phase 6).

Runs the v4 pipeline end-to-end for one candidate arch:
  labels -> splits -> (Phase 1 gamma sweep) -> train -> export.

Until Phases 1-6 are sequenced, this just chains labels+splits+smoke and prints
the next manual step. Full training is invoked via skytrain.engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytrain import labels, splits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, help="C1 | C2 | C3")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--gamma", type=float, default=None,
                    help="omit for Phase 1 sweep; set after Phase 1 freeze")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--skip-splits", action="store_true")
    args = ap.parse_args()

    root = args.data_root
    if root is None:
        from skytrain.config import SKY_ROOT
        root = SKY_ROOT

    if not args.skip_labels:
        labels.main.__wrapped__ if False else None
        rc = labels.main() if False else _run_labels(root)
        if rc:
            return rc
    if not args.skip_splits:
        rc = _run_splits(root)
        if rc:
            return rc

    print("\nNext: run gamma sweep (Phase 1), e.g.")
    print(f"  python -m skytrain.engine --arch {args.arch} "
          f"--sweep-gamma 1.0,1.3,1.6,2.0 --epochs 40 --smoke")
    print("Then freeze gamma and run full training (Phase 2).")
    return 0


def _run_labels(root):
    from skytrain.labels import convert_tree
    from skytrain.config import SKY_ROOT
    n = convert_tree(SKY_ROOT / "work" / "prob",
                     Path(SKY_ROOT).parent / "output" / "teacher_prob16")
    print(f"converted {n} probability maps")
    return 0


def _run_splits(root):
    from skytrain.splits import build_splits
    from skytrain.config import SKY_ROOT
    summary = build_splits(SKY_ROOT, Path(SKY_ROOT).parent / "output" / "split" / "blocks.csv")
    print(f"splits: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
