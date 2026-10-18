"""Evaluate a trained student checkpoint (EXECUTION_PLAN.md §6.4 on-PC part).

Reports per-frame leakage / erosion / F1 after bilinear upsample to 640x480 and
writes a CSV. This is the validation gate used by `skytrain.run eval`.

Run:
  python tools/eval_checkpoint.py --arch C2 --ckpt runs/C2_g1.3/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytrain.engine import eval_checkpoint
from skytrain.config import SKY_ROOT, GAMMA_DEFAULT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--csv", type=Path,
                    default=SKY_ROOT.parent / "output" / "split" / "blocks.csv")
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None,
                    help="per-frame CSV output (default: runs/eval_<arch>.csv)")
    args = ap.parse_args()

    if args.out is None:
        args.out = SKY_ROOT.parent / "runs" / f"eval_{args.arch}.csv"

    agg = eval_checkpoint(args.arch, args.ckpt, args.csv, args.gamma,
                          args.device, args.out)
    # exit non-zero if the hard leakage gate fails (Phase 5 gate)
    return 1 if agg["leakage_mean"] > 0.01 else 0


if __name__ == "__main__":
    raise SystemExit(main())
