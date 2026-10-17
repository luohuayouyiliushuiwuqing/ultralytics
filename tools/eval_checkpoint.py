"""Evaluate a trained student checkpoint (EXECUTION_PLAN.md §6.4 on-PC part).

Reports per-frame leakage / erosion / F1 after bilinear upsample to 640x480, and
optionally shows a triple view (image / teacher-overlay / student-overlay).

Run (after Phase 2 produces runs/<arch>/best.pt):
  python tools/eval_checkpoint.py --arch C2 --ckpt runs/C2/best.pt --csv <val.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytrain.arch import build
from skytrain.data import YSkyDataset
from skytrain.losses import leakage_erosion
from skytrain.splits import read_splits
from skytrain.config import SKY_ROOT, GAMMA_DEFAULT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--csv", type=Path,
                    default=SKY_ROOT.parent / "output" / "split" / "blocks.csv")
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    _, val_frames = read_splits(args.csv)
    model, _ = build(args.arch)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(args.device).eval()

    ds = YSkyDataset(val_frames, training=False, gamma=args.gamma,
                     prob_dir=SKY_ROOT.parent / "output" / "teacher_prob16",
                     img_dir=SKY_ROOT / "dataset" / "images" / "train")
    # NOTE: val frames live under images/{train,val} in v3 layout; use whatever
    # split column resolved. This is a Phase 6 refinement; Phase 0 leaves it stubbed.
    print(f"[eval] {len(ds)} val frames; checkpoint {args.ckpt}")
    print("[eval] full per-frame leakage/erosion wired in Phase 6 (needs trained ckpt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
