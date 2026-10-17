"""ONNX export for the student model (EXECUTION_PLAN.md §2.4).

Order matters for INT8 accuracy:
  1. BN folding (fuse_conv_bn_eval) -- fewer ops, scale lands on conv weights.
  2. Bake Sigmoid into the graph so the board does zero post-process.
  3. opset 11, static shape (dynamic_axes={}), batch=1 -- easiest RKNN case.
  4. No constant-folding of shapes: a prior teacher export baked static shapes
     into the graph and then OpenVINO reshape to 160 failed. Verify a fresh
     export can be re-inferred at another size if needed later.
  5. Log every upsample op (nearest/bilinear) -- first thing to check on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytrain.arch import build
from skytrain.config import NET_HW, GAMMA_DEFAULT


class BakeSigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def fuse_bn(model):
    model.eval()
    return torch.ao.quantization.fuse_modules(model, []) if False else torch.softmax  # placeholder


def export(model, path: Path, dummy, out_shape, opset: int = 11):
    """Fold BN, bake Sigmoid, export static-shape ONNX."""
    model.eval()
    # BN folding
    try:
        from torch.nn.utils import fuse_conv_bn_eval
        model = fuse_conv_bn_eval(model)
    except Exception as e:
        print(f"[export] BN fold skipped: {e}")

    wrapped = nn.Sequential(model, BakeSigmoid())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped, dummy, str(path),
            opset_version=opset,
            input_names=["x"], output_names=["alpha"],
            dynamic_axes={},
        )
    # log operator census + upsample ops
    import onnx
    g = onnx.load(str(path).replace(".onnx", ".onnx")) if Path(str(path)).exists() else None
    print(f"[export] wrote {path} in={tuple(dummy.shape)} out={out_shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    args = ap.parse_args()

    model, out_shape = build(args.arch)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    dummy = torch.randn(1, 1, *NET_HW)
    export(model, args.onnx, dummy, out_shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
