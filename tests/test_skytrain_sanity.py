"""Phase 0 sanity suite (EXECUTION_PLAN.md §6.1, layer 1: implementation correctness).

Run:  python -m pytest tests/test_skytrain_sanity.py -v
Each test maps to a gate in the plan. All are second-scale, no real data needed.
"""

from __future__ import annotations

import numpy as np
import torch

from skytrain.arch import build, SkyHead
from skytrain.data import YSkyDataset
from skytrain.config import NET_HW


def test_output_shape_c2_c3():
    for name, expected in [("C2", (1, 1, 144, 192)), ("C3", (1, 1, 144, 192))]:
        model, _ = build(name)
        x = torch.randn(1, 1, *NET_HW)
        out = model(x)
        assert tuple(out.shape) == expected, f"{name}: {tuple(out.shape)} != {expected}"


def test_output_shape_c1_skyhead():
    # C1 = YOLOv8n p3/p4/p5 features; verify SkyHead alone gives 72x96 at stride 4.
    p3 = torch.randn(1, 64, 288 // 8, 384 // 8)    # stride 8
    p4 = torch.randn(1, 128, 288 // 16, 384 // 16)
    p5 = torch.randn(1, 256, 288 // 32, 384 // 32)
    out = SkyHead()(p3, p4, p5)
    assert tuple(out.shape) == (1, 1, 72, 96), tuple(out.shape)


def test_stride_divisibility():
    # 4:3 zero-pad legal grid must divide by max stride 32.
    assert 288 % 32 == 0
    assert 384 % 32 == 0
    # the illegal 320x240 grid must fail (documented in the plan).
    assert 240 % 32 != 0


def test_normalization_roundtrip():
    # (Y/255 - 0.5)/0.5 at Y=0/127.5/255 -> -1/0/+1
    for Y, exp in [(0.0, -1.0), (127.5, 0.0), (255.0, 1.0)]:
        got = (Y / 255.0 - 0.5) / 0.5
        assert abs(got - exp) < 1e-6, (Y, got, exp)


def test_gamma_monotonic():
    # gamma=1 is identity; gamma>1 pulls sky confidence down (the bias we want);
    # larger gamma pulls further down; 0 and 1 are fixed points.
    assert (0.5 ** 1.0) == 0.5
    assert (0.5 ** 1.3) < 0.5
    assert (0.5 ** 2.0) < (0.5 ** 1.3)
    assert (0.0 ** 1.3) == 0.0
    assert (1.0 ** 1.3) == 1.0


def test_sigmoid_monotonic():
    logits = torch.linspace(-5, 5, 50)
    alpha = torch.sigmoid(logits)
    assert torch.all(alpha[1:] >= alpha[:-1])


def test_geo_alignment():
    H, W = NET_HW
    p = np.tile(np.linspace(0, 1, W).astype(np.float32), (H, 1))
    g = (p * 255).astype(np.float32)
    image = np.stack([g, g, g], -1)
    ds = YSkyDataset([], training=True, gamma=1.0, limited_jitter=False)
    for _ in range(100):
        y_w, p_w = ds._transform(image, p)
        err = float(np.mean(np.abs(y_w.astype(np.float32) - p_w * 255.0)))
        assert err < 1e-3, f"geo misalign {err}"


def test_forward_backward_runs():
    """One batch forward+backward must not error (Phase 0 gate)."""
    from skytrain.losses import SkyLoss
    model, _ = build("C2")
    x = torch.randn(2, 1, *NET_HW)
    t = torch.rand(2, 1, 144, 192)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = SkyLoss(gamma=1.3)
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(x), t)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).all()
