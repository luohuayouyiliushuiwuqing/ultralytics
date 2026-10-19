"""Model architectures for the three v4 candidates (see EXECUTION_PLAN.md §3.2).

C1: YOLOv8n backbone+neck (layers 0-21, COCO-pretrained) + a custom SkyHead that
    emits a single-channel soft alpha at stride 4 (33K params). The backbone is
    loaded separately in engine.py; here SkyHead is the standalone head.
C2: tiny dense-conv U-Net (ReLU6) -- lowest RKNN risk.
C3: tiny depthwise U-Net (ReLU6) -- smallest, but depthwise efficiency on RK3588
    MAC array is an unknown (verified on board in Phase 4).

build(name) -> (model, output_shape) so the rest of the pipeline is arch-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(ci: int, co: int, k: int = 3):
    """YOLOv8-style Conv+BN+SiLU used by C1's SkyHead."""
    return [nn.Conv2d(ci, co, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(co), nn.SiLU()]


class SkyHead(nn.Module):
    """P3/P4/P5 (stride 8/16/32) -> single-channel soft alpha @ stride 4.

    With 1x1x288x384 input, output is (1,1,72,96), exact stride 4, ~33,025 params.
    """

    def __init__(self, ch=(64, 128, 256), hidden: int = 32):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in ch])
        self.fuse = nn.Sequential(*block(hidden, hidden))
        self.ref = nn.Sequential(*block(hidden, hidden))
        self.logit = nn.Conv2d(hidden, 1, 1)

    def forward(self, p3, p4, p5):
        size = p3.shape[-2:]
        x = (self.lat[0](p3)
             + F.interpolate(self.lat[1](p4), size, mode="nearest")
             + F.interpolate(self.lat[2](p5), size, mode="nearest"))
        x = self.ref(self.fuse(x))
        return self.logit(F.interpolate(x, scale_factor=2, mode="nearest"))


def blk(ci: int, co: int, s: int = 1):
    """Dense 3x3 conv block with ReLU6 (RKNN first-class op, naturally bounded)."""
    return [nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU6(inplace=True)]


def dw_blk(ci: int, co: int, s: int = 1):
    """Depthwise-separable block with ReLU6."""
    return [nn.Conv2d(ci, ci, 3, s, 1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU6(True),
            nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU6(True)]


class TinyNet(nn.Module):
    """3-level encoder + skip-connection decoder. No FPN, no attention.

    dw=True yields C3 (depthwise); dw=False yields C2 (dense).
    Input 1x1x288x384 -> output 1x1x144x192 (stride 2).
    """

    def __init__(self, ch=(16, 32, 64), dw: bool = False):
        super().__init__()
        mk = dw_blk if dw else blk
        self.e1 = nn.Sequential(*mk(1, ch[0], 2))
        self.e2 = nn.Sequential(*mk(ch[0], ch[1], 2))
        self.e3 = nn.Sequential(*mk(ch[1], ch[2], 2))
        self.b = nn.Sequential(*mk(ch[2], ch[2]))
        self.d2 = nn.Sequential(*mk(ch[2] + ch[1], ch[1]))
        self.d1 = nn.Sequential(*mk(ch[1] + ch[0], ch[0]))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def up(self, x, ref):
        return F.interpolate(x, ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        d = self.b(c)
        e = self.d2(torch.cat([self.up(d, b), b], 1))
        return self.out(self.d1(torch.cat([self.up(e, a), a], 1)))


def build(name: str):
    """Return (model, output_shape) for a candidate arch."""
    name = name.upper()
    if name == "C1":
        # SkyHead consumes (p3, p4, p5), not the input tensor, so returning it as a
        # standalone model makes engine's `model(x)` call fail with a TypeError. The
        # YOLOv8n backbone+neck that produces those features is not wired here, so
        # C1 fails loudly rather than crashing mid-training. Use SkyHead directly
        # (see tests/test_skytrain_sanity.py), or run C2/C3 end to end.
        raise NotImplementedError(
            "C1 needs the YOLOv8n backbone+neck to feed SkyHead (p3, p4, p5); it is not "
            "wired in this checkout. Use --arch C2 or --arch C3, or call SkyHead directly."
        )
    if name == "C2":
        return TinyNet(ch=(16, 32, 64), dw=False), (1, 1, 144, 192)
    if name == "C3":
        return TinyNet(ch=(32, 64, 128), dw=True), (1, 1, 144, 192)
    raise ValueError(f"unknown arch {name!r}; expected C1/C2/C3")
