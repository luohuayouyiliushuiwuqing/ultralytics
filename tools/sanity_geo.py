"""Phase 0 gate #1: geometry augmentation must align image-Y and teacher-prob.

If the affine applied to the image and to the probability map ever diverge, the
model does not crash and every metric still "looks fine" -- it just silently
learns from mislabeled data. This script proves the SAME affine is used for both.

Proof: build a grayscale image whose Y plane equals the prob map (prob*255). After
_transform, the warped Y must equal the warped prob*255 pointwise (same op, same
matrix). We sample many random affines (hflip / +/-3deg / 0.9-1.1 scale).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from skytrain.data import YSkyDataset


def main() -> int:
    H, W = 288, 384
    p = np.tile(np.linspace(0, 1, W).astype(np.float32), (H, 1))   # gradient along width
    g = (p * 255).astype(np.float32)
    image = np.stack([g, g, g], -1)                                # BGR grayscale, float32

    ds = YSkyDataset([], training=True, gamma=1.0, limited_jitter=False)
    max_err = 0.0
    for _ in range(200):
        y_w, p_w = ds._transform(image, p)
        err = float(np.mean(np.abs(y_w.astype(np.float32) - p_w * 255.0)))
        max_err = max(max_err, err)
        if err > 1e-3:
            print(f"MISALIGN detected: err={err}")
            return 1
    print(f"[geo-align] OK over 200 random affines, max_err={max_err:.2e} (<= 1e-3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
