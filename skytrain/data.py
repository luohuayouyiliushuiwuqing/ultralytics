"""Dataset for v4 sky-mask distillation.

Contract baked into code (not memory), per EXECUTION_PLAN.md §6.1:
  * Input is the Y (luma) plane -> single channel. Rec.601 weights match the ISP.
  * Limited/full range jitter (50%) eats the unconfirmed NV12 range unknown.
  * Geometry augmentation (hflip / +/-3deg / 0.9-1.1 scale) applies the SAME affine
    to image-Y and teacher-prob. Mismatch won't crash; it silently trains garbage,
    so this property is unit-tested (tools/sanity_geo.py, tests/).
  * Probability map is resampled with INTER_LINEAR, NEVER nearest.
  * Label is downsampled to output size with bilinear so the loss domain == output
    domain (saves 4x memory vs training full-res then squeezing).

__getitem__ returns (x, t, p):
  x: (1, H, W) float32 in [-1,1]   -- network input (Y plane)
  t: (1, oh, ow) float32           -- soft target p**gamma
  p: (1, oh, ow) float32           -- raw teacher prob (for metrics)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NET_HW, GAMMA_DEFAULT, IMAGE_EXTS


class YSkyDataset(Dataset):
    def __init__(self, frames, net_hw=NET_HW, training=True, gamma=GAMMA_DEFAULT,
                 prob_dir=None, img_dir=None, out_hw=None, limited_jitter=True):
        self.frames = list(frames)
        self.net_hw = tuple(net_hw)                      # (H, W)
        self.training = training
        self.gamma = gamma
        self.prob_dir = Path(prob_dir) if prob_dir is not None else None
        self.img_dir = Path(img_dir) if img_dir is not None else None
        self.out_hw = tuple(out_hw) if out_hw else (net_hw[0] // 4, net_hw[1] // 4)
        self.limited_jitter = limited_jitter

    # ---- Y plane -------------------------------------------------------------
    def y_plane(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)         # Rec.601, matches ISP Y
        if self.training and self.limited_jitter and random.random() < 0.5:
            # limited (16-235) -> full (0-255) range jitter
            y = (y.astype(np.float32) - 16) * (255.0 / 219.0)
            y = np.clip(y, 0, 255).astype(np.uint8)
        return y

    # ---- geometry (shared between image and prob) ----------------------------
    def _geo(self, h, w):
        """Return (flip, M) used identically for image-Y and prob."""
        flip = self.training and random.random() < 0.5
        if self.training:
            ang = random.uniform(-3, 3)
            s = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        else:
            M = np.eye(2, 3, dtype=np.float32)
        return flip, M

    @staticmethod
    def _warp(img: np.ndarray, flip: bool, M: np.ndarray) -> np.ndarray:
        if flip:
            img = cv2.flip(img, 1)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _transform(self, image_bgr, prob):
        """Apply the SAME affine to Y(image) and prob. Tested for alignment."""
        flip, M = self._geo(*image_bgr.shape[:2])
        y = self._warp(self.y_plane(image_bgr), flip, M)
        p = self._warp(prob, flip, M)
        return y, p

    # ---- item ----------------------------------------------------------------
    def __getitem__(self, i):
        stem = self.frames[i]
        image = None
        for sub in ("train", "val"):                      # v4 re-splits merged v3 train/val
            for ext in IMAGE_EXTS:                        # frames ship as .jpg or .png
                cand = self.img_dir / sub / f"{stem}{ext}"
                if cand.is_file():
                    image = cv2.imread(str(cand))
                    break
            if image is not None:
                break
        dom = stem.split("_")[0]                          # '1_000001' -> '1'
        prob = cv2.imread(str(self.prob_dir / dom / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or prob is None:
            raise FileNotFoundError(f"missing pair for {stem}")
        prob = prob.astype(np.float32) / 65535.0

        y, p = self._transform(image, prob)

        y = cv2.resize(y, (self.net_hw[1], self.net_hw[0]))           # (W,H) order
        p = cv2.resize(p, (self.out_hw[1], self.out_hw[0]),
                       interpolation=cv2.INTER_LINEAR)                  # never nearest

        x = torch.from_numpy((y.astype(np.float32) / 127.5 - 1.0))[None]
        t = torch.from_numpy(p[None] ** self.gamma)
        return x, t, torch.from_numpy(p[None])

    def __len__(self):
        return len(self.frames)
