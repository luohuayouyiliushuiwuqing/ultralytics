"""Loss and metrics for sky-mask distillation.

Main loss (soft label distillation):
    L = BCE(logit, p**gamma) + 0.5 * soft_dice(sigmoid(logit), p)
where p is the teacher probability and p**gamma (gamma ~ 1.3) is the only bias
hyperparameter: it pushes the sky confidence down so the model errs on the side
of "carving more sky", matching the product rule "better to eat sky than leave
ground".

Metrics are DIRECTIONAL (not symmetric IoU):
    leakage = pred_sky ∩ true_ground / true_ground   (ground survives in sky -> bad)
    erosion = pred_ground ∩ true_sky  / true_sky      (sky eaten -> acceptable, minimize)
Both must be computed AFTER bilinear upsample to 640x480 (resolution loss counts).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_bce(logit: torch.Tensor, p: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    target = p ** gamma
    return F.binary_cross_entropy_with_logits(logit, target, reduction="mean")


def soft_dice(pred: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (pred * p).sum()
    den = pred.sum() + p.sum()
    return 1.0 - (2.0 * num + eps) / (den + eps)


class SkyLoss(nn.Module):
    def __init__(self, gamma: float = 1.3, bce_w: float = 1.0, dice_w: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logit: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logit)
        return self.bce_w * soft_bce(logit, p, self.gamma) + self.dice_w * soft_dice(pred, p)


def leakage_erosion(pred_alpha: torch.Tensor, true_alpha: torch.Tensor, thr: float = 0.5):
    """Compute leakage / erosion between two alpha maps at the same resolution.

    pred_alpha, true_alpha: float tensors, same shape, values in [0,1].
    """
    pred_sky = pred_alpha >= thr
    pred_ground = pred_alpha < thr
    true_sky = true_alpha >= thr
    true_ground = true_alpha < thr
    leak = (pred_sky & true_ground).float().sum() / true_ground.float().sum().clamp(min=1.0)
    eros = (pred_ground & true_sky).float().sum() / true_sky.float().sum().clamp(min=1.0)
    return leak, eros
