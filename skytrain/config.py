"""Shared constants for the v4 sky-mask student training pipeline.

The image / probability-map assets live in the sibling repo checkout
(`ultralytics/sky_workflow`), referenced by absolute path so the code can run
either from the worktree or from the original checkout.
"""

from pathlib import Path

# Root of the sky_workflow assets (v3-produced, reused by v4).
SKY_ROOT = Path("C:/Users/Xiao/Desktop/PythonProject/ultralytics/sky_workflow")

NET_HW = (288, 384)        # (H, W) model input tensor 1x1x288x384
OUT_STRIDE = 4             # C1 head outputs at stride 4
GAMMA_DEFAULT = 1.3        # only bias hyperparameter; frozen after Phase 1
EPOCHS = 80                # C2/C3; C1 uses 120
BATCH = 32
DEVICE = "cpu"             # override with --device xpu

# Frames the teacher is known-unstable on (RGB itself wrong): excluded from training.
EXCLUDED = {"3_000001", "3_000002", "3_000003"}
