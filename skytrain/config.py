"""Shared constants for the v4 sky-mask student training pipeline.

All paths are derived from the repository root, so the pipeline runs from any
checkout on any OS (the previous revision hard-coded a Windows absolute path).
Every constant is still overridable from the command line.

Layout this module assumes (see skytrain/README.md):

    datasets/images/            source frames (flat, .jpg/.png/...)
    datasets/work/prob/         teacher .npy probability maps
    datasets/teacher_prob16/    teacher probabilities as 16-bit PNG, per domain
    datasets/skytrain_data/      dataset/images/{train,val} + blocks.csv
    datasets/runs_skytrain/      checkpoints, history, sweep summary
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "datasets"

# Source frames and the teacher probabilities produced from them.
IMAGE_DIR = DATA_ROOT / "images"
PROB_NPY_DIR = DATA_ROOT / "work" / "prob"

# Student pipeline: image root holding dataset/images/{train,val}, soft labels, split, runs.
SKY_ROOT = DATA_ROOT / "skytrain_data"
PROB_DIR = DATA_ROOT / "teacher_prob16"
BLOCKS_CSV = SKY_ROOT / "blocks.csv"
RUNS_DIR = DATA_ROOT / "runs_skytrain"

NET_HW = (288, 384)        # (H, W) model input tensor 1x1x288x384
OUT_STRIDE = 4             # C1 head outputs at stride 4
GAMMA_DEFAULT = 1.3        # only bias hyperparameter; frozen after Phase 1
EPOCHS = 100               # C2/C3
BATCH = 32
DEVICE = "cpu"             # override with --device cuda / xpu

BLOCK = 20                 # contiguous frames per split atom
VAL_EVERY = 4              # every Nth block becomes validation

# Frames the teacher is known-unstable on (RGB itself wrong): excluded from training.
EXCLUDED = {"3_000001", "3_000002", "3_000003"}

# Accepted frame extensions -- source frames are .jpg in some checkouts, .png in others.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
