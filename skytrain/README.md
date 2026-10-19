# skytrain — sky-mask student (soft-label distillation)

Trains a tiny single-channel "sky alpha" network by distilling a large teacher
(`skyseg`), instead of predicting polygons. Soft labels keep the boundary smooth
and, unlike YOLO polygons, can represent holes (ground islands inside sky).

Everything is driven from one entry point:

```bash
python -m skytrain.run all --arch C2 --epochs 100 --device cuda
#   = prepare -> train -> export ONNX
```

`prepare` is idempotent, so re-running the whole command is safe and cheap.

---

## 1. Quick start

```bash
# 1) produce teacher probability maps (.npy) if you do not have them yet
python sky_workflow/build_dataset.py            # writes datasets/work/prob/*.npy

# 2) one command for everything else
python -m skytrain.run all --arch C2 --epochs 100 --device cuda
```

Check the gate afterwards (leakage must be ≤ 1%):

```bash
python -m skytrain.run eval --arch C2 \
  --ckpt datasets/runs_skytrain/C2_g1.3/best.pt
```

Individual stages are still available (see §4).

---

## 2. `datasets/` directory structure

Code lives in `skytrain/` and `sky_workflow/`; **all data and every intermediate
the images generate live under `datasets/`**. Nothing is written next to the code.

```
datasets/
├── images/                     SOURCE frames (flat, .jpg/.png) — never modified
│                               e.g. d3_1015_001.jpg, demo_960.jpg
│
├── work/                       intermediates from the teacher pass
│   ├── mask/                    binary sky masks           (stage 01)
│   ├── prob/                    float32 .npy probability    (stage 01)  ← student input
│   ├── mask_clean/              speck/hole-pruned masks     (stage 02)
│   ├── labels/                  YOLO polygon labels         (stage 04, YOLO-seg path only)
│   └── verify_worst.png        round-trip check sheet      (stage 08)
│
├── teacher_prob16/              STUDENT soft labels — 16-bit PNG, uint16 = prob*65535
│   ├── d3/    <stem>.png        342 total; domain = stem prefix before "_"
│   └── demo/  <stem>.png
│
├── skytrain_data/               STUDENT dataset root (--data-root)
│   ├── blocks.csv               frame_stem -> train/val (block split)
│   └── dataset/images/
│       ├── train/               281 frames, symlinked to datasets/images
│       └── val/                  61 frames
│
├── runs_skytrain/               STUDENT runs (--project)
│   ├── C2_g1.3/{best.pt,history.json,c2_sky.onnx}
│   └── C2_gamma_sweep.{json,png}
│
└── sky_seg/                     YOLO-seg dataset (the *other* path, 06_train_yolo.py)
    ├── data.yaml, manifest.csv
    └── images/{train,val}, labels/{train,val}
```

Frames are **symlinked, never copied**, so the dataset costs no extra disk and
cannot drift from the source tree. If symlinks are unavailable the code copies instead.

---

## 3. Data contract

| Item | Location | Notes |
|---|---|---|
| frame | `skytrain_data/dataset/images/{train,val}/<stem>.<ext>` | any image extension |
| soft label | `teacher_prob16/<domain>/<stem>.png` | 16-bit, `prob*65535` |
| split | `skytrain_data/blocks.csv` | `domain,block_idx,frame_stem,split` |
| domain | `stem.split("_")[0]` | `d3_1015_001` → `d3` |

Splitting is by **contiguous blocks of frames** (default 20 frames/block, every
4th block to val), never random — these are handheld sweeps where neighbouring
frames are near-duplicates, and a random split makes every metric meaningless.

---

## 4. Subcommands

| Command | What it does |
|---|---|
| `prepare` | flat frames + `.npy` → 16-bit labels + `blocks.csv` + `train/val` links |
| `labels` | teacher `.npy` → 16-bit PNG (standalone) |
| `splits` | block split → `blocks.csv` (standalone) |
| `train` | train one arch at a fixed gamma |
| `sweep` | Phase 1: gamma calibration (trains each gamma, plots leakage/erosion) |
| `export` | checkpoint → ONNX (BN folded, Sigmoid baked in) |
| `eval` | per-frame leakage / erosion / F1 on val |
| `viz` | visualise one frame through the ONNX model |
| `test` | Phase 0 sanity suite (geometry alignment, shapes, normalisation) |
| `all` | `prepare` + `train` + `export` |

```bash
python -m skytrain.run train --arch C2 --epochs 100 --device cuda
python -m skytrain.run sweep --arch C2 --sweep-gamma 1.0,1.3,1.6,2.0 --epochs 100
python -m skytrain.run --help
```

---

## 5. Architectures

| Arch | Model | Output | Status |
|---|---|---|---|
| **C2** | dense 3×3 conv U-Net, ReLU6 | 144×192 @ stride 2 | ✅ works, default |
| **C3** | depthwise-separable U-Net, ReLU6 | 144×192 @ stride 2 | ✅ works |
| **C1** | YOLOv8n backbone+neck + `SkyHead` | 72×96 @ stride 4 | ❌ not wired |

**C1 raises `NotImplementedError`.** `SkyHead.forward` consumes `(p3, p4, p5)`, not
the input tensor, and the YOLOv8n backbone that produces those features is not
connected in this checkout (it also needs 3-channel input, while the student
trains on a single Y plane). Previously `build("C1")` returned the head alone and
the trainer crashed mid-run with a `TypeError`; it now fails immediately with a
clear message. `SkyHead` itself is fine and unit-tested directly.

---

## 6. Metrics and the gate

Metrics are **directional**, not symmetric IoU, and are computed *after* bilinear
upsample to 640×480 against the full-resolution teacher:

- `leakage` = predicted-sky ∩ true-ground / true-ground → **bad**, target **≤ 1%**
- `erosion` = predicted-ground ∩ true-sky / true-sky → acceptable, minimise

`gamma` (default 1.3) is the only bias knob: the training target is `p**gamma`,
which pushes sky confidence down so the model errs toward carving more sky.
**Gamma must be calibrated per dataset** with `sweep`; do not assume 1.3 passes.

### Last run on this checkout (342 frames, C2, 100 epochs, gamma 1.3)

```
loss      0.6007 -> 0.2420
leakage   11.42%      (target <= 1%)   -> gate FAILED
erosion    9.37%
ONNX      datasets/runs_skytrain/C2_g1.3/c2_sky.onnx  (1,1,288,384) -> (1,1,144,192)
```

The gate is **not** met. Next step is a gamma sweep (§4); if no gamma reaches ≤1%,
the model or the teacher labels need revisiting, not just the threshold.

---

## 7. Notes / gotchas

- Teacher input is the **Y (luma) plane**, single channel, Rec.601 weights; range
  jitter (limited/full) is applied during training.
- Geometry augmentation applies the **same affine** to image-Y and probability map.
  A mismatch does not crash — it silently trains garbage. `test` covers it.
- Probability maps are always resampled with `INTER_LINEAR`, never nearest.
- `teacher_prob16` uses 16-bit rather than 8-bit because 8-bit quantises too
  coarsely in the 0.1–0.9 boundary band, which is exactly where the alpha must
  stay smooth.
- VAL_EVERY/BLOCK live in `config.py`; changing them requires re-running `prepare`.
