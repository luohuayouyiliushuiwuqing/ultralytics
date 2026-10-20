# Dataset README — sky segmentation

How the sky dataset is produced, what every file means, and which of the two
training paths consumes what. All data lives under `datasets/`; all code lives in
`sky_workflow/` (YOLO-seg path) and `skytrain/` (student distillation path).

---

## 1. Build it

```bash
# teacher masks + probability maps + polygon labels + YOLO-seg dataset (+ verify)
python sky_workflow/build_dataset.py

# student dataset (soft labels, split, train/val links) + train + export ONNX
python -m skytrain.run all --arch C2 --epochs 100 --device cuda
```

Both are idempotent — re-running is safe and cheap.

---

## 2. Directory layout

```
datasets/
├── images/                  SOURCE frames — flat, .jpg/.png, never modified
│                            342 frames: 201 x d3_*, 141 x demo_*
│
├── work/                    intermediates from the teacher pass
│   ├── mask/                 binary sky mask, 255=sky 0=ground      (stage 01)
│   ├── prob/                 float32 .npy probability, 480x640       (stage 01)
│   ├── mask_clean/           speck/hole-pruned masks                 (stage 02)
│   ├── labels/               YOLO polygon .txt                       (stage 04)
│   ├── qc_frames.csv         temporal-smoothness QC report           (stage 03)
│   ├── qc_suspects.png       QC montage                              (stage 03)
│   ├── polygon_stats.csv     per-frame polygon/hole stats            (stage 04)
│   ├── verify_worst.png      round-trip check sheet                  (stage 08)
│   └── excluded.txt          frames a human rejected                 (stage 01 review)
│
├── sky_seg/                 YOLO-seg dataset  -> 06_train_yolo.py
│   ├── data.yaml            nc:1, names: 0=sky
│   ├── manifest.csv         frame,stem,sequence,block,split
│   ├── images/{train,val}/  262 / 80
│   └── labels/{train,val}/  YOLO polygon .txt, one per frame
│
├── teacher_prob16/          student soft labels, 16-bit PNG (uint16 = prob*65535)
│   ├── d3/   201 *.png
│   └── demo/ 141 *.png
│
├── skytrain_data/           student dataset root  -> skytrain
│   ├── blocks.csv           domain,block_idx,frame_stem,split
│   └── dataset/images/{train,val}/   281 / 61  (symlinked to datasets/images)
│
└── runs_skytrain/           student checkpoints, history, sweep summary
    └── C2_g1.3/{best.pt,history.json,c2_sky.onnx}
```

Frames are **symlinked**, never copied: no extra disk, and the dataset cannot drift
from the source tree.

---

## 3. File formats

| File | Format | Meaning |
|---|---|---|
| `work/mask/*.png` | 8-bit grayscale | 255 = sky, 0 = ground (`prob >= 0.5`) |
| `work/prob/*.npy` | float32, 480×640 | teacher sky probability in [0,1] |
| `work/labels/*.txt` | YOLO-seg polygon | `0 x1 y1 x2 y2 ...` normalised to the frame's own W/H; class 0 = sky |
| `teacher_prob16/*.png` | 16-bit grayscale | `uint16 = round(prob * 65535)`; read back `/65535.0` |
| `sky_seg/data.yaml` | YOLO dataset | `path`, `train`, `val`, `nc: 1`, `names` |

**Why 16-bit for soft labels:** 8-bit only has 256 levels (≈0.39% step), 16-bit has
65536 (≈0.0015%). The 0.1–0.9 boundary band is exactly where the soft alpha must stay
smooth. PNG (not `.npy`) because the loader uses cv2 and geometry augmentation must
apply the same affine to image and probability map.

**Why polygons lose information:** YOLO `.txt` polygons are binary and have no
even-odd fill, so (a) the soft boundary disappears and (b) ground islands inside sky
are silently filled as sky — measured at up to **8.81% of a frame**. This is why the
student path distills the dense probability map instead.

---

## 4. Splitting

**By contiguous blocks of frames, never randomly.** These are handheld sweeps where
frame N and N+1 are almost the same picture; a random split puts near-duplicates of
every training frame into validation and makes every metric meaningless.

- YOLO-seg path (`05_build_dataset.py`): 20 frames per block, every 4th block to val → **262 / 80 (23.4%)**
- Student path (`skytrain`): per **domain**, 20 frames per block, every 4th to val → **281 / 61 (17.8%)**

Both splits are per-domain/per-sequence so every domain appears in train *and* val.
If a split comes out empty the stage **fails loudly** rather than shipping an
unusable dataset — tune `--block` / `--val-every`.

---

## 5. Data cleaning (interactive review)

```bash
python sky_workflow/stages/01_label_teacher.py datasets/images \
  -o datasets/work/mask --review --device cpu        # after labeling
  ... --review-only                                   # review existing masks only
```

Shows the frame beside its mask; keys:

| key | action |
|---|---|
| `a` | previous frame |
| `d` | next frame |
| `x` | mark for deletion |
| `u` | undo |
| `q` / `Esc` | quit and save |

`x` **never deletes a file** — it records the frame stem in `datasets/work/excluded.txt`,
which stages 04 and 05 already skip. Every mark is printed and reversible (`u`, or
edit the text file). Needs a real interactive display: on a headless box the review
refuses rather than acting on stray input.

---

## 6. Which path consumes what

| Path | Inputs | Model | Output |
|---|---|---|---|
| **YOLO-seg** (`06_train_yolo.py`) | `sky_seg/` (images + polygon labels) | `yolov8n-sky1-seg.yaml` (nc=1, nm=1) | instance masks, mAP |
| **Student** (`skytrain`) | `skytrain_data/` (images) + `teacher_prob16/` (soft labels) | TinyNet U-Net (C2/C3) | single-channel soft alpha, ONNX |

The YOLO-seg path is the real YOLOv8n-seg stripped to one class and one mask
prototype (86 layers / 3.08 M params / 8.4 GFLOPs). The student path is a
self-written network trained by distillation — see `skytrain/README.md`.

---

## 7. Current quality (this checkout, 342 frames)

**Label fidelity** (stage 08, polygon rendered vs source mask, IoU):

```
train  mean 0.9954  median 0.9972  p05 0.9877  min 0.8668   below 0.97:  6 (2.3%)
val    mean 0.9933  median 0.9977  p05 0.9681  min 0.9555   below 0.97:  5 (6.2%)
```

The residual tail is the hole limitation above, not polygon coarseness — raising
`--max-points` cannot fix it.

**Trained models:**

| Model | Result |
|---|---|
| YOLO-seg (100 ep) | Mask mAP50 **0.686**, mAP50-95 **0.517** |
| Student C2 (100 ep, γ=1.3) | leakage **11.42%** (target ≤1%) — **gate failed** |

---

## 8. Caveats you should not forget

- **The data is overwhelmingly sky.** Median sky coverage ≈ **97%**; 72 of 141
  `demo_*` frames exceed 95%. Most polygon labels degenerate to a full-frame
  rectangle, and a model can score well by predicting "all sky".
- **Mask mAP is measured against teacher-derived labels.** It can never exceed the
  teacher, and it does not prove the teacher was right.
- **The teacher is fixed at 320×320** and needs ImageNet normalisation. A 640×480
  frame is squashed to square before inference and stretched back — the boundary
  lands in the right place, but the teacher judged a distorted image.
- **No COCO pretrained weights are reachable** from this machine (GitHub times out,
  HF mirror 404s), so the YOLO-seg run trained from scratch.
