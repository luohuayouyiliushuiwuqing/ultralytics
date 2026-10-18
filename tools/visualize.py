"""Visualize sky-mask inference on a single image.

Runs the exported ONNX (or a PyTorch .pt) on one real frame and renders a
2x2 composite: original | predicted sky alpha | suppressed (sky dimmed) |
teacher prob (reference). Inference preprocessing mirrors skytrain.data.YSkyDataset
exactly (Y plane, 288x384, (Y/127.5)-1, no jitter/geo).

Usage:
    python tools/visualize.py --image <png> \
        --onnx runs/C2_g1.3/sky_c2.onnx \
        --teacher <prob16 png or .npy> \
        --out runs/visualize/<stem>.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


NET_HW = (288, 384)  # (H, W) model input


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Y plane -> 288x384 -> (Y/127.5)-1 in [-1,1], shape (1,1,288,384)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # Rec.601
    y = cv2.resize(y, (NET_HW[1], NET_HW[0]), interpolation=cv2.INTER_LINEAR)
    x = (y.astype(np.float32) / 127.5 - 1.0)
    return x[None, None]  # (1,1,H,W)


def infer_onnx(sess, image_bgr: np.ndarray) -> np.ndarray:
    x = preprocess(image_bgr)
    out = sess.run(["alpha"], {"x": x.astype(np.float32)})[0]  # (1,1,oh,ow)
    return out[0, 0]  # (oh, ow) in [0,1]


def upsample(alpha: np.ndarray, h: int, w: int) -> np.ndarray:
    a = (alpha * 255).astype(np.uint8)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return a.astype(np.float32) / 255.0


def load_teacher(path: Path, h: int, w: int) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.suffix == ".npy":
        prob = np.load(str(p)).astype(np.float32)
    else:
        prob = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    return upsample(prob, h, w)


def colorize(alpha: np.ndarray) -> np.ndarray:
    """Grayscale mask: white = sky (alpha=1), black = ground (alpha=0)."""
    a = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return np.repeat(a[..., None], 3, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--teacher", default=None, help="16-bit PNG or .npy teacher prob")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = img.shape[:2]

    alpha = infer_onnx(sess, img)            # (oh, ow)
    alpha_up = upsample(alpha, H, W)         # (H, W) in [0,1]

    # overlay: suppress sky by dimming original where alpha high
    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    suppressed = orig * (1.0 - 0.75 * alpha_up[..., None])

    teacher = load_teacher(args.teacher, H, W)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].imshow(orig)
    ax[0, 0].set_title("Original", fontsize=12)
    ax[0, 1].imshow(colorize(alpha_up))
    ax[0, 1].set_title("Predicted sky alpha (ONNX)", fontsize=12)
    ax[1, 0].imshow(suppressed)
    ax[1, 0].set_title("Suppressed (sky dimmed x0.75)", fontsize=12)
    if teacher is not None:
        ax[1, 1].imshow(colorize(teacher))
        ax[1, 1].set_title("Teacher prob (reference)", fontsize=12)
    else:
        ax[1, 1].imshow(orig)
        ax[1, 1].set_title("Original (no teacher given)", fontsize=12)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{Path(args.image).name}  |  model input 1x1x288x384  "
                 f"out 1x1x144x192 -> upsampled {W}x{H}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}  alpha range [{alpha_up.min():.3f}, {alpha_up.max():.3f}]  "
          f"sky-frac={alpha_up.mean():.3f}")


if __name__ == "__main__":
    main()
