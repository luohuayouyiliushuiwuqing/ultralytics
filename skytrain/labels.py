"""Teacher soft-label conversion: float32 .npy probability -> 16-bit PNG.

16-bit (uint16 = round(prob*65535)) is chosen over 8-bit because 8-bit quantizes
too coarsely in the 0.1-0.9 boundary band, which is exactly where the soft alpha
should stay smooth. The .npy files already exist in sky_workflow/work/prob
(2630 frames), so Phase 0 does NOT need to re-run teacher inference.

Output: output/teacher_prob16/{1,2,3}/*.png
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def npy_to_png16(src: Path, dst: Path) -> None:
    """Convert a float32 probability .npy in [0,1] to uint16 PNG (prob*65535)."""
    prob = np.load(src).astype(np.float32)
    prob = np.clip(prob, 0.0, 1.0)
    u16 = np.round(prob * 65535.0).astype(np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(dst), u16)


def convert_tree(npy_root: Path, out_root: Path) -> int:
    """Walk {domain}/*.npy under npy_root, emit 16-bit PNGs under out_root."""
    npy_root, out_root = Path(npy_root), Path(out_root)
    count = 0
    for dom in sorted(npy_root.iterdir()):
        if not dom.is_dir():
            continue
        for npy in sorted(dom.glob("*.npy")):
            dst = out_root / dom.name / (npy.stem + ".png")
            npy_to_png16(npy, dst)
            count += 1
    return count


def load_png16(path: Path) -> np.ndarray:
    """Read a 16-bit probability PNG back to float32 in [0,1]."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 65535.0


def main() -> int:
    import argparse
    from .config import SKY_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", type=Path, default=SKY_ROOT / "work" / "prob")
    ap.add_argument("--out", type=Path, default=SKY_ROOT.parent / "output" / "teacher_prob16")
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    args = ap.parse_args()
    # convert_tree walks all domains; limit applies per domain roughly
    out = convert_tree(args.npy_dir, args.out)
    print(f"converted {out} probability maps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
