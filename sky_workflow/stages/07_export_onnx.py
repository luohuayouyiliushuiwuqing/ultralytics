"""Stage 7 -- export to ONNX, check parity, and optionally fold to a luma input.

    python sky_workflow/stages/07_export_onnx.py --weights sky_workflow/runs/sky_yolov8n/weights/best.pt
    python sky_workflow/stages/07_export_onnx.py --weights ... --fold-luma

Three things a bare `model.export()` does not tell you:

* **Parity.** PyTorch and the exported ONNX are both pushed through the same
  ultralytics post-processor and their masks compared. An export that quietly
  changes the graph shows up as IoU below 1.0 rather than as a nonzero exit code.
* **Operator census.** Everything the RK3588 NPU would have to place. Conv / Relu /
  Concat / Resize land on the NPU; if Softmax, LayerNorm or MatMul appears here,
  that subgraph falls to CPU and INT8 accuracy goes with it.
* **The luma fold.** Chroma carries almost nothing on this dataset (teacher RGB vs
  luma agreed at mean IoU 0.995 over 300 frames), so a 1-channel input lets the
  board skip colour conversion and cuts input bytes to a third. Summing the stem's
  three input columns is *exact* when the channels are identical, so --fold-luma
  measures how far the masks actually drift.

Ultralytics divides pixels by 255 inside the graph. On RKNN, fold that into
`rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]])` so the board feeds
raw uint8 and no CPU preprocessing happens at all.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sky_only import EXTENSIONS  # noqa: E402


def sample_frames(root: Path, count: int, size: int) -> list:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    picked = [files[i] for i in np.linspace(0, len(files) - 1, min(count, len(files))).astype(int)]
    frames = []
    for path in picked:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append((path.name, cv2.resize(image, (size, size))))
    return frames


def sky_mask(model, image: np.ndarray, imgsz: int) -> np.ndarray:
    """Union of every predicted instance mask -- for one semantic class that is the sky."""
    result = model.predict(source=image, imgsz=imgsz, verbose=False)[0]
    composite = np.zeros((imgsz, imgsz), bool)
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return composite
    for item in masks.data.cpu().numpy():
        composite |= item > 0.5
    return composite


def compare(pairs) -> np.ndarray:
    scores = []
    for first, second in pairs:
        union = np.count_nonzero(first | second)
        scores.append(np.count_nonzero(first & second) / union if union else 1.0)
    return np.array(scores)


def fold_luma(model) -> int:
    """Replace the 3-input stem conv with its column sum. Exact for replicated input."""
    stem = model.model.model[0]
    conv = stem.conv if isinstance(stem.conv, nn.Conv2d) else stem.conv.conv
    replacement = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                            bias=conv.bias is not None)
    with torch.no_grad():
        replacement.weight.copy_(conv.weight.detach().sum(dim=1, keepdim=True))
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias.detach())
    if isinstance(stem.conv, nn.Conv2d):
        stem.conv = replacement
    else:
        stem.conv.conv = replacement
    return int(conv.in_channels)


def census(path: Path) -> Counter:
    import onnx

    graph = onnx.load(str(path))
    return Counter(node.op_type for node in graph.graph.node)


def input_shape(path: Path) -> list:
    import onnx

    dims = onnx.load(str(path)).graph.input[0].type.tensor_type.shape.dim
    return [int(d.dim_value) if d.HasField("dim_value") else d.dim_param for d in dims]


def report(label: str, scores: np.ndarray) -> None:
    print(f"  {label}: mean={scores.mean():.4f} median={np.median(scores):.4f} "
          f"p05={np.percentile(scores, 5):.4f} min={scores.min():.4f}   below 0.99: {(scores < 0.99).sum()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=HERE / "runs/sky_yolov8n/weights/best.pt")
    parser.add_argument("-o", "--out", type=Path, default=HERE / "models")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--frames", type=Path, default=Path("input"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--fold-luma", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"{args.weights} missing -- run stages/06_train_yolo.py first", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    frames = sample_frames(args.frames, args.count, args.imgsz)
    if not frames:
        print(f"no frames under {args.frames}", file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    reference = [sky_mask(model, image, args.imgsz) for _, image in frames]

    exported = Path(str(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                                     simplify=True, dynamic=False)))
    final = args.out / "yolov8n_sky_seg.onnx"
    final.write_bytes(exported.read_bytes())
    onnx_model = YOLO(str(final))
    ops = census(final)
    print(f"{final.name}  input {input_shape(final)}")
    print(f"  ops: {dict(ops.most_common(12))}")
    print("  parity, PyTorch vs exported ONNX:")
    report("IoU", compare(zip(reference, [sky_mask(onnx_model, image, args.imgsz) for _, image in frames])))

    if args.fold_luma:
        before = fold_luma(model)
        luma = [sky_mask(model, cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)[:, :, None]
                         .repeat(3, axis=2), args.imgsz) for _, image in frames]
        print(f"\nluma fold on the {before}-channel stem:")
        scores = compare(zip(reference, luma))
        report("IoU(3ch, 1ch)", scores)
        if scores.min() < 0.95:
            print("    the fold moves real boundaries -- retrain with luma input rather than folding after")
        folded = Path(str(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                                       simplify=True, dynamic=False)))
        target = args.out / "yolov8n_sky_seg_luma.onnx"
        target.write_bytes(folded.read_bytes())
        print(f"  -> {target.name}  input {input_shape(target)}  ops: {dict(census(target).most_common(8))}")

    print("\nNext: tools/convert_rknn.py on the Linux side, then board/bench_rknn.py on the board.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
