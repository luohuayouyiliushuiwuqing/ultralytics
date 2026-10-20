# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build a trainable sky segmentation dataset from a folder of images, one command.

    python sky_workflow/build_dataset.py

Code lives in `sky_workflow/`; every artifact the images generate lands under
`datasets/`, so the two never mix and the source frames are never written to:

    datasets/images/             input frames (source of truth, never modified)
    datasets/work/mask/          teacher binary masks          (stage 01)
    datasets/work/prob/           teacher .npy probability maps (stage 01)
    datasets/work/mask_clean/    speck/hole pruned masks        (stage 02)
    datasets/work/labels/        YOLO polygon labels            (stage 04)
    datasets/work/qc_*.csv/.png  quality report                 (stage 03, --qc)
    datasets/work/verify_*.png    round-trip check               (stage 08)
    datasets/sky_seg/            trainable dataset              (stage 05)

Stages run 01 -> 02 -> 04 -> 05 -> 08. Stage 03 is opt-in via --qc because its
exclusions are a human decision: it writes a report, and any frames to drop go
into `datasets/work/excluded.txt` (one path per line, relative to --images), which
stages 04 and 05 read on the next run.

Note that stage 05 deletes and rebuilds its output directory, so --out must never
point at --images or --work; that is checked before anything runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # sky_workflow/
REPO = HERE.parent
STAGES = HERE / "stages"


def _load(name: str):
    """Import a stage module by filename (stage names begin with a digit, so dotted imports are invalid)."""
    path = STAGES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(name: str, *argv) -> int:
    """Run a stage's main() with a synthetic argv, returning its exit code."""
    module = _load(name)
    saved = sys.argv
    sys.argv = [str(STAGES / f"{name}.py"), *(str(a) for a in argv)]
    try:
        return module.main() or 0
    finally:
        sys.argv = saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=REPO / "datasets" / "images", help="input frames")
    parser.add_argument("--work", type=Path, default=REPO / "datasets" / "work", help="intermediates")
    parser.add_argument("-o", "--out", type=Path, default=REPO / "datasets" / "sky_seg", help="trainable dataset")
    parser.add_argument("-m", "--model", type=Path, default=HERE / "models/skyseg_teacher.onnx")
    parser.add_argument("--device", default="cpu", help="cpu is safest; auto/GPU/NPU use OpenVINO when present")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="teacher probability -> sky")
    parser.add_argument("--clean-min-area", type=int, default=150, help="stage 2 component prune size")
    # Measured on 342 frames: 0.0015/120 leaves 16 frames under the 0.97 round-trip
    # IoU floor (train min 0.855); 0.0008/300 cuts that to 11 and lifts val mean to
    # 0.993. Holes are unaffected by these -- they are a polygon-format limit.
    parser.add_argument("--epsilon", type=float, default=0.0008, help="stage 4 approxPolyDP start, fraction of arc length")
    parser.add_argument("--max-points", type=int, default=300, help="stage 4 polygon point budget")
    parser.add_argument("--block", type=int, default=20, help="contiguous frames per split atom")
    parser.add_argument("--val-every", type=int, default=4, help="every Nth block becomes validation")
    parser.add_argument("--limit", type=int, default=0, help="0 = all frames; N = first N (smoke run)")
    parser.add_argument("--qc", action="store_true", help="also run stage 3 quality report")
    parser.add_argument("--skip-teacher", action="store_true", help="reuse --work/mask instead of re-running stage 1")
    args = parser.parse_args()

    images = args.images.resolve()
    work = args.work.resolve()
    out = args.out.resolve()

    if not images.is_dir():
        print(f"no images under {images}", file=sys.stderr)
        return 1
    # Stage 5 rebuilds --out from scratch; pointing it at the sources would delete them.
    if out == images or images in out.parents or out == work or work in out.parents:
        print(f"--out {out} must not sit under --images or --work (stage 5 rebuilds it)", file=sys.stderr)
        return 1

    mask = work / "mask"
    prob = work / "prob"
    mask_clean = work / "mask_clean"
    labels = work / "labels"
    # One list shared by stage 01 (--review writes it), 04 and 05 (they skip what is
    # on it), so a manual cleaning decision flows through the whole pipeline.
    excluded = work / "excluded.txt"

    stage = ["--limit", args.limit] if args.limit else []

    steps = []
    if not args.skip_teacher:
        steps.append(("01 teacher masks", "01_label_teacher",
                      [images, "-o", mask, "--prob", prob, "-m", args.model,
                       "-t", args.threshold, "--device", args.device,
                       "--excluded", excluded, *stage]))
    steps.append(("02 clean masks", "02_clean_labels",
                  [mask, "-o", mask_clean, "--min-area", args.clean_min_area]))
    if args.qc:
        steps.append(("03 quality report", "03_qc_labels",
                      [mask_clean, "--images", images,
                       "--report", work / "qc_frames.csv", "--sheet", work / "qc_suspects.png"]))
    steps.append(("04 masks -> YOLO polygons", "04_masks_to_yolo",
                  ["--images", images, "--masks", mask_clean, "-o", labels,
                   "--epsilon", args.epsilon, "--max-points", args.max_points,
                   "--excluded", excluded, "--stats", work / "polygon_stats.csv"]))
    steps.append(("05 assemble dataset", "05_build_dataset",
                  ["--images", images, "--labels", labels, "-o", out,
                   "--excluded", excluded,
                   "--block", args.block, "--val-every", args.val_every]))
    steps.append(("08 verify round-trip", "08_verify_dataset",
                  ["--dataset", out, "--masks", mask_clean, "--sheet", work / "verify_worst.png"]))

    for title, name, argv in steps:
        print(f"\n=== {title} ({name}.py) ===")
        code = _run(name, *argv)
        if code:
            print(f"stage {name} failed with {code}", file=sys.stderr)
            return code

    print(f"\ndataset ready -> {out / 'data.yaml'}")
    print(f"train with:  python sky_workflow/stages/06_train_yolo.py --data {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
