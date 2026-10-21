# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage 9 -- batch-train several model yamls, then validate and compare them.

    python sky_workflow/stages/09_batch_train.py --epochs 1000 --imgsz 320 --device 0

Why this exists: picking a backbone/head variant by eye is guesswork. The only
question that matters is "which yaml gives the best mask quality per unit of
compute", and answering it needs the same training run repeated for every
candidate. This script does that, then validates each result and prints one
comparison table.

Training is delegated to `06_train_yolo.py` (subprocess) so every candidate
inherits exactly the same trainer settings and save_dir handling -- the comparison
is then apples to apples. Validation and analysis happen here.

Two things are measured per candidate:
  * accuracy  -- from an explicit `val` pass on the best checkpoint
  * cost      -- parameter count and GFLOPs at the training imgsz

Usage
    # train every sky yaml under models/ and compare
    python sky_workflow/stages/09_batch_train.py --epochs 1000 --imgsz 320 --device 0

    # only re-analyse runs that already exist (no training)
    python sky_workflow/stages/09_batch_train.py --skip-train

    # explicit list, mark anything under a mask mAP50 gate
    python sky_workflow/stages/09_batch_train.py --models models/a.yaml models/b.yaml --gate 0.80
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
STAGES = HERE / "stages"


# ---- cost measurement ------------------------------------------------------
def count_gflops(model, imgsz: int) -> float:
    """Count conv MACs with forward hooks; GFLOPs = 2 * MACs (multiply-add)."""
    import torch
    import torch.nn as nn

    total = 0.0
    hooks = []

    def hook(module, _inp, out):
        nonlocal total
        ho, wo = out.shape[-2], out.shape[-1]
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            cin = module.in_channels // module.groups
            total += ho * wo * module.out_channels * cin * kh * kw
        elif isinstance(module, nn.ConvTranspose2d):
            kh, kw = module.kernel_size
            total += ho * wo * module.in_channels * (module.in_channels // module.groups) * kh * kw

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return 2 * total / 1e9


def measure_cost(yaml_path: Path, imgsz: int):
    """Return (params_M, GFLOPs) for a yaml, or (nan, nan) if it cannot be built."""
    try:
        from ultralytics import YOLO

        model = YOLO(str(yaml_path)).model
        params = sum(p.numel() for p in model.parameters()) / 1e6
        return params, count_gflops(model, imgsz)
    except Exception as error:
        print(f"  cost measurement failed for {yaml_path.name}: {type(error).__name__}: {error}")
        return float("nan"), float("nan")


# ---- training / validation -------------------------------------------------
def find_run_dir(project: Path, stem: str) -> Path | None:
    """Ultralytics appends a number when a run name exists; take the newest match."""
    candidates = [d for d in project.iterdir() if d.is_dir() and d.name.startswith(stem)]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def arg_from_run(run_dir: Path, key: str) -> str | None:
    """Read a scalar setting back out of a run's saved args.yaml."""
    args_file = run_dir / "args.yaml"
    if not args_file.is_file():
        return None
    for line in args_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def yaml_from_run(run_dir: Path) -> Path | None:
    """Recover the model yaml a run was trained from."""
    value = arg_from_run(run_dir, "model")
    return Path(value) if value else None


def imgsz_from_run(run_dir: Path) -> int | None:
    """Recover the imgsz a run was trained at -- validating at a mismatched size
    would understate the model (a 640-trained net scored 0.81 at 640 but 0.55 at 320)."""
    value = arg_from_run(run_dir, "imgsz")
    try:
        return int(value) if value else None
    except ValueError:
        return None


def train_one(yaml_path: Path, args) -> Path | None:
    """Delegate training to stage 06 so all candidates share identical settings."""
    stem = yaml_path.stem
    cmd = [
        sys.executable, str(STAGES / "06_train_yolo.py"),
        "--data", str(args.data),
        "--model", str(yaml_path),
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--pretrained", str(args.pretrained),
        "--project", str(args.project),
        "--name", stem,
    ]
    print(f"\n=== training {yaml_path.name} ===")
    started = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    print(f"  finished in {(time.perf_counter() - started) / 60:.1f} min (rc={result.returncode})")
    if result.returncode:
        print(f"  training FAILED for {yaml_path.name}", file=sys.stderr)
        return None
    return find_run_dir(args.project, stem)


def validate(weights: Path, args, imgsz: int, run_name: str) -> dict:
    """Explicit validation pass on a checkpoint; returns the headline metrics."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(args.data), imgsz=imgsz, device=str(args.device), split=args.split,
        project=str(args.project), name=f"{run_name}_val", exist_ok=True,
    )
    def pick(obj, *names):
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return float(value)
        return float("nan")

    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)
    return {
        "mask_map50": pick(seg, "map50") if seg else float("nan"),
        "mask_map": pick(seg, "map") if seg else float("nan"),
        "mask_p": pick(seg, "mp") if seg else float("nan"),
        "mask_r": pick(seg, "mr") if seg else float("nan"),
        "box_map50": pick(box, "map50") if box else float("nan"),
        "box_map": pick(box, "map") if box else float("nan"),
    }


def read_history(run_dir: Path) -> dict:
    """Pull epochs, training time and the peak metrics from results.csv.

    `time` is cumulative seconds, so the total is the last row -- not the sum.
    The peak columns are reported alongside the explicit val score because the two
    can disagree (best.pt stores one checkpoint at an epoch the csv may rank
    differently), and a silent disagreement would make the table read as truth.
    """
    out = {"epochs": 0, "minutes": float("nan"), "hist_map50": float("nan"), "hist_map": float("nan")}
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return out
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return out
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    out["epochs"] = len(data)
    try:
        out["minutes"] = float(data[-1][header.index("time")]) / 60
    except (ValueError, IndexError):
        pass
    for key, column in (("hist_map50", "metrics/mAP50(M)"), ("hist_map", "metrics/mAP50-95(M)")):
        if column in header:
            i = header.index(column)
            out[key] = max(float(r[i]) for r in data)
    return out


# ---- report ---------------------------------------------------------------
def build_report(rows: list, gate: float | None) -> str:
    head = (
        "| model | params(M) | GFLOPs | epochs | train(min) | "
        "val mAP50 | val mAP50-95 | mask P | mask R | box mAP50 | "
        "peak mAP50 | peak mAP50-95 | verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in rows:
        verdict = "-"
        if gate is not None:
            verdict = "PASS" if r["mask_map50"] >= gate else "FAIL"
        lines.append(
            f"| {r['name']} | {r['params']:.3f} | {r['gflops']:.3f} | {r['epochs']} | "
            f"{r['minutes']:.1f} | {r['mask_map50']:.4f} | {r['mask_map']:.4f} | "
            f"{r['mask_p']:.3f} | {r['mask_r']:.3f} | {r['box_map50']:.4f} | "
            f"{r['hist_map50']:.4f} | {r['hist_map']:.4f} | {verdict} |"
        )
    best = max(rows, key=lambda r: (r["mask_map50"] if r["mask_map50"] == r["mask_map50"] else -1))
    lines.append("")
    lines.append(f"Best mask mAP50: **{best['name']}** ({best['mask_map50']:.4f}) -> {best['weights']}")
    if gate is not None:
        lines.append(f"Gate: mask mAP50 >= {gate}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="*", default=None,
                        help="yamls to compare (default: every models/yolov8n-sky1-seg*.yaml)")
    parser.add_argument("--runs", type=Path, nargs="*", default=None,
                        help="analyse these existing run dirs instead (implies --skip-train)")
    parser.add_argument("--data", type=Path, default=HERE.parent / "datasets" / "sky_seg" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", default="none")
    parser.add_argument("--project", type=Path, default=HERE / "runs")
    parser.add_argument("--split", default="val", help="split used for the validation pass")
    parser.add_argument("--skip-train", action="store_true", help="analyse existing runs only")
    parser.add_argument("--gate", type=float, default=None, help="flag models below this mask mAP50")
    parser.add_argument("--out", type=Path, default=None, help="report path (default: <project>/batch_report.md)")
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"no dataset at {args.data}", file=sys.stderr)
        return 1
    models = args.models or sorted((HERE / "models").glob("yolov8n-sky1-seg*.yaml"))
    if not models:
        print(f"no model yamls found under {HERE / 'models'}", file=sys.stderr)
        return 1
    args.project = Path(args.project)
    args.project.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.runs:  # analyse given run dirs; the yaml is recovered from each run's args
        pairs = [(run_dir, yaml_from_run(run_dir)) for run_dir in args.runs]
    else:
        pairs = [(None, yaml_path) for yaml_path in models]

    for run_dir, yaml_path in pairs:
        if yaml_path is None or not Path(yaml_path).is_file():
            print(f"  cannot resolve model yaml for {run_dir}; skipped", file=sys.stderr)
            continue
        yaml_path = Path(yaml_path)
        stem = run_dir.name if run_dir else yaml_path.stem
        print(f"\n---------- {stem} ----------")
        if run_dir is None:
            run_dir = None if args.skip_train else train_one(yaml_path, args)
        if run_dir is None:
            run_dir = find_run_dir(args.project, yaml_path.stem)
        if run_dir is None:
            print(f"  no run directory for {stem}; skipped", file=sys.stderr)
            continue

        weights = run_dir / "weights" / "best.pt"
        if not weights.is_file():
            print(f"  no best.pt at {weights}; skipped", file=sys.stderr)
            continue

        # Validate at the size the model was trained at, otherwise a 640-trained
        # net gets scored at 320 and looks far worse than it is.
        val_imgsz = imgsz_from_run(run_dir) or args.imgsz
        params, gflops = measure_cost(yaml_path, val_imgsz)
        print(f"  params={params:.3f}M  GFLOPs@{val_imgsz}={gflops:.3f}  -> validating {weights.name}")
        metrics = validate(weights, args, val_imgsz, run_dir.name)
        history = read_history(run_dir)
        rows.append({"name": stem, "params": params, "gflops": gflops, "weights": str(weights),
                     **history, **metrics})
        print(f"  mask mAP50={metrics['mask_map50']:.4f}  mask mAP50-95={metrics['mask_map']:.4f}")

    if not rows:
        print("nothing to report", file=sys.stderr)
        return 1

    report = build_report(rows, args.gate)
    print("\n" + report)
    out = args.out or (args.project / "batch_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
