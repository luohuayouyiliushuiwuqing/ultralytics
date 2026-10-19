"""v4 sky-mask student -- unified training + validation workflow.

This is the single entry point for the whole pipeline. Every stage the project
has built so far is exposed as a subcommand; `all` chains them end to end.

    python -m skytrain.run --help
    python -m skytrain.run test                              # Phase 0 gates
    python -m skytrain.run all --arch C2 --device cuda       # prepare + train + export

Subcommands
    prepare  flat frames + teacher .npy -> 16-bit labels, blocks.csv, train/val links
    labels   teacher .npy -> 16-bit PNG soft labels (standalone)
    splits   block-based train/val split -> blocks.csv (standalone)
    train    train one arch at a fixed gamma (Phase 2)
    sweep    Phase 1 gamma calibration (train each gamma + val eval + curve)
    export   checkpoint -> ONNX (BN fold + baked Sigmoid)
    eval     validation: per-frame leakage/erosion/F1 on the val set
    viz      visualize one frame through the ONNX model
    test     Phase 0 sanity suite (geo-align / shapes / normalization)
    all      prepare + train + export
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skytrain.config import (IMAGE_DIR, SKY_ROOT, BLOCKS_CSV, PROB_DIR, RUNS_DIR,
                             GAMMA_DEFAULT, EPOCHS, BATCH, DEVICE, BLOCK, VAL_EVERY)


def _run(main_fn, argv):
    """Invoke a module's `main()` with a synthetic argv (cleanest cross-module call)."""
    saved = sys.argv
    sys.argv = [(main_fn.__module__ or "module")] + list(argv)
    try:
        return main_fn()
    finally:
        sys.argv = saved


def _sub(module: str, fn: str, argv):
    mod = __import__(module, fromlist=[fn])
    return _run(getattr(mod, fn), argv)


# ---- per-stage dispatchers -------------------------------------------------
def cmd_prepare(a):
    argv = ["--images", str(a.images), "--data-root", str(a.data_root),
            "--blocks-csv", str(a.blocks_csv), "--prob-out", str(a.prob_dir),
            "--block", str(a.block), "--val-every", str(a.val_every)]
    if a.skip_probs:
        argv += ["--skip-probs"]
    return _sub("skytrain.prepare", "main", argv)


def cmd_labels(a):
    argv = []
    if a.npy_dir:
        argv += ["--npy-dir", str(a.npy_dir)]
    if a.prob_dir:
        argv += ["--out", str(a.prob_dir)]
    return _sub("skytrain.labels", "main", argv)


def cmd_splits(a):
    argv = []
    if a.data_root:
        argv += ["--input", str(a.data_root)]
    if a.blocks_csv:
        argv += ["--out", str(a.blocks_csv)]
    argv += ["--block", str(a.block), "--val-every", str(a.val_every)]
    return _sub("skytrain.splits", "main", argv)


def cmd_train(a):
    argv = ["--arch", a.arch, "--epochs", str(a.epochs), "--batch", str(a.batch),
            "--device", a.device, "--gamma", str(a.gamma),
            "--data-root", str(a.data_root), "--project", str(a.project),
            "--blocks-csv", str(a.blocks_csv), "--prob-dir", str(a.prob_dir)]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    if a.smoke:
        argv += ["--smoke"]
    return _sub("skytrain.engine", "main", argv)


def cmd_sweep(a):
    argv = ["--arch", a.arch, "--epochs", str(a.epochs), "--batch", str(a.batch),
            "--device", a.device, "--sweep-gamma", a.sweep_gamma,
            "--data-root", str(a.data_root), "--project", str(a.project),
            "--blocks-csv", str(a.blocks_csv), "--prob-dir", str(a.prob_dir)]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    return _sub("skytrain.engine", "main", argv)


def cmd_export(a):
    argv = ["--arch", a.arch, "--ckpt", str(a.ckpt), "--onnx", str(a.onnx)]
    return _sub("skytrain.export", "main", argv)


def cmd_eval(a):
    from skytrain.engine import eval_checkpoint
    agg = eval_checkpoint(a.arch, a.ckpt, a.blocks_csv, a.gamma, a.device, a.out)
    return 1 if agg["leakage_mean"] > 0.01 else 0


def cmd_viz(a):
    argv = ["--image", str(a.image), "--onnx", str(a.onnx), "--out", str(a.out)]
    if a.teacher:
        argv += ["--teacher", str(a.teacher)]
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "visualize.py"), *argv],
        check=False).returncode


def cmd_test(a):
    return subprocess.run(
        [sys.executable, str(ROOT / "tests" / "run_all.py")],
        check=False).returncode


def cmd_all(a):
    # Phase 0: 16-bit soft labels, block split, train/val frame links (idempotent)
    if not a.skip_prepare:
        rc = cmd_prepare(argparse.Namespace(
            images=a.images, data_root=a.data_root, blocks_csv=a.blocks_csv,
            prob_dir=a.prob_dir, block=a.block, val_every=a.val_every,
            skip_probs=a.skip_probs))
        if rc:
            return rc

    # Phase 2: train with the chosen gamma
    rc = cmd_train(argparse.Namespace(
        arch=a.arch, epochs=a.epochs, batch=a.batch, device=a.device,
        gamma=a.gamma, data_root=a.data_root, project=a.project,
        blocks_csv=a.blocks_csv, prob_dir=a.prob_dir, limit=a.limit, smoke=False))
    if rc:
        return rc

    # export -> ONNX (BN fold + baked Sigmoid)
    run_dir = a.project / f"{a.arch}_g{a.gamma}"
    ckpt = run_dir / "best.pt"
    onnx = run_dir / f"{a.arch.lower()}_sky.onnx"
    rc = cmd_export(argparse.Namespace(arch=a.arch, ckpt=ckpt, onnx=onnx))
    if rc:
        return rc

    print(f"\n[all] complete. ONNX -> {onnx}")
    print(f"[all] gate check:  python -m skytrain.run eval --arch {a.arch} --ckpt {ckpt}")
    return 0


# ---- argparse --------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="v4 sky-mask student training+validation workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-root", type=Path, default=SKY_ROOT)
        sp.add_argument("--blocks-csv", type=Path, default=BLOCKS_CSV)
        sp.add_argument("--prob-dir", type=Path, default=PROB_DIR)
        sp.add_argument("--project", type=Path, default=RUNS_DIR)
        sp.add_argument("--device", default=DEVICE)
        return sp

    sp = sub.add_parser("prepare", help="flat frames + teacher probs -> student dataset")
    sp.add_argument("--images", type=Path, default=IMAGE_DIR)
    sp.add_argument("--data-root", type=Path, default=SKY_ROOT)
    sp.add_argument("--blocks-csv", type=Path, default=BLOCKS_CSV)
    sp.add_argument("--prob-dir", type=Path, default=PROB_DIR)
    sp.add_argument("--block", type=int, default=BLOCK)
    sp.add_argument("--val-every", type=int, default=VAL_EVERY)
    sp.add_argument("--skip-probs", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("labels", help="teacher .npy -> 16-bit PNG soft labels")
    sp.add_argument("--npy-dir", type=Path, default=None)
    sp.add_argument("--prob-dir", type=Path, default=PROB_DIR)
    sp.set_defaults(func=cmd_labels)

    sp = sub.add_parser("splits", help="block-based train/val split")
    sp.add_argument("--data-root", type=Path, default=SKY_ROOT)
    sp.add_argument("--blocks-csv", type=Path, default=BLOCKS_CSV)
    sp.add_argument("--block", type=int, default=BLOCK)
    sp.add_argument("--val-every", type=int, default=VAL_EVERY)
    sp.set_defaults(func=cmd_splits)

    sp = common(sub.add_parser("train", help="train one arch @ fixed gamma"))
    sp.add_argument("--arch", required=True)
    sp.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    sp.add_argument("--epochs", type=int, default=EPOCHS)
    sp.add_argument("--batch", type=int, default=BATCH)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--smoke", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = common(sub.add_parser("sweep", help="Phase 1 gamma calibration"))
    sp.add_argument("--arch", required=True)
    sp.add_argument("--sweep-gamma", type=str, default="1.0,1.3,1.6,2.0")
    sp.add_argument("--epochs", type=int, default=EPOCHS)
    sp.add_argument("--batch", type=int, default=BATCH)
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_sweep)

    sp = common(sub.add_parser("export", help="checkpoint -> ONNX"))
    sp.add_argument("--arch", required=True)
    sp.add_argument("--ckpt", type=Path, required=True)
    sp.add_argument("--onnx", type=Path, required=True)
    sp.set_defaults(func=cmd_export)

    sp = common(sub.add_parser("eval", help="validation: leakage/erosion/F1"))
    sp.add_argument("--arch", required=True)
    sp.add_argument("--ckpt", type=Path, required=True)
    sp.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    sp.add_argument("--out", type=Path, default=None, help="per-frame CSV")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("viz", help="visualize one frame through ONNX")
    sp.add_argument("--image", required=True)
    sp.add_argument("--onnx", required=True)
    sp.add_argument("--teacher", default=None)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_viz)

    sub.add_parser("test", help="Phase 0 sanity suite").set_defaults(func=cmd_test)

    sp = sub.add_parser("all", help="prepare + train + export")
    sp.add_argument("--arch", default="C2")
    sp.add_argument("--images", type=Path, default=IMAGE_DIR)
    sp.add_argument("--data-root", type=Path, default=SKY_ROOT)
    sp.add_argument("--blocks-csv", type=Path, default=BLOCKS_CSV)
    sp.add_argument("--prob-dir", type=Path, default=PROB_DIR)
    sp.add_argument("--project", type=Path, default=RUNS_DIR)
    sp.add_argument("--device", default=DEVICE)
    sp.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    sp.add_argument("--epochs", type=int, default=EPOCHS)
    sp.add_argument("--batch", type=int, default=BATCH)
    sp.add_argument("--block", type=int, default=BLOCK)
    sp.add_argument("--val-every", type=int, default=VAL_EVERY)
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--skip-prepare", action="store_true")
    sp.add_argument("--skip-probs", action="store_true")
    sp.set_defaults(func=cmd_all)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
