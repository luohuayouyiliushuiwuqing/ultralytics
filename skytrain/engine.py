"""Training loop shared by all three candidates (EXECUTION_PLAN.md §4).

The loop is intentionally identical across C1/C2/C3; only the model definition
differs (via arch.build). This is the prerequisite for a fair horizontal
comparison in Phase 2 -- if the loop differed, "which candidate is better"
would degrade into "whose loop was tuned better".

Phase 0 only needs this to (a) run a forward+backward on one batch and
(b) not error on xpu; full training runs in Phase 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytrain.arch import build
from skytrain.data import YSkyDataset
from skytrain.losses import SkyLoss, leakage_erosion
from skytrain.splits import read_splits
from skytrain.config import SKY_ROOT, NET_HW, GAMMA_DEFAULT, EPOCHS, BATCH, DEVICE


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running = 0.0
    n = 0
    for x, t, _ in loader:
        x, t = x.to(device), t.to(device)
        opt.zero_grad()
        logit = model(x)
        loss = loss_fn(logit, t)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, frames, prob_dir, thr=0.5):
    """Directional leakage/erosion on the VALIDATION set (EXECUTION_PLAN.md §6).

    Predicted alpha is upsampled to 640x480; the teacher probability is loaded at
    FULL resolution (not the out_hw copy) so the metric reflects true upsample loss.
    `frames` must be in the same order as `loader` (built with shuffle=False).
    """
    model.eval()
    leak_sum = eros_sum = 0.0
    n = 0
    prob_dir = Path(prob_dir)
    H, W = 480, 640
    for i, (x, _, _) in enumerate(loader):
        x = x.to(device)
        logit = model(x)
        alpha = torch.sigmoid(logit)
        a_up = torch.nn.functional.interpolate(
            alpha, (H, W), mode="bilinear", align_corners=False)
        stem = frames[i]
        dom = stem.split("_")[0]
        pp = cv2.imread(str(prob_dir / dom / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        true = torch.from_numpy(pp.astype(np.float32) / 65535.0)[None, None]
        lk, er = leakage_erosion(a_up, true, thr)
        leak_sum += lk.item()
        eros_sum += er.item()
        n += 1
    return leak_sum / max(n, 1), eros_sum / max(n, 1)


@torch.no_grad()
def eval_checkpoint(arch, ckpt_path, blocks_csv, gamma=GAMMA_DEFAULT,
                    device=DEVICE, out_csv=None):
    """Validation gate (EXECUTION_PLAN.md §6.4): per-frame leakage / erosion / F1
    on the VALIDATION set, computed AFTER bilinear upsample to 640x480.

    Returns an aggregate dict; when out_csv is given, also writes a per-frame CSV.
    """
    import torch.nn.functional as F

    model, out_shape = build(arch)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    _, val_frames = read_splits(blocks_csv)
    img_dir = args_data_root() / "dataset" / "images"
    prob_dir = SKY_ROOT.parent / "output" / "teacher_prob16"
    ds = YSkyDataset(val_frames, training=False, gamma=gamma,
                     prob_dir=prob_dir, img_dir=img_dir, out_hw=out_shape[2:])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    H, W = 480, 640

    rows = []
    leak_sum = eros_sum = f1_sum = 0.0
    n = 0
    for i, (x, _, _) in enumerate(loader):
        x = x.to(device)
        alpha = torch.sigmoid(model(x))
        a_up = F.interpolate(alpha, (H, W), mode="bilinear", align_corners=False)
        stem = val_frames[i]
        dom = stem.split("_")[0]
        pp = cv2.imread(str(prob_dir / dom / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        true = torch.from_numpy(pp.astype(np.float32) / 65535.0)[None, None]
        pred_sky = a_up >= 0.5
        true_sky = true >= 0.5
        ts = true_sky.float().sum().clamp(min=1.0)
        tg = (~true_sky).float().sum().clamp(min=1.0)
        lk = (pred_sky & ~true_sky).float().sum() / tg
        er = (~pred_sky & true_sky).float().sum() / ts
        tp = (pred_sky & true_sky).sum().item()
        fp = (pred_sky & ~true_sky).sum().item()
        fn = (~pred_sky & true_sky).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        rows.append((stem, lk.item(), er.item(), f1))
        leak_sum += lk.item()
        eros_sum += er.item()
        f1_sum += f1
        n += 1

    agg = {"arch": arch, "n_val": n,
           "leakage_mean": leak_sum / max(n, 1),
           "erosion_mean": eros_sum / max(n, 1),
           "f1_mean": f1_sum / max(n, 1)}
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_stem", "leakage", "erosion", "f1"])
            w.writerows(rows)
        print(f"[eval] per-frame written -> {out_csv}")
    print(f"[eval] arch={arch} n={n} leakage={agg['leakage_mean']*100:.3f}% "
          f"erosion={agg['erosion_mean']*100:.3f}% f1={agg['f1_mean']:.4f}")
    return agg


def args_data_root():
    """Fallback data root for eval_checkpoint when SKY_ROOT is the default."""
    return SKY_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, help="C1 | C2 | C3")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--sweep-gamma", type=str, default="",
                    help="comma list, e.g. 1.0,1.3,1.6,2.0 (Phase 1)")
    ap.add_argument("--data-root", type=Path, default=SKY_ROOT)
    ap.add_argument("--blocks-csv", type=Path,
                    default=SKY_ROOT.parent / "output" / "split" / "blocks.csv")
    ap.add_argument("--prob-dir", type=Path,
                    default=SKY_ROOT.parent / "output" / "teacher_prob16")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full; N = use first N training frames (smoke check)")
    ap.add_argument("--smoke", action="store_true",
                    help="one batch forward+backward, then exit (Phase 0 gate)")
    ap.add_argument("--project", type=Path, default=SKY_ROOT.parent / "runs")
    args = ap.parse_args()

    if not args.blocks_csv.is_file():
        print(f"{args.blocks_csv} missing -- run skytrain/splits.py first", file=sys.stderr)
        return 1

    train_frames, val_frames = read_splits(args.blocks_csv)
    if args.limit:
        train_frames = train_frames[:args.limit]
        val_frames = val_frames[:max(1, args.limit // 10)]

    gammas = [float(g) for g in args.sweep_gamma.split(",") if g] or [args.gamma]
    results = []

    for gamma in gammas:
        model, out_shape = build(args.arch)
        img_dir = args.data_root / "dataset" / "images"      # Dataset searches train/val
        prob_dir = args.prob_dir
        ds = YSkyDataset(train_frames, training=True, gamma=gamma,
                         prob_dir=prob_dir, img_dir=img_dir,
                         out_hw=out_shape[2:])  # model output H,W (72x96 C1 / 144x192 C2,C3)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

        if args.smoke:
            model = model.to(args.device)
            x, t, _ = next(iter(loader))
            x, t = x.to(args.device), t.to(args.device)
            loss_fn = SkyLoss(gamma=gamma)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            opt.zero_grad()
            logit = model(x)
            loss = loss_fn(logit, t)
            loss.backward()
            opt.step()
            print(f"[smoke] arch={args.arch} gamma={gamma} "
                  f"in={tuple(x.shape)} out={tuple(logit.shape)} "
                  f"expect_out={out_shape} loss={loss.item():.4f} OK")
            continue

        # full training (Phase 2+)
        model = model.to(args.device)
        loss_fn = SkyLoss(gamma=gamma)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
        run_dir = args.project / f"{args.arch}_g{gamma}"
        run_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader, opt, loss_fn, args.device)
            sched.step()
            history.append({"epoch": ep, "loss": loss})
            print(f"epoch {ep}/{args.epochs} loss={loss:.4f}")
        torch.save(model.state_dict(), run_dir / "best.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"saved {run_dir / 'best.pt'}")

        # ---- Phase 1 evaluation gate (val set, full-res teacher) ----
        # batch_size=1: each sample must be matched to its own teacher frame,
        # otherwise frames[i] (batch index) misaligns and leakage is silently wrong.
        val_ds = YSkyDataset(val_frames, training=False, gamma=gamma,
                            prob_dir=prob_dir, img_dir=img_dir, out_hw=out_shape[2:])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        leak, eros = evaluate(model, val_loader, args.device, val_frames, prob_dir)
        print(f"[eval] arch={args.arch} gamma={gamma:.2f} "
              f"leakage={leak*100:.3f}% erosion={eros*100:.3f}%")
        results.append({"arch": args.arch, "gamma": gamma, "leakage": leak,
                        "erosion": eros, "ckpt": str(run_dir / "best.pt")})

    # ---- sweep summary + calibration curve (Phase 1) ----
    if results:
        ok = [r for r in results if r["leakage"] <= 0.01]
        if ok:
            best = min(ok, key=lambda r: r["erosion"])
            note = "PASS: leakage<=1% met; chosen min-erosion among passing"
        else:
            best = min(results, key=lambda r: r["leakage"])
            note = "WARN: no gamma meets leakage<=1%; picked min-leakage; re-run Phase 1 with wider gamma"
        print("\n=== Gamma sweep summary ===")
        for r in results:
            flag = "OK " if r["leakage"] <= 0.01 else "   "
            print(f"  {flag}gamma={r['gamma']:.2f}  leakage={r['leakage']*100:.3f}%  "
                  f"erosion={r['erosion']*100:.3f}%")
        print(f"RECOMMENDED gamma={best['gamma']:.2f}  ({note})")
        summary_path = args.project / f"{args.arch}_gamma_sweep.json"
        with open(summary_path, "w") as f:
            json.dump({"recommended_gamma": best["gamma"], "note": note,
                       "results": results}, f, indent=2)
        print(f"saved {summary_path}")

        # calibration curve: leakage vs gamma, erosion vs gamma
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            g = [r["gamma"] for r in results]
            lk = [r["leakage"] * 100 for r in results]
            er = [r["erosion"] * 100 for r in results]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(g, lk, "o-", color="tab:red", label="leakage (%)")
            ax.plot(g, er, "s-", color="tab:blue", label="erosion (%)")
            ax.axhline(1.0, ls="--", color="gray", label="leakage<=1% target")
            ax.axvline(best["gamma"], ls=":", color="black",
                       label=f"chosen γ={best['gamma']:.2f}")
            ax.set_xlabel("gamma (confidence push-down)")
            ax.set_ylabel("percent")
            ax.set_title(f"Phase 1 gamma calibration — {args.arch}")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            curve = args.project / f"{args.arch}_gamma_sweep.png"
            fig.savefig(curve, dpi=120)
            print(f"saved {curve}")
        except Exception as e:  # matplotlib optional; never block the run
            print(f"[warn] curve plot skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
