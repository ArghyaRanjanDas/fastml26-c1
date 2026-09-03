"""Train + evaluate a binary HH_4b vs. background classifier on the A10.

Reports the headline number for Challenge 1 -- the ROC AUC of the network score
on a held-out slice of eval/ -- plus the parameter count and measured inference
time per event (which is the quantity the FPGA implementation has to beat).

  python train.py --model deepset_plus --rho 256,128 --epochs 20
  python train.py --model deepset_plus --rho 96,40 --tag ds_10k
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import build_cache, load_cache, GROUP_ID, EVENT_FEATURES
from models import MODELS, count_params

OUT_DIR = Path(__file__).resolve().parent / "runs"
GROUP_NAME = {v: k for k, v in GROUP_ID.items()}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dims(spec: str):
    return tuple(int(v) for v in spec.split(",") if v.strip())


@torch.no_grad()
def predict(model, X: torch.Tensor, F: torch.Tensor, device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device, non_blocking=True)
        fb = F[i:i + batch_size].to(device, non_blocking=True)
        out.append(torch.sigmoid(model(xb, fb)).float().cpu().numpy())
    return np.concatenate(out)


def auc_report(scores: np.ndarray, y: np.ndarray, group: np.ndarray, title: str):
    auc = roc_auc_score(y, scores)
    print(f"\n=== {title} ===")
    print(f"  events: {len(y)}  signal: {int(y.sum())}  background: {int((1 - y).sum())}")
    print(f"  BINARY AUC (signal vs all background): {auc:.5f}")

    per_group, sig = {}, y == 1
    for gid, name in sorted(GROUP_NAME.items()):
        if name == "HH_4b":
            continue
        sel = sig | ((y == 0) & (group == gid))
        if (y[sel] == 0).sum() == 0:
            continue
        per_group[name] = float(roc_auc_score(y[sel], scores[sel]))
        print(f"    vs {name:<6s}: AUC {per_group[name]:.5f}"
              f"  ({int((y[sel] == 0).sum())} bkg events)")

    # signal efficiency at fixed background rejection -- the trigger-relevant view
    eff, bkg = {}, scores[y == 0]
    for rej in (0.99, 0.999):
        thr = np.quantile(bkg, rej)
        eff[str(rej)] = float((scores[sig] > thr).mean())
        print(f"    signal eff @ {rej * 100:g}% bkg rejection: {eff[str(rej)]:.4f}")
    return float(auc), per_group, eff


def measure_latency(model, X: torch.Tensor, F: torch.Tensor, device):
    """Batched GPU throughput plus single-event CPU latency."""
    results = {}
    n = min(8192, len(X))
    x, f = X[:n].to(device), F[:n].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x, f)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(x, f)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    results["gpu_batched_us_per_event"] = dt / n * 1e6
    results["gpu_batch_size"] = n

    model.to("cpu")
    xc, fc = X[:1].cpu(), F[:1].cpu()
    with torch.no_grad():
        for _ in range(50):
            model(xc, fc)
        t0 = time.perf_counter()
        for _ in range(500):
            model(xc, fc)
        results["cpu_single_event_us"] = (time.perf_counter() - t0) / 500 * 1e6
    model.to(device)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepset_plus", choices=list(MODELS))
    ap.add_argument("--phi", default="64,32,16")
    ap.add_argument("--rho", default="256,128")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n-particles-use", type=int, default=None,
                    help="feed only the leading N candidates to phi (cache stays at 16). "
                         "Halving particles halves the phi cost, which is the FPGA bill.")
    ap.add_argument("--event-scale", type=float, default=1.0,
                    help="fixed multiplier on the event features at the concat point")
    ap.add_argument("--pool-norm", action="store_true",
                    help="BatchNorm the pooled vector before concat (free on FPGA)")
    ap.add_argument("--no-event-features", action="store_true",
                    help="ablation: pooled vector only, no HT/mass/dxy features")
    ap.add_argument("--train-tag", default="train300k")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-signal", type=int, default=300_000)
    ap.add_argument("--n-background", type=int, default=300_000)
    ap.add_argument("--n-eval-signal", type=int, default=100_000)
    ap.add_argument("--n-eval-background", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="run name (default: model name)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.tag or args.model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # ------------------------------------------------------------------ data
    build_cache(args.train_tag, "train", args.n_signal, args.n_background)
    build_cache(args.eval_tag, "eval", args.n_eval_signal, args.n_eval_background)
    Xtr, Ftr, ytr, gtr, meta_tr = load_cache(args.train_tag)
    Xev, Fev, yev, gev, meta_ev = load_cache(args.eval_tag)
    if args.n_particles_use:
        # candidates are pT-sorted descending, so a head slice is the leading N.
        # Event features stay computed from the full 16 -- they are event-level
        # quantities and cost nothing per particle in firmware.
        Xtr, Xev = Xtr[:, :args.n_particles_use], Xev[:, :args.n_particles_use]
    print(f"train: {Xtr.shape} + {Ftr.shape}   eval(held-out): {Xev.shape} + {Fev.shape}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = int(len(perm) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr_t, Ftr_t = torch.from_numpy(Xtr[tr_idx]), torch.from_numpy(Ftr[tr_idx])
    ytr_t = torch.from_numpy(ytr[tr_idx])
    Xva_t, Fva_t = torch.from_numpy(Xtr[val_idx]), torch.from_numpy(Ftr[val_idx])
    yva, gva = ytr[val_idx], gtr[val_idx]
    Xev_t, Fev_t = torch.from_numpy(Xev), torch.from_numpy(Fev)

    n_particles, n_features = Xtr.shape[1], Xtr.shape[2]
    n_event_features = Ftr.shape[1]

    # ----------------------------------------------------------------- model
    use_evt = not args.no_event_features
    model = MODELS[args.model](
        n_features=n_features, n_event_features=n_event_features,
        phi_dims=dims(args.phi), rho_dims=dims(args.rho),
        dropout=args.dropout, use_event_features=use_evt,
        event_scale=args.event_scale, pool_norm=args.pool_norm,
    ).to(device)
    n_params = count_params(model)
    # phi runs once per particle, so this product -- not the parameter count --
    # is what sets DSP/LUT usage on the FPGA.
    pd, d0 = dims(args.phi), n_features
    phi_macs = 0
    for h in pd:
        phi_macs += d0 * h
        d0 = h
    phi_macs *= n_particles
    print(f"\nrun '{run}'  model={args.model}  phi={args.phi}  rho={args.rho}  "
          f"dropout={args.dropout}  event_features={use_evt}  "
          f"event_scale={args.event_scale}  pool_norm={args.pool_norm}")
    print(f"trainable params: {n_params:,}   phi MACs/event: {phi_macs:,} "
          f"({n_particles} particles x {phi_macs // n_particles})")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, Ftr_t, ytr_t),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader), eta_min=args.lr * 1e-3)
    criterion = nn.BCEWithLogitsLoss()

    ckpt = OUT_DIR / f"{run}_best.pt"
    best_auc, history = -1.0, []
    print(f"training {args.epochs} epochs, {len(loader)} steps/epoch")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.perf_counter(), 0.0, 0
        for xb, fb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb, fb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            seen += len(yb)

        val_auc = roc_auc_score(yva, predict(model, Xva_t, Fva_t, device))
        history.append(dict(epoch=epoch, train_loss=tot / seen, val_auc=float(val_auc)))
        flag = ""
        if val_auc > best_auc:
            best_auc, flag = val_auc, "  *"
            torch.save(model.state_dict(), ckpt)
        print(f"  epoch {epoch:2d}/{args.epochs}  train_loss={tot / seen:.4f}  "
              f"val_auc={val_auc:.5f}  ({time.perf_counter() - t0:.1f}s){flag}", flush=True)

    model.load_state_dict(torch.load(ckpt))
    print(f"\nbest validation AUC: {best_auc:.5f}  ({ckpt})")

    # ------------------------------------------------------------ evaluation
    auc_report(predict(model, Xva_t, Fva_t, device), yva, gva, "held-out validation (train split)")
    ev_scores = predict(model, Xev_t, Fev_t, device)
    eval_auc, per_group, eff = auc_report(ev_scores, yev, gev, "held-out EVAL slice (eval/ directory)")

    timing = measure_latency(model, Xev_t, Fev_t, device)
    print(f"\n=== cost ===")
    print(f"  trainable params      : {n_params:,}")
    print(f"  GPU batched (bs={timing['gpu_batch_size']}) : "
          f"{timing['gpu_batched_us_per_event']:.4f} us/event")
    print(f"  CPU single event      : {timing['cpu_single_event_us']:.1f} us/event")

    summary = dict(run=run, model=args.model, phi=dims(args.phi), rho=dims(args.rho),
                   dropout=args.dropout, use_event_features=use_evt,
                   event_scale=args.event_scale, pool_norm=args.pool_norm,
                   event_feature_names=list(EVENT_FEATURES) if use_evt else [],
                   params=n_params, phi_macs=phi_macs, n_particles=n_particles,
                   n_particles_use=args.n_particles_use, eval_auc=eval_auc, val_auc=float(best_auc),
                   per_background_auc=per_group, signal_eff=eff,
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   train_meta=meta_tr, eval_meta=meta_ev, timing=timing, history=history)
    (OUT_DIR / f"{run}_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT_DIR / f"{run}_eval_scores.npy", ev_scores)

    print("\n" + "=" * 66)
    print(f"  {run}: BINARY AUC (HH_4b vs background, eval slice) = {eval_auc:.5f}")
    print(f"  params = {n_params:,}  phi_macs = {phi_macs:,}  particles = {n_particles}"
          f"   |   {timing['cpu_single_event_us']:.1f} us/event (CPU, batch 1)")
    print("=" * 66)


if __name__ == "__main__":
    main()
