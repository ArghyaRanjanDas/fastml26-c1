"""Train + evaluate a binary HH_4b vs. background classifier on the A10.

Reports the headline number for Challenge 1 -- the ROC AUC of the network score
on a held-out slice of eval/ -- plus the parameter count and measured inference
time per event (which is the quantity the FPGA implementation has to beat).

  python team/train.py --model deepset --epochs 20
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

from data import build_cache, load_cache, GROUP_ID
from models import MODELS, count_params

OUT_DIR = Path(__file__).resolve().parent / "runs"
GROUP_NAME = {v: k for k, v in GROUP_ID.items()}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X: torch.Tensor, device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size].to(device, non_blocking=True))
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def auc_report(scores: np.ndarray, y: np.ndarray, group: np.ndarray, title: str):
    auc = roc_auc_score(y, scores)
    print(f"\n=== {title} ===")
    print(f"  events: {len(y)}  signal: {int(y.sum())}  background: {int((1 - y).sum())}")
    print(f"  BINARY AUC (signal vs all background): {auc:.5f}")

    sig = y == 1
    for gid, name in sorted(GROUP_NAME.items()):
        if name == "HH_4b":
            continue
        sel = sig | ((y == 0) & (group == gid))
        if (y[sel] == 0).sum() == 0:
            continue
        print(f"    vs {name:<6s}: AUC {roc_auc_score(y[sel], scores[sel]):.5f}"
              f"  ({int((y[sel] == 0).sum())} bkg events)")

    # signal efficiency at fixed background rejection -- the trigger-relevant view
    bkg = scores[y == 0]
    for rej in (0.99, 0.999):
        thr = np.quantile(bkg, rej)
        print(f"    signal eff @ {rej * 100:g}% bkg rejection: {(scores[sig] > thr).mean():.4f}")
    return auc


def measure_latency(model, n_features: int, n_particles: int, device):
    """Batched GPU throughput plus single-event CPU latency."""
    results = {}

    x = torch.randn(8192, n_particles, n_features, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
    results["gpu_batched_us_per_event"] = dt / len(x) * 1e6
    results["gpu_batch_size"] = len(x)

    cpu_model = model.to("cpu")
    xc = torch.randn(1, n_particles, n_features)
    with torch.no_grad():
        for _ in range(50):
            cpu_model(xc)
        t0 = time.perf_counter()
        for _ in range(500):
            cpu_model(xc)
        results["cpu_single_event_us"] = (time.perf_counter() - t0) / 500 * 1e6
    model.to(device)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepset", choices=list(MODELS))
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
    Xtr, ytr, gtr, meta_tr = load_cache(args.train_tag)
    Xev, yev, gev, meta_ev = load_cache(args.eval_tag)
    print(f"train: {Xtr.shape}  eval(held-out): {Xev.shape}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = int(len(perm) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr_t = torch.from_numpy(Xtr[tr_idx])
    ytr_t = torch.from_numpy(ytr[tr_idx])
    Xva_t = torch.from_numpy(Xtr[val_idx])
    yva = ytr[val_idx]
    gva = gtr[val_idx]
    Xev_t = torch.from_numpy(Xev)

    n_particles, n_features = Xtr.shape[1], Xtr.shape[2]

    # ----------------------------------------------------------------- model
    model = MODELS[args.model]().to(device)
    n_params = count_params(model)
    print(f"\nmodel: {args.model}   trainable params: {n_params:,}")
    print(model)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader), eta_min=args.lr * 1e-3)
    criterion = nn.BCEWithLogitsLoss()

    ckpt = OUT_DIR / f"{run}_best.pt"
    best_auc, history = -1.0, []
    print(f"\ntraining {args.epochs} epochs, {len(loader)} steps/epoch")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.perf_counter(), 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            seen += len(yb)

        val_scores = predict(model, Xva_t, device)
        val_auc = roc_auc_score(yva, val_scores)
        history.append(dict(epoch=epoch, train_loss=tot / seen, val_auc=val_auc))
        flag = ""
        if val_auc > best_auc:
            best_auc, flag = val_auc, "  *"
            torch.save(model.state_dict(), ckpt)
        print(f"  epoch {epoch:2d}/{args.epochs}  train_loss={tot / seen:.4f}  "
              f"val_auc={val_auc:.5f}  ({time.perf_counter() - t0:.1f}s){flag}", flush=True)

    model.load_state_dict(torch.load(ckpt))
    print(f"\nbest validation AUC: {best_auc:.5f}  ({ckpt})")

    # ------------------------------------------------------------ evaluation
    auc_report(predict(model, Xva_t, device), yva, gva, "held-out validation (train split)")
    ev_scores = predict(model, Xev_t, device)
    eval_auc = auc_report(ev_scores, yev, gev, "held-out EVAL slice (eval/ directory)")

    timing = measure_latency(model, n_features, n_particles, device)
    print(f"\n=== cost ===")
    print(f"  trainable params      : {n_params:,}")
    print(f"  GPU batched (bs={timing['gpu_batch_size']}) : "
          f"{timing['gpu_batched_us_per_event']:.4f} us/event")
    print(f"  CPU single event      : {timing['cpu_single_event_us']:.1f} us/event")

    summary = dict(run=run, model=args.model, params=n_params, eval_auc=float(eval_auc),
                   val_auc=float(best_auc), epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, train_meta=meta_tr, eval_meta=meta_ev, timing=timing,
                   history=history)
    (OUT_DIR / f"{run}_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT_DIR / f"{run}_eval_scores.npy", ev_scores)

    print("\n" + "=" * 62)
    print(f"  {args.model.upper()} BINARY AUC (HH_4b vs background, eval slice) = {eval_auc:.5f}")
    print(f"  params = {n_params:,}   |   {timing['cpu_single_event_us']:.1f} us/event (CPU, batch 1)")
    print("=" * 62)


if __name__ == "__main__":
    main()
