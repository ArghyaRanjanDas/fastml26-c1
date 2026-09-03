"""Train a teacher on train1M, evaluate on eval100k, dump soft targets.

  python train_teacher.py --model deepset --tag ds_big --epochs 40
  python train_teacher.py --model part --tag part_s0 --epochs 50 --ema 0.999 --compile --publish

Writes runs/<tag>/{best.pt, summary.json, logits_<cache>.npy}; with --publish also copies
the logits to team/teacher/soft_targets_<cache>.npy (float32 teacher logits, cache row order).

GPU etiquette: this A100 is shared with a higher-priority trainer, so the process caps
itself at half the device memory and everything is sized to stay well below that.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (CACHE_TAGS, HERE, RUNS, GROUP_ID, load_cache, auc_report, quick_auc,
                    write_json, _check_constants)
from models import MODELS, count_params

GPU_MEMORY_FRACTION = 0.5


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of parameters *and* float buffers (BatchNorm stats)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        # warm-up: decay ramps from ~0 so early evaluations are not dominated by the random init
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        src = [p for p in model.parameters()] + [b for b in model.buffers() if b.dtype.is_floating_point]
        dst = [p for p in self.module.parameters()] + [b for b in self.module.buffers() if b.dtype.is_floating_point]
        torch._foreach_lerp_(dst, src, 1.0 - d)
        for s, d in zip(model.buffers(), self.module.buffers()):
            if not s.dtype.is_floating_point:
                d.copy_(s)


@torch.no_grad()
def predict(model, X: torch.Tensor, Fe: torch.Tensor, batch_size: int = 8192, device="cuda") -> np.ndarray:
    """Logits (float32, same row order as the inputs).  X/F may live on CPU or GPU."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device, non_blocking=True)
        fb = Fe[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out.append(model(xb, fb).float())
    return torch.cat(out).cpu().numpy().astype(np.float32)


def build_model(args, n_event: int):
    if args.model == "deepset":
        return MODELS["deepset"](n_event=n_event, rich=not args.raw_feats, dropout=args.dropout)
    return MODELS["part"](n_event=n_event, rich=not args.raw_feats, d=args.d, n_heads=args.heads,
                          n_blocks=args.blocks, n_cls_blocks=args.cls_blocks, dropout=args.dropout,
                          mlp_ratio=args.mlp_ratio, pair_hidden=args.pair_hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepset", choices=list(MODELS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-tag", default="train1M")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-train", type=int, default=None, help="use only a random N-row subset of the cache (smoke tests)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-ratio", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--tt-weight", type=float, default=1.0, help="loss weight on tt background events")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off); eval/soft targets use the EMA")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--raw-feats", action="store_true", help="feed only the 5 cached per-candidate features")
    # ParT-lite
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--cls-blocks", type=int, default=2)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--pair-hidden", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-soft-targets", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="copy this run's logits to team/teacher/soft_targets_<cache>.npy")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _check_constants()
    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {torch.cuda.get_device_name(0)}  memory cap: {GPU_MEMORY_FRACTION:.0%}")
    print("args:", json.dumps(vars(args)))

    # ------------------------------------------------------------------ data
    X, Fe, y, g, meta_tr = load_cache(args.train_tag)
    rng = np.random.default_rng(args.seed)
    if args.n_train:   # random subset: the cache is stored per process, so a head slice is all signal
        sub = np.sort(rng.choice(len(X), args.n_train, replace=False))
        X, Fe, y, g = X[sub], Fe[sub], y[sub], g[sub]
    perm = rng.permutation(len(X))
    n_val = int(len(perm) * args.val_frac)
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    Xtr = torch.from_numpy(X[tr_idx]).to(device)
    Ftr = torch.from_numpy(Fe[tr_idx]).to(device)
    ytr = torch.from_numpy(y[tr_idx]).to(device)
    gtr = torch.from_numpy(g[tr_idx].astype(np.int64)).to(device)
    Xva, Fva = torch.from_numpy(X[val_idx]).to(device), torch.from_numpy(Fe[val_idx]).to(device)
    yva, gva = y[val_idx], g[val_idx]
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  "
          f"(signal frac train {float(ytr.mean()):.3f})  GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ----------------------------------------------------------------- model
    model = build_model(args, Fe.shape[1]).to(device)
    n_params = count_params(model)
    print(f"model {args.model}: {n_params:,} trainable params")
    train_model = torch.compile(model) if args.compile else model
    ema = EMA(model, args.ema) if args.ema > 0 else None
    eval_model = ema.module if ema else model

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith("cls_token") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)
    steps_per_epoch = len(Xtr) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = int(steps_per_epoch * args.warmup_epochs)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    eps = args.label_smoothing
    tt_id = GROUP_ID["tt"]

    ckpt = out_dir / "best.pt"
    best, history = -1.0, []
    print(f"training {args.epochs} epochs x {steps_per_epoch} steps (batch {args.batch_size})", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_model.train()
        t0, tot = time.perf_counter(), torch.zeros((), device=device)
        order = torch.randperm(len(Xtr), device=device)
        for s in range(steps_per_epoch):
            idx = order[s * args.batch_size:(s + 1) * args.batch_size]
            xb, fb, yb, gb = Xtr[idx], Ftr[idx], ytr[idx], gtr[idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit = train_model(xb, fb)
            target = yb * (1 - eps) + 0.5 * eps
            w = torch.where(gb == tt_id, torch.full_like(yb, args.tt_weight), torch.ones_like(yb))
            loss = (F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            if ema:
                ema.update(model)
            tot += loss.detach()
        train_loss = float(tot) / steps_per_epoch
        va = predict(eval_model, Xva, Fva)
        auc, auc_tt = quick_auc(va, yva, gva)
        if not np.isfinite(auc):
            raise RuntimeError("validation AUC is not finite -- check the split / the loss")
        flag = ""
        if auc > best:
            best, flag = auc, "  *"
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
                        "epoch": epoch, "val_auc": auc, "args": vars(args)}, ckpt)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_auc=auc, val_auc_tt=auc_tt,
                            lr=sched.get_last_lr()[0], secs=time.perf_counter() - t0))
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  val_auc={auc:.5f}  "
              f"val_auc_tt={auc_tt:.5f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"({time.perf_counter() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB){flag}",
              flush=True)

    # ------------------------------------------------------------ evaluation
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    if ema:
        ema.module.load_state_dict(state["ema"])
    print(f"\nbest epoch {state['epoch']}  val AUC {state['val_auc']:.5f}")
    val_auc, val_pg, val_eff = auc_report(predict(eval_model, Xva, Fva), yva, gva,
                                          "held-out validation (10% of train1M)")
    # how much does the teacher over-fit its own training rows?  (matters for distillation)
    tr_logits = predict(eval_model, Xtr, Ftr)
    fit_auc, fit_tt = quick_auc(tr_logits, y[tr_idx], g[tr_idx])
    print(f"  train-slice AUC {fit_auc:.5f} (vs tt {fit_tt:.5f}); mean|logit| train {np.abs(tr_logits).mean():.3f}")

    Xev, Fev, yev, gev, _ = load_cache(args.eval_tag)
    ev_logits = predict(eval_model, torch.from_numpy(Xev), torch.from_numpy(Fev))
    eval_auc, eval_pg, eval_eff = auc_report(ev_logits, yev, gev, f"held-out EVAL slice ({args.eval_tag})")

    summary = dict(run=args.tag, model=args.model, params=n_params, args=vars(args),
                   best_epoch=state["epoch"], val_auc=val_auc, val_per_group=val_pg, val_eff=val_eff,
                   train_fit=dict(auc=fit_auc, auc_tt=fit_tt, mean_abs_logit=float(np.abs(tr_logits).mean())),
                   eval_auc=eval_auc, eval_per_group=eval_pg, eval_eff=eval_eff,
                   history=history, train_meta=meta_tr)
    write_json(out_dir / "summary.json", summary)

    # ---------------------------------------------------------- soft targets
    del Xtr, Ftr, ytr, gtr, Xva, Fva
    torch.cuda.empty_cache()
    if not args.no_soft_targets:
        for tag in CACHE_TAGS:
            if tag == args.eval_tag:
                logits = ev_logits
            else:
                Xc, Fc, yc, gc, _ = load_cache(tag)
                logits = predict(eval_model, torch.from_numpy(Xc), torch.from_numpy(Fc))
                a, att = quick_auc(logits, yc, gc)
                print(f"  soft targets {tag}: AUC {a:.5f} (vs tt {att:.5f})")
            np.save(out_dir / f"logits_{tag}.npy", logits.astype(np.float32))
        if args.publish:
            publish(args.tag, summary)

    print("\n" + "=" * 66)
    print(f"  {args.tag}: EVAL AUC = {eval_auc:.5f}   vs QCD {eval_pg.get('QCD', 0):.5f}  "
          f"vs tt {eval_pg.get('tt', 0):.5f}  vs Wjets {eval_pg.get('Wjets', 0):.5f}   params {n_params:,}")
    print("=" * 66, flush=True)


def publish(run: str, summary: dict):
    src = RUNS / run
    for tag in CACHE_TAGS:
        shutil.copyfile(src / f"logits_{tag}.npy", HERE / f"soft_targets_{tag}.npy")
    write_json(HERE / "soft_targets_meta.json",
               dict(source_run=run, model=summary["model"], params=summary["params"],
                    eval_auc=summary["eval_auc"], eval_per_group=summary["eval_per_group"],
                    format="float32 teacher logits, one per cache row, same order as team/cache/<tag>/X.npy",
                    label_smoothing=summary["args"]["label_smoothing"]))
    print(f"  published soft targets from '{run}' -> {HERE}/soft_targets_*.npy")


if __name__ == "__main__":
    main()
