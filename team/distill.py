"""Teacher -> student distillation into the deployable DeepSet shape.

The student is the shape we can actually synthesize (phi 32-16-8, rho 32-16,
16 particles); the teacher is unconstrained.  Teacher logits are computed once
up front and cached, so distillation costs the same per epoch as ordinary
training -- the teacher never runs inside the loop.

Binary distillation with temperature T: the soft target is sigmoid(z_t / T) and
the student is scored at z_s / T, with the KD term scaled by T^2 so its gradient
magnitude stays comparable to the hard-label term as T varies (Hinton et al.).

  python distill.py --teacher teacher_1M --tag student_T3 --temperature 3 --alpha 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

from data import load_cache, EVENT_FEATURES
from models import DeepSetPlus, count_params
from train import (OUT_DIR, auc_report, predict, measure_latency, dims, set_seed,
                   OFFICIAL_MIX)


def build_from_summary(summary: dict) -> DeepSetPlus:
    return DeepSetPlus(
        n_features=5, n_event_features=len(EVENT_FEATURES),
        phi_dims=tuple(summary["phi"]), rho_dims=tuple(summary["rho"]),
        dropout=summary["dropout"], use_event_features=summary["use_event_features"],
        event_scale=summary.get("event_scale", 1.0),
        pool_norm=summary.get("pool_norm", False), pool=summary.get("pool", "mean"),
    )


@torch.no_grad()
def teacher_logits(model, X, F, device, use_evt, batch_size=16384):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        fb = torch.from_numpy(F[i:i + batch_size]).to(device) if use_evt else None
        out.append(model(xb, fb).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", help="run tag of a teacher trained here (team/runs/)")
    ap.add_argument("--soft-targets", metavar="DIR",
                    help="use precomputed teacher logits from a directory holding "
                         "soft_targets_<cache tag>.npy + soft_targets_meta.json "
                         "(one logit per cache row, same order as cache/<tag>/X.npy). "
                         "Mutually exclusive with --teacher.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--phi", default="32,16,8")
    ap.add_argument("--rho", default="32,16")
    ap.add_argument("--n-particles-use", type=int, default=16)
    ap.add_argument("--event-scale", type=float, default=0.2)
    ap.add_argument("--no-event-features", action="store_true")
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the KD term")
    ap.add_argument("--train-tag", default="train1M")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="mean", choices=["mean", "sum", "meanmax"])
    ap.add_argument("--gpu-batches", action="store_true")
    ap.add_argument("--mixture", default="cache", choices=["cache", "official"],
                    help="'official' reweights background events so the effective mixture "
                         "matches the organizers' eval parquet (QCD .0912/W .3638/tt .5450)")
    ap.add_argument("--tt-weight", type=float, default=1.0,
                    help="multiply the hard-label term for tt events by this")
    ap.add_argument("--disagree-weight", action="store_true",
                    help="weight by 1 + |teacher - student| logit gap (batch-normalised)")
    args = ap.parse_args()

    if bool(args.teacher) == bool(args.soft_targets):
        ap.error("give exactly one of --teacher or --soft-targets")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, alpha = args.temperature, args.alpha

    Xtr, Ftr, ytr, gtr, meta_tr = load_cache(args.train_tag)
    Xev, Fev, yev, gev, meta_ev = load_cache(args.eval_tag)

    # ---------------------------------------------------------------- teacher
    if args.soft_targets:
        d = Path(args.soft_targets)
        tmeta = json.loads((d / "soft_targets_meta.json").read_text())
        # A derived cache (train1M_s) keeps its parent's row order, so the parent's
        # soft targets apply unchanged; fall back to derived_from when there is no
        # file for the derived tag itself.
        st = d / f"soft_targets_{args.train_tag}.npy"
        if not st.exists() and meta_tr.get("derived_from"):
            st = d / f"soft_targets_{meta_tr['derived_from']}.npy"
            print(f"  no targets for '{args.train_tag}'; using its parent "
                  f"'{meta_tr['derived_from']}' (same row order)")
        zt_tr = np.load(st).astype(np.float32).ravel()
        if len(zt_tr) != len(Xtr):
            raise SystemExit(f"soft targets have {len(zt_tr):,} rows but cache "
                             f"'{args.train_tag}' has {len(Xtr):,} -- wrong cache or stale targets")
        teacher_name = tmeta.get("source_run", str(d))
        teacher_auc = tmeta.get("eval_auc")
        print(f"teacher (precomputed) '{teacher_name}': {tmeta.get('params','?')} params, "
              f"eval AUC {teacher_auc:.5f}" if teacher_auc else f"teacher '{teacher_name}'")
        print(f"  per-group: {tmeta.get('eval_per_group')}")
    else:
        tsum = json.loads((OUT_DIR / f"{args.teacher}_summary.json").read_text())
        teacher = build_from_summary(tsum).to(device)
        teacher.load_state_dict(torch.load(OUT_DIR / f"{args.teacher}_best.pt", map_location=device))
        teacher.eval()
        t_np = tsum.get("n_particles_use") or tsum["n_particles"]
        t_evt = tsum["use_event_features"]
        teacher_name, teacher_auc = args.teacher, tsum["eval_auc"]
        print(f"teacher '{args.teacher}': {count_params(teacher):,} params, "
              f"{t_np} particles, pool={tsum.get('pool','mean')}, eval AUC {teacher_auc:.5f}")
        t0 = time.perf_counter()
        zt_tr = teacher_logits(teacher, Xtr[:, :t_np], Ftr, device, t_evt)
        print(f"cached teacher logits for {len(zt_tr):,} train events in {time.perf_counter()-t0:.1f}s")
        del teacher
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- student
    use_evt = not args.no_event_features
    npart = args.n_particles_use
    student = DeepSetPlus(
        n_features=Xtr.shape[2], n_event_features=Ftr.shape[1],
        phi_dims=dims(args.phi), rho_dims=dims(args.rho), dropout=0.0,
        use_event_features=use_evt, event_scale=args.event_scale, pool_norm=True,
        pool=args.pool,
    ).to(device)
    n_params = count_params(student)
    print(f"student: {n_params:,} params, phi={args.phi} rho={args.rho} "
          f"{npart} particles | T={T} alpha={alpha}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = int(len(perm) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tX = torch.from_numpy(Xtr[tr_idx][:, :npart]); tF = torch.from_numpy(Ftr[tr_idx])
    tY = torch.from_numpy(ytr[tr_idx]); tZ = torch.from_numpy(zt_tr[tr_idx])
    w = np.ones(len(tr_idx), dtype=np.float32)
    if args.mixture == "official":
        # the cache is an even three-way background split, so scale each group by
        # official_fraction / (1/3). Signal weight stays 1, preserving the
        # signal-to-background balance the model was tuned at.
        gg = gtr[tr_idx]
        for gid, name in ((0, "QCD"), (2, "tt"), (3, "Wjets")):
            w[(ytr[tr_idx] == 0) & (gg == gid)] = OFFICIAL_MIX[name] * 3.0
        print(f"  official mixture: QCD x{OFFICIAL_MIX['QCD']*3:.3f}, "
              f"W x{OFFICIAL_MIX['Wjets']*3:.3f}, tt x{OFFICIAL_MIX['tt']*3:.3f}")
    if args.tt_weight != 1.0:
        w[gtr[tr_idx] == 2] *= args.tt_weight
        print(f"  tt-weight {args.tt_weight}: {int((gtr[tr_idx] == 2).sum()):,} tt events")
    tW = torch.from_numpy(w)
    if args.gpu_batches:
        # The box is shared with the CPU lane; GPU-resident batching keeps the
        # cores free and is ~5 s/epoch here against ~9 s (or worse under load).
        gX, gF, gy, gz, gW = (tX.to(device), tF.to(device), tY.to(device),
                              tZ.to(device), tW.to(device))
        n_train, bs = len(gX), args.batch_size
        steps = n_train // bs

        def batches():
            perm = torch.randperm(n_train, device=device)
            for si in range(steps):
                i = perm[si * bs:(si + 1) * bs]
                yield gX[i], gF[i], gy[i], gz[i], gW[i]
    else:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tX, tF, tY, tZ, tW), batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
            persistent_workers=True)
        steps = len(loader)

        def batches():
            return loader
    Xva_t = torch.from_numpy(Xtr[val_idx][:, :npart])
    Fva_t = torch.from_numpy(Ftr[val_idx])
    yva, gva = ytr[val_idx], gtr[val_idx]
    Xev_t, Fev_t = torch.from_numpy(Xev[:, :npart]), torch.from_numpy(Fev)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps, eta_min=args.lr * 1e-3)
    hard = nn.BCEWithLogitsLoss()

    ckpt = OUT_DIR / f"{args.tag}_best.pt"
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        student.train()
        t0, tot, seen = time.perf_counter(), 0.0, 0
        for xb, fb, yb, zb, wb in batches():
            xb, fb = xb.to(device, non_blocking=True), fb.to(device, non_blocking=True)
            yb, zb = yb.to(device, non_blocking=True), zb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            zs = student(xb, fb if use_evt else None)
            kw = wb
            if args.disagree_weight:
                with torch.no_grad():
                    d = (zb - zs).abs()
                    kw = kw * (1.0 + d / d.mean().clamp(min=1e-6))
            kd = (Fn.binary_cross_entropy_with_logits(
                      zs / T, torch.sigmoid(zb / T), reduction="none") * (T * T) * kw).mean()
            hl = (Fn.binary_cross_entropy_with_logits(zs, yb, reduction="none") * wb).mean()
            loss = alpha * kd + (1.0 - alpha) * hl
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            seen += len(yb)

        val_auc = roc_auc_score(yva, predict(student, Xva_t, Fva_t, device))
        history.append(dict(epoch=epoch, loss=tot / seen, val_auc=float(val_auc)))
        flag = ""
        if val_auc > best:
            best, flag = val_auc, "  *"
            torch.save(student.state_dict(), ckpt)
        print(f"  epoch {epoch:2d}/{args.epochs}  loss={tot/seen:.4f}  "
              f"val_auc={val_auc:.5f}  ({time.perf_counter()-t0:.1f}s){flag}", flush=True)

    student.load_state_dict(torch.load(ckpt))
    ev = predict(student, Xev_t, Fev_t, device)
    eval_auc, per_group, eff, off_auc = auc_report(ev, yev, gev, f"student '{args.tag}' -- held-out EVAL slice")
    timing = measure_latency(student, Xev_t, Fev_t, device)

    pd_, d0, phi_macs = dims(args.phi), 5, 0
    for h in pd_:
        phi_macs += d0 * h
        d0 = h
    phi_macs *= npart

    summary = dict(run=args.tag, model="deepset_plus", distilled_from=teacher_name,
                   teacher_auc=teacher_auc, teacher_source="soft_targets" if args.soft_targets else "local",
                   temperature=T, alpha=alpha, phi=list(dims(args.phi)), rho=list(dims(args.rho)),
                   dropout=0.0, use_event_features=use_evt, event_scale=args.event_scale,
                   pool_norm=True, pool=args.pool, params=n_params, phi_macs=phi_macs,
                   n_particles=npart, n_particles_use=npart, eval_auc=eval_auc,
                   val_auc=float(best), per_background_auc=per_group, official_auc=off_auc, signal_eff=eff,
                   tt_weight=args.tt_weight, disagree_weight=args.disagree_weight,
                   mixture=args.mixture,
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   train_meta=meta_tr, eval_meta=meta_ev, timing=timing, history=history)
    (OUT_DIR / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT_DIR / f"{args.tag}_eval_scores.npy", ev)
    print(f"\n  {args.tag}: even-thirds {eval_auc:.5f}  OFFICIAL {off_auc:.5f}  "
          f"({n_params:,} params, T={T}, alpha={alpha})")


if __name__ == "__main__":
    main()
