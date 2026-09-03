"""Quantization-aware training for the deployable DeepSet.

Post-training rescaling cannot fix a dynamic range that training never
constrained (see fpga/FIXED-POINT.md).  QAT constrains it directly: weights and
activations are fake-quantized in the forward pass with a straight-through
estimator, so the optimizer sees the rounding/saturation error and works around it.

The fake quantizer reproduces `ap_fixed<W,I,AP_RND,AP_SAT>` exactly as
`quantsim.py` models it -- and quantsim is calibrated against real Vitis closure
numbers -- so the AUC reported here is the AUC the firmware should reach.

Integer widths are per-tensor (hls4ml `granularity="name"`), sized from the float
model's measured ranges and then frozen, and written into the export json as a
precision map so synth.py can apply exactly what was trained.

  python qat.py --init kd_best --bits 16 --epochs 12 --tag qat16
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import load_cache, EVENT_FEATURES
from models import DeepSetPlus, count_params
from train import OUT_DIR, auc_report, set_seed
from export import build_from_summary, fold

HERE = Path(__file__).resolve().parent
EXPORT = HERE / "export"


class _FakeQuant(torch.autograd.Function):
    """ap_fixed<W,I,AP_RND,AP_SAT> in the forward pass, identity gradient."""

    @staticmethod
    def forward(ctx, x, step, lo, hi):
        return torch.clamp(torch.round(x / step) * step, lo, hi)

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def fq(x, W: int, I: int):
    step = 2.0 ** (I - W)
    return _FakeQuant.apply(x, step, -(2.0 ** (I - 1)), 2.0 ** (I - 1) - step)


def ibits(v, headroom: int = 1) -> int:
    m = float(np.abs(v).max()) if isinstance(v, np.ndarray) else float(v.abs().max())
    return max(2, int(np.ceil(np.log2(m + 1e-12))) + 1 + headroom)


class QuantDeepSet(nn.Module):
    """Wraps a folded DeepSetPlus and runs it with fake quantization everywhere.

    Layer names match the Keras graph synth.py builds (phi0.., pool, concat,
    rho0.., score), so the emitted precision map drops straight into hls4ml.
    """

    def __init__(self, flat: DeepSetPlus, bits: int, ranges: dict):
        super().__init__()
        self.m = flat
        self.bits = bits
        self.I = dict(ranges)
        self.phis = [l for l in flat.phi if isinstance(l, nn.Linear)]
        self.rhos = [l for l in flat.rho if isinstance(l, nn.Linear)]
        self.use_evt = flat.use_event_features

    def forward(self, x, f=None):
        B = self.bits
        h = fq(x, B, self.I["input_particles"])
        for i, lyr in enumerate(self.phis):
            w = fq(lyr.weight, B, self.I[f"phi{i}_w"])
            b = fq(lyr.bias, B, self.I[f"phi{i}_b"])
            h = torch.relu(fq(torch.nn.functional.linear(h, w, b), B, self.I[f"phi{i}_out"]))
        h = fq(h.mean(dim=1), B, self.I["pool"])
        if self.use_evt:
            h = torch.cat([h, fq(f, B, self.I["input_event"])], dim=1)
        for i, lyr in enumerate(self.rhos):
            w = fq(lyr.weight, B, self.I[f"rho{i}_w"])
            b = fq(lyr.bias, B, self.I[f"rho{i}_b"])
            h = torch.relu(fq(torch.nn.functional.linear(h, w, b), B, self.I[f"rho{i}_out"]))
        w = fq(self.m.out.weight, B, self.I["score_w"])
        b = fq(self.m.out.bias, B, self.I["score_b"])
        return fq(torch.nn.functional.linear(h, w, b), B, self.I["score_out"]).squeeze(-1)


@torch.no_grad()
def measure_ranges(flat: DeepSetPlus, X, F, use_evt) -> dict:
    """Per-tensor integer widths from the float model's observed ranges."""
    r = {"input_particles": ibits(X), "input_event": ibits(F) if use_evt else 2}
    phis = [l for l in flat.phi if isinstance(l, nn.Linear)]
    rhos = [l for l in flat.rho if isinstance(l, nn.Linear)]
    h = X
    for i, lyr in enumerate(phis):
        r[f"phi{i}_w"], r[f"phi{i}_b"] = ibits(lyr.weight), ibits(lyr.bias)
        z = lyr(h)
        r[f"phi{i}_out"] = ibits(z)
        h = torch.relu(z)
    h = h.mean(dim=1)
    r["pool"] = ibits(h)
    if use_evt:
        h = torch.cat([h, F], dim=1)
    for i, lyr in enumerate(rhos):
        r[f"rho{i}_w"], r[f"rho{i}_b"] = ibits(lyr.weight), ibits(lyr.bias)
        z = lyr(h)
        r[f"rho{i}_out"] = ibits(z)
        h = torch.relu(z)
    r["score_w"], r["score_b"] = ibits(flat.out.weight), ibits(flat.out.bias)
    r["score_out"] = ibits(flat.out(h))
    return r


@torch.no_grad()
def qpredict(model, X, F, device, use_evt, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        fb = F[i:i + bs].to(device) if use_evt else None
        out.append(torch.sigmoid(model(xb, fb)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="run tag to start from (float student)")
    ap.add_argument("--bits", type=int, default=16, help="total width W of ap_fixed<W,I>")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-tag", default="train1M")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soft-targets", default="teacher",
                    help="keep distilling during QAT from these soft targets ('' to disable)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = json.loads((OUT_DIR / f"{args.init}_summary.json").read_text())
    model = build_from_summary(summary)
    model.load_state_dict(torch.load(OUT_DIR / f"{args.init}_best.pt", map_location="cpu"))
    flat = fold(model.eval()).eval().to(device)
    use_evt = flat.use_event_features
    npart = summary.get("n_particles_use") or summary["n_particles"]

    Xtr, Ftr, ytr, gtr, meta_tr = load_cache(args.train_tag)
    Xev, Fev, yev, gev, meta_ev = load_cache(args.eval_tag)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr))
    n_val = int(len(perm) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xva = torch.from_numpy(Xtr[val_idx][:, :npart]); Fva = torch.from_numpy(Ftr[val_idx])
    yva, gva = ytr[val_idx], gtr[val_idx]
    Xev_t = torch.from_numpy(Xev[:, :npart]); Fev_t = torch.from_numpy(Fev)

    cal = torch.from_numpy(Xtr[tr_idx][:8192][:, :npart]).to(device)
    calF = torch.from_numpy(Ftr[tr_idx][:8192]).to(device)
    ranges = measure_ranges(flat, cal, calF, use_evt)
    print(f"init '{args.init}' float AUC {summary['eval_auc']:.5f} | QAT at "
          f"ap_fixed<{args.bits},I> with per-tensor I:")
    print("  " + "  ".join(f"{k}={v}" for k, v in ranges.items()))

    qmodel = QuantDeepSet(flat, args.bits, ranges).to(device)
    print(f"pre-QAT (weights unchanged, quantized forward): "
          f"AUC {roc_auc_score(yev, qpredict(qmodel, Xev_t, Fev_t, device, use_evt)):.5f}")

    tensors = [torch.from_numpy(Xtr[tr_idx][:, :npart]), torch.from_numpy(Ftr[tr_idx]),
               torch.from_numpy(ytr[tr_idx])]
    use_kd = bool(args.soft_targets)
    if use_kd:
        z = np.load(Path(args.soft_targets) / f"soft_targets_{args.train_tag}.npy").astype(np.float32).ravel()
        tensors.append(torch.from_numpy(z[tr_idx]))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    opt = torch.optim.AdamW(qmodel.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader),
                                                       eta_min=args.lr * 1e-2)
    hard = nn.BCEWithLogitsLoss()
    T, alpha = args.temperature, args.alpha
    ckpt = OUT_DIR / f"{args.tag}_best.pt"
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        qmodel.train()
        tot = seen = 0
        for batch in loader:
            xb, fb, yb = (t.to(device, non_blocking=True) for t in batch[:3])
            opt.zero_grad(set_to_none=True)
            zs = qmodel(xb, fb if use_evt else None)
            loss = hard(zs, yb)
            if use_kd:
                zt = batch[3].to(device, non_blocking=True)
                kd = nn.functional.binary_cross_entropy_with_logits(
                    zs / T, torch.sigmoid(zt / T)) * (T * T)
                loss = alpha * kd + (1 - alpha) * loss
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(yb); seen += len(yb)
        va = roc_auc_score(yva, qpredict(qmodel, Xva, Fva, device, use_evt))
        flag = ""
        if va > best:
            best, flag = va, "  *"
            torch.save(flat.state_dict(), ckpt)
        print(f"  epoch {epoch:2d}/{args.epochs}  loss={tot/seen:.4f}  val_auc={va:.5f}{flag}",
              flush=True)

    flat.load_state_dict(torch.load(ckpt))
    qmodel = QuantDeepSet(flat, args.bits, ranges).to(device)
    ev = qpredict(qmodel, Xev_t, Fev_t, device, use_evt)
    eval_auc, per_group, eff = auc_report(ev, yev, gev,
                                          f"QAT '{args.tag}' ap_fixed<{args.bits},I> -- EVAL slice")

    prec = {}
    for k, I in ranges.items():
        prec[k] = f"ap_fixed<{args.bits},{I},AP_RND,AP_SAT>"
    out = dict(summary)
    out.update(run=args.tag, qat=True, bits=args.bits, integer_bits=ranges,
               precision_map=prec, init_from=args.init, eval_auc=eval_auc,
               float_eval_auc=summary["eval_auc"], per_background_auc=per_group,
               signal_eff=eff, val_auc=float(best), params=count_params(flat),
               n_particles=npart, n_particles_use=npart, pool_norm=False, event_scale=1.0)
    (OUT_DIR / f"{args.tag}_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n  {args.tag}: QAT AUC = {eval_auc:.5f} at ap_fixed<{args.bits},I>  "
          f"(float init {summary['eval_auc']:.5f})")


if __name__ == "__main__":
    main()
