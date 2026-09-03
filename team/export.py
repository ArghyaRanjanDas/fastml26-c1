"""Export a trained model for the hls4ml / Vitis HLS path.

Writes, per the contract in team/fpga/README.md:
  team/export/model_<params>.json   architecture + preprocessing constants
  team/export/model_<params>.pt     state_dict, Linear layers in phi..rho..out order
  team/export/eval_sample.npz       5000 preprocessed eval inputs + labels + scores

The trained model carries a BatchNorm on the pooled vector.  `synth.py` maps
`*.weight` keys positionally onto Keras Conv1D/Dense layers, so a BatchNorm in
the state_dict would both break that mapping and ask the firmware to synthesize
something it does not need to.  At inference BatchNorm is a fixed per-channel
affine, so we **fold it exactly into the first rho Linear** and export a
BatchNorm-free model.  The fold is verified to machine precision before writing.

  python export.py --run C1_8p
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import (EVENT_FEATURES, EVENT_STANDARDIZE, EVENT_TRANSFORM, EVENT_CLIP,
                  PT_LOG_SCALE, ETA_SCALE, DXY_CLIP, load_cache)
from models import DeepSetPlus, count_params

HERE = Path(__file__).resolve().parent
RUNS, EXPORT = HERE / "runs", HERE / "export"


def fold(model: DeepSetPlus) -> DeepSetPlus:
    """Return an equivalent model with the pooled BatchNorm and event_scale folded away.

    BatchNorm at inference is y = a*x + b with a = gamma/sqrt(var+eps),
    b = beta - a*running_mean.  The first rho Linear sees concat([a*h+b, s*f]),
    so scaling its pooled columns by a, its event columns by s, and adding
    W[:, :P] @ b to the bias reproduces it exactly with no BatchNorm.
    """
    flat = DeepSetPlus(
        n_features=model.phi[0].in_features,
        n_event_features=model.n_event_features if model.use_event_features else 0,
        phi_dims=tuple(l.out_features for l in model.phi if isinstance(l, torch.nn.Linear)),
        rho_dims=tuple(l.out_features for l in model.rho if isinstance(l, torch.nn.Linear)),
        dropout=0.0, pool=model.pool, use_event_features=model.use_event_features,
        event_scale=1.0, pool_norm=False,
    )
    flat.load_state_dict({k: v for k, v in model.state_dict().items()
                          if not k.startswith("norm.")}, strict=False)

    P = model.pooled_dim
    W = model.rho[0].weight.data.clone()
    c = model.rho[0].bias.data.clone()

    if isinstance(model.norm, torch.nn.BatchNorm1d):
        bn = model.norm
        a = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
        b = bn.bias.data - a * bn.running_mean.data
        c = c + W[:, :P] @ b
        W[:, :P] = W[:, :P] * a
    if model.use_event_features and model.event_scale != 1.0:
        W[:, P:] = W[:, P:] * model.event_scale

    flat.rho[0].weight.data, flat.rho[0].bias.data = W, c
    return flat


def build_from_summary(summary: dict) -> DeepSetPlus:
    return DeepSetPlus(
        n_features=5, n_event_features=len(EVENT_FEATURES),
        phi_dims=tuple(summary["phi"]), rho_dims=tuple(summary["rho"]),
        dropout=summary["dropout"], use_event_features=summary["use_event_features"],
        event_scale=summary.get("event_scale", 1.0), pool_norm=summary.get("pool_norm", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag under team/runs/")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--n-sample", type=int, default=5000)
    ap.add_argument("--name", default=None,
                    help="override the export stem (default model_<params>); use when two "
                         "runs share a parameter count, e.g. the 8-particle variant")
    ap.add_argument("--sample-name", default="eval_sample.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = json.loads((RUNS / f"{args.run}_summary.json").read_text())
    model = build_from_summary(summary)
    model.load_state_dict(torch.load(RUNS / f"{args.run}_best.pt", map_location="cpu"))
    model.eval()

    flat = fold(model).eval()
    n_params = count_params(flat)
    npart = summary.get("n_particles_use") or summary["n_particles"]
    use_evt = summary["use_event_features"]

    # ------------------------------------------------- data + closure check
    X, F, y, g, _ = load_cache(args.eval_tag)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n_sample, len(X)), replace=False)
    Xs, Fs, ys = X[idx][:, :npart], F[idx], y[idx]
    xt, ft = torch.from_numpy(Xs), torch.from_numpy(Fs)

    with torch.no_grad():
        s_orig = torch.sigmoid(model(xt, ft if use_evt else None)).numpy()
        s_flat = torch.sigmoid(flat(xt, ft if use_evt else None)).numpy()
    max_diff = float(np.abs(s_orig - s_flat).max())
    print(f"BatchNorm fold check: max|folded - original| = {max_diff:.3e}")
    assert max_diff < 1e-5, "fold is not exact -- refusing to export"

    sd = flat.state_dict()
    keys = [k for k in sd if k.endswith("weight")]
    print(f"exported weight keys (synth.py maps these positionally): {keys}")

    EXPORT.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"model_{n_params}"
    torch.save(sd, EXPORT / f"{stem}.pt")

    spec = dict(
        # --- keys required by team/fpga/README.md -------------------------
        phi=list(summary["phi"]),
        rho=list(summary["rho"]),
        n_features=5,
        n_particles=npart,
        n_event_features=len(EVENT_FEATURES) if use_evt else 0,
        # --- everything else needed to reproduce the model exactly --------
        run=args.run,
        params=n_params,
        eval_auc=summary["eval_auc"],
        architecture=dict(
            phi_activation="relu", rho_activation="relu", output_activation="sigmoid",
            pooling="mean (GlobalAveragePooling1D over particles)",
            event_features_concat="after pooling" if use_evt else None,
            batchnorm="folded into the first rho Linear; none to synthesize",
            layer_order=keys,
        ),
        particle_features=["log1p(pt)/8", "eta/4", "clip(dxy,+-2)/2", "cos(phi)", "sin(phi)"],
        particle_norm=dict(pt_log_scale=PT_LOG_SCALE, eta_scale=ETA_SCALE, dxy_clip=DXY_CLIP),
        particle_ordering="leading n_particles candidates by descending pT",
        event_feature_names=list(EVENT_FEATURES) if use_evt else [],
        event_feature_norm=(
            {n: dict(transform=EVENT_TRANSFORM[n], mean=EVENT_STANDARDIZE[n][0],
                     std=EVENT_STANDARDIZE[n][1]) for n in EVENT_FEATURES}
            if use_evt else {}),
        event_feature_clip=EVENT_CLIP if use_evt else None,
        event_features_computed_from_n_candidates=16 if use_evt else None,
        fold_check_max_abs_diff=max_diff,
    )
    (EXPORT / f"{stem}.json").write_text(json.dumps(spec, indent=2))

    arrays = dict(X=Xs, y=ys, scores=s_flat, particles=Xs)
    if use_evt:
        arrays["F"] = Fs
        arrays["event"] = Fs
    np.savez_compressed(EXPORT / args.sample_name, **arrays)

    from sklearn.metrics import roc_auc_score
    print(f"\nexported '{args.run}' ({n_params:,} params, {npart} particles, "
          f"event_features={use_evt})")
    print(f"  {EXPORT / (stem + '.pt')}")
    print(f"  {EXPORT / (stem + '.json')}")
    print(f"  {EXPORT / args.sample_name}  ({len(Xs)} events, "
          f"AUC on this sample {roc_auc_score(ys, s_flat):.5f})")


if __name__ == "__main__":
    main()
