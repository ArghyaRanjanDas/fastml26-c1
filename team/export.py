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
from equalize import equalize
import input_spec
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


def build_from_summary(summary: dict, sd: dict | None = None) -> DeepSetPlus:
    """Rebuild the trained model.

    Widths are read off the checkpoint when one is given rather than assumed:
    the per-candidate channel count and the event-feature count both changed
    when c2's rich inputs landed, and older summaries do not record them.
    """
    phi_dims, rho_dims = tuple(summary["phi"]), tuple(summary["rho"])
    pool = summary.get("pool", "mean")
    n_feat = summary.get("n_features", 5)
    n_evt = summary.get("n_event_features", len(EVENT_FEATURES))
    if sd is not None:
        n_feat = sd["phi.0.weight"].shape[1]
        pooled = phi_dims[-1] * (2 if pool == "meanmax" else 1)
        n_evt = (sd["rho.0.weight"].shape[1] - pooled
                 if summary["use_event_features"] else n_evt)
    return DeepSetPlus(
        n_features=n_feat, n_event_features=n_evt,
        phi_dims=phi_dims, rho_dims=rho_dims,
        dropout=summary["dropout"], use_event_features=summary["use_event_features"],
        event_scale=summary.get("event_scale", 1.0), pool_norm=summary.get("pool_norm", False),
        pool=pool,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run tag under team/runs/")
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--cache-meta", default=None,
                    help="cache tag whose meta.json holds the extra-feature constants "
                         "(defaults to --eval-tag)")
    ap.add_argument("--n-sample", type=int, default=5000)
    ap.add_argument("--name", default=None,
                    help="override the export stem (default model_<params>); use when two "
                         "runs share a parameter count, e.g. the 8-particle variant")
    ap.add_argument("--sample-name", default="eval_sample.npz")
    ap.add_argument("--equalize", action="store_true",
                    help="cross-layer equalization. Fixes the range problem (max|W| 184 -> 9, "
                         "max|preact| 115 -> 9) but measurably trades it for underflow: 25%% of "
                         "rho0 weights end up below the ap_fixed<16,6> step. Off by default -- "
                         "per-layer precision (hls4ml granularity='name') is the better lever, "
                         "and QAT is the real fix. See quantsim.py.")
    ap.add_argument("--equalize-events", type=int, default=4096,
                    help="calibration events for the equalization activation ranges")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = json.loads((RUNS / f"{args.run}_summary.json").read_text())
    sd_in = torch.load(RUNS / f"{args.run}_best.pt", map_location="cpu")
    model = build_from_summary(summary, sd_in)
    model.load_state_dict(sd_in)
    model.eval()

    flat = fold(model).eval()
    n_params = count_params(flat)
    npart = summary.get("n_particles_use") or summary["n_particles"]
    use_evt = summary["use_event_features"]

    # ------------------------------------------------- data + closure check
    X, F, y, g, cache_meta = load_cache(args.eval_tag)
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

    if args.equalize:
        n_cal = min(args.equalize_events, len(xt))
        print("cross-layer equalization (function-preserving, no retraining):")
        equalize(flat, xt[:n_cal], ft[:n_cal], use_evt)
        with torch.no_grad():
            s_eq = torch.sigmoid(flat(xt, ft if use_evt else None)).numpy()
        eq_diff = float(np.abs(s_eq - s_orig).max())
        print(f"  equalization check: max|equalized - original| = {eq_diff:.3e}")
        assert eq_diff < 1e-4, "equalization changed the function -- refusing to export"
        s_flat, max_diff = s_eq, max(max_diff, eq_diff)

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
        n_features=int(X.shape[2]),
        n_particles=npart,
        n_event_features=int(F.shape[1]) if use_evt else 0,
        # --- everything else needed to reproduce the model exactly --------
        run=args.run,
        params=n_params,
        eval_auc=summary["eval_auc"],
        architecture=dict(
            phi_activation="relu", rho_activation="relu", output_activation="sigmoid",
            pooling=("mean+max concatenated (GlobalAveragePooling1D + GlobalMaxPooling1D)"
                     if summary.get("pool") == "meanmax"
                     else "mean (GlobalAveragePooling1D over particles)"),
            event_features_concat="after pooling" if use_evt else None,
            batchnorm="folded into the first rho Linear; none to synthesize",
            equalized=args.equalize,
            equalization="per-channel cross-layer scaling (ReLU positive homogeneity); "
                         "function-preserving, keeps weights/activations inside ap_fixed range",
            layer_order=keys,
        ),
        particle_features=[f for f, _ in
                           [(c, None) for c in cache_meta.get(
                               "particle_channels",
                               ["log_pt", "eta", "dxy", "cos_phi", "sin_phi"])]],
        particle_norm=dict(pt_log_scale=PT_LOG_SCALE, eta_scale=ETA_SCALE, dxy_clip=DXY_CLIP),
        particle_ordering="leading n_particles candidates by descending pT",
        event_feature_names=(list(cache_meta.get("event_features", EVENT_FEATURES))
                             if use_evt else []),
        event_feature_norm=(
            {n: dict(transform=EVENT_TRANSFORM[n], mean=EVENT_STANDARDIZE[n][0],
                     std=EVENT_STANDARDIZE[n][1]) for n in EVENT_FEATURES}
            if use_evt else {}),
        event_feature_clip=EVENT_CLIP if use_evt else None,
        event_features_computed_from_n_candidates=16 if use_evt else None,
        fold_check_max_abs_diff=max_diff,
        particle_channels=list(cache_meta.get(
            "particle_channels", ["log_pt", "eta", "dxy", "cos_phi", "sin_phi"])),
        pool=summary.get("pool", "mean"),
        input_spec=input_spec.build(
            npart, list(cache_meta.get("event_features", EVENT_FEATURES)) if use_evt else [],
            json.loads((Path(__file__).resolve().parent / "cache" /
                        (args.cache_meta or args.eval_tag) / "meta.json").read_text())),
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
