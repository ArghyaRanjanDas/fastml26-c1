"""Average the logits of several runs, evaluate, and (optionally) publish as soft targets.

  python ensemble.py --runs part_s0 part_s1 part_s2 --publish
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import CACHE_TAGS, HERE, RUNS, load_cache, auc_report, quick_auc, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or "ens_" + "+".join(args.runs)

    per_cache = {}
    for tag in CACHE_TAGS:
        stack = np.stack([np.load(RUNS / r / f"logits_{tag}.npy") for r in args.runs])
        per_cache[tag] = stack.mean(0).astype(np.float32)

    Xev, Fev, yev, gev, _ = load_cache(args.eval_tag)
    members = {}
    for r in args.runs:
        a, att = quick_auc(np.load(RUNS / r / f"logits_{args.eval_tag}.npy"), yev, gev)
        members[r] = dict(eval_auc=a, eval_auc_tt=att)
        print(f"  {r:<24s} eval AUC {a:.5f}  vs tt {att:.5f}")
    auc, pg, eff = auc_report(per_cache[args.eval_tag], yev, gev, f"ensemble of {len(args.runs)} ({name})")

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    for tag, logits in per_cache.items():
        np.save(out / f"logits_{tag}.npy", logits)
    summary = dict(run=name, model="ensemble", members=members, eval_auc=auc, eval_per_group=pg, eval_eff=eff)
    write_json(out / "summary.json", summary)
    if args.publish:
        for tag, logits in per_cache.items():
            np.save(HERE / f"soft_targets_{tag}.npy", logits)
        write_json(HERE / "soft_targets_meta.json",
                   dict(source_run=name, model="ensemble (mean of member logits)", members=members,
                        eval_auc=auc, eval_per_group=pg,
                        format="float32 teacher logits, one per cache row, same order as team/cache/<tag>/X.npy"))
        print(f"  published -> {HERE}/soft_targets_*.npy")


if __name__ == "__main__":
    main()
