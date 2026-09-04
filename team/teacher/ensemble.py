"""Average the logits of several runs, evaluate, and (optionally) publish as soft targets.

  python ensemble.py --runs part_s0 part_s1 part_s2 --publish
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import CACHE_TAGS, HERE, RUNS, binary_score, load_cache, auc_report, official_auc, quick_auc, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval-tag", default="eval100k")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tags", nargs="+", default=list(CACHE_TAGS),
                    help="cache tags to average and (with --publish) write out")
    args = ap.parse_args()
    name = args.name or "ens_" + "+".join(args.runs)

    def member_score(run, tag):
        """A member's binary logit. A 4-class member contributes its HH-vs-background
        log-odds, which is the same quantity the binary head emits, so the mean is
        meaningful across head types."""
        z = np.load(RUNS / run / f"logits_{tag}.npy")
        return binary_score(z) if z.ndim == 2 else z

    per_cache = {}
    for tag in args.tags:
        stack = np.stack([member_score(r, tag) for r in args.runs])
        per_cache[tag] = stack.mean(0).astype(np.float32)

    Xev, Fev, yev, gev, _ = load_cache(args.eval_tag)
    members, n_params, label_smoothing = {}, 0, None
    for r in args.runs:
        a, att = quick_auc(member_score(r, args.eval_tag), yev, gev)
        ms = json.loads((RUNS / r / "summary.json").read_text())
        members[r] = dict(eval_auc=a, eval_auc_tt=att, model=ms["model"], params=ms["params"])
        n_params += ms["params"]
        label_smoothing = ms["args"]["label_smoothing"]
        print(f"  {r:<24s} eval AUC {a:.5f}  vs tt {att:.5f}")
    auc, pg, eff = auc_report(per_cache[args.eval_tag], yev, gev, f"ensemble of {len(args.runs)} ({name})")
    print(f"  official-mixture AUC: {official_auc(pg):.5f}")

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    for tag, logits in per_cache.items():
        np.save(out / f"logits_{tag}.npy", logits)
    summary = dict(run=name, model="ensemble", params=n_params, members=members,
                   eval_auc=auc, eval_per_group=pg, eval_eff=eff,
                   eval_auc_official=official_auc(pg), tags=list(args.tags))
    write_json(out / "summary.json", summary)
    if args.publish:
        for tag, logits in per_cache.items():
            np.save(HERE / f"soft_targets_{tag}.npy", logits)
        write_json(HERE / "soft_targets_meta.json",
                   dict(source_run=name, model="ensemble (mean of member logits)",
                        params=n_params, n_members=len(args.runs), members=members,
                        eval_auc=auc, eval_per_group=pg, eval_eff=eff,
                        label_smoothing=label_smoothing,
                        format="float32 teacher logits, one per cache row, same order as team/cache/<tag>/X.npy",
                        usage="student score = sigmoid(logit); for KD at temperature T use sigmoid(logit / T)"))
        print(f"  published -> {HERE}/soft_targets_*.npy")


if __name__ == "__main__":
    main()
