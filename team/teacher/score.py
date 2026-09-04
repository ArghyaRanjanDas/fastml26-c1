"""Score cache tags with a trained teacher run and write logits_<tag>.npy into its run dir.

train_teacher.py only scores the three standard caches; this adds any other tag (train4M)
without retraining.  Also used to refresh logits after a cache is added.

  python score.py --run part4c_4M_off_s0 --tags train4M
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import RUNS, load_cache, quick_auc, binary_score, official_auc, auc_report
from models import MODELS
from train_teacher import build_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.3)
    a = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    device = torch.device("cuda")
    d = RUNS / a.run
    summary = json.loads((d / "summary.json").read_text())
    args = argparse.Namespace(**summary["args"])
    model = build_model(args, 11).to(device)
    state = torch.load(d / "best.pt", map_location=device)
    # evaluation used the EMA weights when EMA was on, so score with the same ones
    model.load_state_dict(state["ema"] if state.get("ema") else state["model"])
    model.eval()
    print(f"{a.run}: {summary['model']} n_classes={summary.get('n_classes', 1)} "
          f"best epoch {summary['best_epoch']}")

    for tag in a.tags:
        X, F, y, g, meta = load_cache(tag)
        logits = predict(model, torch.from_numpy(X), torch.from_numpy(F), a.batch_size)
        np.save(d / f"logits_{tag}.npy", logits.astype(np.float32))
        score = binary_score(logits) if logits.ndim == 2 else logits
        auc, pg, _ = auc_report(score, y, g, f"{a.run} on {tag}")
        print(f"  -> {d / f'logits_{tag}.npy'}  shape {logits.shape}  "
              f"even-thirds {auc:.5f}  official {official_auc(pg):.5f}")


if __name__ == "__main__":
    main()
