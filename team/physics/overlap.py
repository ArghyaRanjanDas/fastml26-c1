"""Do any events in a train cache also appear in the eval slice?

`train/` and `eval/` are different directories, but that is an argument about
file paths, not about events. The parquet files carry `source_file` and
`source_row`, which identify the row in the upstream COLLIDE2V dataset each event
came from, so the question can be settled exactly: take the (source_file,
source_row) pairs of the events a cache actually consumed and intersect them with
the eval slice's.

`stream_process` reads a process directory in sorted filename order and stops at
its budget, so "the events consumed" is the first N rows of that concatenation --
which is what this reproduces.

  python overlap.py --train-events 4000000 --eval-events 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import DATA_ROOT, SIGNAL, BACKGROUND   # noqa: E402

COLS = ["source_file", "source_row"]


def ids(split: str, directory: str, budget: int):
    """(source_file, source_row) of the first `budget` events, as one int64 key."""
    out, seen = [], 0
    d = DATA_ROOT / split / directory
    for path in sorted(d.glob(f"{d.name}_*.parquet")):
        if seen >= budget:
            break
        for b in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=COLS):
            a = ak.from_arrow(b)
            f = ak.to_numpy(a["source_file"]).astype(np.int64)
            r = ak.to_numpy(a["source_row"]).astype(np.int64)
            take = min(len(f), budget - seen)
            # source_row is int16; shift by 2^16 with an offset so negatives are safe
            out.append((f[:take] << 17) | (r[:take] + 65536))
            seen += take
            if seen >= budget:
                break
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def budgets(n_signal: int, n_background: int):
    b = [(p, n_signal) for p in SIGNAL]
    b += [(p, int(round(n_background * p.weight / 3.0))) for p in BACKGROUND]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-signal", type=int, default=4_000_000)
    ap.add_argument("--train-background", type=int, default=4_000_000)
    ap.add_argument("--eval-signal", type=int, default=100_000)
    ap.add_argument("--eval-background", type=int, default=100_000)
    ap.add_argument("--only", default=None, help="restrict to one process directory")
    a = ap.parse_args()

    tb = dict((p.directory, n) for p, n in budgets(a.train_signal, a.train_background))
    eb = dict((p.directory, n) for p, n in budgets(a.eval_signal, a.eval_background))

    rows = []
    print(f"{'process':<46s} {'train kept':>11s} {'eval kept':>10s} {'dup in train':>13s} {'overlap':>8s}")
    for proc in SIGNAL + BACKGROUND:
        d = proc.directory
        if a.only and a.only not in d:
            continue
        t = ids("train", d, tb[d])
        e = ids("eval", d, eb[d])
        dup = int(len(t) - len(np.unique(t)))
        n_over = int(np.isin(e, t, assume_unique=False).sum())
        rows.append(dict(process=d, group=proc.group, train_kept=int(len(t)),
                         train_requested=int(tb[d]), eval_kept=int(len(e)),
                         train_internal_duplicates=dup, overlap=n_over))
        print(f"{d:<46s} {len(t):>11,} {len(e):>10,} {dup:>13,} {n_over:>8,}", flush=True)

    (HERE / "overlap.json").write_text(json.dumps(rows, indent=1))
    tot = sum(r["overlap"] for r in rows)
    print(f"\ntotal overlapping events: {tot}")


if __name__ == "__main__":
    main()
