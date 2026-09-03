# team/ — FastML26 Challenge 1 working code

| file | what it does |
|---|---|
| `data.py` | Streams capped event samples out of the 118 GB parquet dataset and caches them as `.npy`. Never loads a full process. |
| `models.py` | Binary classifiers (DeepSet baseline, small MLP). One logit, `BCEWithLogitsLoss`. |
| `train.py` | Trains on the A10, reports eval-slice AUC + per-background AUC + params + latency. |
| `RESULTS.md` | Running table of every experiment. |

## Data pipeline notes

* Only the leaf columns `L1T_PUPPIPart.{pt,eta,phi,dxy}` and `label` are read out of
  parquet. The `L1T_PUPPIPart` struct has 14 subfields; selecting leaves instead of
  the whole struct is ~4× faster.
* Every event in this dataset has ≥415 particles, already sorted by descending pT,
  so keeping 16 is pure truncation. The zero-pad path is kept for generality.
* Preprocessing uses **fixed constants**, not min/max fitted on the loaded batch as
  the intro notebook does: `log1p(pt)/8`, `eta/4`, `clip(dxy,±2)/2`, `cos φ`, `sin φ`.
  Train and inference must agree, and the FPGA build later wants everything inside
  ~[-1,1] for quantization.
* Background mixture defaults to an even split over the QCD / tt / W+jets groups.
  Real trigger rates are QCD-dominated; the even split stops the classifier from
  ignoring tt and W entirely, and per-group AUCs are reported separately so the
  mixture choice never hides anything.

## Usage

```bash
# build a cache explicitly (train.py does this automatically too)
python data.py --tag train300k --split train --n-signal 300000 --n-background 300000
python data.py --tag eval100k  --split eval  --n-signal 100000 --n-background 100000

# train + evaluate
python train.py --model deepset --epochs 20
```

Caches live in `team/cache/` and are gitignored.
