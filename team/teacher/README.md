# Teacher lane (Purdue AF A100, agent `hh4b`)

Unconstrained teachers for distillation into the FPGA student. Everything here consumes
**exactly** the student's inputs (`team/cache/<tag>/X.npy` = 16 candidates × 5 features,
`F.npy` = 11 event features); the teacher only adds *derived* per-candidate and pairwise
quantities computed on the fly from those tensors (`common.py`), so its logits are valid
soft targets for the student.

## Deliverables

| file | content |
|---|---|
| `soft_targets_train1M.npy` | float32 teacher logits, one per row of `team/cache/train1M`, same order |
| `soft_targets_train300k.npy` | same for `train300k` (599,999 rows) |
| `soft_targets_eval100k.npy` | same for `eval100k` (200,000 rows) |
| `soft_targets_meta.json` | which run produced them, its eval AUCs, label smoothing used |

Student score = `sigmoid(logit)`; for KD at temperature T use `sigmoid(logit / T)`.
The teacher was trained with label smoothing 0.05, so its probabilities saturate around
0.975 / 0.025 rather than 1 / 0.

## Code

- `common.py` — cache I/O, inversion of `data.preprocess`, rich per-candidate features
  (ln pt/HT, ln E, Δφ/Δη to the leading candidate, |dxy|), ParT pairwise features
  (ln ΔR, ln kT, ln z, ln m²), AUC report identical to `team/train.py`.
- `models.py` — `BigDeepSet` (φ 128-64-32, mean+max pool, ρ 256-128-64) and `ParTLite`
  (d=128, 4 particle-attention blocks × 8 heads with pairwise attention bias, 2 class-attention
  blocks, concat mean-pooled tokens + event-feature MLP, MLP head).
- `train_teacher.py` — AdamW, warm-up + cosine, label smoothing 0.05, bf16 autocast, EMA,
  `torch.compile`; caps itself at 50 % of the GPU (shared card). Writes `runs/<tag>/`.
- `ensemble.py` — mean of member logits, evaluate, `--publish` to the files above.

```bash
PY=/work/users/das214/fastml26/venv/bin/python
$PY train_teacher.py --model deepset --tag ds_big_s0 --epochs 40 --lr 2e-3 --wd 0.01 --ema 0.999 --publish
$PY train_teacher.py --model part --tag part_s0 --epochs 50 --ema 0.999 --compile
$PY ensemble.py --runs part_s0 part_s1 part_s2 --publish
```

## Results

See the "Teacher lane" section of `../RESULTS.md`.
