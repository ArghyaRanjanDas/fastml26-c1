# Challenge 1 — HH→4b vs. background, results log

Metric is the **binary ROC AUC** of the network score (signal = `HH_4b`, background =
QCD + tt + W+jets pooled), measured on a held-out slice built from the `eval/`
directory — never from any file used for training.

Default mixture: 300k signal, 300k background split evenly across the three
background groups (QCD / tt / W+jets). Eval slice: 100k signal + 100k background,
same mixture. Inputs are the 16 highest-pT `L1T_PUPPIPart` candidates,
5 features each (log-pT, η, dxy, cos φ, sin φ).

| model | params | eval AUC | train events | notes |
|---|---|---|---|---|
| DeepSet (intro baseline, binary) | 44,401 | **0.88261** | 600k (300k sig / 300k bkg) | φ 5→64→32→16, mean-pool, ρ 16→256→128→32→1. 20 epochs, AdamW 2e-3 + cosine. 0.21 µs/event GPU batched, 165 µs/event CPU batch-1. |

## Per-background breakdown (DeepSet baseline, eval slice)

| background | AUC |
|---|---|
| QCD_HT250toInf | 0.93109 |
| tt (3 channels) | 0.74616 |
| W+jets (2 channels) | 0.97059 |

Signal efficiency at 99% background rejection: 0.174; at 99.9%: 0.036.

**Read:** tt is by far the hardest background (0.746) — it also has real b-jets, so
the dxy/displacement handle that separates signal from QCD does not help there.
Improving the pooled AUC most likely means giving the network something that sees
event-level structure (multiplicity, HT, pairwise masses) rather than a wider ρ.

## Reproduce

```bash
cd ~/fastml26-hackathon/team
python train.py --model deepset --epochs 20
```

`train.py` builds the parquet caches on first run (~90 s for 600k train events)
and writes `runs/<tag>_summary.json` plus the raw eval scores.
