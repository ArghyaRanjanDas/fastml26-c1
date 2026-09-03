# Data pipeline notes (`data.py`)

* Only the leaf columns `L1T_PUPPIPart.{pt,eta,phi,dxy}` and `label` are read out of
  parquet. The `L1T_PUPPIPart` struct has 14 subfields; selecting leaves instead of
  the whole struct is ~4× faster.
* Every event in this dataset has ≥415 particles, already sorted by descending pT,
  so keeping the leading 16 is pure truncation. The zero-pad path is kept for generality.
* Per-particle preprocessing uses **fixed constants**, not min/max fitted on the loaded
  batch as the intro notebook does: `log1p(pt)/8`, `eta/4`, `clip(dxy,±2)/2`, `cos φ`,
  `sin φ`. Train and inference must agree, and the firmware wants everything in ~[-1,1].
* Background mixture defaults to an even split over the QCD / tt / W+jets groups. Real
  trigger rates are QCD-dominated; the even split stops the classifier from ignoring tt
  and W, and per-group AUCs are always reported so the mixture never hides anything.

## Event-level features (11, `EVENT_FEATURES`)

`ht`, `lead_pt1..4`, `n_cand`, `sum_abs_dxy`, `max_abs_dxy`, `mean_abs_dxy`, `m2`, `m4`
— the last two are the invariant masses of the leading 2 and leading 4 candidates in the
massless approximation, computed from physical units before the per-particle scaling.
Verified against the analytic back-to-back case (pT 30 & 20 at η=0 → m = 48.990).

Each is transformed (`log1p` or linear) and then **standardized** with frozen constants
(`EVENT_STANDARDIZE`, re-measurable with `python data.py --fit-event-norm`). The first
version squashed them to [0,1] instead, which left `ht` at mean 0.69 / std 0.06 and
measurably hurt the AUC.

Two things worth knowing:

* **`n_cand` is dead.** It is identically 16 in every event of every process (measured
  std exactly 0), because every event has ≥415 candidates. It is kept because it stops
  being dead the moment `n_particles` drops or a pT threshold is applied, and it is
  emitted as 0 rather than dividing by zero.
* **Event features are always computed from the full 16-candidate list**, even when the
  particle branch is fed fewer (`--n-particles-use 8`). They are event-level quantities
  an L1 trigger already has, and computing them once per event costs nothing in firmware
  — unlike φ, which is replicated per particle.
