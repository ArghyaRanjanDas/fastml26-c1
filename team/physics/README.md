# c2 lane — the tt problem

The baseline `B1e_16p_1M` is 0.930 vs QCD and 0.972 vs W+jets, but **0.759 vs tt**.
tt is the background that looks like the signal: high HT, high multiplicity, real
b-quarks. This directory is the search for what still separates them at trigger level.

* `features.py` — 39 candidate features computed from the leading 16 PUPPI candidates.
  Reads nothing new: the cached `X` from `team/data.py` is an invertible transform of
  (pt, eta, phi, dxy), so `decode()` recovers physical units from the existing cache.
  Verified against the cached `m2`/`m4`/`ht` columns (mean |Δ| ~1e-4 in standardized units).
* `rank.py` — three ranking stages: `alone` (single-feature AUC vs tt), `marginal`
  (gain on top of B1e_16p's 11 event features, GBDT stand-in for rho), `greedy`
  (forward selection, so correlated features don't all claim the same handle).
* `cache/` — memoized feature arrays per cache tag (gitignored).

Jets are a greedy pT-ordered cone (R=0.4) rather than real anti-kT: with 16
candidates the two agree except in pathological overlaps, and the cone version is a
fixed number of vectorized passes over all events at once.
