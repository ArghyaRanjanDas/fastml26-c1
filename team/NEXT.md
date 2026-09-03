# Round 3 (from the orchestrator, 2026-09-03T21:11Z)
Weak spot: tt background (AUC vs tt 0.749 vs QCD 0.931, W+jets 0.971). Fitting candidate: B1e_16p (2,057 params, 0.8838).
1. Teacher -> student distillation: train an unconstrained teacher (e.g. phi 128-64-32, rho 256-128-64, 16 particles, event feats, 1M+1M events, 40+ epochs; optionally add a second pooling (max) alongside mean). Then distill into the B1e_16p shape with soft targets (KL at T=2-4, mixed with BCE). Report student AUC vs the 0.8838 from scratch.
2. tt focus: report the background mixture used; try tt-upweighting (e.g. tt x2 in the loss) and a 1-vs-tt diagnostic; add tt-sensitive event features only if they help the tt AUC without hurting overall.
3. QAT in ~/hlsenv (QKeras): rebuild the best student in Keras (Conv1D k=1 phi, GAP, concat, Dense rho), train quantization-aware at 8-bit and 6-bit weights/activations; export .h5 + json to team/export/qat_*; report AUC loss vs float.
4. Append RESULTS.md rows; push team main after each step.

## Ownership (teammates are exploring independently — assume no contributions from them)
- **c1** (GPU): teacher training, distillation into the B1e_16p shape, QKeras QAT + exports. Owns `train.py`, `models.py`, `export/`.
- **c2** (CPU only — `CUDA_VISIBLE_DEVICES=""`): the tt̄ problem. Owns `team/physics/`.
  Goal: find what separates HH→4b from tt̄ at trigger level, measured as AUC vs tt̄ (currently 0.749).
  Ideas in priority order: (a) semi/fully-leptonic tt̄ carries a lepton + MET — is there an isolated
  high-pT candidate or a MET-like imbalance among the 16 PUPPI candidates? (b) cluster candidates into
  jets (anti-kT-like greedy, R=0.4) and form the two best bb pair masses — HH has two ~125 GeV pairs,
  tt̄ has a W (80) + top (173) structure; (c) dxy-based b-likeness per jet; (d) angular structure
  (ΔR between pairs, sphericity). Deliver: a ranked feature table (AUC vs tt̄ each, alone and added
  to B1e_16p's 11 features), the top 3 implemented in data.py behind a flag, and a RESULTS.md row
  for B1e_16p + new features. Coordinate through git; never edit c1's files without a note.
