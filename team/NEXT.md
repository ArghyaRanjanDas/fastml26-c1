# Round 3 (from the orchestrator, 2026-09-03T21:11Z)
Weak spot: tt background (AUC vs tt 0.749 vs QCD 0.931, W+jets 0.971). Fitting candidate: B1e_16p (2,057 params, 0.8838).
1. Teacher -> student distillation: train an unconstrained teacher (e.g. phi 128-64-32, rho 256-128-64, 16 particles, event feats, 1M+1M events, 40+ epochs; optionally add a second pooling (max) alongside mean). Then distill into the B1e_16p shape with soft targets (KL at T=2-4, mixed with BCE). Report student AUC vs the 0.8838 from scratch.
2. tt focus: report the background mixture used; try tt-upweighting (e.g. tt x2 in the loss) and a 1-vs-tt diagnostic; add tt-sensitive event features only if they help the tt AUC without hurting overall.
3. QAT in ~/hlsenv (QKeras): rebuild the best student in Keras (Conv1D k=1 phi, GAP, concat, Dense rho), train quantization-aware at 8-bit and 6-bit weights/activations; export .h5 + json to team/export/qat_*; report AUC loss vs float.
4. Append RESULTS.md rows; push team main after each step.
