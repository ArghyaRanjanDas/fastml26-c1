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

## Teacher lane on the Purdue AF A100 (agent `hh4b` there) — added by the orchestrator
Purpose: train the best possible **teacher** with no FPGA constraint, and hand c1 soft targets.
- Inputs: the same caches c1 built (`team/cache/{train1M,train300k,eval100k}`, X.npy = [N,16,5], F.npy = 11 event feats, y.npy), copied to `/work/users/das214/fastml26/team/cache`. Same preprocessing as c1 (see team/data.py) — the student must see identical inputs.
- Architectures, in order: (1) big DeepSet (φ 128-64-32, ρ 256-128-64, mean+max pooling) as a floor;
  (2) **Particle-Transformer-lite**: per-candidate embedding d=128, 4 attention blocks, 8 heads, with a pairwise
  interaction bias from (ΔR, ln m², ln kT) of candidate pairs — the ParT recipe that is SOTA for jet tagging;
  class-attention pooling; concat event feats; MLP head. (3) ensemble of the best 3 seeds.
- Training: 1M+1M, AdamW, cosine LR, 40-60 epochs, label smoothing 0.05, mixed precision. Report AUC overall and vs tt.
- Deliverable for c1: `team/teacher/soft_targets_train1M.npy` (teacher logits, float32, same row order as the cache)
  + `soft_targets_train300k.npy`, `soft_targets_eval100k.npy`, plus the teacher's RESULTS.md row. Push to `team` main.
- Optional if time: a teacher with 32 candidates (privileged information) — the student still uses 16.
