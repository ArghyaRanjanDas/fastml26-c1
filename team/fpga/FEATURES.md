# The preprocessing block, costed per feature

`model_2777_rich` synthesizes at **HLS AUC 0.906, 253k LUT / 1,692 DSP / 0.42 µs** — better
*and* smaller than the plain student (0.881, 319k LUT / 1,724 DSP). This file prices the
block that produces its inputs: the 11 per-candidate channels and the 19 event features,
each with the exact formula from raw `(pt, η, φ, dxy)` and an operation count.

Names and constants are those in `team/export/model_2777_rich.json`. Feature values come
from `team/data.py` (`--rich-particles --extra-features`); what each is worth in AUC is in
the c2 section of `team/RESULTS.md`.

**Cost columns**: `A/C` = adds, subtracts, compares, muxes, sign strips (LUT fabric, no
DSP) · `M` = multiplies · `LUT` = a table lookup (log, exp, cosh, sin/cos). Divisions are
counted separately because there are, deliberately, **none**.

## Two things that make the whole block cheap

1. **Every standardization folds away.** Each event feature is delivered as
   `clip((transform(raw) − μ) / σ, ±5)` and then multiplied by `event_scale` and fed
   straight into a Linear. `(x − μ)/σ · s` is affine, so μ, σ and the scale fold into that
   Linear's weights and bias at export time. What survives in firmware is **only the clip**:
   2 compares per feature, 38 in total. The same is true of the per-candidate `/8`, `/4`,
   `/2` scalings — all shifts, and foldable into φ's first layer.
2. **`log1p(pt)` is computed once.** Channel 0 is `log1p(pt)/8`, so any feature needing
   `log1p(pt_i)` — `lead_pt1..4`, and `iso_lead_pt`, which is `log1p` of a candidate's pT —
   is a shift of a value already on the wire. No second log.

## Per-candidate channels — cost **per candidate**, ×16

| ch | name | formula from raw | A/C | M | LUT | note |
|---|---|---|---|---|---|---|
| 0 | `log_pt` | `log1p(pt) / 8` | 0 | 0 | 1 | `/8` is a shift |
| 1 | `eta` | `η / 4` | 0 | 0 | 0 | shift |
| 2 | `dxy` | `clip(dxy, ±2) / 2` | 2 | 0 | 0 | two compares |
| 3 | `cos_phi` | `cos φ` | 0 | 0 | 1 | 0 if the L1 object already carries cos/sin |
| 4 | `sin_phi` | `sin φ` | 0 | 0 | 1 | ″ |
| 5 | `lnz` | `ln(pt / HT) / 4` | 1 | 0 | 0 | `(8·ch0 − lnHT)/4`; `lnHT` is one per-event LUT |
| 6 | `lnE` | `log1p(pt cosh η) / 8` | 1 | 0 | 1 | rewritten as `ch0 + LUT(ln cosh η)/8` — see note |
| 7 | `cos_dphi_lead` | `cos(φ − φ₁)` | 1 | 2 | 0 | `c_i c_1 + s_i s_1` |
| 8 | `sin_dphi_lead` | `sin(φ − φ₁)` | 1 | 2 | 0 | `s_i c_1 − c_i s_1` |
| 9 | `deta_lead` | `(η − η₁) / 2` | 1 | 0 | 0 | subtract + shift |
| 10 | `abs_dxy` | `\|ch2\|` | 1 | 0 | 0 | sign strip |
| | **per candidate** | | **8** | **4** | **4** | |
| | **×16 candidates** | | **128** | **64** | **64** | |

Plus, once per event for `lnz`: `HT = Σ pt` (**15 A**) and `ln HT` (**1 LUT**).

*Two rewrites are used above.* `8·ch0 = ln(1+pt)`, not `ln pt`; the difference is
`ln(1+1/pt) ≤ ln(1+1/4.4) = 0.207` since every kept candidate has pT ≥ 4.4 GeV, i.e.
≤ 0.05 in `lnz` units and ≤ 0.026 in `lnE` units, a positive constant-sign offset the
first Linear absorbs. And `ln(pt cosh η) = ln pt + ln cosh η`, which turns `lnE` from
(cosh LUT + multiply + log LUT) into (one LUT + one add). If bit-exactness with the
trained float model is wanted instead, `ln pt = LUT(ch0)` costs one more small table per
candidate and the naive `lnE` costs +16 M +16 LUT.

## Event features — cost **per event**

| # | name | formula | A/C | M | LUT | note |
|---|---|---|---|---|---|---|
| 0 | `ht` | `Σ pt` → log1p | 15 + 2 | 0 | 1 | |
| 1-4 | `lead_pt1..4` | `pt` of candidates 0-3 → log1p | 8 | 0 | 0 | `= 8·ch0` of those candidates |
| 5 | `n_cand` | count of filled slots | 0 | 0 | 0 | **identically 16 → σ = 0 → emitted as 0. A dead input; drop it and save a ρ column** |
| 6 | `sum_abs_dxy` | `Σ \|dxy\|` → log1p | 15 + 2 | 0 | 1 | |
| 7 | `max_abs_dxy` | `max \|dxy\|` → log1p | 15 + 2 | 0 | 1 | comparator tree |
| 8 | `mean_abs_dxy` | `sum_abs_dxy / 16` | 2 | 0 | 0 | shift — never a divider (`n_cand` is fixed) |
| 9 | `m2` | mass of leading 2, massless | 5 | 8 | 5 | see four-vector note |
| 10 | `m4` | mass of leading 4, massless | 11 | 12 | 1 | shares the four-vectors with `m2` |
| 11 | `iso_lead_pt` | pT of the most isolated hard candidate | ~755 | 390 | 0 | the ΔR table — see below |
| 12 | `n_iso` | # hard candidates with `iso < 0.15` | 31 | 0 | 0 | reuses that table; `0.15·pt` as shift-add |
| 13-18 | `p12..p34_lndR` | `ln ΔR` for the 6 leading-4 pairs | 30 | 12 | 6 | |
| | **total** | | **~895** | **422** | **15** | |

**`m2` / `m4` (rows 9-10).** Build massless four-vectors for the leading 4 candidates:
`E = pt cosh η`, `px = pt·c`, `py = pt·s`, `pz = pt sinh η` — 4 M and 2 LUTs each, 16 M and
8 LUTs for four candidates. `m² = E² − px² − py² − pz²` is 4 M + 3 A per mass, and
`log1p(√m²)` is one LUT that takes `m²` directly, so no square root. `m2` uses the first
two four-vectors, `m4` all four; the four-vectors are shared.

**`iso_lead_pt` / `n_iso` (rows 11-12)** — the only expensive part, and worth it: this one
scalar is +0.024 AUC vs tt on its own, more than all 24 leading-4 pair numbers together.

```
iso_i       = (Σ_{j≠i, ΔR(i,j) < 0.4} pt_j) / pt_i
iso_lead_pt = pt_k ,  k = argmin over { i : pt_i > 10 GeV } of iso_i
n_iso       = #{ i : pt_i > 10 GeV and iso_i < 0.15 }
```

Three rewrites keep it inside budget:

* **The cone test needs no ΔR, no √, no atan2 and no 2π wrap.** Use
  `ΔR'² = (η_i−η_j)² + 2(1 − cos Δφ_ij) < 0.16`, with `cos Δφ_ij = c_i c_j + s_i s_j` from
  channels 3-4. `2(1−cos x)` equals `x²` to fourth order (0.1578 vs 0.16 at the cone edge).
  **Measured, not assumed:** the two cone definitions give +0.0361 and +0.0365 AUC vs tt.
  Per unordered pair: 1 A (Δη), 1 M (square), 2 M (cos), 2 A, 1 compare.
  120 pairs → **360 M, 480 A/C**, fully parallel — one combinational stage, no search.
* **No divisions.** `iso_i < 0.15` becomes `Σpt_i < 0.15·pt_i`, and 0.15 ≈ 1/8 + 1/32 is
  two shifts and an add. The argmin over ratios becomes a cross-multiplied comparison,
  `Σpt_i·pt_j < Σpt_j·pt_i` — 2 M per comparison, 15 comparisons in a tournament tree,
  **30 M** and no divider anywhere.
* **No log on the output.** `log1p(iso_lead_pt)` is `8·ch0` of the winning candidate: a mux
  driven by the tournament, then the folded affine.

Cone pT accumulation is 16 accumulators × 15 conditional adds = **240 A**; the hard-candidate
mask is 16 compares. If only `log_pt` is on the wire and not `pt`, add 16 `expm1` LUTs.

**Do not shrink the search.** Allowing only the leading 8 (or 4) candidates to *be* the
isolated one halves (quarters) the table but costs real AUC vs tt: 0.7146 with all 16 seeds
→ 0.6910 with 8 → 0.6688 with 4. In leptonic tt the lepton is often not among the four
hardest candidates.

**`p_ij_lndR` (rows 13-18).** As exported these are the textbook
`½ ln(Δη² + Δφ²)` over the 6 pairs among the leading 4. With φ itself on the wire that is,
per pair: 1 A (Δη) + 1 M (square) + 1 A (Δφ) + 2 C + 1 A (2π wrap) + 1 M (square) + 1 A +
1 LUT (`½ln`) — **2 M, 5 A/C, 1 LUT**, six fixed pairs of known positions, no sort, no search.
If φ is *not* available and only `cos φ`/`sin φ` are, use the same `2(1−cos Δφ)` substitution
as the cone: 3 M, 3 A, 1 LUT per pair and no wrap logic. That variant measures the same
(+0.0158 vs +0.0156 AUC vs tt), and `data.py` can emit either.

## Totals, against the layers they feed

| block | A/C | M | LUT | divides | sequential depth |
|---|---|---|---|---|---|
| 11 per-candidate channels (×16) | 128 | 64 | 64 | 0 | 2 (needs HT and candidate 1 first) |
| per-event scalars, rows 0-10 | ~77 | 20 | 9 | 0 | 1 |
| isolation table + tournament (rows 11-12) | ~786 | 406 | 0 | 0 | 2 |
| 6 pair distances (rows 13-18) | 30 | 12 | 6 | 0 | 1 |
| **preprocessing total** | **~1,020** | **~500** | **~79** | **0** | **≤ 4 stages** |
| φ 32-16-8 over 11 channels × 16 | — | 15,872 | — | — | 3 |
| ρ 32-16-1 on 16 pooled + 19 event | — | ~1,700 | — | — | 3 |

**The preprocessing block is ~500 multiplies against φ's 15,872 — about 3% — with no
dividers and at most four combinational stages.** Against a synthesized 0.42 µs it is not
what threatens the 1 µs budget; the φ width is. Two cheaper φ variants that keep every one
of these features are in RESULTS.md (`c2_canon_narrow`, φ 24-12-8 → 10,368 MACs, *below*
the plain student's 12,800, at AUC 0.9015 / tt 0.8008).

Free saving available now: **`n_cand` is a dead input** (identically 16, σ = 0, emitted as
0). Dropping it removes a column from ρ's first Linear at exactly zero AUC cost.

## Rejected, with the reason

| candidate | verdict | why |
|---|---|---|
| `ln kT`, `ln m²` of the leading-4 pairs | drop | +0.000 on top of `ln ΔR` — same information in other coordinates |
| `ln z` of the leading-4 pairs | drop | −0.000. Worth nothing anywhere |
| full 16×16×4 pairwise block | drop | pooling all 120 pairs keeps only +0.009 of the +0.023 the explicit leading-6 pairs give; the value is in *which* pair, and 1,024 inputs do not fit |
| jet clustering (anti-kT / cone) | drop | every jet-derived feature (`dm_W`, `dm_top`, `m_bb`, `n_jets`) is worth ≤ +0.010, and clustering is 6 *sequential* passes over the pair table — the one thing the latency budget cannot absorb |
| seed-restricted isolation | drop | 2×/4× cheaper, −0.024/−0.046 AUC vs tt |
| dijet-pairing mass (`m_bb1`, `dm_higgs`) | drop | +0.001/+0.000. The candidates are particle-flow objects, not jets: two of the leading four routinely come from the same jet |
