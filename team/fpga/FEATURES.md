# The feature block, priced for firmware

Everything the round-3 student consumes beyond the raw candidate list, with the exact
formula, what it costs in an FPGA, and — where one exists — a rewrite that is cheaper in
fixed point and **measured** to lose nothing. Written for the Friday slide: this is the
argument that the feature block fits inside the 1 µs budget alongside φ and ρ.

Companion files: `team/physics/features.py` (event features), `team/physics/derived.py`
(per-candidate channels), `team/data.py` (the canonical pipeline, flags `--rich-particles`
and `--extra-features`), `team/RESULTS.md` (what each feature is worth).

## What arrives

Per candidate the L1 PUPPI stream gives **pt, η, φ, dxy**. The existing export contract
feeds the network five already-normalized channels, computed upstream of the model:

| ch | value | from |
|---|---|---|
| 0 | `u = log1p(pt) / 8` | log LUT |
| 1 | `e = η / 4` | shift |
| 2 | `d = clip(dxy, ±2) / 2` | clamp + shift |
| 3 | `c = cos φ` | LUT (or already in the object) |
| 4 | `s = sin φ` | LUT |

Candidates arrive **sorted by descending pt**, and every event in this dataset has 16 of
them with pt ≥ 4.4 GeV — both facts are used below.

Cost classes: **A** add/subtract/compare/shift · **M** multiply · **D** divide ·
**L** LUT (log, exp, cosh).

---

## 1. The six derived per-candidate channels (φ input 5 → 11)

Worth **+0.037 AUC vs tt, +0.0136 overall** in the 2,057-parameter student — the largest
single gain in round 3. All six are O(1) per candidate: no search, no clustering, no
sequential passes.

| ch | exact formula | naive cost | **fixed-point rewrite** | rewritten cost |
|---|---|---|---|---|
| 5 `lnz` | `ln(pt_i / HT) / 4`, `HT = Σ pt_j` | 16 A (sum) + 2 L + 1 D | `(8u_i − ln HT) / 4` — `8u_i` **is** `ln pt_i` to 0.21 (see note), `ln HT` is one LUT for the whole event, `/4` is a shift | 16 A + **1 L** + 16 A |
| 6 `lnE` | `log1p(pt_i cosh η_i) / 8` | 1 M + 2 L | `u_i + LUT(η_i)/8` with `LUT = ln cosh η` — because `ln(pt cosh η) = ln pt + ln cosh η` | **1 L + 1 A** per candidate, no multiply |
| 7 `cos Δφ` | `cos(φ_i − φ_1)` | trig | `c_i c_1 + s_i s_1` | **2 M + 1 A** |
| 8 `sin Δφ` | `sin(φ_i − φ_1)` | trig | `s_i c_1 − c_i s_1` | **2 M + 1 A** |
| 9 `Δη/2` | `(η_i − η_1) / 2` | — | `2(e_i − e_1)` | **1 A + shift** |
| 10 `|dxy|/2` | `|dxy_i| / 2` | — | `|d_i|` — strip the sign bit | **free** |

Per event: **64 multiplies, ~50 adds, 17 LUT lookups**. Compare with φ itself at
32-16-8 over 16 candidates: **12,800 multiply-accumulates**. The derived block is under
**1 %** of the layer it feeds.

*Note on `ln pt` vs `log1p(pt)`.* `8u = ln(1+pt)`, and `ln(1+pt) − ln(pt) = ln(1+1/pt) ≤
ln(1+1/4.4) = 0.207` because every candidate has pt ≥ 4.4 GeV. Divided by the channel's
own scale that is ≤ 0.05 for `lnz` and ≤ 0.026 for `lnE`, a constant-sign offset the first
Linear absorbs. If exactness is wanted instead, `ln pt = LUT(u)` is one more small LUT.

*What this costs where it hurts.* These are φ-input channels, so they widen the block the
FPGA lane already has at 91 % of one SLR: φ MACs go 12,800 → 15,872 (+24 %). Two variants
that are **cheaper than today's baseline** while keeping the channels are in RESULTS.md
(`c2_canon_narrow`, φ 24-12-8 = 10,368 MACs; `c2_rich_8p`, 8 candidates = 7,936).

## 2. Max-pooling alongside mean

`z = [mean_i φ(x_i), max_i φ(x_i)]`. **Comparators only, zero DSP**, one extra 8-wide
vector into ρ (+256 MACs). Worth +0.0005 on its own and +0.0012 on top of the derived
channels — small, but it is nearly free.

## 3. `iso_lead_pt`, `n_iso` — the isolated-candidate tag

The tt background is 2/3 semi- or fully-leptonic, and a lepton is a hard candidate with
nothing around it. This is the single most valuable *event-level* number found:
**+0.024 AUC vs tt on its own**, more than all 24 leading-4 pair numbers together.

```
iso_i    = (Σ_{j≠i, ΔR(i,j)<0.4} pt_j) / pt_i
iso_lead_pt = pt_k,  k = argmin_{i : pt_i > 10 GeV} iso_i
n_iso       = #{ i : pt_i > 10 GeV and iso_i < 0.15 }
```

Three things make this affordable:

1. **The cone test needs no ΔR and no atan2.** Use
   `ΔR'^2 = (η_i − η_j)^2 + 2(1 − cos Δφ_ij) < 0.16`, with
   `cos Δφ_ij = c_i c_j + s_i s_j` from channels 3-4. No square root, no 2π wrap.
   `2(1 − cos x)` equals `x²` to fourth order (0.1578 vs 0.16 at the cone edge).
   **Measured**: the two cones give +0.0361 and +0.0365 AUC vs tt — the same number.
   Per unordered pair: 1 A (Δη) + 1 M (square) + 2 M (cos) + 2 A + 1 compare.
   120 pairs → **360 M, ~360 A, 120 compares**, i.e. ~3 % of φ, and fully parallel —
   one combinational stage, not a sequential search.
2. **No divisions.** `iso_i < 0.15` becomes `Σpt < 0.15 · pt_i` (a constant multiply, or
   a shift-add: 0.15 ≈ 1/8 + 1/32). The argmin over ratios becomes a cross-multiplied
   comparison, `Σpt_i · pt_j < Σpt_j · pt_i` — 2 M per comparison, 15 comparisons in a
   tournament tree, **30 M** and no divider.
3. **No log on the output.** The feature is standardized as
   `(log1p(iso_lead_pt) − 3.0265) / 0.7147`, and `log1p(pt_k)` is `8·u_k` — the channel
   already at the input. Select `u_k` from the tournament, then one affine step.

The cone pT sums do need `pt_j` itself; if only `u_j` is on the wire, that is 16 `expm1`
LUTs (or take pt from the PUPPI object directly, where it already exists).

**Restricting the search is not worth it.** Allowing only the leading 8 (or 4) candidates
to *be* the isolated one cuts the table 2× (4×) but costs real AUC vs tt:
0.7146 (16 seeds) → 0.6910 (8) → 0.6688 (4). The lepton is often not among the four
hardest candidates. Pay for the full table.

## 4. `p_ij_lndRc` — six pair distances among the leading 4 candidates

```
lndR'_ij = ½ ln( (η_i − η_j)^2 + 2(1 − cos Δφ_ij) ),  (i,j) ∈ {12,13,14,23,24,34}
```

Same trick as the cone, and the same verdict from the measurement: the textbook
`½ ln(Δη² + Δφ²)` gives +0.0156 AUC vs tt and this multiply-only form gives **+0.0158**.
`data.py` therefore ships the firmware definition as the canonical one, so the trained
model and the trigger compute the identical number.

Per pair: 1 A + 1 M (Δη²) + 2 M + 2 A (cos Δφ) + 1 A + **1 log LUT**, and the ½ is a
shift. Six fixed pairs of *known* positions — no sorting, no search: **18 M, 6 L**.

Feeding `ΔR'^2` directly and dropping the log is possible (it is monotone), but the log
is what compresses three decades into the ±5 standardized range the datapath likes; one
small LUT is the cheaper side of that trade.

## 5. What was rejected, and why

| candidate | verdict | reason |
|---|---|---|
| ln kT, ln m² of the leading-4 pairs | drop | +0.000 on top of ln ΔR' — same information, different coordinates |
| ln z of the leading-4 pairs | drop | −0.000. Worth nothing anywhere |
| full 16×16×4 pairwise block | drop | pooling all 120 pairs keeps only +0.009 of the +0.023 the explicit leading-6 pairs give; the value is in *which* pair, and 1,024 inputs do not fit |
| jet clustering (anti-kT / cone) | drop | every jet-derived feature (`dm_W`, `dm_top`, `m_bb`, `n_jets`) is worth ≤ +0.010, and clustering is 6 *sequential* passes over the pair table — the one thing the latency budget cannot absorb |
| seed-restricted isolation | drop | 2×/4× cheaper, but −0.024/−0.046 AUC vs tt |

## 6. Total for the block

| stage | multiplies | LUTs | sequential depth |
|---|---|---|---|
| 6 derived per-candidate channels | 64 | 17 | 1 (needs HT first) |
| isolation pair table + tournament | ~390 | 16 (expm1, if pt is not on the wire) | 2 |
| 6 leading-4 pair distances | 18 | 6 | 1 |
| **total** | **~470** | **~39** | **≤ 3 combinational stages** |
| φ (32-16-8, 11 channels, ×16) | 15,872 | — | 3 |

**The whole derived-feature block is ~3 % of the multiplies in φ and adds at most three
combinational stages.** Against a synthesized baseline of 75-78 cycles (0.39 µs) at
ap_fixed<22,10>, it is not what threatens the 1 µs budget — the φ width is.
