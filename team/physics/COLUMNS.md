# What is actually in the C1_HH4b parquet files

We use 4 of the 14 subfields of one of 16 top-level columns. This is the full inventory,
what each thing is, and — the question that prompted it — whether any generator-level
truth is in there to distil from. Schema is identical in `train/` and `eval/`.

**Short answer on truth: no.** There are no generator particles, no gen jets, no parton
four-vectors, no pileup branch, and the two branches that *look* like MC bookkeeping
(`Event.Weight`, `Event.CrossSection`) are stored as float16 and have overflowed to 0/inf.
The closest thing to a b label is `L1T_JetPuppiAK4.BTag`, which is a **reconstructed**
tag, not truth — still usable as a privileged distillation target, but it must not be
described as gen-matched. Details and the measured fractions are in the last two sections.

## Top-level columns

| column | subfields | what it is |
|---|---|---|
| `L1T_PUPPIPart` | 14 | the particle-flow candidates. **This is our input.** |
| `L1T_JetPuppiAK4` | 11 | PUPPI anti-kT R=0.4 jets, *with* `BTag` |
| `L1T_JetPuppiAK8` | 11 | the R=0.8 version |
| `L1T_JetAK4`, `L1T_JetAK8` | 10, 11 | the non-PUPPI jets |
| `L1T_Electron` | 11 | reconstructed electrons, with isolation and impact parameter |
| `L1T_MuonTight` | 10 | reconstructed muons, same |
| `L1T_PhotonTight` | 6 | reconstructed photons |
| `L1T_MET`, `L1T_PUPPIMET` | 3 | missing transverse energy (`MET`, `Eta`, `Phi`) |
| `L1T_ScalarHT` | 1 | `HT` |
| `Event` | 2 | `Weight`, `CrossSection` — **both overflowed, see below** |
| `label` | — | int8: 0 QCD, 1 HH_4b, 2 tt, 3 W+jets |
| `source_file`, `source_row` | — | provenance |

Every collection is a struct of `large_list` per event, `halffloat` for the continuous
fields, so **everything is float16 on disk** — the overflow problems below are all the
same problem.

## `L1T_PUPPIPart` — the 14 candidate subfields

| subfield | used? | notes |
|---|---|---|
| `pt`, `eta`, `phi`, `dxy` | **yes** | the four we build all 11 channels from |
| `pdgId` | no | values are `{0, 11, 13, 22, 211}` — **neutral hadron, electron, muon, photon, charged hadron**. Unsigned. This is a per-candidate lepton tag, sitting unused (see below) |
| `dxysig` | no | dxy significance — the natural b-tagging variable. **Overflows float16**: p99 of \|dxysig\| is 2,460 (QCD) to 10,032 (HH) and some entries are `inf`. Usable only with a hard clip |
| `charge` | no | |
| `e`, `mass` | no | candidate energy and mass (we assume massless) |
| `dz`, `error_dz` | no | longitudinal impact parameter and its error |
| `puppi_weight`, `pt_weighted` | no | PUPPI pileup weight and the weighted pT |
| `funique_id` | no | |

Two of these are worth someone's time:

* **`pdgId`.** Electron and muon candidates are flagged directly. Fraction of the leading-16
  candidates that are electrons/muons: HH 1.67 % / 1.94 %, semi-leptonic tt 2.85 % / 3.24 %,
  hadronic tt 1.04 % / 1.14 %, QCD 0.21 % / 0.24 %, W+jets 1.52 % / 1.60 %. Our best
  single event feature, `iso_lead_pt`, is a hand-built proxy for exactly this — a two-bit
  `is_electron` / `is_muon` per candidate would be free in firmware.
* **`dxysig`.** A per-candidate impact-parameter significance is the standard b-tag input
  and we are using the unnormalized `dxy` instead. It needs clipping before use.

## The jet collections

`L1T_JetPuppiAK4` carries `Eta, Phi, PT, Mass, Charge, BTag, BTagPhys, NCharged, NNeutrals,
Constituents, ConstituentsIdx`.

* **`BTag`** is a 6-bit working-point mask from the *reconstructed* tagger. It separates
  processes exactly as a b-tagger should — jets per event with any bit set: **HH 2.94,
  hadronic tt 2.73, semi-leptonic tt 2.44, QCD 1.00, W+jets 0.33**.
* **`BTagPhys`** is Delphes' physics-flavour flag and *does not work here*: 0.94 (HH),
  1.18 (tt), 0.76 (QCD), 0.29 (W) jets per event, and per candidate 14.3 / 15.4 / 14.4 %
  — flat across processes. Whatever it is filled with, it is not a clean b-flavour truth
  flag, and it should not be used as a label.
* **`ConstituentsIdx`** does not index `L1T_PUPPIPart` in any way we could verify: for the
  leading jet of the first HH event, 3 of 4 indices are in range and the "constituents"
  they select sit up to Δη = 4.9 from the jet axis. Candidate→jet association has to be
  done by ΔR matching, not by these indices.

## `Event.Weight` / `Event.CrossSection` — unusable

Measured on the first fragment of each process (5,000 events):

| process | `Weight` | `CrossSection` |
|---|---|---|
| HH_4b | 2 distinct values, 78 % zero, else 0.00745 | 98 % `inf` |
| tt (hadronic) | 2 distinct, 82 % zero, else 1091 | 97 % `inf` |
| QCD | 2 distinct, 91 % zero, else `inf` | 92 % `inf` |
| W+jets | 2 distinct, 59 % zero, else `inf` | 99 % `inf` |

float16 saturates at 65,504, and cross sections in pb do not fit. **There is no way to
build a physically weighted mixture from these files.** Any statement about "real trigger
rates" has to come from external cross sections, not from the dataset.

## Per-candidate b labels: what can actually be built

Since there is no gen truth, the only available b label is ΔR < 0.4 matching of each
candidate to a jet whose reconstructed `BTag` is set. Fraction of the leading-16 candidates
so matched (5,000 events per process):

| process | in any AK4 jet | matched to a `BTag` jet | matched to a `BTagPhys` jet |
|---|---|---|---|
| HH_4b | 69.7 % | **49.9 %** | 14.3 % |
| tt hadronic | 77.8 % | 42.0 % | 16.2 % |
| tt semi-leptonic | 72.9 % | 39.5 % | 15.4 % |
| QCD | 68.8 % | 20.1 % | 14.4 % |
| W+jets | 16.3 % | 4.0 % | 3.2 % |

Signal candidates are b-matched half the time against 40 % in tt — a 10-point separation,
which is real but much softer than the *event*-level jet count (2.94 vs 2.44 tagged jets).
`BTagPhys` separates nothing and is not a usable label.

**So the auxiliary-b-head idea is still on the table, but reframed.** It would distil a
reconstructed tagger, not generator truth, and the per-candidate label is noisy. Two
cheaper things point the same way and should be tried first: `pdgId` as a per-candidate
lepton flag, and clipped `dxysig` as a per-candidate displacement significance. Both are
inputs the student could carry itself, which beats needing a teacher at all.

## Is any of this a legal input?

The challenge README does not restrict inputs to `L1T_PUPPIPart`; the intro notebook merely
chose it. Everything under `L1T_*` is a Level-1 trigger object, so on the face of it
`L1T_PUPPIMET`, `L1T_ScalarHT`, `L1T_Electron/MuonTight`, and the jet `BTag` are as legal
as the candidates — and several of them hand us, for free, quantities this lane spent the
day reconstructing (MET, HT, an isolated lepton, a b-tag). **Worth one question to the
organizers before anyone builds on it.** `BTagPhys` is the one field that is clearly
truth-derived and must stay out of the model either way.
