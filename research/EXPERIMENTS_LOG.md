# SENPAI Research Results — icml-appendix-charlie-pai2d-r4

## 2026-04-28 09:50 — PR #633: Domain-conditional FiLM at last-block decoder (infrastructure merge)
- Branch: `charliepai2d4-thorfinn/domain-film` (deleted on merge)
- Student: charliepai2d4-thorfinn
- **Outcome: MERGED (squash, commit 6b12744). Infrastructure merge — `--domain_film` flag plumbed at default False; BASELINE.md unchanged at #549's 54.12.**

### Headline (paired in same PR, EMA-evaluated, full merged stack: warmup3 + β=0.5)
| Run | best ep | val_avg | test_avg | Δ paired | vs #549 |
|---|---|---|---|---|---|
| baseline-ref-surface-film | 29 | 59.77 | 50.81 | (control) | +10.4% / +6.9% |
| **domain-film** | 33 | **55.37** | **48.33** | **-7.36% / -4.88%** | +2.31% / +1.66% (within noise) |
| matched-epoch (ep 29) | 29 | 58.22 | — | -2.59% | +7.6% |

### Per-split val (all 4 gain — paired)
| Split | surface-film | domain-film | Δ |
|---|---|---|---|
| val_single_in_dist     | 63.55 | 59.82 | -5.87% |
| val_geom_camber_rc     | 73.65 | 67.62 | -8.18% |
| val_geom_camber_cruise | 40.74 | 37.48 | **-8.02%** |
| val_re_rand            | 61.15 | 56.58 | -7.47% |

### Per-detected-domain mechanism check (the load-bearing diagnostic)
| Bucket | Domain | n | surface-film | domain-film | Δ |
|---|---|---|---|---|---|
| val_single_in_dist | single | 100 | 63.55 | 59.82 | -5.86% |
| val_geom_camber_rc | rc_tandem | 100 | 73.65 | 67.62 | -8.18% |
| val_geom_camber_cruise | cruise_tandem | 100 | 40.74 | 37.48 | **-8.02%** |
| val_re_rand | rc_tandem | 58 | 83.49 | 77.81 | -6.80% |
| val_re_rand | cruise_tandem | 42 | 38.31 | 34.88 | **-8.96%** (largest) |

**Hypothesis confirmed**: cruise-tandem (smallest y_std ≈ 164) gains most percentage-wise — the "explicit per-regime decoder affine helps most where the y_std distribution differs most from training average" prediction holds.

### Volume MAE: trunk/decoder separation extends to domain conditioning
| Split | surface-film vol_p | domain-film vol_p | Δ |
|---|---|---|---|
| val_single_in_dist | 74.70 | 73.23 | -1.97% |
| val_geom_camber_rc | 80.42 | 76.42 | -4.97% |
| val_geom_camber_cruise | 42.14 | 39.19 | **-7.00%** |
| val_re_rand | 61.47 | 58.01 | -5.63% |

Volume MAE improves uniformly — the decoder benefit isn't surface-specific; it's a general "domain-conditional decoder affine" mechanism.

### Domain detection rule fixed
Thorfinn discovered that the assignment's AoA-sign rule mis-labels ~47% of cruise-tandem samples (cruise AoA -5° to +6° overlaps raceCar -10° to 0°). Switched to **z_min rule**: cruise has freestream z ≈ -9.5; raceCar has ground at z=0 so z_min ≥ 0. Gives 100% clean partitioning verified per-split. Documented in `detect_domain` docstring.

### Why merge as infrastructure (not as new BASELINE)
- Absolute 55.37 is +2.3% above #549 (54.12) — within ~5pp noise floor.
- Paired surface-film baseline-ref came in at 59.77 (vs #484's 57.37 with same FiLM but no warmup) — suggests **warmup3+FiLM combination is somewhere between 57-60**, well above #549's 54.12. This is a separate FiLM-vs-warmup interaction issue, which alphonse's #687 (full-stack with warmup3 + β=0.3 + FiLM) is investigating directly.
- `--domain_film` flag plumbed at default False (no behavior change at default). Future PRs can `--domain_film` to test on different stacks. Mechanism evidence (paired -7.36%, all 8 splits, per-domain breakdown) is strong enough to make this a useful round-2 lever even if it's not a strict baseline-update.

### Round-2 candidates from thorfinn's follow-ups
- **Hybrid FiLM** (surface × domain): 6 sets of (γ, β). Combines both axes.
- **Learned domain detection**: small MLP over per-sample features producing 3-way softmax. Cleaner than hand-crafted z_min rule.
- **Decoder-only FiLM at last 2 blocks**: now twice-validated mechanism (#484 surface, #633 domain); #594's "FiLM-all-blocks loses on wall-clock" doesn't necessarily extend to "last-2-blocks FiLM".

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=633 records, 36 lines from domain-film run).

## 2026-04-28 09:30 — PR #623: Higher peak LR ∈ {7e-4, 1e-3} on top of warmup=3
- Branch: `charliepai2d4-alphonse/higher-lr-warmup3` (deleted on close)
- Student: charliepai2d4-alphonse
- **Outcome: CLOSED** (paired lr=7e-4 wins -2.96% val real but within ~5pp noise floor; absolute lags merged baseline #549).

### Headline (3-arm sweep, EMA-evaluated, paired in same PR)
| Run | best ep | val_avg | test_avg | Δ paired vs lr=5e-4 | Δ vs #549 |
|---|---|---|---|---|---|
| baseline-ref-lr5e-4 (paired ref) | 33 | 57.51 | 50.03 | (control) | +6.27% / +5.24% |
| **lr7e-4-warmup3** | 33 | **55.81** | **48.08** | **-2.96% / -3.89%** | +3.13% / +1.13% |
| lr1e-3-warmup3 | 33 | 56.45 | 48.09 | -1.85% / -3.87% | +4.31% / +1.16% |

### Critical finding: noise floor confirmed
**alphonse's same-pod reproduce of #549 lands at 57.51 vs recorded 54.12** — a 3.4pt gap on val. Cross-run noise on this stack (compile + bf16 + dynamic shapes + cudagraphs, no seeding) is **~3-4pt on val_avg, ~2.5pt on test_avg** — exactly the magnitude of the predicted lr-effect. The "lr=7e-4 paired wins by 2.96%" claim is real on paired-noise (same hardware, same code, runs back-to-back) but indistinguishable from absolute baseline noise without multi-seed retests.

### Mechanism: warmup absorbs higher LR cleanly (predicted, confirmed)
Grad-norm trajectories epochs 1-5 are essentially identical across the 3 runs:
| epoch | lr=5e-4 gn_mean | lr=7e-4 gn_mean | lr=1e-3 gn_mean |
|---|---|---|---|
| 1 | 18.5 | 21.4 | 17.3 |
| 2 | 54.9 | 44.2 | 39.5 |
| 5 | 27.8 | 23.1 | 18.5 |

- No NaN, no explosion at lr=1e-3 (predicted catastrophic failure didn't materialize).
- Counter-intuitively, lr=1e-3 has the LOWEST grad-clip fraction at epoch 5 (87% vs lr=5e-4's 93%) — higher peak LR pushes through the high-grad regime faster.
- All 3 runs descending monotonically through ep33, no late-epoch overshoot at lr=1e-3.

### Why close
- Paired -2.96% is real but within the same-stack noise floor (~3-4pt).
- Absolute lr=7e-4 = 55.81 vs merged #549 = 54.12 = +3.13% behind.
- Merging would set a new default lr=7e-4 that REPRODUCES at 55.81, worse than #549's recorded 54.12.
- Multi-seed retest could resolve, but expensive (3-6 runs).

### Round-2 / immediate follow-ups
- **Full-stack measurement** (alphonse's #2): warmup3 + β=0.3 + FiLM + lr=7e-4 — untested combined config, highest-EV single experiment. Assigned next.
- **Multi-seed retest infrastructure** (alphonse's #1): plumb `--seed` flag, run 3-seed paired triplicates. Round-2 candidate if noise issues persist.

### Cross-stack data point (warmup3 + β=0.5 + no FiLM)
alphonse's lr=7e-4 reproduce-of-#549-stack lands at val=55.81, test=48.08. This is the first direct measurement of "warmup3 + β=0.5 + lr=7e-4 + no FiLM" — adds to our cross-config map.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=623 records, 105 lines from all 3 sweep arms).

## 2026-04-28 09:10 — PR #599: Huber β finer-still sweep ∈ {0.1, 0.2, 0.4}
- Branch: `charliepai2d4-askeladd/huber-beta-finest` (deleted on close)
- Student: charliepai2d4-askeladd
- **Outcome: CLOSED** (β-axis bottomed out: β=0.2 wins paired by -0.6% val within noise; β=0.1 overshoots).

### Headline (4-arm sweep + paired ref, EMA-evaluated)
| β | best ep | val_avg | test_avg | Δ vs paired β=0.3-ref |
|---|---|---|---|---|
| 0.1 | 29 | 57.77 | 48.95 | +1.8% / +1.7% (worst) |
| **0.2** | 33 | **56.40** | **47.85** | **-0.6% / -0.5%** (within noise but lowest) |
| 0.3-ref (paired) | 33 | 56.74 | 48.11 | (control) |
| 0.4 | 33 | 57.14 | 49.43 | +0.7% / +2.7% |

### Combined β grid — U-shape with trough at β=0.2
| β | val_avg | trend |
|---|---|---|
| 0.1 | 57.77 | overshoots (loss collapses to pure L1, gradient signal flat) |
| 0.2 | **56.40** | ✓ optimum (interior) |
| 0.3 | 56.74 | ≈tie with 0.2 |
| 0.4 | 57.14 | monotone worse |
| 0.5+ | 57.50+ (#467+) | monotone toward MSE worse |

The 5-point monotone descent from #539 ({0.3, 0.5, 0.7, 1.0, 2.0}) extends to 0.2 and reverses at 0.1. Smooth-L1 axis is now fully mapped.

### Per-channel mechanism (val_geom_camber_cruise — heavy-tail-pressure split)
| β | mae_surf_Ux | mae_surf_Uy | mae_surf_p |
|---|---|---|---|
| **0.1** | **0.41** ✓ | **0.26** ✓ | 38.19 |
| **0.2** | 0.43 | 0.27 | **37.17** ✓ |
| 0.3 | 0.45 | 0.28 | 37.93 |
| 0.4 | 0.48 | 0.30 | 39.60 |

**Velocity prefers β→0 (monotone improvement); pressure has interior optimum at β=0.2.** Mechanism: at β=0.1 the loss collapses to pure L1 for residuals >0.1; in normalized space most pressure residuals are >0.1, so gradient signal becomes constant ±1 and learning dynamics on moderate-residual pressure nodes degrades. Velocity residuals stay closer to 0.1 normalized, so pure-L1 still has informative gradients there.

### Per-channel β is the next experiment
Different β per output channel: β_velocity ≈ 0.1 (or smaller) + β_pressure ≈ 0.2 (interior optimum). Predicted gain modest (−0.5% to −2%) since per-channel optimums are close to the joint optimum, but mechanism-grounded follow-up. Assigned to askeladd next.

### Why close
- β=0.2 vs β=0.3-ref paired is within ~5pp variance floor.
- vs current merged baseline #549 (54.12 val, has warmup3): askeladd's β=0.2 = 56.40 is +4.2% behind because askeladd didn't pass `--warmup_epochs 3` (despite the assignment instruction).
- β-axis is mapped — no new merge candidate from this sweep alone.

### Stability all-clean
| β | last-6 std |
|---|---|
| 0.1 | 1.35 |
| 0.2 | **1.21** (smoothest) |
| 0.3 | 1.40 |
| 0.4 | 1.29 |

NaN-free across all 4 runs; β=0.1's discontinuous-curvature concern from the assignment didn't materialize.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=599 records, 136 lines from all 4 sweep arms).

## 2026-04-28 08:30 — PR #453: w_p ramp 0.5→1.0 (rebased onto post-#549)
- Branch: `charliepai2d4-fern/pchannel-p-ramp05-10` (deleted on close)
- Student: charliepai2d4-fern
- **Outcome: CLOSED** (paired -1.80% within noise floor; rebase dropped --film and --warmup_epochs 3; absolute 57.86 vs merged 54.12 = +6.9%).

### Headline (epoch 33 of 33, EMA-evaluated, paired in same PR)
| Run | val_avg | test_avg |
|---|---|---|
| baseline-ref-film (no film, no warmup despite name) | 58.92 | 51.01 |
| w-ramp-huber-rebased (w_p ramp 0.5→0.83) | 57.86 | 49.87 |
| **Δ paired** | **-1.80%** | **-2.23%** |
| Δ vs merged #549 | +6.9% | +4.9% |

### Per-channel mechanism survives but shrinks
| Channel | pre-rebase Δ (vs β=1.0 base) | post-rebase Δ (vs β=0.5 base) |
|---|---|---|
| val surf_Ux | **-13.1%** | -4.2% |
| val surf_Uy | -7.3% | -3.9% |
| val surf_p | -3.5% | -1.8% |
| Velocity:pressure gain ratio | ~4× | ~2× |

Mechanism preserved (velocity gains > pressure gains), but magnitude compressed — Huber β=0.5 already does part of the velocity-vs-pressure rebalancing the ramp used to handle.

### Mechanism rotation: early-velocity → late-pressure
| Epoch | Pre-rebase ramp Δ vs base | Post-rebase ramp Δ vs base |
|---|---|---|
| 1 | -2% | **+13%** (low w_p starves pressure when Huber already underweights tail) |
| 13 | -6.1% (peak) | +0.8% |
| 30 | -3.9% | -1.7% |
| 33 | -3.5% | **-1.8%** (final, ramp pulls ahead late) |

**Mechanism rotated**: pre-rebase was "free velocity gradient early"; post-rebase is "refine pressure late". Trajectory: ramp behind early, catches up mid-training, pulls ahead from epoch 30 as w_p approaches 1.0.

### Per-split distribution flipped
| Split | Pre-rebase val Δ | Post-rebase val Δ |
|---|---|---|
| val_single_in_dist | **-7.72%** (biggest gain) | **+1.50%** (regresses) |
| val_geom_camber_rc | -1.44% (smallest) | **-5.00%** (biggest gain) |
| val_geom_camber_cruise | -3.34% | -1.34% |
| val_re_rand | -1.24% | -1.48% |

With FiLM + Fourier providing in-distribution capacity, the ramp's marginal benefit migrated to OOD camber splits where pressure prediction is still the bottleneck.

### Why close
- Paired -1.80% is at the empirical 5pp variance floor — borderline mechanism evidence.
- Absolute 57.86 vs merged 54.12 (#549, has warmup3 but not in fern's run) = +6.9% behind.
- Rebase missed --film and --warmup_epochs 3 (despite "baseline-ref-film" naming, no --film flag was passed). Fully-stacked compounding measurement would have predicted ~53-54 (well within noise of #549) given the diminishing-returns trajectory.
- Mechanism findings (rotation, per-split flip) are valuable for round-2 design but not a winning lever on their own.

### Round-2 candidates from fern's analysis
- Smarter ramp shapes: shorter ramp window post-warmup, or post-cosine-tail late refinement.
- Per-channel β (askeladd's queued #4 from #467): different β for pressure vs velocity, exploits the same heavy-tail-pressure asymmetry without the schedule complexity.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=453 records, 35 lines from rebased run).

## 2026-04-28 07:50 — PR #594: FiLM at all 5 block boundaries (mid-network specialization)
- Branch: `charliepai2d4-thorfinn/film-all-blocks` (deleted on close)
- Student: charliepai2d4-thorfinn
- **Outcome: CLOSED** (wall-clock +2.12% val regression; matched-epoch -1.92% but throughput cost eats gain).

### Headline (epoch 31 vs 34, EMA-evaluated, both runs in PR)
| Metric | baseline-ref-film-last-only | film-all-blocks | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (best, EMA, wall-clock) | 56.80 (ep 34) | 58.01 (ep 31) | +2.12% (loses) |
| `val_avg/mae_surf_p` (matched at ep 31) | 59.14 | 58.01 | **-1.92%** (wins, within predicted band) |
| `test_avg/mae_surf_p` (wall-clock) | 48.59 | 50.41 | +3.77% (loses) |
| Per-epoch | 54.0 s | 57.1 s | +5.7% |
| Total epochs | 34 | 31 | -3 |
| Peak GPU memory | 24.25 GB | 29.0 GB | +20% |
| Param count | 671 K | 673.6 K | +2.6 K (+0.4%) |

### Mechanism — supported in matched-epoch, eaten by throughput
- Matched-epoch FiLM gain: ~-1.92% at ep 31 (inside predicted -1% to -3% band).
- Throughput cost: +5.7%/epoch ≈ 3 epochs in 30-min cap.
- Cosine-tail value: -4% over those 3 epochs (baseline drops 59.14 → 56.80 over ep 31→34).
- Net wall-clock effect: -1.92% (FiLM gain) + +4% (lost cosine-tail) ≈ +2% (matches the observed +2.12%).

### Per-split val: regression concentrated on OOD splits
| Split | Δ |
|---|---|
| val_single_in_dist     | +1.46% |
| val_geom_camber_rc     | -0.25% (≈flat) |
| val_geom_camber_cruise | **+4.79%** |
| val_re_rand            | **+4.04%** |

### Per-split test: same OOD-concentrated regression pattern
| Split | Δ |
|---|---|
| test_single_in_dist     | +1.59% |
| test_geom_camber_rc     | +1.53% |
| test_geom_camber_cruise | **+7.30%** (largest) |
| test_re_rand            | **+6.75%** |

### Volume MAE: "both modes gain" mechanism doesn't extend
- vol_p IMPROVES on in-domain splits (-1.27% rc, -1.37% single)
- vol_p REGRESSES on OOD splits (+5.95% cruise, +3.32% re_rand)
- Surface MAE regresses across all channels and splits — the clean trunk/decoder separation from #484 breaks down with intermediate FiLMs.

### Why close
- Wall-clock budget is the operating regime; matched-epoch wins don't translate when throughput cost eats epochs.
- OOD regression is a structural concern — early-block FiLMs may amplify noise where domain-discriminative signal is weaker, OR over-specialize the trunk in ways that hurt cross-Re / cross-camber generalization.
- Throughput optimization (gather-based dual-mode select) could potentially recover the matched-epoch gain — parked as round-2 if domain-conditional FiLM (next experiment) suggests the mechanism family is worth deeper investment.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=594 records, 33 lines from film-all-blocks run).

## 2026-04-28 07:30 — PR #549: Linear warmup + cosine sweep — **NEW BASELINE**
- Branch: `charliepai2d4-alphonse/warmup-cosine-sweep` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit c234239). NEW BASELINE: val_avg=54.12 with warmup=3, -2.4% vs #539 on val, -0.9% on test.**

### Headline (4-arm sweep, EMA-evaluated)
| Run | warmup | best ep | val_avg | test_avg | Δ vs paired warmup0 | Δ vs #539 |
|---|---|---|---|---|---|---|
| baseline-ref-warmup0 | 0 | 33 | 58.34 | 50.32 | (control) | +5.25% (within seed noise) |
| warmup2 | 2 | 34 | 55.66 | 47.86 | -4.59% / -4.90% | +0.42% / -0.25% |
| **warmup3** | **3** | 34 | **54.12** | **47.54** | **-7.23%** / -5.50% | **-2.36%** / -0.92% |
| warmup5 | 5 | 34 | 54.91 | 48.44 | -5.87% / -3.74% | -0.94% / +0.96% |

### Mechanism diagnosis (the load-bearing artifact — grad-norm trajectory)
| epoch | warmup0 mean (max) | warmup3 mean (max) |
|---|---|---|
| 1 | **40.0 (max 178)** | 20.2 (max 40) |
| 2 | 31.3 | 52.4 |
| 3 | 25.5 | 51.2 |
| 4 | 24.9 | 38.0 |
| 5 | 21.6 | 28.7 |

**Confirmed**: warmup keeps epoch-1 grad norms 2-2.5× smaller and far less spiky (max 35-43 vs 178). The "huge per-step variance from cold AdamW m/v" observation from #466 is now empirically demonstrated — warmup tames it as predicted. Warmup variants then hit higher grad-norm regime in epochs 2-3 as LR ramps up — that's the "first useful work" window.

### Per-epoch trajectory (val_avg, EMA)
| epoch | warmup0 | warmup3 | crossover? |
|---|---|---|---|
| 1  | 236.6 | 389.3 | warmup behind (low-LR ramp) |
| 5  | 116.4 | 133.2 | warmup behind |
| 10 | 88.1  | 92.4  | warmup catching up |
| 20 | 68.7  | **67.9** | **warmup ahead** |
| 30 | 60.1  | **57.7** | warmup pulls ahead |
| 33-34 (best) | 58.34 | **54.12** | warmup wins |

**Crossover at epoch 20** — warmup variants start far behind in the warmup ramp, but pull ahead in the cosine tail. Cleaner training trajectory generalizes to harder OOD splits (warmup3 dominates val_geom_camber_rc by -9.9% vs warmup0).

### Per-split val (warmup3 winner)
| Split | warmup0 | warmup3 | Δ |
|---|---|---|---|
| val_single_in_dist     | 60.82 | 58.49 | -3.8% |
| val_geom_camber_rc     | 72.05 | 65.31 | **-9.9%** |
| val_geom_camber_cruise | 40.81 | 36.39 | -12.0% |
| val_re_rand            | 59.68 | 56.27 | -5.7% |

### Per-split test (warmup3 winner)
| Split | warmup0 | warmup3 |
|---|---|---|
| test_single_in_dist     | 53.99 | 52.37 |
| test_geom_camber_rc     | 63.21 | 60.09 |
| test_geom_camber_cruise | 34.22 | 31.05 |
| test_re_rand            | 49.86 | 46.65 |

### Cross-axis caveats
- **Branch is post-#467 but pre-#484** (no FiLM). alphonse's 54.12 lacks FiLM contribution.
- **Used β=0.5 explicitly** (not β=0.3). Branch was pre-#539.
- vs current merged baseline #539 (β=0.3 + FiLM = 55.43): -2.4% val, -0.9% test. Even without FiLM and without β=0.3, warmup3 wins.
- **Combined config `--huber_beta 0.3 --film --warmup_epochs 3` is untested** but predicted to compound to ~52-53 range.

### Cumulative round-1 trajectory
| PR | val_avg | Δ from prior best |
|---|---|---|
| #287 | 126.67 | (first) |
| ... | (intermediate) | |
| #539 |  55.43 |  -3.4% |
| #549 |  **54.12** |  **-2.4%** (val); -0.9% (test) |
| **Cumulative** | | **-57.3% from #287, -60.1% from published-baseline-equivalent** |

### Open follow-ups identified
- **Higher peak LR with warmup3** (alphonse's #2): the cleaner from-zero ramp may now permit lr=7e-4 or 1e-3 without the epoch-1 explosion frieren's #307 hit. Strongest predicted lever — assigned next.
- **Per-step warmup** (vs per-epoch): smoother LR ramp at finer granularity. Round-2.
- **Lower start_factor**: tested 1e-3; could try 0.1 for less-extreme ramp.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=549 records, 36 lines from warmup3 best run).

## 2026-04-28 07:05 — PR #528: Cosine eta_min sweep
- Branch: `charliepai2d4-frieren/cosine-eta-min-sweep` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (mechanism real but below noise floor; pre-#484 branch).

### Headline (4-arm sweep, EMA-evaluated)
| Run | val_avg | test_avg | best_epoch | LR @ term |
|---|---|---|---|---|
| eta=0 (cosine→0, ref) | 64.46 | 55.55 | 32 | 1.4e-4 |
| eta=1e-5 | 63.29 | 54.89 | 33 | 1.4e-4 (floor never engaged) |
| eta=1e-4 | 64.22 | 55.25 | 33 | 2.0e-4 (plateau active by ep 25-28) |
| **eta=5e-4** (constant LR) | **63.15** | 55.38 | 33 | 5.0e-4 |
| #368 baseline anchor | 62.94 | 54.73 | 33 | — |

Best variant (eta=5e-4) is +0.21 val (+0.33%) above PR #368 — within noise floor (~2pp empirically observed).

### Tail-trajectory mechanism (last 5 epochs val_avg, EMA)
| epoch | eta=0 | eta=1e-5 | eta=1e-4 | **eta=5e-4** |
|---|---|---|---|---|
| 28 | 67.84 | 66.19 | 67.10 | 67.32 |
| 32 | 64.46 | 64.73 | 64.74 | **63.59** |
| Δ ep28→32 | -3.38 | -1.46 | -2.36 | **-3.73** |

eta=5e-4 has the fastest late descent — the "model still has gradient direction late in training" hypothesis is qualitatively supported.

### Per-split mechanism — interesting tradeoff
- `val_geom_camber_rc` (OOD camber holdout): **monotone improvement with eta_min** (79.5 → 78.98 → 77.59 → 75.66). Hardest-OOD split benefits most from continued learning at high LR.
- `test_re_rand`: **monotone REGRESSION with eta_min** (55.64 → 55.24 → 55.86 → 56.30). Cleaner Re-rand samples overshoot the optimum at high constant LR.
- Per-split tradeoff suggests a **two-stage schedule** (cosine then constant floor) might dodge both failure modes — frieren's follow-up #3 candidate.

### Why close
- Mechanism is qualitatively real but magnitude is below the ~2% empirical noise floor (eta=0 vs eta=1e-5 differ by 1.17 mae despite identical LR schedules).
- **Pre-#484 branch**: absolute 63.15 is +14% above the current merged baseline (55.43, #539). Even with mechanism-clean transfer, the absolute number can't compete.
- Frieren's clean run/sample/init-noise-floor analysis is the most useful artifact — this PR establishes that single-seed sweeps can't resolve <2% effects on this hardware/data.

### Round-2 candidates from frieren's follow-ups
- **Two-stage schedule** (cosine first, then constant at moderate floor): mechanism-clean, addresses both per-split failure modes. Strongest candidate.
- **Multi-seed retest of eta=5e-4 vs eta=0**: only way to resolve sub-noise effects. Expensive (6 runs).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=528 records, 139 lines from all 4 sweep arms).

## 2026-04-28 06:45 — PR #539: Huber β finer sweep — β=0.3 wins, **NEW BASELINE**
- Branch: `charliepai2d4-askeladd/huber-beta-finer` (deleted on merge)
- Student: charliepai2d4-askeladd
- **Outcome: MERGED (squash, commit 893ea4c). NEW BASELINE: val_avg=55.43 with β=0.3, -3.4% vs #484 on val, -2.0% on test.**

### Headline (3-arm sweep + paired ref, EMA-evaluated)
| β | best epoch | val_avg | test_avg | Δ vs paired ref |
|---|---|---|---|---|
| **0.3** | 34 | **55.43** | **47.98** | **-6.2% val / -4.9% test** |
| 0.5 (paired ref) | 33 | 59.12 | 50.46 | (control) |
| 0.7 | 33 | 60.32 | 52.19 | +2.0% val / +3.4% test |

### Combined 5-point β grid (this PR + #467)
| β | val_avg | test_avg |
|---|---|---|
| **0.3** | **55.43** | **47.98** |
| 0.5 | 57.50 | 50.51 (#467) |
| 0.7 | 60.32 | 52.19 |
| 1.0 | 63.13 | 55.16 (#467) |
| 2.0 | 68.53 | 60.44 (#467) |

**Strict monotone β trend toward L1.** No interior optimum.

### Per-channel mechanism (val_geom_camber_cruise — biggest gain split)
| β | mae_surf_Ux | mae_surf_Uy | mae_surf_p |
|---|---|---|---|
| **0.3** | 0.451 | 0.278 | **36.67** |
| 0.5-ref | 0.495 | 0.314 | 41.46 |

β=0.3 wins on all 3 channels. Pressure gain -11.6% — largest, consistent with heavy-tailed residuals benefiting most from L1-leaning shape.

### Stability check
- Last-6 EMA val_avg std: β=0.3 = **1.32**, β=0.5-ref = 1.48, β=0.7 = 1.54.
- β=0.3 is the **smoothest** of the three. Discontinuous-curvature concern from PR body did NOT materialize.

### Config default flip — verified
A no-flag `python train.py --epochs 50` run produces `config.yaml` with `huber_beta: 0.5` (proof committed in PR). Default flipped from 1.0 to 0.5 successfully.

### Cross-axis caveat
Branch is post-#467 but pre-#484 (FiLM landed in between). β=0.3 measurement at 55.43 is **without FiLM**. vs current merged baseline #484 (57.37 val with β=0.5 + FiLM): β=0.3 wins -3.4% on val even without FiLM. Combined β=0.3 + FiLM (the post-merge config when reproduced explicitly) should compound to ~54-55 range.

### Cumulative round-1 trajectory
| PR | val_avg | Δ from prior best |
|---|---|---|
| #287 | 126.67 | (first baseline) |
| #308 | 106.40 | -16.2% |
| #381 |  98.85 |  -7.1% |
| #401 |  66.89 | -32.3% |
| #289 |  63.33 |  -5.3% |
| #368 |  62.94 |  -0.6% |
| #467 |  57.50 |  -8.7% |
| #484 |  57.37 |  -0.2% (val); test -3.1% |
| #539 |  **55.43** |  -3.4% (val); test -2.0% |
| **Cumulative** | | **-56.2% from #287, -59.2% from published-baseline-equivalent** |

### Open questions / follow-ups
- **β finer-still grid {0.1, 0.2, 0.4}**: monotone trend says optimum may be even further toward L1 (β→0+). Saturation likely; smallest predicted gain. Assigning next.
- **Per-channel β**: pressure has fattest tails so might want sharpest β. Smaller predicted gain but mechanism-clean. Round-2 candidate.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=539 records, 36 lines for the β=0.3 best run).

## 2026-04-28 06:30 — PR #512: Fourier n_freqs sweep — n=4 wins, mechanism partially confirmed
- Branch: `charliepai2d4-edward/fourier-nfreqs-sweep` (still in flight after revision)
- Student: charliepai2d4-edward
- **Outcome: SENT BACK** (n=4 wins on n=8 base by -1.96%; pre-#467 / pre-#484 branch; rebase + re-run with β=0.5 + FiLM).

### Headline (4-arm sweep, EMA-evaluated, paired in same PR)
| n_freqs | best epoch | val_avg/mae_surf_p | Δ vs n=8 ref | test_avg/mae_surf_p | n_params |
|---|---|---|---|---|---|
| **4** | 32 | **61.70** | **-1.96%** | **53.50** | 666,455 |
| 6 | 32 | 62.37 | -0.91% | 54.01 | 668,503 |
| 8 (paired ref) | 33 | 62.80 | (control) | 54.90 | 670,551 |
| 12 | 32 | 62.61 | -0.52% | 54.17 | 674,647 |

n=8 ref reproduces #368's 62.94 within run-to-run noise.

### Mechanism: cruise-split detail (the high-freq-noise diagnostic)
| n_freqs | val_cruise/mae_surf_p | test_cruise/mae_surf_p |
|---|---|---|
| 4  | 44.36 | 37.18 |
| 6  | **43.68** | **36.65** ← test best |
| 8  | 45.04 | 38.12 ← worst |
| 12 | **43.30** ← val best | 37.75 |

**Refined hypothesis**: n=8 IS a local pessimum on cruise (+1.5pp vs n=4/6), confirming high-freq bands hurt at this band count. But the pattern isn't strictly monotonic — n=12 actually ties or beats n=8 on every metric. Edward's reading: "the model can apparently learn to ignore the highest bands when there are enough of them, but n=8 lands in a bad sweet spot where the highest bands are large enough to inject noise but not redundant enough to be ignored."

### Per-split val (n=4 winner)
| Split | n=4 | n=8 ref | Δ |
|---|---|---|---|
| val_single_in_dist     | 63.17 | 66.12 | -4.5% |
| val_geom_camber_rc     | 76.90 | 77.16 | -0.3% |
| val_geom_camber_cruise | 44.36 | 45.04 | -1.5% |
| val_re_rand            | 62.39 | 62.90 | -0.8% |

### Per-split test (n=4 winner)
| Split | n=4 | n=8 ref | Δ |
|---|---|---|---|
| test_single_in_dist     | 57.27 | 57.22 | +0.1% (essentially flat) |
| test_geom_camber_rc     | 65.73 | 68.95 | **-4.7%** |
| test_geom_camber_cruise | 37.18 | 38.12 | -2.5% |
| test_re_rand            | 53.81 | 55.33 | -2.7% |

All 4 test splits gain at n=4.

### Why send back rather than merge
- Branch is pre-#467 / pre-#484. Absolute val 61.70 is +7.6% above current merged baseline (#484 at 57.37 with β=0.5 + FiLM).
- Mechanism is clearly orthogonal: Fourier features are computed outside the model on x_norm before the model forward; β=0.5 changes loss curve shape; FiLM is an affine inside the model. Three independent layers, no expected interference.
- Predicted post-rebase val: -1.10 mae gap from sweep → expected **56.3-57.0 range** with β=0.5 + FiLM + n=4.
- Edward will also flip the Config default `n_freqs=8 → 4` to make the merged baseline reproduce with the winning value (same fix as #539 huber_beta default flip).

### Useful follow-ups identified
- **Learnable Gaussian Fourier features (Tancik 2020)**: replace fixed `2^k π` ladder with `B ∼ N(0, σ²I)` learned/tuned. The largest predicted gain remaining on the Fourier axis per edward's read.
- **Per-axis n_freqs**: x and z play different physical roles (chord vs pitch); separate band counts exploit that.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=512 records, 137 lines from all 4 sweep arms).

## 2026-04-28 06:15 — PR #484: Surface-conditional FiLM rebased onto post-#467 — **NEW BASELINE**
- Branch: `charliepai2d4-thorfinn/surface-film` (deleted on merge)
- Student: charliepai2d4-thorfinn
- **Outcome: MERGED (squash, commit dc9e0e5). NEW BASELINE: val_avg=57.37, -0.23% vs #467 on val, -3.06% on test_avg.**

### Headline (epoch 33, EMA-evaluated, both runs in PR)
| Run | val_avg | test_avg | params |
|---|---|---|---|
| baseline-ref-beta05 (paired) | 59.17 | 50.44 | 670,551 |
| **film-beta05** | **57.37** | **48.96** | 671,063 (+512) |
| Δ vs paired | **-3.05%** | **-2.92%** | +0.08% |
| vs merged #467 | -0.23% | **-3.06%** | — |

### Per-split val (4/4 gain at β=0.5 — vs 3/4 at β=1.0 in pre-rebase)
| Split | Δ |
|---|---|
| val_single_in_dist     | **-3.65%** |
| val_geom_camber_rc     | **-3.51%** (was +0.65% noise at β=1.0; now clean gain) |
| val_geom_camber_cruise | -1.30% |
| val_re_rand            | -3.01% |

### Per-split test (4/4 gain)
| Split | Δ |
|---|---|
| test_single_in_dist     | -1.71% |
| test_geom_camber_rc     | -1.94% |
| test_geom_camber_cruise | -3.92% |
| test_re_rand            | **-4.79%** (largest test gain) |

### Volume MAE (replicates the trunk/decoder-separation finding)
| Split | mae_vol_p Δ |
|---|---|
| val_single_in_dist     | -4.92% |
| val_geom_camber_rc     | -1.15% |
| val_geom_camber_cruise | -2.03% |
| val_re_rand            | -1.70% |

Smaller magnitude than at β=1.0 (-11.4% on single_in_dist there) because β=0.5 already sharpens the volume head — exactly the redundancy predicted in the send-back. Mechanism unchanged: shared trunk → tiny domain-conditional decoder → no interference cost.

### Analysis
- **Mechanism survives β=0.5 cleanly**: paired -3.05% gain comparable to the pre-rebase -2.34% at β=1.0. FiLM's specialization is roughly **additive in absolute MAE units** with whatever β=0.5 already delivered.
- **Predicted compounding range was 55-57; landed at 57.37** — upper bound. Partial redundancy (FiLM and β=0.5 both operate on heavy-tailed-pressure regions but via different mechanisms: affine modulation vs loss curve shape).
- **All 4 val splits gain** (vs 3/4 at β=1.0) — even camber_rc, which had a tiny noise regression before, now gains -3.51% cleanly.
- **+0.5% wall-clock + 512 params** is essentially free. Compile diagnostics clean (3 unique graphs, no recompile spiral, no CUDAGraph OOM since cudagraph_skip is in baseline).
- **Compounding evidence: 7 stacked levers** all positive — Fourier + Huber β=0.5 + EMA + clip + bf16 + compile + FiLM.

### Cumulative round-1 trajectory
| PR | val_avg | Δ from prior best |
|---|---|---|
| #287 | 126.67 | (first baseline) |
| #308 | 106.40 | -16.2% |
| #381 |  98.85 |  -7.1% |
| #401 |  66.89 | -32.3% |
| #289 |  63.33 |  -5.3% |
| #368 |  62.94 |  -0.6% |
| #467 |  57.50 |  -8.7% |
| #484 |  **57.37** |  -0.2% (val); -3.1% (test_avg, 50.51 → 48.96) |
| **Cumulative** | | **-54.7% from #287, -57.6% from published-baseline-equivalent** |

### Open follow-ups identified
- **FiLM at all 5 block boundaries** (~5×512 = 2,560 params, mid-network specialization). Strongest extension; assigned next.
- **Surface-only vs symmetric FiLM ablation** — with vol MAE also improving, the symmetric form is now the expected default. Ablation cleanly answers whether vol_film is doing real work.
- **Stack with fern's #453 (per-channel surf_weight ramp) once it lands** — three orthogonal surface-mechanism levers.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=484 records, 35 lines from final film-beta05 run).

## 2026-04-28 05:35 — PR #466: cudagraph_skip + cosine_epochs flag (revision merge)
- Branch: `charliepai2d4-alphonse/tmax32-cudagraph-skip` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit e0a902b). Infrastructure merge — BASELINE stays at #467's 57.50.**

### Headline (cudagraph-skip-only run, EMA-evaluated)
| Metric | Value | vs #289 anchor |
|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 64.20 | +1.37% (within ~5pp variance floor) |
| `test_avg/mae_surf_p` (EMA) | 55.56 | +0.20% |
| Per-epoch wall-clock | 53.87 s | matches #289 (54.4 s) |
| Crashes / launches | 0 / 1 | (second clean 1/1 launch under cudagraph_skip) |
| dynamo unique_graphs | 3 | clean |

### What this PR adds
- **`cudagraph_skip_dynamic_graphs=True`**: redundant with #467 (which already merged this), but git auto-deduplication handled the conflict cleanly.
- **`--cosine_epochs` CLI flag, default 50**: novel addition. No behavior change at the default; useful for future depth/capacity experiments to override.

### Why merge
- The cosine_epochs flag is plumbed for future use (e.g. depth experiments that legitimately want a shorter horizon).
- Two clean 1/1 launches confirm cudagraph_skip is throughput-neutral.
- BASELINE.md remains at #467's 57.50 since this is pure infrastructure (parallel to #372 bf16, #358 scoring fix).

### Round-2 lever identified by alphonse
- Their follow-up #1 (cosine to lr_min > 0): frieren is running this in PR #528 (eta_min sweep).
- Their follow-up #3 (warmup + cosine): alphonse's next assignment (PR #547 or similar).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=466 records, includes the final cudagraph-skip-only run, 36 lines).

## 2026-04-28 05:25 — PR #484: Surface-conditional FiLM in last TransolverBlock
- Branch: `charliepai2d4-thorfinn/surface-film` (still in flight after revision)
- Student: charliepai2d4-thorfinn
- **Outcome: SENT BACK** (mechanism strong, paired -2.34% val / -3.23% test, but auto-merge CONFLICTING with #467; rebase to post-β=0.5 baseline + re-run).

### Headline (epoch 33, EMA-evaluated, both runs in PR)
| Run | val_avg | test_avg | n_params |
|---|---|---|---|
| baseline-ref (no FiLM) | 63.21 | 55.57 | 662,359 |
| surface-film | **61.73** | **53.78** | 662,871 (+512) |
| Δ vs paired | **-2.34%** | **-3.23%** | +0.08% |

### Per-split val (3/4 gain)
| Split | Δ |
|---|---|
| val_single_in_dist     | **-5.25%** (largest) |
| val_geom_camber_rc     | +0.65% (smallest, within noise) |
| val_geom_camber_cruise | -3.82% |
| val_re_rand            | -1.45% |

### Per-split test (4/4 gain)
| Split | Δ |
|---|---|
| test_single_in_dist     | **-5.67%** |
| test_geom_camber_rc     | -1.26% (val regression on rc reverses on test → noise read confirmed) |
| test_geom_camber_cruise | -4.32% |
| test_re_rand            | -2.08% |

### Bonus mechanism finding: volume MAE also improves
| Split | mae_vol_p Δ |
|---|---|
| val_single_in_dist     | **-11.42%** (large) |
| val_geom_camber_rc     | -3.44% |
| val_geom_camber_cruise | -1.20% |
| val_re_rand            | -2.43% |

The PR predicted vol MAE should match baseline. **Wrong**: gamma_vol/beta_vol are also learned, so the shared mlp2 receives different affine inputs from surface vs volume nodes → trunk produces one shared representation, FiLM-modulated decoder acts as a tiny domain-conditional decoder for **both** modes. The structural insight from #436's post-mortem ("no parallel pathway → no trunk interference") is empirically validated AND extended: domain-conditional decoder helps everywhere, not just the surface side it was designed for.

### Why send back rather than merge
- Auto-merge CONFLICTING with PR #467 (β=0.5 just merged). Both changes touch train.py model class + Config; need manual rebase.
- Absolute val 61.73 is +7.4% above the new merged baseline (#467 at 57.50). The paired comparison is what makes this interesting; the rebased FiLM + β=0.5 combo is the cleaner test.
- Mechanisms are mathematically orthogonal (FiLM = affine modulation in last block; β = loss curve shape), so they should compound. Expected post-rebase val: 55-57 range.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=484 records, 70 lines from both runs).

## 2026-04-28 05:05 — PR #467: Huber β sweep — β=0.5 wins, **NEW BASELINE (-8.65% vs #368)**
- Branch: `charliepai2d4-askeladd/huber-beta-sweep` (deleted on merge)
- Student: charliepai2d4-askeladd
- **Outcome: MERGED (squash, commit eb5168f). NEW BASELINE: val_avg=57.50 with β=0.5, -8.65% vs #368.**
- Bonus: cudagraph_skip_dynamic_graphs=True bundled into the merge as throughput-neutral robustness fix (parallel to alphonse #466's pending revision).

### Headline (3-arm β sweep, EMA-evaluated, paired in same PR)
| β | best epoch | val_avg | test_avg | Δ val vs ref | Δ test vs ref |
|---|---|---|---|---|---|
| **0.5** | 34 | **57.50** | **50.51** | **-8.91%** | **-8.41%** |
| 1.0 (paired ref) | 33 | 63.13 | 55.16 | (control) | (control) |
| 2.0 | 33 | 68.53 | 60.44 | +8.55% | +9.57% |

vs current merged baseline #368 (val=62.94, test=54.73): **β=0.5 lands at -8.65% val, -7.71% test**. Strict monotone β order on every val + test split + every output channel.

### Per-split val Δ (β=0.5 − β=1.0 ref)
| Split | Δ |
|---|---|
| val_single_in_dist     | -9.45% |
| val_geom_camber_rc     | -5.87% |
| val_geom_camber_cruise | **-12.08%** (largest gain — predicted by sharper-near-zero hypothesis) |
| val_re_rand            | -9.69% |

### Per-split test Δ
| Split | Δ |
|---|---|
| test_single_in_dist     | -8.43% |
| test_geom_camber_rc     | -6.24% |
| test_geom_camber_cruise | -11.89% |
| test_re_rand            | -8.78% |

### Per-channel mechanism (cruise camber, the biggest β=0.5 winner)
| Channel | β=0.5 | β=1.0 | β=2.0 | Δ(0.5 vs 1.0) |
|---|---|---|---|---|
| mae_surf_Ux | 0.466 | 0.543 | 0.613 | -14.2% |
| mae_surf_Uy | 0.294 | 0.351 | 0.382 | -16.3% |
| mae_surf_p  | 38.46 | 43.75 | 50.78 | **-12.1%** |

β=0.5 helps **all three channels** — pressure, Ux, Uy — not just heavy-tailed pressure. The 'L1-like loss might over-emphasize tiny errors' worry didn't materialize.

### Analysis
- **Mechanism confirmed and stronger than predicted** (predicted -1% to -4%, actual -8.91%). Likely amplification factor: EMA + cosine-tail interactions with β shape compound at the now-33-34-epoch budget.
- **Cruise gains biggest** (-12.1% val mae_surf_p) — exactly the sharper-near-zero prediction. raceCar camber gains least (-5.87%, fattest pressure tail benefits less from sharpening).
- **β=2.0 (more L2-like) hurts ALL channels**, including velocity. Confirms the loss-shape effect is genuine, not just tail-dominance.
- **Throughput-neutral**: 53.7-54.1 s/epoch across all 3 β values vs #289's 54.4 s. cudagraph_skip flag applied throughout (askeladd hit the known compile flakiness on first launch and bundled the fix).
- **Compounding evidence**: Huber β=0.5 + Fourier + EMA + clip + bf16 + compile = 6 stacked levers, all positive, no observable interference.

### Cumulative round-1 trajectory
| PR | val_avg | Δ from prior best |
|---|---|---|
| #287 | 126.67 | (first baseline) |
| #308 | 106.40 | -16.2% |
| #381 |  98.85 |  -7.1% |
| #401 |  66.89 | -32.3% |
| #289 |  63.33 |  -5.3% |
| #368 |  62.94 |  -0.6% |
| #467 |  **57.50** |  **-8.7%** |
| **Cumulative** | | **-54.6% from #287, -57.5% from published-baseline-equivalent** |

### Open infrastructure note
- Merged train.py has β as a CLI flag with default `huber_beta=1.0`. Reproducing the merged baseline requires `--huber_beta 0.5`. Same flag-vs-default issue as alphonse's #287 surf_weight=25 (which became orphaned in subsequent PRs). Asking askeladd to flip the Config default to 0.5 in their next PR (β finer sweep + default flip).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=467 records, 107 lines from all 3 runs).

## 2026-04-28 04:50 — PR #477: Conservative widening n_hidden=144 on post-#289 + compile
- Branch: `charliepai2d4-frieren/wider144-compile` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (+6.4% vs paired baseline-ref; second paired-experiment confirmation that wider loses to budget on this 30-min frontier).

### Headline (epoch 29 vs 33, EMA-evaluated, both runs in PR)
| Run | Best epoch | val_avg | test_avg | per-epoch |
|---|---|---|---|---|
| baseline-ref (n=128) | 33 | **62.87** | **54.66** | 55.0 s |
| wider144 (n=144) | 29 | 66.87 | 58.02 | 62.1 s (+13%) |
| Δ | -4 epochs | **+6.4%** | +6.1% | |

baseline-ref reproduced merged #289 (63.33) cleanly within run-to-run noise.

### Same-epoch trajectory (capacity at fixed epoch IS gained, but offset by cosine-tail loss)
| Epoch | baseline-ref | wider144 | Δ (wider − base) |
|---|---|---|---|
| 5  | 138.96 | 135.08 | -3.88 |
| 10 | 103.85 | 100.79 | -3.06 |
| 16 |  86.47 |  83.93 | -2.54 |
| 20 |  79.57 |  77.25 | -2.32 |
| 25 |  73.59 |  70.39 | -3.20 |
| 29 |  68.03 |  **66.87** (its best) | -1.16 |
| 30 |  65.78 | — | (wider stopped) |
| 33 |  **62.87** (its best) | — | |

Baseline-ref drops 5.16 mae across epochs 30-33 (cosine-tail), wiping out the 1-3 mae capacity advantage from earlier epochs.

### Mechanism (the load-bearing finding)
- **+13% per-epoch tax ≈ 12% share of cosine-tail epochs** (4/33). Frontier dead-on — this is not a measurement artifact.
- **Two paired experiments now**: #431 (wider160, +14% tax → +2.5% val regression) and this PR (wider144, +13% tax → +6.4% val regression). Both confirm capacity > budget loses on this 30-min schedule.
- **Capacity gain narrows over training**: epoch 5 advantage 3.88, epoch 29 only 1.16. Models converge toward similar regime; the wider model's extra capacity stops paying off in late training.

### Lessons / next steps
- **Don't chase wider** at this schedule (frieren's #1). 152/136 would land on the same frontier.
- **Schedule changes are higher-leverage than capacity changes** at this budget. Specifically: eta_min > 0 (alphonse's #466 follow-up #1, frieren's #3) — let the cosine tail continue learning at low-but-nonzero lr instead of truncating to 0. **Assigning that to frieren next.**
- Depth instead of width (frieren's #4): different compute frontier, parking.
- Slice_num bump (frieren's #5): parking until cudagraph_skip lands.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=477 records, 31 lines from wider144 run).

## 2026-04-28 04:35 — PR #466: cosine T_max retune (32) + cudagraph_skip_dynamic_graphs flag
- Branch: `charliepai2d4-alphonse/tmax32-cudagraph-skip` (still in flight after revision)
- Student: charliepai2d4-alphonse
- **Outcome: SENT BACK** (cudagraph_skip is clean win; cosine_epochs=32 regresses +6.7%; revise to keep cudagraph_skip + revert cosine_epochs default).

### Headline (clean A/B at fixed cudagraph_skip flag)
| Metric | #289 baseline | baseline-ref-tmax50 | tmax32-cudagraph-skip | Δ tmax32 vs tmax50 |
|---|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 63.33 | 64.07 | **68.34** | **+6.7%** |
| `test_avg/mae_surf_p` (EMA) | 55.45 | 55.47 | 59.54 | +7.4% |
| Per-epoch wall-clock | 54.4 s | 53.8 s | 53.8 s | matches |
| Crashes / launches | — | 0/1 | 0/1 | both clean |
| LR at termination | ~1.35e-4 | 1.30e-4 | 4.80e-6 | tmax32 reaches lr≈0 ✓ |

### Why cosine_epochs=32 regressed
- At T_max=50, lr ≈ 1.3e-4 at epoch 33 and the model is still dropping ~0.9 mae/epoch (EMA).
- At T_max=32, lr ≈ 0 from epoch 32 onward and EMA gains only ~0.05 mae across the final two epochs.
- Hypothesis assumed late-stage "fine-tuning" was being cut off. The data says the opposite: the model is in the **bulk-learning regime** at epoch 33, not fine-tuning — there is no late-stage low-LR phase to extend.
- All 4 val splits move the same direction; mechanism-coherent regression.

### What's keeping
- **`cudagraph_skip_dynamic_graphs=True`**: cleanly throughput-neutral (53.8 s/epoch vs #289's 54.4 s, within run-to-run noise), peak memory unchanged at 23.84 GB, 3 unique dynamo graphs (no recompile spiral), 1/1 launches succeed in both runs. Will land as throughput infrastructure (parallel to #372 bf16 merge).
- **`--cosine_epochs` CLI flag**: useful for future depth/capacity-increasing experiments that genuinely want a shorter horizon. Default reverts to 50.

### Mechanism follow-up identified
Alphonse's strongest follow-up: **cosine to lr_min > 0** (e.g. `eta_min=1e-4`). The LR trajectory shows the model wants to keep learning at ~1e-4; lifting the floor of the late-stage tail rather than truncating the schedule is the right read of the data. Queued as round-2 candidate.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=466 records, 71 lines from both runs).

## 2026-04-28 04:20 — PR #368: Fourier positional encoding rebased onto post-#289 — **NEW BASELINE**
- Branch: `charliepai2d4-edward/fourier-pos-encoding` (deleted on merge)
- Student: charliepai2d4-edward
- **Outcome: MERGED (squash, commit 430cd62). NEW BASELINE: val_avg=62.94, -0.62% vs #289; test_avg=54.73, -1.30%.**

### Headline (epoch 33 of 33, EMA-evaluated)
| Metric | This run | PR #289 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **62.94** | 63.33 | **-0.62%** |
| `test_avg/mae_surf_p` (EMA) | **54.73** | 55.45 | **-1.30%** |
| Per-epoch (steady-state) | 54.6 s | 54.4 s | matches |
| Total epochs | 33 | 32 | +1 |
| Peak GPU memory | 24.2 GB | 23.8 GB | +0.4 GB (wider preprocess) |

### Equal-epoch trajectory (the load-bearing evidence)
| Epoch | PR #289 | Fourier+#289 | Δ |
|---|---|---|---|
| 5  | 137 | 117 | **-14.4%** |
| 10 | 101 |  93 | -8.4% |
| 15 |  88 |  80 | -9.1% |
| 20 |  78 |  75 | -3.4% |
| 25 |  71 |  69 | -2.6% |
| 30 |  65 |  64 | -0.6% |
| 33 | n/a |  63 | new best |

### Per-split val (epoch 33, EMA, vs #289)
| Split | Fourier+#289 | #289 | Δ |
|---|---|---|---|
| val_single_in_dist     | 67.25 | 69.14 | **-2.7%** |
| val_geom_camber_rc     | 75.50 | 75.30 | +0.3% |
| val_geom_camber_cruise | 45.70 | 45.70 | 0.0% |
| val_re_rand            | 63.31 | 63.20 | +0.2% |

### Per-split test (post-fix scoring, EMA)
| Split | mae_surf_p | Δ vs #289 |
|---|---|---|
| test_single_in_dist     | 59.05 | **-3.3%** |
| test_geom_camber_rc     | 66.47 | **-3.1%** |
| test_geom_camber_cruise | 38.68 | +3.2% (highest-freq bands may add noise on easiest split) |
| test_re_rand            | 54.73 | +0.1% |

### Analysis
- **Mechanism interpretation: Fourier features accelerate convergence rather than raise asymptote.** Through epochs 5-15 the gap is -8 to -14%; cosine decay narrows the gap as both runs near their respective minima. Edward correctly notes T_max retune (alphonse #466) would let the Fourier advantage persist into the cosine tail.
- **Test side stronger than val** (-1.30% vs -0.62%), and gains concentrate on the **hardest splits** (single_in_dist and geom_camber_rc, the two raceCar-dominated splits with the heaviest pressure tails). Mechanism-coherent: Fourier features should help most where pressure has sharp local features (suction peaks at leading edges).
- **No throughput regression** despite +8K params on the preprocess MLP — compile + dynamic=True absorbed the change cleanly.
- **Compounding evidence**: Fourier compounds with Huber (#289), EMA (#381), compile (#401), bf16 (#372) without observable interference. 4 stacked levers all positive.

### Cumulative trajectory (round 1 to date)
| PR | val_avg/mae_surf_p | Δ from prior best |
|---|---|---|
| #287 | 126.67 | (first baseline) |
| #308 | 106.40 | -16.2% |
| #381 |  98.85 |  -7.1% |
| #401 |  66.89 | -32.3% |
| #289 |  63.33 |  -5.3% |
| #368 |  62.94 |  -0.6% |
| **Cumulative** | | **-50.3% from #287, -53% from published-baseline-equivalent** |

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=368 records, 35 lines from final rebased run).

## 2026-04-28 04:00 — PR #453: Linear w_p ramp 0.5 → 1.0 over training
- Branch: `charliepai2d4-fern/pchannel-p-ramp05-10` (still in flight)
- Student: charliepai2d4-fern
- **Outcome: SENT BACK** (mechanism strong, but auto-merge CONFLICTING with #289; +1.1% vs merged baseline within noise).

### Headline (epoch 33 of 33, EMA-evaluated, both runs in PR)
| Run | val_avg/mae_surf_p (EMA) | test_avg/mae_surf_p (EMA) |
|---|---|---|
| baseline-ref (paired) | 66.35 | 58.02 |
| w_p ramp | **64.03** | **56.01** |
| Δ vs paired | **-3.49%** | -3.47% |
| vs merged #289 anchor | +1.1% (within noise) | +1.0% |

### Per-channel mechanism diagnostic (the load-bearing data)
| Channel | baseline-ref | ramp | Δ |
|---|---|---|---|
| val surf_Ux | 1.07 | 0.93 | **-13.1%** |
| val surf_Uy | 0.50 | 0.46 | **-7.3%** |
| val surf_p  | 66.35 | 64.03 | -3.5% |

**~4× velocity:pressure ratio** is the freed-velocity-gradient mechanism signature. Predicted by the physics-coupling argument from #422 post-mortem. Independent confirmation.

### Per-split val Δ (ramp − baseline-ref)
| Split | Δ on mae_surf_p |
|---|---|
| val_single_in_dist     | **-7.72%** (largest) |
| val_geom_camber_rc     | -1.44% (smallest — OOD camber irreducible geometric gap) |
| val_geom_camber_cruise | -3.34% |
| val_re_rand            | -1.24% |

### Same-epoch trajectory (val_avg)
- Epoch 1: ramp +4.86% (low w_p means less pressure focus initially)
- Epoch 13: ramp -6.09% (max gap, w_p ≈ 0.62)
- Epoch 33: ramp -3.49% (gap shrinks as w_p approaches 1.0)
- Ramp monotonically ahead from epoch 2 to 33

### Schedule caveat
- `w_p` only reached **0.827** at the timeout cap (not 1.0).
- Early-velocity phase clearly observed; late-pressure refinement phase only partially executed.
- Fern's analysis: "the early-velocity phase carries most of the value" — consistent with the trajectory data showing peak gain at mid-training.

### Why send back rather than close
- Auto-merge conflicts with #289 (Huber replacing MSE). Manual conflict resolution required.
- Mechanism evidence is strong (per-channel, per-split, per-epoch trajectory all coherent).
- Predicted compounding with Huber (orthogonal levers — Huber=loss shape, w_p_ramp=channel weight) → rebased run highly likely to clear the 63.33 baseline.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=453 records, 35 lines from both runs).

## 2026-04-28 03:50 — PR #436: Additive surface decoder (preds_vol + is_surface * preds_surf)
- Branch: `charliepai2d4-thorfinn/additive-surf-head` (deleted on close)
- Student: charliepai2d4-thorfinn
- **Outcome: CLOSED** (+3.47% vs paired baseline-ref; trunk-interference identified as deeper failure mode).

### Headline (epoch 32 of 32, EMA-evaluated, both runs in PR)
| Run | val_avg/mae_surf_p (EMA) | test_avg/mae_surf_p (EMA) |
|---|---|---|
| baseline-ref (paired) | 65.16 | 57.20 |
| additive-surf-head | **67.43** | 58.65 |
| Δ vs paired | **+3.47%** | +2.54% |
| vs merged #401 anchor | +0.81% (within noise) | +1.37% |

### Per-split val Δ (additive − baseline-ref)
| Split | Δ on mae_surf_p |
|---|---|
| val_single_in_dist     | +2.32% |
| val_geom_camber_rc     | **+5.45%** (worst) |
| val_geom_camber_cruise | +2.22% |
| val_re_rand            | +3.35% |

### Volume regression (the smoking gun)
| Split | mae_vol_p Δ |
|---|---|
| val_geom_camber_rc | **+6.06%** |
| test_geom_camber_rc | **+6.01%** |
- The surf_head literally can't write to volume nodes (mask gates it), but volume metrics regressed → only explainable by trunk-interference.

### Sanity checks (all passed)
| Check | Predicted | Observed | OK |
|---|---|---|---|
| Epoch-1 val_avg parity | within ~1% | +1.72% | ✓ (#379 was +8.3%) |
| Per-epoch wall-clock | ~57-60 s | 55.5 s steady | ✓ |
| Param count | ~677 K (+15 K) | 679,514 (+17 K) | ✓ |
| Dynamo unique graphs | ≤5 | **3** (no recompile) | ✓ |

### Analysis
- **The substitutive→additive fix worked at the design level**: epoch-1 mismatch dropped from #379's +8.3% to this PR's +1.7%. Confirms the zero-init parallel-head pathway starts at parity with the volume head.
- **Trunk interference is the deeper bottleneck**: surf_head reads `fx_pre_final` (penultimate-block output); its gradients flow back through that hidden state and through the entire trunk. Two objectives competing for trunk capacity → worse vol on hard splits AND worse surf at this 32-epoch budget.
- **Adding 17 K params to a parallel head is a poor capacity spend** — same params on the trunk (frieren's #477 wider144) avoid the interference problem entirely.
- **Architectural lesson for round 2**: any "extra capacity for surface" should modulate inside the trunk (e.g. FiLM scale/shift conditioned on `is_surface`), not branch off as a parallel pathway. Assigned next.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=436 records, 34 lines from both runs).

## 2026-04-28 03:30 — PR #431: Moderate widening (n_hidden 128→160) on bf16+EMA pre-#401
- Branch: `charliepai2d4-frieren/wider160-bf16` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (within-noise vs same-pod baseline-ref; doesn't beat current merged baseline #289 at 63.33).

### Headline (epoch 16 vs 18, EMA-evaluated, both runs in PR)
| Run | Best epoch | val_avg/mae_surf_p (EMA) | Δ vs baseline-ref |
|---|---|---|---|
| baseline-ref (n_hidden=128) | 18 | **85.74** | (control) |
| wider160 (n_hidden=160) | 16 | 87.89 | +2.51% (within noise) |

### Same-epoch (16) per-split val (the load-bearing diagnostic)
| Split | baseline-ref e16 | wider160 e16 | Δ |
|---|---|---|---|
| val_single_in_dist     | 100.85 | 99.78 | -1.07 |
| val_geom_camber_rc     | 102.44 | 98.61 | **-3.83** (largest gain) |
| val_geom_camber_cruise |  71.00 | 68.69 | -2.31 |
| val_re_rand            |  87.22 | 84.49 | -2.73 |

### Analysis
- **Capacity hypothesis is supported at fixed-epoch budget**: wider160 beats baseline by uniform ~2.5 mae units across all splits from epoch ~5 onward, with held-out raceCar cambers gaining most (-3.83). This is the mechanism signature the PR predicted.
- **Budget-bounded ranking favors the smaller model**: +14% per-epoch tax (117 s vs 103 s) costs 2 epochs of cosine tail decay. Those 2 missing epochs at low-LR pull baseline ~5 mae lower (epoch 16 → 18).
- **Memory (38 GB) << prediction (50-65 GB)**: the bottleneck is compute, not memory. Future capacity experiments have lots of VRAM headroom.
- **Useful seed/hardware noise floor estimate**: same-pod baseline-ref (85.74) was 13.2% better than the published #381 baseline (98.85). Most of that gap is the 5 extra epochs bf16 buys; some is run-to-run noise. Round-1 variance floor confirmed at ~5pp.

### Pre-#401 / pre-#289 caveat
- Branched after #372 (bf16) but before #401 (compile) and #289 (Huber). Absolute 87.89 is +39% above current merged baseline.
- Frieren's follow-up #2 (n_hidden=144 with throughput recovery from compile) is the right next test — assigned next.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=431 records, 17 lines from wider160 run).

## 2026-04-28 03:10 — PR #289: Huber/SmoothL1 (β=1.0) rebased onto post-#401 — **NEW BASELINE**
- Branch: `charliepai2d4-askeladd/huber-loss` (deleted on merge)
- Student: charliepai2d4-askeladd
- **Outcome: MERGED (squash, commit 906a2c1). NEW BASELINE: val_avg=63.33, -5.31% vs #401.**

### Headline (epoch 32 of 50, EMA-evaluated)
| Metric | Huber + stack | PR #401 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **63.33** | 66.89 | **-5.31%** |
| `test_avg/mae_surf_p` (EMA) | **55.45** | 57.86 | -4.16% |
| Per-epoch wall-clock | 54.4 s (median) | 54.6 s | (matches) |
| Total epochs | 32 | 33 | -1 (slower compile-warmup epoch 1) |
| Peak GPU memory | 23.8 GB | 23.8 GB | unchanged |

### Per-split val (epoch 32, EMA, vs #401)
| Split | Huber+stack | #401 | Δ |
|---|---|---|---|
| val_single_in_dist     | 69.14 | 75.99 | **-9.0%** (heaviest pressure tails — Huber's mechanism) |
| val_geom_camber_rc     | 75.30 | 77.53 | -2.9% |
| val_geom_camber_cruise | 45.70 | 48.50 | -5.8% |
| val_re_rand            | 63.20 | 65.51 | -3.5% |

### Per-split test (post-fix scoring, EMA)
| Split | mae_surf_p | Δ vs #401 |
|---|---|---|
| test_single_in_dist     | 61.05 | -4.5% |
| test_geom_camber_rc     | 68.58 | -2.9% |
| test_geom_camber_cruise | 37.48 | -7.9% (largest test gain) |
| test_re_rand            | 54.69 | -2.7% |

### Analysis
- **Compounding evidence**: Huber on the pre-#308 base gave -9.9%; rebased onto post-#401 (which has EMA+clip), Huber gives -5.31%. The gap (-4.6%) is the part of Huber's gain that EMA already absorbs. Mechanism: both Huber's bounded tail and EMA's late-epoch smoothing dampen variance from heavy-tailed pressure samples; they overlap partially.
- **Per-split pattern preserved**: val_single_in_dist (raceCar high-Re) wins biggest, exactly the mechanism signature.
- **All 4 test splits gain** including test_geom_camber_cruise (now finite post-#358 scoring fix).
- **Late-epoch stability nearly identical to baseline** (last-6-epoch std 1.78 vs 1.95) — EMA is dominating the noise floor; Huber's stabilization advantage narrowed.
- **Variance floor**: at the threshold (~5pp from askeladd's earlier seed estimate), but consistent direction and magnitude across every per-split per-channel breakdown.

### Compile flakiness flag (separate from results)
2 of 4 launches crashed at the rebased stack: CUDAGraph private-pool blowup with `mode="reduce-overhead"` + `dynamic=True` + variable mesh sizes. Askeladd suggests `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` to eliminate this failure mode (~10-15% perf cost). Queued for alphonse's next PR (cosine T_max retune + cudagraph robustness).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=289 records, 34 lines from rebased run).

## 2026-04-28 03:05 — PR #435: Deeper Transolver (n_layers=8) + DropPath 0.1 under compile
- Branch: `charliepai2d4-alphonse/deeper8-droppath01-compile` (deleted on close)
- Student: charliepai2d4-alphonse
- **Outcome: CLOSED** (val_avg=87.43 = +30% vs #401; schedule mismatch dominates).

### Headline (epoch 22 of 22, EMA-evaluated)
| Metric | This run | PR #401 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 87.43 | 66.89 | **+30.7% worse** |
| `test_avg/mae_surf_p` (EMA) | 77.93 | 57.86 | +34.7% |
| Per-epoch | 80.7 s | 54.6 s | +48% |
| Total epochs | 22 | 33 | -11 |
| First compile + forward | 41.1 s | 8.8 s | (under `mode="default"` instead of reduce-overhead) |

### Compile crisis identified by alphonse
- **Round 1 OOM at depth=8** with `mode="reduce-overhead"`: CUDAGraph Trees private pools (one per unique padded mesh size) exceeded 95 GB. 9 distinct sizes, 68.67 GB allocated in private pools → allocator failed.
- **Workaround**: switched to `mode="default"` + `dynamic=True` (skips CUDAGraph capture, keeps Inductor compile + kernel fusion). Trade: ~10-15% throughput cost.
- This is the same flakiness askeladd flagged in #289. Suggests a common fix: `cudagraph_skip_dynamic_graphs=True`.

### Schedule mismatch (the key insight)
- Cosine T_max=50 (configured), but only 22 epochs reachable at depth=8.
- LR at termination: ~0.59× peak — model never reaches the cosine tail's fine-tuning regime that #401 (33 epochs) benefits from.
- Per-epoch loss still descending ~1.5-3 mae units/epoch in the last 5 epochs → even +5-10 epochs wouldn't close the 20-mae gap to baseline at the still-elevated LR.
- **Depth's actual contribution is essentially untestable in this run**: can't distinguish "depth doesn't help" from "depth helps but only after LR settles, which we never reached."

### Why close, what to do next
- Per-split breakdown shows uniform 17-24 mae regression across all 4 val splits — the predicted "held-out cambers gain most" pattern did NOT hold. Cleanly negative on the headline metric.
- Alphonse's follow-up #1 (cosine T_max retune) is the right infrastructure fix and helps every depth/capacity-increasing experiment, not just deeper-8. Assigned next.
- Once T_max retune lands, depth experiments are revisitable (n_layers=6 or 7 first, since they're a better budget compromise than 8).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=435 records, 24 lines).

## 2026-04-28 02:50 — PR #422: Per-channel pressure downweight (w_p = 0.5)
- Branch: `charliepai2d4-fern/pchannel-p-w05` (deleted on close)
- Student: charliepai2d4-fern
- **Outcome: CLOSED** (same-epoch val_avg delta within noise; absolute 85.08 is +27% above current baseline #401's 66.89).

### Headline (epoch 18, no compile — branch predates #401)
| Metric | This run | PR #381 baseline | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA, best) | 85.08 | 98.85 (epoch 13) | -13.9% (mostly from 5 extra epochs due to less node contention) |
| `val_avg/mae_surf_p` (epoch 13, same-epoch) | 97.02 | 98.85 | **-1.85%** (within ~5pp noise floor) |
| `test_avg/mae_surf_p` (EMA) | 75.27 | 87.81 | -14.3% |

### Same-epoch (13) per-channel (the load-bearing finding)
| Channel | #381 epoch 13 | w_p=0.5 epoch 13 | Δ |
|---|---|---|---|
| mae_surf_p | 98.85 | 97.02 | -1.85% (within noise) |
| mae_surf_Ux (4-split mean) | 1.465 | 1.316 | **-10.2%** |
| mae_surf_Uy (4-split mean) | 0.702 | 0.646 | **-7.9%** |
| mae_vol_Ux | 4.298 | 3.990 | -7.2% |
| mae_vol_Uy | 1.872 | 1.740 | -7.0% |
| mae_vol_p | 99.38 | 101.68 | +2.3% (modest regression on vol pressure) |

### Analysis
- **Mechanism supported**: velocity channels gain -7 to -10% at same-epoch — directly from freed gradient mass under reduced pressure weighting. Mirror image of thorfinn's #310 (3× pressure → starves velocity → pressure hurt).
- **Pressure gain is borderline-noise** (-1.85% on val_avg). The "physics coupling" idea would predict pressure improves over time as velocity learns better; the run was timeout-capped at epoch 18 (still descending) so it's plausible that more epochs would show pressure gains.
- **Wall-clock confound**: per-epoch was 104 s vs 142 s baseline. Fern explicitly attributed this to lower node contention; the actual code change is one element-wise multiply, ≈0 cost.
- **All 4 splits improved at the surface across all 3 channels** in the best-ckpt comparison. Generalization on held-out cambers wasn't damaged.

### Why close (not send back)
- Same-epoch val_avg is within ~5pp variance floor → not a clear winner on the primary metric.
- Absolute 85.08 doesn't beat current baseline #401 (66.89, has compile).
- Even if rebased onto #401, the small same-epoch effect predicts only ~65.7 (≈3% off 66.89, within noise).
- Mechanism finding is captured for future work. **Fern's own follow-up #2** (ramp w_p 0.5 → 1.0 over training) is the cleaner test that decouples early/late training regimes — assigned next.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=422 records, 19 lines).

## 2026-04-28 02:35 — PR #421: EMA(0.995) + NO clip (attribution ablation)
- Branch: `charliepai2d4-nezuko/ema995-noclip` (deleted on close)
- Student: charliepai2d4-nezuko
- **Outcome: CLOSED** (val_avg=109.99 vs #381's 98.85 = +11.3%, well outside variance floor).

### Headline (epoch 18 of 50, EMA-evaluated, no compile — branch predates #401)
| Metric | This run | PR #381 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 109.99 | 98.85 | **+11.3% worse** |
| `test_avg/mae_surf_p` (EMA) | 98.58 | 87.81 | +12.3% worse |
| Per-epoch | ~104 s | 142 s | (bf16 from #372 stacked, just no compile) |
| Total epochs | 18 | 13 | (more epochs, still worse) |

### Per-split val Δ vs PR #381 (no-clip − clip=10)
| Split | Δ |
|---|---|
| val_single_in_dist     | +10.2% |
| val_geom_camber_rc     | +5.5% |
| val_geom_camber_cruise | **+22.1%** (largest hit) |
| val_re_rand            | +11.3% |

### Attribution conclusion (the load-bearing finding)
1. **gn_max with no clipping doesn't explode** (max 564 vs #381's 767). Clip's role was NOT runaway-gradient suppression at our AdamW/lr/loss settings.
2. **Clip is acting as a per-batch dampener** that materially helps generalization. Removing it regresses broadly across all 8 splits, largest on the smallest-magnitude cruise camber split (+22%) — overshoot signature.
3. **EMA alone (decay=0.995, no clip) gives 109.99** — worse than even #308's 106.40 (decay=0.999, clip=1.0). The two stabilizers compose; neither alone reproduces the joint effect.
4. **Round-2 compound is clear**: surf_weight=25 (alphonse #287) + EMA(0.995) + clip(10.0) (#381) all established as load-bearing. PR #437 assigned to nezuko to test the compound.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=421 records, 19 lines).

## 2026-04-28 02:15 — PR #401: torch.compile + bf16 + EMA + clip — **NEW BASELINE (huge jump)**
- Branch: `charliepai2d4-alphonse/compile-bf16-emaclip` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit 5f2edca). NEW BASELINE: val_avg/mae_surf_p = 66.89, -37.1% vs #308, -32.3% vs #381.**

### Headline (epoch 33 of 50, EMA-evaluated)
| Metric | Value | vs #308 | vs #381 |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **66.89** | **-37.1%** | -32.3% |
| `test_avg/mae_surf_p` (EMA) | **57.86** | -38.4% | -34.1% |
| Median per-epoch | **54.6 s** | 2.58× faster | 2.6× faster |
| Total epochs | **33** | +20 | +20 |
| Peak GPU memory | 23.8 GB | -43% (lots of headroom) | -43% |
| First compile overhead | 8.8 s | (well below 60-180 s prediction) | |

### Per-split val (epoch 33, EMA)
| Split | mae_surf_p | Δ vs #308 |
|---|---|---|
| val_single_in_dist     | 75.99 | -41.7% |
| val_geom_camber_rc     | 77.53 | -35.2% |
| val_geom_camber_cruise | 48.50 | -39.9% |
| val_re_rand            | 65.51 | -30.9% |

### Per-split test (post-fix scoring, EMA)
| Split | mae_surf_p | Δ vs #308 |
|---|---|---|
| test_single_in_dist     | 63.90 | -43.3% |
| test_geom_camber_rc     | 70.64 | -32.0% |
| test_geom_camber_cruise | 40.68 | -38.7% |
| test_re_rand            | 56.21 | -39.5% |

### Compile diagnostics (the key win)
- **Only 3 unique dynamo graphs** under `dynamic=True` — one set covers all 74K-242K mesh sizes. The 'dynamic-mode recompilation eats the speedup' worry didn't materialize.
- **Per-epoch dropped 141 s → 54.6 s** (PR #308 vs this). That's **2.58×**. With 30-min cap, total epochs went 13 → 33.
- **No graph breaks, no eager fallback.** `torch._dynamo.utils.counters['stats']['unique_graphs']=3, frames.total=3, frames.ok=3`.
- **CUDA Graph re-recording** (`cudagraph_recorded_non_static_inputs=5838`) runs per concrete shape but is much cheaper than dynamo recompilation; peak memory bounded at 23.8 GB.

### Analysis
- **Cosine schedule's tail finally reachable.** PR #308 stopped at epoch 13 with cosine still near peak; #401 reaches epoch 33, well into the decay regime. EMA + small lr is the dominant late-training mechanism, and we've finally given it room to operate.
- **Best epoch = 33 = last epoch.** Curve was still descending. There's likely more available with longer wall-clock; round 2 may want to revisit budget allocation.
- **Implementation choices were all correct**: hold `_model_base`/`_ema_base` references for parameters/state_dict (avoids `_orig_mod.` prefix), `dynamic=True` collapses mesh-size variation cleanly, `reduce-overhead` was the right mode.
- **Round-1 character changed.** "14-epoch ranking exercise" → "33-epoch ranking exercise". Architectural-scale PRs that previously busted budget (wider, deeper) should be revisited.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=401 records, 35 lines).

## 2026-04-28 02:10 — PR #379: Surface-aware decoder head (substitutive design)
- Branch: `charliepai2d4-thorfinn/surface-aware-decoder` (deleted on close)
- Student: charliepai2d4-thorfinn
- **Outcome: CLOSED** (-0.52% vs own baseline-ref — within noise; +94% vs new merged baseline 66.89).

### Headline (epoch 14, EMA-evaluated, both runs in PR)
| Run | val_avg | test_avg | Δ |
|---|---|---|---|
| baseline-ref | 129.13 | 117.97 | (matched control) |
| surface-aware decoder | 128.46 | 117.30 | -0.52% / -0.57% (within ~5pp variance floor) |

### Per-split val (surf-decoder − baseline-ref)
| Split | Δ |
|---|---|
| val_single_in_dist     | **-5.93%** (helps) |
| val_geom_camber_rc     | -2.36% (helps) |
| val_geom_camber_cruise | **+8.58%** (hurts) |
| val_re_rand            | +1.73% (hurts) |

### Analysis
- **Mechanism identified by thorfinn**: substitutive design (`torch.where(is_surface, surf_pred, vol_pred)`) plus zero-init wastes the volume head's pretrained signal on surface nodes. The surface head has to re-derive what the backbone already knows; at 14 epochs that re-derivation eats most of the gain.
- **Per-split pattern is informative**: surface decoder helps high-amplitude raceCar splits (where surface pressure has sharp suction peaks needing extra capacity), hurts low-amplitude cruise splits (where the volume head's prediction was already correct and the surface head erased it).
- **Right idea, wrong design**: thorfinn's suggested fix is additive (`preds = preds_vol + is_surface[..., None] * preds_surf`). Cleaner test of the same hypothesis. Assigned as follow-up.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=379 records, 15 lines).

## 2026-04-28 02:08 — PR #368: Fourier positional encoding (rebased onto post-#308) — sent back AGAIN
- Branch: `charliepai2d4-edward/fourier-pos-encoding` (still in flight)
- Student: charliepai2d4-edward
- **Outcome: SENT BACK** (rebase to post-#401 + re-run; under-trained at 10 epochs due to GPU contention).

### Headline (epoch 10 of 14, EMA-evaluated)
| Metric | Value | Notes |
|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 110.89 | (10 epochs only, +4.2% vs #308 at 13 epochs) |
| `test_avg/mae_surf_p` (EMA) | 100.61 | |
| Per-epoch (clean) | ~141 s | matches #308 — Fourier features add ~8K params, negligible cost |

### Equal-epoch comparison vs PR #308 (the load-bearing data)
| Epoch | PR #308 EMA+clip | Fourier+EMA+clip | Δ |
|---|---|---|---|
| 5  | 194.60 | 183.99 | -5.4% |
| 7  | 156.64 | 144.33 | -7.9% |
| 9  | 132.65 | 119.60 | -9.8% |
| **10** | **124.63** | **110.89** | **-11.0%** |

### Analysis
- **Compounding signal is real.** Equal-epoch trajectory diverges in Fourier's favor from epoch 3 onward, gap grows to -11.0% at epoch 10. Extrapolating the same e10→e13 trajectory to PR #308's 13 epochs would land at ~92-95 — well beating the 106.40 baseline at #308's time.
- **GPU contention robbed 3 epochs.** First 3 epochs at 302/299/256s instead of 141s due to a sibling launch's orphan run (acknowledged in edward's writeup; per launch-isolation rules I won't reuse that run's number).
- **Decision: send back, NOT merge** — at 10 epochs the absolute number doesn't beat baseline, and we now have a much stronger baseline (PR #401, 66.89). Edward's branch needs to rebase onto post-#401 to fairly evaluate Fourier+compile+EMA. If the equal-epoch pattern holds with compile's 33-epoch budget, the rebased run could land in the 58-64 range.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=368 records, 11 lines).

## 2026-04-28 01:50 — PR #382: Larger batch (bs=8) + sqrt-scaled lr (7e-4)
- Branch: `charliepai2d4-frieren/batch8-lr7e-4` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (GPU compute-saturated at bs=4; per-epoch only -1.8%; EMA artifact regressed val by 52%).

### Headline (epoch 13/13, EMA-evaluated)
| Metric | bs=4 baseline-ref | bs=8 lr=7e-4 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | 109.66 | **167.20** | **+52.5% worse** |
| `test_avg/mae_surf_p` (EMA) | 97.26 | 150.37 | +54.6% |
| Per-epoch wall-clock | 141.5 s | 138.9 s | **−1.8%** (no throughput win) |
| Total epochs | 13 | 13 | 0 (zero extra epochs in budget) |
| Peak GPU memory | 42.1 GB | 84.2 GB | +100% |

### Two clean findings
1. **GPU compute-saturated at bs=4 on this model+hardware.** Doubling batch ≈ doubles per-step time when the GPU is already saturated; net compute per epoch stays flat. Memory headroom doesn't translate to throughput. This is exactly why PR #372 (bf16 autocast) drops per-epoch time — it cuts per-step compute, not parallelism.
2. **EMA averaging artifact at bs=8 with decay=0.999.** Half the steps means decay=0.999 covers ~5.3 epoch-equivalents of weight history (vs ~2.7 at bs=4), so EMA is biased toward early-noisy weights. Online (non-EMA) val regressed only ~12% vs the 52% EMA regression — the lion's share of the regression is EMA dynamics, not slower convergence. Frieren's principled fix: lower decay to ~0.998 or 0.997 to keep effective averaging window fixed in batches.

### Per-split val Δ (bs=8 − bs=4)
| Split | Δ |
|---|---|
| val_single_in_dist     | **+85% worse** |
| val_geom_camber_rc     | +42% |
| val_geom_camber_cruise | +41% |
| val_re_rand            | +32% |

### Lessons
- Memory headroom should go to capacity, not throughput, on this hardware.
- Batch-scaling experiments need EMA decay retuning for the new step count.
- bf16 autocast (PR #372) is the right way to cash in memory for throughput; it drops per-step compute. This is now in the merged baseline; future PRs implicitly benefit.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=382 records, 14 lines).

## 2026-04-28 01:30 — PR #381: EMA decay 0.995 + grad clip 10.0 — **NEW BASELINE**
- Branch: `charliepai2d4-nezuko/ema995-gradclip10` (deleted on merge)
- Student: charliepai2d4-nezuko
- **Outcome: MERGED (squash, commit a620ba1). NEW BASELINE: val_avg/mae_surf_p = 98.85, -7.1% vs #308.**

### Headline metrics (epoch 13 of 13, EMA-evaluated)
| Metric | Value | vs #308 baseline |
|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **98.85** | **-7.10%** |
| `test_avg/mae_surf_p` (EMA) | **87.81** | -6.58% |
| All 4 splits improve | -3.6% to -10.3% | (cleanest gain on val_single_in_dist and val_geom_camber_cruise) |

### EMA crossover
| Epoch | EMA−online | Notes |
|---|---|---|
| 1 | +5.23 | EMA lags |
| **2** | **−15.83** | **First crossover** (vs ~epoch 10 in #308) |
| 5 | −57.46 | EMA leads strongly |
| 13 | −15.13 | EMA settled in lead |

### Clip behavior at max_norm=10
| Epoch | gn_mean | gn_max | clip% |
|---|---|---|---|
| 1 | 107.55 | 484.51 | 99.5% |
| 7 |  66.59 | 719.29 | 97.1% |
| 13 |  44.16 | 406.11 | 87.5% |

### Analysis
- **Faster EMA decay clearly works**: crossover at epoch 2 vs ~10 in #308. The EMA spends 11 epochs in the useful lagging-but-leading regime instead of ~3.
- **Clip is NOT cleanly released**: at max_norm=10, clip% is 87-100%. So the −7.1% win is the joint effect of (faster EMA) + (10× looser dampening), not pure EMA. Attribution is still ambiguous.
- **Clip role qualitatively shifted**: at threshold 10, gn_mean (44-107) is mostly above the threshold but gn_max (344-767) is dramatically above — so most batches are dampened lightly and rare outliers are dampened hard. Closer to clip's intended outlier-protection role than #308's blanket unit-norm at threshold=1.
- **Mechanism hypothesis confirmed**: nezuko predicted faster decay would push EMA into useful regime sooner; the EMA−online divergence trace (down to −57 at epoch 5) is direct evidence.

### Attribution still open — ablation queued
PR (nezuko's next) will run **EMA-only (decay=0.995, no clip)** to isolate EMA's standalone contribution. If 98.85 holds without clip, EMA is doing the work; if it tanks, clip-as-dampener is doing more than expected.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=381 records, 14 lines).

## 2026-04-28 01:25 — PR #388: Arcsinh pressure target reparametrization
- Branch: `charliepai2d4-fern/arcsinh-pressure` (deleted on close)
- Student: charliepai2d4-fern
- **Outcome: CLOSED** (+15.1% regression on val_avg, +17.5% on test_avg).

### Headline (epoch 13 of 13, EMA-evaluated, on post-#308 baseline)
| Metric | Value | vs #308 baseline |
|---|---|---|
| `val_avg/mae_surf_p` | 122.50 | **+15.1% worse** |
| `test_avg/mae_surf_p` | 110.40 | +17.5% worse |

### Per-split val Δ vs PR #308
| Split | Δ |
|---|---|
| val_single_in_dist     | **+28.4% worse** |
| val_geom_camber_rc     | **+22.2% worse** |
| val_geom_camber_cruise | **-10.3% better** |
| val_re_rand            | +9.6% worse |

### Analysis
- **Mechanism identified by fern**: encoded-space loss precision doesn't translate to physical-space MAE precision uniformly across the |p| range. For p=2000 (raceCar high-Re), encoded p=8.29; a 0.1 normalized-encoded error decodes through `sinh` (Jacobian ≈ |p| at large |p|) to ~190 physical MAE.
- **Trade is fundamentally wrong-direction for the metric**: equal-weight 4-split MAE has 3 of 4 splits dominated by raceCar, so amplifying tail precision losses there sinks val_avg.
- **Clean negative result with sharp post-mortem.** Fern's three suggested follow-ups all useful: (1) per-channel y_std rescale w_p<1 — assigned next; (2) per-sample target normalization — overlaps with tanjiro's #378; (3) per-split-weighted checkpoint selection — parking as round-2 lever.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=388 records, 14 lines).

## 2026-04-28 01:05 — PR #372: bf16 autocast on forward (throughput infrastructure)
- Branch: `charliepai2d4-alphonse/bf16-autocast` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit 91d8a4e). Infrastructure merge — BASELINE stays at #308's 106.40.**

### Headline metrics (epoch 19 of 50, timeout-capped)
| Metric | bf16+surf_w=25 (no EMA) | alphonse #287 (no bf16, surf_w=25, no EMA) | #308 baseline (EMA, surf_w=10) |
|---|---|---|---|
| `val_avg/mae_surf_p` | 108.93 | 126.67 | **106.40** |
| `test_avg/mae_surf_p` | 99.16 | 114.88 | 93.99 |
| Per-epoch wall-clock | **97.3 s** | 131.8 s | 141 s |
| Epochs in 30-min cap | **19/50** | 14/50 | 13/50 |
| Peak GPU memory | **33.0 GB** | 42.1 GB | 42.1 GB |

### Why merge as infrastructure (not strict metric win)
- 108.93 is +2.4% above the canonical baseline (106.40), but that's **within the ~5pp variance floor** established by askeladd's two Huber seeds.
- Equal-config Δ vs alphonse #287: **-14%** on identical surf_weight=25/no-EMA setup. Clean throughput-axis win.
- Three-way merge with #308's EMA composes cleanly — autocast wraps `model({"x": x_norm})["preds"]` which now applies to both live model (training) and EMA model (evaluation).
- BASELINE.md keeps 106.40 as canonical metric; bf16 is now standing infrastructure in train.py.

### Per-split val (epoch 19, no EMA)
| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist     | 134.11 | 155.67 |
| val_geom_camber_rc     | 115.26 | 130.40 |
| val_geom_camber_cruise |  84.19 | 108.35 |
| val_re_rand            | 102.17 | 114.85 |

### Per-split test (post-fix scoring)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 118.15 |
| test_geom_camber_rc     | 106.14 |
| test_geom_camber_cruise |  72.74 |
| test_re_rand            |  99.60 |

### Analysis
- **Throughput was the bottleneck, exactly as predicted.** 1.36× speedup (slightly below the 1.5-2× target — Transolver's softmax/layernorm stay in fp32 under autocast, and dataloader/host transfer don't accelerate). Memory drops 22% from halved bf16 activations.
- **5 extra epochs of cosine-decayed training** is what produces the −14% equal-config improvement; the cosine tail does the heavy lifting.
- **Run was still descending at epoch 19** (val_avg=108.93 was the last epoch and a new best). T_max=50 means we're seeing only 38% of the cosine decay; even more headroom.
- alphonse correctly identified the cosine T_max retuning follow-up — set T_max ≈ achievable epoch count so the lr decay actually finishes.

### Implications for in-flight PRs
- **#381 nezuko, #382 frieren, #378 tanjiro, #379 thorfinn, #388 fern, #289/368 (rebasing askeladd/edward)**: depending on when each branch was created, they may or may not have bf16. New assignments after this merge inherit it automatically.
- The throughput recovery makes round 2 PRs (compounding wins) much more attractive — we can now run more epochs and see the cosine schedule converge.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=372 records, 20 lines).

## 2026-04-28 00:55 — PR #289: SmoothL1 (Huber β=1.0) loss replacing MSE
- Branch: `charliepai2d4-askeladd/huber-loss` (sent back to draft)
- Student: charliepai2d4-askeladd
- **Outcome: SENT BACK** (clean -9.9% on loss-formulation axis; rebase+re-run with EMA+clip).

### Headline (epoch 14 of 14, askeladd's own A/B)
| Metric | MSE baseline-ref (askeladd) | Huber β=1.0 | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 128.49 | **115.76** | **-9.9%** |
| Best epoch | 13 | 14 | — |

### Per-split val Δ (Huber − MSE, primary metric only)
| Split | Δ on mae_surf_p |
|---|---|
| val_single_in_dist     | -7.6% |
| val_geom_camber_rc     | -5.5% |
| val_geom_camber_cruise | **-16.2%** (largest gain) |
| val_re_rand            | -12.9% |

### Late-epoch stability (epochs 9-14)
| | std | max-min |
|---|---|---|
| MSE | 17.6 | 45.4 |
| Huber | 8.0 | 19.5 |

### Analysis
- **Cleanly positive on the loss-formulation axis.** Predicted -5% to -10% on val_avg; got -9.9%. Every per-split per-channel metric improves; velocity channels gain even more relatively (-15% to -38%) because Huber's median-tracking generalizes across all channels.
- **Cruise wins biggest** (-16% on cruise mae_surf_p) — exactly the mechanism askeladd predicted: under MSE, low-amplitude cruise samples are out-competed for gradient by high-Re raceCar tails; Huber's bounded tail evens the playing field.
- **2× lower late-epoch variance** confirms Huber's stabilization claim.
- **Variance floor established**: askeladd's two Huber seeds spread by ~5pp on val_avg. **Round-1 winners by less than ~5% are within run-to-run noise** — keep this in mind for ranking decisions going forward.
- **Branch predates #308.** This run lacks EMA+clip — same situation as edward's #368. Rebased re-run will tell us whether Huber and EMA compound (expected: yes; predict 95-105 range).
- Independent diagnosis of the inf-y bug (now fixed in #358).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=289 records, 15 lines from initial run).

## 2026-04-28 00:50 — PR #368: Fourier positional encoding on (x,z): 8-freq sin/cos
- Branch: `charliepai2d4-edward/fourier-pos-encoding` (sent back to draft)
- Student: charliepai2d4-edward
- **Outcome: SENT BACK** (rebase to post-#308 advisor + re-run with EMA+clip).

### Headline (epoch 14 of 14, timeout-capped)
| Metric | Value | vs equal-config baseline |
|---|---|---|
| `val_avg/mae_surf_p` | 117.68 | better than alphonse #287 at epoch 14 (126.67) — Fourier features show positive signal |
| `val_avg/mae_surf_p` vs current baseline | 117.68 | +10.6% worse than #308 (106.40) — but the comparison isn't fair (no EMA/clip on this branch) |
| `test_avg/mae_surf_p` | 108.69 | post-#358 scoring fix → all 4 splits finite |
| Per-epoch time | 132 s | comparable to baseline (141 s) |
| Peak GPU memory | 42.5 GB | unchanged |
| n_params | 670,551 (+8K vs baseline from wider preprocess) | within budget |

### Per-split val (epoch 14)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     | 133.28 |
| val_geom_camber_rc     | 127.03 |
| val_geom_camber_cruise |  93.21 |
| val_re_rand            | 117.18 |

### Analysis
- **Fourier features showed positive signal**: 117.68 at epoch 14 beats alphonse #287's 126.67 at the same epoch on the same surf_weight=10 config (~7% gain).
- **Branch predates the new baseline:** edward branched after #358 but before #308. So this run lacks EMA + grad clip. Merging Fourier features with the post-#308 train.py is exactly the experiment we need.
- **Edward identified an open follow-up bug**: normalized-space loss in `evaluate_split` propagates NaN on the cruise test split for losses, even though the MAE side is now clean post-#358. Cosmetic for ranking but worth fixing eventually.
- **Sent back rather than closed**: the rebased run is a clean test of "do Fourier features compound with EMA?" Expected outcome range: 95-110 if compounding works, ~110-115 if EMA was doing all the work.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=368 records, 15 lines from initial run; rebased run will produce a new set).

## 2026-04-28 00:40 — PR #304: Deeper Transolver: n_layers 5->8 with DropPath 0.1
- Branch: `charliepai2d4-fern/deeper-model-droppath` (deleted on close)
- Student: charliepai2d4-fern
- **Outcome: CLOSED** (per-epoch wall-clock too high → only 9/50 epochs; worse than baseline at equal-epoch).

### Headline (epoch 8 of 9 completed, timeout-capped)
| Metric | Value | vs. baseline |
|---|---|---|
| `val_avg/mae_surf_p` | 159.62 | +50% vs PR #308 (106.40) |
| `val_avg/mae_surf_p` (epoch 8 equal-budget) | 159.62 | +11% vs nezuko #308 epoch 8 (143.61, online weights) |
| `test_avg/mae_surf_p` | NaN¹ → 163.23 (3 splits) | — |
| Per-epoch time | **210 s** | vs nezuko 141 s — **~1.5× slower** |
| Peak GPU memory | 64.5 GB | within budget |

¹ Branch predates PR #358 scoring fix; same inf-y sample 20 in test_geom_camber_cruise.

### Per-split val (epoch 8)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     | 199.65 |
| val_geom_camber_rc     | 178.49 |
| val_geom_camber_cruise | 118.37 |
| val_re_rand            | 141.97 |

### Analysis
- **Depth + DropPath integration is healthy** — no NaN, no divergence, val curve descended monotonically through epoch 8.
- **Throughput is the kill criterion (again)**. Same lesson as edward's #300 (wider) and tanjiro's #309 (more slices): per-epoch cost above ~150s makes experiments uncompetitive on absolute val_avg in the 30-min cap.
- **Worse at equal-epoch**: fern epoch 8 (159.62) vs nezuko epoch 8 online (143.61) — depth doesn't even win where wall-clock is matched.
- **Independent diagnosis** of the inf-y bug — sixth student to hit it. All resolved by PR #358.
- **Parking depth as round-2** — once throughput-recovery PRs land (#372 bf16, #382 larger batch), n_layers=8 fits more comfortably.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=304 records, 10 lines).

## 2026-04-28 00:25 — PR #308: EMA (decay 0.999) + grad clip max_norm 1.0 — **NEW BASELINE**
- Branch: `charliepai2d4-nezuko/ema-grad-clip` (deleted on merge)
- Student: charliepai2d4-nezuko
- **Outcome: MERGED (squash, commit 5bdb284). NEW BASELINE: val_avg/mae_surf_p = 106.40, -16.2% vs PR #287.**

### Headline metrics (epoch 13 of 14, EMA-evaluated)
| Metric | Value | vs #287 baseline | vs published |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **106.40** | **-16.2%** | best on branch |
| `test_avg/mae_surf_p` (EMA) | **93.99**  | -18.2% (vs #287 114.88) | best on branch |
| Wall-clock | ~33 min total | comparable | |
| Peak GPU memory | 42.1 GB | unchanged | |

### Per-split val (epoch 13, EMA)
| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist     | 130.44 | 133.99 |
| val_geom_camber_rc     | 119.63 | 120.78 |
| val_geom_camber_cruise |  80.75 |  74.44 |
| val_re_rand            |  94.78 |  91.91 |

### Per-split test (post-fix scoring, EMA)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 112.78 |
| test_geom_camber_rc     | 103.87 |
| test_geom_camber_cruise |  66.35 |
| test_re_rand            |  92.98 |

### Analysis
- **Monotonically descending**: 322 → 106.40 across 13 epochs, every epoch a new best. No instability.
- **EMA pays off late**: from epoch 10 onward EMA is consistently 13-20 units better than online weights (nezuko's own diagnostic). Epochs 1-4 the EMA lags online (decay=0.999 warmup).
- **Crucial caveat — clipping is a hidden lr dampener.** `max_norm=1.0` clips 100% of batches; pre-clip gn_mean ≈ 50-100 vs threshold 1.0. So the optimizer is doing essentially unit-norm SGD on top of AdamW. The 16% gain therefore cannot be cleanly attributed to EMA alone. **Ablations queued.**
- **Compounds with #287**: surf_weight=25 (alphonse) and EMA+clip (nezuko) are independent changes; combination is a clear round-2 candidate.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=308 records, 14 lines).

## 2026-04-28 00:20 — PR #307: 5-epoch linear warmup + cosine schedule, peak lr 1e-3
- Branch: `charliepai2d4-frieren/warmup-cosine-1e3` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (val_avg=134.58, ~26% worse than the new baseline 106.40).

### Headline (epoch 13 of 14, timeout-capped)
| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 134.58 |
| `test_avg/mae_surf_p` | 123.24 |

### Analysis
- **Warmup itself is stable** — 1e-3 with 5-epoch linear warmup did not diverge; only a small blip at epoch 5 when full lr fires (recovers in one epoch).
- **Cosine never decays in the budget**: T_max=50 means lr only drops 1.0 → 0.924 across 14 epochs. The "warmup→peak→decay" cycle the hypothesis predicted never plays out — this run is essentially constant-peak-lr.
- **Below baseline anyway**: even with stable warmup at 2× peak lr, the run lags the prior baseline (PR #287, 126.67) and far behind the new baseline (PR #308, 106.40).
- Independent diagnosis of the same inf-y test bug (now fixed in PR #358).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=307 records, 15 lines).

## 2026-04-28 00:05 — PR #310: Per-channel surface loss weights: 3x weight on surface pressure
- Branch: `charliepai2d4-thorfinn/per-channel-surf-weights` (deleted on close)
- Student: charliepai2d4-thorfinn
- **Outcome: CLOSED** (+13.0% regression on val_avg/mae_surf_p).
- Hypothesis: weight surface-p 3× over surface-Ux/Uy in the loss to bias optimization toward the metric. Predicted Δ -3% to -8%.

### Headline metrics (epoch 13/14, both runs timeout-capped at 14 epochs)
| Metric | thorfinn baseline-ref (sw=10, p_w=1) | thorfinn 3x p (sw=10, p_w=3) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 130.54 (epoch 12) | **147.56** (epoch 13) | **+13.0% worse** |
| `val_avg/mae_surf_Ux` | 2.13 | 2.88 | +35.2% |
| `val_avg/mae_surf_Uy` | 0.92 | 1.09 | +18.7% |
| `val_avg/mae_vol_p` | 131.24 | 153.05 | +16.6% |
| `test_avg/mae_surf_p` | NaN¹ | NaN¹ | — |

¹ Branch predates PR #358 scoring fix; same inf-y sample 20 in test_geom_camber_cruise.

### Per-split val (3x − baseline)
| split | mae_surf_p Δ |
|---|---|
| val_single_in_dist | +7.0% |
| val_geom_camber_rc | +36.9% |
| val_geom_camber_cruise | -1.5% |
| val_re_rand | +8.0% |

### Analysis
- **Hypothesis cleanly disproved**: 3 of 4 val splits regressed materially, only `val_geom_camber_cruise` saw a tiny improvement (-1.5%). Surface velocities also degraded — the gradient-mass-starvation story (heavy weight on p starves Ux/Uy learning, which feeds back into pressure via Navier-Stokes coupling) is the most plausible explanation.
- **Useful side data:** thorfinn's own baseline-ref run (surf_weight=10, p_weight=1) gave val_avg=130.54 at epoch 12, **independently confirming** that alphonse's PR #287 (val_avg=126.67 at surf_weight=25) is a ~3% improvement over surf_weight=10 at the same wall-clock.
- **Lesson:** with the existing 10× surf_weight, layering more p-specific weight on top dominates the gradient signal too aggressively in a 14-epoch budget regime. If revisited, try `surf_weight=5, surf_p_weight=2` so total surface-p effective weight matches baseline.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=310 records, 15 lines).

## 2026-04-27 23:55 — PR #309: More slice tokens: slice_num 64->128, n_head 4->8
- Branch: `charliepai2d4-tanjiro/more-slices` (deleted on close)
- Student: charliepai2d4-tanjiro
- **Outcome: CLOSED** (slower per epoch and not better at equal-epoch comparison).
- Hypothesis: doubling slice tokens + heads (with halved head_dim) gives more physical-regime "slots". Predicted Δ -3% to -7%.

### Headline metrics (epoch 6 of 50, timeout-capped at 8 epochs / 33.4 min)
| Metric | Value | vs. alphonse #287 baseline |
|---|---|---|
| `val_avg/mae_surf_p`  | 168.47 (epoch 6) | +33% vs 126.67 — but unfair due to fewer epochs |
| `val_avg/mae_surf_p` (epoch 8 equal-budget) | 170.54 | +4.5% vs alphonse epoch 8 (163.24); alphonse used surf_weight=25 |
| `test_avg/mae_surf_p` | 154.97 (post-fix scoring) | — |
| Per-epoch time | **250 s** | vs alphonse 131 s — **~1.9× slower** |
| Peak GPU memory | 82.3 GB | within budget but high |

### Analysis
- **Throughput is the kill criterion.** At 250 s/epoch only 8 epochs fit the 30-min cap (vs alphonse's 14). Even granting the full architectural advantage at equal-epoch (which doesn't show up — tanjiro is *worse* at epoch 8), the total run is fewer-epoch-and-no-better.
- **Late-epoch instability:** epoch 6→7 jumped 168→197 then 170 at epoch 8. Plausibly head_dim=16 (n_head=8 with n_hidden=128) is too narrow.
- **Independent bug diagnosis** matched edward's earlier finding on `data/scoring.py` (`inf * 0 = NaN`). Tanjiro applied a workaround in `train.py::evaluate_split` to drop bad samples before scoring; the proper fix landed in PR #358.
- **Tanjiro's suggested decomposition** (slice-count alone vs head-count alone) is the right experimental design IF we ever return to this axis. Parking for now since the binding constraint is throughput, not slice count.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=309 records, 9 lines).

## 2026-04-27 23:50 — PR #287: surf_weight 10 -> 25 to refocus loss on surface MAE
- Branch: `charliepai2d4-alphonse/surf-weight-up` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit e4a0c18). First baseline on this branch.**
- Hypothesis: Raise surf_weight to direct gradient toward surface error. Predicted Δ -3% to -7%.

### Headline metrics (epoch 14 of 50, timeout-capped at 30 min)
| Metric | Value |
|---|---|
| `val_avg/mae_surf_p`  | **126.67** |
| `test_avg/mae_surf_p` | **114.88** (corrected; 1 NaN-y test sample skipped) |
| Wall-clock | 30.8 min train + 20s test |
| Peak GPU memory | 42.1 GB / 96 |

### Per-split val
| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist     | 155.79 | 172.08 |
| val_geom_camber_rc     | 134.23 | 147.34 |
| val_geom_camber_cruise |  98.89 | 131.81 |
| val_re_rand            | 117.77 | 133.50 |

### Per-split test (post-NaN-fix)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 136.43 |
| test_geom_camber_rc     | 124.14 |
| test_geom_camber_cruise |  83.63 |
| test_re_rand            | 115.33 |

### Analysis
- **Curve was monotonically descending through epoch 14** (256.83 → 126.67), and the last 3 epochs (153.65 → 140.48 → 126.67) all set new bests. Run was meaningfully under-trained — would likely drop further if the cosine tail completed.
- **Volume MAE not pathologically harmed** (vol_p ≈ 146 mean across val). The surf_weight bump didn't starve the volume branch.
- **Test < val**: 114.88 < 126.67. The cruise test split is ~15 pts better than its val counterpart. Plausibly small-val-set noise (val 100 vs test 200), but worth watching as more PRs land.
- **Important meta-observation from alphonse**: round 1 is effectively a **14-epoch** ranking exercise rather than 50-epoch. Every comparison this round inherits this caveat. BASELINE.md updated to flag.
- Independent diagnosis of the scoring NaN bug, exactly matching edward's earlier finding.

JSONL summary: `research/EXPERIMENT_METRICS.jsonl` (PR=287 records, 16 lines).

## 2026-04-27 23:30 — PR #358: Maintenance: fix data/scoring.py NaN propagation through inf*0 mask
- Branch: `charliepai2d4-edward/fix-scoring-nan-mask` (deleted on merge)
- Student: charliepai2d4-edward
- **Outcome: MERGED (squash, commit 010235e).**

Maintenance fix, not an experiment. Replaces `err * mask_float` with `torch.where(mask, err, 0)` in `data/scoring.py::accumulate_batch`. New `data/test_scoring.py` with 4 tests covers: inf-in-p (reproduces test_geom_camber_cruise sample 20 failure mode), NaN-in-y, bit-equality on all-finite inputs (no-op proof), end-to-end finalized-MAE finiteness. All 4 pass. Empirical OLD vs NEW on the same inf-injected batch confirms the fix. Diff: 3 additions / 2 deletions in scoring.py + 104-line test file.

This unblocks `test_avg/mae_surf_p` for every other in-flight PR — they will now produce a finite test ranking metric. Existing in-flight branches will need to either rebase to pick up this fix, or accept that `test_geom_camber_cruise/mae_surf_p` will remain NaN until the scoring change reaches their runtime.

## 2026-04-27 23:20 — PR #300: Wider Transolver: n_hidden 128->192, slice_num 64->96
- Branch: `charliepai2d4-edward/wider-model` (deleted on close)
- Student: charliepai2d4-edward
- Hypothesis: Increase Transolver capacity (n_hidden 128→192, slice_num 64→96, ~1.48 M params) on the under-utilized 96 GB GPU. Predicted Δ on `val_avg/mae_surf_p`: -5% to -10% vs. published baseline.
- **Outcome: CLOSED** (under-trained, not directly comparable to baseline).

### Best validation metrics (epoch 9 of 50; training stopped at 30-min cap)
| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 193.90 | 2.510 | 0.982 | 174.91 |
| val_geom_camber_rc     | 157.52 | 3.349 | 1.300 | 147.12 |
| val_geom_camber_cruise | 104.49 | 2.052 | 0.669 |  99.36 |
| val_re_rand            | 125.49 | 2.583 | 0.937 | 118.76 |
| **val_avg**            | **145.35** | 2.624 | 0.972 | 135.04 |

### Test metrics (best ckpt)
| split | mae_surf_p |
|---|---|
| test_single_in_dist     | 169.31 |
| test_geom_camber_rc     | 139.23 |
| test_geom_camber_cruise | NaN ← *scoring bug, see below* |
| test_re_rand            | 125.83 |
| **test_avg (3 valid)**  | **144.79** |

JSONL summary: `research/EXPERIMENT_METRICS.jsonl` (PR=300 records).

### Analysis
- **Run hit `SENPAI_TIMEOUT_MINUTES=30` cap at end of epoch 9** (~205 s/epoch). Validation was still improving sharply (epoch 8 → 9 dropped 16 pts on `val_avg/mae_surf_p`); the model had not converged. We can't compare a 9-epoch result to a hypothetical 50-epoch baseline.
- **No overfitting yet** — train loss still well above val, consistent with under-training. The held-out camber gap (rc=157.5, cruise=104.5) wasn't catastrophic.
- **Peak GPU memory: 63 GB / 96 GB**, no OOM.
- **Critical bug found in `data/scoring.py`**: sample 20 of `test_geom_camber_cruise` has 761 inf values in the p channel of ground-truth y. The masking logic uses `error * mask` (float-mask multiply), and `inf * 0 = NaN` in IEEE-754, poisoning the float64 accumulator for that channel. Every wide- or deep-model experiment on this branch will hit the same NaN.
- Decision rationale: closing rather than re-running; assigning edward to fix the scoring bug next as the higher-leverage action. A more conservative widening (n_hidden=144-160) can be revisited once we have a fully-trained baseline number on this branch from one of the cheap in-flight experiments (alphonse, askeladd).
