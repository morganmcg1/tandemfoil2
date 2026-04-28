# SENPAI Research Results — willow-pai2d-r1

## 2026-04-28 07:00 — PR #324 v4 (merged, NEW BASELINE): EMA decay=0.999 with every-2-epochs validation gating

- branch: `willowpai2d1-nezuko/ema-and-grad-clip` (deleted on merge)
- hypothesis: every-2-epochs gating recovers the schedule budget that v3's
  per-epoch swap-validate-swap was eating (~7 s/epoch). Predicted post-fix
  val_avg in 55-56 range.

### Results

| Metric | Value | vs PR #531 (per-Re sqrt baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **52.1155** (epoch 36 of 36) | **−3.65%** (better than 55-56 prediction) |
| `test_avg/mae_surf_p` | **45.0018** | **−3.00%** |
| Per-epoch wall | ~52.9 s | +6% (vs 50 s baseline; EMA val ~3-4 s amortized) |
| Epochs completed | 36 / 50 | matches baseline budget exactly |
| Peak GPU memory | 24.1 GB | unchanged |
| W&B run | `qsplc76j` | |

Cumulative improvement: **−63.9% val_avg / −65.7% test_avg** since original PR #312 reference.

### Per-split val deltas vs PR #531

| Split | Δ |
|---|---|
| val_single_in_dist | −4.02% |
| val_geom_camber_rc | **−5.10%** (finally a non-trivial rc gain) |
| val_geom_camber_cruise | −3.24% |
| val_re_rand | −1.74% |

### EMA val curve smoothness (textbook variance reduction)

18 EMA validation points (epochs 2, 4, ..., 36), every successive value
strictly decreased. Raw val swings more (epoch 30 raw=62.77, epoch 32
raw=58.17 — bigger swings than EMA's 55.97 → 54.60 over the same window).

### Analysis & conclusions

- **Merged. New round baseline at val_avg=52.12.** Better than the 55-56
  prediction.
- **Four-version arc on this PR** is the cleanest example of diagnostic-
  driven iteration in the round: v1 EMA decay=0.9999 catastrophic
  (warmup pathology) → v2 decay=0.999 + drop grad_clip but on stale FF
  baseline → v3 rebase to L1 but blocked by val overhead → v4
  every-2-epochs gating recovers full schedule. Each step had a clear
  hypothesis from prior failure.
- **rc-camber convergence story**: the four interventions that have
  moved rc-camber best (FF -3.3%, compile -25.8% via schedule, per-Re
  sampling -5.0%, EMA -5.1%) collectively confirm rc has **multiple
  failure components** — schedule, representation, distribution, and
  parameter-trajectory variance. None dominates; each contributes.
- v3 rc-camber test gain (−6.07%) on pure-L1 alone shrunk to −2.57% on
  the per-Re+L1 baseline → per-Re sampling absorbed part of the v3 rc
  gain, EMA still adds on the splits where per-Re sampling helped less.
- Followups: grad_clip alone (assigned PR #614 — original PR v1 had it
  bundled, never tested in isolation under L1's already-bounded gradient
  regime). Decay sweep, aux-head composition, T_max alignment — all queued.



## 2026-04-28 06:13 — PR #529 (sent back): Surface-only auxiliary p head on PR #407 baseline

- branch: `willowpai2d1-alphonse/surface-only-aux-p-head` (in flight as draft after send-back)
- hypothesis: small aux MLP head trained on surface-pressure-only Huber
  loss + inference blending of main and aux predictions on surface nodes.
  Decouples extra surface-p signal from main loss gradient ratio.
  Predicted -2 to -5%.

### Results (vs PR #407 Huber+T_max=37 baseline; assignment-time)

| Run | val_avg/mae_surf_p | Δ vs PR #407 | test_avg | Δ vs PR #407 | best ep | W&B |
|---|---|---|---|---|---|---|
| Full (aux loss + inference blend, α=0.5) | **66.16** | **−5.13%** | **58.44** | **−3.39%** | 36 | `8u8s1ecw` |
| Ablation (aux loss only, no inference blend) | 68.44 | −1.86% | 60.03 | −0.74% | 33 | `wfjngkjo` |

vs current baseline (PR #531 at val_avg=54.09): **+22.3%** — assignment-time
goal-post shift; rebase needed.

### Mechanism decomposition

The clean ablation isolates two independent mechanisms:
- **Aux loss term alone**: ~1.9% on val (better backbone representations
  via additional supervisor on the surface-p subspace).
- **Inference blend alone**: ~3.3% additional on top (main and aux heads
  make different surface-p errors; averaging gives consistent gain).

Volume MAE_p slightly *improved* (-1.6%) — confirms "more surface signal
without distorting volume gradients" hypothesis.

### Per-split structure

- **Cruise benefits most from blend** (-8.1% surf_p val) — main and aux
  heads make sufficiently different errors for averaging to help.
- **rc essentially flat** — main and aux heads agree where they're both
  wrong; rc remains representation-limited (consistent signal across
  PR #503 closing, PR #527 schedule, PR #531 sampling).
- single_in_dist -4.6%, val_re_rand modest improvement.

### Capacity / cost

| Metric | Baseline | This PR |
|---|---|---|
| Total params | 0.67 M | 0.69 M (+3%) |
| Cold compile | ~10 s | 10.05 s (no extra recompiles) |
| Per-epoch wall | ~49 s | ~50 s |
| Peak GPU memory | 24.1 GB | 24.7 GB (+0.6 GB) |

### Analysis & conclusions

- **Sent back, not merged.** Same goal-post-shift situation as fern,
  edward, nezuko, etc. — assignment-time baseline (PR #407) was
  superseded by L1 (PR #504) → T_max=50 (PR #541) → per-Re sampling
  (PR #531) during the run.
- **Mechanism is orthogonal** to all three (loss formulation, schedule
  shape, sampler weighting) so should still give a meaningful win on
  the new baseline — though smaller than the 5.1% on Huber, since pure
  L1 already addresses some of the small-residual fine-tuning the aux
  head was previously helping with.
- Send-back instructions: rebase onto current advisor branch + switch
  aux loss from SmoothL1 to pure L1 (consistency with merged loss) +
  use `--epochs 50` (matches T_max=50). Predicted post-rebase val_avg:
  51-53 at 50-70% stacking efficiency.
- Followups (aux_weight sweep, blend_alpha sweep, aux velocity head)
  queued for after the rebased run lands.

## 2026-04-28 06:12 — PR #531 v2 (merged, NEW BASELINE): Per-Re sqrt sampling on pure L1

- branch: `willowpai2d1-fern/per-re-weighted-sampling` (deleted on merge)
- hypothesis: rebase per-Re sqrt sampling onto pure L1 baseline.
  Predicted post-rebase val_avg in 53-56 range if at 70-100% stacking
  efficiency.

### Results

| Metric | Value | vs PR #541 (current baseline) | vs PR #504 (orig L1 baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **54.0914** (epoch 37 of 37) | **−3.79%** | −5.58% |
| `test_avg/mae_surf_p` | **46.3959** | **−4.18%** | **−9.65%** |
| Per-epoch wall | ~48 s | ≈baseline | ≈baseline |
| Peak GPU memory | 24.1 GB | unchanged | unchanged |
| W&B run | `ncn6snxe` | | |

Cumulative improvement: **−62.5% val_avg / −64.6% test_avg** since
original PR #312 reference.

### Per-split val deltas vs PR #504

| Split | Δ |
|---|---|
| val_single_in_dist | **−8.62%** |
| val_geom_camber_rc | −5.01% |
| val_geom_camber_cruise | −7.17% |
| val_re_rand | −1.88% |

### Per-split test deltas vs PR #504

| Split | Δ |
|---|---|
| test_single_in_dist | −7.58% |
| test_geom_camber_rc | **−9.79%** (vs flat under Huber!) |
| test_geom_camber_cruise | **−11.66%** |
| test_re_rand | −10.36% |

### Analysis & conclusions

- **Merged. New round baseline at val_avg=54.09.** Per-Re sampling stacks
  cleanly on pure L1 with **94% efficiency** (vs ~91% on Huber).
- **Test gain consistently bigger than val gain** under both losses
  (Huber: test −4.4% / val −5.9%; L1: test −9.65% / val −5.58%).
  Sampler reweighting helps **generalization** more than training-set
  fit — exactly the kind of effect we want for paper-facing test_avg.
- **rc-camber test response flipped under L1**: from nearly flat (−0.07%)
  under Huber to clean improvement (−9.79%) under L1. Mechanism:
  - Under Huber's quadratic-in-tail profile, narrow-Re splits got little
    reweighting benefit because gradient was already compressed.
  - Under L1's constant gradient, re-emphasized high-Re samples push
    useful updates everywhere.
  Adds another data point: rc-camber failure mode has **multiple
  components**, schedule-truncation alone isn't the only one — sampling
  emphasis also moves it under the right loss.
- BASELINE.md updated. Followup assigned: linear-Re bracket (PR #591).



## 2026-04-28 06:01 — PR #541 (merged, NEW BASELINE): L1 T_max sweep — `--epochs 50` confirmed

- branch: `willowpai2d1-edward/l1-tmax-sweep` (deleted on merge)
- hypothesis: pure L1's constant-magnitude gradient predicts T_max=50
  (lr stays positive throughout) beats T_max=37 (cosine→0). Two-run sweep
  to settle the schedule alignment question for L1, predicted ~+0.5% to +2%
  for T_max=37 vs T_max=50.

### Results

| Run | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | end lr | W&B |
|---|---|---|---|---|---|
| `l1-tmax37` | 36 / 37 | **58.0402** | 49.6875 | 0 (cosine zero) | `6ihh5b0s` |
| **`l1-tmax50-rerun`** | **37 / 37** | **56.2167** | **48.4232** | 7.89e-5 (~16% peak) | `pwi9gy9f` |

Direct comparison: T_max=50 wins by **−1.8235 absolute, −3.14% relative**
on val_avg vs T_max=37. Test gap: **−2.54%**. Exceeded predicted 0.5-2%.

vs current PR #504 baseline (57.29):
- tmax50-rerun: val_avg = 56.22 → **−1.07%** (fresh seed lined up favorably)
- test_avg = 48.42 → **−5.70%** vs 51.35 baseline

### Cumulative improvement

**−61.0% on val_avg / −63.1% on test_avg** since original PR #312 reference
(144.21 → 56.22, 131.18 → 48.42).

### Analysis & conclusions

- **Merged. New round baseline at val_avg=56.22.** Two findings:
  1. T_max=50 confirmed correct for pure L1 (3.14% better than T_max=37).
     Mechanism: L1's `sign(r)` constant-magnitude gradient keeps making
     progress at small residuals; the late-training low-LR tail extracts
     continued refinement rather than settling. T_max=37 zeroes lr
     prematurely. **Per-epoch jumps in T_max=50's last few epochs are
     the LARGEST of the run** (epoch 36→37: −5.4% in one epoch with
     lr ≈ 8e-5).
  2. **Single-seed variance ≈ 1%** (PR #504 yi5upb1e=57.29 vs PR #541
     pwi9gy9f=56.22 same config, different seed). Future single-seed
     <1% wins should be treated as noise; only ≥2% claims are robust.
- **rc-camber barely moved** (−0.73% val / −0.17% test) — schedule
  changes don't help rc, confirming **rc is representation-limited,
  not residual-refinement-limited**. We need geometry-side or capacity-
  side experiments for rc, not schedule/loss tweaks.
- **Per-split structure matches mechanism**: cruise (-6.16%) and
  single-in-dist (-3.54%) win biggest where small residuals dominate.
  re_rand wins moderately (-3.69%). rc minimal.
- BASELINE.md updated to recommend `--epochs 50` for pure L1 (was
  `--epochs 37` from fern's PR #407 era when Huber was active).
- Followup assigned: `--epochs 70` probe (PR #584) — tests if even
  longer schedule continues the trend, with lr ≈ 45% of peak at the cap
  vs T_max=50's ~16%.



## 2026-04-28 05:13 — PR #324 v3 (sent back): EMA decay=0.999 rebased on pure L1 baseline

- branch: `willowpai2d1-nezuko/ema-and-grad-clip` (in flight as draft after send-back)
- hypothesis: EMA(decay=0.999) on top of pure L1 baseline. Predicted -2 to -5%.

### Results

| Metric | Value | vs PR #504 (pure L1, current baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **58.71** (epoch 32 of 32 wall-cap, still descending) | **+2.49%** (regression) |
| `test_avg/mae_surf_p` | **50.21** | **−2.23%** (better) |
| Test Δ on `test_geom_camber_rc` | 64.10 vs 71.57 | **−6.07%** (huge OOD win) |
| Test Δ on `test_re_rand` | 48.67 vs 50.83 | **−3.11%** |
| Per-epoch wall | ~57 s (vs 49 s baseline; +8 s EMA val overhead) | +14% |
| Epochs completed | 32 / 50 (vs 36 baseline) | -4 epochs |
| Peak GPU memory | 24.2 GB | unchanged |
| W&B run | `tq07pkuf` | |

### Analysis & conclusions

- **Sent back** for EMA val overhead optimization. Val regression is
  schedule-budget, not mechanism failure:
  - Val curve still descending at end (epochs 28-32: 61.7 → 58.7,
    ~0.7-0.9/epoch decay).
  - Projection to 36 epochs: val_avg ≈ 55.9 (would beat baseline).
  - EMA shadow fully warm (`0.999^12000 ≈ 6e-6` initial contamination).
- **Mechanism is confirmed**: EMA preferentially helps OOD splits
  (test_geom_camber_rc −6.07%, test_re_rand −3.11%), exactly where
  variance-reduction-in-parameter-space predicts.
- **Val/test ratio improved** (1.116 → 1.169): EMA produces a model
  more conservative on val (less overfit to val-specific noise) that
  transfers better to held-out test.
- Send-back instructions: gate EMA swap-validate-swap on epoch parity
  (every 2nd epoch), recovers ~3-4 s/epoch → ~36 reachable epochs.
  Predicted post-fix val_avg in 55-56 range.

## 2026-04-28 05:13 — PR #544 (closed): surf_weight=15 on pure-L1 baseline

- branch: `willowpai2d1-thorfinn/sw15-on-pure-l1-baseline` (deleted on close)
- hypothesis: round-1 sw=15 directional finding from PR #333 transfers to
  pure L1 baseline. Predicted -1 to -3%.

### Results

| Metric | Value | vs PR #504 (pure L1, current baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **62.26** (epoch 30 of 31) | **+8.69%** (clear regression) |
| `test_avg/mae_surf_p` | 53.76 | +4.69% |
| Surface Ux MAE (val_avg) | 0.85 | **+9.4%** (val_single_in_dist: **+28%**!) |
| Surface Uy MAE (val_avg) | 0.42 | +4.6% |
| W&B run | `jvl533ne` | |

### Analysis & conclusions

- **Closed.** sw=15 doesn't transfer from MSE to L1.
- **Mechanism**: L1 has constant-magnitude gradient `sign(r)/N`. With
  sw=15, the surface gets 1.5× constant pull throughout training (not
  decaying with residual like under MSE). This accelerates early
  surface fit (epoch 19: val 79.3 vs baseline ~107) but causes
  **oscillation near convergence** (epoch 25-31 val swings 62-80) instead
  of clean descent.
- **Velocity-pressure coupling biting** (askeladd #451 prediction): surface
  Ux regressed +9.4% val / +28% on single_in_dist, surface Uy regressed
  +4.6%. Distorting volume velocity (chasing surface fit) breaks
  Navier-Stokes-respecting flow → hurts surface pressure prediction.
- **Three-point monotonic curve confirms** under pure L1: sw=10 baseline
  → sw=15 (+8.7%) → sw=50 effective via PR #451 (+59%). Surface
  overweighting under L1 is bad; sw=10 may be near or below optimum.
- Reassigned thorfinn to **sw=8 probe** (PR #570) — single-flag test of
  whether sw<10 is the right direction.
- Followup #1 from this PR (annealed sw=15→10 warmup) held for if
  sw<10 doesn't move the metric.

## 2026-04-28 05:11 — PR #443 (closed): Gaussian RFF (K=16, σ=10) replacing deterministic FF

- branch: `willowpai2d1-tanjiro/gaussian-random-fourier` (deleted on close)
- hypothesis: Gaussian RFF outperforms deterministic 2^k π ladder on
  CFD/coordinate-MLP tasks (Tancik et al. 2020). Predicted -2 to -5%.

### Results (vs PR #407 Huber+T_max=37 baseline)

| Metric | Value | vs PR #407 (rebase target) | vs PR #504 (pure L1, current) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **71.19** (epoch 36 of 36) | **+2.07%** (regression) | +24.3% (much worse) |
| `test_avg/mae_surf_p` | 62.36 | +3.11% | +21.0% |
| Per-epoch wall | ~49 s | same | same |
| Peak GPU memory | 24.1 GB | same | same |
| W&B run | `xk8r2y6c` | | |

### Per-split val deltas vs PR #407 baseline

| Split | Δ |
|---|---|
| val_single_in_dist | **+7.49%** (worse — in-dist regression) |
| val_geom_camber_rc | −2.97% (slightly better — OOD geometry) |
| val_geom_camber_cruise | +1.69% |
| val_re_rand | +2.71% |

### Analysis & conclusions

- **Closed.** Mechanism analysis is the keeper:
  - Foils in chord-frame coordinates are strongly axis-aligned →
    deterministic 2^k π ladder catches dominant pressure-field
    frequency directions.
  - Isotropic Gaussian RFF spreads frequency budget across off-axis
    directions the network can't usefully exploit on this geometry.
  - σ=10 puts too much energy at frequencies the network can't usefully
    use, hurting in-dist fit while marginally helping OOD generalization
    — the wrong tradeoff for our equal-weight metric.
- **Train-loss curve is *slower* per epoch with Gaussian RFF** (epoch 1
  train surf 0.30 vs 0.25 for deterministic at same point). Tancik's
  faster-convergence claim doesn't generalize to this CFD-on-foil
  setting.
- Rebase to pure L1 baseline wouldn't change qualitative finding —
  saved 30-min run by closing instead.
- **Followups #1 (σ sweep), #2 (K=8 Gaussian)**: skipped — same
  axis-alignment argument predicts these would also lose.
- **Followup #3 (lower σ for smooth-target tasks)**: also skipped — the
  argument predicts smaller loss but still loss vs deterministic.
- **Followup #4 (FF on saf)**: assigned as next experiment (PR #564) —
  saf is a coordinate-like feature aligned with foil-surface position;
  the deterministic-ladder argument applies even more directly than for
  (x, z).

## 2026-04-28 04:50 — PR #531 round-1 (sent back): Per-Re weighted sampling on Huber+T_max=37

- branch: `willowpai2d1-fern/per-re-weighted-sampling` (in flight as draft after send-back)
- hypothesis: weight training samples within each domain by sqrt(Re/Re_median[domain])
  to emphasize high-Re (high-pressure-magnitude) tail. Predicted -1 to -4%.

### Results (on Huber+T_max=37 baseline, BEFORE pure L1 merge)

| Metric | Value | vs PR #407 (Huber, rebase target) | vs PR #504 (pure L1, current) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **65.61** (epoch 37 of 37) | **−5.91%** | **+14.5%** (regression) |
| `test_avg/mae_surf_p` | 57.82 | −4.41% | +12.6% |
| Per-epoch wall | ~48 s | same | same |
| Peak GPU memory | ~24 GB | same | same |
| W&B run | `3xhvmzbl` | | |

### Per-split deltas vs Huber baseline

| Split | val Δ | test Δ | Re range |
|---|---|---|---|
| val_single_in_dist | −5.57% | −4.90% | 100K-5M |
| val_geom_camber_rc | −6.30% | −0.07% (test flat) | 1M-5M (narrow) |
| val_geom_camber_cruise | **−7.65%** | **−9.38%** | 122K-5M (broadest) |
| val_re_rand | −4.60% | −5.28% | Re-stratified |

**Mechanism confirmed**: per-split improvement scales with the Re-spread
of the split. Broadest Re range (cruise) wins most; narrow Re range
(rc-camber) wins least on test. Direct dose-response signal that the
within-domain Re emphasis is doing the work, not some confound.

### Diagnostics

- Per-domain Re medians: racecar_single 2.53M, racecar_tandem 2.73M, cruise 2.90M (all in 100K-5M range, upper-skewed).
- Within-domain p90/p10 weight ratio: 2.1-2.8× (gentler than PR's 3-5× prediction; reflects upper-skewed Re distribution).
- Domain mass remains balanced after reweighting.

### Analysis & conclusions

- **Sent back, not merged.** Same goal-post-shift situation as edward,
  alphonse, askeladd, nezuko before: assigned-time baseline (PR #407
  Huber) was superseded by pure L1 (PR #504) shortly after fern's run
  started.
- **Mechanism is sampler-side, fully orthogonal to loss formulation.**
  Per-Re sampling composes with compile/FF/L1 cleanly. Predicted post-
  rebase val_avg: 53-56 if at 70-100% stacking efficiency (compile+FF+L1
  stacks at ~91%). Worst case ~57-58 if mostly orthogonal and marginal
  lever shrinks at lower-MAE regime.
- Send-back instructions: rebase onto pure L1 baseline + use `--epochs 50`
  (matches PR #504's T_max=50) instead of `--epochs 37`.
- Followups queued: linear-Re bracket (PR's followup #1, lined up if rebased
  result lands in 5% band); per-sample Re-weighted loss (followup #2);
  cross-domain Re weighting (followup #4).

## 2026-04-28 04:48 — PR #509 (closed): batch_size=8 + lr=7.07e-4 revisit on Huber+compile+FF

- branch: `willowpai2d1-thorfinn/batch-size-8-on-compile-baseline` (deleted on close)
- hypothesis: compile fuses kernels and changes the memory regime; bsz=8
  with sqrt-scaled lr might amortize per-step compute and give a real
  per-epoch speedup that PR #360 couldn't see. Predicted -1 to -5%.

### Results

| Metric | Value | vs PR #314 baseline (Huber+compile+FF, the rebase target) | vs PR #504 (pure L1, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **74.02** (epoch 34 of 34) | **+6.0%** | **+29.2%** (regression) |
| `test_avg/mae_surf_p` | 63.47 | +2.8% | +23.6% |
| Per-epoch wall (mean) | **53.98 s** | **+10.2%** (slower!) | +10.2% (slower) |
| Per-batch wall (steady) | 0.260 s | ×1.99 vs bsz=4 | n/a |
| Peak GPU memory | 48.3 GB | ×2.00 vs bsz=4 | n/a |
| Epochs completed | 34 / 50 | -2 | n/a |
| W&B run | `48uy33yz` | | |

### Analysis & conclusions

- **Closed.** Decisive falsification with conclusive throughput data.
  Per-batch time scaled ×1.99 with batch size doubling — kernel-launch
  overhead was already negligible at bsz=4 with compile, so bsz=8 has no
  amortization room. Peak memory scaled ×2.00 — compile fused autograd
  intermediates intra-step but did not compress them across batch dim.
- **Trainer is fundamentally HBM-bandwidth + padding-cost bound at every
  baseline measured** (32.9 / 42 / 84 GB pre-compile, 24.1 / 48.3 GB on
  compile). compile reduced the constant-factor of HBM traffic per token
  but did not change linear scaling with token count.
- Per-epoch wall going UP (not down) when batch-size doubled is the final
  signature: with half as many batches × 2× per-batch cost, the integral
  is slightly worse because validation pass + epoch-boundary overhead
  amortize over fewer batches.
- **Batch-size scaling is now permanently retired** as a throughput knob
  for this trainer. Compile reduced its single-step cost but did not move
  the bottleneck. If future throughput is wanted, **per-token cost
  reduction** (variable-mesh attention kernel, sparser slice tokens,
  reduce pad-mask waste) is the only remaining lever — substantial
  implementation work, not on the immediate roadmap.
- Reassigned thorfinn to **sw=15 on pure-L1 baseline** (PR #544): clean
  re-test of their round-1 directional surf_weight finding on the new
  best baseline (val_avg=57.29). No rebase, single-flag change.

## 2026-04-28 04:40 — PR #504 (merged, NEW BASELINE): Pure L1 loss replacing SmoothL1(β=1.0)

- branch: `willowpai2d1-edward/pure-l1-loss` (deleted on merge)
- hypothesis: pure L1 (`(pred - y_norm).abs()`) is simpler than
  SmoothL1(β=1.0); should match or beat. Predicted Δ in [−1%, +1%].

### Results

| Metric | Value | Δ vs prior baseline (PR #407, T_max=37) | Δ vs PR #314 baseline (Huber, T_max=50) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **57.2858** (epoch 36 of 36 wall-cap) | **−17.86%** | **−17.96%** |
| `test_avg/mae_surf_p` | **51.3504** | **−15.10%** | **−16.79%** |
| Per-epoch wall | ~49 s | same | same |
| Peak GPU memory | 24.1 GB | same | same |
| Epochs completed | 36 / 50 | (note: this run used --epochs 50, not 37) | same |
| W&B run | `yi5upb1e` | — | — |

Cumulative improvement: **−60.3% val_avg / −60.9% test_avg** since original PR #312 reference (144.21 → 57.29, 131.18 → 51.35).

### Per-split val deltas vs Huber baseline

| Split | Δ |
|---|---|
| val_single_in_dist | **−18.6%** |
| val_geom_camber_rc | −11.7% |
| val_geom_camber_cruise | **−27.0%** |
| val_re_rand | −17.8% |

Pure L1 disproportionately helps the easier splits (cruise, single-in-dist)
where residuals shrink fastest — exactly the regime where SmoothL1's
gradient was vanishing.

### Mechanism (the keeper insight from edward's writeup)

> SmoothL1's gradient is `r/β` for `|r| < β`, which **vanishes** as
> residuals shrink. Pure L1's gradient is `sign(r)` everywhere — full
> unit magnitude even for tiny residuals. In normalized (unit-std)
> space, once the model trains reasonably well, residuals are `|r| ≪ 1`,
> so SmoothL1 spends most of training in its quadratic regime
> (effectively a downweighted MSE), while pure L1 keeps pushing every
> residual toward zero with the same step magnitude.

The per-split asymmetry directly confirms this: cruise splits (easiest,
fastest residual shrinkage) won most. Hardest split (rc-camber) won
least — its residuals stay larger so SmoothL1 was already in its L1 tail.

### Schedule alignment may have flipped for pure L1

PR #407 merged `--epochs 37` (T_max=37, cosine reaches lr=0) for Huber.
Pure L1 keeps gradient at unit magnitude even at tiny residuals, so it
*can* keep refining at low LR — but it stops at lr=0. Mechanism predicts
T_max=50 (lr at end ≈ 27% of peak) is better for pure L1 than T_max=37.

The merged result (val_avg=57.29) used `--epochs 50` per the original
assignment (which predated PR #407). BASELINE.md updated to recommend
`--epochs 50` for reproduction.

**edward PR #541 in flight**: direct two-run sweep `--epochs 37` vs
`--epochs 50` with pure L1 to settle the schedule question.

### Analysis & conclusions

- **Merged. New round baseline at val_avg=57.29.** Biggest single-lever
  win since compile (−24.4%). And it's the **simpler** formulation.
- **The schedule-alignment story is loss-dependent.** What was right for
  Huber (T_max=37, late-LR settle) may be wrong for L1 (T_max=50, lr stays
  positive). Future stacked PRs need to track this.
- Cumulative now ~60% improvement. Round 3 has been remarkably productive.
- Followups assigned: T_max sweep with L1 (edward #541). β sweep on
  Huber (queued, lower priority now that pure L1 won), seed-variance
  multi-seed (queued, lower priority since qualitative win is robust).



## 2026-04-28 04:27 — PR #407 (merged, NEW BASELINE): Cosine T_max alignment via `--epochs 37`

- branch: `willowpai2d1-fern/cosine-tmax-alignment` (deleted on merge)
- hypothesis: cosine T_max=epochs misaligned with achievable budget; align
  via `--epochs 37` to let cosine actually decay. Predicted -1 to -3%.

### Results

| Metric | Value | Δ vs prior baseline (PR #314, T_max=50) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **69.7385** (epoch 36 of 37 completed) | **−0.13%** |
| `test_avg/mae_surf_p` | **60.4829** | **−2.0%** |
| Per-epoch wall | ~49 s | same |
| Epochs completed | 37 / 37 (full schedule!) | +1 from binding |
| Peak GPU memory | 24.1 GB | same |
| LR at best epoch | 3.6e-6 (0.72% of peak) | vs 9e-5 / ~18% under T_max=50 |
| W&B run | `7xvc5hfl` | — |

Cumulative improvement: **−51.7% val_avg / −53.9% test_avg** since PR #312.

### Late-stage settle pattern (mechanism worked)

Six consecutive epochs (31-36) each set a new best:
71.69 → 71.00 → 70.56 → 70.45 → 69.82 → **69.74** → 69.76

The model converged monotonically through the low-LR tail — exactly the
schedule-shape behavior the hypothesis predicted.

### Analysis & conclusions

- **Merged. New round baseline at val_avg=69.74.** PR has zero file
  changes — the experiment is a CLI-flag adjustment, not a code change.
  Merged the empty branch (gives credit) and updated BASELINE.md to:
  - Record new metric (val_avg=69.74, test_avg=60.48)
  - Document `--epochs 37` as the recommended setting in the reproduce command
  - Add fern's followup #1 as a checklist item: future merges that change
    per-epoch wall need to re-evaluate `--epochs N`.
  - Did **not** hardcode `epochs=37` into Config default — fragile to
    future budget shifts.
- **val gain (0.13%) is at noise level**, but the test gain (2.0%) is
  consistent across all 4 splits and the qualitative cosine-tail
  signature (6 consecutive epoch-bests in the low-LR tail) confirms the
  schedule mechanism is doing real work.
- The schedule-shape decision is downstream-relevant regardless of
  magnitude: every future hypothesis on this branch can now be evaluated
  on a properly-decaying schedule rather than the truncated cosine.
- Two students caught baseline drift on this PR (tanjiro #443, fern
  #407) by asking before running. Reinforces the rebase-discipline norm.

## 2026-04-28 04:25 — PR #503 (closed): Half-step capacity (h=160, L=5, heads=5, slices=80) on compile+FF

- branch: `willowpai2d1-alphonse/halfstep-capacity-on-compile-ff` (deleted on close)
- hypothesis: with compile speedup + 70 GB VRAM headroom, the bigger
  model that PR #393 couldn't test should now be feasible. Predicted
  -2 to -5%.

### Results

| Metric | Value | vs PR #416 (compile+FF baseline) | vs PR #314 (Huber baseline, current) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **90.62** (epoch 27 of 27, still descending) | **+12.07%** | +29.8% |
| `test_avg/mae_surf_p` | 82.40 | +12.24% | +33.5% |
| Per-epoch wall | ~66 s | +35% | +35% |
| Epochs completed | 27 / 50 | -10 | -10 |
| Peak GPU memory | 31.6 GB | +31% | +31% |
| Param count | 1.03 M | +54% | +54% |
| W&B run | `12x5vmre` | | |

### Analysis & conclusions

- **Closed.** Capacity scale-up at this scale is now **conclusively ruled
  out** for this trainer — two independent runs (PR #393 on bf16, this
  PR on compile+FF) both regress to roughly the same +12% val band.
- **Wall budget tax structurally larger than cosine-tail benefit.** The
  bigger model is +35% slower per epoch even after compile fusion;
  cosine T_max=epochs schedule means it gets less of the high-lr
  exploration phase relative to its convergence needs, not more.
- **rc-camber is not capacity-limited.** Was the OOD split the bigger
  model was meant to help most with; regressed +14.9% / +21.0% (val/test).
  Combined signal across PR #327 (FF: −3.3%), PR #324 v2 (EMA: −16%),
  PR #503 (capacity: +14.9%) — rc-camber failure mode looks like a
  **gradient-stability problem** (hence EMA helps), not representation
  or capacity. **Targeted-architecture rc-camber experiments dequeued**.
- **Capacity direction redirected** toward novel architecture (auxiliary
  heads, mesh-aware encoders, geometry-conditioned attention), not
  width/depth scaling.



## 2026-04-28 04:14 — PR #451 (closed): Surface-only pressure weighting (1, 1, 5) on surf_loss

- branch: `willowpai2d1-askeladd/surface-only-pressure-weight` (deleted on close)
- hypothesis: restrict (1,1,5) channel weighting to surf_loss only (vol_loss
  unchanged) to recover the cruise/re_rand wins from PR #313 v1 without the
  Ux/Uy starvation that ruined PR #313 v2. Predicted -1 to -4%.

### Results (vs PR #416 baseline, before Huber merge)

| Metric | Value | vs PR #416 (compile+FF, rebase target) | vs PR #314 (Huber+compile+FF, current) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **91.25** (epoch 36 of 37) | **+12.86%** (regression) | +30.7% (much worse) |
| `test_avg/mae_surf_p` | 81.16 | +10.55% | +31.5% |
| Surface Ux MAE (val_avg) | 1.76 | **+37.3%** | n/a |
| Surface Uy MAE (val_avg) | 0.87 | **+38.3%** | n/a |
| W&B run | `64c83ffi` | | |

### Per-split val deltas vs PR #416 baseline

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | +17.83% | **+33.7%** | **+46.5%** |
| val_geom_camber_rc | +10.42% | **+41.5%** | +35.8% |
| val_geom_camber_cruise | +13.71% | +36.2% | +38.5% |
| val_re_rand | +9.78% | +35.5% | +33.7% |

### Analysis & conclusions

- **Closed.** All 8 splits (4 val + 4 test) regressed 9-18%. Decisive
  falsification.
- **Mechanism (askeladd's diagnosis):** surface velocity and surface
  pressure share gradient structure through the boundary layer / pressure-
  coefficient relation. The (1, 1, 5) weighting on surf_loss makes pressure
  dominate the surface-node gradient by 5×, starving surface Ux/Uy. Worse
  surface velocity prediction → worse surface pressure prediction
  (geometric coupling), which shows up *as* a surface-pressure regression
  at the metric we rank on.
- **Channel-weighted MSE family conclusively ruled out** at convergence:
  - PR #313 v1: hurt volume velocity (vol_loss weighted)
  - PR #313 v2: didn't transfer post-bf16 (small wins were schedule
    artifacts that disappeared at convergence)
  - PR #451 (this run): hurt surface velocity (surf_loss weighted)
- The original PR #313 v1 cruise + re_rand wins were **schedule-truncation
  artifacts**, not real wins from the (1,1,5) ratio. At 14 epochs the model
  never converged, so distorted gradient ratios produced different (and
  seemingly better) trajectories. With compile-unlocked 37-epoch budget,
  those trajectories converge to a worse local optimum at every channel-
  weight ratio.
- Reassigned askeladd to lr=3e-4 (their followup #4): tests whether the
  surface-gradient sharp-edge implication of this PR's failure mode also
  applies to the unweighted baseline.
- Auxiliary surface-pressure loss term (askeladd's followup #2) — *adding*
  a pressure term rather than reweighting — queued for a future slot if
  lr-tuning doesn't move the metric.

## 2026-04-28 03:52 — PR #333 (closed by student-bot after send-back)

After my send-back at 03:50, the student-bot (morganmcg1) auto-closed
the PR at 03:52:40Z. Likely reason: rebase complexity (advisor branch
moved through bf16 → FF → compile → Huber merges since the original
assignment) made an in-place rebase of the round-1 sweep branch
impractical. Reassigning thorfinn to a fresh experiment (`batch_size=8
on Huber+compile+FF baseline`, PR #509) rather than re-issuing the sw=15
single-run experiment — the surf_weight directional finding from the
round-1 sweep is recorded for if we ever need to revisit it, but the
expected delta on the much-improved baseline is small (~1-2%) and a
batch-size scaling re-investigation has higher expected information
value with the new memory regime.

## 2026-04-28 03:50 — PR #333 (sent back, round-1 ablation): surf_weight ∈ {15, 25, 40} on PR #312 base

- branch: `willowpai2d1-thorfinn/surf-weight-sweep` (in flight as draft)
- hypothesis: surf_weight > 10 puts more gradient on surface, improves
  surface-pressure MAE.

### Results (vs PR #312 reference, 14 epochs, no bf16/FF/compile)

| surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| 15 | **130.25** (best) | 119.35 |
| 25 | 136.02 | 125.34 |
| 40 | 133.52 | 122.19 |
| (10) baseline PR #312 | 144.21 | 131.18 |

### Analysis & conclusions

- **Sent back, not closed.** Sweep is a clean round-1 ablation: sw=15
  wins, sw>15 shows diminishing returns and volume-channel degradation
  (test mae_vol_p 128.74 → 142.42 → 149.33 for sw=15/25/40). Surface
  vs volume tradeoff is favorable at sw=15, reverses by sw=40.
- vs current Huber+compile+FF baseline (69.83), all three sweep results
  are +86% or worse (stale base).
- Single-seed val noise is large (~80 points epoch-to-epoch); ranking
  among {15, 25, 40} is inside the noise band.
- Send-back: rebase + run **sw=15 only** on the Huber+compile+FF
  baseline. If the directional finding (sw=15 > sw=10) transfers, merge;
  if not, default sw=10 stays.

## 2026-04-28 03:42 — PR #314 round-3 (merged, NEW BASELINE): SmoothL1/Huber + bf16 + FF + `torch.compile`

- branch: `willowpai2d1-edward/huber-loss` (deleted on merge)
- hypothesis: stack SmoothL1 (β=1.0) on top of compile+FF baseline.
  Predicted post-rebase val_avg in 65-75 range based on prior 91% stacking
  efficiency (FF + Huber).

### Results

| Metric | Value | Δ vs prior baseline (PR #416, compile+FF) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **69.8310** (epoch 35 of 36) | **−13.62%** |
| `test_avg/mae_surf_p` | **61.7177** | **−15.93%** |
| Per-epoch wall (steady) | ~49 s | ≈baseline |
| Cold compile time | 32.5 s | up from 27.85 s on compile+FF (SmoothL1 graph nodes) |
| Epochs completed | 36 / 50 | −1 (essentially same) |
| Peak GPU memory | **24.1 GB steady** | ≈baseline (memory anomaly RESOLVED) |
| W&B run | `fs3tf90w` (`smoothl1-beta1-on-compile-ff`) | — |

Cumulative improvement: **−51.6% val_avg / −53.0% test_avg** since original PR #312 reference (144.21 → 69.83, 131.18 → 61.72).

### Stack composition

| Stack | val_avg | Δ vs FF baseline (106.92) |
|---|---|---|
| FF alone | 106.92 | — |
| Compile alone (on FF) | 80.85 | −26.07 |
| Huber alone (on FF) | 92.32 | −14.60 |
| **Compile + Huber (on FF) — this run** | **69.83** | **−37.09** |
| Sum-of-individuals | — | −40.67 |
| **Capture ratio** | — | **91.2%** |

The 91% capture ratio is identical to the FF + Huber stack alone (also
91%), indicating the three mechanisms (compile = execution graph; FF =
input representation; Huber = loss gradient profile) live on genuinely
orthogonal axes. The remaining 9% overlap is the shared "more cosine
schedule reaches the model" mechanism.

### Memory anomaly resolved

Rounds 1 & 2 showed transient ~95-98 GB peaks attributable to SmoothL1
autograd intermediates (`|x-y|`, `min(|x-y|, β)`, where-mask). Compile
flat-lines at 24.1 GB across all 36 epochs because the inductor graph
**fuses these intermediates into a single kernel** without materializing
them as separate tensors. ~78 GB headroom now exists at the new baseline.

This resolves my pre-rebase memory prediction error (88-90 GB expected;
actual 24.1 GB) — naive memory accounting undercounts compile's kernel-
fusion benefits when the loss has compound autograd intermediates.

### Per-split val (epoch 35 best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist | 76.34 | 0.92 | 0.48 | 83.08 |
| val_geom_camber_rc | 81.78 | 1.37 | 0.66 | 89.00 |
| val_geom_camber_cruise | 52.16 | 0.70 | 0.40 | 53.36 |
| val_re_rand | 69.04 | 1.04 | 0.52 | 68.97 |
| **val_avg** | **69.83** | 1.01 | 0.52 | 73.60 |

Per-split structure mirrors prior runs: cruise easiest, single-in-dist
hardest. **Ux/Uy MAE also at strong levels** — Huber+compile+FF improves
all three velocity/pressure surface channels, not just `surf_p`.

### Analysis & conclusions

- **Merged. New round baseline.** Cumulative −51.6% across the round.
- Three orthogonal mechanisms compose to ~91% efficiency. The "stack
  effect ratio" stayed constant from 2-mechanism to 3-mechanism stacks,
  which is a strong empirical signal that the next intervention is
  unlikely to capture much less than its measured single-lever delta.
- **78 GB VRAM headroom** opens batch-size scaling investigation again
  (PR #360 ruled it out without compile; memory math is now different).
- Followups assigned: pure L1 on this baseline (edward followup #2,
  PR #504). β sweep, EMA stack, and batch-size revisit are queued.
- Still a clear schedule-budget ROI: cosine T_max=50 with 36 epochs
  reachable means lr at end is 9e-5 (~18% of peak), still misaligned.
  Fern's #407 with --epochs 37 is the precise alignment.



## 2026-04-28 02:56 — PR #416 round-2 (merged, NEW BASELINE): `torch.compile(dynamic=True)` + bf16 + FF

- branch: `willowpai2d1-alphonse/torch-compile-pilot` (deleted on merge)
- hypothesis: stack `torch.compile(dynamic=True)` on top of FF baseline.
  Predicted -2 to -5% from compile alone, but rebased run expected to land
  in 80-85 (combined with FF speedup).

### Results

| Metric | Value | Δ vs prior baseline (PR #327, FF) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **80.8506** (epoch 37 of 37 completed) | **−24.4%** |
| `test_avg/mae_surf_p` | **73.4107** | **−24.2%** |
| Per-epoch wall (steady) | ~49 s | **−50%** (2.0× speedup) |
| Cold compile time | 27.85 s (one-time) | up from 9.26 s on bf16-only (FF graph nodes) |
| Epochs completed | **37 / 50** | +18 (+95%) |
| Peak GPU memory | 24.1 GB | −9.2 GB (kernel fusion, FF concat near-free at +0.3 GB) |
| Recompiles in 30 min | 1 (grad_mode flip at first eval) | None shape-driven across 74K-242K node range |
| W&B run | `ewq3guz2` (`compile-bf16-ff-bsz4`) | — |

Cumulative improvement: **−44.0% val_avg / −44.0% test_avg** since original PR #312 reference (144.21 → 80.85, 131.18 → 73.41).

### Per-split delta vs FF baseline (val)

| Split | FF baseline | Compile + FF | Δ |
|---|---|---|---|
| val_single_in_dist | 117.22 | 84.20 | **−28.2%** |
| val_geom_camber_rc | 125.94 | 93.39 | **−25.8%** |
| val_geom_camber_cruise | 80.26 | 65.95 | **−17.8%** |
| val_re_rand | 104.27 | 79.86 | **−23.4%** |
| **val_avg** | **106.92** | **80.85** | **−24.4%** |

### Stack composition vs single-lever predictions

| Stack | val_avg/mae_surf_p | Δ vs raw bf16 |
|---|---|---|
| bf16 only (#359) | 121.85 | — |
| bf16 + FF (#327) | 106.92 | −12.2% (FF alone) |
| bf16 + compile (off-base, sent back) | 87.20 | −28.4% (compile alone) |
| **bf16 + FF + compile (this run)** | **80.85** | **−33.7% (stack)** |

Subadditive stacking (~83% of naive sum). Both interventions partly cash in
on "more cosine decay reaches the model" — compile gives 2× more epochs,
FF gives faster per-epoch convergence — so they overlap on that mechanism
but throughput-wise are orthogonal (steady batch wall identical, peak GB
+0.3 GB vs compile-only).

### Analysis & conclusions

- **Merged. New round baseline.** Largest single win. Implementation gem:
  `_orig_mod.` prefix strip on save/load keeps W&B model artifacts portable
  into non-compiled modules.
- **Big update on rc-camber understanding:** FF on its own only relieved
  the OOD rc-camber split by −3.3%, suggesting a "camber→pressure mapping"
  representation bottleneck. Compile + FF gets to **−25.8%** on the same
  split — so the rc-camber gap was *schedule-truncation-bound*, not a
  representation bottleneck. The "camber-aware feature embedding" round-2
  experiment is **dequeued**. Throughput unlock alone closed the gap.
- VRAM headroom is now ~70 GB. Capacity scale-up (the parked PR #393
  territory) becomes feasible again.
- **Cosine T_max=50 is now even more misaligned** — the model now reaches
  37 epochs but the schedule still decays as if for 50, so lr at epoch 37
  is ~7.9e-5 (still 16% of peak rather than 0).
- Followup queued: `mode="reduce-overhead"` (CUDA Graphs on dynamic-shape
  compiled graph). Assigned as alphonse's next experiment.



## 2026-04-28 03:29 — PR #481 (closed): `torch.compile(mode="reduce-overhead")` on compile+FF baseline

- branch: `willowpai2d1-alphonse/compile-reduce-overhead` (deleted on close)
- hypothesis: CUDA Graphs via `mode="reduce-overhead"` on top of
  `torch.compile(dynamic=True)` shaves residual Python overhead. Predicted
  0% to -5%.

### Results

| Metric | Value | vs PR #416 (compile+FF baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **84.31** (epoch 35 of 35) | **+4.27%** (regression) |
| `test_avg/mae_surf_p` | 75.75 | +3.19% |
| Per-batch wall (steady) | ~0.121 s | −7% (real shave) |
| Per-epoch wall | ~50.6 s | +3% (no win) |
| Epochs completed | 35 / 50 | −2 |
| Peak GPU memory | 24.1 GB | unchanged |
| W&B run | `30nsmdrn` | |

### CUDA-Graph diagnostics

- 9 distinct CUDAGraph captures during epoch 1 (one per concrete padded
  mesh size).
- 1 recompile event (grad_mode flip, same one-shot as PR #416).
- No graph breaks or capture failures. reduce-overhead is *functional* with
  dynamic=True on PT 2.10; it just doesn't help.

### Analysis & conclusions

- **Closed.** Hypothesis falsified for the dynamic-shape regime.
- Three-mechanism explanation:
  1. **9 CUDAGraph captures eat the shave**: with variable mesh sizes,
     reduce-overhead records a static graph per shape on first encounter.
  2. **Default-mode compile already removed dispatch overhead**, so
     reduce-overhead's ceiling is small (~7% per-batch).
  3. **Validation + dataloader dominate per-epoch budget** post-default-
     compile — saving 3 s of train wall doesn't move the 56 s/epoch
     total much.
- **Throughput frontier essentially exhausted**: bf16 (merged), FF
  (merged), `torch.compile(dynamic=True)` (merged). reduce-overhead,
  bsz=8, bucketing all ruled out. Remaining throughput candidates would
  require padding inputs to fixed shapes (conflicts with simplicity) or
  fundamental architectural rewrites.
- 30-min cap will stay binding until SENPAI_MAX_EPOCHS or
  SENPAI_TIMEOUT_MINUTES change.

## 2026-04-28 03:16 — PR #324 round-2 (sent back AGAIN): EMA decay=0.999 on FF (no compile)

- branch: `willowpai2d1-nezuko/ema-and-grad-clip` (in flight as draft after second send-back)
- hypothesis: EMA decay=0.999 (window matched to ~7K-step training budget)
  with grad_clip dropped per one-hypothesis-per-PR. Predicted -1% to -4%.

### Results (EMA on FF baseline, BEFORE compile merge)

| Metric | Value | vs PR #327 (FF, prior baseline) | vs PR #416 (compile+FF, NEW baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **98.0023** (epoch 19 of 19) | **−8.34%** | **+21.2%** (regression) |
| `test_avg/mae_surf_p` | 88.1348 | −8.97% | +20.1% |
| Per-epoch wall | ~97 s | ≈baseline | +2× (no compile speedup) |
| Peak GPU memory | 33.3 GB | ≈baseline | n/a |
| Best epoch | 19 of 19 (still improving at timeout) | | |
| W&B run | `2xrtj5yt` (`ema999-only`) | | |

### Per-split delta vs FF baseline (val)

| Split | Δ |
|---|---|
| val_single_in_dist | −3.34% |
| val_geom_camber_rc | **−16.00%** (the split FF struggled most with) |
| val_geom_camber_cruise | −3.20% |
| val_re_rand | −8.69% |

### Analysis & conclusions

- **Sent back again.** Result is a clean EMA win at val_avg=98.00 on the
  FF baseline, but compile merged at `9b92e31` while nezuko was running
  rebased to `0941a04` per my prior instructions. Now needs to rebase
  onto compile+FF baseline.
- **Predicted post-rebase val_avg: 70-77.** EMA gave −8.34% on FF; compile
  gave −24.4% on FF. Mechanisms look orthogonal (compile=throughput +
  extra cosine decay reaching the model; EMA=validation-time parameter
  averaging). Stacking at ~80% efficiency lands near 70; mostly orthogonal
  near 73.
- **Useful per-split signal**: EMA preferentially helps rc-camber
  (−16%) — exactly the split FF on its own (−3.3%) and even compile+FF
  (−25.8%) don't fully resolve. EMA's variance reduction in parameter
  space addresses a different rc-camber failure mode. Round-3 hint: full
  4-stack (compile + FF + bf16 + EMA) may be particularly strong on
  rc-camber.
- v1 failure (decay=0.9999 → val=302 from random-init contamination) was
  textbook EMA-warmup pathology; v2 fix is correct and produces clean
  monotonic EMA val curve from epoch 1.
- Followups (bias correction at higher decay, grad_clip alone) queued.

## 2026-04-28 02:54 — PR #314 round-2 (sent back again): SmoothL1/Huber + FF on bf16

- branch: `willowpai2d1-edward/huber-loss` (in flight as draft after second send-back)
- hypothesis: SmoothL1 stacks with FF (orthogonal: FF=upstream features,
  Huber=downstream gradient profile). Predicted ~90.

### Results (Huber + FF on bf16, BEFORE compile merge)

| Metric | Value | vs PR #327 (FF, prior baseline) | vs PR #416 (compile+FF, NEW baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **92.3221** (epoch 19 of 19) | **−13.65%** | **+14.2%** (regression) |
| `test_avg/mae_surf_p` | 84.3064 | −12.92% | +14.8% |
| Per-epoch wall | ~97 s | ≈baseline | +2× (no compile speedup) |
| Peak GPU memory | **~98.2 GB** transient | +5 GB vs FF baseline | over-cap concern |
| W&B run | `eohccx3j` | | |

### Stacking decomposition

| Stack | val_avg | Δ vs raw bf16 |
|---|---|---|
| bf16 only | 121.85 | — |
| bf16 + FF | 106.92 | −12.2% |
| bf16 + Huber | 104.27 | −14.4% |
| **bf16 + FF + Huber** | **92.32** | **−24.2%** |
| bf16 + FF + compile (NEW BASELINE) | 80.85 | −33.7% |

FF + Huber stacking is ~91% of sum-of-individuals (very clean). But the
compile merge happened simultaneously and pushed the baseline past Huber's
best. Edward needs to add Huber on top of compile+FF.

### Analysis & conclusions

- **Sent back, second time.** Result is a clean Huber+FF win, but compile
  merged at the same time and now leads by 11.5 absolute MAE.
- Memory peak ~98.2 GB is concerning — only 4.5 GB headroom on 102.6 GB
  GPU. With compile dropping peak by ~9 GB on FF baseline, the
  compile+FF+Huber peak should land ~88-90 GB. Need to verify
  experimentally. **Cannot scale batch_size with FF + Huber.**
- Predicted post-rebase val_avg: **65-75**. Mechanisms remain orthogonal.
- Followups (β sweep, pure L1) queued for after this rebased run lands.

## 2026-04-28 02:28 — PR #324 (sent back): EMA(0.9999) + grad-clip(1.0) on bf16

- branch: `willowpai2d1-nezuko/ema-and-grad-clip` (in flight as draft after send-back)
- hypothesis: EMA shadow weights + gradient clipping for stability and
  small-data generalization. Predicted -1 to -4%.

### Results

| Metric | Value | vs PR #359 (bf16, rebase target) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **301.99** (epoch 19 of 19) | **+147.8%** (terrible) |
| `test_avg/mae_surf_p` | 285.33 | n/a |
| Train loss curves | healthy (vol 1.81→0.32, surf 1.04→0.17) | normal |
| Pre-clip grad norms | mean 35-116, peak 350-750 | clip firing every batch at 50-100× ratio |
| W&B run | `xmo5f7x7` (`ema9999-clip1`) | |

### Analysis & conclusions

- **Sent back (not closed)** despite +148% regression — diagnosis is
  exemplary and hypothesis isn't falsified.
- **Root cause: EMA-warmup pathology**. At decay=0.9999 over 7,125 steps,
  `0.9999^7125 ≈ 0.490` → ~49% of EMA shadow weights are still random
  init at the "best" checkpoint. Linear-blend prediction
  `0.49 × ~500 (random init val) + 0.51 × 121.85 ≈ 307` matches observed
  302 within 2%. Textbook EMA-warmup failure on a too-short budget.
- **Original PR violated one-hypothesis-per-PR.** My assignment bundled
  EMA + grad_clip together. Send-back drops grad_clip so the EMA test
  is clean; grad_clip alone could be a future experiment if EMA wins.
- Send-back instructions: rebase onto FF, ema_decay=0.999 (window 1K
  steps → 0.08% init contribution at 7125 steps), drop grad_clip.
- Predicted rebased result: -0.5 to -3% on val_avg, ie around 103-106
  on the FF baseline. Will become the new round baseline if it lands.

## 2026-04-28 01:58 — PR #416 (sent back): `torch.compile(dynamic=True)` pilot

- branch: `willowpai2d1-alphonse/torch-compile-pilot` (in flight as draft after send-back)
- hypothesis: torch.compile(dynamic=True) gives single-graph dynamic-shape
  handling for variable mesh sizes; expected 1.3-1.7× speedup on top of
  bf16. Predicted -2 to -5% on val_avg.

### Results (vs bf16 baseline; ran on pre-FF advisor branch)

| Metric | Value | vs PR #359 (bf16, rebase target) | vs PR #327 (FF, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **87.2042** (epoch 37 of 37 completed) | **−28.4%** | **−18.5% (still beats FF baseline!)** |
| `test_avg/mae_surf_p` | **78.4099** | **−29.5%** | **−19.0%** |
| Per-epoch wall (steady) | **48.6 s** mean | **−50%** (2.0× speedup) | −50% |
| Epochs completed | **37 / 50** | **+18** | +18 |
| Peak GPU memory | **23.8 GB** | **−9.1 GB** (less!) | less |
| W&B run | `bpkl3tch` (`compile-bf16-bsz4`) | | |

### Compile diagnostics (clean signal)

| Quantity | Value |
|---|---|
| First batch wall (cold compile) | 9.26 s |
| Steady-state batch wall | mean 0.119 s, max 0.156 s |
| Recompiles in entire 30-min run | **1** (one-time, `grad_mode` flip at first val) |
| Shape recompiles | **None** despite 74K-242K node range |
| Graph breaks / eager fallbacks | None observed |

### Per-split val (best epoch 37)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 97.02 | 1.18 | 0.64 |
| val_geom_camber_rc | 99.04 | 1.94 | 0.82 |
| val_geom_camber_cruise | 67.35 | 0.84 | 0.49 |
| val_re_rand | 85.41 | 1.41 | 0.65 |
| **val_avg** | **87.20** | 1.34 | 0.65 |

### Analysis & conclusions

- **Sent back, not merged.** alphonse's branch was created off the
  pre-FF advisor branch (between bf16 and FF merges), so this run is
  bf16 + compile, **no FF**. Even so, val_avg=87.20 already beats the
  current FF baseline 106.92 by 18.5%. Compile is the largest single
  lever found in the round so far.
- **Mechanism is mostly the cosine schedule, not the compile itself.**
  2.0× per-epoch speedup → 37 epochs → cosine decays to ~16% of peak lr
  vs ~78% at the bf16-baseline best epoch. Compile is the lever that
  exposes the schedule. The student called this out clearly in the
  analysis.
- Implementation gem: the `_orig_mod.` prefix strip on `state_dict()`
  save/load — without that fix, the W&B model artifact would be silently
  unloadable into a non-compiled module. Exactly the right level of
  detail for shipping torch.compile to production training.
- Peak VRAM dropped 9.1 GB from kernel fusion. Capacity scale-up just
  got more headroom (~72 GB free).
- The right next test is **compile + FF stacked**. Mechanisms look
  orthogonal (throughput vs feature representation). Predicted val_avg
  in the 80-85 range after rebase + re-run.
- Other queued followups: `mode="reduce-overhead"` (CUDA Graphs), capacity
  scale-up revisited, cosine T_max alignment compose with compile.

## 2026-04-28 01:54 — PR #314 (sent back): SmoothL1 / Huber loss (β=1.0)

- branch: `willowpai2d1-edward/huber-loss` (in flight as draft after send-back)
- hypothesis: replace MSE with SmoothL1 (β=1.0) to align loss with the MAE
  metric and bound gradients on high-Re outliers. Predicted -2 to -5%.

### Results (vs bf16 baseline, but BEFORE FF merge)

| Metric | Value | vs PR #359 (bf16) | vs PR #327 (FF, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **104.2658** (epoch 17 of 19) | **−14.4%** | **−2.5% (still beats FF baseline!)** |
| `test_avg/mae_surf_p` | **92.1301** | **−17.1%** | **−4.8%** |
| Per-epoch wall | ~97 s | ≈baseline | ≈baseline |
| Peak GPU memory | ~95.7 GB transient (epochs 1-2), ~47 GB steady | +14 GB steady | +14 GB steady |
| Epochs completed | 19 / 50 | same | same |
| W&B run | `czpoam0v` (`smoothl1-beta1`) | | |

### Per-split val (epoch 17 best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist | 136.57 | 1.83 | 0.82 | 131.62 |
| val_geom_camber_rc | 112.14 | 2.13 | 0.96 | 114.45 |
| val_geom_camber_cruise | 75.12 | 0.98 | 0.55 | 71.05 |
| val_re_rand | 93.23 | 1.56 | 0.75 | 89.04 |
| **val_avg** | **104.27** | 1.62 | 0.77 | 101.54 |

### Analysis & conclusions

- **Sent back, not merged.** Edward rebased onto pre-FF advisor branch
  (tip 0069451) but the FF merge happened at the same time (f17992d).
  Edward's run is essentially Huber+bf16 with no FF. Even so, the result
  (104.27) is *already below the FF baseline* (106.92) — Huber alone is
  a stronger lever than FF.
- The right test is **Huber on top of FF** (orthogonal mechanisms): FF
  improves spatial frequency representation, Huber bounds gradients on
  high-Re outliers. Predicted to stack to val_avg ~90.
- Memory anomaly (transient 95.7 GB peak in epochs 1-2) is from the
  caching allocator; SmoothL1 has more autograd intermediates than MSE.
  Steady-state +14 GB. Means we can't safely scale batch_size with Huber
  without instrumentation — flagged for future PRs.
- Per-split improvement is **uniform** (unlike pressure-weighting which
  was heavily split-dependent). All four splits improve substantially.

## 2026-04-28 01:54 — PR #313 (closed, rebased run): Pressure-weighted MSE on bf16

- branch: `willowpai2d1-askeladd/pressure-channel-loss-weight` (deleted on close)
- hypothesis: same as original #313 (pressure-weighted MSE 5x p), but rebased
  onto bf16 baseline.

### Results

| Metric | Value | vs PR #359 (bf16, rebase target) | vs PR #327 (FF, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | 122.5350 (epoch 19 of 19) | **+0.56%** (flat) | **+14.6%** (regression) |
| `test_avg/mae_surf_p` | 112.7175 | +1.41% | +16.4% |
| W&B run | `wf3av0ps` | | |

### Per-split deltas vs bf16 baseline

| Split | Δ |
|---|---|
| val_single_in_dist | +4.19% |
| val_geom_camber_rc | **+5.26%** (sign flip — was −14% pre-bf16) |
| val_geom_camber_cruise | −5.65% |
| val_re_rand | −3.78% |

### Analysis & conclusions

- **Closed.** Excellent diagnosis from askeladd: pressure weighting and
  bf16 are *not* orthogonal — both addressed the same underlying issue
  (high-Re pressure samples dominating gradient). Once bf16 closed the
  rc-camber gap (170.34 → 130.28 between PR #312 and PR #359), the
  additional pressure weighting just starves Ux/Uy. Surface Ux/Uy errors
  went +25-40% across every split — that's the smoking gun.
- The orthogonality assumption in my send-back was wrong. Documented for
  the future.
- Followup queued and assigned: surface-only pressure weighting (askeladd
  PR #451). Restricting (1,1,5) to surf_loss only might preserve cruise /
  re_rand wins without volume Ux/Uy damage.

## 2026-04-28 01:50 — PR #327 (merged, NEW BASELINE): Sinusoidal Fourier features for (x, z), K=8

- branch: `willowpai2d1-tanjiro/fourier-features-positions` (deleted on merge)
- hypothesis: concatenate sinusoidal Fourier features
  `[sin(2^k π x), cos(2^k π x), sin(2^k π z), cos(2^k π z)] for k=0..7` to
  the per-node feature vector. Predicted -2 to -6%.

### Results

| Metric | Value | Δ vs prior baseline (PR #359, bf16) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **106.9223** (epoch 19 of 19) | **−12.2%** |
| `test_avg/mae_surf_p` | **96.8186** | **−12.9%** |
| Per-epoch wall | 97-100 s (mean ~98) | ≈baseline (FF compute trivial) |
| Peak GPU memory | 33.3 GB / 96 GB | +0.4 GB (negligible) |
| Param count | 0.67 M | +0.01 M |
| Epochs completed | 19 / 50 | same |
| W&B run | `nbyicdne` (`ff-K8-bf16`) | — |

### Per-split surface MAE (val, vs prior bf16 baseline)

| Split | Δ |
|---|---|
| val_single_in_dist | **−17.0%** (141.24 → 117.22) |
| val_geom_camber_rc | −3.3% (130.28 → 125.94) |
| val_geom_camber_cruise | **−19.6%** (99.83 → 80.26) |
| val_re_rand | −10.2% (116.04 → 104.27) |

### Analysis & conclusions

- **Merged. New round baseline.** Largest single win of the round so far.
- Tanjiro rebased onto current advisor branch before running so this is
  apples-to-apples vs the bf16 baseline. Excellent practice.
- FF computed in fp32 *before* the bf16 autocast scope so sin/cos aren't
  quantised. Right composition pattern.
- Train loss drops faster per epoch with FF: by epoch 5 train_surf is ~0.41
  vs ~0.50 for bf16-baseline at the same point. Canonical Tancik et al.
  (2020) Fourier-features behaviour.
- **Per-split asymmetry is informative:** cruise and single-in-dist
  improve ~17–20%, val_re_rand 10%, but rc-camber held-out only 3-4%. The
  rc holdout is bottlenecked by *camber → pressure mapping* under unseen
  geometry, not by spatial-frequency representation. Targeted future
  experiment direction.
- Followup directions queued: Gaussian random Fourier features (Tancik's
  variant, often beats deterministic ladder), K sweep, FF on saf/dsdf.

## 2026-04-28 01:03 — PR #393 (closed): Half-step capacity scale-up on bf16 (h=160, L=5, heads=5, slices=80)

- branch: `willowpai2d1-alphonse/halfstep-capacity-on-bf16` (deleted on close)
- hypothesis: with bf16 throughput unlocked + ~63 GB headroom, the bigger
  model that fern's #318 couldn't test is finally feasible. Predicted -2 to -7%.

### Results

| Metric | Value | vs PR #359 (bf16 baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **131.05** (epoch 11 of 14) | **+7.55%** |
| `test_avg/mae_surf_p` | 114.89 | +3.37% |
| Per-epoch wall (mean) | 135 s | +39% |
| Peak GPU memory | 45.6 GB / 96 GB | +38.6% |
| Param count | 1.02 M | +54% |
| Epochs completed | 14 / 50 | -5 |
| W&B run | `nbkqn78z` | — |

### Analysis & conclusions

- **Closed.** val regression past 5% threshold, but well-diagnosed —
  schedule-budget mismatch dominates: best at epoch 11 of 14 with val
  curve still descending; lr was still ~41% of peak at the timeout because
  `T_max=50` doesn't decay in 14 epochs.
- test_avg only +3.37% (vs val +7.55%) — the bigger model isn't
  fundamentally worse, just under-converged.
- **Capacity hypothesis parked, not abandoned.** Once fern's #407 cosine
  T_max alignment lands, retest capacity against the proper schedule.
- Per-epoch wall scales roughly linearly with parameter count (1.55× params
  → 1.39× wall) — no compute pathology.

## 2026-04-28 00:49 — PR #384 (closed): Domain-bucketed batch sampler

- branch: `willowpai2d1-fern/domain-bucketed-sampler` (deleted on close)
- hypothesis: bucket batches by domain so each batch is homogeneous in mesh
  size, cutting padding waste from `pad_collate`. Predicted ~1.2-1.5×
  per-epoch speedup; predicted -2% to -6% on val_avg.

### Results

| Metric | Value | vs PR #359 baseline (bf16) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **125.91** (epoch 15 of 16) | **+3.3%** (worse) |
| `test_avg/mae_surf_p` | 115.40 | +3.8% (worse) |
| Per-epoch wall (mean) | **115.2 s** | **+17%** (slower!) |
| Peak GPU memory | **42.1 GB** | **+28%** (more) |
| Epochs completed | 16 / 50 | -3 |
| W&B run | `00fl62dc` | — |

vs old (pre-bf16) baseline: −12.7%, but ranking is now against the new
post-bf16 baseline.

### Analysis & conclusions

- **Closed.** Hypothesis falsified — bucketing made throughput *worse*, not
  better. Two-mechanism explanation from fern is convincing:
  1. **CUDA caching-allocator fragmentation.** Cycling between 3 max_n
     shapes (4×210K, 4×127K, 4×85K) defeats the allocator's pool reuse —
     the old WeightedRandomSampler had ~80% of batches padding to ~210K so
     the allocator settled on a single dominant pool. Bucketing forces
     three pools.
  2. **Dataloader pipeline mismatch.** GPU step time varies ~2.5× across
     domains while pad_collate worker time is roughly constant; bucketing
     breaks the worker-GPU pipeline overlap.
- The +28% memory regression is consistent with allocator fragmentation;
  the +17% wall regression is consistent with both mechanisms.
- **Throughput-via-sampler is ruled out** as a quick win on this trainer.
  Future throughput attempts should look at `torch.compile`, attention
  flavor swaps, or gradient checkpointing (when scaling capacity), not
  sampler shape.
- Followups parked: length-bucket only the cruise outlier, sort-and-bucket
  NLP-style, memory_history diagnostic. None promising enough to assign
  immediately.

## 2026-04-28 00:20 — PR #313 (sent back): Pressure-channel-weighted MSE (5x p)

- branch: `willowpai2d1-askeladd/pressure-channel-loss-weight`
- hypothesis: weight pressure 5× in the per-channel MSE to align loss with
  the metric. Predicted -3% to -8%.

### Results (pre-bf16 run)

| Metric | Value | vs PR #312 (old baseline) | vs PR #359 (NEW baseline 121.85) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **138.1556** (epoch 14 of 14) | **−4.20%** | **+13.4%** |
| `test_avg/mae_surf_p` | 125.9776 | −3.97% | +13.3% |
| W&B run | `gxcli1lf` (`p-weight-5x`) | | |

### Per-split deltas (val, vs old baseline)

| Split | Δ vs baseline |
|---|---|
| val_single_in_dist | **+2.4%** (regressed) |
| val_geom_camber_rc | **−14.0%** (huge OOD geometry win) |
| val_geom_camber_cruise | −5.4% |
| val_re_rand | +1.2% (flat) |

### Analysis & conclusions

- **Sent back, not merged or closed.** Win on prior baseline was clean
  (−4.2% on val_avg) but the run preceded the bf16 merge (PR #359). Vs the
  new bf16 baseline (121.85), this is +13.4% — but pressure weighting and
  bf16 are orthogonal, so expected behavior is they stack.
- **Action:** rebase onto post-bf16 advisor branch and re-run.
- Excellent per-split characterisation: pressure weighting helps most where
  pressure dominates the surface-error budget (geom_camber_rc, the hardest
  split). Slight regression on the easier in-dist split because relative
  velocity-channel gradient drops.
- Suggested followups (queued): sweep over weights (1,1,3 / 1,1,5 / 1,1,8),
  decoupled surf vs vol channel weights.

## 2026-04-28 00:21 — PR #359 (merged, new baseline): bf16 autocast on forward + loss

- branch: `willowpai2d1-alphonse/bf16-autocast` (deleted on merge)
- hypothesis: throughput, not capacity, is the binding constraint at 30-min
  cap. bf16 autocast on forward + loss should shorten per-epoch wall enough
  to actually exercise the cosine schedule and improve val_avg/mae_surf_p.
  Predicted -3% to -10%.

### Results

| Metric | Value | Δ vs prior baseline (PR #312) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **121.8478** (epoch 16 of 19 completed) | **−15.5%** |
| `test_avg/mae_surf_p` | **111.1495** | **−15.3%** |
| Per-epoch wall | 96–99 s (mean ~97 s) | **−26%** |
| Epochs completed | 19 / 50 | +5 epochs |
| Peak GPU memory | 32.9 GB / 96 GB | **−22%** (~63 GB headroom) |
| Optimizer steps | 19 × 375 = 7,125 | n/a (≈+36% vs baseline 5,250) |
| W&B run | `ot9decu8` (`bf16-bsz4`) | — |

Per-split val (best ckpt): single 141.24 / rc 130.28 / cruise 99.83 / re_rand 116.04.
Per-split test: single 123.73 / rc 121.54 / cruise 85.65 / re_rand 113.68.

### Analysis & conclusions

- **Merged as new round-1 baseline.** BASELINE.md updated.
- bf16 autocast was the single highest-value experiment of the round so far —
  prediction of "throughput first, then capacity" is now solidly evidenced.
- The 26% per-epoch speedup is below the upper end of the predicted 1.5–2×;
  the model is small (0.66M params reported by the printed banner) so
  CPU-side dataloader/normalization is a meaningful share of the step.
  Plenty of compute upside remains.
- No bf16 numerical issues observed. All forward steps and per-split val
  losses were finite. The single `loss=NaN` print on `test_geom_camber_cruise`
  is from `train.py::evaluate_split`'s normalised-loss accumulator (which
  doesn't filter non-finite-y) — same cosmetic bug fern flagged on PR #360,
  unrelated to bf16.
- **Headroom snapshot post-merge**: 63 GB of GPU memory free, 31 epochs of
  unused budget per run if the schedule could decay properly. Capacity
  scale-up experiments that were untestable at the old throughput are now
  feasible.

## 2026-04-28 00:02 — PR #360 (closed): Larger batch (bsz=8, lr=7.07e-4)

- branch: `willowpai2d1-fern/batch-size-8-lr-scaled` (deleted on close)
- hypothesis: doubling batch size from 4→8 with sqrt-scaled lr gives a
  meaningful per-epoch wall-clock speedup, letting more cosine schedule run
  inside the 30-min cap. Predicted -2% to -7%.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **148.7170** (epoch 13 of 14 completed) |
| `test_avg/mae_surf_p` | 136.3675 |
| Per-epoch wall | 129.5 s (vs baseline 131 s — **-1.1%, no win**) |
| Optimizer steps | 2444 (vs baseline 5250 — **-53%**) |
| Peak GPU memory | 84.2 GB / 96 GB |
| Stability | clean (no NaN, lr=7.07e-4 was safe) |
| W&B run | `6977miuh` (`bsz8-lr7e-4`) |

vs baseline: val_avg **+3.12%**, test_avg **+3.95%**. Below 5% close
threshold, but ruled-out direction with strong diagnosis.

### Analysis & conclusions

- **Closed.** Negative result, but high information value.
- **Trainer is not kernel-launch-bound at bsz=4.** Doubling B doubled HBM
  traffic and padded-node count, so per-step compute roughly doubled →
  per-epoch time barely changed → fewer gradient updates → worse
  convergence.
- **Padding waste is the real bottleneck.** `pad_collate` pads to max-N in
  the batch; with cruise meshes at ~210K and raceCar single at ~85K, every
  random-composition batch is padded to ~210K nodes regardless of how many
  small samples it contains. Activations scaled near-linearly with B (42 →
  84 GB).
- **Redirect:** fern's own followup — *domain-bucketed batch sampler* — is
  the right next move; assigned as PR #384.
- The wider+deeper scale-up is parked until throughput is unblocked.
- Cosmetic NaN: train.py's `evaluate_split` still uses `(pred-y_norm)**2`
  with no finite-y filter, so the printed test/cruise normalised loss
  shows NaN. Doesn't affect MAE rankings (those go through the patched
  accumulator). Not fixed.

## 2026-04-27 23:15 — PR #312: Round-1 reference baseline (Transolver default config)

- branch: `willowpai2d1-alphonse/baseline-default`
- hypothesis: establish a clean reference number on the advisor branch by
  running the default Transolver config unchanged.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **144.2118** (epoch 10 of 14 completed) |
| `test_avg/mae_surf_p` | **131.1823** (best val checkpoint) |
| Wall time | 30-min `SENPAI_TIMEOUT_MINUTES` binding (~131 s/epoch) |
| Peak GPU memory | 42.1 GB / 96 GB |
| W&B run | `x33nmv34` ([link](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/x33nmv34)) |

Per-split val (best checkpoint): single 169.70 / rc 170.34 / cruise 110.70 /
re_rand 126.11. Per-split test: single 150.39 / rc 155.05 / cruise 93.29 /
re_rand 125.99.

### Analysis & conclusions

- **Merged** as round-1 baseline. BASELINE.md updated.
- **Bug found and fixed.** Alphonse diagnosed a `0 * Inf = NaN` poisoning
  in `data/scoring.py` triggered by `test_geom_camber_cruise/000020.pt`
  (the only Inf-pressure sample in the corpus). I cherry-picked the
  documented fix into commit `b78f404` on the advisor branch — sibling PRs
  now report finite `test_avg/mae_surf_p`.
- **Throughput is the binding constraint** at the current model size.
  Only 14 of 50 epochs ran; the cosine schedule barely decayed; VRAM is
  >50 GB underutilised. AMP/bf16, larger batch, or `torch.compile` are the
  obvious first knobs and should be the highest-priority next experiment.
- Among val splits, cruise is easiest, then re_rand, then rc/single. Val
  and test rankings agree, which is a useful sanity check on the four-track
  split design.

## 2026-04-27 23:22 — PR #321 (round 1, sent back): 5-epoch LR warmup + cosine to 0 with peak lr=1e-3

- branch: `willowpai2d1-frieren/lr-warmup-and-higher-peak`
- hypothesis: warmup + higher peak lr (1e-3) improves over default no-warmup
  cosine from 5e-4. Predicted -2% to -5% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **148.38** (epoch 14 of 14 completed) |
| `test_avg/mae_surf_p` | NaN (scoring bug + non-finite cruise pred) |
| Wall time | 30-min cap binding (~132 s/epoch) |
| Peak GPU memory | 42.1 GB / 96 GB |
| W&B run | `4ba8w3wb` (`warmup5-peak1e3`) |

vs baseline (PR #312, 144.21): **+2.9% regression**. Under the 5% close
threshold; sent back, not closed.

### Analysis & conclusions

- **Sent back** for variation: peak=7e-4 instead of 1e-3.
- Frieren caught a real bug in my pseudocode (`LinearLR` requires
  `end_factor ≤ 1`; my literal version would also leave cosine annealing
  from `1e-5` to `0` due to base_lrs capture). Their reimplementation is
  semantically correct.
- The peak=1e-3 caused a val regression at epochs 6-7 right after warmup
  ended (val: 178 → 254 → 259 → 178), which is exactly what the student
  flagged as a likely problem. peak=7e-4 should be calmer.
- Schedule never reached cosine tail (only 14/50 epochs ran, lr at end
  was still ~9e-4). Throughput PRs (#359 bf16, #360 bsz=8) will let
  warmup+cosine actually be evaluated to convergence in a future round.
- Frieren also separately reported non-finite *predictions* on a cruise
  test sample (vol_loss=inf in normalized space). The scoring fix in
  b78f404 removes the test-cruise NaN from the *scoring*-side, but a
  blown-up prediction is a separate model-stability concern that should
  be calmer with a lower peak lr.

## 2026-04-27 23:16 — PR #318: Wider+deeper Transolver (h=192, L=6, heads=6, slices=96)

- branch: `willowpai2d1-fern/wider-deeper-transolver` (deleted on close)
- hypothesis: scale Transolver up — h=192, L=6, heads=6, slice_num=96 —
  predicted -3% to -8% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **175.8511** (epoch 3 of 7 completed) |
| `test_avg/mae_surf_p` | **NaN** (epoch-3 checkpoint produced non-finite p on a cruise test sample, *and* pre-fix scoring NaN-poisoning was active) |
| Wall time | 30-min cap binding (~275 s/epoch, 2.1× baseline cost) |
| Peak GPU memory | 83.8 GB / 96 GB |
| Param count | 1.72 M |
| W&B run | `rzn96bqj` (`h192-l6-h6-s96`) |

vs baseline (PR #312): val_avg = 175.85 vs 144.21 → **+22% regression**.

### Analysis & conclusions

- **Closed.** Result is well past the >5% close threshold, but the
  underlying hypothesis (more capacity ⇒ better generalization) was *not
  fairly tested*: only 7 of 50 epochs ran in the budget, the model was still
  oscillating downward, and the cosine LR barely moved.
- Fern's analysis was excellent: hypothesis untestable at this throughput,
  not falsified.
- **Redirect:** the right next step is throughput improvement first
  (AMP/bf16, larger batch given 50+ GB headroom, possibly `torch.compile`),
  *then* a half-step scale-up. I've assigned that as fern's round-1.5
  experiment.
