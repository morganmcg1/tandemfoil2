# SENPAI Research Results — charlie-pai2d-r5

## 2026-04-28 00:55 — PR #380: Best-val checkpoint averaging (top-3) — **REQUEST CHANGES (rebase + val-on-averaged)**

- Branch: `charliepai2d5-frieren/ckpt-avg-top3` (on L1-only, not L1+warmup)

### Results (on L1 baseline — pre-warmup)

| metric | value | vs L1 baseline (101.87 / 102.61) | vs current baseline (94.54 / 91.85) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (single best) | 104.43 | +2.5% (worse, run-to-run noise) | +10.5% (worse) |
| `val_avg/mae_surf_p` (averaged) | **not measured** | — | — |
| `test_avg/mae_surf_p` (3 clean) | **91.13** | **−11.2%** ✓ | −0.8% (small win) |

Top-3 averaged epochs: 12 (val=104.43), 13 (108.96), 14 (108.42). Per-epoch wall: 131.1s (unchanged from L1 baseline). Averaging adds < 1% overhead.

### Decision

Send back for:
1. **Rebase onto current advisor** (L1 + warmup + budget-matched cosine). Squash-merging now would revert PR #296's warmup scheduler — same mechanic issue as thorfinn's #365.
2. **Add val-on-averaged-model evaluation.** The current implementation only runs the averaged model on test, so we can't rank by `val_avg/mae_surf_p`. Student's own follow-up #3 — easy addition, one extra `evaluate_split` pass.

The technique works. Test improvement is real and large (−11.2% vs L1). Stacked on L1+warmup it should give a clean new test-side best. The val-on-averaged measurement closes the only methodological gap.

---

## 2026-04-28 00:20 — PR #278 (rerun): surf_p_weight=5 on top of L1 — **CLOSE (hypothesis falsified)**

- Branch: `charliepai2d5-alphonse/pressure-surface-weight` (rebased onto L1, not onto current L1+warmup)

### Results

| metric | value | vs L1 baseline (101.87) | vs current baseline (94.54) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (best ep 13/14) | **108.63** | +6.6% (worse) | +14.9% (worse) |
| `val_single_in_dist/mae_surf_p` | 134.66 | +7.5% | — |
| `val_geom_camber_rc/mae_surf_p` | 132.11 | **+22.3%** | — |
| `val_geom_camber_cruise/mae_surf_p` | 70.31 | −6.6% | — |
| `val_re_rand/mae_surf_p` | 97.44 | −1.5% | — |
| `test_avg/mae_surf_p` (3 clean) | 112.49 | +9.6% | — |

### Decision

Close. Hypothesis cleanly falsified: `surf_p_weight=5` on L1 is **+6.6% worse** than L1 baseline (past the 5% close threshold), with the dominant cost on `val_geom_camber_rc` (+22.3%). Student's analysis is excellent — under L1, gradient magnitudes are sign-based and per-element, so 5× channel weighting routes 71% of surface gradient onto `p`, starving Ux/Uy. Since the model is parameter-shared across channels, degraded velocity learning hurts the joint flow representation that pressure prediction relies on.

The same gradient-budget reasoning predicts that any `surf_p_weight > 1` under L1 trades Ux/Uy starvation for pressure emphasis with no good operating point — channel weighting and L1 don't compose well. Reassigned alphonse to **gradient clipping `max_norm=1.0`** (PR #387) — a no-cost stability hypothesis that may also reduce the test-time non-finite-prediction patterns alphonse helped diagnose.

---

## 2026-04-28 00:15 — PR #365: Fourier positional features (8 freqs, normalized x,z) — **REQUEST CHANGES (rebase mechanic only)**

- Branch: `charliepai2d5-thorfinn/fourier-features`
- Hypothesis: 8-band sinusoidal Fourier features on normalized node positions relax MLP spectral bias and improve surface-pressure fidelity.

### Results (on L1 baseline — pre-warmup; not rebased onto current advisor)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 13/14) | **89.30** |
| `val_single_in_dist/mae_surf_p` | 108.97 (-13.0% vs L1) |
| `val_geom_camber_rc/mae_surf_p` | 98.80 (-8.5%) |
| `val_geom_camber_cruise/mae_surf_p` | 67.31 (-10.6%) |
| `val_re_rand/mae_surf_p` | 82.12 (-17.0%) |
| `test_avg/mae_surf_p` (3 clean) | **88.94** (-13.3% vs L1) |
| Per-epoch wall (s) | 131.91 (vs 131.82 baseline — essentially free) |
| Peak GPU memory (GB) | 42.36 (vs 42.11 — +0.6%) |

All four val splits improved monotonically. Result substantially exceeded the predicted 2–5% delta (~12.3% achieved).

### Decision

**Send back for rebase only — the experiment was right, the merge mechanic is wrong.** Thorfinn's branch was created from L1-only (post-PR-#293 but pre-PR-#296), so squash-merging now would revert PR #296's warmup scheduler. Beats current baseline (94.54) by 5.6% even without warmup; rerun on top of L1+warmup is expected to produce a clear new best. No experiment changes — pure git mechanic.

After the rebased rerun lands, this is likely the round-2 winner.

---

## 2026-04-28 00:05 — PR #296 (rerun): Linear warmup → cosine, peak lr 1e-3, --epochs 14 — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-fern/lr-warmup-1e3` (rebased onto post-L1 advisor)
- Hypothesis: with the schedule matched to the wall-clock budget, warmup → cosine decay should let the model converge into a low-LR refinement regime that L1's plain cosine-over-50 can't reach.

### Results (on top of L1 baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 12/14) | **94.5397** |
| `val_single_in_dist/mae_surf_p` | 114.295 |
| `val_geom_camber_rc/mae_surf_p` | 105.456 |
| `val_geom_camber_cruise/mae_surf_p` | 70.448 |
| `val_re_rand/mae_surf_p` | 87.961 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **91.853** |

Best epoch landed at end-of-epoch-12, LR ≈ 2.5e-4 (mid-cosine-decay) — schedule worked exactly as designed.

### Decision

Merge. Beats the L1-only baseline by **−7.2% val** and **−10.5% test (3-clean-split)**. Two clean orthogonal axes (loss + schedule) now stacked. The `test_geom_camber_cruise/p` NaN is unchanged from the cohort-wide pre-existing data issue.

Reassigned fern to `weight_decay 1e-4 → 5e-4` (PR #385) — single-axis test of whether stronger regularization helps the OOD camber splits.

---

## 2026-04-28 00:05 — PR #303: EMA weights (decay 0.999) — **REQUEST CHANGES (rebase onto L1+warmup)**

- Branch: `charliepai2d5-tanjiro/ema-weights`
- Hypothesis: per-step EMA of model weights with decay 0.999 should improve generalization by 2–5%.

### Results (on pre-L1 MSE baseline — student honestly noted not rebased)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50, EMA model) | **127.65** |
| `test_avg/mae_surf_p` (3 clean) | **125.63** |
| **EMA vs live diagnostic** | epoch 5: live wins by 32; epoch 10: **EMA wins by 4.66%** ✓ |

The EMA-vs-live tracking confirmed the predicted 2–5% delta empirically. The hypothesis works mechanically — the issue is just that this run was on MSE not L1+warmup.

### Decision

Send back. EMA is loss/schedule-agnostic, so the 4–5% relative delta should stack on top of L1+warmup. Action: rebase onto the new advisor branch (which has L1 + warmup + `epochs=14` budget) and rerun with `--ema_decay 0.999 --lr 1e-3 --epochs 14`. Keep the every-5-epoch live-vs-EMA diagnostic — it's a great instrumentation choice we want to retain.

Independent diagnosis of the cruise NaN matches the cohort-wide finding.



## 2026-04-27 23:30 — PR #293: L1 loss in normalized space (alignment with MAE eval metric) — **MERGE (winner)**

- Branch: `charliepai2d5-edward/l1-loss`
- Hypothesis: replace MSE with L1 in normalized space; MAE-aligned with the eval metric, more robust to high-Re outliers.

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **101.868** |
| `val_single_in_dist/mae_surf_p` | 125.264 |
| `val_geom_camber_rc/mae_surf_p`  | 108.034 |
| `val_geom_camber_cruise/mae_surf_p` |  75.262 |
| `val_re_rand/mae_surf_p` | 98.912 |
| `test_avg/mae_surf_p` (4-split, with NaN) | NaN |
| `test_avg/mae_surf_p` (3 clean splits) | **102.606** |
| `test_single_in_dist/mae_surf_p` | 113.966 |
| `test_geom_camber_rc/mae_surf_p` |  99.998 |
| `test_geom_camber_cruise/mae_surf_p` | NaN (data bug) |
| `test_re_rand/mae_surf_p` | 93.854 |

Metric summary: `models/model-l1-loss-20260427-223415/metrics.yaml`

### Analysis

Pure L1 swap, no other changes. Training was numerically clean from epoch 1 (no Huber fallback needed). Validation `val_avg/mae_surf_p` descended monotonically across all 14 reached epochs (266 → 209 → 184 → 171 → 161 → 135 → 142 → 140 → 125 → 124 → 112 → 107 → 106 → 102) and was still trending down at the 30-min timeout. Edward did detective work and identified a pre-existing data + scoring bug that affects the round: `test_geom_camber_cruise` sample 20 has 761 non-finite values in the `p` channel of GT, and `data/scoring.accumulate_batch` computes `err = (pred - y).abs()` *before* masking, which lets NaN propagate into the per-channel sums. Same pattern hit fern (#296) and thorfinn (#305). Read-only constraint on `data/scoring.py` means the fix has to be flagged for the human team or solved via a sanitization pre-step in `train.py`.

### Decision

Merge — clear round-1 winner. New baseline `val_avg/mae_surf_p = 101.87`, 3-clean-split `test_avg/mae_surf_p = 102.61`. The cruise NaN is a pre-existing artifact, not L1's fault, and edward's stability investigation confirmed the model itself produces only finite predictions on that split.

---

## 2026-04-27 23:30 — PR #305: Finer attention: slice_num 64→128, n_head 4→8 — **CLOSE**

- Branch: `charliepai2d5-thorfinn/slices-heads-2x`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 8/50) | **160.676** |
| `val_single_in_dist/mae_surf_p`     | 219.613 |
| `val_geom_camber_rc/mae_surf_p`     | 179.649 |
| `val_geom_camber_cruise/mae_surf_p` | 108.617 |
| `val_re_rand/mae_surf_p` | 134.825 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **162.22** |

Metric summary: `models/model-slices-heads-2x-20260427-223358/metrics.yaml`

### Analysis

Per-epoch wall time was ~252 s vs ~131 s for edward / fern — almost exactly 2× the baseline cost. Inside the 30-min `SENPAI_TIMEOUT_MINUTES` cap this gives only 8 epochs vs 14. Worse, the test split exposed the dim_head=16 instability the PR pre-warned about: model produced non-finite predictions on at least one cruise test sample, `surf_loss=NaN` and `vol_loss=+Inf` on that split. Even granting that the model is far from converged at epoch 8, the per-epoch unit economics make this a poor fit for the current timeout regime.

### Decision

Close. The configuration is fundamentally too slow per epoch to compete with the loss-formulation winners, and the dim_head=16 fragility makes test scoring unreliable. The natural fallback (`n_hidden=192` to restore dim_head=24) overlaps with askeladd's running PR #290, so reassigning thorfinn to a non-overlapping hypothesis is the better use of the slot.

---

## 2026-04-27 23:55 — PR #299: Deeper Transolver: n_layers 5 → 8 — **CLOSE**

- Branch: `charliepai2d5-frieren/deeper-8-layers` (closed)

### Results (on pre-L1 MSE baseline; two replicate runs)

| Run | best `val_avg/mae_surf_p` | best epoch | epochs/30min | per-epoch wall |
|---|---:|---:|---:|---:|
| #1 | 146.31 | 9 | 9 | ~206s |
| #2 (headline) | **139.29** | 9 | 9 | ~206s |

Run #2 per-split val: `val_single_in_dist=169.55`, `val_geom_camber_rc=146.73`, `val_geom_camber_cruise=113.17`, `val_re_rand=127.71`. 3-clean-split test mean: 141.48. test_geom_camber_cruise NaN (same root cause as round-1 cohort).

### Decision

Close. Per-epoch wall time ~206 s (same scale as askeladd's wider-192) → only 9 of 50 epochs reached. Both replicates ~37% worse than the L1 baseline (`val_avg = 101.87`). The val curve was still descending at the cap, so this is again an under-converged snapshot — but as with the wider-192 close (#290) and the slices+heads close (#305), capacity-heavy hypotheses are *structurally* penalized in the 30-min timeout regime: they can't accumulate enough SGD steps to beat the cheaper-per-epoch baselines.

Reassigned frieren to **best-val checkpoint averaging (top-3)** (PR #380) — a no-per-epoch-cost technique that fits the budget regime and addresses the per-epoch noise we saw in their training trajectory.

Worth noting: frieren's run-to-run variance (146.31 → 139.29 from two replicates with the same config) is a useful data point. Single-run round-1 numbers should be treated as having ~5% inherent noise, not as point estimates.

---

## 2026-04-27 23:35 — PR #301: Bump surf_weight 10 to 30 — **REQUEST CHANGES (rebase onto L1)**

- Branch: `charliepai2d5-nezuko/surf-weight-30`
- Hypothesis: push the surface/volume balance harder onto surface fidelity to align with the surface-only eval metric.

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **141.556** |
| `val_single_in_dist/mae_surf_p` | 156.905 |
| `val_geom_camber_rc/mae_surf_p` | 148.448 |
| `val_geom_camber_cruise/mae_surf_p` | 122.728 |
| `val_re_rand/mae_surf_p` | 138.141 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **141.27** |

### Decision

Worse than the L1 baseline of `101.87`, but the change was tested on MSE — we don't know what it does on top of L1. The hypothesis "more surface emphasis improves the surface-only metric" is plausibly orthogonal to the loss type (with L1, gradients are sign-based, so the optimal `surf_weight` may shift). Rebase onto `icml-appendix-charlie-pai2d-r5` (now has L1) and rerun with `--surf_weight 30.0`. Pure CLI flag — trivial rebase.

Excellent independent diagnosis of the cruise NaN scoring path (`err * surf_mask` propagates `NaN * 0 = NaN`); same root cause as edward's PR #293 finding.

---

## 2026-04-27 23:35 — PR #290: Wider Transolver: n_hidden 128→192, slice_num 64→96 — **CLOSE**

- Branch: `charliepai2d5-askeladd/wider-hidden-192`

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 8/9 reached) | **152.238** |
| `val_single_in_dist/mae_surf_p` | 198.823 |
| `val_geom_camber_rc/mae_surf_p` | 155.683 |
| `val_geom_camber_cruise/mae_surf_p` | 120.887 |
| `val_re_rand/mae_surf_p` | 133.559 |
| `test_avg/mae_surf_p` (3 clean) | **151.69** |

### Analysis

Per-epoch wall time was ~205 s vs ~131 s for the loss-formulation winners — the 30-min cap allowed only 9 epochs vs ~14 for the cheaper-per-epoch baselines. Best-val came at epoch 8, still descending, so this is an under-trained snapshot. Even projecting forward, the wider model is structurally penalized by the wall-clock budget: the L1 baseline reached `val_avg = 101.87` in 14 epochs at the same wall time, ~33% better than this wider 8-epoch number.

### Decision

Close. Capacity-heavy hypotheses cannot win in the current 30-min timeout regime — every minute of GPU spent on extra width is a minute not spent annealing through the cosine schedule. Reassigned askeladd to `drop-path 0.1` regularization (PR #369), which has zero per-epoch cost and is well-matched to the small-dataset regime.

Independent NaN observation matches edward / alphonse / nezuko's diagnosis of the `data/scoring.py` bug.

---

## 2026-04-27 23:35 — PR #278: Pressure-channel surface weighting (surf_p_weight=5) — **REQUEST CHANGES (rebase onto L1)**

- Branch: `charliepai2d5-alphonse/pressure-surface-weight`
- Hypothesis: up-weight the pressure channel inside the surface loss by 5× to align gradients with the eval metric.

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 12/50) | **156.16** |
| `val_single_in_dist/mae_surf_p` | 195.74 |
| `val_geom_camber_rc/mae_surf_p` | 162.81 |
| `val_geom_camber_cruise/mae_surf_p` | 131.15 |
| `val_re_rand/mae_surf_p` | 134.94 |
| `test_avg/mae_surf_p` (3 clean) | **149.65** |

### Decision

Worse than L1 baseline of `101.87`, but the change was on MSE. The pressure-channel-weighting code is a per-element broadcast tensor that composes the same way regardless of whether `abs_err` comes from L1 or MSE — should rebase cleanly. Sent back: rebase onto `icml-appendix-charlie-pai2d-r5` (now has L1) and rerun.

Best independent diagnosis of the cruise NaN bug — found that `test_geom_camber_cruise` sample 20 has `-inf` in 761 volume-cell pressure GT values, scoring path: `inf * 0 = NaN` in IEEE 754. Same root-cause edward identified; alphonse's writeup pinpoints volume-cell vs surface and the exact `data/scoring.py:49–50` lines.

---

## 2026-04-27 23:30 — PR #296: Linear warmup then cosine, peak lr 1e-3 — **REQUEST CHANGES (send back)**

- Branch: `charliepai2d5-fern/lr-warmup-1e3`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **137.319** |
| `val_single_in_dist/mae_surf_p`     | 175.812 |
| `val_geom_camber_rc/mae_surf_p`     | 150.559 |
| `val_geom_camber_cruise/mae_surf_p` |  99.339 |
| `val_re_rand/mae_surf_p` | 123.565 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **136.998** |

Metric summary: `models/model-lr-warmup-1e3-20260427-223514/metrics.yaml`

### Analysis

The hypothesis is reasonable but the schedule isn't matched to the budget: `cosine T_max = MAX_EPOCHS - warmup_epochs = 45`, while only 14 epochs were ever run. So warmup occupied epochs 1–5, and epochs 6–14 ran at near-peak LR (~9.4e-4 → 8.2e-4) — effectively a "warmup + plateau at ~1e-3" run rather than the intended warmup+decay. `val_avg/mae_surf_p` was still descending at the timeout. We can't tell whether the schedule helps until cosine actually decays into the wall budget.

### Decision

Send back — set `--epochs 14` so cosine T_max scales to the actually-reachable budget and we get a clean read on the schedule. Same student branch, same hypothesis, just a one-line config tweak.
