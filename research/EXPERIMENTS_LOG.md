# SENPAI Research Results

## Track: icml-appendix-charlie-pai2e-r2

This log records all reviewed experiments for this research track.
Primary metric: `val_avg/mae_surf_p` (lower is better).

---

## 2026-04-29 06:00 — PR #1036: mlp_ratio=3: wider MLP feedforward per Transolver block [CLOSED — DEAD END]

- **Branch:** `charliepai2e2-alphonse/mlp-ratio-3` (CLOSED)
- **Hypothesis:** Increasing `mlp_ratio` from 2 to 3 (MLP hidden width 256→384) provides additional feedforward capacity to encode nonlinear flow features; gains expected in camber-generalization splits.

### Results Table

| Split | Baseline (PR #1001) | mlp_ratio=3 (best e15) | Delta |
|-------|--------------------:|------------------------:|------:|
| val_single_in_dist | 104.94 | 110.40 | +5.46 |
| val_geom_camber_rc | 102.21 | 108.05 | +5.84 |
| val_geom_camber_cruise | 74.27 | 76.61 | +2.34 |
| val_re_rand | 91.02 | 93.93 | +2.91 |
| **val_avg** | **93.1083** | **97.25** | **+4.14** |
| val_avg (ckpt_avg K=3) | — | 97.10 | +4.0 |
| test_avg | 82.37 | 86.58 | +4.21 |

**Params:** 883,149 vs baseline 718,669 (+22.9%). Metrics path: `target/metrics/charliepai2e2-alphonse-mlp-ratio-3/`.

### Analysis & Conclusions

Regression is uniform across all splits, including camber OOD splits — no extrapolation benefit from wider MLP. Two explanations:
1. **Underfit at fixed budget**: 23% more parameters trained for the same 15-epoch cosine schedule reaches a worse local minimum. The val curve was flat at LR floor by e15, confirming the minimum is genuinely worse rather than 'needs more steps'.
2. **Inductive-bias mismatch**: At n_hidden=128/n_head=2 (head_dim=64), mlp_ratio=2 (256-dim MLP hidden) is well-matched to the attention output. Pushing to 384-dim adds capacity the 64-dim attention slices can't populate usefully.

**Decision:** CLOSED. mlp_ratio=2 confirmed as optimal for current dimensions. Student suggests mlp_ratio=1 and SwiGLU at mlp_ratio=2 as future hypotheses.

---

## 2026-04-29 05:00 — PR #1009: Relative MAE on surface pressure (eps=0.1, node-level Re-normalization) [CLOSED — DEAD END]

- **Branch:** `charliepai2e2-fern/relative-mae-pressure-loss` (CLOSED)
- **Hypothesis:** Apply relative MAE loss on the surface pressure channel: `|pred_p - y_p| / (|y_p| + eps)` with eps=0.1. Hypothesis was that percentage-error training provides intrinsic Re-regime normalization, reducing the dominance of high-Re high-pressure samples and improving OOD generalization across Re regimes.
- **Outcome:** DEAD END — val_avg=102.45 vs baseline 93.1083 (+9.34, +10% WORSE). All splits degraded.

### Results Table (best epoch)

| Split | Baseline (PR #1001) | This run | Delta |
|-------|--------------------:|----------:|------:|
| val_single_in_dist | 104.94 | 128.51 | +23.57 |
| val_geom_camber_rc | 102.21 | 112.54 | +10.33 |
| val_geom_camber_cruise | 74.27 | 77.28 | +3.01 |
| val_re_rand | 91.02 | 91.47 | +0.45 |
| **val_avg** | **93.1083** | **102.45** | **+10.0%** |
| test_avg | 82.37 | 93.79 | +11.4 |

surf_loss_pres / surf_loss_vel ratio: 3.80× (pressure channel dominated training signal)

### Why the Hypothesis Failed

1. **Objective misalignment**: Training optimizes percentage error; evaluation measures absolute MAE. Minimizing relative error on low-pressure nodes (near-zero) does not minimize absolute error on high-pressure nodes (leading-edge suction peaks), which dominate mae_surf_p.
2. **Near-zero amplification**: eps=0.1 is insufficient for surface points where |y_p| ≈ 0.1–1.0; those nodes dominate gradients with artificially amplified terms.
3. **Pressure channel dominance**: surf_loss_pres/vel ratio ballooned to 3.8× vs ~1× at baseline — relative MAE on pressure channel overwhelms velocity loss.
4. **Re-weight double-compression**: Per-sample Re-weighting already suppresses high-Re samples; relative MAE further suppresses the high-pressure gradients on those same samples, compounding the mismatch.

**Decision:** CLOSED — objective misalignment is structural. Any relative-error loss must be carefully calibrated against the absolute-MAE eval metric. If revisited: consider hybrid alpha*MSE + (1-alpha)*rMAE with alpha≈0.7–0.8 to maintain gradient direction from the dominant absolute-error term.

---

## 2026-04-29 03:00 — PR #965: Relative MAE surf-p loss (auto Re-regime normalization) [CLOSED — DEAD END]

- **Branch:** `charliepai2e2-edward/relative-mae-surf-p-loss`
- **Hypothesis:** Replace absolute squared-error loss for the surface pressure channel with relative MAE (`|pred_p - y_p| / (|y_p| + ε)`) in original (denormalized) space. Motivation: pressure values vary enormously across Re regimes; relative loss auto-normalizes each sample's contribution by its own pressure magnitude, providing intrinsic Re-regime balance without explicit sample weights.
- **Outcome:** DEAD END — `val_avg/mae_surf_p = 178.14` vs baseline 94.78; +87.9% regression. All splits degraded sharply. Dominant failure mode: gradient explosion.

### Results Table (best epoch 14)

| Split | Baseline (PR #931) | This run (best epoch 14) | Delta |
|-------|-------------------:|-------------------------:|------:|
| `val_single_in_dist` | 104.91 | 228.51 | +117.8% |
| `val_geom_camber_rc` | 105.49 | 185.92 | +76.2% |
| `val_geom_camber_cruise` | 77.32 | 145.60 | +88.3% |
| `val_re_rand` | 91.41 | 152.54 | +66.9% |
| **`val_avg/mae_surf_p`** | **94.78** | **178.14** | **+87.9%** |
| `test_avg/mae_surf_p` | 85.22 | 166.34 | +95.2% |

Note: `test_geom_camber_cruise` surf_loss / vol_loss produced NaN/Inf — extreme outputs in normalized space on some cruise samples.

### Gradient Explosion Analysis

| Metric | This run | Baseline |
|--------|----------:|---------:|
| epoch grad_norm_mean | 630–1530 | 30–85 |
| epoch grad_norm_max | 7,900–14,900 | 370–1,000 |
| effective clip ratio (mean / max_norm=5) | ~150–300x | ~6–17x |

Relative MAE produces gradients ~15–25× larger than squared-error baseline. Every single batch was aggressively clipped, severely impairing optimizer ability to follow loss geometry.

### Why the Hypothesis Failed

1. **Pressure crosses zero.** At surface points where `y_p ≈ 0`, denominator collapses to `P_EPS=1.0`, making relative loss effectively unbounded MAE in original units. These near-zero spatial points dominate the gradient — a point with raw pressure error of 100 contributes a gradient term ~4500× larger than under normalized squared error.
2. **Within-sample imbalance is inverted.** Relative loss amplifies low-magnitude spots (near stagnation lines) and damps high-magnitude spots (leading-edge suction peaks — the aerodynamically important ones). The per-sample Re-weighting from PR #931 already handles cross-sample balance fine; spatial reweighting inverts the priority structure.
3. **`P_EPS = 1.0` is too small** relative to typical |y_p| of order 100–1000. A larger P_EPS (e.g. ~y_std_p/10 ≈ 68) would cap gradients at low-pressure points but partially defeats the auto-normalization motivation.

- Wall clock: 30.8 min (14/50 epochs)
- Peak GPU memory: 42.1 GB
- Metrics: `metrics/charliepai2e2-edward-relative-mae-surf-p-loss/` (on student branch)

### Conclusions

Relative MAE on a quantity that crosses zero with O(thousand)-magnitude tails is a poor fit for SGD-style training. Gradient dynamics are dominated by the wrong spatial points. Abandon this loss formulation.

Student's best suggested follow-up: **Huber/SmoothL1 loss on denormalized pressure** — bounded gradients at large errors, quadratic near zero, can't blow up at any spatial location. Worth testing in a future PR.

**Decision:** CLOSED — clear dead end.

---

## 2026-04-28 20:15 — PR #764: Larger model capacity: n_hidden 128→256

- **Branch:** `charliepai2e2-alphonse/larger-model-capacity`
- **Hypothesis:** Doubling n_hidden (128→256) gives ~4× parameters per layer; wider model should capture finer boundary-layer and pressure-gradient features, lowering `val_avg/mae_surf_p`.
- **Outcome:** INFORMATIONAL — Run terminated by 30-min wall-clock cap at epoch 9/50. Model was clearly undertrained.

### Results Table

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 159.6098 | 2.9486 | 0.8582 |
| val_geom_camber_rc | 151.6750 | 3.2153 | 1.2054 |
| val_geom_camber_cruise | 108.3837 | 2.5150 | 0.6284 |
| val_re_rand | 128.3367 | 2.7760 | 0.9236 |
| **avg** | **137.0013** | 2.8637 | 0.9039 |

- Test split: `test_avg/mae_surf_p = NaN` (non-finite prediction on `test_geom_camber_cruise` poisoned accumulator — undertraining side-effect)
- Parameters: 2,600,279 (~2.60M, ~4× stock)
- Peak VRAM: 73.95 GB / 96 GB
- Epochs completed: 9 / 50 (~3.7 min/epoch with n_hidden=256)
- Validation trajectory: ep1=222.19 → ep9=137.00 (still declining ~17 units/epoch at cutoff)
- W&B run: d0bkhxgp | Metrics: `metrics/charliepai2e2-alphonse-larger-model-capacity-d0bkhxgp.jsonl`

### Analysis

The wider model showed correct learning dynamics — val_avg/mae_surf_p fell monotonically in the final epochs (153.99→137.00) — but hit the 30-min wall clock after only 9/50 epochs. Cosine LR had barely decayed (5.0e-4→4.61e-4), so the schedule was effectively still at full rate. With ~17 units/epoch still dropping, the true converged value is unknown. VRAM is not the bottleneck (73.95 GB peak with headroom). The NaN on test_geom_camber_cruise is an undertraining artifact — large-magnitude pressure predictions overflow before stabilization.

**This run establishes the first measured number for the track: val_avg/mae_surf_p = 137.0013 at epoch 9, n_hidden=256. It is a soft working baseline, not a converged result.**

**Decision:** Closed as informational. Alphonse re-assigned to repeat the n_hidden=256 run with bf16/AMP to fit 50 epochs within the wall-clock budget, enabling a true converged comparison.

---

## 2026-04-28 20:55 — PR #778: Gradient clipping (clip_grad_norm=1.0) [MERGED — NEW BASELINE]

- **Branch:** `charliepai2e2-tanjiro/gradient-clipping`
- **Hypothesis:** Pre-clip gradient norms are extreme on high-Re samples; `clip_grad_norm_(params, 1.0)` after backward stabilizes optimization and unlocks better convergence within the 30-min budget.
- **Outcome:** WIN — `val_avg/mae_surf_p` = **104.7457** at epoch 14/50, a 24% improvement over the original baseline 137.0013.

### Results Table (best-val checkpoint at epoch 14)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 105.24 |
| val_geom_camber_rc | 97.21 |
| val_geom_camber_cruise | 98.39 |
| val_re_rand | 118.15 |
| **avg** | **104.7457** |

- Test split: `test_avg/mae_surf_p = NaN` (`test_geom_camber_cruise` pressure NaN — pre-existing scoring bug, NOT a model regression). Other test splits clean: test_single_in_dist=101.63, test_geom_camber_rc=108.01, test_re_rand=94.42.
- Pre-clip gradient norms: 40–900× above 1.0 threshold on every step — confirms gradient explosion is the dominant pathology on high-Re samples.
- Epochs completed: 14/50 (still improving at cutoff — undertrained).
- Metrics: `metrics/tanjiro-gradient-clipping/`

### Analysis

A single line of code (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` immediately before `optimizer.step()`) cut val_avg/mae_surf_p by ~24%. The dominant issue in the original baseline was gradient explosion from high-Re samples, not model capacity or schedule. This is the kind of simple high-impact fix that reframes the entire research track: every other experiment should rebase onto this baseline before being judged.

**Decision:** MERGED. New baseline = 104.7457. All in-flight experiments need to rebase onto this baseline.

---

## 2026-04-28 20:55 — PR #765: More physics slices (slice_num 64→128) [CLOSED]

- **Branch:** `askeladd/more-slices`
- **Hypothesis:** Doubling slice_num gives finer physics-aware decomposition.
- **Outcome:** FAIL — val_avg/mae_surf_p = 157.8745, worse than even the original baseline of 137.0013.
- **Decision:** CLOSED. Likely amplified gradient instability (more slice projection paths = more places for gradients to blow up). Worth retrying after gradient clipping is in baseline.

---

## 2026-04-28 20:55 — PR #767: Higher surf_weight (10→50) [SENT BACK]

- **Branch:** `fern/higher-surf-weight`
- val_avg/mae_surf_p = 127.16. Beat original baseline (137.0013) but not new baseline (104.7457).
- **Decision:** Sent back — rebase onto clipping baseline and re-run. Surface-weight tuning may compound with stable gradients.

---

## 2026-04-28 19:00 — PR #768: Lower LR + warmup (5e-4→1e-4 + 500-step warmup) [CLOSED — DEAD END]

- **Branch:** `frieren/lower-lr-warmup`
- **Hypothesis:** Lower peak LR (1e-4) + 500-step linear warmup would protect orthogonal in_project_slice initialization from destructive early updates, finding better local minima.
- **Round 1** (pre-gradient-clipping): lr=1e-4 + warmup → val_avg/mae_surf_p = 125.35. Beat old baseline 137.0013.
- **Round 2** (post-gradient-clipping, rebased on PR #778):

| Run | val_avg/mae_surf_p | Delta vs 104.7457 |
|-----|-------------------|--------------------|
| lr=5e-4 + 500-step warmup (epoch 12) | 111.0786 | +6.33 (worse) |
| lr=2e-4 + 500-step warmup (epoch 14) | 115.0022 | +10.26 (worse) |

Per-split (lr=5e-4 + warmup, best epoch 12):
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 130.93 |
| val_geom_camber_rc | 123.76 |
| val_geom_camber_cruise | 84.34 |
| val_re_rand | 105.29 |
| **avg** | **111.08** |

- **Analysis:** Warmup and gradient clipping are substitutes, not complements. Pre-clipping, warmup protected the model from explosive gradients in early training. Post-clipping, the warmup ramp wastes the first 1.3 epochs of compute in a 14-epoch budget. Both warmup variants landed in the 111–115 band.
- **Key insight from frieren:** Current cosine is tuned for 50 epochs but only ~14 complete (28% progress). LR never anneals. This is an independent improvement worth pursuing → assigned as PR #875 (schedule-to-budget-cosine).
- **Decision:** CLOSED. Warmup is not useful post-clipping. Follow-up: budget-aligned cosine annealing (PR #875).

---

## 2026-04-28 21:30 — PR #780: Higher MLP ratio: mlp_ratio 2→4 [SENT BACK]

- **Branch:** `charliepai2e2-thorfinn/higher-mlp-ratio`
- **Hypothesis:** Doubling the FFN expansion factor (mlp_ratio 2→4) gives the model more representational capacity per layer at minimal parameter cost (~2.6K extra params). Expected to lower val_avg/mae_surf_p.
- **Outcome:** SENT BACK — val_avg/mae_surf_p = 135.296 at epoch 13/50. Beat old baseline (137.0013) narrowly but not the new clipping baseline (104.7457). Run was without gradient clipping.

### Results Table (best checkpoint, epoch 13)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 172.640 |
| val_geom_camber_rc | 144.293 |
| val_geom_camber_cruise | 99.672 |
| val_re_rand | 124.578 |
| **avg** | **135.296** |

- Model size: 0.99M params (~+2.6K vs mlp_ratio=2)
- Peak VRAM: 52.2 GB
- Per-epoch time: ~148s, runtime 32.09 min
- test_avg/mae_surf_p: NaN (test_geom_camber_cruise scoring bug; 3-split manual avg = 130.75)
- W&B run_id: 496z6hmp

### Analysis

The val curve was monotonically descending (233.6 → 135.3) — no sign of overfitting or instability. The result narrowly beats the old baseline (137.0) but sits well above the new clipping baseline (104.7). This run was on the pre-clipping codebase. mlp_ratio=4 may still be a valid improvement once gradient clipping is in place — retest required. Student provided excellent NaN bug report corroborating fern and nezuko's findings.

**Decision:** Sent back. Rebase onto clipping baseline (merge icml-appendix-charlie-pai2e-r2 which includes clip_grad_norm=1.0), keep mlp_ratio=4, re-run. Target: val_avg/mae_surf_p < 104.7457.

---

## 2026-04-28 21:35 — PR #772: Per-channel output affine: learnable scale+bias [SENT BACK]

- **Branch:** `charliepai2e2-nezuko/per-channel-output-scale`
- **Hypothesis:** Adding learnable per-channel scale and bias to the output head gives the model explicit calibration capacity per physical channel (Ux, Uy, p), potentially helping the pressure channel which operates on different physical scales.
- **Outcome:** SENT BACK — val_avg/mae_surf_p = 138.497 at epoch 12/50. Above both old baseline (137.0013) and new clipping baseline (104.7457). Run was without gradient clipping.

### Results Table (best checkpoint, epoch 12)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 172.870 |
| val_geom_camber_rc | 152.966 |
| val_geom_camber_cruise | 107.674 |
| val_re_rand | 120.478 |
| **avg** | **138.497** |

### Learned per-channel output affine (best checkpoint)

| Channel | out_scale | out_bias |
|---------|----------:|---------:|
| Ux | 1.0203 | -0.0157 |
| Uy | 1.0835 | -0.0293 |
| p  | 1.0636 | -0.0258 |

- Peak VRAM: 42.1 GB
- test_avg/mae_surf_p: 125.331 (finite — student added NaN workaround in train.py filtering non-finite-y samples at evaluate_split)

### Analysis

The learned scales are non-trivial (2-8% above 1.0) — the model does exploit calibration capacity. However, without the clipping baseline, the result can't be fairly evaluated. The scale shift on Uy (1.084) > p (1.064) > Ux (1.020) pattern is interesting — all three shift in the same direction suggesting broad output amplification rather than per-channel decoupling. With stable gradients from clipping, the affine calibration may find more channel-specific corrections. Student also filed a detailed NaN bug report confirming the data/scoring.py root cause.

**Decision:** Sent back. Rebase onto clipping baseline. Keep per-channel affine + NaN workaround in train.py. Target: val_avg/mae_surf_p < 104.7457.

---

## 2026-04-28 21:00 — PR #800: n_hidden=256 + bf16 AMP [SENT BACK]

- **Branch:** `charliepai2e2-alphonse/n-hidden-256-bf16`
- **Hypothesis:** n_hidden=256 with bf16 AMP for 2× speedup, completing the 50-epoch schedule within 30 min.
- **Outcome:** REJECT — bf16 only gave 1.47× speedup (slice-attention pattern not bandwidth-bound), epochs 12/50, val_avg/mae_surf_p = 145.4196. Worse than both old baseline (137.0) AND new baseline (104.7457). test_geom_camber_cruise pressure NaN reappeared.

### Analysis

Capacity expansion alone is not the answer when gradients explode. The student's own analysis correctly identified gradient clipping as the missing ingredient — and indeed PR #778 confirmed clipping alone delivers a 24% improvement on stock arch. Need to retest n_hidden=256 + bf16 + clipping combination on the new baseline.

**Decision:** SENT BACK. Rebase onto clipping baseline, re-run with n_hidden=256 + bf16, target val_avg/mae_surf_p < 104.7457.

---

## 2026-04-28 23:15 — PR #897: Schedule-to-budget cosine T_max=15 [SENT BACK — MARGINAL WIN, CONFLICT]

- **Branch:** `charliepai2e2-alphonse/schedule-to-budget-cosine-tmax15`
- **Hypothesis:** The stock cosine schedule is tuned for T_max=50 but only ~14 epochs complete in the 30-min budget. Setting T_max=15 (aligned to achievable epoch budget) allows the LR to complete its full cosine arc — decaying from 5e-4 → ~0 over the reachable window instead of barely moving. This should improve convergence within the wall-clock constraint.
- **Outcome:** MARGINAL WIN — val_avg/mae_surf_p = **104.4004** vs baseline **104.7457** (Δ = -0.345, -0.33%). Merge was attempted but FAILED due to merge conflicts on `icml-appendix-charlie-pai2e-r2`. Sent back for rebase.

### Results Table (best checkpoint)

| Split | Baseline (PR #778) | T_max=15 (PR #897) | Delta |
|-------|-------------------|-------------------|-------|
| val_single_in_dist | 105.24 | 112.69 | +7.45 ✗ |
| val_geom_camber_rc | 97.21 | 98.24 | +1.03 ✗ |
| val_geom_camber_cruise | 98.39 | 95.21 | -3.18 ✓ |
| val_re_rand | 118.15 | 111.47 | -6.68 ✓ |
| **avg** | **104.7457** | **104.4004** | **-0.345** |

### Mechanism Validation

- T_max=15 means the cosine schedule completes its full arc in ~14 achieved epochs
- LR decays from 5e-4 → 5.46e-6 (full decay to near-zero vs ~4.2e-4 with T_max=50)
- Student confirmed gradient clipping baseline was correctly inherited
- OOD generalization improved: val_re_rand -6.68, val_geom_camber_cruise -3.18
- In-distribution performance regressed slightly: val_single_in_dist +7.45

### Analysis

The mechanism is validated: a budget-aligned cosine schedule does improve mean performance. The marginal aggregate gain (+0.33% improvement) masks a meaningful split pattern — OOD splits benefit from the aggressive final LR decay, while in-dist regression suggests the low final LR (5.46e-6) may be slightly too low, causing underfitting in the late epochs. The optimal T_max is likely between 15 and 50 — student suggested T_max ∈ {20, 25, 30} as natural next experiments. T_max=20–25 would set the final LR at 1e-5–5e-5 rather than 5e-6, potentially retaining the OOD win without the in-dist regression.

**Decision:** Attempted merge → FAILED (conflicts with icml-appendix-charlie-pai2e-r2). Sent back as draft for rebase. Once rebased and re-run confirms the result, should merge (it is a winner). Baseline NOT updated yet — awaiting rebase confirmation.

---

## 2026-04-28 23:20 — PR #898: Less-aggressive gradient clipping (max_norm=5.0) [CLOSED — DEAD END]

- **Branch:** `charliepai2e2-askeladd/less-aggressive-clip-norm`
- **Hypothesis:** max_norm=1.0 may be too aggressive, cutting down genuine gradients alongside explosive ones. Loosening to max_norm=5.0 allows larger but still bounded gradient steps, potentially improving training signal quality without the instability of unclipped gradients.
- **Outcome:** FAIL — val_avg/mae_surf_p = **114.9909** vs baseline **104.7457** (Δ = +10.25, +9.8% worse). Clear dead end.

### Results Table

| Split | Baseline (PR #778, norm=1.0) | max_norm=5.0 (PR #898) | Delta |
|-------|------------------------------|------------------------|-------|
| val_single_in_dist | 105.24 | 105.39 | +0.15 |
| val_geom_camber_rc | 97.21 | 127.01 | +29.80 ✗ |
| val_geom_camber_cruise | 98.39 | 117.89 | +19.50 ✗ |
| val_re_rand | 118.15 | 109.65 | -8.50 ✓ |
| **avg** | **104.7457** | **114.9909** | **+10.25** |

### Training Trajectory

Non-monotonic: val_avg went 143.6 (ep1) → 118.1 (ep4) → 114.0 (ep6) → 116.2 (ep8) → ~115 (plateau). Failed to push through the 110-level floor the tight clip achieves. This pattern is consistent with Adam's internal state (β₁, β₂ EMA) being calibrated to the scale of gradients when max_norm=1.0; loosening the clip changes the effective LR implicitly.

### Analysis

The result confirms that max_norm=1.0 is not over-tightening — it is a structural component of the current recipe, not an arbitrary guard. The Adam optimizer's β₁=0.9, β₂=0.999 effective step sizes are calibrated to gradients of magnitude ≤1.0 per dimension. Pre-clip norms average 43–114 per epoch with maxima 300–900; at max_norm=5.0, steps are still clipped but at 5× the scale, making Adam's effective learning rate 5× larger. This is why val_geom_camber_rc and val_geom_camber_cruise regressed severely (OOD splits are more sensitive to effective LR inflation) while val_re_rand improved marginally (in-dist OOD benefits from slightly larger updates). The fix is not in the clipping threshold — it is in exploring orthogonal improvements that work within the stable gradient regime.

Student suggestions for follow-up: (1) sweep max_norm 1.0–5.0, (2) max_norm=5.0 + lower lr, (3) lower β₂=0.99, (4) no clip + lower lr. These are lower priority than other directions given this result.

**Decision:** CLOSED. max_norm=1.0 is the correct setting. Askeladd now idle — reassigned to T_max sweep experiment.


## 2026-04-28 — PR #974: n_head=4→8 (head_dim=32→16), all else fixed
- Branch: `charliepai2e2-nezuko/n-head-8-attention` (CLOSED)
- Hypothesis: Doubling attention heads (4→8) at fixed n_hidden=128 (head_dim 32→16) is "architecturally free" — same parameter count, more attention diversity, expected to help OOD splits.
- Results (ckpt_avg epochs 9-10-11; only 11 epochs completed in timeout):

| Metric | n_head=8 (#974) | Baseline (#931) | Delta |
|--------|----------------:|----------------:|------:|
| **val_avg/mae_surf_p** | **109.18** | **94.78** | **+14.40 (+15.2% worse)** |
| val_single_in_dist | 129.81 | 104.91 | +24.90 (+23.7%) |
| val_geom_camber_rc | 122.32 | 105.49 | +16.83 (+16.0%) |
| val_geom_camber_cruise | 84.59 | 77.32 | +7.27 (+9.4%) |
| val_re_rand | 100.00 | 91.41 | +8.59 (+9.4%) |
| **test_avg/mae_surf_p** | **97.65** | **85.22** | **+12.43 (+14.6% worse)** |

- Notes: n_head=8 ran ~36% slower per epoch (170s vs 125s), so only 11 epochs fit in the 30-min timeout vs 14 for baseline — cosine tail never executed. Two compounding negatives: head_dim=16 too narrow (uniform regression across all splits) AND truncated schedule.
- Analysis: Clear dead end. head_dim=32 appears near a sweet spot for this problem at n_hidden=128. Orthogonal direction: n_head=2 / head_dim=64 (assigned to edward, PR #1001).
- Status: CLOSED — clear negative result, +15.2% val regression.

## 2026-04-28 — PR #978: T_max=15→20 on cosine schedule, all else fixed
- Branch: `charliepai2e2-askeladd/tmax20-on-compound` (SENT BACK with T_max=18 follow-up)
- Hypothesis: T_max=15 → LR ~5.5e-6 at epoch 14. T_max=20 raises floor to ~1.0e-4 at epoch 14.
- Results (ckpt_avg epochs 12-13-14):

| Metric | T_max=20 (#978) | Baseline T_max=15 (#931) | Delta |
|--------|----------------:|-------------------------:|------:|
| **val_avg/mae_surf_p** | **95.84** | **94.78** | **+1.05 (close-but-no-cigar)** |
| val_single_in_dist | 110.13 | 104.91 | +5.22 |
| val_geom_camber_rc | 106.19 | 105.49 | +0.70 |
| val_geom_camber_cruise | 75.48 | 77.32 | -1.84 (better) |
| val_re_rand | 91.54 | 91.41 | +0.13 (near-neutral) |
| test_avg/mae_surf_p | 85.71 | 85.22 | +0.49 |

- Analysis: ckpt_avg K=3 and in-flight LR motion anti-correlated. At T_max=15, final-3-epoch LR is frozen (~1.6e-5→5.5e-6); at T_max=20 it spans 1.7e-4→1.0e-4 (>10× spread), diluting best single checkpoint.
- Status: SENT BACK — try T_max=18 (midpoint: ~1.5e-5 LR floor, ~3× spread across ckpt_avg window).
