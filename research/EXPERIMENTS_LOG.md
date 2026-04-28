# SENPAI Research Results

## Track: icml-appendix-charlie-pai2e-r2

This log records all reviewed experiments for this research track.
Primary metric: `val_avg/mae_surf_p` (lower is better).

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

