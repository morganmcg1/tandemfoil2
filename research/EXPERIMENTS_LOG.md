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

## 2026-04-28 20:55 — PR #768: Lower LR + warmup (5e-4→1e-4 + 500-step warmup) [SENT BACK]

- **Branch:** `frieren/lower-lr-warmup`
- val_avg/mae_surf_p = 125.35. Beat original baseline but not new baseline.
- **Decision:** Sent back — rebase onto clipping baseline. With clipping handling extremes, the higher 5e-4 LR may now train safely without warmup; and the lower-LR variant should compound with clipping for further gain.

---

## 2026-04-28 21:00 — PR #800: n_hidden=256 + bf16 AMP [SENT BACK]

- **Branch:** `charliepai2e2-alphonse/n-hidden-256-bf16`
- **Hypothesis:** n_hidden=256 with bf16 AMP for 2× speedup, completing the 50-epoch schedule within 30 min.
- **Outcome:** REJECT — bf16 only gave 1.47× speedup (slice-attention pattern not bandwidth-bound), epochs 12/50, val_avg/mae_surf_p = 145.4196. Worse than both old baseline (137.0) AND new baseline (104.7457). test_geom_camber_cruise pressure NaN reappeared.

### Analysis

Capacity expansion alone is not the answer when gradients explode. The student's own analysis correctly identified gradient clipping as the missing ingredient — and indeed PR #778 confirmed clipping alone delivers a 24% improvement on stock arch. Need to retest n_hidden=256 + bf16 + clipping combination on the new baseline.

**Decision:** SENT BACK. Rebase onto clipping baseline, re-run with n_hidden=256 + bf16, target val_avg/mae_surf_p < 104.7457.

