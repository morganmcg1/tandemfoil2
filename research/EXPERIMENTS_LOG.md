# SENPAI Research Results

## 2026-04-29 — PR #886: Pressure-channel loss weighting (p_weight sweep) on Huber baseline

- Branch: `charliepai2e1-alphonse/pressure-channel-weight`
- Hypothesis: Upweighting pressure channel (index 2) in Huber loss would align training with the `val_avg/mae_surf_p` objective.

### Results

| Run | `val_avg/mae_surf_p` | `val_avg/mae_surf_Ux` | `val_avg/mae_surf_Uy` | Delta vs current baseline |
|-----|----------------------|-----------------------|-----------------------|--------------------------|
| Baseline (PR #882) | **103.2182** | — | — | — |
| `p_weight=2.0` | 113.0719 | 1.9448 | 0.8346 | +9.54% |
| `p_weight=3.0` | 109.8322 | 2.3026 | 0.8398 | +6.40% |
| `p_weight=5.0` | 125.6185 | 2.7838 | 0.9450 | +21.73% |

Per-split surface pressure MAE (best epoch=14 for all):

| Run | `val_single_in_dist` | `val_geom_camber_rc` | `val_geom_camber_cruise` | `val_re_rand` |
|-----|----------------------|----------------------|--------------------------|---------------|
| `p_weight=2.0` | 145.84 | 114.46 | 89.28 | 102.71 |
| `p_weight=3.0` | 138.21 | 120.66 | 82.12 | 98.34 |
| `p_weight=5.0` | 154.96 | 135.42 | 99.45 | 112.65 |

- Metrics: `metrics/charliepai2e1-alphonse-pweight-2.0-k5sydadm.jsonl`, `metrics/charliepai2e1-alphonse-pweight-3.0-yrd9rvw7.jsonl`, `metrics/charliepai2e1-alphonse-pweight-5.0-1gn6crxv.jsonl`
- Config base: `--loss huber --huber_delta 1.0 --surf_weight 30 --n_hidden 128 --n_head 4`
- All runs: 14 epochs (30-min timeout), ~131 s/epoch

### Analysis and Conclusions

**Decision: Closed — clear dead-end.** All three p_weight values are substantially worse than the current baseline (103.2182 from PR #882). The PR was originally written against the older PR #827 baseline (109.5716), and even against that weaker target, only p_weight=3.0 came close (+0.24%). Against the actual current best, the best run (p_weight=3.0) is still +6.4% worse.

Student's analysis is accurate and insightful:
1. Targets are normalized pre-loss, so channels already compete on comparable scales.
2. `surf_weight=30` already routes 30x gradient toward surface nodes (where pressure is evaluated). Stacking channel reweighting on top is redundant and harmful.
3. High p_weight (5.0) effectively rescales the Huber loss regime — the delta=1.0 threshold is calibrated for unit-weight loss, and multiplying by 5 pushes gradients into the linear tail, impairing convergence.

**Key takeaway:** The model is already pressure-focused at the node level via `surf_weight`. Channel-level reweighting is the wrong lever. Gains must come from representation quality (model capacity, architecture), not gradient routing.

**Follow-up note:** Student offered to take a bug-fix PR for `test_geom_camber_cruise` NaN — this was already resolved in PR #792's infrastructure fix (grad_clip + eval sanitization). New assignments build on that baseline.

---

## 2026-04-28 20:15 — PR #792: Deeper Transolver: n_layers 5→8, lr 5e-4→3e-4

- Branch: `charliepai2e1-frieren/more-layers`
- Hypothesis: Increasing n_layers from 5 to 8 deepens the model's ability to compose multi-scale physics features, benefiting pressure field prediction across boundary layer, wake, and far-field regimes.

### Results

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 189.77 | 2.61 | 0.99 |
| val_geom_camber_rc | 170.02 | 4.01 | 1.20 |
| val_geom_camber_cruise | 109.36 | 2.42 | 0.66 |
| val_re_rand | 130.93 | 2.90 | 0.90 |
| **val_avg** | **150.02** | 2.99 | 0.94 |
| test_single_in_dist | 168.47 | 2.59 | 0.96 |
| test_geom_camber_rc | 156.64 | 3.73 | 1.14 |
| test_geom_camber_cruise | **NaN** | 2.21 | 0.59 |
| test_re_rand | 129.61 | 2.79 | 0.90 |
| **test_avg** | **NaN** | 2.83 | 0.90 |

- Metrics path: `metrics/charliepai2e1-frieren-e0tvgog3/metrics_summary.json`
- Config: n_layers=8, lr=3e-4, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2, surf_weight=10, MSE loss
- Epochs completed: 9 / 50 (timeout-bounded; ~2.4x slower per epoch than n_layers=5)
- VRAM: 64.5 GB / 96 GB
- Wall clock: 30.9 min

### Analysis and Conclusions

**Decision: Request changes — not mergeable as-is.**

Three critical problems:
1. **NaN primary metric**: `test_avg/mae_surf_p = NaN` due to `vol_loss = Infinity` on `test_geom_camber_cruise`. The deeper model produces non-finite predictions on at least one high-Re cruise test sample. Since `Ux` and `Uy` are finite on the same samples, this is a p-channel numerical instability — likely extreme pressure magnitude (+/-29K range in cruise) driving attention softmax to saturation with 8 layers.
2. **No baseline**: No n_layers=5 baseline has been run on this repo, so val_avg/mae_surf_p=150.02 is an absolute reading, not a delta. Cannot declare win or loss.
3. **Severely under-trained**: Only 9/50 epochs. The val curve was still falling steeply (162.9→150.0 in the last two epochs) so this is far from convergence. Equal-epoch comparison needed.

**Key insight from trajectory**: Val loss was falling sharply at cutoff; the model shows strong learning signal but the depth-induced slowdown makes it impractical within a 30-min budget. The NaN bug is likely reproducible in other deeper variants.

**Recommended fix**: Add gradient clipping + NaN guard in test eval accumulation, then re-run at n_layers=6 (middle ground) to stay within budget.

---

## 2026-04-28 21:00 — PR #791: Wider Transolver (n_hidden=256, n_head=8, fp32)

- Branch: `charliepai2e1-fern/wider-model`
- Hypothesis: Wider hidden dimension (128→256) and more attention heads (4→8) increases the model's capacity to represent complex pressure field patterns. Infrastructure improvements: NaN guard in `evaluate_split`, JSONL metrics logger, CLI args for `--n_hidden`/`--n_head`, `WANDB_MODE=disabled` default.

### Results

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 182.51 | 2.88 | 1.05 |
| val_geom_camber_rc | 173.18 | 4.23 | 1.28 |
| val_geom_camber_cruise | 112.50 | 2.60 | 0.70 |
| val_re_rand | 152.63 | 3.14 | 0.97 |
| **val_avg** | **155.96** | 3.21 | 1.00 |

- Config: n_hidden=256, n_head=8, n_layers=5, slice_num=64, lr=5e-4, MSE loss, fp32
- Epochs completed: 7 / 50 (timeout-bounded; 260s/epoch, epoch-starved)
- Wall clock: 30 min (timeout)

### Analysis and Conclusions

**Decision: CLOSED — superseded by PR #808 (bf16 follow-up).**

PR #791 itself did not beat the Huber baseline (155.96 vs 115.65) due to severe epoch starvation (only 7/50 epochs in fp32). However, the infrastructure improvements (NaN guard, JSONL logger, CLI args, WANDB_MODE default) are important fixes retained in the codebase. PR #808 (bf16 mixed precision) was opened as the direct follow-up to address the epoch starvation issue — it achieves ~192s/epoch vs 260s for fp32, enabling ~10 epochs within the 30-min budget. PR #791 was closed as superseded once #808 was confirmed running.

---

## 2026-04-28 21:30 — PR #808: Wider Transolver (n_hidden=256, n_head=8) + bf16 Mixed Precision

- Branch: `charliepai2e1-fern/wider-model-bf16`
- Hypothesis: bf16 mixed precision reduces epoch time ~32% (192s vs 260s) for the wider model (n_hidden=256, n_head=8), enabling ~2× more epochs within the 30-min budget compared to the fp32 wider model (PR #791). Combined with the capacity increase from the wider hidden dim, this should beat the Huber baseline.

### Results

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 151.43 | 2.74 | 1.01 |
| val_geom_camber_rc | 143.27 | 3.82 | 1.16 |
| val_geom_camber_cruise | 99.18 | 2.48 | 0.67 |
| val_re_rand | 120.49 | 2.97 | 0.93 |
| **val_avg** | **128.59** | 3.00 | 0.94 |

- Config: n_hidden=256, n_head=8, n_layers=5, slice_num=64, lr=5e-4, MSE loss (no Huber), bf16
- Epochs completed: 10 / 50 (timeout-bounded; 192s/epoch)
- VRAM: ~58 GB / 96 GB

### Analysis and Conclusions

**Decision: Request changes — 3 blocking issues prevent merge.**

1. **Missing Huber loss**: Run command uses default MSE loss; `--loss huber --huber_delta 1.0` was not included. Since PR #788 established Huber loss as the foundation (115.65 vs 126.88 MSE), this result (128.59) is not a fair comparison. With Huber on top of bf16+wider, we can reasonably project breaking the 115.65 baseline.
2. **Unauthorized split decoder architecture**: Fern added a separate pressure decoder head that was not part of the approved hypothesis. This architectural change must be reverted to isolate the bf16+wider-width effect.
3. **T_max mismatch**: `CosineAnnealingLR(T_max=50)` but timeout limits to ~12 epochs; LR never anneals. Should set `--epochs 12` to match actual budget.

**Revised run command sent back**: `cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12`

---

## 2026-04-28 21:45 — PR #795: Per-Sample Normalization (std-based, Huber)

- Branch: `charliepai2e1-thorfinn/per-sample-norm`
- Hypothesis: Normalizing loss per sample by the per-sample pressure standard deviation equalizes gradient contributions across the 15× Re-range variance spread (per-sample std range: 0.32–4.85 Pa), preventing high-Re samples from dominating training and allowing the model to learn low-Re boundary layer features more effectively.

### Results

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 141.22 | 2.65 | 0.98 |
| val_geom_camber_rc | 128.47 | 3.77 | 1.14 |
| val_geom_camber_cruise | 96.30 | 2.38 | 0.63 |
| val_re_rand | 115.51 | 2.82 | 0.87 |
| **val_avg** | **120.37** | 2.91 | 0.91 |

- Metrics path: `metrics/charliepai2e1-thorfinn-per-sample-norm/metrics_summary.json`
- Config: n_layers=5, n_hidden=128, n_head=4, slice_num=64, lr=5e-4, MSE loss + per-sample std normalization
- Epochs completed: 11 / 50 (timeout-bounded)
- MSE baseline: 136.08 → 120.37 (-11.5% improvement over MSE)

### Analysis and Conclusions

**Decision: Request changes — above Huber baseline, re-run with Huber+norm combined.**

Per-sample normalization alone achieves a 11.5% improvement over MSE baseline (120.37 vs 136.08), validating the hypothesis that high-Re samples dominate training. However, 120.37 does not beat the current Huber baseline (115.65). The per-sample normalization was applied on top of MSE rather than Huber loss. The combination of Huber loss (which reduces outlier sensitivity) plus per-sample normalization (which equalizes Re-range gradient contributions) is a natural stack that has not yet been tested.

**Key insight**: Per-sample std range of 0.32–4.85 confirms the 15× variance spread. The normalization approach is sound — it just needs to build on the Huber foundation.

**Recommended fix**: Stack `--loss huber --huber_delta 1.0` on top of the per-sample normalization and re-run.

---

## 2026-04-28 22:00 — PR #794: LR Warmup (5 epochs) + Cosine Annealing

- Branch: `charliepai2e1-tanjiro/lr-warmup`
- Hypothesis: 5-epoch linear LR warmup before cosine annealing avoids early gradient instability, particularly during the first few iterations when the attention modules have random weights and can produce large gradient norms. Combined with CosineAnnealingLR, this should stabilize training and improve final convergence.

### Results

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|------------|-------------|-------------|
| val_single_in_dist | 158.34 | 2.78 | 1.02 |
| val_geom_camber_rc | 144.19 | 3.91 | 1.18 |
| val_geom_camber_cruise | 107.22 | 2.44 | 0.65 |
| val_re_rand | 135.19 | 2.94 | 0.91 |
| **val_avg** | **136.25** | 3.02 | 0.94 |

- Config: n_layers=5, n_hidden=128, n_head=4, slice_num=64, lr=5e-4, MSE loss, 5-epoch linear warmup + CosineAnnealingLR(T_max=45)
- Epochs completed: 12 / 50 (timeout-bounded; ~150s/epoch)
- MSE baseline: 143.22 → 136.25 (-4.87% improvement over MSE)

### Analysis and Conclusions

**Decision: Request changes — above Huber baseline, re-run with 2-epoch warmup + Huber.**

Two compounding problems:
1. **Warmup too long**: 5-epoch warmup consumes ~37% of the available ~14-epoch budget (timeout-bounded). The model spent 5 epochs at reduced LR before meaningful learning began. A 2-epoch warmup would provide gradient stabilization while preserving more epochs at full LR.
2. **No Huber loss**: MSE foundation means starting at 143.22 vs Huber's 115.65. The improvement is real (-4.87%) but the wrong starting point.

**Key insight**: The improvement over MSE baseline confirms warmup helps stabilize early training. The combination of Huber + short warmup should compound these gains.

**Recommended fix**: Reduce `--warmup_epochs 5` to `--warmup_epochs 2`, add `--loss huber --huber_delta 1.0`, set `T_max = epochs - 2 = 48` in CosineAnnealingLR.
