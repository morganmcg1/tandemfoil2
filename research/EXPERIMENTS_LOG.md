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

