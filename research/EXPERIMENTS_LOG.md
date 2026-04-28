# SENPAI Research Results

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
