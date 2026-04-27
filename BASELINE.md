# TandemFoilSet Baseline — icml-appendix-charlie-pai2c-r5

## Primary metric: `val_avg/mae_surf_p` (lower is better)

## Current Best

**Source:** Stock Transolver (train.py defaults — no modifications)
**val_avg/mae_surf_p:** ~103 (estimated from sibling track)
**test_avg/mae_surf_p:** ~103 (estimated)

> Note: This is a fresh track — no experiments have run yet.
> Baseline will be updated once PR results come in.

## Stock Transolver Config (starting point)

- n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- Loss: MSE (vol_loss + 10.0 * surf_loss)
- Scheduler: CosineAnnealingLR(T_max=epochs)
- No AMP, no gradient accumulation

## Known Winning Recipe (sibling track kagent_v_students — validated)

All changes stack. Each was validated with multi-seed experiments:

| Change | Val Improvement | Mechanism |
|--------|----------------|-----------|
| L1 loss (sw=1) | ~21% | Matches MAE metric; correct gradient for heavy-tailed pressure |
| AMP + grad_accum=4 | ~2x throughput | More epochs in 30-min budget |
| Fourier PE (m=160, σ=0.7) | ~17% | Better spatial encoding for CFD coordinates |
| SwiGLU FFN | ~6% | Better gradient flow, compute-matched |
| slice_num=16 | ~12% | Faster per-step + regularization compound |
| n_layers=3 | ~19% | Shallower = more epochs; compute-constrained regime |
| n_head=1 | ~10% | Wider single head generalizes better |

**Full recipe best result (triple compound nl=3/sn=8/nh=1):**
- val_avg/mae_surf_p: ~48.1
- test_avg/mae_surf_p: ~40.9

## Per-Split Reference (from kagent PR #32 triple compound probe)

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | ~73 | ~49 |
| geom_camber_rc | ~74 | ~55 |
| geom_camber_cruise | ~36 | ~25 |
| re_rand | ~57 | ~41 |
| **avg** | **~48.1** | **~42.5** |

_Will be updated with actual results from first experiments on this track._
