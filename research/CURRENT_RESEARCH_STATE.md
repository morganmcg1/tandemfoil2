# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-27
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus
Round 1 covers four orthogonal axes that compound well: loss formulation, loss weighting, architecture capacity, and optimization. The published Transolver in `train.py` is a small model (128 hidden, 5 layers) trained with MSE on normalized targets even though we evaluate on MAE — both gaps are obvious low-cost wins.

## Round 1 hypotheses
| Student | Slug | Axis | Predicted Δ on `val_avg/mae_surf_p` |
|---|---|---|---|
| alphonse | surf-weight-up | Loss weighting (surf_weight 10→25) | -3% to -7% |
| askeladd | huber-loss | Loss formulation (MSE→SmoothL1/Huber) | -5% to -10% |
| edward   | wider-model   | Architecture width (n_hidden 128→192, slice_num 64→96) | -5% to -10% |
| fern     | deeper-model-droppath | Architecture depth (n_layers 5→8 + DropPath 0.1) | -3% to -8% |
| frieren  | warmup-cosine-1e3 | Optimization (linear warmup + peak lr 1e-3) | -2% to -6% |
| nezuko   | ema-grad-clip | Optimization (EMA decay 0.999 + grad clip 1.0) | -3% to -8% |
| tanjiro  | more-slices   | Architecture (slice_num 64→128, n_head 4→8) | -3% to -7% |
| thorfinn | per-channel-surf-weights | Loss weighting (3× surface pressure) | -3% to -8% |

## Potential next research directions (post-round 1)
- Compounding the round-1 winners (e.g. Huber + EMA + warmup).
- Heavy-tail-aware pressure handling: per-sample y-std normalization, log-pressure target, or focal weighting on extreme |p|.
- Fourier / RFF positional encoding on (x, z) to give the model multi-scale spatial frequency info — currently only raw position + signed-arc-length.
- Surface-aware decoder: separate surface-only head with extra capacity, since `mae_surf_p` is what we're scored on.
- Domain-conditional FiLM modulation of attention slices (single vs. raceCar tandem vs. cruise tandem) — the three regimes differ by orders of magnitude in y-std.
- Gradient-aware loss scaling (GradNorm or DWA) across surface and volume to stop one branch dominating.
- Test-time augmentation: average predictions from x↔−x flipped meshes (after re-orienting AoA / stagger).
- Better val-track-aware checkpoint selection: weight `val_geom_camber_*` higher since they're the harder generalization tracks.
