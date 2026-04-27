# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-27 23:35
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus
Round 1 covers four orthogonal axes that compound well: loss formulation, loss weighting, architecture capacity, and optimization. The published Transolver in `train.py` is a small model (128 hidden, 5 layers) trained with MSE on normalized targets even though we evaluate on MAE — both gaps are obvious low-cost wins.

**Resolved infrastructure issue:** PR #358 (edward) merged 2026-04-27 — `data/scoring.py` now uses `torch.where` masking instead of float-mask multiplication, so `inf * 0 = NaN` no longer poisons the float64 accumulator. New `data/test_scoring.py` has 4 surgical regression tests. Existing in-flight PRs branched **before** the merge will still produce NaN on `test_geom_camber_cruise/mae_surf_p`; we'll rebase or cherry-pick on a per-PR basis at review time. Future assignments branch from the post-fix advisor branch.

## Round 1 hypotheses
| Student | PR | Slug | Axis | Predicted Δ | Status |
|---|---|---|---|---|---|
| alphonse | #287 | surf-weight-up | Loss weighting (surf_weight 10→25) | -3% to -7% | WIP |
| askeladd | #289 | huber-loss | Loss formulation (MSE→SmoothL1/Huber) | -5% to -10% | WIP |
| edward   | #300 | wider-model | Architecture width (192/96) | -5% to -10% | **CLOSED** — under-trained 9/50 epochs at 30-min cap |
| edward   | #358 | fix-scoring-nan-mask | Maintenance fix to data/scoring.py | n/a (unblocks test_avg) | **MERGED** 010235e |
| edward   | #368 | fourier-pos-encoding | Architecture/input (8-freq Fourier features on (x,z), fun_dim 22→54) | -3% to -8% | WIP |
| fern     | #304 | deeper-model-droppath | Architecture depth (n_layers 5→8 + DropPath 0.1) | -3% to -8% | WIP |
| frieren  | #307 | warmup-cosine-1e3 | Optimization (linear warmup + peak lr 1e-3) | -2% to -6% | WIP |
| nezuko   | #308 | ema-grad-clip | Optimization (EMA decay 0.999 + grad clip 1.0) | -3% to -8% | WIP |
| tanjiro  | #309 | more-slices | Architecture (slice_num 64→128, n_head 4→8) | -3% to -7% | WIP |
| thorfinn | #310 | per-channel-surf-weights | Loss weighting (3× surface pressure) | -3% to -8% | WIP |

## Lessons from edward's #300
- The 30-min cap is binding — at ~205 s/epoch a 1.5 M-param wider model only fits ~9 of 50 epochs. Architecture-scaling experiments must verify epochs/budget BEFORE we widen further.
- 96 GB peak ~63 GB → memory is not the binding constraint, time is. Good news: a future widening can spend memory on a smaller batch size to recover speed (more samples per second via larger batch is the wrong axis when the bottleneck is per-step compute).
- The `inf * 0 = NaN` mask trap in `data/scoring.py` was sample-data driven (inf in p of one cruise-test sample); fix in #358 will unblock test rankings.

## Potential next research directions (post-round 1)
- Compounding the round-1 winners (e.g. Huber + EMA + warmup) once we have a comparable baseline number.
- Conservative widening that fits the 30-min budget: e.g. n_hidden=144, slice_num=80 — under 1.3 M params, should run 50 epochs in under 30 min.
- Heavy-tail-aware pressure handling: per-sample y-std normalization, log-pressure target, or focal weighting on extreme |p|.
- Fourier / RFF positional encoding on (x, z) to give the model multi-scale spatial frequency info — currently only raw position + signed-arc-length.
- Surface-aware decoder: separate surface-only head with extra capacity, since `mae_surf_p` is what we're scored on.
- Domain-conditional FiLM modulation of attention slices (single vs. raceCar tandem vs. cruise tandem) — the three regimes differ by orders of magnitude in y-std.
- Gradient-aware loss scaling (GradNorm or DWA) across surface and volume to stop one branch dominating.
- Test-time augmentation: average predictions from x↔−x flipped meshes (after re-orienting AoA / stagger).
- Better val-track-aware checkpoint selection: weight `val_geom_camber_*` higher since they're the harder generalization tracks.
