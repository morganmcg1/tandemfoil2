# SENPAI Research State

- **Date:** 2026-04-27
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file; treat this as an open research track.

## Current research focus

Establish the empirical headroom on TandemFoilSet by attacking the unmodified `train.py` baseline along several orthogonal axes in parallel. The eval metric is `val_avg/mae_surf_p` — surface pressure MAE averaged across the four validation tracks (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). The paper-facing rank is `test_avg/mae_surf_p` from the best-val checkpoint.

Round 1 deliberately uses single-axis changes so the contributions of each lever can be measured cleanly. Winners merge sequentially; subsequent rounds will stack and explore second-order interactions.

## Round 1 — single-axis sweep (8 hypotheses)

| Axis | Student | Hypothesis |
|------|---------|------------|
| Loss formulation | edward | L1 loss in place of MSE — directly aligned with the MAE evaluation metric |
| Loss formulation | alphonse | Per-channel pressure-up-weighting inside the surface loss |
| Loss formulation | nezuko | `surf_weight` 10 → 30 — push the optimizer harder onto surface fidelity |
| Architecture | askeladd | Wider model: `n_hidden` 128 → 192, `slice_num` 64 → 96 |
| Architecture | frieren | Deeper model: `n_layers` 5 → 8 |
| Architecture | thorfinn | Finer attention: `slice_num` 64 → 128, `n_head` 4 → 8 |
| Optimization | fern | Linear warmup (5 ep) then cosine, peak `lr` 5e-4 → 1e-3 |
| Optimization | tanjiro | EMA weights (decay 0.999) for eval and test |

## Potential next research directions

- **Stack winners.** Best loss × best architecture × best optimizer.
- **Pressure-conditioned heads.** Predict velocity and pressure with separate output heads or distinct loss balancing per channel.
- **Geometry-aware features.** Stronger use of the signed-distance / arc-length descriptors (dims 2–11) that are already present.
- **Frequency / Fourier features** on node position to recover finer pressure gradients near the surface.
- **Domain conditioning.** Explicit token or FiLM conditioning on the (raceCar single | raceCar tandem | cruise tandem) domain — currently inferred from features alone.
- **Larger effective batch.** Gradient accumulation trades wall-clock per step for stability across the 3 domains.
- **Mesh-aware augmentation.** Random subsampling of mesh nodes during training as a regularizer (the model is permutation-invariant by design).
- **Improved checkpointing.** SWA at the tail of training; checkpoint averaging.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Experiment metrics are committed as JSONL under `models/<experiment>/metrics.jsonl` and pulled back into `/research/EXPERIMENT_METRICS.jsonl` on review.
- No W&B / external loggers — local JSONL only.
