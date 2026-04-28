# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 113.157** (EMA, epoch 13/50, timeout-cut)
- **`test_avg/mae_surf_p` = 99.322**
- Set by **PR #374** (`charliepai2d1-tanjiro/grad-clip-1p0`), merged 2026-04-28 00:43 UTC.
- Beats all four val splits and all four test splits vs prior baseline.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.999) shadow weights** drive validation, best-checkpoint selection, and final test eval. Companion raw-model val is logged for free-lunch attribution.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation. Required because `data/scoring.py:accumulate_batch` is read-only and IEEE 754 `NaN*0 = NaN` defeats its per-sample mask. Affects `test_geom_camber_cruise` index 20 (`y[:,2]` non-finite).
- **Gradient clipping at `max_norm=1.0`** between `loss.backward()` and `optimizer.step()`. Pre-clip grad norms cluster at 50–100× `max_norm` for the entire run on this base — clip is acting as an effective LR cap, damping outlier steps. Pre-clip mean-per-epoch norm logged as `train/grad_norm`.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #374 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 133.853 | 1.530 | 0.757 | 116.513 | 1.503 | 0.707 |
| geom_camber_rc | 132.712 | 2.686 | 1.041 | 114.145 | 2.519 | 0.965 |
| geom_camber_cruise | 84.919 | 1.074 | 0.524 | 69.410 | 1.029 | 0.474 |
| re_rand | 101.144 | 1.841 | 0.766 | 97.220 | 1.627 | 0.720 |
| **avg** | **113.157** | 1.783 | 0.772 | **99.322** | 1.670 | 0.717 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with EMA(0.999) shadow + NaN-safe pre-pass + grad-clip(1.0) per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep13 of 50 configured.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. New baseline at val=113.157 (−14.45 %), test=99.322 (−15.86 %). Diagnostic: pre-clip grad norms 50–100× `max_norm`, so clip is acting as an effective LR cap.
