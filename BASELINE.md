# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 107.957** (EMA, epoch 13/50, timeout-cut)
- **`test_avg/mae_surf_p` = 95.675**
- Set by **PR #408** (`charliepai2d1-fern/higher-lr-1e3`), merged 2026-04-28 01:41 UTC.
- Fern's measured run was on `lr=1e-3 + max_norm=1.0` (the post-#374 base). Squash-merge composed `lr=1e-3` (from #408) with `max_norm=0.5` (from #402) → current baseline `train.py` is `lr=1e-3 + max_norm=0.5 + EMA(0.999) + NaN-safe pre-pass`. Future re-runs of the unmodified baseline may land slightly different from 107.957 / 95.675 due to that compose; the recorded numbers are the comparison floor.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **EMA(0.999) shadow weights** drive validation, best-checkpoint selection, and final test eval. Companion raw-model val is logged for free-lunch attribution.
- **`evaluate_split` NaN-safe pre-pass:** drops samples with non-finite ground truth from `mask` and zeros their `y` before loss/MAE computation. Required because `data/scoring.py:accumulate_batch` is read-only and IEEE 754 `NaN*0 = NaN` defeats its per-sample mask. Affects `test_geom_camber_cruise` index 20 (`y[:,2]` non-finite).
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()`. Pre-clip mean grad norm ~44 across training at `lr=1e-3` (vs ~73 at lr=5e-4) — AdamW preconditioner adapts to higher LR by inflating per-step magnitude internally, so raw grads land smaller. Clip is still firing aggressively (~60–110× over `max_norm`).

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #408 best-EMA-epoch checkpoint, lr=1e-3 + max_norm=1.0)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 125.675 | 1.661 | 0.719 | 109.532 | 1.556 | 0.714 |
| geom_camber_rc | 122.662 | 2.609 | 1.008 | 108.057 | 2.480 | 0.929 |
| geom_camber_cruise | 82.973 | 1.052 | 0.513 | 69.100 | 1.041 | 0.465 |
| re_rand | 100.517 | 1.750 | 0.754 | 96.011 | 1.549 | 0.708 |
| **avg** | **107.957** | 1.768 | 0.748 | **95.675** | 1.656 | 0.704 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with EMA(0.999) + NaN-safe pre-pass + grad-clip(0.5) + lr=1e-3 per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep13 of 50 configured.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) shadow + NaN-safe pre-pass. First measured baseline (val=132.276, test=118.041).
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): added `clip_grad_norm_(model.parameters(), max_norm=1.0)`. Baseline at val=113.157 (−14.45 % vs #356), test=99.322 (−15.86 %).
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): tightened `max_norm=1.0 → 0.5`. Baseline at val=110.822 (−2.07 % vs #374), test=97.955 (−1.38 %). Diminishing-returns curve on clipping lever now mapped.
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): bumped `lr=5e-4 → 1e-3`. New baseline at val=107.957 (−2.59 % vs #402), test=95.675 (−2.33 %). Confirmed mechanism: AdamW preconditioner adapts to higher LR by inflating per-step magnitude; clip envelope at `max_norm` controls effective step magnitude. "Higher LR safe under clip" hypothesis confirmed.
