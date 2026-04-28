# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 63.218** (EMA, epoch 14/50, timeout-cut)
- **`test_avg/mae_surf_p` = 55.398**
- Set by **PR #491** (`charliepai2d1-fern/tf32-matmul-precision`), merged 2026-04-28 05:17 UTC.
- Beats prior baseline (#352) by −1.47 % val / −0.95 % test (modest gain, throughput multiplier is the durable win).

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count).
- **Optimizer: Lion** (sign-of-momentum) `lr=1.7e-4`, `weight_decay=3e-4`, `betas=(0.9, 0.99)`.
- **Loss**: `MSE_vol + 10.0 * SmoothL1_surf` (β=1.0 in normalized space).
- Schedule: cosine annealing over `epochs` (T_max=50)
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **TF32 matmul precision**: `torch.set_float32_matmul_precision('high')` enables Blackwell tensor-core fp32 matmul. Throughput: 131 s/epoch steady-state (−13 % vs eager), **14 epochs in 30-min budget**.
- **EMA(0.99)** shadow weights drive validation, best-checkpoint selection, and final test eval.
- **`evaluate_split` NaN-safe pre-pass**: drops samples with non-finite ground truth.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()` — under Lion this only smooths the momentum buffer (sign-update is invariant to gradient magnitude); kept for lineage.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #491 best-EMA-epoch checkpoint)

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p | test mae_surf_Ux | test mae_surf_Uy |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 71.414 | 0.736 | 0.420 | 62.626 | 0.731 | 0.388 |
| geom_camber_rc | 77.627 | 1.184 | 0.567 | 70.013 | 1.135 | 0.530 |
| geom_camber_cruise | 42.063 | 0.531 | 0.293 | 34.847 | 0.477 | 0.270 |
| re_rand | 61.766 | 0.853 | 0.428 | 54.106 | 0.797 | 0.422 |
| **avg** | **63.218** | 0.826 | 0.427 | **55.398** | 0.785 | 0.403 |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with TF32 + Lion + SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + lr=1.7e-4 + SmoothL1(β=1.0)/MSE-vol per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep14 of 50 configured. Wall clock ~131 s/epoch.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) + NaN-safe pre-pass. val=132.276 / test=118.041.
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): `clip_grad_norm_(1.0)`. val=113.157 (−14.45 %).
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %).
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %).
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %).
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): GELU MLP → SwiGLU at matched params. val=89.349 (−9.36 %).
- 2026-04-28 03:46 — **PR #430** (tanjiro/lion-optimizer): AdamW → Lion. val=67.737 (−24.19 %). Biggest single-PR delta.
- 2026-04-28 04:33 — **PR #352** (edward/smoothl1-surface): SmoothL1(β=1.0)/MSE-vol. val=64.158 (−5.28 %).
- 2026-04-28 05:17 — **PR #491** (fern/tf32-matmul-precision): `torch.set_float32_matmul_precision('high')`. val=**63.218** (−1.47 %) / test=**55.398** (−0.95 %). Permanent throughput multiplier (−13 % per-epoch, 14 epochs in budget vs 12); metric Δ at noise floor.
