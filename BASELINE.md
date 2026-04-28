# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #314, edward, 2026-04-28)

SmoothL1 (Huber, β=1.0) loss stacked on top of bf16 + FF K=8 +
`torch.compile(dynamic=True)`. Two-line change inside the autocast block;
clean 91% stacking efficiency with the existing compile+FF baseline.

- **`val_avg/mae_surf_p` = 69.8310** at epoch 35 (of 36 completed)
- **`test_avg/mae_surf_p` = 61.7177** (best val checkpoint)
- W&B run: [`fs3tf90w` / `smoothl1-beta1-on-compile-ff`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/fs3tf90w)
- Per-epoch wall: ~49 s steady state (cold compile 32.5 s + 80 s for epoch 1)
- Peak GPU memory: **24.1 GB** / 102.6 GB (~78 GB headroom — compile fuses
  SmoothL1 autograd intermediates so the rounds 1-2 transient memory spike
  is gone entirely)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 36/50 epochs.

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 76.3403 | 0.9221 | 0.4845 |
| val_geom_camber_rc | 81.7820 | 1.3709 | 0.6643 |
| val_geom_camber_cruise | 52.1583 | 0.6955 | 0.3972 |
| val_re_rand | 69.0436 | 1.0396 | 0.5212 |
| **val_avg** | **69.8310** | 1.0070 | 0.5168 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 69.3043 | 0.9284 | 0.4676 |
| test_geom_camber_rc | 71.9660 | 1.3267 | 0.6208 |
| test_geom_camber_cruise | 44.2844 | 0.6889 | 0.3593 |
| test_re_rand | 61.3159 | 0.9273 | 0.4900 |
| **test_avg** | **61.7177** | 0.9678 | 0.4844 |

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| **PR #314 (edward): + SmoothL1 β=1.0** | **69.83** | **−13.6%** |

Cumulative: **−51.6% on val_avg / −53.0% on test_avg** since PR #312.

The four orthogonal mechanisms compose to ~91% of sum-of-individuals
efficiency. The remaining ~9% overlap appears to come from the shared
"more cosine schedule reaches the model" mechanism (compile gives 2×
epochs; FF speeds per-epoch convergence; Huber bounds gradient outliers
that previously dominated MSE; bf16 enables the rest).

## Default config (matches PR #314)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- **Loss**: SmoothL1 / Huber (β=1.0) per-element loss in normalized space,
  with surface vs. volume split via `surf_weight`. Inside
  `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Fourier features (K=8)** for normalized (x, z), computed in fp32
  outside the autocast scope, concatenated to the per-node feature vector.
  Per-node feature dim: 24 → 56.
- **`torch.compile(model, dynamic=True)`** wrapper applied right after
  `model.to(device)` (gated on `not cfg.debug`). Save/load via
  `getattr(model, "_orig_mod", model).state_dict()` so the W&B model
  artifact is portable into a non-compiled module.
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22 + 4*8 = 54`, `out_dim=3`).

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-huber-compile-ff-bf16 \
  --wandb_name baseline-huber-compile-ff-bf16
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap is **still binding** at 36/50 epochs. Cosine T_max
  alignment (in flight as fern PR #407, --epochs 37 confirmed) may release
  another small win from the schedule tail.
- VRAM headroom is now 78 GB (24.1 / 102.6). The previous ban on batch-
  size scaling (PR #360 ruled it out without compile) deserves
  re-investigation under compile, since memory math has fundamentally
  changed and the trainer may be in a different regime now.
- `data/scoring.py` patched (`b78f404`).
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.

## Prior baselines (superseded)

- **PR #312** (alphonse, original): val_avg=144.21, test_avg=131.18.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15.
- **PR #327** (tanjiro, FF K=8): val_avg=106.92, test_avg=96.82.
- **PR #416** (alphonse, compile+FF): val_avg=80.85, test_avg=73.41.
