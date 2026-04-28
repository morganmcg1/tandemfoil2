# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-28 (after PR #576 merged)

## Current best — Round 0, PR #576 (nezuko H16 arcsinh-on-pressure × bf16+compile × FiLM × EMA, Run E)

| Metric | Value | Δ vs prior baseline (PR #343) |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` (active=ema) | **59.31** | **−26.7%** |
| `val_avg_raw/mae_surf_p` | 59.55 | −26.4% |
| **`test_avg/mae_surf_p`** | **52.98** | **−27.2%** |
| `test/test_single_in_dist/mae_surf_p` | 66.72 | −15.2% |
| `test/test_geom_camber_rc/mae_surf_p` | 66.58 | −21.7% |
| `test/test_geom_camber_cruise/mae_surf_p` | 29.68 | **−44.8%** |
| `test/test_re_rand/mae_surf_p` | 48.95 | −33.4% |
| `test_avg/mae_surf_Ux` | 0.87 | improvement |
| `test_avg/mae_surf_Uy` | 0.44 | improvement |
| `test_avg/mae_vol_p` | 60.99 | improvement |
| Throughput | 54.9 s/epoch | within 2% of PR #343 (arcsinh ~free) |
| Peak GPU memory | 23.85 GB | unchanged |

- **W&B run:** [`pl7c6y23`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r4/runs/pl7c6y23) (`willowpai2d4-nezuko/h16-arcsinh-500-on-merged`)
- **Best epoch:** 33 of 33 actually trained (run hit 30-min wall clock)
- **Active source at best epoch:** EMA by 0.24 pts (essentially tied with raw at convergence)

## Cumulative round-0 progress vs original baseline

| | val_avg/mae_surf_p | test_avg/mae_surf_p |
|--|--|--|
| Original (vanilla Transolver, pre-#344 baseline) | ~125-130 | ~113-119 |
| PR #344 (warmup+cosine+NaN fix) | 120.97 | 109.92 |
| PR #404 (FiLM-on-Re + wd=5e-4) | 119.36 | 107.54 |
| PR #442 (EMA decay=0.99) | 109.19 | 98.47 |
| PR #343 (bf16+compile) | 80.91 | 72.73 |
| **PR #576 (arcsinh-on-pressure)** | **59.31** | **52.98** |

**Cumulative improvement: ~52% on val, ~56% on test over five merges in round 0.** PR #576's super-additive compound (val −26.7% on top of #343's −25.7%) is the round's largest single-PR effect *as measured against the prior baseline*; the throughput merge made the arcsinh mechanism much more valuable than it was on the post-#404 path.

## What changed from prior baseline (PR #343)

The merged code on `icml-appendix-willow-pai2d-r4` now includes:

1. **arcsinh-compressed pressure target.** During training, the pressure channel of `y` is transformed via `y_p_t = arcsinh(y_p / scale)` with `scale=500.0` (default). Stats `(y_mean[2], y_std[2])` are recomputed at startup over a 200-sample subset of `train_ds` to match the transformed-target distribution. Loss is then MSE in the transformed-then-normalized space. At evaluation, predictions are denormalized and inverse-transformed via `p_phys = scale * sinh(p_t)` before MAE accumulation.
2. **`--arcsinh_p_scale` CLI flag** (default 500.0; 0 = disabled). Standalone flag that doesn't affect any other behavior.
3. **Stats recomputation respects the active scale.** When `arcsinh_p_scale > 0`, the pressure mean/std are recomputed before any `evaluate_split` call (raw eval, EMA eval, end-of-run test eval). When 0, the original stats are used unchanged.
4. **Cumulative from PR #343 + #442 + #404 + #344:** bf16 autocast, torch.compile (mode="default", dynamic=True), EMA decay=0.99 + every-other-epoch eval, Re-conditional FiLM modulation, linear warmup + per-step cosine-to-zero schedule, defensive `nan_to_num` in `evaluate_split`, `--seed` CLI flag.

## Recommended training command (reproduces current best)

```bash
cd target/ && python train.py \
    --agent <student-name> \
    --batch_size 4 \
    --amp_dtype bf16 \
    --compile True \
    --film_re True \
    --use_ema True --ema_decay 0.99 --ema_eval_every 2 \
    --arcsinh_p_scale 500.0 \
    --epochs 37 \
    --lr 7e-4 \
    --weight_decay 5e-4 \
    --seed 123 \
    --wandb_name "<student-name>/<experiment-tag>"
```

**The four mechanisms compound super-additively** (per PR #576's analysis): bf16+compile + arcsinh especially reinforce because the compressed loss landscape rewards more gradient steps. Future hypothesis comparisons should layer their changes on top of `--arcsinh_p_scale 500.0` and the rest of the merged config.

## EMA's marginal value at full convergence is now ~0

Per PR #343 and PR #576 measurements:
- PR #343 Run G: EMA-vs-raw gap 16 pts at epoch 1 → 0.13 pts at epoch 33; raw was selected as active.
- PR #576 Run E: EMA-vs-raw gap 24.8 pts at epoch 1 → 0.24 pts at epoch 33; EMA was selected by 0.24 pts.

EMA's noise-smoothing effect approaches zero as the model converges. We keep `--use_ema True` as the default because:
- It's a defensive measure for any future PR that introduces optimization noise.
- The active-checkpoint logic auto-selects raw or EMA per epoch.
- The dual-eval cost is amortized via `--ema_eval_every 2`.

But don't expect EMA to add headline numbers when the underlying training is already converged.

## Setup recap

| Setting | Value |
|---------|-------|
| Model | Transolver + Re-conditional FiLM (~0.75M params) |
| Optimizer | AdamW, weight_decay=5e-4 |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10, **pressure target arcsinh-compressed at scale=500** |
| Schedule | Linear warmup (5%) + cosine-to-zero, per-step (`LambdaLR`) |
| Epochs (recommended) | 37 (cosine reaches lr=0 at our 30-min wall clock) |
| Forward dtype | bf16 (`--amp_dtype bf16`) |
| torch.compile | `mode="default", dynamic=True` (`--compile True`) |
| EMA decay | 0.99 (half-life ~0.2 epoch) |
| EMA eval frequency | every 2 epochs (`--ema_eval_every 2`) |
| Recommended `--lr` | 7e-4 |
| Recommended `--seed` | 123 |
| Recommended `--arcsinh_p_scale` | 500.0 |
| Primary metric | `val_avg/mae_surf_p` — selects best of raw vs EMA |
| Paper metric | `test_avg/mae_surf_p` — uses the model that produced the best val |

## Validation/test splits

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains

## Round-0 history

- **PR #344 (edward H2):** linear warmup + per-step cosine + NaN fix → val=120.97, test=109.92.
- **PR #404 (edward H11):** Re-conditional FiLM + wd=5e-4 → val=119.36, test=107.54.
- **PR #442 (thorfinn H12):** EMA decay=0.99 + every-other-epoch eval → val_ema=109.19, test=98.47.
- **PR #343 (askeladd H6):** bf16 + torch.compile (mode="default", dynamic=True) → val=80.91, test=72.73.
- **PR #576 (nezuko H16):** arcsinh-on-pressure scale=500 → **val_ema=59.31, test=52.98** (super-additive compound with #343).

## Key methodological tooling shipped this round

- **`--seed` CLI flag** (PR #404) — seed-controlled comparison protocol now demonstrated **eight times** with clean baseline reproduction (most recent: PR #576 Run E reproduces PR #343 internals in expected ways via stats-recomputation paths).
- **`--ema_decay`, `--ema_eval_every`, `--ema_warmup_steps` flags** (PR #442) — EMA is default-on with active-checkpoint selection.
- **`--amp_dtype`, `--compile`, `--compile_mode`, `--grad_accum_steps` flags** (PR #343) — bf16, torch.compile, gradient accumulation all configurable.
- **`--arcsinh_p_scale` flag** (PR #576) — pressure target transformation with proper stats recomputation.
- **Defensive `nan_to_num` in `evaluate_split`** (PR #344) — robust against `test_geom_camber_cruise` sample 20's `-inf` GT.
- **bf16 fp32-fallback in `evaluate_split`** (PR #343) — defensive against bf16 producing non-finite preds at masked-in positions.
