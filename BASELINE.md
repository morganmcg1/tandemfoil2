# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #462 (charliepai2d3-edward) — **L1 surface loss + 8-frequency
Fourier features + matched cosine schedule + gradient clipping
(`max_norm=1.0`)**. Run with `--epochs 14 --lr 5e-4` (default LR).
*Note: branched off pre-#447 advisor, so the measurement does not
include EMA*; the post-merge advisor includes EMA from #447.

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **80.06** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **70.04** |
| Per-epoch wallclock | ~131 s |
| Peak GPU memory (batch=4) | 42.38 GB |
| Wallclock total | ~30.6 min |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 93.59 |
| val_geom_camber_rc     | 92.33 |
| val_geom_camber_cruise | 57.74 |
| val_re_rand            | 76.57 |
| **val_avg**            | **80.06** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 82.41 |
| test_geom_camber_rc     | 79.66 |
| test_geom_camber_cruise | 49.50 |
| test_re_rand            | 68.56 |
| **test_avg**            | **70.04** |

**Recommended reproduce command** (round-3-best six-lever stack):

```bash
cd target/
python train.py --epochs 14 --lr 7.5e-4 --experiment_name baseline_ref
```

This invokes the post-merge advisor (which has L1 + FF + EMA + grad
clipping baked into `train.py`) with matched cosine and the bumped
peak LR. **Important caveat**: PR #462's measurement was on the
pre-#447 advisor (no EMA) and used default lr=5e-4. The post-merge
advisor adds EMA + the lr=7.5e-4 recommendation; the actual
**six-lever-stack** number on the new advisor is **untested** but
expected to land below 80.06 since EMA was a +9% standalone lever.

## Round 3 progress

| Round | val | test | Lever | Δ vs prior |
|-------|----:|-----:|-------|--:|
| Pre-r3 | TBD | — | — | — |
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine `--epochs 14` (CLI) | −1.06% / −0.33% |
| PR #447 |  82.97 |  73.58 | + EMA(0.999) | −8.7% / −9.0% |
| PR #461 |  80.28 |  70.92 | + lr=7.5e-4 (CLI) | −3.2% / −3.6% |
| **PR #462 (current)** | **80.06** | **70.04** | **+ grad clipping max_norm=1.0** | **−0.27% / −1.24%** |

## Round-3 proven levers (cumulative — six stacked levers)

1. **L1 surface loss** (PR #280) — loss formulation aligned with metric.
2. **8-freq spatial Fourier features** (PR #400) — spectral bias mitigation.
3. **Matched cosine `--epochs 14`** (PR #389, CLI) — full LR decay.
4. **EMA-of-weights, decay=0.999** (PR #447) — late-training trajectory averaging.
5. **Peak LR `lr=7.5e-4`** (PR #461, CLI) — bumped from `5e-4` default.
6. **Gradient clipping max_norm=1.0** (PR #462) — stability; clip fires throughout training (grad norms ~27× max_norm even at epoch 14).

The advisor `train.py` bakes in 1, 2, 4, 6 by default. Levers 3 and 5
are CLI flags (`--epochs 14 --lr 7.5e-4`).

## Reference (unmodified Transolver) configuration

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `fun_dim` | `X_DIM - 2 + 4 * NUM_FOURIER_FREQS` = 22 + 32 = **54** |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, **MSE volume + L1 surface** |
| Input encoding | raw 24-d `x` + 8-frequency Fourier of `(x, z)` |
| Weight averaging | **EMA(decay=0.999)** at every step, swap for val/test eval |
| Gradient clipping | **`clip_grad_norm_(max_norm=1.0)`** before optimiser step |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Default LR | **5e-4** (override with `--lr 7.5e-4` for the round-3 best config) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Compose pattern map (round-3 finding)

Six round-3 PRs tested compose effects on top of L1+FF. The pattern
that emerges:

| compose | with FF | example |
|---------|---------|---------|
| **Distributional** (broad gain across all splits) | additive | matched cosine + lr=7.5e-4 (PR #461), grad clipping (PR #462) |
| **Trajectory averaging** | clean orthogonal | EMA × FF (PR #447) |
| **L1-only-OOD-camber-targeted** at high doses | destructive on rc-camber | wd=1e-3 (PR #437), beta2=0.95 (PR #446) |
| **L1-only-OOD-camber-targeted** at low doses | additive | wd=5e-4 (PR #469, validated but lost to current by merge timing) |

**Round-5 assignment heuristic**:
- Prefer levers that are **distributional**, **trajectory-averaging**,
  or **mechanistically different from existing regularisers** —
  these compose with FF.
- Lower confidence in levers whose L1-only effect is **specifically
  OOD-camber-targeted via regularisation magnitude** (wd, beta2) at
  high doses — these often interfere with FF on rc-camber.
- Test **mechanistically-different regulariser axes** (DropPath,
  stochastic depth, dropout, MixUp on input space) before concluding
  the OOD-camber bottleneck is exhausted.

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
