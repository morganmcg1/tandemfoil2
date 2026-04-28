# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #506 (charliepai2d3-nezuko) — **L1 surface loss + 12-frequency Fourier
positional features + EMA(0.999) + matched cosine + lr=7.5e-4 + grad
clipping (max_norm=1.0)**. Run with `--epochs 14 --lr 7.5e-4` on the
post-merge advisor.

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **78.80** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **69.13** |
| Per-epoch wallclock | ~133 s |
| Peak GPU memory (batch=4) | 42.51 GB |
| Wallclock total | 31.0 min |
| Param count | 674,647 (+4,096 from FF=12 expansion) |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 92.73 |
| val_geom_camber_rc     | 89.86 |
| val_geom_camber_cruise | 57.32 |
| val_re_rand            | 75.30 |
| **val_avg**            | **78.80** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 78.60 |
| test_geom_camber_rc     | 80.12 |
| test_geom_camber_cruise | 49.29 |
| test_re_rand            | 68.50 |
| **test_avg**            | **69.13** |

**Recommended reproduce command**:

```bash
cd target/
python train.py --epochs 14 --lr 7.5e-4 --experiment_name baseline_ref
```

The post-merge advisor `train.py` has L1 + 12-freq FF + EMA(0.999) +
grad clipping baked in. The two CLI flags supply matched cosine
(`--epochs 14`) and the bumped peak LR (`lr=7.5e-4`).

## Round 3 progress

| Round | val | test | Lever | Δ vs prior |
|-------|----:|-----:|-------|--:|
| Pre-r3 | TBD | — | — | — |
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine `--epochs 14` (CLI) | −1.06% / −0.33% |
| PR #447 |  82.97 |  73.58 | + EMA(0.999) | **−8.7% / −9.0%** |
| PR #461 |  80.28 |  70.92 | + lr=7.5e-4 (CLI) | −3.2% / −3.6% |
| PR #462 |  80.06 |  70.04 | + grad clipping max_norm=1.0 | −0.27% / −1.24% |
| **PR #506 (current)** | **78.80** | **69.13** | **+ NUM_FOURIER_FREQS=12** | **−1.57% / −1.30%** |

**Cumulative round-3 improvement: −41.7% on val, −43.9% on test.**

## Round-3 proven levers (cumulative — seven stacked levers)

1. **L1 surface loss** (PR #280)
2. **8→12-freq spatial Fourier features** (PR #400 → PR #506)
3. **Matched cosine `--epochs 14`** (PR #389, CLI)
4. **EMA-of-weights, decay=0.999** (PR #447)
5. **Peak LR `lr=7.5e-4`** (PR #461, CLI)
6. **Gradient clipping max_norm=1.0** (PR #462)
7. **NUM_FOURIER_FREQS=12** (PR #506) — refinement of lever #2.

The advisor `train.py` bakes in 1, 2, 4, 6, 7 by default. Levers 3
and 5 are CLI flags (`--epochs 14 --lr 7.5e-4`).

## Compose pattern map (round-3 finding, comprehensive)

Round-3 PRs revealed multiple compose patterns:

| compose pattern | with FF/EMA | examples | result |
|----------------|---------|----------|:--|
| Distributional / trajectory averaging | additive | matched cosine + lr=7.5e-4 (#461), grad clipping (#462), FF freq bump (#506) | merged |
| Magnitude-based regulariser, small dose | additive | wd=5e-4 standalone (#469) | partial — saturates on full stack (#500) |
| Magnitude-based regulariser, large dose | destructive on rc-camber | wd=1e-3 (#437), beta2=0.95 (#446) | closed |
| Loss-shape regulariser | overlaps with EMA | L1-volume × EMA (#492) | closed |
| LR overshoot | regression | lr=1e-3 × EMA (#489) | closed |
| Direction-only-update cliff | under-convergence | max_norm=0.5 (#499), DropPath 0.1 (#501) | closed |
| Schedule × averaging interference | OOD regression | matched cosine × EMA (#476) | closed |
| Saturated regularisation overlap | no marginal value | wd=5e-4 × full stack (#500) | closed |
| Input encoding on already-rich features | net-flat / regression | log(Re) FF (#432) | closed |

**Round-5 assignment heuristic**:
- Prefer levers that are **distributional**, **trajectory-averaging**,
  or **mechanistically different** from existing regularisers.
- Magnitude-based regularisers (wd, beta2) compose with FF only at
  small doses; large doses interfere on rc-camber.
- Schedule × averaging × magnitude-regulariser interactions are
  non-trivial; the canonical 6-lever-with-EMA stack hides
  significant interference (matched cosine × EMA per PR #476).
- Per-split signal is load-bearing for compose decisions.

## Reference (unmodified Transolver) configuration

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `fun_dim` | `X_DIM - 2 + 4 * NUM_FOURIER_FREQS` = 22 + 48 = **70** (FF=12) |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, **MSE volume + L1 surface** |
| Input encoding | raw 24-d `x` + 12-frequency Fourier of `(x, z)` |
| Weight averaging | **EMA(decay=0.999)** at every step, swap for val/test eval |
| Gradient clipping | **`clip_grad_norm_(max_norm=1.0)`** before optimiser step |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Default LR | **5e-4** (override with `--lr 7.5e-4` for the round-3 best config) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
