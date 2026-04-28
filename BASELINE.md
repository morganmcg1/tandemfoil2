# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #461 (charliepai2d3-askeladd) — **L1 surface loss + 8-frequency
Fourier features + matched cosine schedule (`--epochs 14`) + bumped
peak LR (`lr=7.5e-4`, 1.5× the default)**. All other knobs at
unmodified defaults.

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **80.28** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **70.92** |
| Per-epoch wallclock | ~131 s |
| Peak GPU memory (batch=4) | 42.38 GB |
| Wallclock total | 30.69 min |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 89.76 |
| val_geom_camber_rc     | 90.03 |
| val_geom_camber_cruise | 62.42 |
| val_re_rand            | 78.92 |
| **val_avg**            | **80.28** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 78.18 |
| test_geom_camber_rc     | 80.52 |
| test_geom_camber_cruise | 53.88 |
| test_re_rand            | 71.12 |
| **test_avg**            | **70.92** |

**Recommended reproduce command**:

```bash
cd target/
python train.py --epochs 14 --lr 7.5e-4 --experiment_name baseline_ref
```

This invokes the post-merge advisor (which has L1 + FF + **EMA** in
`train.py`) with matched cosine and the bumped peak LR. **Important
caveat**: PR #461's measurement was on the post-#400 advisor (had FF
and matched cosine via askeladd's own #389) but **before PR #447
merged** (didn't have EMA). So the headline 80.28 is L1+FF + matched
cosine + lr=7.5e-4 *without EMA*. The post-#461 advisor includes
EMA from #447. The actual five-lever-stack number on the post-merge
advisor is **untested** but expected to land below 80.28 (since EMA
was a clean +9% lever in PR #447).

## Round 3 progress

| Round | val | test | Lever | Δ vs prior |
|-------|----:|-----:|-------|--:|
| Pre-r3 | TBD | — | — | — |
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine `--epochs 14` (CLI) | **−1.06% / −0.33%** |
| PR #447 |  82.97 |  73.58 | + EMA(0.999) | **−8.7% / −9.0%** |
| **PR #461 (current)** | **80.28** | **70.92** | **+ lr=7.5e-4 (CLI, no EMA in measurement)** | **−3.2% / −3.6%** |

## Round-3 proven levers (cumulative — five stacked levers)

1. **L1 surface loss** (PR #280) — loss formulation aligned with metric.
2. **8-freq spatial Fourier features** (PR #400) — spectral bias mitigation.
3. **Matched cosine `--epochs 14`** (PR #389) — full LR decay (CLI flag).
4. **EMA-of-weights, decay=0.999** (PR #447) — late-training trajectory averaging.
5. **Peak LR `lr=7.5e-4`** (PR #461) — bumped from `5e-4` default; works under matched cosine without warmup (cosine self-warmup from peak handles ep1-3).

The advisor `train.py` bakes in 1, 2, and 4 by default. Levers 3 and 5
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
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Default LR | **5e-4** (override with `--lr 7.5e-4` for the round-3 best config) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Convergent OOD-camber narrative — partially refuted (PR #437 finding)

Five round-3 levers all improved `val_geom_camber_rc` on the L1
baseline. PR #437 showed wd × FF **overlap destructively** on
rc-camber while composing on cruise/in-dist. PR #447 (EMA × FF)
showed clean **additive compose** (most orthogonal pair this round).
PR #461 shows **distributional gain** (broad across all 4 splits,
not concentrated on rc-camber) when LR is bumped on matched cosine.

Three different compose patterns from three different lever pairs:
overlap, additive, distributional. The regularisation/optimisation
landscape is multi-dimensional. Per-split analysis is load-bearing.

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
