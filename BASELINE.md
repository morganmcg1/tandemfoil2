# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #447 (charliepai2d3-fern) — **L1 surface loss + 8-frequency Fourier
positional features + EMA-of-weights with decay=0.999** (budget-aware,
derived from `EMA_DECAY = 1 − 1/(0.2 × total_steps)` for ~5K steps).
All other knobs at unmodified defaults (`bs=4`, `lr=5e-4`,
`weight_decay=1e-4`, `surf_weight=10`, `n_hidden=128`, `n_layers=5`,
`n_head=4`, `slice_num=64`, `mlp_ratio=2`, cosine T_max=50).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **82.97** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **73.58** |
| Per-epoch wallclock | ~132 s |
| Peak GPU memory (batch=4) | 42.4 GB |
| Wallclock total | ~30.8 min (timeout-bound at epoch 14) |
| Param count | 670,551 (EMA shadow ~2.6 MB extra, fp32) |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     |  99.44 |
| val_geom_camber_rc     |  93.14 |
| val_geom_camber_cruise |  61.06 |
| val_re_rand            |  78.22 |
| **val_avg**            | **82.97** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 88.79 |
| test_geom_camber_rc     | 81.54 |
| test_geom_camber_cruise | 52.36 |
| test_re_rand            | 71.62 |
| **test_avg**            | **73.58** |

Reproduce:

```bash
cd target/
python train.py --experiment_name baseline_ref
```

(L1 surface loss, 8-frequency Fourier features, and EMA-of-weights
with decay=0.999 are all baked into `train.py`. The scoring fix is on
the advisor branch since commit `2eb5c7f` — `test_avg/*_p` lands as a
clean number.)

## Important note: matched cosine `--epochs 14`

PR #389 (askeladd, merged 2026-04-28 02:28) demonstrated that
`--epochs 14` (matched cosine schedule with T_max=14) was a free
−11.4% gain on the L1-only baseline. PR #447 above was measured with
the default `--epochs 50` schedule (cosine T_max=50, never reaches the
tail under the 30-min wallclock cap).

**Round-4 untested**: L1+FF+EMA + matched cosine `--epochs 14`. Likely
gives another small additive gain (the matched-cosine effect on L1+FF
is bracketed by the L1-only matched-cosine result of −11.4% and the
small L1+FF win of −1%). Round-4 priority is to measure this. fern's
next assignment tests it directly.

## Round 3 progress

| Round | Best val_avg/mae_surf_p | Best test_avg/mae_surf_p | Lever | Δ vs prior |
|-------|------------------------:|-------------------------:|-------|----:|
| Pre-r3 | TBD | — | — | — |
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine `--epochs 14` (CLI) | **−1.06% / −0.33%** |
| **PR #447 (current)** | **82.97** | **73.58** | **+ EMA (decay=0.999)** | **−8.7% / −9.0%** |

EMA is the largest single-lever win since L1 surface loss (PR #280's
−24.1%). It uniquely targets the persistent `val_single_in_dist`
bottleneck — closes 15.2% of that gap, more than any other round-3
lever. Mechanistically orthogonal to L1 (loss shape), FF (input
encoding), and matched cosine (schedule shape).

## Round-3 proven levers (cumulative, four stacked)

1. **L1 surface loss** (PR #280) — loss formulation aligned with metric.
2. **8-freq spatial Fourier features** (PR #400) — spectral bias mitigation.
3. **Matched cosine `--epochs 14`** (PR #389) — let LR fully decay (CLI flag).
4. **EMA-of-weights, decay=0.999** (PR #447) — late-training trajectory averaging.

## Convergent OOD-camber narrative — partially refuted (PR #437 finding)

Five round-3 levers all improved `val_geom_camber_rc` on the L1
baseline. PR #437 showed wd × FF **overlap destructively** on
rc-camber while composing on cruise/in-dist. This contradicts the
naive "stack all five additively" round-5 plan. Per-split analysis is
load-bearing.

PR #447's per-split signal **is the cleanest "additive compose"** of
round 3: EMA gain on L1 (−10.4%, PR #396) and EMA gain on L1+FF
(−9.7%, this PR) are within 1% — EMA's mechanism is fully orthogonal
to FF.

## Reference (unmodified Transolver) configuration

Defaults baked into `train.py` after PR #280 + #400 + #447 merges
(PR #389 was CLI-only and didn't change defaults):

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
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
