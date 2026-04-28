# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #657 (charliepai2d3-fern) — **L1 + 12-freq spatial FF + EMA(0.997)
+ matched cosine + lr=7.5e-4 + grad clipping (max_norm=5.0) + decoupled
head LR (2× on `mlp2` + `ln_3`) + aux log-p loss (weight=0.25) + layer
scale (CaiT-style, γ_init=1e-4 on attn + mlp residual branches)**.

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 13/14 — timeout) | **67.29** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **58.39** |
| Per-epoch wallclock | ~142 s |
| Peak GPU memory (batch=4) | 47.50 GB |
| Wallclock total | ~31 min (timeout-bounded; ep 14 not run) |

Per-split val (best epoch 13):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 73.91 |
| val_geom_camber_rc     | 79.82 |
| val_geom_camber_cruise | 49.58 |
| val_re_rand            | 65.84 |
| **val_avg**            | **67.29** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 63.55 |
| test_geom_camber_rc     | 70.41 |
| test_geom_camber_cruise | 41.32 |
| test_re_rand            | 58.29 |
| **test_avg**            | **58.39** |

**Critical caveat**: PR #657 was timeout-bounded at epoch 13/14
(per-epoch wallclock rose from ~132 s to ~142 s with layer scale).
Best-val at epoch 13 was still actively descending (epoch 12 → 13:
−1.57%); a full 14-epoch run (or 16 with looser timeout) could
improve further. Round-5 should consider raising `SENPAI_TIMEOUT_MINUTES`
or trimming eval frequency.

**Mechanistic insight from PR #657**: layer scale is a per-channel
learnable scalar `γ` (init 1e-4) on each residual branch. Diagnostic
from final-epoch γ values:
- `γ_mlp` grew to ~3.0e-2 (avg, all blocks); `γ_attn` grew to ~8.0e-3
  (avg). MLP branches dominate residual contribution by ~3×.
- Block 1 has the highest γ in both branches; block 2 deliberately
  downweights its attention branch (γ_attn=4.7e-3). The model
  expresses real per-block heterogeneity that PR #578 had partially
  captured via the head's 2× LR.
- Layer scale composes orthogonally with PR #578's decoupled head LR
  — γ parameters live in `backbone_params` group at LR=7.5e-4, while
  head's mlp2+ln_3 stay at 1.5e-3.

The win is uniform across all 4 splits (val: −7% to −15%; test: −7%
to −17%), with the largest improvement on `val_geom_camber_cruise`
(−14.6%) — the same split that PR #578 had mildly regressed (+3.44%).
**Layer scale recovers and extends those gains.** This is the largest
single-knob delta seen since PR #280 (L1 surface loss, −24.1% on val).

**Recommended reproduce command**:

```bash
cd target/
python train.py --epochs 14 --lr 7.5e-4 --experiment_name baseline_ref
```

The post-merge advisor `train.py` bakes in L1 + 12-freq FF +
EMA(0.997) + grad clipping max_norm=5.0 + aux log-p (weight=0.25) +
decoupled head LR (2×) + **layer scale (γ_init=1e-4)**. The two CLI
flags supply matched cosine (`--epochs 14`) and the bumped peak LR
(`lr=7.5e-4`).

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
| PR #506 |  78.80 |  69.13 | + NUM_FOURIER_FREQS=12 | −1.57% / −1.30% |
| PR #534 |  78.60 |  67.77 | + EMA_DECAY=0.997 (schedule × EMA fix) | −0.25% / −1.97% |
| PR #572 |  77.78 |  67.71 | + aux log-p loss (weight=0.25) | −1.06% / −0.09% |
| PR #596 |  77.01 |  67.78 | + max_norm=5.0 (loosened clip) | −0.99% / +0.10% |
| PR #578 |  75.78 |  66.27 | + decoupled head LR (2×) | −1.60% / −2.23% |
| **PR #657 (current)** | **67.29** | **58.39** | **+ layer scale (γ_init=1e-4)** | **−11.21% / −11.89%** |

**Cumulative round-3 improvement: −50.2% on val, −52.6% on test.**

## Round-3 proven levers (cumulative — twelve stacked levers)

1. **L1 surface loss** (PR #280)
2. **8→12-freq spatial Fourier features** (PR #400 → PR #506)
3. **Matched cosine `--epochs 14`** (PR #389, CLI)
4. **EMA-of-weights, decay=0.999** (PR #447)
5. **Peak LR `lr=7.5e-4`** (PR #461, CLI)
6. **Gradient clipping max_norm=1.0** (PR #462)
7. **NUM_FOURIER_FREQS=12** (PR #506) — refinement of lever #2.
8. **EMA_DECAY=0.997** (PR #534) — schedule × EMA interference fix.
9. **Auxiliary log-pressure loss (weight=0.25)** (PR #572) — per-split tradeoff lever.
10. **Loosened gradient clipping (max_norm=5.0)** (PR #596) — clip × LR joint optimum shift.
11. **Decoupled head LR (2× on `mlp2`+`ln_3`)** (PR #578) — head adapts faster than backbone.
12. **Layer scale (γ_init=1e-4)** (PR #657) — per-channel learnable residual gating; largest single-knob delta of round 3 since L1 surface loss.

The advisor `train.py` bakes in 1, 2, 4, 6, 7, 8, 9, 10, 11, 12 by
default. Levers 3 and 5 are CLI flags (`--epochs 14 --lr 7.5e-4`).

## Compose pattern map (round-3 finding, comprehensive)

Round-3 PRs revealed multiple compose patterns:

| compose pattern | with FF/EMA | examples | result |
|----------------|---------|----------|:--|
| Distributional / trajectory averaging | additive | matched cosine + lr=7.5e-4 (#461), grad clipping (#462), FF freq bump (#506) | merged |
| Per-channel residual gating (deterministic) | clean orthogonal | layer scale γ_init=1e-4 (#657) | merged |
| Per-parameter-group LR | additive | decoupled head LR (#578) | merged |
| Magnitude-based regulariser, small dose | additive | wd=5e-4 standalone (#469) | partial — saturates on full stack (#500) |
| Magnitude-based regulariser, large dose | destructive on rc-camber | wd=1e-3 (#437), beta2=0.95 (#446) | closed |
| Loss-shape regulariser | overlaps with EMA | L1-volume × EMA (#492) | closed |
| LR overshoot | regression | lr=1e-3 × EMA (#489) | closed |
| Direction-only-update cliff | under-convergence | max_norm=0.5 (#499), DropPath 0.1 (#501) | closed |
| Schedule × averaging interference | OOD regression | matched cosine × EMA (#476) | closed |
| Saturated regularisation overlap | no marginal value | wd=5e-4 × full stack (#500) | closed |
| Input encoding on already-rich features | net-flat / regression | log(Re) FF (#432) | closed |
| Magnitude-dependent compressor (Pareto trade) | partial improvement | head wd 5e-4 (#656), max_norm=10 (#616), eta_min=5e-5 (#617), slice_num=32 (#642), ext-head 2× (#639), anneal-noise (#607) | all closed |
| Activation swap | regression | SiLU (#663, +12.4% on stack) | closed |
| BF16 precision (any guard) | regression on stack | full BF16 (#587), narrow guard (#606), broad guard (#626), attn-FP32 (#655) | all closed |

**Round-5 assignment heuristic**:
- Prefer **deterministic per-channel gating** levers (layer scale was
  the breakthrough; SwiGLU, gated MLP variants are mechanism-aligned).
- Prefer **per-parameter-group** levers that compose with existing
  decoupled head LR (#578).
- Per-split signal remains load-bearing for compose decisions; the
  Pareto-frontier pattern (cruise wins, in-dist + rc-camber lose) is
  resolved at the merged stack level by layer scale.
- **Avoid** activation swaps, BF16 attempts, and magnitude-dependent
  regularisers — all 12+ closed PRs in these categories show the
  round-3 stack is now at a balanced state.

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
| Loss | `vol_loss + 10.0 * surf_loss + 0.25 * log_p_aux`, **MSE volume + L1 surface + aux log-p** |
| Input encoding | raw 24-d `x` + 12-frequency Fourier of `(x, z)` |
| Weight averaging | **EMA(decay=0.997)** at every step, swap for val/test eval |
| Gradient clipping | **`clip_grad_norm_(max_norm=5.0)`** before optimiser step |
| Layer scale | **γ_init=1e-4 on each residual branch (attn + mlp)** |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Default LR | **5e-4** (override with `--lr 7.5e-4` for the round-3 best config) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
