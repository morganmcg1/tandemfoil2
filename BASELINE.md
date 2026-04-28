# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #389 (charliepai2d3-askeladd) — **L1 surface loss + matched cosine
schedule** (`--epochs 14`, T_max=14). All other knobs at unmodified
defaults (`bs=4`, `lr=5e-4`, `weight_decay=1e-4`, `surf_weight=10`,
`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
`mlp_ratio=2`).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **90.90** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **80.84** |
| Per-epoch wallclock | ~131 s |
| Peak GPU memory (batch=4) | 42.14 GB |
| Wallclock total | ~30.6 min |
| Reproducibility | 3 seeded re-runs at 90.90 / 91.47 / 91.94 (~1% spread) |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 105.82 |
| val_geom_camber_rc     | 100.82 |
| val_geom_camber_cruise |  71.37 |
| val_re_rand            |  85.60 |
| **val_avg**            | **90.90** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 94.78 |
| test_geom_camber_rc     | 88.30 |
| test_geom_camber_cruise | 59.67 |
| test_re_rand            | 80.62 |
| **test_avg**            | **80.84** |

## Important reproducibility note

PR #389 measured **L1 + matched cosine without FF** (branched off the
L1-only advisor, pre-#400). The current advisor `train.py` includes
both L1 surface loss (PR #280) **and** 8-frequency Fourier positional
features (PR #400). So:

- **Best confirmed measurement (this row)**: val 90.90 / test 80.84 from
  L1 + matched cosine, **without** FF.
- **Best confirmed measurement on FF**: val 91.87 / test 81.11 from
  L1 + FF + `--epochs 50` (PR #400).
- **Untested (round-4 priority)**: L1 + FF + matched cosine. If FF and
  matched cosine compose, the result should land below 90.90.

**Recommended reproduce command** for round-4 PRs assigned against the
new baseline:

```bash
cd target/
python train.py --epochs 14 --experiment_name baseline_ref
```

This invokes the post-#400 advisor (L1 + FF) with matched cosine
schedule. The measured number will replace the 90.90 estimate once the
first round-4 PR confirms it.

## Round 3 progress

| Round | Best val_avg/mae_surf_p | Best test_avg/mae_surf_p | Lever | Δ vs prior |
|-------|------------------------:|-------------------------:|-------|----:|
| Pre-r3 | TBD | — | — | — |
| PR #306 (merged) | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 (merged) | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 (merged) |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| **PR #389 (merged, current)** | **90.90** | **80.84** | **+ matched cosine `--epochs 14`** | **−1.06% / −0.33%** |

Notes:

- PR #389 measurement was on L1-only baseline; the val 90.90 win vs the
  L1+FF baseline (91.87) is small (~1%) and below the seed-noise floor
  measured in this round (~5-13%). Treat the current best as
  approximately tied between PR #389's config and PR #400's config.
- The matched cosine fix is **structural** — every other round-3 PR was
  running cosine T_max=50 with the actual budget capped at epoch 14, so
  the LR never decayed below ~81% of peak. The next round-4 PR to
  measure will use the post-#400 advisor (L1+FF) with `--epochs 14`,
  which is the cleanest single-config measurement of "all proven levers
  stacked".

## Convergent OOD-camber generalisation signal (round-3 finding)

Five different round-3 levers produced the **same per-split improvement
pattern**: dominant win on `val_geom_camber_rc` (the unseen-front-foil-
camber raceCar tandem track), flat-or-mild on the other splits.

| PR | lever | `val_geom_camber_rc` Δ |
|----|-------|-----------------------:|
| #400 (merged) | spatial Fourier features | −20.8% |
| #389 (merged) | matched cosine | **−19.4%** |
| #419 (closed) | AdamW(beta2=0.95) | −13.6% |
| #395 (closed) | weight_decay=1e-3 | −11.9% |
| #423 (under review) | gradient clipping | **−15.0%** |

Five different mechanisms (input encoding / schedule / optimiser /
regularisation / stability), same direction of effect. Whatever's
bottlenecking `val_geom_camber_rc` is responsive to "make optimisation
more effective in the limited budget we have". Round-4 compose tests
(in flight) will reveal whether these levers each hit independent
paths or share a common dynamic.

## Reference (unmodified Transolver) configuration

Defaults baked into `train.py` after PR #280 + PR #400 merges (PR #389
was CLI-only and didn't change defaults):

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
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Default epochs | **50** (override with `--epochs 14` for matched cosine) |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
