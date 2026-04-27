# Baseline — icml-appendix-charlie-pai2c-r2

## Current best

**`val_avg/mae_surf_p = 102.71`** (epoch 12; training cut by 30-min wall clock at epoch 14)

- **PR**: #213 — Physical-space L1 surface loss (volume stays MSE)
- **Student**: charliepai2c2-nezuko
- **Merged**: 2026-04-27 (commit efa6e3c)
- **Param count**: ~1.18M (same arch as the previous baseline; loss-only change)
- **Peak VRAM**: 42.13 GB
- **Wall-clock**: 30.6 min (hit timeout; val curve was still trending down)

### Per-split val (best epoch 12)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist | 125.02 | 2.27 | 1.27 | 130.67 |
| val_geom_camber_rc | 109.49 | 3.29 | 1.72 | 114.21 |
| val_geom_camber_cruise | 80.47 | 1.63 | 1.06 | 74.28 |
| val_re_rand | 95.85 | 2.45 | 1.31 | 91.05 |
| **val_avg** | **102.71** | | | |

### Per-split test (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist | 108.28 | 2.15 | 1.19 |
| test_geom_camber_rc | 99.19 | 3.20 | 1.64 |
| test_geom_camber_cruise | NaN ⚠️ (logged) / **67.62** (NaN-safe, 199/200) | 1.59 | 1.01 |
| test_re_rand | 90.99 | 2.27 | 1.36 |
| **test_avg** | **NaN ⚠️ (logged) / 91.52 (NaN-safe)** | | |

`test_avg/mae_surf_p` is logged as NaN due to the same `data/scoring.py::accumulate_batch` bug both #216 and #213 identified (`NaN * 0 = NaN` in PyTorch, contaminates the masked-out entries before reduction). Manual NaN-safe re-evaluation gives `test_avg ≈ 91.52`. **Rank PRs on `val_avg/mae_surf_p` until human team patches the read-only `data/scoring.py`.**

### Improvement vs. previous baseline (PR #216, val_avg = 130.057)

| Split | Previous | Current | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 161.146 | 125.02 | **–22.4%** |
| val_geom_camber_rc | 142.415 | 109.49 | **–23.1%** |
| val_geom_camber_cruise | 98.282 | 80.47 | **–18.1%** |
| val_re_rand | 118.383 | 95.85 | **–19.0%** |
| **val_avg** | **130.057** | **102.71** | **–21.0%** |

Consistent gains across all four val splits — exactly the cross-split robustness we want from a Round-1 result.

### Metrics references

- Local summary: `models/model-charliepai2c2-nezuko-surface-pressure-l1-loss-20260427-195051/metrics.yaml`
- JSONL: `models/model-charliepai2c2-nezuko-surface-pressure-l1-loss-20260427-195051/metrics.jsonl`
- Centralized: `research/EXPERIMENT_METRICS.jsonl`

### Reproduce command

```bash
cd target && python train.py \
    --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
    --batch_size 4 --surf_weight 10.0 \
    --experiment_name surface-pressure-l1-loss \
    --agent charliepai2c2-nezuko
```

(The merged advisor branch already includes the wider-shallower architecture from #216 plus the L1-surface-loss change from #213.)

## Primary ranking metric

`val_avg/mae_surf_p` — equal-weight surface-pressure MAE across:
- `val_single_in_dist`
- `val_geom_camber_rc` (M=6–8 holdout)
- `val_geom_camber_cruise` (M=2–4 holdout)
- `val_re_rand` (stratified Re holdout)

Test counterpart: `test_avg/mae_surf_p` — paper-facing number from best validation checkpoint. Currently NaN-contaminated by `data/scoring.py` bug; use NaN-safe manual re-evaluation for paper numbers until patched.

## History

- _2026-04-27_: Branch opened. Round 1 in flight.
- _2026-04-27 19:52 UTC_: PR #216 merged — first empirical baseline. `val_avg/mae_surf_p = 130.0568` (wider-shallower architecture). Run timed out at epoch 11/50; baseline-establisher.
- _2026-04-27 20:30 UTC_: PR #213 merged — physical-space L1 surface loss (volume stays MSE). New best: `val_avg/mae_surf_p = 102.71` (–21.0% vs prior baseline). Consistent across all 4 splits.
