# Baseline — icml-appendix-charlie-pai2c-r2

## Current best

**`val_avg/mae_surf_p = 130.0568`** (epoch 11/50; training timed out at 30 min wall clock)

- **PR**: #216 — Wider-shallower Transolver (`n_hidden=192, n_layers=4, n_head=6`)
- **Student**: charliepai2c2-tanjiro
- **Merged**: 2026-04-27 (commit d354f0a)
- **Param count**: 1.18M
- **Peak VRAM**: 51.79 GB
- **Wall-clock**: 30.3 min (hit timeout — training was still improving ~5%/epoch when it stopped)

### Per-split val (best epoch 11)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist | 161.146 | 1.946 | 0.855 | 166.393 | 5.524 | 2.370 |
| val_geom_camber_rc | 142.415 | 2.640 | 1.129 | 138.864 | 5.511 | 2.862 |
| val_geom_camber_cruise | 98.282 | 1.608 | 0.637 | 99.226 | 4.110 | 1.427 |
| val_re_rand | 118.383 | 2.113 | 0.882 | 118.886 | 4.769 | 2.092 |
| **val_avg** | **130.057** | 2.077 | 0.876 | 130.842 | 4.978 | 2.188 |

### Per-split test (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---:|---:|---:|---:|---:|---:|
| test_single_in_dist | 144.256 | 1.860 | 0.804 | 146.616 | 5.047 | 2.135 |
| test_geom_camber_rc | 127.587 | 2.601 | 1.035 | 127.743 | 5.440 | 2.696 |
| test_geom_camber_cruise | NaN ⚠️ | 1.529 | 0.591 | NaN ⚠️ | 3.895 | 1.318 |
| test_re_rand | 115.976 | 1.999 | 0.841 | 115.300 | 4.649 | 1.943 |
| **test_avg** | **NaN** ⚠️ | 1.997 | 0.818 | **NaN** ⚠️ | 4.758 | 2.023 |

`test_avg/mae_surf_p` is NaN-contaminated by a single bad sample (`test_geom_camber_cruise/000020.pt` — 761 Inf pressure values), not a model failure. Mean over the 3 finite test splits: `mae_surf_p ≈ 129.27`. **Rank PRs on `val_avg/mae_surf_p` until the scoring bug is patched in `data/scoring.py` (read-only — needs human team).**

### Metrics references

- Local summary: `models/model-wider-shallower-arch-20260427-191514/metrics.yaml`
- JSONL: `models/model-wider-shallower-arch-20260427-191514/metrics.jsonl`
- Centralized: `research/EXPERIMENT_METRICS.jsonl` (all-experiments rollup)

### Reproduce command

```bash
cd target && python train.py \
    --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
    --batch_size 4 --surf_weight 10.0 \
    --experiment_name wider-shallower-arch \
    --agent charliepai2c2-tanjiro
```

## Primary ranking metric

`val_avg/mae_surf_p` — equal-weight surface-pressure MAE across:
- `val_single_in_dist`
- `val_geom_camber_rc` (M=6–8 holdout)
- `val_geom_camber_cruise` (M=2–4 holdout)
- `val_re_rand` (stratified Re holdout)

Test counterpart: `test_avg/mae_surf_p` — paper-facing number from best validation checkpoint. Currently NaN-contaminated; see note above.

## Update protocol

When a PR beats the current `val_avg/mae_surf_p`, update this file with:
- PR number and merge date
- New best `val_avg/mae_surf_p` and `test_avg/mae_surf_p`
- Per-split `mae_surf_p` for diagnostics
- Path to metrics summary (`models/<experiment>/metrics.yaml`)

## History

- _2026-04-27_: Branch opened. Round 1 in flight.
- _2026-04-27 19:52 UTC_: PR #216 merged — first empirical baseline. `val_avg/mae_surf_p = 130.0568`. Run timed out at epoch 11/50; subsequent runs should consider matching `T_max` to the realistic wall-clock budget.
