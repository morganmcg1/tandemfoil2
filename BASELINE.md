# Baseline — charlie-pai2d-r5

This advisor track starts from a pristine `train.py`. No round-1 measurements yet — first cohort of PRs will establish the empirical baseline by running modifications against it.

## Reference configuration (unmodified `train.py`)

| Block | Value |
|-------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` | 2 (x, z) |
| `fun_dim` | 22 (X_DIM − 2) |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| `batch_size` | 4 |
| Loss | MSE in normalized space |
| `surf_weight` | 10.0 (volume + 10 × surface) |
| Sampler | WeightedRandomSampler — equal weight across raceCar single, raceCar tandem, cruise tandem domains |
| Epochs cap | 50 (or `SENPAI_TIMEOUT_MINUTES` wall clock) |

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE across the four validation splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Lower is better. Best-validation checkpoint is what gets evaluated on the test splits.

The paper-facing rank is `test_avg/mae_surf_p`, computed once at the end of training from the best-validation checkpoint.

## Best result

**PR #293 — `l1-loss` (edward), merged 2026-04-27**

- `val_avg/mae_surf_p` = **101.868** (best epoch 14/50, run terminated by 30-min wall timeout while still improving)
- `test_avg/mae_surf_p` = **NaN** (4-split) / **102.606** (mean of 3 clean splits — `test_geom_camber_cruise` is a pre-existing data + scoring NaN issue affecting *every* PR this round, not specific to L1)
- Per-split val: `val_single_in_dist=125.26`, `val_geom_camber_rc=108.03`, `val_geom_camber_cruise=75.26`, `val_re_rand=98.91`
- Change: replace MSE `(pred - y_norm)**2` with L1 `(pred - y_norm).abs()` in both training and `evaluate_split` (in normalized space; MAE accumulators in physical space are unchanged).

The L1 change is a clean single-axis swap with no other modifications — `n_hidden=128, n_layers=5, n_head=4, slice_num=64`, `lr=5e-4`, `surf_weight=10.0`, `batch_size=4`, plain CosineAnnealingLR(T_max=50), unchanged.

## Known issue affecting test scoring

`test_geom_camber_cruise` returns `NaN` for `mae_surf_p` and `mae_vol_p` because of a non-finite ground truth on at least one sample (Edward's diagnosis: sample 20 has 761 NaN values in the `p` channel of GT). `data/scoring.accumulate_batch` computes `(pred - y).abs()` before masking, so NaN propagates into the per-channel sum even when the surrounding code intends to skip the sample. `data/scoring.py` is read-only per program constraints, so for now we rank on the 3-clean-split test mean as a stable indicator. Worth flagging to the human team for an upstream fix.
