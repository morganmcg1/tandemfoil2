# Baseline — icml-appendix-charlie-pai2c-r2

## Status

**Empirical floor not yet established.** Round 1 of this advisor branch is in flight; baseline will be set by the first PR that produces a clean `val_avg/mae_surf_p`.

## Seed configuration (`train.py` defaults)

- Architecture: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Output fields: `[Ux, Uy, p]`, output_dims `[1, 1, 1]`
- Training: `lr=5e-4` (cosine annealing to 0, no warmup), `weight_decay=1e-4`, `batch_size=4`, AdamW
- Loss: `vol_loss + 10.0 * surf_loss` (MSE in normalized space)
- Epochs: 50 (or `SENPAI_MAX_EPOCHS` cap), wall clock cap `SENPAI_TIMEOUT_MINUTES`
- Sampler: `WeightedRandomSampler` from `load_data` (balances raceCar single, raceCar tandem, cruise tandem)

## Primary ranking metric

`val_avg/mae_surf_p` — equal-weight surface-pressure MAE across:
- `val_single_in_dist`
- `val_geom_camber_rc` (M=6–8 holdout)
- `val_geom_camber_cruise` (M=2–4 holdout)
- `val_re_rand` (stratified Re holdout)

Test counterpart: `test_avg/mae_surf_p` — paper-facing number, evaluated from best validation checkpoint.

## Reproduce command

```bash
cd target && python train.py \
    --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
    --batch_size 4 --surf_weight 10.0 \
    --experiment_name baseline-seed
```

## Update protocol

When a PR beats the current `val_avg/mae_surf_p`, update this file with:
- PR number and merge date
- New best `val_avg/mae_surf_p` and `test_avg/mae_surf_p`
- Per-split `mae_surf_p` for diagnostics
- Path to metrics summary (`models/<experiment>/metrics.yaml`)

## History

- _2026-04-27_: Branch opened. Baseline TBD (Round 1 in flight).
