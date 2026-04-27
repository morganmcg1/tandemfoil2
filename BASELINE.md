# Baseline — TandemFoilSet (willow-pai2d-r5)

**Status:** TBD — first round of experiments will calibrate baseline metrics.

## Reference configuration (default `train.py`)

The baseline is the default Transolver in `train.py` at HEAD of `icml-appendix-willow-pai2d-r5`:

- **Model:** Transolver, `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2` (**~662k params** — measured from PR #331)
- **Optimizer:** AdamW `lr=5e-4`, `weight_decay=1e-4`
- **Schedule:** CosineAnnealingLR with `T_max=epochs`
- **Batch size:** 4
- **Loss:** MSE in normalized space, `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10`
- **Training:** `epochs=50`, capped by `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Sampling:** `WeightedRandomSampler` over balanced domain weights

## Reproduce command

```bash
cd /workspace/senpai/target
python train.py --epochs 50 --wandb_group baseline_v0
```

## Primary metric

**`val_avg/mae_surf_p`** — equal-weight mean of surface pressure MAE across the four validation splits:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

Lower is better. The matching test metric `test_avg/mae_surf_p` is computed at the end of every run from the best validation checkpoint.

## Best results

_(updated as winning PRs merge; first round of experiments will populate this section)_

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| —  | —                  | —                   | First round in flight |
