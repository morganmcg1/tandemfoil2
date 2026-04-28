# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **101.9253** | `8lyryo5g` | #752 |
| `test_avg/mae_surf_p` | **NaN** (3-split mean: 100.83) | `8lyryo5g` | #752 |
| Best epoch | 14 / 50 (timeout) | | |
| Wall time | 30.9 min | | |

## Per-split val (epoch 14, run `8lyryo5g`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|-------------|-------------|
| `val_single_in_dist` | 133.248 | 1.643 | 0.627 |
| `val_geom_camber_rc` | 109.255 | 1.944 | 0.801 |
| `val_geom_camber_cruise` | 76.132 | 0.801 | 0.421 |
| `val_re_rand` | 89.066 | 1.328 | 0.594 |
| **val_avg** | **101.925** | **1.429** | **0.611** |

## Per-split test (best checkpoint, epoch 14)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 121.175 |
| `test_geom_camber_rc` | 95.506 |
| `test_geom_camber_cruise` | **NaN** (Inf vol_loss → NaN propagation) |
| `test_re_rand` | 85.812 |
| **test_avg (3 finite splits)** | **100.83** |
| `test_avg/mae_surf_p` (all 4) | NaN — blocked by cruise NaN/Inf bug |

## Configuration (post-#752)

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| **Loss** | **L1 (absolute error) on normalized space**, vol + surf_weight × surf |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |

## Reproduce

```bash
cd target/
python train.py --agent willowpai2e4-askeladd \
  --wandb_name "willowpai2e4-askeladd/l1-loss"
```

## Open issues

- **Cruise-test NaN/Inf bug:** `test_geom_camber_cruise/vol_loss` is Inf in
  every round-1 run (regardless of MSE or L1), which propagates NaN into
  `mae_surf_p` for that split and corrupts `test_avg/mae_surf_p`. The model
  emits Inf on at least one cruise-test sample. Until guarded, the headline
  test metric is unreportable across the round.
