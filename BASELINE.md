# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **99.2257** | `m46h5g4s` | #754 |
| `test_avg/mae_surf_p` | **NaN** (3-split mean: 99.34) | `m46h5g4s` | #754 |
| Best epoch | 12 / 50 (timeout @ ep 14) | | |
| Wall time | 30.77 min | | |

## Per-split val (epoch 12, run `m46h5g4s`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|-------------|-------------|
| `val_single_in_dist` | 116.675 | 1.358 | 0.657 |
| `val_geom_camber_rc` | 113.935 | 2.820 | 0.934 |
| `val_geom_camber_cruise` | 75.015 | 1.129 | 0.487 |
| `val_re_rand` | 91.279 | 1.850 | 0.693 |
| **val_avg** | **99.226** | **1.789** | **0.693** |

## Per-split test (best checkpoint, epoch 12)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 106.775 |
| `test_geom_camber_rc` | 104.872 |
| `test_geom_camber_cruise` | **NaN** (cruise GT bug, fix in flight at #797) |
| `test_re_rand` | 86.370 |
| **test_avg (3 finite splits)** | **99.34** |
| `test_avg/mae_surf_p` (all 4) | NaN — blocked by cruise NaN bug |

## Configuration (post-#754)

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
| **`channel_weights`** | **[1.0, 1.0, 3.0]** for [Ux, Uy, p] |
| Loss | L1 (absolute error) on normalized space, vol + surf_weight × surf |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |

## Delta vs prior baseline (PR #752 L1)

| Metric | L1-only (#752) | L1 + ch=[1,1,3] (#754) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 101.93 | **99.23** | **−2.65%** |
| 3-split test mean | 100.83 | 99.34 | −1.48% |
| `val_avg/mae_surf_Ux` | 1.429 | 1.789 | +25.2% |
| `val_avg/mae_surf_Uy` | 0.611 | 0.693 | +13.4% |

The 3× pressure weight stacks with L1: net pressure improvement, with
acceptable velocity-channel regression (we don't rank on velocity).
Biggest val gain on `val_single_in_dist` (133.25 → 116.68, −12.4%) — the
heaviest-tail split where pressure outliers dominate.

## Reproduce

```bash
cd target/
python train.py --agent willowpai2e4-fern \
  --wandb_name "willowpai2e4-fern/p-channel-3x-on-L1"
```

## Open issues

- **Cruise-test NaN bug:** `test_geom_camber_cruise/000020.pt` has 761 NaN
  values in the GT p-channel (out of 225K nodes). The cruise-test
  prediction for that sample also goes Inf in some runs. Both paths feed
  through `data/scoring.py:accumulate_batch` where `NaN * 0 = NaN`
  propagates into the channel-sum. Fix in flight at PR #797 (askeladd, scope
  expanded to handle both paths). Until guarded, `test_avg/mae_surf_p` is
  unreportable.
