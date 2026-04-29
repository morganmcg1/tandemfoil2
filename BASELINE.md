# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** charlie-pai2e-r5, PR #1013 (compound configuration reproduced as anchor for this round)

**Primary metric:** `val_avg/mae_surf_p = 47.7385`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=15) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 49.68 |
| val_geom_camber_rc | 60.82 |
| val_geom_camber_cruise | 30.55 |
| val_re_rand | 49.90 |
| **val_avg** | **47.7385** |

**Training:** ~21 min, 50 epochs, batch_size=4

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4
```

## Merge History

(None yet — round in progress)
