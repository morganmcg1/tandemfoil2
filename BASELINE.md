# TandemFoilSet Baseline Metrics

## 2026-04-29 — PR #1112: Attention dropout=0.1 for OOD slice regularization

**Student:** charliepai2f4-edward  
**Branch:** charliepai2f4-edward/attention-dropout  
**Model:** Transolver (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, dropout=0.1)  
**Best epoch:** 13 / 50 (training timed out at ~14 epochs)  
**Metric summary:** `models/model-attention-dropout-20260429-100507/metrics.yaml`

### Primary metric (lower is better)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **129.531** |

### Per-split surface pressure MAE (val)

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | 159.429 |
| val_geom_camber_rc | 155.559 |
| val_geom_camber_cruise | 92.955 |
| val_re_rand | 110.181 |

### Per-split surface MAE (val, all channels)

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p |
|-------|-------------|-------------|------------|
| val_single_in_dist | 1.832 | 0.847 | 159.429 |
| val_geom_camber_rc | 3.366 | 1.140 | 155.559 |
| val_geom_camber_cruise | 1.489 | 0.594 | 92.955 |
| val_re_rand | 2.028 | 0.859 | 110.181 |

### Test split surface pressure MAE

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 139.912 |
| test_geom_camber_rc | 138.778 |
| test_geom_camber_cruise | NaN (corrupt sample 000020.pt — inf values in ground truth) |
| test_re_rand | 110.986 |
| **test_avg/mae_surf_p** | **NaN** (due to corrupt sample) |

Note: `test_geom_camber_cruise/000020.pt` has 761 inf values in the ground-truth `y` tensor, causing `scoring.py` to produce NaN for the cruise test split. Valid test splits (3 of 4) show consistent generalization.

### Reproduce

```bash
cd target/ && python train.py \
  --experiment_name attention-dropout \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --dropout 0.1 --lr 5e-4 --surf_weight 10 --batch_size 4
```
