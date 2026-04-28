# Baseline — icml-appendix-charlie-pai2e-r3

## Current Best
- **Source**: MAE/L1 loss replacing MSE (PR #835, nezuko)
- **PR**: #835
- **Primary**: `val_avg/mae_surf_p` = **104.058** (lower is better)
- **Test (corrected, NaN-sample skipped)**: `test_avg/mae_surf_p` = **92.608**

### Best checkpoint metrics (val, from reeval_summary)

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist | 1.718 | 0.723 | 124.701 | 134.457 |
| val_geom_camber_rc | 2.189 | 0.885 | 116.841 | 114.777 |
| val_geom_camber_cruise | 1.669 | 0.431 | 76.934 | 67.139 |
| val_re_rand | 1.973 | 0.644 | 97.756 | 91.352 |
| **val_avg** | **1.887** | **0.671** | **104.058** | **101.931** |

### Test metrics (NaN-sample-skipped workaround, `test_clean_with_workaround`)

| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| test_single_in_dist | 108.838 | 121.768 |
| test_geom_camber_rc | 105.324 | 102.900 |
| test_geom_camber_cruise | 66.099 (1 sample skipped) | 58.095 |
| test_re_rand | 90.170 | 85.656 |
| **test_avg** | **92.608** | **92.105** |

- **Metric summary**: `target/runs/mae-loss-metrics/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`
  (MAE loss swap is hard-coded in train.py; no extra CLI flag needed)

## Default config (the bar this PR beat)
```
Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)
AdamW(lr=5e-4, weight_decay=1e-4)
CosineAnnealingLR(T_max=epochs)
batch_size=4, surf_weight=10, epochs=50
WeightedRandomSampler over 3 domains (rc-single, rc-tandem, cruise-tandem)
loss = vol_mae + 10 · surf_mae  (in normalized space, MAE/L1 since PR #835)
```

## Metrics to report
Each PR must report from the best-val-checkpoint test evaluation:
- `test_avg/mae_surf_p` (primary)
- per-split `test/<split>/mae_surf_p`
- per-split `test/<split>/mae_vol_{Ux,Uy,p}`
- training wall-clock (min) and peak VRAM (GB)

## Notes
- Branch is `icml-appendix-charlie-pai2e-r3`; PRs target it as base; merges squash into it.
- 4 val splits = `val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`.
- Camber-holdout splits (`val_geom_camber_*`) are typically hardest — front-foil shape unseen at train time.
- Tag every PR with `charlie-pai2e-r3`.
