# Baseline — icml-appendix-charlie-pai2e-r3

## Current Best
- **Source**: Multi-scale RFF (σ₁=1.0, σ₂=5.0, num_freq=32) on normalized (x,z) coords + EMA decay=0.99 + per-sample Re-aware RMS loss normalization (PR #928, frieren)
- **PR**: #928
- **Primary**: `val_avg/mae_surf_p` = **83.338** (lower is better)
- **Test (3-split mean, excl. cruise NaN)**: `test_avg/mae_surf_p` = **82.635**

### Best checkpoint metrics (val, best EMA checkpoint, epoch 13)

| Split | mae_surf_p | Δ vs PR #895 |
|---|---|---|
| val_single_in_dist | 99.354 | -4.59% |
| val_geom_camber_rc | 93.856 | -3.79% |
| val_geom_camber_cruise | 60.324 | -7.11% |
| val_re_rand | 79.819 | -3.02% |
| **val_avg** | **83.338** | **-4.46%** |

### Test metrics (best-val checkpoint, epoch 13; EMA weights; NaN in cruise due to known 1-sample bug)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 91.795 |
| test_geom_camber_rc | 84.131 |
| test_geom_camber_cruise | NaN (1-sample GT bug) |
| test_re_rand | 71.978 |
| **test_avg (3-split excl. cruise)** | **82.635** |

- **Metric summary**: `target/runs/multi-scale-rff-normalized/training_metrics.jsonl`
- **Reproduce**: `cd target/ && WANDB_MODE=offline EMA_DECAY=0.99 python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`
  (Multi-scale RFF σ₁=1.0, σ₂=5.0, num_freq=32, normalized coords, seed=42; EMA decay=0.99; per-sample Re-aware RMS normalization; Cosine T_max=15 + 1-epoch warmup)

## Previous Best (PR #895)
- **Source**: EMA of model weights (decay=0.99) for evaluation (PR #895, thorfinn)
- **PR**: #895
- **Primary**: `val_avg/mae_surf_p` = **87.233** (lower is better)
- **Test (3-split mean, excl. cruise NaN)**: `test_avg/mae_surf_p` = **85.166**

### Best checkpoint metrics (val, epoch 13)

| Split | mae_surf_p | Δ vs PR #919 |
|---|---|---|
| val_single_in_dist | 104.130 | -0.81% |
| val_geom_camber_rc | 97.553 | +2.14% |
| val_geom_camber_cruise | 64.943 | -2.11% |
| val_re_rand | 82.304 | -1.56% |
| **val_avg** | **87.233** | **-0.44%** |

### Test metrics (raw, NaN in cruise due to known 1-sample bug)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 92.469 |
| test_geom_camber_rc | 86.491 |
| test_geom_camber_cruise | NaN (1-sample GT bug) |
| test_re_rand | 76.539 |
| **test_avg (3-split excl. cruise)** | **85.166** |

- **Metric summary**: `target/runs/ema-decay-0.99/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`
  (EMA decay=0.99, per-sample Re-aware RMS normalization, T_max=15 + 1-epoch warmup)

## Previous Best (PR #919)
- **Source**: Per-sample Re-aware loss normalization (PR #919, fern)
- **PR**: #919
- **Primary**: `val_avg/mae_surf_p` = **87.614** (lower is better)
- **Test (3-split mean, excl. cruise NaN)**: `test_avg/mae_surf_p` = **84.461**

### Best checkpoint metrics (val, epoch 14)

| Split | mae_surf_p | Δ vs PR #889 |
|---|---|---|
| val_single_in_dist | 104.985 | -11.13% |
| val_geom_camber_rc | 95.516 | -4.76% |
| val_geom_camber_cruise | 66.346 | -6.66% |
| val_re_rand | 83.608 | -5.05% |
| **val_avg** | **87.614** | **-7.18%** |

### Test metrics (raw, NaN in cruise due to known 1-sample bug)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 93.332 |
| test_geom_camber_rc | 83.844 |
| test_geom_camber_cruise | NaN (1-sample GT bug) |
| test_re_rand | 76.206 |
| **test_avg (3-split excl. cruise)** | **84.461** |

- **Metric summary**: `target/runs/re-aware-sample-loss-weighting/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`
  (Per-sample Re-aware RMS normalization merged in train.py; T_max=15 + 1-epoch warmup)

## Previous Best (PR #889)
- **Source**: Cosine T_max=15 + 1-epoch warmup for 30-min budget (PR #889, fern)
- **PR**: #889
- **Primary**: `val_avg/mae_surf_p` = **94.387** (lower is better)
- **Test (3-split mean, excl. cruise NaN)**: `test_avg/mae_surf_p` = **92.232**

### Best checkpoint metrics (val, epoch 14)

| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist | 118.130 | — |
| val_geom_camber_rc | 100.284 | — |
| val_geom_camber_cruise | 71.079 | — |
| val_re_rand | 88.053 | — |
| **val_avg** | **94.387** | — |

### Test metrics (raw, NaN in cruise due to known 1-sample bug)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 106.075 |
| test_geom_camber_rc | 89.965 |
| test_geom_camber_cruise | NaN (1-sample GT bug) |
| test_re_rand | 80.655 |
| **test_avg (3-split excl. cruise)** | **92.232** |

- **Metric summary**: `target/runs/cosine-tmax-fix-warmup/metrics.jsonl`
- **Reproduce**: `cd target/ && python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`
  (T_max=15 + 1-epoch warmup hard-coded in train.py since PR #889)

## Previous Best (PR #835)
- **Primary**: `val_avg/mae_surf_p` = **104.058**
- **Test (corrected, NaN-sample skipped)**: `test_avg/mae_surf_p` = **92.608**

### Best checkpoint metrics (val, from reeval_summary)

| Split | mae_surf_Ux | mae_surf_Uy | mae_surf_p | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist | 1.718 | 0.723 | 124.701 | 134.457 |
| val_geom_camber_rc | 2.189 | 0.885 | 116.841 | 114.777 |
| val_geom_camber_cruise | 1.669 | 0.431 | 76.934 | 67.139 |
| val_re_rand | 1.973 | 0.644 | 97.756 | 91.352 |
| **val_avg** | **1.887** | **0.671** | **104.058** | **101.931** |

- **Reproduce**: `cd target/ && python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`

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
