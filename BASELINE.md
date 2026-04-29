# Baseline (icml-appendix-charlie-pai2f-r1)

Round 1 of the `charlie-pai2f-r1` track. The first round-1 winner has been
merged into `train.py`: thorfinn's regime-matched schedule (PR #1101). All
round-2 experiments compare against this baseline.

## Current best (round-1 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **125.438** (epoch 14/14, still descending) | #1101 | Schedule: 1-ep warmup, T_max=13, eta_min=5e-6 (lr/100) |
| `test_avg/mae_surf_p` | **112.988** (4 splits, all finite) | #1101 | NaN-safe scoring rebase verified |

Per-split val (epoch 14):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 151.429 | 1.922 | 0.899 |
| `val_geom_camber_rc` | 132.769 | 2.756 | 1.159 |
| `val_geom_camber_cruise` | 99.904 | 1.237 | 0.627 |
| `val_re_rand` | 117.650 | 1.950 | 0.906 |
| **avg** | **125.438** | 1.966 | 0.898 |

Per-split test:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 130.679 | 1.742 | 0.874 |
| `test_geom_camber_rc` | 122.823 | 2.671 | 1.085 |
| `test_geom_camber_cruise` | 84.042 | 1.182 | 0.584 |
| `test_re_rand` | 114.410 | 1.822 | 0.859 |
| **avg** | **112.988** | 1.854 | 0.851 |

Notes:
- Best checkpoint is the **final** epoch — model still descending under the
  30-min cap. Headroom available if longer wall-clock or throughput gains are
  realized.

## Default config (`train.py` at HEAD, post-merge of #1101)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | LinearLR warmup (1 ep, 5e-7 → 5e-4) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Batch size | 4 |
| Surf weight (loss) | 10.0 |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` ≈ 14 effective epochs) |
| Sampler | WeightedRandomSampler (balanced across 3 domains) |
| Loss | MSE on normalized targets, vol + surf_weight·surf |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Params | ~0.66M |

Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
Test-time metric for paper: `test_avg/mae_surf_p`.

## Reproduce

```
cd target/ && python train.py --agent <student> --experiment_name "<student>/baseline-default"
```

(All defaults; do NOT pass `--lr`, `--batch_size`, `--surf_weight`, etc.)
