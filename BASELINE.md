# Baseline (icml-appendix-charlie-pai2f-r1)

Two winners merged into `train.py`:
- **PR #1101 (thorfinn)** — regime-matched schedule (warmup=1, T_max=13, eta_min=lr/100)
- **PR #1138 (frieren)** — Random Fourier Features on (x, z), n_freq=32, sigma=1.0

All subsequent experiments compare against this stacked baseline.

## Current best (round-2 winner — merged 2026-04-29 12:42)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **108.543** (epoch 14/14, still descending) | #1138 | RFF on (x,z) at n_freq=32, sigma=1.0 |
| `test_avg/mae_surf_p` | **96.942** (4 splits, all finite) | #1138 | All-finite paper-facing test metric |

Per-split val (epoch 14):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 125.815 | 1.686 | 0.750 |
| `val_geom_camber_rc` | 114.589 | 2.385 | 0.956 |
| `val_geom_camber_cruise` | 86.371 | 1.289 | 0.585 |
| `val_re_rand` | 107.397 | 1.797 | 0.775 |
| **avg** | **108.543** | 1.789 | 0.766 |

Per-split test:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 104.402 | 1.612 | 0.698 |
| `test_geom_camber_rc` | 106.273 | 2.367 | 0.908 |
| `test_geom_camber_cruise` | 74.043 | 1.196 | 0.541 |
| `test_re_rand` | 103.048 | 1.654 | 0.753 |
| **avg** | **96.942** | 1.707 | 0.725 |

Notes:
- frieren's RFF run was launched on the pre-#1101-merge train.py. After the
  squash merge, the merged train.py has **RFF + schedule together** for the
  first time — next runs may show further compounding gains below 108.5.
- Best checkpoint is the **final** epoch — model still descending under the
  30-min cap.

## Improvement chain

| Stage | val_avg | test_avg | PR |
|---|---|---|---|
| Provisional round-1 best (confounded) | 133.892 | 132.106 (3 finite) | #1095 (sent back) |
| Round-1 winner: regime-matched schedule | 125.438 | 112.988 | #1101 ← merged |
| Round-2 winner: RFF (on top of schedule) | **108.543** | **96.942** | #1138 ← merged |

Round-1→Round-2 cumulative improvement: **-19.0% on val, -14.2% on test**.

## Default config (`train.py` at HEAD, post-merge of #1138)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | LinearLR warmup (1 ep, 5e-7 → 5e-4) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Batch size | 4 |
| Surf weight (loss) | 10.0 |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` ≈ 14 effective epochs) |
| Sampler | WeightedRandomSampler (balanced across 3 domains) |
| Loss | MSE on normalized targets, vol + surf_weight·surf |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **RFF on (x,z) n_freq=32 sigma=1.0** |
| Params | ~0.68M (RFF preprocess MLP slightly wider than baseline 0.66M) |

Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
Test-time metric for paper: `test_avg/mae_surf_p`.

## Reproduce

```
cd target/ && python train.py --agent <student> --experiment_name "<student>/baseline-default"
```

(All defaults; do NOT pass `--lr`, `--batch_size`, `--surf_weight`, etc.)
