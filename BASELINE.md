<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **93.1083** | #1001 | `charliepai2e2-edward/n-head-2-wider-heads` | n_head=2 (head_dim=64); epoch 16/50 timeout; -1.76% vs PR #931 baseline |

Set by reducing attention heads from 4 to 2 (head_dim doubles from 32 to 64): wider per-head attention over the 128-dim hidden space. The best single epoch was epoch 16 (last), suggesting the curve was still falling and more epochs may help further. Improvement concentrated in `val_geom_camber_rc` (-2.91) and `val_re_rand` (-0.49), with a slight regression in `val_single_in_dist` (+0.83). Checkpoint averaging over epochs 14-15-16 (ckpt_avg K=3) slightly degrades vs best single by +0.34 — the best val is best single epoch 16. Built on top of PR #931 compound stack (AdamW, CosineAnnealingLR T_max=15, max_norm=5.0, per-sample Re-weighting).

Per-split breakdown (best single epoch 16):
- `val_single_in_dist`: mae_surf_p = 104.94
- `val_geom_camber_rc`: mae_surf_p = 102.21
- `val_geom_camber_cruise`: mae_surf_p = 74.27
- `val_re_rand`: mae_surf_p = 91.02

Per-split breakdown (ckpt_avg epochs 14-15-16):
- `val_single_in_dist`: mae_surf_p = 105.74
- `val_geom_camber_rc`: mae_surf_p = 102.58
- `val_geom_camber_cruise`: mae_surf_p = 74.57
- `val_re_rand`: mae_surf_p = 90.92
- `val_avg`: mae_surf_p = 93.45

Test metrics (best epoch 16 checkpoint):
- `test_single_in_dist`: mae_surf_p = 87.68
- `test_geom_camber_rc`: mae_surf_p = 94.77
- `test_geom_camber_cruise`: mae_surf_p = 61.94
- `test_re_rand`: mae_surf_p = 85.09
- `test_avg`: mae_surf_p = 82.37

**Reproduce:** `cd target/ && WANDB_MODE=offline python train.py --agent charliepai2e2-edward --wandb_name charliepai2e2-edward/n-head-2-wider-heads`

## Baseline Architecture (current best: PR #1001)

| Parameter | Value |
|-----------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | **2** |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-5 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `T_max` | 15 |
| `eta_min` | 1e-6 |
| `max_norm` | 5.0 |
| Per-sample Re-weighting | enabled (`w_i = 1/(log_Re - log_Re_min + 1)`) |
| `ckpt_avg K` | 3 |

## Primary Metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better).

## History

| Date | PR | val_avg/mae_surf_p | Config | Notes |
|------|----|--------------------|--------|-------|
| 2026-04-29 | #1001 | **93.1083** | n_head=2 + T_max=15 + max_norm=5.0 + per-sample Re-weighting | Epoch 16/50; best single epoch; -1.76% vs PR #931; rc/re_rand improved |
| 2026-04-28 | #931 | 94.7833 | stock + T_max=15 + max_norm=5.0 + per-sample Re-weighting | Epoch 14/50; ckpt_avg K=3; -2.81% vs prior; single_in_dist/rc improved, cruise minor regression |
| 2026-04-29 | #911 | 97.5181 | stock + T_max=15 + max_norm=5.0 | Epoch 14/50; -6.92% vs prior; cosine tail executed fully; cruise/re_rand +large, single_in_dist/rc -moderate |
| 2026-04-28 | #899 | 104.6986 | stock + clip_grad_norm=1.0 + ckpt_avg K=3 | Epoch 14/50; checkpoint average of epochs 12,13,14; tiny margin over prior baseline; technique confirmed effective |
| 2026-04-28 | #778 | 104.7457 | stock + clip_grad_norm=1.0 | Epoch 14/50; 30-min wall-clock cap; undertrained; clear win — gradient explosion was the dominant issue |
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
