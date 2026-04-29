<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **94.7833** | #931 | `charliepai2e2-fern/per-sample-adaptive-clip` | Per-sample Re-weighted loss (downweight high-Re gradients); epoch 14/50 timeout; -2.81% vs prior baseline |

Set by per-sample Re-weighting: loss for each sample is divided by its own `log(Re)` before averaging within a batch, so high-Re samples (which produce large raw gradients) contribute proportionally less than low-Re samples. The improvement was concentrated in `val_single_in_dist` (-11.35) and `val_geom_camber_rc` (-1.87), with a small regression in `val_geom_camber_cruise` (+2.16) and near-neutral `val_re_rand` (+0.10). Checkpoint averaging over epochs 12-13-14 (ckpt_avg) added -2.09 vs best single epoch (epoch 13 = 96.87). Built on top of PR #911 (T_max=15 + max_norm=5.0).

Per-split breakdown (ckpt_avg epochs 12-13-14):
- `val_single_in_dist`: mae_surf_p = 104.91
- `val_geom_camber_rc`: mae_surf_p = 105.49
- `val_geom_camber_cruise`: mae_surf_p = 77.32
- `val_re_rand`: mae_surf_p = 91.41

Test metrics (ckpt_avg epochs 12-13-14):
- `test_single_in_dist`: mae_surf_p = 96.86
- `test_geom_camber_rc`: mae_surf_p = 93.28
- `test_geom_camber_cruise`: mae_surf_p = 64.54
- `test_re_rand`: mae_surf_p = 86.18
- `test_avg`: mae_surf_p = 85.22

**Reproduce:** `cd target/ && python train.py --agent charliepai2e2-fern --wandb_name charliepai2e2-fern/per-sample-adaptive-clip`

## Baseline Architecture (stock Transolver from train.py)

| Parameter | Value |
|-----------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-5 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |

## Primary Metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 validation splits (lower is better).

## History

| Date | PR | val_avg/mae_surf_p | Config | Notes |
|------|----|--------------------|--------|-------|
| 2026-04-28 | #931 | **94.7833** | stock + T_max=15 + max_norm=5.0 + per-sample Re-weighting | Epoch 14/50; ckpt_avg K=3; -2.81% vs prior; single_in_dist/rc improved, cruise minor regression |
| 2026-04-29 | #911 | 97.5181 | stock + T_max=15 + max_norm=5.0 | Epoch 14/50; -6.92% vs prior; cosine tail executed fully; cruise/re_rand +large, single_in_dist/rc -moderate |
| 2026-04-28 | #899 | 104.6986 | stock + clip_grad_norm=1.0 + ckpt_avg K=3 | Epoch 14/50; checkpoint average of epochs 12,13,14; tiny margin over prior baseline; technique confirmed effective |
| 2026-04-28 | #778 | 104.7457 | stock + clip_grad_norm=1.0 | Epoch 14/50; 30-min wall-clock cap; undertrained; clear win — gradient explosion was the dominant issue |
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
