<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **97.5181** | #911 | `charliepai2e2-fern/combined-lr-clip-fix` | T_max=15 + max_norm=5.0; epoch 14/50 timeout; -6.92% vs prior baseline |

Set by combining two structural fixes: (1) aligning `CosineAnnealingLR(T_max=15, eta_min=1e-6)` to the actual ~14-epoch compute budget so the cosine tail executes fully, and (2) relaxing gradient clipping from `max_norm=1.0` to `max_norm=5.0` to allow richer gradient signal from high-Re samples. The compound fix delivered a clean -7.23 absolute (-6.92%) improvement. The gains were concentrated in `val_geom_camber_cruise` (-23.2) and `val_re_rand` (-27.0); the single-foil splits (`val_single_in_dist`, `val_geom_camber_rc`) regressed by ~10 each, suggesting a capacity redistribution effect from looser clipping.

Per-split breakdown (best checkpoint, epoch 14):
- `val_single_in_dist`: mae_surf_p = 116.26
- `val_geom_camber_rc`: mae_surf_p = 107.36
- `val_geom_camber_cruise`: mae_surf_p = 75.16
- `val_re_rand`: mae_surf_p = 91.29

Test metrics (best checkpoint, epoch 14):
- `test_single_in_dist`: mae_surf_p = 103.06
- `test_geom_camber_rc`: mae_surf_p = 94.98
- `test_geom_camber_cruise`: mae_surf_p = 63.63
- `test_re_rand`: mae_surf_p = 88.38
- `test_avg`: mae_surf_p = 87.51

**Reproduce:** `cd target/ && python train.py --agent charliepai2e2-fern --wandb_name charliepai2e2-fern/combined-lr-clip-fix`

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
| 2026-04-29 | #911 | **97.5181** | stock + T_max=15 + max_norm=5.0 | Epoch 14/50; -6.92% vs prior; cosine tail executed fully; cruise/re_rand +large, single_in_dist/rc -moderate |
| 2026-04-28 | #899 | 104.6986 | stock + clip_grad_norm=1.0 + ckpt_avg K=3 | Epoch 14/50; checkpoint average of epochs 12,13,14; tiny margin over prior baseline; technique confirmed effective |
| 2026-04-28 | #778 | 104.7457 | stock + clip_grad_norm=1.0 | Epoch 14/50; 30-min wall-clock cap; undertrained; clear win — gradient explosion was the dominant issue |
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
