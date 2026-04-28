<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — icml-appendix-charlie-pai2e-r2

## Current Best

| Metric | Value | PR | Branch | Notes |
|--------|-------|----|--------|-------|
| `val_avg/mae_surf_p` | **137.0013** | #764 | `charliepai2e2-alphonse/larger-model-capacity` | epoch 9/50 only (30-min timeout); model still in steep descent (-17/epoch); **undertrained** |

This is the first measured number for this track. It was set with n_hidden=256 (4× stock capacity). The model had not converged — val loss was still dropping at ~17 units/epoch at cutoff. This number will be superseded once the in-flight wave completes fuller runs.

**Effective working baseline for second-wave assignment:** treat 137.0 as a soft floor to beat, knowing it reflects undertrained n_hidden=256, not converged stock architecture.

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
| 2026-04-28 | #764 | 137.0013 | n_hidden=256 | Epoch 9/50; 30-min wall-clock cap; undertrained; first measured number |
