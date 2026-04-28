<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — TandemFoilSet (pai2e-r5)

## 2026-04-28 21:00 — PR #798: L1 loss: align training objective with MAE metric

- **Surface MAE:** Ux=1.3095, Uy=0.5908, p=**97.4483** (val_avg)
- **val_avg/mae_surf_p:** 97.4483 (best checkpoint, epoch 14/50)
- **Metric summary:** `metrics/charliepai2e5-alphonse-l1-loss-2dl6j00h.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`

### Per-split validation (best checkpoint, epoch 14)

| Split | surf Ux | surf Uy | surf p |
|-------|--------:|--------:|-------:|
| val_single_in_dist     | — | — | 126.6157 |
| val_geom_camber_rc     | — | — | 110.4532 |
| val_geom_camber_cruise | — | — |  65.8819 |
| val_re_rand            | — | — |  86.8424 |
| **avg**                | **1.3095** | **0.5908** | **97.4483** |

**Improvement over previous baseline:** 128.8320 → 97.4483 (−24.4%)

---

## Previous Best: PR #738 — surf_weight=20 (charliepai2e4-edward/higher-surf-weight-20)

**Primary metric: `val_avg/mae_surf_p` = 128.8320** (best checkpoint, epoch 13/14) — *superseded by PR #798*

### Model configuration (baseline)

| Parameter | Value |
|-----------|-------|
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| surf_weight | **20.0** (was 10.0 — upgraded in PR #738) |
| batch_size | 4 |
| loss | MSE (vol + surf_weight * surf) |
| scheduler | CosineAnnealingLR T_max=MAX_EPOCHS |
| EMA | None |
| bf16 | None |
| gradient clipping | None |

### Validation metrics (best checkpoint, epoch 13)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 2.2478 | 0.9177 | 157.1632 | 6.5801 | 2.7364 | 183.4634 |
| val_geom_camber_rc     | 3.0704 | 1.0829 | 136.6349 | 6.1857 | 3.2678 | 140.8892 |
| val_geom_camber_cruise | 1.9694 | 0.7030 | 100.2357 | 4.9168 | 1.8728 | 125.3909 |
| val_re_rand            | 2.4887 | 0.8736 | 121.2942 | 5.6338 | 2.4523 | 130.2818 |
| **avg**                | **2.4441** | **0.8943** | **128.8320** | **5.8291** | **2.5823** | **145.0063** |

### Reproduce command

```bash
cd target/ && python train.py --surf_weight 20.0
```

Note: Run time was 30.59 min for 14 epochs (timeout at 30 min). Full-epoch baseline on more epochs would
show continued improvement.
