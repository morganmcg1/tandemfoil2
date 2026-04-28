<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — TandemFoilSet (pai2e-r5)

## 2026-04-28 23:59 — PR #901: Cosine LR T_max budget align: T_max 50→15 (NEW BEST)

- **Surface MAE:** Ux=0.9077, Uy=0.4877, p=**71.2882** (val_avg, best checkpoint epoch 13)
- **val_avg/mae_surf_p:** 71.2882 — **−7.78% vs previous best (77.2954)**
- **Metric summary:** `metrics/charliepai2e5-askeladd-cosine-tmax-15-9b1s4s0x.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`
  *(One-line change: `CosineAnnealingLR(T_max=15)` instead of `T_max=MAX_EPOCHS=50`)*

### Per-split validation (best checkpoint, epoch 13)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 0.7788 | 0.4462 |  79.4120 | 4.0014 | 1.7347 | 108.5898 |
| val_geom_camber_rc     | 1.4460 | 0.6702 |  83.1787 | 4.6980 | 2.4878 |  99.8770 |
| val_geom_camber_cruise | 0.4725 | 0.3372 |  54.1816 | 2.6017 | 1.0577 |  57.2435 |
| val_re_rand            | 0.9296 | 0.4974 |  68.3805 | 3.4593 | 1.7086 |  77.6605 |
| **avg**                | **0.9077** | **0.4877** | **71.2882** | **3.6901** | **1.7472** | **85.8427** |

### Model configuration

| Parameter | Value |
|-----------|-------|
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | Lion |
| lr | 3e-4 |
| weight_decay | 1e-2 |
| surf_weight | 20.0 |
| batch_size | 4 |
| loss | L1 (vol + surf_weight * surf) |
| scheduler | **CosineAnnealingLR T_max=15** (aligned to ~14-epoch timeout budget) |
| gradient clipping | clip_grad_norm max_norm=1.0 |
| EMA | None |
| bf16 | None |

**Improvement over previous baseline (PR #799):** 77.2954 → 71.2882 (−7.78%)

---

## 2026-04-28 23:00 — PR #799: Lion optimizer + L1 loss + gradient clipping

- **Surface MAE:** Ux=1.3128, Uy=0.5040, p=**77.2954** (val_avg)
- **val_avg/mae_surf_p:** 77.2954 (best checkpoint, epoch 14/50 — timeout-bound, still descending)
- **Metric summary:** `metrics/charliepai2e5-askeladd_lion-l1-clip-plhsfvbu.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`
  *(Note: train.py dataclass defaults updated to lr=3e-4, weight_decay=1e-2 for Lion; loss=L1; clip_grad_norm max_norm=1.0)*

### Per-split validation (best checkpoint, epoch 14)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 1.3596 | 0.4770 |  92.0183 | 4.1521 | 1.6284 | 100.6224 |
| val_geom_camber_rc     | 1.6130 | 0.6790 |  87.7708 | 4.6558 | 2.2368 | 102.6982 |
| val_geom_camber_cruise | 1.0149 | 0.3605 |  57.9690 | 2.9002 | 1.0248 |  63.2044 |
| val_re_rand            | 1.2637 | 0.4993 |  71.4235 | 3.6930 | 1.5748 |  79.8195 |
| **avg**                | **1.3128** | **0.5040** | **77.2954** | **3.8503** | **1.6162** | **86.5861** |

### Model configuration

| Parameter | Value |
|-----------|-------|
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | **Lion** |
| lr | **3e-4** |
| weight_decay | **1e-2** |
| surf_weight | 20.0 |
| batch_size | 4 |
| loss | **L1** (vol + surf_weight * surf) |
| scheduler | CosineAnnealingLR T_max=MAX_EPOCHS |
| gradient clipping | **clip_grad_norm max_norm=1.0** |
| EMA | None |
| bf16 | None |

**Improvement over previous baseline (PR #798):** 97.4483 → 77.2954 (−20.68%)

---

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
