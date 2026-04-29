<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Baseline Metrics — TandemFoilSet (pai2e-r5)

## 2026-04-29 04:43 — PR #1013: n_layers=2 + n_layers=1 depth floor sweep (NEW BEST)

- **Surface MAE (n_layers=1):** val_avg/mae_surf_p = **47.7385** (best checkpoint epoch 50, final epoch)
- **Surface MAE (n_layers=2):** val_avg/mae_surf_p = **47.9717** (best checkpoint epoch 41, hit timeout)
- **val_avg/mae_surf_p (n_layers=1):** 47.7385 — **−24.3% vs previous best (63.0588)**
- **Metric summary:** `metrics/charliepai2e5-tanjiro-n-layers-1-bf16-k2glbsqy.jsonl` (n_layers=1), `metrics/charliepai2e5-tanjiro-n-layers-2-bf16-a3ikk3qi.jsonl` (n_layers=2)
- **Reproduce (n_layers=1):** `cd target/ && python train.py --n_layers 1 --bf16 --surf_weight 28.0 --lr 3e-4 --weight_decay 1e-2 --batch_size 4`
- **Reproduce (n_layers=2):** `cd target/ && python train.py --n_layers 2 --bf16 --surf_weight 28.0 --lr 3e-4 --weight_decay 1e-2 --batch_size 4`
  *(Note: `--optimizer lion --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2` are train.py defaults)*

### Per-split validation — n_layers=1 (best checkpoint, epoch 50)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | — | — |  49.6805 | — | — | — |
| val_geom_camber_rc     | — | — |  60.8209 | — | — | — |
| val_geom_camber_cruise | — | — |  30.5543 | — | — | — |
| val_re_rand            | — | — |  49.8983 | — | — | — |
| **avg**                | — | — | **47.7385** | — | — | — |

### Per-split validation — n_layers=2 (best checkpoint, epoch 41)

| Split | surf Ux | surf Uy | surf p |
|-------|--------:|--------:|-------:|
| val_single_in_dist     | — | — | 49.4117 |
| val_geom_camber_rc     | — | — | 64.3695 |
| val_geom_camber_cruise | — | — | 29.4214 |
| val_re_rand            | — | — | 48.6844 |
| **avg**                | — | — | **47.9717** |

### Model configuration (n_layers=1 winner)

| Parameter | Value |
|-----------|-------|
| **n_layers** | **1** (was 3 — key change; depth floor) |
| n_hidden | 128 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | Lion |
| lr | 3e-4 |
| weight_decay | 1e-2 |
| surf_weight | 28.0 |
| batch_size | 4 |
| loss | L1 (vol + surf_weight * surf) |
| scheduler | CosineAnnealingLR T_max=15 (cycles 3× over 50 epochs — misaligned, fix in #1014) |
| gradient clipping | clip_grad_norm max_norm=1.0 |
| EMA | decay=0.995 |
| bf16 | True |
| epochs run | 50 (9.0 GB VRAM peak — model so small it fits easily) |

**Improvement over previous baseline (PR #913):** 63.0588 → 47.7385 (−24.3%)

### Depth-shallowing full trajectory

| n_layers | val_avg/mae_surf_p |
|----------|-----------------:|
| 6 (old)  | ~70.x |
| 5        | 67.25 (PR #926) |
| 4        | 65.37 (PR #913) |
| 3        | 63.06 (PR #913) |
| 2        | 47.97 (PR #1013) |
| **1**    | **47.74 (PR #1013)** ⭐ |

---

## 2026-04-29 04:00 — PR #913: n_layers=3 + bf16 autocast on Lion+L1+clip+T_max=15+EMA+sw=28 baseline (NEW BEST)

- **Surface MAE:** Ux=0.8735, Uy=0.4147, p=**63.0588** (val_avg, best checkpoint epoch 29)
- **val_avg/mae_surf_p:** 63.0588 — **−6.22% vs previous best (67.2490)**
- **Metric summary:** `metrics/tanjiro-nlayers3-bf16-wyals8i4.jsonl`
- **Reproduce:** `cd target/ && python train.py --n_layers 3 --bf16 --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4`

### Per-split validation (best checkpoint, epoch 29)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 0.7230 | 0.3749 |  69.5556 | 3.4483 | 1.4977 |  96.2418 |
| val_geom_camber_rc     | 1.4011 | 0.6057 |  77.6902 | 4.1075 | 2.1391 |  91.8509 |
| val_geom_camber_cruise | 0.4399 | 0.2571 |  41.7779 | 2.3158 | 0.9109 |  47.9011 |
| val_re_rand            | 0.9299 | 0.4210 |  63.2113 | 3.1335 | 1.4993 |  69.7065 |
| **avg**                | **0.8735** | **0.4147** | **63.0588** | **3.2513** | **1.5118** | **76.4251** |

### Model configuration

| Parameter | Value |
|-----------|-------|
| **n_layers** | **3** (was 5 — key change) |
| n_hidden | 128 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | Lion |
| lr | 3e-4 |
| weight_decay | 1e-2 |
| surf_weight | 28.0 |
| batch_size | 4 |
| loss | L1 (vol + surf_weight * surf) |
| scheduler | CosineAnnealingLR T_max=15 |
| gradient clipping | clip_grad_norm max_norm=1.0 |
| EMA | decay=0.995 |
| **bf16** | **True** (autocast bfloat16 — key change, enables ~1.7× more epochs in budget) |
| epochs run | 29 (vs 14 in fp32 — bf16 gives more epochs within timeout) |

**Improvement over previous baseline (PR #926):** 67.2490 → 63.0588 (−6.22%)

---

## 2026-04-29 02:30 — PR #926: surf_weight=28 on Lion+T_max=15+EMA baseline

- **Surface MAE:** Ux=0.8727, Uy=0.4681, p=**67.2490** (val_avg, best checkpoint epoch 14)
- **val_avg/mae_surf_p:** 67.2490 — **−4.35% vs previous best (70.3212)**
- **Metric summary:** `metrics/charliepai2e5-alphonse-sw-sweep-28-84a7c3zn.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4`

### Per-split validation (best checkpoint, epoch 14)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 0.7106 | 0.4310 |  73.0754 | 3.9637 | 1.7670 | 109.9681 |
| val_geom_camber_rc     | 1.4350 | 0.6685 |  82.5966 | 4.8815 | 2.5525 | 100.0313 |
| val_geom_camber_cruise | 0.4319 | 0.3063 |  47.7165 | 2.6182 | 1.0685 |  60.0248 |
| val_re_rand            | 0.9131 | 0.4668 |  65.6074 | 3.6150 | 1.7411 |  79.3861 |
| **avg**                | **0.8727** | **0.4681** | **67.2490** | **3.7696** | **1.7823** | **87.3526** |

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
| **surf_weight** | **28.0** (was 20.0 — key change) |
| batch_size | 4 |
| loss | L1 (vol + surf_weight * surf) |
| scheduler | CosineAnnealingLR T_max=15 |
| gradient clipping | clip_grad_norm max_norm=1.0 |
| EMA | decay=0.995 (inherited from merged PR #801) |
| bf16 | None |

**Improvement over previous baseline (PR #801):** 70.3212 → 67.2490 (−4.35%)

---

## 2026-04-29 01:00 — PR #801: EMA model averaging (decay=0.995) for better generalization (NEW BEST)

- **Surface MAE:** Ux=0.8997, Uy=0.4874, p=**70.3212** (val_avg, best checkpoint epoch 14)
- **val_avg/mae_surf_p:** 70.3212 — **−1.36% vs previous best (71.2882)**
- **Metric summary:** `metrics/charliepai2e5-edward-ema-decay-0.995-q7m9lxrl.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`
  *(EMA decay=0.995 applied on top of Lion+L1+clip+CosineAnnealingLR T_max=15 baseline)*

### Per-split validation (best checkpoint, epoch 14)

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|-------|--------:|--------:|-------:|-------:|-------:|------:|
| val_single_in_dist     | 0.7521 | 0.4461 |  74.0550 | 3.7432 | 1.6326 | 106.1984 |
| val_geom_camber_rc     | 1.4649 | 0.6900 |  87.1819 | 5.0641 | 2.5006 | 103.4538 |
| val_geom_camber_cruise | 0.4522 | 0.3227 |  51.8277 | 2.4878 | 1.0466 |  55.9177 |
| val_re_rand            | 0.9295 | 0.4907 |  68.2204 | 3.5078 | 1.6900 |  76.9039 |
| **avg**                | **0.8997** | **0.4874** | **70.3212** | **3.7007** | **1.7174** | **85.6185** |

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
| scheduler | CosineAnnealingLR T_max=15 |
| gradient clipping | clip_grad_norm max_norm=1.0 |
| **EMA** | **decay=0.995** |
| bf16 | None |

**Improvement over previous baseline (PR #901):** 71.2882 → 70.3212 (−1.36%)

---

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
