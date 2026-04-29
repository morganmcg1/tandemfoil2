# Baseline — TandemFoilSet CFD Surrogate (icml-appendix-charlie-pai2e-r4)

Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## 2026-04-28 20:00 — PR #738: Surface loss weight: 10 → 20 to prioritize surface MAE

- **Branch:** charliepai2e4-edward/higher-surf-weight-20
- **Best epoch:** 13 of 14 (30-min timeout — model still converging)
- **Surface MAE (val, best ckpt):** Ux=2.4441, Uy=0.8943, **p=128.8320**
- **Volume MAE (val, best ckpt):** Ux=5.8291, Uy=2.5823, p=145.0063
- **val_avg/mae_surf_p: 128.8320** ← current best
- **Metric summary:** `target/metrics/charliepai2e4-edward-higher-surf-weight-20-wnnqnvav.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 2.2478 | 0.9177 | 157.1632 | 6.5801 | 2.7364 | 183.4634 |
| val_geom_camber_rc     | 3.0704 | 1.0829 | 136.6349 | 6.1857 | 3.2678 | 140.8892 |
| val_geom_camber_cruise | 1.9694 | 0.7030 | 100.2357 | 4.9168 | 1.8728 | 125.3909 |
| val_re_rand            | 2.4887 | 0.8736 | 121.2942 | 5.6338 | 2.4523 | 130.2818 |
| **avg**                | **2.4441** | **0.8943** | **128.8320** | **5.8291** | **2.5823** | **145.0063** |

### Notes
- First merged experiment on this track — establishes the baseline.
- `surf_weight=20` (doubled from default 10) pushes surface accuracy below volume accuracy on every split (avg surf/vol ratio 0.889).
- `test_geom_camber_cruise` pressure is NaN (single-sample overflow in scoring pipeline — not specific to this config).
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, cosine over 50 epochs.

---

## 2026-04-28 — PR #812: Lower LR 5e-4 → 2e-4 with surf_weight=20 for smoother convergence

- **Branch:** charliepai2e4-edward/lower-lr-2e-4-surf-weight-20
- **Best epoch:** 14 of 14 (30-min timeout — best=last suggests further gains with more time)
- **Surface MAE (val, best ckpt):** Ux=1.8782, Uy=0.7963, **p=112.9366**
- **Volume MAE (val, best ckpt):** Ux=5.2301, Uy=2.6142, p=132.0728
- **val_avg/mae_surf_p: 112.9366** ← current best (was 128.8320, **−12.3%**)
- **Metric summary:** `metrics/charliepai2e4-edward-lower-lr-2e-4-surf-weight-20-5vwlbqdz.jsonl`
- **Reproduce:** `cd target/ && python train.py --lr 2e-4 --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.6891 | 0.8207 | 132.7534 | 5.3934 | 2.6561 | 165.0441 |
| val_geom_camber_rc     | 2.4081 | 0.9279 | 125.5523 | 5.7861 | 3.1474 | 129.0143 |
| val_geom_camber_cruise | 1.5603 | 0.6291 |  91.5413 | 4.6879 | 2.0143 | 116.7482 |
| val_re_rand            | 1.8552 | 0.8074 | 101.9034 | 5.0529 | 2.6389 | 117.5847 |
| **avg**                | **1.8782** | **0.7963** | **112.9366** | **5.2301** | **2.6142** | **132.0728** |

### Notes
- Reducing LR from 5e-4 to 2e-4 with `surf_weight=20` gave a 12.3% improvement on the primary metric.
- Best epoch = last epoch: model was still converging at the 30-min wall-clock cap.
- Cosine annealing `T_max` set to actual achievable epochs (~14), not MAX_EPOCHS=50.
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.

---

## 2026-04-28 — PR #735: Wider hidden dim 192, deeper 6 layers, surf_weight=20 (alphonse)

- **Branch:** charliepai2e4-alphonse/wider-hidden-256 (fork: morganmcg1/TandemFoilSet-Balanced)
- **Best epoch:** 9 of ~11 achievable
- **Surface MAE (val, best ckpt):** Ux=2.0889, Uy=0.9002, **p=128.3833**
- **Volume MAE (val, best ckpt):** Ux=5.6798, Uy=2.5484, p=138.1776
- **val_avg/mae_surf_p: 128.3833** — beat old PR #738 baseline (128.8320, −0.35%) but below current best PR #812 (112.9366)
- **Metric summary:** `metrics/charliepai2e4-alphonse-wider192-deep6-sw20-tmax11-qme35ium.jsonl`
- **Reproduce:** `cd target/ && python train.py --n_hidden 192 --n_layers 6 --surf_weight 20.0`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 2.0038 | 0.9508 | 156.3134 | 6.4149 | 2.7374 | 181.2448 |
| val_geom_camber_rc     | 2.7272 | 1.0610 | 143.9932 | 5.9975 | 3.1286 | 137.1337 |
| val_geom_camber_cruise | 1.6257 | 0.6789 |  97.4374 | 4.8069 | 1.8280 | 122.9547 |
| val_re_rand            | 1.9991 | 0.9101 | 115.7893 | 5.5000 | 2.4998 | 111.3773 |
| **avg**                | **2.0889** | **0.9002** | **128.3833** | **5.6798** | **2.5484** | **138.1776** |

### Notes
- Widening hidden dim from 128 to 192 and adding a 6th layer beat the very first baseline (PR #738: 128.83) by 0.35%.
- However, this run predates the LR reduction in PR #812 — combining wider architecture with lr=2e-4 is a strong candidate for further improvement.
- PR closed manually (metrics cherry-picked): fork branch had structural conflicts due to divergent repo layout (train.py at root vs target/).
- Best epoch was not the last — model had headroom; best=epoch 9 of 11.
- Architecture: `n_hidden=192`, `n_layers=6`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `surf_weight=20`, cosine T_max=11.

---

## 2026-04-28 23:00 — PR #845: Combine per-sample norm loss with surf_weight=20 for additive gain

- **Branch:** charliepai2e4-fern/per-sample-norm-loss-plus-surf-weight-20
- **Best epoch:** 14 of 14 (final epoch best — smooth monotonic descent through full cosine anneal)
- **Surface MAE (val, best ckpt):** Ux=1.5402, Uy=0.7110, **p=105.9649**
- **Volume MAE (val, best ckpt):** Ux=4.7621, Uy=2.2293, p=120.9768
- **val_avg/mae_surf_p: 105.9649** ← current best (was 112.9366, **−6.2%**)
- **Metric summary:** `metrics/charliepai2e4-fern-per-sample-norm-loss-plus-surf-weight-20-0b1lixif.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0 --epochs 14`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.5439 | 0.7060 | 130.9864 | 5.5467 | 2.3723 | 159.9004 |
| val_geom_camber_rc     | 2.1869 | 0.9438 | 118.4032 | 5.6667 | 2.9503 | 125.0687 |
| val_geom_camber_cruise | 0.9228 | 0.4906 |  79.1174 | 3.5359 | 1.4811 |  94.2555 |
| val_re_rand            | 1.5073 | 0.7034 |  95.3524 | 4.2992 | 2.1133 | 104.6828 |
| **avg**                | **1.5402** | **0.7110** | **105.9649** | **4.7621** | **2.2293** | **120.9768** |

### Notes
- Per-sample normalized loss (divides each sample's MSE by per-sample target variance) combined with `surf_weight=20`.
- These two changes address orthogonal imbalances: sample-level Re-regime rebalancing (per-sample norm) + node-class surface emphasis (surf_weight).
- Combination stacks additively: per-sample norm alone at sw=10 gave 110.37 (PR #747); sw=20 alone gave 128.83 (PR #738); combined gives 105.96.
- `T_max=14` properly calibrated to run length — LR fully annealed to 0 at epoch 14 (vs PR #747 T_max=50 leaving LR high at cutoff).
- Loss tag in JSONL config: `loss_kind: "per_sample_norm_mse"`.
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`.
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `surf_weight=20.0`, cosine T_max=14.
- `test_geom_camber_cruise` p NaN is pre-existing scoring pipeline issue (same as PRs #738, #747).

---

## 2026-04-29 00:22 — PR #868: Combine per-sample norm loss with lr=2e-4 for additive gain

- **Branch:** edward/per-sample-norm-plus-lr-2e-4
- **Best epoch:** 14 of 14 (final epoch best — model still improving through full cosine anneal)
- **Surface MAE (val, best ckpt):** Ux=1.5446, Uy=0.7109, **p=102.9421**
- **Volume MAE (val, best ckpt):** Ux=4.8035, Uy=2.2793, p=117.7792
- **val_avg/mae_surf_p: 102.9421** ← current best (was 105.9649, **−2.85%**)
- **Metric summary:** `metrics/charliepai2e4-edward-per-sample-norm-plus-lr-2e-4-c37s8dpo.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0 --lr 2e-4 --loss_kind per_sample_norm_mse --epochs 14`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.4580 | 0.6938 | 129.4600 | 5.4347 | 2.4817 | 158.3350 |
| val_geom_camber_rc     | 2.2424 | 0.9548 | 111.1449 | 5.7793 | 3.0469 | 123.6362 |
| val_geom_camber_cruise | 0.9653 | 0.4930 |  78.0567 | 3.6348 | 1.4672 |  87.6694 |
| val_re_rand            | 1.5129 | 0.7020 |  93.1066 | 4.3652 | 2.1212 | 101.4761 |
| **avg**                | **1.5446** | **0.7109** | **102.9421** | **4.8035** | **2.2793** | **117.7792** |

### Notes
- Confirms per-sample norm + lr=2e-4 + T_max=14 compose constructively over PR #845.
- Gain was modest (−2.85%) not the hypothesised 6–10%, indicating the two optimizations share some gradient-magnitude-reduction mechanism and are not fully orthogonal.
- `val_single_in_dist` remains the dominant bottleneck at 129.46 — ~31% of the average.
- Best epoch = last epoch: model still improving, suggests more epochs could push lower within the timeout budget.
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, 0.66M params.
- Optimizer: AdamW, `lr=2e-4`, `weight_decay=1e-4`, `surf_weight=20.0`, cosine T_max=14.
- `test_geom_camber_cruise` p NaN is pre-existing scoring pipeline issue.

---

## 2026-04-29 — PR #871: Grad-clip + lr=2e-4 + per-sample norm: triple stabilization

- **Branch:** askeladd/grad-clip-per-sample-norm-lr2e4
- **Best epoch:** 14 of 14 (final epoch best — model still improving, clipping active 100% of batches)
- **Surface MAE (val, best ckpt):** Ux=1.4046, Uy=0.6882, **p=95.6617**
- **Volume MAE (val, best ckpt):** Ux=4.4974, Uy=2.1069, p=112.3271
- **val_avg/mae_surf_p: 95.6617** ← current best (was 102.9421, **−7.1%** — sub-100 achieved)
- **Metric summary:** `metrics/charliepai2e4-askeladd-grad-clip-lr2e4-per-sample-norm-surf20-ufu7itlb.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0 --lr 2e-4 --grad_clip 1.0 --loss_kind per_sample_norm_mse --epochs 14`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.3427 | 0.6899 | 115.2536 | 5.1963 | 2.2326 | 148.4179 |
| val_geom_camber_rc     | 2.0870 | 0.9032 | 104.6421 | 5.3562 | 2.8916 | 120.9540 |
| val_geom_camber_cruise | 0.7879 | 0.4812 |  73.5144 | 3.3225 | 1.3328 |  81.4751 |
| val_re_rand            | 1.4008 | 0.6785 |  89.2369 | 4.1147 | 1.9706 |  98.4614 |
| **avg**                | **1.4046** | **0.6882** | **95.6617** | **4.4974** | **2.1069** | **112.3271** |

### Notes
- Gradient clipping (max_norm=1.0) confirmed critical: pre-clip norms were 35–105× above max_norm throughout training (grad_clip_frac=1.0 every epoch).
- Achieves the sub-100 stretch goal: val_avg/mae_surf_p = 95.66 < 100.
- Best epoch = last epoch: model still improving at epoch 14 — T_max=14 LR schedule fully annealed to 0.
- Pre-clip grad norm declines monotonically: 105.7 (ep1) → 36.8 (ep14) — clipping reduces gradient scale steadily.
- Test results: `test_geom_camber_cruise` p NaN (pre-existing scoring pipeline overflow); `test_single_in_dist` p=99.98; `test_geom_camber_rc` p=92.41; `test_re_rand` p=85.15.
- Architecture unchanged: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, 662K params.
- Optimizer: AdamW, `lr=2e-4`, `weight_decay=1e-4`, `surf_weight=20.0`, `grad_clip=1.0`, cosine T_max=14.

---

## 2026-04-29 — PR #935: Extended epochs=18, T_max=18 on current best: per_sample_norm_mse + lr=2e-4

- **Branch:** charliepai2e4-frieren/extended-epochs-18-per-sample-norm-lr-2e-4
- **Best epoch:** 14 of 18 (30-min wall-clock timeout; best=last — model still improving at cutoff)
- **Surface MAE (val, best ckpt):** Ux=1.2913, Uy=0.6843, **p=94.6918**
- **Volume MAE (val, best ckpt):** Ux=4.3653, Uy=1.9829, p=110.2781
- **val_avg/mae_surf_p: 94.6918** ← current best (was 95.6617, **−1.01%**)
- **Metric summary:** `target/metrics/charliepai2e4-frieren-extended-epochs-18-per-sample-norm-lr-2e-4-alsxfigk.jsonl`
- **Reproduce:** `cd target/ && python train.py --surf_weight 20.0 --lr 2e-4 --grad_clip 1.0 --weight_decay 1e-4 --epochs 18`

### Per-split breakdown

| Split | surf Ux | surf Uy | surf p | vol Ux | vol Uy | vol p |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 1.2984 | 0.6816 | 109.0267 | 4.9698 | 2.1102 | 144.2117 |
| val_geom_camber_rc     | 1.9591 | 0.8680 | 108.3369 | 5.1580 | 2.7046 | 115.3989 |
| val_geom_camber_cruise | 0.7416 | 0.4688 |  73.1648 | 3.1743 | 1.2715 |  79.3267 |
| val_re_rand            | 1.1660 | 0.7089 |  88.2387 | 4.1590 | 1.8454 |  92.1751 |
| **avg**                | **1.2913** | **0.6843** | **94.6918** | **4.3653** | **1.9829** | **110.2781** |

### Notes
- Architecture change: spatial_bias input dimension 2D→4D (`nn.Linear(4, 32)` and `raw_xy = x[:,:,:4]`), adding saf_0 and saf_1 airfoil shape coordinates for richer spatial context.
- **Key gain mechanism:** T_max=18 with epochs≤18 means the LR has not fully decayed to 0 at epoch 14 (timeout cutoff). PR #871 used T_max=14, which parks LR≈0 at epoch 14. The longer cosine cycle keeps a positive LR through the timeout, yielding continued gradient flow.
- Best epoch = last epoch: model still improving at 30-min wall-clock cap — further gains expected with longer budget.
- `test_geom_camber_cruise` p NaN is pre-existing scoring pipeline issue (degenerate target std → div-by-zero).
- Architecture: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`, spatial_bias 4D.
- Optimizer: AdamW, `lr=2e-4`, `weight_decay=1e-4`, `surf_weight=20.0`, `grad_clip=1.0`, cosine T_max=18.
