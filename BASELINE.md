# TandemFoilSet Baseline — willow-pai2e-r5

Lower `val_avg/mae_surf_p` is better.

---

## 2026-04-29 01:55 — PR #850: Lower surf_weight 10→3 on Huber+BF16 stack

- **val_avg/mae_surf_p:** 101.563 ← **current best**
- **val per-split surf p MAE:**
  - `val_single_in_dist`: 120.51
  - `val_geom_camber_rc`: 107.95
  - `val_geom_camber_cruise`: 82.16
  - `val_re_rand`: 95.64
- **test_avg/mae_surf_p:** 89.918 ← **best clean 4-split test**
- **test per-split surf p MAE:**
  - `test_single_in_dist`: 102.846
  - `test_geom_camber_rc`: 94.352
  - `test_geom_camber_cruise`: 70.128
  - `test_re_rand`: 92.346
- **Best epoch:** 17 of 17 (still descending at 30-min timeout)
- **Model:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (663K params + distance features)
- **Training:** `lr=1e-3 (peak), warmup_epochs=5, warmup_start_lr=1e-4, eta_min=1e-6, weight_decay=1e-4, batch_size=4, surf_weight=3.0, epochs=50, amp=True, amp_dtype=bf16, huber_delta=1.0`
- **Changes vs prior:** `surf_weight` lowered from 10.0 to 3.0. Lower surface weight forces volume residual to inform attention more, exploiting the global pressure-Poisson relationship.
- **W&B run:** `6rh7dzkx`
- **Reproduce:**
  ```bash
  cd target/ && python train.py --agent willowpai2e5-edward \
    --amp --amp_dtype bf16 \
    --surf_weight 3.0 \
    --wandb_name "willowpai2e5-edward/sw3-huber" \
    --wandb_group lower-surf-weight-huber-stack
  ```

---

## 2026-04-28 23:40 — PR #739: Replace MSE with Huber loss (delta=1.0) for high-Re outlier robustness

- **val_avg/mae_surf_p:** 110.594
- **val per-split surf p MAE:**
  - `val_single_in_dist`: 130.87
  - `val_geom_camber_rc`: 115.14
  - `val_geom_camber_cruise`: 92.61
  - `val_re_rand`: 103.76
- **test_avg/mae_surf_p:** 101.299 ← **best clean 4-split test**
- **test per-split surf p MAE:**
  - `test_single_in_dist`: 124.544
  - `test_geom_camber_rc`: 99.385
  - `test_geom_camber_cruise`: 80.195
  - `test_re_rand`: 101.070
- **Best epoch:** 16 of 17 completed (30-min timeout; val still descending at cutoff)
- **Throughput:** 33.1 GB peak VRAM (same as BF16 baseline)
- **Model:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (663K params + distance features)
- **Training:** `lr=1e-3 (peak), warmup_epochs=5, warmup_start_lr=1e-4, eta_min=1e-6, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, amp=True, amp_dtype=bf16, huber_delta=1.0`
- **Changes vs prior:** Huber loss (δ=1.0) replacing MSE in both training loop and evaluate_split. Stacked on BF16 + distance features + warmup+cosine baseline.
- **W&B run:** `l95azbnv`
- **Reproduce:**
  ```bash
  cd target/ && python train.py --agent willowpai2e5-frieren \
    --amp --amp_dtype bf16 \
    --huber_delta 1.0 \
    --wandb_name "willowpai2e5-frieren/huber-loss-d1.0-rebased" \
    --wandb_group huber-loss
  ```

---

## 2026-04-28 21:29 — PR #811: Enable bf16 mixed precision for 1.5-2x training throughput

- **val_avg/mae_surf_p:** 127.402
- **val per-split surf p MAE:**
  - `val_single_in_dist`: 151.791
  - `val_geom_camber_rc`: 147.898
  - `val_geom_camber_cruise`: 93.729
  - `val_re_rand`: 116.189
- **test_avg/mae_surf_p:** 116.211 ← **best clean 4-split test**
- **test per-split surf p MAE:**
  - `test_single_in_dist`: 141.142
  - `test_geom_camber_rc`: 134.121
  - `test_geom_camber_cruise`: 79.094
  - `test_re_rand`: 110.488
- **Best epoch:** 17 of 50 (30-min timeout; 1.20× speedup vs fp32 baseline)
- **Throughput:** 110.02 s/epoch (vs 131.96 s fp32); peak VRAM 33.1 GB (63 GB headroom)
- **Model:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (663K params + distance features)
- **Training:** `lr=1e-3 (peak), warmup_epochs=5, warmup_start_lr=1e-4, eta_min=1e-6, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, amp=True, amp_dtype=bf16`
- **Changes vs prior:** BF16 autocast on forward+loss; no GradScaler (bf16 safe dynamic range); pred cast to fp32 for eval. Zero NaN/Inf events.
- **W&B run:** `newqt8dd`
- **Reproduce:**
  ```bash
  cd target/ && python train.py --agent willowpai2e5-askeladd \
    --amp --amp_dtype bf16 \
    --wandb_name "willowpai2e5-askeladd/bf16-mixed-precision" \
    --wandb_group mixed-precision-bf16
  ```

---

## 2026-04-28 20:25 — PR #737: Add 5-epoch linear warmup + peak lr=1e-3 before cosine decay

- **val_avg/mae_surf_p:** 127.872
- **val per-split surf p MAE:**
  - `val_single_in_dist`: 149.241
  - `val_geom_camber_rc`: 146.033
  - `val_geom_camber_cruise`: 96.362
  - `val_re_rand`: 119.852
- **test_avg/mae_surf_p:** NaN (dataset bug in cruise sample 20; 3-split avg ≈ 123.59)
- **Best epoch:** 14 of 50 (30-min timeout; model still improving steeply at cutoff)
- **Model:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (663K params + distance features)
- **Training:** `lr=1e-3 (peak), warmup_epochs=5, warmup_start_lr=1e-4, eta_min=1e-6, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50`
- **Changes vs prior:** Warmup+cosine LR schedule with peak lr=1e-3 (stacked on top of PR #763 features + NaN-safe eval)
- **W&B run:** `5b22tecz`
- **Reproduce:**
  ```bash
  cd target/ && python train.py --agent willowpai2e5-fern \
    --wandb_name "willowpai2e5-fern/lr-warmup-cosine-peak1e-3" \
    --wandb_group lr-warmup-cosine
  ```

---

## 2026-04-28 20:15 — PR #763: Add physics-informed distance features to Transolver input

- **val_avg/mae_surf_p:** 141.4181
- **val per-split surf p MAE:**
  - `val_single_in_dist`: 177.0157
  - `val_geom_camber_rc`: 157.4591
  - `val_geom_camber_cruise`: 105.7864
  - `val_re_rand`: 125.4113
- **test_avg/mae_surf_p:** 126.5598
- **test per-split surf p MAE:**
  - `test_single_in_dist`: 148.3098
  - `test_geom_camber_rc`: 145.5500
  - `test_geom_camber_cruise`: 91.0172
  - `test_re_rand`: 121.3623
- **Best epoch:** 12 of 50 (30-min timeout hit at epoch 13)
- **Model:** `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (663K params)
- **Training:** `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50`
- **Changes vs prior:** Added 3 derived input features (dist_to_surface, log_dist_to_surface, is_tandem); AUG_X_DIM=27; NaN-safe evaluate_split fix for dataset bug in test_geom_camber_cruise sample 20.
- **W&B run:** `072wo9xb`
- **Reproduce:**
  ```bash
  cd target/ && python train.py --agent willowpai2e5-thorfinn \
    --wandb_name "willowpai2e5-thorfinn/feature-engineering-dist-tandem" \
    --wandb_group feature-engineering
  ```
