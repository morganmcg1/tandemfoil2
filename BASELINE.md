# TandemFoilSet Baseline — willow-pai2e-r5

Lower `val_avg/mae_surf_p` is better.

---

## 2026-04-28 20:25 — PR #737: Add 5-epoch linear warmup + peak lr=1e-3 before cosine decay

- **val_avg/mae_surf_p:** 127.872 ← **current best**
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
