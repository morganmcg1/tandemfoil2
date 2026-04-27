# Baseline — `icml-appendix-willow-pai2c-r5`

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits, lower is better).

---

## 2026-04-27 20:21 — PR #227: Smooth-L1 (Huber β=1.0) surface loss

**First established baseline for this research track.**

- **`val_avg/mae_surf_p` = 112.1574** (best epoch 14 of 14, timeout-limited at 30 min)
- **Per-split val surface pressure MAE:**

| Split | `mae_surf_p` | `mae_surf_Ux` | `mae_surf_Uy` |
|---|---|---|---|
| `val_single_in_dist`     | 147.6473 | 1.4333 | 0.7497 |
| `val_geom_camber_rc`     | 112.0473 | 2.1004 | 0.9512 |
| `val_geom_camber_cruise` |  89.5626 | 1.0164 | 0.5885 |
| `val_re_rand`            |  99.3723 | 1.5758 | 0.7462 |

- **Test metrics (best checkpoint):** `test_avg/mae_surf_p = NaN` (cruise split poisoned by one sample's infinite pressure prediction; partial average over 3 finite splits = 107.7879). See cross-track learnings — NaN-guard should be added in evaluate_split for future runs.
- **W&B run:** `6bylngu8` ([link](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/6bylngu8))
- **Student:** willowpai2c5-nezuko
- **Reproduce:**
```bash
cd target/ && python train.py \
  --agent willowpai2c5-nezuko \
  --wandb_name huber-surface-loss \
  --wandb_group loss-form-sweep
```

**Architecture:** default Transolver (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`)  
**Loss:** Huber (Smooth-L1 β=1.0) on surface, MSE on volume; `surf_weight=10.0`  
**Optimizer:** AdamW `lr=5e-4, wd=1e-4`, cosine annealing `T_max=50`  
**Epochs achieved:** 14 of 50 (30-min wall-clock timeout binding)  
**Note:** Alphonse's vanilla MSE baseline (PR #184) still in flight. When it lands, if it beats 112.16, this entry will be superseded. The Huber change is small and worth keeping regardless of attribution.

---
