# Baseline Tracker — TandemFoilSet CFD Surrogate

Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## 2026-04-29 10:50 — PR #1088: Increase surf_weight from 10 to 25 for surface MAE focus

- **Student**: charliepai2f2-edward
- **Branch**: charliepai2f2-edward/surf-weight-sweep-25
- **Change**: `surf_weight: 10.0 → 25.0` (single line change in Config dataclass)

### Best Validation Metrics (epoch 13/50, 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **127.6661** |
| val_avg/mae_vol_p | 139.9394 |
| val_avg/mae_surf_Ux | 2.2548 |
| val_avg/mae_surf_Uy | 0.9431 |
| val_avg/mae_vol_Ux | 5.8663 |
| val_avg/mae_vol_Uy | 2.6935 |

Per-split surface pressure MAE:

| Split | mae_surf_p | mae_vol_p |
|-------|------------|-----------|
| val_single_in_dist | 157.82 | 178.70 |
| val_geom_camber_rc | 135.65 | 146.43 |
| val_geom_camber_cruise | 99.26 | 112.71 |
| val_re_rand | 117.94 | 121.91 |

### Test Metrics (3 of 4 splits clean; test_geom_camber_cruise NaN due to upstream corrupted GT sample)

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 137.04 |
| test_geom_camber_rc | 122.18 |
| test_geom_camber_cruise | NaN (upstream data bug) |
| test_re_rand | 117.39 |
| 3-split avg | 125.54 |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~131s/epoch)
- Peak GPU memory: 42.12 GB
- Metrics JSONL: `target/models/model-charliepai2f2-edward-surf-weight-25-20260429-095003/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=5e-4, wd=1e-4, batch=4, CosineAnnealingLR(T_max=50)

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-edward --experiment_name "charliepai2f2-edward/surf-weight-25"
# with surf_weight=25.0 set in Config dataclass
```

---

## 2026-04-29 12:20 — PR #1091: Add stochastic depth (drop_path 0→0.1) for OOD generalization

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/stochastic-depth-regularization
- **Change**: DropPath linear schedule (0.0→0.1 across 5 TransolverBlocks) + budget-aware CosineAnnealingLR (T_max estimated from warm-up timing, eta_min=1e-6) + surf_weight=25.0 + NaN-safe eval workaround

### Best Validation Metrics (epoch 13/50, 30-min timeout)

| Metric | Value | vs prior baseline |
|--------|-------|-------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **121.89** | **-5.78 (-4.5%)** |
| val_avg/mae_surf_Ux | 1.97 | — |
| val_avg/mae_surf_Uy | 0.89 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 154.41 | 130.06 |
| geom_camber_rc | 128.38 | 118.20 |
| geom_camber_cruise | 95.98 | 79.45 |
| re_rand | 108.78 | 110.64 |
| **avg** | **121.89** | **109.59** |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~135s/epoch); best at epoch 13
- Budget-aware cosine schedule: T_max=11 estimated after 2 warm-up epochs (1529s remaining / 135.4s per epoch), eta_min=1e-6
- Peak GPU memory: 42.1 GB (single H100, batch=4)
- Params: 0.66 M (DropPath adds no parameters)
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-sd-cosine-budget-aware-20260429-114039/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=5e-4, wd=1e-4, batch=4, surf_weight=25.0, DropPath(0.0→0.1 linear), budget-aware CosineAnnealingLR

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-nezuko --experiment_name "charliepai2f2-nezuko/sd-cosine-budget-aware"
# with DropPath linear schedule 0.0→0.1, budget-aware CosineAnnealingLR, surf_weight=25.0
```

---

## 2026-04-29 13:10 — PR #1098: Grad clip + higher LR (1e-3) for stable fast convergence

- **Student**: charliepai2f2-tanjiro
- **Branch**: charliepai2f2-tanjiro/grad-clip-higher-lr-rebased
- **Change**: lr 5e-4→1e-3 + grad_clip=1.0 (on top of PR #1091 baseline: DropPath 0→0.1 + budget-aware cosine + surf_weight=25 + NaN-safe eval)

### Best Validation Metrics (epoch 14/50, 30-min timeout)

| Metric | Value | vs prior baseline |
|--------|-------|-------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **100.41** | **-21.48 (-17.6%)** |
| val_avg/mae_vol_p | 120.81 | — |
| val_avg/mae_surf_Ux | 1.4978 | — |
| val_avg/mae_surf_Uy | 0.7422 | — |
| val_avg/mae_vol_Ux | 4.9283 | — |
| val_avg/mae_vol_Uy | 2.2782 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 120.68 | 104.32 |
| geom_camber_rc | 111.80 | 98.04 |
| geom_camber_cruise | 75.99 | 63.06 |
| re_rand | 93.15 | 88.91 |
| **avg** | **100.41** | **88.58** |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30); best at epoch 14
- Budget-aware cosine schedule: T_max=11 after 2-epoch warmup, eta_min=1e-6 (inherited from PR #1091)
- Peak GPU memory: 42.11 GB (single H100, batch=4)
- Params: 662,359
- Metrics JSONL: `target/models/model-charliepai2f2-tanjiro-grad-clip-higher-lr-rebased-20260429-123256/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=1e-3, wd=1e-4, batch=4, grad_clip=1.0, surf_weight=25.0, DropPath(0.0→0.1 linear), budget-aware CosineAnnealingLR

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-tanjiro --experiment_name "charliepai2f2-tanjiro/grad-clip-higher-lr-rebased" --grad_clip 1.0
# lr=1e-3 is the new default in Config; --grad_clip 1.0 required
```

---

## Notes on NaN in test_geom_camber_cruise

One corrupted GT sample (`000020.pt`) in `.test_geom_camber_cruise_gt/` has NaN in the pressure channel. `data/scoring.py:accumulate_batch` propagates this NaN because `NaN * 0.0 = NaN` in IEEE float — the mask does not fully guard it. Since `data/scoring.py` is read-only, future experiments should apply `nan_to_num()` or clamp predictions in train.py before the scoring call, or report 3-split test averages when test_geom_camber_cruise is corrupted.
