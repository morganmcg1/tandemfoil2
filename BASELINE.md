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

## Notes on NaN in test_geom_camber_cruise

One corrupted GT sample (`000020.pt`) in `.test_geom_camber_cruise_gt/` has NaN in the pressure channel. `data/scoring.py:accumulate_batch` propagates this NaN because `NaN * 0.0 = NaN` in IEEE float — the mask does not fully guard it. Since `data/scoring.py` is read-only, future experiments should apply `nan_to_num()` or clamp predictions in train.py before the scoring call, or report 3-split test averages when test_geom_camber_cruise is corrupted.
