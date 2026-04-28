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
