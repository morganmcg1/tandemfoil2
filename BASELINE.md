# TandemFoilSet Baseline — icml-appendix-willow-pai2c-r2

**Last updated:** PR #194 (willowpai2c2-thorfinn — Huber loss delta=0.5)

## Current Best Configuration

```
n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
Optimizer: AdamW + CosineAnnealingLR
Loss: Huber (delta=0.5) for both vol_loss and surf_loss (normalized space)
```

## Current Best Metrics (PR #194, W&B run: sl8j1hlp)

| Metric | val_avg | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand |
|--------|---------|--------------------|--------------------|------------------------|-------------|
| mae_surf_p | **100.40** | 125.63 | 107.12 | 76.03 | 92.81 |

| Test Metric | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand |
|-------------|---------------------|---------------------|------------------------|--------------|
| mae_surf_p | 111.54 | 97.51 | NaN* | 88.47 |

*test_geom_camber_cruise is NaN due to a pre-existing scoring bug with sample 000020.pt (non-finite p ground truth). All other splits are clean.

## Original MSE Baseline (PR #185, 3-seed anchor)

| Metric | val_avg (mean ± std) | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand |
|--------|---------|--------------------|--------------------|------------------------|-------------|
| mae_surf_p | 130.67 ± 3.28 | ~162 | ~140 | ~99 | ~120 |

W&B runs: cz83xuzc (seed0), ao4sksim (seed1), vrphtsf4 (seed2)

Seed variance: ±3.28 on val_avg, ±2.93 on test_avg (≈2.5% of mean — need ≥3% improvement to be convincing with 1-seed run).

## Improvement History

| PR | Description | val_avg/mae_surf_p | Delta vs MSE baseline |
|----|-------------|-------------------|-----------------------|
| #185 | MSE baseline anchor (3-seed mean) | 130.67 | — |
| #191 | FiLM conditioning on log(Re)+NACA | 117.65 | −9.96 (−8.6%) |
| #194 | Huber loss delta=0.5 | **100.40** | −30.27 (−22.0%) |
