# TandemFoilSet Baseline — icml-appendix-willow-pai2c-r2

**Status:** Pending — alphonse's baseline anchor (PR #185) will populate this file.

## Default Transolver Configuration

```
n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50
Optimizer: AdamW + CosineAnnealingLR
Loss: MSE (vol_loss + surf_weight * surf_loss, normalized space)
```

## Baseline Metrics (TBD after PR #185)

| Metric | val_avg | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand |
|--------|---------|--------------------|--------------------|------------------------|-------------|
| mae_surf_p | TBD | TBD | TBD | TBD | TBD |
| mae_surf_Ux | TBD | — | — | — | — |
| mae_surf_Uy | TBD | — | — | — | — |

| Test Metric | test_avg | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand |
|-------------|---------|---------------------|--------------------|------------------------|--------------|
| mae_surf_p | TBD | TBD | TBD | TBD | TBD |

W&B run: TBD — see PR #185.

Seed variance: TBD (3-seed run, expected ± ~1-2%).
