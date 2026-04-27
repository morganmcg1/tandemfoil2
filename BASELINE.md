# Baseline Metrics

## Current Best

- **Branch**: icml-appendix-charlie-pai2c-r3
- **PR**: #209 — EMA weight averaging (decay=0.999) — charliepai2c3-nezuko
- **val_avg/mae_surf_p**: 133.66 (epoch 14/50 — run was timeout-limited, still improving)
- **test_avg/mae_surf_p**: 119.58 (corrected, NaN-safe GT masking applied)

**Note:** This is the first experiment on this track. The EMA model includes a critical
bug fix for a NaN-poisoning issue in `evaluate_split` affecting `test_geom_camber_cruise`
(1 sample with 761 NaN GT pressure values). The fix is now part of `train.py` on the advisor
branch and all future experiments will inherit it.

**Note:** The run timed out after 14/50 epochs (~30 min wall clock, ~132 s/epoch) with
val_avg/mae_surf_p still monotonically decreasing. The model had not converged. Comparison
against a vanilla baseline without EMA is not yet available.

## 2026-04-27 18:20 — PR #209: EMA weight averaging (decay=0.999)

- **Surface MAE (val, best checkpoint epoch 14):**
  - Ux=2.21, Uy=0.86, p=133.66 (avg across 4 val splits)
- **Surface MAE (test, NaN-corrected):**
  - Ux=2.07, Uy=0.82, p=119.58 (avg across 4 test splits)
- **Per-split val breakdown (epoch 14):**

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist     | 171.74 | 2.02 | 0.88 |
| val_geom_camber_rc     | 146.90 | 3.18 | 1.12 |
| val_geom_camber_cruise | 100.14 | 1.46 | 0.60 |
| val_re_rand            | 115.87 | 2.17 | 0.84 |
| **val avg**            | **133.66** | 2.21 | 0.86 |

- **Per-split test breakdown (NaN-corrected):**

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist     | 143.91 | 1.92 | 0.81 |
| test_geom_camber_rc     | 132.09 | 3.05 | 1.06 |
| test_geom_camber_cruise |  85.50 | 1.32 | 0.55 |
| test_re_rand            | 116.84 | 1.99 | 0.83 |
| **test avg**            | **119.58** | 2.07 | 0.82 |

- **Metric summary:** `target/models/model-charliepai2c3-nezuko-ema-weight-averaging-20260427-192048/metrics.jsonl`
- **Reproduce:** `cd target/ && python train.py --agent charliepai2c3-nezuko --experiment_name ema-weight-averaging`

## Baseline Model Config

```python
model_config = dict(
    space_dim=2,
    fun_dim=22,   # X_DIM - 2 = 24 - 2
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

## Baseline Hyperparameters

```
lr=5e-4
weight_decay=1e-4
batch_size=4
surf_weight=10.0
epochs=50
optimizer=AdamW
scheduler=CosineAnnealingLR (T_max=epochs)
EMA decay=0.999 (applied to model weights; all val/test evaluated from EMA model)
```

## Notes

The primary metric is `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE
across four validation splits). Lower is better.

A critical NaN-masking bug fix is included in `train.py` (merged from PR #209). It
handles the single NaN-GT sample in `test_geom_camber_cruise` by sanitizing GT and
masking predictions at padding positions before accumulating losses. This fix is a
no-op on all val splits and 3 of 4 test splits.

All students must beat **val_avg/mae_surf_p = 133.66** to be considered a winner.
