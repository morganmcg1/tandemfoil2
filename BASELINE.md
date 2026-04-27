# Baseline Metrics

## Current Best

- **Branch**: icml-appendix-charlie-pai2c-r3
- **PR**: #261 — Fourier PE on (x,z): spatial frequency encoding for mesh nodes — charliepai2c3-nezuko
- **val_avg/mae_surf_p**: 115.02 (best epoch 14 of 14 completed, timeout-limited at 30 min, still improving)
- **test_avg/mae_surf_p**: 104.32

**Note:** Fourier PE (Tancik et al. 2020 random Fourier features) on (x,z) node coordinates gives a
12.7% improvement in val_avg/mae_surf_p over the vanilla MSE baseline (131.71 → 115.02). Best epoch
was 14/14 — the model was still improving monotonically when the timeout hit. Future experiments
can expect even larger gains if this improvement compounds with longer training or combined changes.

All students must beat **val_avg/mae_surf_p = 115.02** to be considered a winner.

## 2026-04-27 21:21 — PR #261: Fourier PE on (x,z): spatial frequency encoding for mesh nodes

- **Surface MAE (val, best checkpoint epoch 14):**
  - Ux=1.88, Uy=0.79, p=115.02 (avg across 4 val splits)
- **Surface MAE (test):**
  - Ux=1.76, Uy=0.74, p=104.32 (avg across 4 test splits)
- **Per-split val breakdown (epoch 14):**

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist     | 129.62 | 1.79 | 0.77 |
| val_geom_camber_rc     | 124.19 | 2.58 | 0.96 |
| val_geom_camber_cruise |  94.50 | 1.29 | 0.62 |
| val_re_rand            | 111.77 | 1.87 | 0.80 |
| **val avg**            | **115.02** | **1.88** | **0.79** |

- **Per-split test breakdown:**

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist     | 115.05 | 1.60 | 0.72 |
| test_geom_camber_rc     | 115.02 | 2.48 | 0.89 |
| test_geom_camber_cruise |  77.89 | 1.23 | 0.57 |
| test_re_rand            | 109.33 | 1.71 | 0.79 |
| **test avg**            | **104.32** | **1.76** | **0.74** |

- **Metric summary:** `models/model-fourier-pe-spatial-encoding-20260427-203909/metrics.jsonl`
- **Reproduce:** `cd target/ && python train.py --agent charliepai2c3-nezuko --experiment_name fourier-pe-spatial-encoding`
- **Model config:** num_fourier_freqs=16, fourier_sigma=1.0, n_hidden=128, n_layers=5, n_head=4, slice_num=64
- **Best epoch:** 14/14 (still improving at timeout — model not converged)
- **Peak VRAM:** 42.32 GB | **Params:** 670,039 (+7.7K vs vanilla)

## 2026-04-27 19:00 — PR #193: Vanilla baseline anchor

- **Surface MAE (val, best checkpoint epoch 11):**
  - Ux=2.39, Uy=0.91, p=131.71 (avg across 4 val splits)
- **Surface MAE (test):** NaN (pre-NaN-fix run; test_geom_camber_cruise pressure NaN)
- **Per-split val breakdown (epoch 11):**

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist     | 162.74 | 2.39 | 0.91 |
| val_geom_camber_rc     | 141.71 | 2.95 | 1.20 |
| val_geom_camber_cruise | 107.43 | 1.67 | 0.66 |
| val_re_rand            | 114.97 | 2.20 | 0.93 |
| **val avg**            | **131.71** | 2.30 | 0.92 |

- **Metric summary:** `models/model-vanilla-baseline-anchor-20260427-194339/metrics.jsonl`
- **Reproduce:** `cd target/ && python train.py --agent charliepai2c3-alphonse --experiment_name vanilla-baseline-anchor`

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

All students must beat **val_avg/mae_surf_p = 131.71** to be considered a winner.
