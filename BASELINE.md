# Baseline Metrics

## Current Best

- **Branch**: icml-appendix-charlie-pai2c-r3
- **PR**: None (initial baseline — no experiments run yet on this track)
- **val_avg/mae_surf_p**: TBD (awaiting first experiment results)
- **test_avg/mae_surf_p**: TBD (awaiting first experiment results)

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
```

## Notes

This is the initial baseline Transolver configuration from `train.py`.
All students should beat this baseline. The primary metric is `val_avg/mae_surf_p`
(equal-weight mean surface pressure MAE across four validation splits).
Lower is better.
