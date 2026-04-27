# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** No experiments completed on this branch yet. Round 1 is in flight.

## Reference configuration (the published Transolver in `train.py`)

Reproduce: `cd target && python train.py --epochs 50`

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| Batch size | 4 |
| Surf weight | 10.0 |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Loss | MSE in normalized space, `vol + surf_weight * surf` |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

## Best metrics

_None yet — first round in progress. Update after the first winner merges._

| Metric | Value | PR | Date |
|---|---|---|---|
| `val_avg/mae_surf_p` | TBD | — | — |
| `test_avg/mae_surf_p` | TBD | — | — |
