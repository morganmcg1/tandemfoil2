# Baseline (icml-appendix-willow-r1)

The current best metrics on the `icml-appendix-willow-r1` track. Lower is better.

## Current best: vanilla Transolver baseline (`train.py` HEAD)

Round 1 of `willow` is starting from the vanilla baseline — no merged research yet on this branch. The first improvement to land becomes the new track baseline.

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | TBD (run vanilla baseline as reference) |
| `test_avg/mae_surf_p` | TBD |
| Source | `train.py` at `icml-appendix-willow-r1` HEAD |

### Vanilla configuration

- Loss: MSE, surf_weight=10
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4, cosine annealing
- Architecture: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
- Training: batch_size=4, epochs=50, 30-min wall-clock cap
- No AMP, no Fourier features, no SwiGLU

## Reference numbers from prior research (`kagent_v_students` branch — CLOSED)

For context only — these are NOT applied to `icml-appendix-willow-r1`. Provided so students know the proven winning recipe and target.

| Recipe | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| Vanilla anchor (closest analogue) | ~88 | ~78 |
| L1 loss + AMP + Fourier σ=0.7,m=160 + SwiGLU + nl=3, sn=8 | **49.443** (2-seed mean) | **42.450** |

The proven winning recipe was budget-bound — every winner hit best val at the terminal epoch.
