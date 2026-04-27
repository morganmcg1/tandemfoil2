# Baseline — icml-appendix-willow-pai2c-r4

- **Track:** `icml-appendix-willow-pai2c-r4` (willow r4)
- **Last updated:** 2026-04-27 (launch)
- **Best PR:** TBD — no experiments completed yet
- **Best `val_avg/mae_surf_p`:** TBD
- **Best `test_avg/mae_surf_p`:** TBD

## Reference baseline (seeded `train.py`)

The starting point against which round-1 PRs are measured is the seeded Transolver in `train.py`:

| Hyperparameter | Value |
|---|---|
| `n_layers` | 5 |
| `n_hidden` | 128 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `surf_weight` | 10.0 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| `batch_size` | 4 |
| `epochs` (default) | 50 |
| Loss | MSE in normalized space, vol + surf_weight × surf |
| Sampler | WeightedRandomSampler (3-domain balanced) |

## Notes

- All round-1 hypotheses test single-axis modifications to this baseline so attribution is clean.
- Round-1 results will populate this file with the actual baseline run's metrics for future reference.
- Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
- Paper-facing metric: `test_avg/mae_surf_p`, computed from the best validation checkpoint.
