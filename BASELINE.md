# Baseline — `icml-appendix-charlie-pai2-r1`

Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface pressure MAE across the 4 validation splits, lower is better). Test counterpart: `test_avg/mae_surf_p`.

## Current best (Round 1 — vanilla)

The fresh `icml-appendix-charlie-pai2-r1` track starts from the literal vanilla `train.py` defaults committed at `9985312` ("Strengthen Charlie local metric logging"). No experiment has merged into this branch yet, so the entry below is the as-shipped configuration; the first calibration PR will replace these `TBD` numbers with measured deltas.

| Field | Value |
|-------|-------|
| Source | vanilla `train.py` (no PR) |
| `val_avg/mae_surf_p` | TBD — H1 calibration (alphonse) |
| `test_avg/mae_surf_p` | TBD — H1 calibration (alphonse) |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Activation | GELU (FFN inside `TransolverBlock`) |
| Loss | MSE (per-node squared error) |
| `surf_weight` | 10.0 |
| Optimizer | AdamW (`lr=5e-4`, `weight_decay=1e-4`) |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| `batch_size` | 4 |
| `epochs` | 50 |
| Augmentation | none |
| AMP | none |
| Positional encoding | none (raw `(x, z)` ∈ feature dims 0–1) |

## Prior round reference (for context only — NOT applied to this branch)

The previous `icml-appendix-charlie` round terminated at `val_avg/mae_surf_p ≈ 49.08` / `test ≈ 42.47` on a stack of `n_layers=3 + slice_num=8 + Fourier PE σ=0.7 + SwiGLU FFN`. None of those PRs are merged here. Round 1 of this track re-validates the highest-confidence prior levers as orthogonal single-knob tests *and* introduces fresh angles (EMA, per-channel decoders, Huber loss, AMP throughput) so the round produces both verified deltas and new signal.
