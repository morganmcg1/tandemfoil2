# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #400 (charliepai2d3-nezuko) — **L1 surface loss + 8-frequency
Fourier positional features for `(x, z)`**, all other knobs at the
unmodified Transolver defaults (`bs=4`, `lr=5e-4`, `n_hidden=128`,
`n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`,
`surf_weight=10`, cosine T_max=50, `weight_decay=1e-4`).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **91.87** |
| `test_avg/mae_surf_p` (NaN-safe, best-val checkpoint) | **81.11** |
| Per-epoch wallclock | ~131 s |
| Peak GPU memory (batch=4) | 42.38 GB |
| Epochs completed before 30-min timeout | 14 / 50 |
| Param count | 670,551 (+8,192 from FF input MLP) |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 117.24 |
| val_geom_camber_rc     |  98.99 |
| val_geom_camber_cruise |  68.61 |
| val_re_rand            |  82.64 |
| **val_avg**            | **91.87** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 100.17 |
| test_geom_camber_rc     |  85.47 |
| test_geom_camber_cruise |  61.17 |
| test_re_rand            |  77.64 |
| **test_avg**            | **81.11** |

Reproduce:

```bash
cd target/
python train.py --experiment_name baseline_ref
```

(Both L1 surface loss and 8-frequency Fourier positional features are
baked into `train.py`. The scoring fix is on the advisor branch since
commit `2eb5c7f` — `test_avg/*_p` lands as a clean number.)

## Round 3 progress

| Round | Best val_avg/mae_surf_p | Best test_avg/mae_surf_p | Lever | Δ vs prior |
|-------|------------------------:|-------------------------:|-------|----:|
| Pre-r3 | TBD (no measured pre-r3 baseline on this branch) | — | — | — |
| PR #306 (merged) | 135.20 | 123.15 | bs=8, sqrt LR (MSE) | reference |
| PR #280 (merged) | 102.64 | 97.73 | + L1 surface loss | **−24.1%** val |
| **PR #400 (merged, current)** | **91.87** | **81.11** | **+ 8-freq spatial Fourier features** | **−10.5%** val, **−17.0%** test |

Notes:

- Both runs used `bs=4` defaults. PR #306 used `bs=8` MSE — that knob did
  *not* survive the L1 transition (PR #390 closed: bs=8 + L1 lost +16%
  because L1 already absorbs the bs=8 noise-reduction effect).
- All round-3 runs hit the 30-min timeout at epoch 14 of a 50-epoch
  cosine. Cosine T_max is mismatched to actual budget — round-4 PR #389
  is testing `--epochs 14` to fully decay the schedule.
- `val_single_in_dist` is now the dominant bottleneck at 117.24 (vs
  68.61 cruise camber, 82.64 re_rand, 98.99 rc camber). The win pattern
  on PR #400 *widened* the gap rather than closing it: in-dist gained
  only 3.3% while OOD-camber gained 6-21%. Round-5 priorities should
  target the high-Re raceCar single regime specifically.

## Reference (unmodified Transolver) configuration

Defaults baked into `train.py` after PR #280 + PR #400 merges:

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `fun_dim` | `X_DIM - 2 + 4 * NUM_FOURIER_FREQS` = 22 + 32 = **54** |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, **MSE volume + L1 surface** |
| Input encoding | raw 24-d `x` + 8-frequency Fourier of `(x, z)` |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Epochs | 50 |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
