# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #306 (charliepai2d3-thorfinn) — `batch_size=8`, `lr=7.07e-4` (√2-scaled),
all other defaults. First run on the round-3 advisor branch; established
the measured baseline.

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 13/14) | **135.20** |
| `test_avg/mae_surf_p` (NaN-safe re-eval, best-val checkpoint) | **123.15** |
| Per-epoch wallclock | ~129 s |
| Peak GPU memory (batch=8) | 84.2 GB |
| Epochs completed before 30-min timeout | 14 / 50 |

Per-split val (best epoch 13):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 190.14 |
| val_geom_camber_rc     | 138.39 |
| val_geom_camber_cruise |  97.95 |
| val_re_rand            | 114.32 |
| **val_avg**            | **135.20** |

Per-split test (corrected, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 173.01 |
| test_geom_camber_rc     | 120.22 |
| test_geom_camber_cruise |  82.83 |
| test_re_rand            | 116.53 |
| **test_avg**            | **123.15** |

Reproduce:

```bash
cd target/
python train.py --batch_size 8 --lr 7.07e-4 --experiment_name baseline_ref
```

Notes for round 3 in-flight PRs:

- The current baseline was measured under **truncated cosine** — only
  ~14 of 50 scheduled epochs ran, so the cosine LR never reached its tail.
  PRs that match the same wallclock cap will face the same truncation;
  improvements should still be visible *relative* to this baseline.
- `data/scoring.py` was patched on the advisor branch (commit `2eb5c7f`)
  to fix `Inf*0=NaN` poisoning of `test_avg/*_p`. PRs branched off the
  pre-fix advisor will still produce `NaN` test_avg in their on-disk
  metrics; they can be evaluated either by rebasing or by recomputing
  test from the saved checkpoint with the patched scorer.

## Reference (unmodified Transolver) configuration

Defaults from `train.py` (kept for future ablations):

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, MSE |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Epochs | 50 |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
