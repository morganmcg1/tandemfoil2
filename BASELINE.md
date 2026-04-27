# Baseline — icml-appendix-willow-pai2c-r4

- **Track:** `icml-appendix-willow-pai2c-r4` (willow r4)
- **Last updated:** 2026-04-27 21:15
- **Best PR:** #217 — H9: OneCycleLR warmup + gradient clipping (Variant A, max_lr=1e-3)
- **Best `val_avg/mae_surf_p`:** 117.62
- **Best `test_avg/mae_surf_p`:** NaN (cruise split bug — 3-split finite avg: 114.89)

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

## 2026-04-27 21:15 — PR #217: H9: OneCycleLR warmup + gradient clipping

- **Student:** willowpai2c4-frieren
- **W&B run:** `jumwxlx7` (Variant A: max_lr=1e-3, pct_start=0.10, grad_clip=1.0)
- **Best epoch:** 14 / 50 (hit 30-min SENPAI_TIMEOUT_MINUTES cap)

| Split | val `mae_surf_p` | test `mae_surf_p` |
|---|---:|---:|
| `single_in_dist` | 140.28 | 124.35 |
| `geom_camber_rc` | 132.30 | 119.57 |
| `geom_camber_cruise` | 85.04 | NaN (cruise bug) |
| `re_rand` | 112.87 | 100.75 |
| **avg** | **117.62** | **NaN** (3-split finite avg: 114.89) |

- **Surface MAE:** Ux=2.0714, Uy=0.8161, p=117.62
- **Reproduce:**
```bash
cd "target/" && python train.py --epochs 50 --max_lr 1e-3 --pct_start 0.10 --grad_clip_norm 1.0 \
    --agent willowpai2c4-frieren --wandb_group h9-onecycle \
    --wandb_name "willowpai2c4-frieren/h9-peak1e3-w10"
```

**Notes:** Run completed only 14/50 epochs (30-min timeout). OneCycleLR was mid-anneal at truncation (lr ≈ 9e-4, past peak). `test_geom_camber_cruise/mae_surf_p` is NaN due to a pre-existing scoring bug where non-finite model predictions are not skipped before accumulation. The 3 finite test splits average 114.89. This is the first completed experiment and sets the round-1 baseline.

---

## Notes

- All round-1 hypotheses test single-axis modifications to this baseline so attribution is clean.
- Round-1 results will populate this file with the actual baseline run's metrics for future reference.
- Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
- Paper-facing metric: `test_avg/mae_surf_p`, computed from the best validation checkpoint.
