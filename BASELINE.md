# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #287 (surf_weight=25) is the current best and the de facto round-1 baseline.

> **Round-1 budget caveat.** `SENPAI_TIMEOUT_MINUTES=30` is binding for every run on the published Transolver. The 50-epoch cosine schedule is set up by `train.py` but training stops at ~14 epochs in practice, well before the schedule's tail. Round 1 is therefore a 14-epoch ranking exercise. Comparisons across PRs in round 1 are apples-to-apples *only* if they hit the same wall-clock limit.

## Current best (PR #287, alphonse, merged 2026-04-27)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **126.67** | 14 / 50 (timeout-capped) |
| `test_avg/mae_surf_p` | **114.88** | best ckpt (epoch 14), 1 NaN-y test sample skipped |
| Wall-clock | 30.8 min train + 20 s test eval | |
| Peak GPU memory | 42.1 GB | |

### Per-split val (epoch 14)
| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 155.79 | 2.23 | 0.94 | 172.08 |
| val_geom_camber_rc     | 134.23 | 3.02 | 1.10 | 147.34 |
| val_geom_camber_cruise |  98.89 | 1.70 | 0.61 | 131.81 |
| val_re_rand            | 117.77 | 2.17 | 0.86 | 133.50 |

### Per-split test (best ckpt, post-fix scoring)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 136.43 |
| test_geom_camber_rc     | 124.14 |
| test_geom_camber_cruise |  83.63 |
| test_re_rand            | 115.33 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50 --surf_weight 25.0`

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50) |
| Batch size | 4 |
| **Surf weight** | **25.0** (raised from the published default of 10.0) |
| Epochs (configured / completed) | 50 / ~14 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Loss | MSE in normalized space, `vol + surf_weight * surf` |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-surf-weight-25-20260427-225335/metrics.jsonl`

## Reference configuration (published `train.py` defaults)

Reproduce: `cd target && python train.py --epochs 50`

The published config differs only in `--surf_weight 10.0` (default). No round-1 experiment has run a *direct* baseline at the 30-min cap with surf_weight=10, so we do not have a separately-measured number for the published default on this branch. PR #287 is the cleanest baseline available; future round-1 PRs should beat 126.67 on val_avg/mae_surf_p.

## Update history

| PR | val_avg/mae_surf_p | Notes |
|---|---|---|
| #287 (merged) | **126.67** | First baseline. surf_weight=25, 14/50 epochs, timeout-capped. |
