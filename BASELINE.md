# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #308 (EMA decay=0.999 + grad clip max_norm=1.0) is the current best.

> **Round-1 budget caveat.** `SENPAI_TIMEOUT_MINUTES=30` is binding for every run. The 50-epoch cosine schedule is set up by `train.py` but training stops at ~14 epochs in practice, well before the schedule's tail. Round 1 is therefore a 14-epoch ranking exercise. Comparisons across PRs in round 1 are apples-to-apples *only* if they hit the same wall-clock limit.

## Current best (PR #308, nezuko, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **106.40** (EMA-evaluated) | 13 / 50 (timeout-capped at 14) |
| `test_avg/mae_surf_p` | **93.99** (EMA-evaluated) | best ckpt = epoch 13 |
| Wall-clock | ~33 min total (~141 s/epoch) | |
| Peak GPU memory | 42.1 GB | |

### Per-split val (epoch 13, EMA weights)
| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 130.44 | 1.53 | 0.74 | 133.99 |
| val_geom_camber_rc     | 119.63 | 2.37 | 0.98 | 120.78 |
| val_geom_camber_cruise |  80.75 | 1.00 | 0.51 |  74.44 |
| val_re_rand            |  94.78 | 1.63 | 0.74 |  91.91 |

### Per-split test (best EMA checkpoint, post-fix scoring)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 112.78 |
| test_geom_camber_rc     | 103.87 |
| test_geom_camber_cruise |  66.35 |
| test_re_rand            |  92.98 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50` (the EMA + grad-clip changes are in `train.py` from PR #308).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50) |
| Batch size | 4 |
| Surf weight | 10.0 (note: this run used the *published* default, not 25.0 from #287) |
| Epochs (configured / completed) | 50 / ~13-14 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Loss | MSE in normalized space, `vol + surf_weight * surf` |
| **EMA** | decay=0.999; eval + test use EMA weights |
| **Grad clip** | max_norm=1.0 (on `clip_grad_norm_`; **fires on 100% of batches** — acting as implicit lr dampener, not outlier suppression) |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-charliepai2d4-nezuko-ema999-gradclip1-20260427-232213/metrics.jsonl`

> **Note on attribution.** The 16% gain over PR #287 cannot be cleanly attributed to EMA alone — `max_norm=1.0` clips every batch, so the optimizer is doing essentially unit-norm SGD. Ablations are queued: EMA-only at decay=0.995, clip-only at max_norm=10, etc.

## Compoundable wins still on the table

PR #287 (surf_weight=25) was merged independently before #308 landed; the artifact files are in `models/model-surf-weight-25-20260427-225335/`. **The two changes are orthogonal** — combining surf_weight=25 with EMA+clip is a likely round-2 candidate.

## Update history

| PR | val_avg/mae_surf_p | Notes |
|---|---|---|
| #287 (merged) | 126.67 | surf_weight 10→25, 14/50 epochs, timeout-capped. |
| #308 (merged) | **106.40** | EMA(0.999) + grad clip 1.0, 13/50 epochs, EMA-evaluated. **-16.2% vs #287.** |
| #372 (merged, infrastructure) | 108.93 (no EMA, surf_weight=25, 19 epochs) | **bf16 autocast** on the model forward in train + eval. Equal-config win: -14% vs #287 (same surf_weight=25, no EMA) at 19 vs 14 epochs. **Canonical baseline metric remains #308's 106.40 (EMA-evaluated).** Bf16 is now in `train.py` for all future runs; future PRs implicitly include bf16+EMA+clip+scoring fix. |
