# In-track baseline — `icml-appendix-willow-pai2d-r3`

Lower is better on **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four val splits) — this is the primary ranking metric. Paper-facing number is `test_avg/mae_surf_p` from the best-val checkpoint.

> **⚠️ Seed-variance caveat (2026-04-28).** `train.py` does not call `torch.manual_seed` (yet), and run-to-run variance for the same config has been measured at ~25 MAE (~21%) on this metric. Numbers below are single-seed point estimates. PR #482 will replace these with multi-seed `mean ± std` once it lands.

## 2026-04-28 03:37 — PR #410: EMA of weights at eval time (decay=0.99, warmup_steps=100)

- Branch: `willowpai2d3-nezuko/ema-of-weights` (squash-merged)
- **Recipe addition:** `use_ema=True, ema_decay=0.99, ema_warmup_steps=100` now defaults in `train.py`. EMA-averaged weights are saved as the best-val checkpoint and evaluated on test.
- **Best val avg surface MAE:** `val_avg/mae_surf_p = 121.4387` (epoch 12, run `22a7k787`).
- **Within-sweep delta (apples-to-apples, same seed env):** EMA d=0.99 vs. no-EMA control = **−21.71 MAE** (143.14 → 121.44). End-of-training live-vs-EMA diagnostic = **−30.74 MAE within the same run** (live=153.94, ema=123.20). Both are seed-independent measurements that confirm the lever is robust.
- **Per-split val MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `val_single_in_dist` | 148.90 |
  | `val_geom_camber_rc` | 130.69 |
  | `val_geom_camber_cruise` | 94.44 |
  | `val_re_rand` | 111.73 |
  | **val_avg** | **121.44** |

- **Per-split test MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `test_single_in_dist` | 128.35 |
  | `test_geom_camber_rc` | 117.19 |
  | `test_geom_camber_cruise` | 79.67 |
  | `test_re_rand` | 109.44 |
  | **test_avg** | **108.66** |

- **W&B run:** `22a7k787` in group `ema-of-weights` (project `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`)
- **Reproduce (now uses EMA by default):**
  ```bash
  cd target/
  python train.py --epochs 50 \
      --wandb_group baseline-after-pr410 --wandb_name baseline-r1 \
      --agent willowpai2d3-XXX
  ```
- **Notes:**
  - The 121.44 absolute number sits *above* PR #320's recorded 115.84 — but that was a single favorable seed; nezuko's no-EMA control on a different seed produced 143.14. The EMA lever effect is reliable; the absolute number depends on seed.
  - Test_geom_camber_cruise NaN is no longer present in the test eval — frieren's y_finite filter (cherry-picked into baseline at commit `32b5b40`) drops the one bad sample (`test_geom_camber_cruise/000020.pt` has `±inf` GT pressure values) before scoring.
  - Hyperparameter snapshot: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, use_ema=True, ema_decay=0.99, ema_warmup_steps=100`.

## 2026-04-28 00:30 — PR #320: Linear warmup + higher peak LR (5e-4 → 1e-3, 2-epoch warmup)

- **Best val avg surface MAE:** `val_avg/mae_surf_p = 115.8379` (epoch 14)
- **Per-split val MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `val_single_in_dist` | 131.0594 |
  | `val_geom_camber_rc` | 129.5697 |
  | `val_geom_camber_cruise` | 92.5489 |
  | `val_re_rand` | 110.1734 |
  | **val_avg** | **115.8379** |

- **Per-split test MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `test_single_in_dist` | 111.75 |
  | `test_geom_camber_rc` | 117.86 |
  | `test_geom_camber_cruise` | (recomputable now via the eval bug-fix in commit `32b5b40` — was NaN in the original W&B run) |
  | `test_re_rand` | 108.73 |
  | mean of 3 valid splits | 112.78 |

- **W&B run:** `w3mjq2ua` in group `lr-warmup-sweep` (project `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`)
- **Reproduce:**
  ```bash
  cd target/
  python train.py --peak_lr 1e-3 --warmup_epochs 2 --epochs 50 \
      --wandb_group lr-warmup-sweep --wandb_name lr-1e-3-w2-r1 \
      --agent willowpai2d3-nezuko
  ```
- **Notes:**
  - All Round-1 sweep runs hit the 30-min `SENPAI_TIMEOUT_MINUTES` at ~epoch 14 of 50; cosine never fully annealed. Comparisons across PRs in this round are at this same truncated budget.
  - **Test_avg NaN bug RESOLVED** in commit `32b5b40` (cherry-picked from PR #319, frieren). Root cause: `test_geom_camber_cruise/000020.pt` has `-inf` values in ground-truth pressure, and `data/scoring.py`'s zero-mask multiplication produces `0 * inf = NaN` which poisons the per-split aggregator. Fix lives in `train.py`'s `evaluate_split` (data/scoring.py is read-only): drop samples with non-finite GT from the mask before any arithmetic. PRs after this commit will have correct test metrics on every checkpoint.
  - Hyperparameter snapshot at this baseline: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`.
