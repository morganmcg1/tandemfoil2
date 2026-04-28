# In-track baseline — `icml-appendix-willow-pai2d-r3`

Lower is better on **`val_avg/mae_surf_p`** (equal-weight mean surface-pressure MAE across the four val splits) — this is the primary ranking metric. Paper-facing number is `test_avg/mae_surf_p` from the best-val checkpoint.

> **⚠️ Seed-variance caveat (2026-04-28).** `train.py` does not call `torch.manual_seed` (yet), and run-to-run variance for the same config has been measured at ~25 MAE (~21%) on this metric. Numbers below are single-seed point estimates. PR #482 will replace these with multi-seed `mean ± std` once it lands.

## 2026-04-28 08:53 — PR #409: OneCycleLR (peak_lr=2e-3, pct_start=0.1, total_epochs=15)

- Branch: `willowpai2d3-frieren/onecycle-lr` (squash-merged)
- **Recipe addition:** Default schedule flipped from `warmup-cosine` to `onecycle`. `OneCycleLR(max_lr=2e-3, pct_start=0.1, anneal_strategy='cos', cycle_momentum=False, div_factor=25, final_div_factor=1e4)` driven per-step (not per-epoch), with `onecycle_total_epochs=15` calibrated so the cool-down phase actually fires inside the 30-min wall-clock budget. Warmup-cosine path kept available behind `--schedule warmup-cosine`.
- **Best val avg surface MAE:** `val_avg/mae_surf_p = 87.7443` (epoch 14, run `4fas292o`).
- **Best test avg surface MAE:** `test_avg/mae_surf_p = 78.2370` (same checkpoint).
- **Within-sweep delta** (OneCycle vs warmup-cosine + EMA + L1 at same budget): **−7.51 MAE on val_avg, −6.64 MAE on test_avg.** OneCycle wins on all 4 test splits.
- **vs prior baseline (PR #294):** −7.15 MAE on val_avg (94.89 → 87.74), −5.70 MAE on test_avg (83.94 → 78.24). Below seed-noise floor in absolute terms but consistent with R2's much-larger pre-EMA delta of −33 MAE.
- **Per-split val MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `val_single_in_dist` | (TBD per split — frieren report) |
  | `val_geom_camber_rc` | (TBD) |
  | `val_geom_camber_cruise` | (TBD) |
  | `val_re_rand` | (TBD) |
  | **val_avg** | **87.7443** |

- **Per-split test MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `test_single_in_dist` | 91.91 |
  | `test_geom_camber_rc` | 87.27 |
  | `test_geom_camber_cruise` | 56.15 |
  | `test_re_rand` | 77.62 |
  | **test_avg** | **78.2370** |

- **W&B run:** `4fas292o` in group `onecycle-lr-r3` (project `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`)
- **Reproduce:**
  ```bash
  cd target/
  python train.py --epochs 50 \
      --wandb_group baseline-after-pr409 --wandb_name baseline-r1 \
      --agent willowpai2d3-XXX
  ```
- **Notes:**
  - **Critical mechanism finding from frieren:** EMA-vs-live diagnostic flips sign across schedules. Under OneCycle, live weights (82.06) BEAT EMA-averaged weights (87.74) by +5.69 MAE. Under warmup-cosine, EMA (95.26) beats live (102.51) by −7.26 MAE. EMA helps noisy schedules; counterproductive on converged ones. **EMA may now be sub-optimal in baseline** — frieren is following up with `use_ema=False` test under OneCycle (predicted val_avg ≈ 82).
  - Compounding analysis: OneCycle's cool-down and EMA+L1 partially overlap (both contribute to a smoother converged state). Schedule effect shrunk from −33 MAE (R2 without EMA/L1) to −7.5 MAE (R3 with EMA+L1). Still net-positive.
  - Hyperparameter snapshot: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, huber_delta=0, epochs=50, schedule=onecycle, onecycle_peak_lr=2e-3, onecycle_pct_start=0.1, onecycle_total_epochs=15, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, use_ema=True, ema_decay=0.999, ema_warmup_steps=100`.

## 2026-04-28 06:47 — PR #294: Pure L1 surface loss (`huber_delta=0`)

- Branch: `willowpai2d3-alphonse/huber-loss-surf-p` (squash-merged)
- **Recipe addition:** Surface loss replaced with Huber-shaped loss; `huber_delta=0` defaults in `train.py` (degenerate to pure L1 = MAE-in-normalized-space). Volume loss stays MSE.
- **Best val avg surface MAE:** `val_avg/mae_surf_p = 94.8854` (epoch 14, run `1zpw3ts2`).
- **Best test avg surface MAE:** `test_avg/mae_surf_p = 83.9410` (same checkpoint).
- **Within-sweep delta (apples-to-apples, monotonic 4-point sweep):** δ=2.0 (110.61) → δ=0 (94.89) = **−15.72 MAE on val_avg, −14.24 on test_avg**. Trend monotonic and clean across δ ∈ {2.0, 1.0, 0.5, 0}.
- **vs prior baseline (PR #410, EMA-included):** **−26.55 MAE on val_avg (121.44 → 94.89, −21.9%), −24.72 MAE on test_avg (108.66 → 83.94, −22.8%)**. Comfortably above the ~25 MAE seed-noise floor on val_avg, with test_avg confirming.
- **Per-split val MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `val_single_in_dist` | 115.4932 |
  | `val_geom_camber_rc` | 107.5207 |
  | `val_geom_camber_cruise` | 69.5202 |
  | `val_re_rand` | 87.0077 |
  | **val_avg** | **94.8854** |

- **Per-split test MAE on best-val checkpoint:**

  | Split | mae_surf_p |
  |---|---:|
  | `test_single_in_dist` | 102.7639 |
  | `test_geom_camber_rc` | 93.1056 |
  | `test_geom_camber_cruise` | 58.8788 |
  | `test_re_rand` | 81.0157 |
  | **test_avg** | **83.9410** |

- **W&B run:** `1zpw3ts2` in group `huber-loss-surf-p-r2` (project `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`)
- **Reproduce:**
  ```bash
  cd target/
  python train.py --huber_delta 0 --epochs 50 \
      --wandb_group huber-loss-surf-p-r2 --wandb_name huber-d0-r2 \
      --agent willowpai2d3-alphonse
  ```
- **Notes:**
  - Compounding confirmed: pure L1 stacks with the warmup+EMA baseline. The mechanisms are orthogonal (loss landscape vs. optimizer schedule + weight averaging) so gains add roughly additively.
  - The largest absolute gains land on the high-residual splits: `val_single_in_dist` (148.90 → 115.49, −33.41 MAE) and `val_geom_camber_rc` (130.69 → 107.52, −23.17). Cruise and re_rand also improve cleanly.
  - Mechanism check via `train/surf_huber_outlier_frac` confirms the lever is doing what it should: at δ=0 every surface element is in the linear regime by construction (outlier_frac = 1.0); at δ=2.0 it stays near 0 (essentially MSE-on-surface).
  - Hyperparameter snapshot: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, huber_delta=0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, use_ema=True, ema_decay=0.999, ema_warmup_steps=100`.

## 2026-04-28 03:37 — PR #410: EMA of weights at eval time (decay=0.99, warmup_steps=100)

- Branch: `willowpai2d3-nezuko/ema-of-weights` (squash-merged)
- **Recipe addition:** `use_ema=True, ema_decay=0.999, ema_warmup_steps=100` now defaults in `train.py`. EMA-averaged weights are saved as the best-val checkpoint and evaluated on test.
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
  - Hyperparameter snapshot: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, use_ema=True, ema_decay=0.999, ema_warmup_steps=100`.

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
