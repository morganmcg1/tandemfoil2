# Round Baseline — `icml-appendix-charlie-pai2d-r2`

Lower is better. Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across the four val splits). Paper-facing metric is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-28 04:15 — PR #479: Bias-corrected EMA (decay_target=0.99, warmup_steps=10) — orthogonal compound lever

- **Best `val_avg/mae_surf_p`** (target to beat — conservative): **72.414** (fern's δ=0.25 measurement on EMA(0.99)+SwiGLU pre-DropPath)
- **Bias-corrected EMA standalone measurement**: 81.251 on EMA(0.99)+SwiGLU pre-DropPath baseline (−2.37% vs 83.223). Strict superset of the existing EMA(0.99) — `warmup_steps=0` reduces to current behavior.
- **Recipe**: huber(δ=0.25) + **bias-corrected EMA (decay_target=0.99, warmup_steps=10)** + SwiGLU FFN + DropPath(0→0.1) + AdamW betas (0.9, 0.95) + NaN-safe. The EMA constructor signature changed to accept `warmup_steps`.

## 2026-04-28 03:55 — PR #480: AdamW betas (0.9, 0.95) — earlier orthogonal compound

- **β₂=0.95 standalone measurement**: 77.951 on EMA(0.99)+SwiGLU baseline (−6.34% vs 83.223).
- **Caveat**: post-merge baseline number not directly measured — both fern's δ=0.25 (72.414) and nezuko's β₂=0.95 (77.951) were measured on the same pre-DropPath, single-axis stack. The combined stack (δ=0.25 + DropPath + β₂=0.95 + ...) will be measured by round-6/7 PRs. Predicted combined val_avg: ~67–72 if levers compound additively, ~72 if partially redundant.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name post-betas095 --agent <name>
  ```

## 2026-04-28 03:25 — Previous baseline (PR #463, huber δ=0.25)

- **Best `val_avg/mae_surf_p`**: 72.414 (epoch 13, measured pre-DropPath; post-merge stack adds DropPath, expected at-or-better)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **63.082** (same caveat — pre-DropPath measurement; post-merge stack expected at-or-better)
- **Per-split val MAE for `p` (pre-DropPath measurement)**:
  - `val_single_in_dist`: 87.914 (−11.03% vs EMA(0.99)+SwiGLU baseline)
  - `val_geom_camber_rc`: 83.323 (−13.76%)
  - `val_geom_camber_cruise`: 50.222 (−17.88% — biggest split improvement)
  - `val_re_rand`: 68.199 (−10.62%)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 76.918
  - `test_geom_camber_rc`: 72.750
  - `test_geom_camber_cruise`: 42.154
  - `test_re_rand`: 60.506
- **Recipe**: huber(**δ=0.25**, was 1.0) + EMA(decay=0.99) + SwiGLU FFN + DropPath(0→0.1, last block kept) + NaN-safe `evaluate_split`. δ change is the only diff vs the post-DropPath state.
- **δ profile** (closes the question): δ=2 → 107.6, δ=1 → 88.2 (pre-EMA), δ=0.5 → 87.3 (pre-EMA), δ=1 → 83.2 (post-EMA), **δ=0.25 → 72.4 (post-EMA, this PR)**. Profile is monotone toward L1 with **non-diminishing returns** in this regime — δ=0.5→0.25 delivered ~10× more improvement than δ=1→0.5 did. The smaller quadratic region handles the heavy-tailed pressure error distribution; EMA(0.99)'s fast tracking compounds especially well with a more L1-like loss.
- **Note on measurement vs merged stack**: this val_avg was measured on EMA(0.99)+SwiGLU+huber(δ=0.25)+NaN-safe (no DropPath, since fern branched before DropPath merged). The current advisor branch now has DropPath added on top of fern's δ change. Since DropPath is a generic regularizer and orthogonal to loss reformulation, the merged compound is expected to be at-or-better-than 72.414. Round-7 PRs will measure the actual post-merge baseline.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-delta-025 --agent <name>
  ```

## 2026-04-28 03:10 — Previous baseline (PR #455, DropPath)

- **Best `val_avg/mae_surf_p`**: 80.480 (epoch 14, measured directly)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **72.328**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 92.907 (−5.98% vs EMA(0.99) baseline)
  - `val_geom_camber_rc`: 95.534 (−1.12%, near noise)
  - `val_geom_camber_cruise`: 57.237 (−6.41%)
  - `val_re_rand`: 76.241 (−0.08%, flat)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 85.502 (−4.82%)
  - `test_geom_camber_rc`: 85.565 (+1.38% — within noise)
  - `test_geom_camber_cruise`: 49.233 (−3.17%)
  - `test_re_rand`: 69.012 (−2.17%)
- **Recipe**: huber(δ=1.0) + EMA(decay=0.99) + SwiGLU FFN + **DropPath (per-block linear schedule 0→0.1, last block always kept)** + NaN-safe `evaluate_split` filter. Effective per-block drop = `[0.0, 0.025, 0.05, 0.075, 0.0]`. Param-identical to baseline (no new learnable parameters).
- **Mechanism (refined)**: DropPath acts as a generic regularizer here (val curve is parallel to baseline with a uniform offset, not a late-training kick). The OOD-camber splits (`val_geom_camber_rc`, `val_re_rand`) showed near-flat improvements — the gain is concentrated on `val_single_in_dist` (in-dist) and `val_geom_camber_cruise` (easier OOD). Implies the camber_rc / re_rand bottleneck is NOT implicit-ensembling-shaped; it's likely data-coverage / extrapolation-shaped.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name stochastic-depth-01 --agent <name>
  ```

## 2026-04-28 02:15 — Previous baseline (PR #426, EMA(0.99))

- **Best `val_avg/mae_surf_p`**: 83.223 (epoch 13)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **73.904**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 98.815 (−7.13% vs SwiGLU baseline)
  - `val_geom_camber_rc`: 96.612 (−3.78% vs SwiGLU baseline)
  - `val_geom_camber_cruise`: 61.160 (−5.04% vs SwiGLU baseline)
  - `val_re_rand`: 76.304 (−6.60% vs SwiGLU baseline)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 89.833
  - `test_geom_camber_rc`: 84.398
  - `test_geom_camber_cruise`: 50.843
  - `test_re_rand`: 70.541
- **Recipe**: huber(δ=1.0) + EMA(decay=0.99 — was 0.999) + SwiGLU FFN inside `TransolverBlock` + NaN-safe `evaluate_split` filter. All other defaults unchanged.
- **Mechanism**: under the default cosine `T_max=50` schedule with only 13 reachable epochs in the 30-min budget, EMA(0.999)'s ~1.85-epoch half-life means the EMA is heavily anchored to random-init weights for the first ~2 epochs. EMA(0.99) (half-life ~0.18 epochs) tracks the live model immediately, removing the bias-from-cold-start. Gain is uniform across all 4 val + test splits.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-decay-099 --agent <name>
  ```

## 2026-04-28 01:20 — Previous baseline (PR #391, SwiGLU FFN)

- **Best `val_avg/mae_surf_p`**: 88.227 (epoch 13)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **78.338**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 106.398 (−15.78% vs EMA)
  - `val_geom_camber_rc`: 100.406 (−8.23% vs EMA — finally moved after being flat under EMA alone)
  - `val_geom_camber_cruise`: 64.409 (−16.34% vs EMA)
  - `val_re_rand`: 81.696 (−11.86% vs EMA)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 96.439
  - `test_geom_camber_rc`: 88.064
  - `test_geom_camber_cruise`: 54.011
  - `test_re_rand`: 74.837
- **Recipe**: huber(δ=1.0) + EMA(decay=0.999, eval+ckpt) + LLaMA-style SwiGLU FFN (gate × value, bias=False, intermediate=176) inside `TransolverBlock`. NaN-safe `evaluate_split` workaround active. All other defaults unchanged (lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, slice_num=64, n_layers=5, n_hidden=128, n_head=4, mlp_ratio=2). Param-matched (+1.3% n_params: 670K vs prior 660K).
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name swiglu-mlp --agent <name>
  ```

Loss curve was still descending monotonically at the 30-min cap — model is under-trained, more epochs likely give further gains.

## 2026-04-28 00:25 — Previous baseline (PR #363, EMA-eval)

- **Best `val_avg/mae_surf_p`**: 101.350 (epoch 14)
- **`test_avg/mae_surf_p`** (paper-facing): pending finite re-measurement on the EMA-merged baseline (cruise NaN here because PR #361 had not landed when this run started); **3-split test mean = 100.030** — `single_in_dist=113.32, geom_camber_rc=97.44, re_rand=89.33`.
- **Per-split val MAE for `p` (EMA, epoch 14)**:
  - `val_single_in_dist`: 126.323 (−5.76% vs huber)
  - `val_geom_camber_rc`: 109.406 (−0.07%, flat)
  - `val_geom_camber_cruise`: 76.988 (−6.93% vs huber)
  - `val_re_rand`: 92.682 (−5.19% vs huber)
- **Recipe**: huber(δ=1.0) loss in normalized space + EMA copy of weights (decay 0.999), checkpoint = EMA weights. All other defaults unchanged from the merged baseline.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-eval --agent <name>
  ```

## 2026-04-27 23:30 — Previous baseline (PR #282 + #361)

- **Best `val_avg/mae_surf_p`**: 105.999 (PR #282 huber-loss)
- **`test_avg/mae_surf_p`**: 97.957 (first finite measurement, PR #361 NaN-safe eval rerun)
- **Per-split val surface MAE for `p`**:
  - `val_single_in_dist`: 134.048
  - `val_geom_camber_rc`: 109.479
  - `val_geom_camber_cruise`: 82.718
  - `val_re_rand`: 97.751
- **Per-split val Ux / Uy / p (surface)**: see `research/EXPERIMENTS_LOG.md`
- **Model**: Transolver, 0.66M params, default config (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`).
- **Optimizer**: AdamW, lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, epochs=50 (timeout-truncated at 14/50 epochs).
- **Loss**: Huber(δ=1.0) on normalized targets, applied identically in train and val/test eval.
- **Metrics path**: `models/model-charliepai2d2-edward-huber-loss-20260427-223516/{metrics.jsonl,metrics.yaml}`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-loss --agent <name>
  ```

## 2026-04-28 00:10 — PR #361 follow-up: per-split test surface MAE for `p` (first finite test_avg)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| `test_single_in_dist`     | 123.760 | 1.737 | 0.746 |
| `test_geom_camber_rc`     | 104.946 | 2.090 | 0.877 |
| `test_geom_camber_cruise` |  66.144 | 0.959 | 0.480 |
| `test_re_rand`            |  96.978 | 1.532 | 0.706 |
| **avg**                   | **97.957** | **1.579** | **0.702** |

PR #361 added a 3-line filter in `train.py:evaluate_split` that drops samples with any non-finite `y` from the batch before calling `accumulate_batch`. The `data/scoring.py:accumulate_batch` Inf-times-0 propagation bug remains (file is read-only); the workaround triggers exactly once per test pass — on `test_geom_camber_cruise` sample 20 (761 non-finite `y[p]` volume nodes; surface `p` and Ux/Uy unaffected) — and is a no-op everywhere else.

## Ranking note

Future PRs are scored against `val_avg/mae_surf_p < 105.999` (recipe high-water mark from PR #282), **not** against the 108.103 RNG draw from PR #361. The val computation path on PR #361 is byte-identical to the merged recipe (the workaround does not trigger on any val sample); the +1.99% delta is purely run-to-run variance under a 14-epoch timeout-truncated training.
