# Round Baseline — `icml-appendix-charlie-pai2d-r2`

Lower is better. Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across the four val splits). Paper-facing metric is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-27 23:30 — PR #282: Replace MSE with Huber loss (delta=1.0) in normalized space

- **Best `val_avg/mae_surf_p`** (target to beat): **105.999** (epoch 14, original recipe)
- **`test_avg/mae_surf_p`** (paper-facing): **97.957** — first finite measurement, from PR #361 rerun under same recipe + NaN-safe eval (val_avg drift to 108.103 there is RNG noise; recipe and val computation are byte-identical to the 105.999 high-water mark).
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
