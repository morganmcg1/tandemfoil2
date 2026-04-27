# Round Baseline — `icml-appendix-charlie-pai2d-r2`

Lower is better. Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across the four val splits). Paper-facing metric is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-27 23:30 — PR #282: Replace MSE with Huber loss (delta=1.0) in normalized space

- **Best `val_avg/mae_surf_p`**: **105.999** (epoch 14)
- **`test_avg/mae_surf_p`**: NaN (scoring.py Inf*0 propagation; mean over 3 finite test splits = 105.418)
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

## Known issues affecting `test_avg/mae_surf_p`

- `test_geom_camber_cruise` sample 20 has 761 non-finite values in `y[p]` volume nodes. `data/scoring.py:accumulate_batch` is read-only and propagates Inf via `Inf * 0 = NaN` in the masked sum, poisoning the per-channel test sum.
- Workaround: filter samples with non-finite `y` in `train.py:evaluate_split` before calling `accumulate_batch`. This will be added in a round-2 PR; subsequent runs should report finite `test_avg/mae_surf_p`.
