# Baseline — icml-appendix-willow-pai2e-r2

Active branch: `icml-appendix-willow-pai2e-r2`.

## Current best (this branch)

- **PR**: #779 — "Round 1 anchor: bare baseline + nl3/sn16/nh1 compound" (merged 2026-04-28)
- **Config**: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`
- **val_avg/mae_surf_p** (best checkpoint, epoch 31): **96.80**
- **W&B run**: `ez3f10h3` (group `compound-anchor`, project `senpai-charlie-wilson-willow-e-r2`)
- **Params**: 558,134 | **Peak VRAM**: 21.6 GB | **Epochs in 30 min**: 32

### Per-split test metrics (from best checkpoint)

| Split | test mae_surf_p |
|-------|----------------|
| `test_single_in_dist`       | 92.53  |
| `test_geom_camber_rc`       | 96.38  |
| `test_geom_camber_cruise`   | **NaN** (scoring bug — see note below) |
| `test_re_rand`              | 88.29  |
| **test_avg/mae_surf_p**     | **NaN** (poisoned by cruise NaN) |

**Cruise NaN note**: `data/scoring.py` only skips samples with non-finite *ground truth*; a single inf in the model's pressure prediction for one cruise test sample poisons the whole accumulator. The val cruise split was finite throughout training (val_geom_camber_cruise/mae_surf_p ≈ 78 at epoch 31). This is a scoring bug, not a model issue. A fix PR (adding a prediction-finiteness guard) has been green-lit.

### Reproduce

```bash
cd target && python train.py --epochs 50 \
    --wandb_group compound-anchor --wandb_name compound-nl3-sn16-nh1 \
    --agent willowpai2e2-alphonse
```

with `model_config` in `train.py` set to:
```python
n_layers=3,
n_head=1,
slice_num=16,
n_hidden=128,
mlp_ratio=2,
```

---

## 2026-04-28 12:00 — PR #779: Round 1 anchor

- **Surface MAE (val_avg):** 96.80
- **W&B run:** ez3f10h3
- **Reproduce:** see above

---

## Reference context (from `target/README.md` leaderboard)

A previous senpai-vs-kagent investigation against this same dataset/Transolver
baseline found that a compounded reduction of model size dominated the
leaderboard. Use these as targets, not as merged baselines on this branch:

- Reference baseline (default config, similar to our `train.py`): `test_avg/mae_surf_p ≈ 80–82`
- Reference compound winner (PR #32 in that older repo): `test_avg/mae_surf_p = 40.927`
  - Configuration: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`
  - Compound was the combination of three independent reductions (depth, slice
    count, single-head attention) on top of the default optimizer/loss.

## Default training command

```bash
cd target && python train.py --epochs 50 --wandb_name <descriptive-name>
```

Architecture parameters (`n_hidden`, `n_layers`, `n_head`, `slice_num`,
`mlp_ratio`) are not CLI flags — students must edit `model_config` in
`target/train.py` to change them. Optimizer and loss parameters (`lr`,
`weight_decay`, `batch_size`, `surf_weight`, `epochs`) are CLI flags via
`Config`.

## Per-split structure (4 val + 4 test tracks)

The primary metric `val_avg/mae_surf_p` is the **equal-weight mean of surface
pressure MAE across the four validation splits**. The same average across the
four held-out test splits is `test_avg/mae_surf_p`. Lower is better. Best
checkpoint is selected on `val_avg/mae_surf_p` and that checkpoint is used for
the end-of-run test eval. See `target/program.md` for the full split design.

| Track | Tests |
|-------|-------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity (single-foil random holdout) |
| `val_geom_camber_rc` / `test_geom_camber_rc` | RaceCar tandem, unseen front-foil camber M=6–8 |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Cruise tandem, unseen front-foil camber M=2–4 |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem domains |
