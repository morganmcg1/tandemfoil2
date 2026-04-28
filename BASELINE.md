# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #324 nezuko, 2026-04-28)

EMA (decay=0.999) shadow weights with every-2-epochs validation gating,
stacked on top of pure L1 + bf16 + FF K=8 + `torch.compile(dynamic=True)` +
cosine T_max=50 + per-Re sqrt sampling. Validation, checkpoint, and test
eval all use EMA weights. Every-2-epochs gating recovers the schedule
budget that v3's swap-validate-swap was eating.

- **`val_avg/mae_surf_p` = 52.1155** at epoch 36 (of 36 completed, wall-cap)
- **`test_avg/mae_surf_p` = 45.0018** (best EMA val checkpoint)
- W&B run: [`qsplc76j` / `ema999-on-l1-every2-rebased`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/qsplc76j)
- Per-epoch wall: ~52.9 s steady state (49 s train + ~3-4 s amortized EMA val)
- Peak GPU memory: 24.1 GB / 102.6 GB (~78 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 36/50 epochs (matches baseline budget exactly).

### Per-split surface MAE (val, best EMA checkpoint = epoch 36)

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 54.4680 |
| val_geom_camber_rc | 65.0996 |
| val_geom_camber_cruise | 34.2148 |
| val_re_rand | 54.6796 |
| **val_avg** | **52.1155** |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 46.9381 |
| test_geom_camber_rc | 59.9922 |
| test_geom_camber_cruise | 28.7984 |
| test_re_rand | 44.2783 |
| **test_avg** | **45.0018** |

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| PR #314 (edward): + SmoothL1 β=1.0 | 69.83 | −13.6% |
| PR #407 (fern): + cosine T_max=37 (Huber-era) | 69.74 | −0.13% |
| PR #504 (edward): SmoothL1 → pure L1 | 57.29 | −17.96% |
| PR #541 (edward): T_max=50 confirmed for L1, fresh seed | 56.22 | −1.07% |
| PR #531 (fern): + per-Re sqrt sampling | 54.09 | −3.79% |
| **PR #324 (nezuko): + EMA decay=0.999 every-2-epochs gating** | **52.12** | **−3.65%** |

Cumulative: **−63.9% on val_avg / −65.7% on test_avg** since PR #312.

### EMA mechanism

EMA shadow weights (decay=0.999) updated after every `optimizer.step()`. At
13,500 training steps in 36 epochs × 375 batches, `0.999^13500 ≈ 1.4e-6`
initial-weight contamination — fully warm. Validation, checkpoint, and
end-of-run test eval all use EMA weights. Every-2-epochs gating: the
swap-validate-swap pattern runs only on even-numbered epochs (2, 4, ...,
36), saving ~3-4 s/epoch on amortized cost. Best-tracking is monotonically
correct — every successive EMA val improved over the prior, with `best`
landing on epoch 36 (last EMA val before timeout).

EMA val curve is monotonically descending across all 18 EMA validation
points (textbook variance-reduction signature). Raw val curve swings more
(epoch 30 raw=62.77, epoch 32 raw=58.17 — bigger swings than EMA's
55.97 → 54.60 over the same window).

### Per-split signal: rc-camber convergence

The four interventions that have moved rc-camber best are now visible:

| Intervention | val_geom_camber_rc Δ |
|---|---|
| FF K=8 (PR #327) | −3.3% |
| Compile (PR #416, schedule unblock) | −25.8% (vs FF alone) |
| Per-Re sqrt sampling (PR #531) | −5.0% (vs L1+T_max=50 alone) |
| EMA (this PR) | −5.1% (vs per-Re+L1 alone) |

rc-camber's failure mode has multiple components: schedule-truncation
(addressed by compile), spatial frequency representation (modest help from
FF), high-Re distributional emphasis (per-Re sampling helps), and
parameter-trajectory variance (EMA helps). Each lever moves rc somewhat
but none dominates.

## Default config (matches PR #324)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- **`--epochs 50`** (T_max=50; lr ends at ~16% of peak which pure L1 uses
  productively)
- **EMA decay=0.999** with every-2-epochs validation gating (`ema_val_interval=2`)
- **Per-Re sampling**: WeightedRandomSampler weights = `(1/group_size) ×
  sqrt(Re / Re_median[domain])` — built once at startup
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- **Loss**: pure L1 `(pred - y_norm).abs()` per-element loss in normalized
  space, with surface vs. volume split via `surf_weight`. Inside
  `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Fourier features (K=8)** for normalized (x, z), computed in fp32
  outside the autocast scope, concatenated to the per-node feature vector.
  Per-node feature dim: 24 → 56.
- **`torch.compile(model, dynamic=True)`** wrapper applied right after
  `model.to(device)` (gated on `not cfg.debug`). Save/load via
  `getattr(model, "_orig_mod", model).state_dict()` (now wrapping
  EMA-state-dict pattern).
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22 + 4*8 = 54`, `out_dim=3`).

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-ema-l1-rew \
  --wandb_name baseline-ema-l1-rew
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap binding at 36/50 epochs.
- Single-seed variance ≈ ±1% on val_avg.
- VRAM headroom is now 78 GB (24.1 / 102.6).
- `data/scoring.py` patched (`b78f404`).
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.

## Prior baselines (superseded)

- **PR #312** (alphonse, original): val_avg=144.21, test_avg=131.18.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15.
- **PR #327** (tanjiro, FF K=8): val_avg=106.92, test_avg=96.82.
- **PR #416** (alphonse, compile+FF): val_avg=80.85, test_avg=73.41.
- **PR #314** (edward, Huber+compile+FF): val_avg=69.83, test_avg=61.72.
- **PR #407** (fern, T_max=37 on Huber): val_avg=69.74, test_avg=60.48.
- **PR #504** (edward, pure L1): val_avg=57.29, test_avg=51.35.
- **PR #541** (edward, T_max=50 for L1, fresh seed): val_avg=56.22, test_avg=48.42.
- **PR #531** (fern, per-Re sqrt sampling): val_avg=54.09, test_avg=46.40.
