# Baseline — charlie-pai2d-r5

This advisor track starts from a pristine `train.py`. No round-1 measurements yet — first cohort of PRs will establish the empirical baseline by running modifications against it.

## Reference configuration (unmodified `train.py`)

| Block | Value |
|-------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` | 2 (x, z) |
| `fun_dim` | 22 (X_DIM − 2) |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| `batch_size` | 4 |
| Loss | MSE in normalized space |
| `surf_weight` | 10.0 (volume + 10 × surface) |
| Sampler | WeightedRandomSampler — equal weight across raceCar single, raceCar tandem, cruise tandem domains |
| Epochs cap | 50 (or `SENPAI_TIMEOUT_MINUTES` wall clock) |

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE across the four validation splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Lower is better. Best-validation checkpoint is what gets evaluated on the test splits.

The paper-facing rank is `test_avg/mae_surf_p`, computed once at the end of training from the best-validation checkpoint.

## Best result

**PR #496 — `bf16-amp-fp32-loss` (alphonse), merged 2026-04-28**

- `val_avg/mae_surf_p` = **73.2916** (best epoch 18/24)
- `test_avg/mae_surf_p` = **NaN** (4-split) / **69.493** (mean of 3 clean splits)
- Per-split val: `val_single_in_dist=87.09`, `val_geom_camber_rc=85.04`, `val_geom_camber_cruise=50.89`, `val_re_rand=70.16`
- Per-split test (3 clean): `test_single_in_dist=70.96`, `test_geom_camber_rc=75.62`, `test_re_rand=61.89`
- **−0.83% val / −1.25% test** vs previous baseline (PR #464). All 3 clean test splits improved.
- Change: bf16 mixed precision via `torch.autocast` wrapping **only** the model forward; `pred` cast back to fp32 before the loss accumulator. Backward, optimizer, and validation all stay in fp32. **Effectively reaches 18 epochs at the 30-min wall budget** (vs 14 in fp32 baseline) — 24% per-epoch speedup unlocks 4 extra cosine-decay epochs.
- Two-attempt history: round-1 with all-bf16 (autocast wrapping forward+loss) had test regression of +1.59% from low-precision loss accumulation. Refinement (cast `pred` back to fp32 before `(pred - y_norm).abs()`) recovered the test improvement cleanly while preserving the speedup.

### Diagnostic finding (alphonse's per-split analysis)

The 2.0-point test improvement vs the all-bf16 round-1 mapped almost directly onto the splits *furthest from val*: `test_single_in_dist` −3.84, `test_geom_camber_rc` −1.84, `test_re_rand` −0.32. Consistent with the bf16-loss-accumulator-noise-acts-as-implicit-regularizer story — small per-step noise in the loss accumulator pushed the model toward a slightly different basin that underperformed on held-out foils. The fix removed that noise without changing model precision.

Full reference config now: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `fun_dim=54`, `lr=1e-3` (peak, linear warmup), `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, `grad_clip_norm=0.5`, **`amp_bf16=True`** (bf16 forward, fp32 loss/backward), **L1** loss in normalized space, **SequentialLR(LinearLR warmup × 5 ep, CosineAnnealingLR T_max=epochs−5)**, **`--epochs 24`** (Config default; reaches ~18 epochs at 30-min wall budget), 8-band Fourier features on normalized `(x, z)`.

Reproduce:
```bash
cd target/ && python train.py \
  --agent charliepai2d5-alphonse \
  --experiment_name bf16-amp-fp32-loss
```

(All knobs are Config defaults.)

### Previous best

**PR #464 — `grad-clip-0p5` (alphonse), merged 2026-04-28**

- `val_avg/mae_surf_p` = 73.9087 (best epoch 14/14)
- `test_avg/mae_surf_p` (3-split mean) = 70.371
- Change: `grad_clip_norm` 1.0 → 0.5 (Config default).

**PR #387 — `grad-clip-1-on-fourier` (alphonse), merged 2026-04-28**

- `val_avg/mae_surf_p` = 74.4437 (best epoch 14/14)
- `test_avg/mae_surf_p` (3-split mean) = 72.137
- Change: gradient clipping at `max_norm=1.0` between `loss.backward()` and `optimizer.step()`. Per-epoch grad-norm telemetry in JSONL.

**PR #301 — `surf-weight-30-on-fourier` (nezuko), merged 2026-04-28**

- `val_avg/mae_surf_p` = 76.6771 (best epoch 14/14)
- `test_avg/mae_surf_p` (3-split mean) = 73.395
- Change: `surf_weight: 10.0 → 30.0` (Config default).
- Tradeoff: `val_avg/mae_vol_p` regressed by +13.2%.

**PR #365 — `fourier-features-on-l1-warmup` (thorfinn), merged 2026-04-28**

- `val_avg/mae_surf_p` = 87.8551 (best epoch 12/14)
- `test_avg/mae_surf_p` (3-split mean) = 84.222
- Change: 8-band sinusoidal Fourier features on normalized `(x, z)` positions.

**PR #296 — `lr-warmup-1e3-budget` (fern), merged 2026-04-28**

- `val_avg/mae_surf_p` = 94.5397 (best epoch 12/14)
- `test_avg/mae_surf_p` (3-split mean) = 91.853
- Change: linear warmup over 5 epochs (1e-5 → 1e-3) → cosine decay over 9 epochs (1e-3 → 0), with `--epochs 14` budget-matched.

**PR #293 — `l1-loss` (edward), merged 2026-04-27**

- `val_avg/mae_surf_p` = 101.868 (epoch 14/50, run terminated by 30-min wall timeout while still improving)
- `test_avg/mae_surf_p` (3-split mean) = 102.606
- Change: replace MSE `(pred - y_norm)**2` with L1 `(pred - y_norm).abs()` in both training and `evaluate_split`.

## Known issue affecting test scoring

`test_geom_camber_cruise` returns `NaN` for `mae_surf_p` and `mae_vol_p` because of a non-finite ground truth on at least one sample (Edward's diagnosis: sample 20 has 761 NaN values in the `p` channel of GT). `data/scoring.accumulate_batch` computes `(pred - y).abs()` before masking, so NaN propagates into the per-channel sum even when the surrounding code intends to skip the sample. `data/scoring.py` is read-only per program constraints, so for now we rank on the 3-clean-split test mean as a stable indicator. Worth flagging to the human team for an upstream fix.
