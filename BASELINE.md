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

**PR #627 — `preprocess-depth-1-on-lion` (edward), merged 2026-04-28** — ⭐ new best

- `val_avg/mae_surf_p` = **53.7986** (best epoch 18/24)
- `test_avg/mae_surf_p` = **NaN** (4-split) / **52.165** (mean of 3 clean splits)
- Per-split val: `val_single_in_dist=54.3136`, `val_geom_camber_rc=70.8552`, `val_geom_camber_cruise=35.2098`, `val_re_rand=54.8159`
- Per-split test (3 clean): `test_single_in_dist=47.9103`, `test_geom_camber_rc=63.2333`, `test_re_rand=45.3507`
- **−4.27% val / −2.18% test** vs previous baseline (PR #612). Largest gain on `val_single_in_dist` (−9.93%), `val_geom_camber_cruise` (−4.86%), `val_re_rand` (−2.83%). `val_geom_camber_rc` essentially flat (−0.29%).
- Change: add **1 residual hidden layer to the preprocess MLP** (`preprocess_layers=1, res=True`), adding ~65,792 params (+9.8%) at the input boundary. All other knobs identical to PR #612 Lion baseline.
- Model: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, **`preprocess_layers=1`**, `fun_dim=54`, `lr=3e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, `grad_clip_norm=0.5`, `amp_bf16=True`, L1 loss, warmup→cosine, `--epochs 24`, 8-band Fourier features, `optimizer_name="lion"`.
- Note: best epoch 18/24, wall-time bound — val still falling at epoch 18.

Reproduce:
```bash
cd target/ && python train.py \
  --agent charliepai2d5-edward \
  --experiment_name preprocess-depth-1-on-lion \
  --epochs 24
```

### Previous best

**PR #612 — `lion-3e-4` (alphonse), merged 2026-04-28** — ⭐ largest single-PR delta

- `val_avg/mae_surf_p` = **56.1948** (best epoch 18/24)
- `test_avg/mae_surf_p` = **NaN** (4-split) / **53.330** (mean of 3 clean splits)
- Per-split val: `val_single_in_dist=60.30`, `val_geom_camber_rc=71.06`, `val_geom_camber_cruise=37.01`, `val_re_rand=56.41`
- Per-split test (3 clean): `test_single_in_dist=51.62`, `test_geom_camber_rc=61.95`, `test_re_rand=46.42`
- **−23.3% val / −23.3% test** vs previous baseline (PR #496). **All four val splits and all three clean test splits improved by 16-31%.** Largest gain on `val_single_in_dist` (-30.8%), `val_geom_camber_cruise` (-27.3%), `test_single_in_dist` (-27.3%), `test_re_rand` (-25.0%).
- Change: replace AdamW with **Lion optimizer** (Chen et al. 2023, sign-based update with momentum), `lr` 1e-3 → 3e-4 (Lion's standard ~3× smaller LR scaling). Per-epoch wall essentially unchanged (101.4s vs 100s).

### Diagnostic finding (alphonse's analysis)

**Why Lion crushes AdamW on this stack so dramatically (vs the typical 1-3% Lion paper bonus):**

Two layers of intuition:
1. **L1 loss + sign-update + grad-clip-0.5 form a triple-quantized chain.** L1 produces unit-magnitude per-element gradients; clip-norm-0.5 caps total L2; Lion's `sign()` makes parameter updates unit-magnitude per param. Every weight gets the same per-step treatment — scale-invariant per parameter, composing naturally with the existing L1+clip regime.
2. **AdamW's RMSProp normalization was redundant and noisy.** Under L1 loss, all per-element gradients have magnitude 1, so RMSProp's adaptive normalization is mostly idle and adds noise from its EMA estimate. Lion replaces this with `sign(momentum)` — exactly what RMSProp converges to in this regime, but cleaner.

The val curve is visibly smoother than AdamW's (3 small bumps in 17 transitions vs typical AdamW wobble). Best epoch is the last reached (18/24), so still improving — Lion has more headroom with longer schedules.

Full reference config now: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `fun_dim=54`, **`lr=3e-4`** (peak, linear warmup), `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, `grad_clip_norm=0.5`, `amp_bf16=True`, **L1** loss in normalized space, **SequentialLR(LinearLR warmup × 5 ep, CosineAnnealingLR T_max=epochs−5)**, `--epochs 24`, 8-band Fourier features on normalized `(x, z)`, **`optimizer_name="lion"`** (Config default).

Reproduce:
```bash
cd target/ && python train.py \
  --agent charliepai2d5-alphonse \
  --experiment_name lion-3e-4
```

(All knobs are Config defaults.)

### Previous best

**PR #496 — `bf16-amp-fp32-loss` (alphonse), merged 2026-04-28**

- `val_avg/mae_surf_p` = 73.2916 (best epoch 18/24)
- `test_avg/mae_surf_p` (3-split mean) = 69.493
- Change: bf16 mixed precision (forward only) with fp32 loss accumulator. 24% per-epoch speedup → 18 epochs reachable.

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
