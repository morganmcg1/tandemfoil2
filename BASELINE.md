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

**PR #464 — `grad-clip-0p5` (alphonse), merged 2026-04-28**

- `val_avg/mae_surf_p` = **73.9087** (best epoch 14/14)
- `test_avg/mae_surf_p` = **NaN** (4-split) / **70.371** (mean of 3 clean splits — same pre-existing `test_geom_camber_cruise` GT-NaN)
- Per-split val: `val_single_in_dist=81.66`, `val_geom_camber_rc=87.95`, `val_geom_camber_cruise=54.47`, `val_re_rand=71.55`
- Per-split test (3 clean): `test_single_in_dist=71.95`, `test_geom_camber_rc=75.93`, `test_re_rand=63.24`
- **−0.71% val / −2.45% test** vs the previous baseline (PR #387). Test improvement is the dominant signal — tighter clipping smooths small-batch L1 noise, payoff is larger on lower-noise test splits than on val.
- Change: pure CLI/Config tweak — `grad_clip_norm` 1.0 → 0.5. Halves the per-step update magnitude in the (always-active) clipped regime.

### Per-split note

The single_in_dist gain is large (−5.79% val) but two camber-OOD splits regressed slightly (val_geom_camber_rc +2.4%, val_geom_camber_cruise +2.2%). Net is positive on val_avg, and uniformly positive on test (where each split has 2× the samples and lower noise). Worth tracking on subsequent merges — if the camber-OOD regression compounds, may need to reverse this axis.

### Diagnostic finding

Pre-clip ‖∇‖ trajectory matches PR #387's (peak ~270 at warmup top, end ~60–63) — confirming that clipping is a per-step magnitude bound, not a gradient-computation change. The optimizer sees identical gradients; just the applied update magnitude is halved when `max_norm=0.5` vs `1.0`. Clipping ratio is now 120–540 : 1, even more in pure direction-only mode.

Full reference config now: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `fun_dim=54`, `lr=1e-3` (peak, linear warmup), `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, **`grad_clip_norm=0.5`**, **L1** loss in normalized space, **SequentialLR(LinearLR warmup × 5 ep, CosineAnnealingLR T_max=epochs−5)**, `--epochs 14`, **8-band Fourier features on normalized (x, z)**.

Reproduce:
```bash
cd target/ && python train.py \
  --agent charliepai2d5-alphonse \
  --experiment_name grad-clip-0p5 \
  --epochs 14
```

(All other knobs are Config defaults.)

### Previous best

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
