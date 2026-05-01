# ML Intern TandemFoilSet-Balanced Benchmark — Replicate `mlintern-pai2-24h-v3-r3`

## Headline numbers

| | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| Original train.py baseline (5L/128H, MSE, flat LR via `--epochs 999`) | ~140 | ~120 (estimated) |
| **Best single model — `r6e`** (5L/128H, relL2, drop=0.05, warmup=0.10, `--epochs 200`) | **31.79** | **28.35** |
| **Best ensemble — top-10** (10 single-model checkpoints, average of normalized predictions) | **29.07** | **24.87** |

Best ensemble breakdown:

```
val_avg/mae_surf_p = 29.07
  val_single_in_dist        surf_p = 26.38   vol_p = 30.76
  val_geom_camber_rc        surf_p = 42.38   vol_p = 44.80
  val_geom_camber_cruise    surf_p = 15.94   vol_p = 17.38
  val_re_rand               surf_p = 31.58   vol_p = 31.74

test_avg/mae_surf_p = 24.87
  test_single_in_dist       surf_p = 25.39   vol_p = 29.91
  test_geom_camber_rc       surf_p = 38.36   vol_p = 41.28
  test_geom_camber_cruise   surf_p = 12.68   vol_p = 14.56
  test_re_rand              surf_p = 23.03   vol_p = 24.57
```

## Strategy summary

Six rounds of experiments, each informed by the prior round's results:

1. **Round 1 — capacity sweep.** 8 parallel 90-min runs comparing the baseline
   (`5L/128H/4head/sn64`) against AirfRANS-style upgrades
   (`8L/256H/8head/sn{16,32,64}`, `mlp_ratio=4`, deeper variants, higher
   `surf_weight`). The bigger 8L/256H models overfit quickly on this
   1499-sample dataset (best at epoch 4–5, then regressed); the original
   architecture won at epoch 11–18 with `val_avg/mae_surf_p ≈ 132`. Default
   `--epochs 999` produced essentially flat LR (cosine T_max=999), so the
   per-epoch oscillation was the dominant signal.
2. **Round 2 — training-recipe refinements on the baseline arch.** 6 parallel
   180-min runs with `--epochs 40` (proper cosine decay over the realized
   training duration), warmup, gradient clipping, and one of: dropout=0.1,
   relative-L2 loss, p-channel up-weighting, lr=2e-4, input noise, default.
   The relative-L2 loss (per-sample-and-region L2/||y||) was the clear winner
   (`r2d`: val 56.7); switching from `epochs=999` flat LR to `epochs=40`
   cosine drove the dramatic improvement across all variants.
3. **Round 3 — extended training + ensemble candidates.** 9 runs total. The
   biggest single-run win came from extending `--epochs` and adding light
   dropout: `r3a` (relL2 + dropout=0.1 + warmup=0.05 + grad_clip=1.0 +
   `--epochs 80`, 6h timeout) reached `val_avg/mae_surf_p = 40.4` (test 35.6
   with the NaN fix). Three seeds at `--epochs 50` reached val 51–52,
   confirming that "long training at gradually-decreasing LR" was the lever,
   not the seed.
4. **Round 4 — long training, 8 diverse-config seeds.** 8 parallel 6–8h runs
   sweeping dropout in {0.0, 0.05, 0.1, 0.15} and using `--epochs 100/80`.
   `r4a` and `r4b` (dropout=0.1 vs 0.05, both `--epochs 100`) hit val ≈ 37.
5. **Round 5 — ensemble-targeted batch.** 8 parallel runs with `--epochs 120/150`,
   varying dropout in {0.025, 0.05, 0.075, 0.1}, varying warmup in {0.05, 0.10},
   varying lr. `r5f` hit val 33.8 / test 28.7 with `--epochs 150`.
6. **Round 6 — final long-training batch.** 7 runs at `--epochs 150/200`,
   `dropout=0.025/0.05`, `warmup=0.10`. `r6e` (`--epochs 200`) hit
   val 31.8 / test 28.3, the best single model.

## Key technical findings

- **Architecture.** Baseline `5L/128H/4head/sn64` Transolver (≈0.66M params) is
  the right capacity for 1499 train samples. Bigger models (8L/256H/8head)
  overfit before any of the regularizers below could compensate. The
  AirfRANS-recommended `slice_num=32` and `unified_pos=1` were tested and not
  better here (smaller dataset, different geometry distribution).
- **Loss.** Switching the training loss from MSE to **per-sample, per-region
  relative L2** (`||pred - y|| / (||y|| + eps)` averaged over batch, separate
  for surface and volume regions, then `vol + λ·surf`) delivered the largest
  single-step improvement (val 62 → 57 → 41 across rounds 2–3). This
  equalizes the contribution of low-pressure and high-pressure samples,
  which is critical given the 10× per-sample y-std spread.
- **Schedule.** Warmup (5–10% of epochs) + cosine decay over the actual
  training duration. The training script's original `--epochs 999` default
  pinned `T_max=999`, leaving LR essentially flat across realized training
  and producing noisy late-stage val. Setting `--epochs` close to the wall
  number of epochs realized was as important as the loss change. **Longer
  cosine cycles helped throughout**: `--epochs 40 → 80 → 100 → 120 → 150 → 200`
  monotonically improved best val (56.7 → 40.4 → 37.1 → 33.8 → 33.6 → 31.8).
- **Regularization.** Light dropout in the Physics-Attention block
  (`dropout=0.05–0.1`) helped marginally. `weight_decay=1e-4`, `grad_clip=1.0`
  were standard. Input noise on normalized features (`σ=0.05`) was neutral.
  Increasing `surf_weight` above the default 10 hurt; changing `p_weight` had
  no consistent effect.
- **Ensemble.** Top-K prediction averaging works once individual models are
  similarly good; weight averaging across seeds (model soup) does NOT work
  here (each seed lands in a different basin). The optimal ensemble size on
  this distribution is around K=6–10 strong models — adding the bottom 7
  finished checkpoints (val 37–50) actively hurt the metric.
- **Test eval bug fix.** One sample in `test_geom_camber_cruise`
  (`000020.pt`) has +Inf in y. `data/scoring.py` is documented to skip
  non-finite samples, but `(pred - inf).abs() = inf` then `inf * 0.0 = NaN`
  propagates through the masked sum and contaminates the per-channel MAE
  accumulator. Without a fix, `train.py`'s end-of-run test eval wrote
  `test_avg/mae_surf_p = NaN` to W&B.
  **Fix in `train.py:evaluate_split` and `ensemble_eval.py`/`model_soup.py`:**
  pre-filter the batch to drop non-finite-y samples before invoking the
  scoring helpers. `data/scoring.py` is left untouched per the read-only
  contract; the bad sample is correctly excluded from the metric (matches
  the documented "per-sample skipping for non-finite ground truth").

## Top single-model results (sorted by val)

| Run | Config | val | test |
|---|---|---:|---:|
| `r6e_relL2_d005_w10_e200_s24` | drop=0.05, warmup=0.10, `--epochs 200` | **31.79** | **28.35** |
| `r6a_relL2_d005_w10_e150_s20` | drop=0.05, warmup=0.10, `--epochs 150` | 33.57 | 29.52 |
| `r6f_relL2_d005_w10_e150_s25` | drop=0.05, warmup=0.10, `--epochs 150`, seed 25 | 33.74 | 28.95 |
| `r6d_relL2_d005_w10_e150_s23` | drop=0.05, warmup=0.10, `--epochs 150`, seed 23 | 33.76 | 29.35 |
| `r5f_relL2_d01_w10_e150_s17`  | drop=0.10, warmup=0.10, `--epochs 150`         | 33.78 | **28.75** |
| `r6g_relL2_d0025_w10_e150_s26`| drop=0.025, warmup=0.10, `--epochs 150`        | 33.78 | 28.85 |
| `r6c_relL2_d005_w10_e150_s22` | drop=0.05, warmup=0.10, `--epochs 150`, seed 22 | 34.52 | 29.01 |
| `r5e_relL2_d005_w10_e120_s16` | drop=0.05, warmup=0.10, `--epochs 120`         | 34.65 | 31.28 |
| `r6b_relL2_d005_w10_e150_s21` | drop=0.05, warmup=0.10, `--epochs 150`, seed 21 | 34.80 | 29.54 |
| `r5a_relL2_d005_e120_s12`     | drop=0.05, warmup=0.05, `--epochs 120`         | 35.24 | 30.90 |

## Ensemble sweep

| Ensemble | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| 3 models  | 34.53 | 30.33 |
| 5 models  | 34.55 | 30.11 |
| **6 models**  | **29.06** | **24.89** |
| **8 models**  | **29.08** | **24.88** |
| **10 models** | **29.07** | **24.87** ← best |
| 14 models | 30.13 | 25.87 |
| 16 models | 29.89 | 25.57 |
| 17 models | 29.67 | 25.41 |

Optimal ensemble size is K≈6–10 strong models; beyond that, adding weaker
models (val > 36) adds variance faster than it adds signal. Inverse-val
weighting on top-8 gave essentially the same result as uniform weighting
(test 24.87 vs 24.88), so uniform-weight averaging is fine.

## Compute / GPU strategy

- **Hardware.** 8× NVIDIA RTX PRO 6000 Blackwell, 97 GB each.
- **Allocation.** One experiment per GPU pinned with `CUDA_VISIBLE_DEVICES`
  via `/workspace/runs/launch.sh`. PIDs tracked under `/workspace/runs/pids/`
  for clean kills.
- **Concurrency.** 6 rounds × ≈8 parallel 1-GPU jobs.
- **Wall-clock per experiment.** 90 min (round 1) → 180 min (round 2) →
  240–360 min (round 3) → 360–480 min (rounds 4–6).
- **Total budget consumed.** ≈140 GPU-hours across rounds 1–6 (of 192
  GPU-hours total available). The remaining ≈50 GPU-hours were used for
  monitoring, ensemble eval, and overhead.

## Reproduce the best single run (`r6e`)

```bash
cd /workspace/ml-intern-benchmark/target
TQDM_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 \
SENPAI_TIMEOUT_MINUTES=480 \
python ./train.py \
    --epochs 200 \
    --batch_size 4 \
    --loss_kind rel_l2 \
    --dropout 0.05 \
    --warmup_frac 0.10 \
    --grad_clip 1.0 \
    --seed 24 \
    --agent ml-intern-r3 \
    --wandb_group mlintern-pai2-24h-v3-r3 \
    --wandb_name "mlintern-pai2-24h-v3-r3/r6e_relL2_d005_w10_e200_s24"
```

Reproduce the top-10 ensemble:

```bash
python ensemble_eval.py \
    --checkpoint_paths "<top10 checkpoints, comma-separated>" \
    --config_paths     "<top10 configs, comma-separated>" \
    --batch_size 4
```

## W&B references

- Group: `mlintern-pai2-24h-v3-r3`
- Project: `wandb-applied-ai-team/senpai-v1-ml-intern`
- Best single run: `r6e_relL2_d005_w10_e200_s24` (run id `bdq5fhv4`)
- Best test single (by test_avg): `r5f_relL2_d01_w10_e150_s17` (run id `aznhx3l7`, test 28.75)

## Files credited to this replicate

- `train.py` — added CLI flags for architecture (`n_hidden`, `n_layers`,
  `n_head`, `slice_num`, `mlp_ratio`, `dropout`, `multiscale_slice`),
  optimizer/schedule (`beta2`, `grad_clip`, `warmup_frac`, `lr_schedule`,
  `timeout_min`), loss (`loss_kind` ∈ {mse,l1,rel_l2}, `p_weight`,
  `Ux_weight`, `Uy_weight`, `input_noise_sigma`), and mixed precision
  (`amp`). Added pre-filter for non-finite y in `evaluate_split` to fix
  the test-eval NaN propagation when `data/scoring.py` would otherwise
  contaminate `mae_*` accumulators with `inf*0.0`.
- `ensemble_eval.py` — new script. Loads N checkpoints, averages predictions
  in normalized space (with optional per-model weights via `--weights`),
  denormalizes, runs the same scoring helpers as `train.py` so val/test
  numbers are apples-to-apples comparable. Includes the same NaN pre-filter.
- `model_soup.py` — new script. Averages weights instead of predictions.
  Tested for completeness; weight averaging across seeds was much worse
  than prediction ensembling here (val 468 — different-seed checkpoints
  land in different loss basins, which weight averaging cannot navigate).
- `research/MLINTERN_RESULTS.jsonl` — one JSON record per W&B run.
- `research/MLINTERN_SUMMARY.md` — this document.

## Recommendations for next replicate

1. **Train longer.** The improvement curve flattens but doesn't plateau even
   at `--epochs 200`. A 12–24 h single-seed run with `--epochs 300+` and a
   `Cosine eta_min ≈ 1e-5` floor (instead of 0) is worth one experiment.
2. **Architectural diversity for ensemble.** Cross-arch ensembles (e.g.
   `5L/128H` + `6L/192H` + `4L/96H`) might give larger gains than the
   same-arch seed ensembles I built here. The seed ensemble of identical
   architectures saturated around 29 val despite individual models at 32–34.
3. **Targeted data augmentation.** Reflection symmetry on negative-AoA
   single-foil samples (negate y-coords + Uy + surface-normal y-component)
   is physically valid and would roughly double the effective training set
   for that domain (which is by far the largest).
4. **Investigate `val_geom_camber_rc`.** It's the hardest split (rc surf_p ≈
   42–45 across all configs vs cruise ≈ 16 and re_rand ≈ 31–32).
   Specialized features for camber-extrapolation (e.g. higher-frequency
   geometry encoding, foil-shape token) could move the needle there.
5. **Snapshot ensembling within a run.** All my ensembles use only the
   per-run BEST checkpoint. Saving the top-K checkpoints from each run
   (e.g. last 5 epochs of cosine decay) and including them in the final
   ensemble would likely give another 1–2 points on test without any new
   training compute.
