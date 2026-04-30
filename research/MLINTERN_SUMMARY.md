# ML Intern TandemFoilSet-Balanced Benchmark — Replicate `mlintern-pai2-24h-v3-r3`

## Strategy summary

Iterative experiment program optimizing `val_avg/mae_surf_p` (equal-weight surface
pressure MAE across the four val splits) with `test_avg/mae_surf_p` reported on
the best ensemble of single-model checkpoints at the end. All training was run
locally on the pai2 pod across 8× RTX PRO 6000 (97 GB) GPUs in parallel, one
1‑GPU job per experiment.

Five rounds, each informed by the prior round's results:

1. **Round 1 — capacity sweep.** 8 parallel 90‑min runs comparing the baseline
   (`5L/128H/4head/sn64`) against AirfRANS-style upgrades (`8L/256H/8head/sn{16,32,64}`),
   higher `surf_weight`, deeper variants, and `mlp_ratio=4`. The bigger 8L/256H
   models overfit quickly on this 1499-sample dataset (best at epoch 4–5, then
   regressed); the original baseline architecture won at epoch 11–18 with
   `val_avg/mae_surf_p ≈ 132`. Default `--epochs 999` produced essentially flat
   LR (cosine T_max=999), so the per-epoch oscillation was the dominant signal.
2. **Round 2 — training-recipe refinements on the baseline arch.** 6 parallel
   180‑min runs with `--epochs 40` (proper cosine decay over the realized
   training duration), warmup, gradient clipping, and one of: dropout=0.1,
   relative-L2 loss, p-channel up-weighting, lr=2e-4, input noise, default.
   The relative-L2 loss (per-sample-and-region L2/||y||) was the clear winner
   (`r2d`: val 56.7); switching from `epochs=999` flat LR to `epochs=40`
   cosine drove the dramatic improvement across all variants.
3. **Round 3 — extended training + ensemble candidates.** 9 runs total. The
   biggest single-run win came from extending `epochs` and adding light dropout:
   `r3a` (relL2 + dropout=0.1 + warmup=0.05 + grad_clip=1.0 + `--epochs 80`,
   6h timeout) reached `val_avg/mae_surf_p = 40.4` (test 35.6 with the NaN
   fix). Three seeds at `--epochs 50` reached val 51–52, confirming that the
   "long training at gradually-decreasing LR" was the lever, not the seed.
4. **Round 4 — long training, 8 diverse-config seeds.** 8 parallel 6–8h runs
   sweeping dropout in {0.0, 0.05, 0.1, 0.15} and using `--epochs 100/80`.
   `r4a` and `r4b` (dropout=0.05 vs 0.1, both `--epochs 100`) both hit val ≈ 37.
5. **Round 5 — ensemble-targeted final batch.** 8 parallel runs with
   `--epochs 120/150` and the best round-4 configs (dropout in {0.05, 0.075,
   0.1, 0.15}, varying warmup and seeds) to populate a strong ensemble.
   Running at time of writing.

## Key technical findings

- **Architecture.** Baseline `5L/128H/4head/sn64` Transolver (≈0.66M params) is
  the right capacity for 1499 train samples. Bigger models (8L/256H/8head)
  overfit before any of the regularizers below could compensate. The
  AirfRANS-recommended `slice_num=32` and `unified_pos=1` were tested and not
  better here (smaller dataset, different geometry distribution).
- **Loss.** Switching the training loss from MSE to **per-sample relative L2**
  (`||pred - y|| / (||y|| + eps)` averaged over batch, separate for surface
  and volume regions, then `vol + λ·surf`) delivered the largest single-step
  improvement (val 62 → 57 → 41 across rounds 2–3). This equalizes the
  contribution of low-pressure and high-pressure samples, which is critical
  given the 10× per-sample y-std spread.
- **Schedule.** Warmup (5–10% of epochs) + cosine decay over the actual
  training duration. The training script's original `--epochs 999` default
  pinned `T_max=999`, leaving LR essentially flat across realized training
  and producing noisy late-stage val. Setting `--epochs` close to the wall
  number of epochs realized was as important as the loss change.
- **Regularization.** Light dropout in the Physics-Attention block
  (`dropout=0.05–0.1`) helped marginally. `weight_decay=1e-4`, `grad_clip=1.0`
  were standard. Input noise on normalized features (`σ=0.05`) was neutral.
  Increasing `surf_weight` above the default 10 hurt; changing `p_weight` had
  no consistent effect.
- **Test eval bug fix.** One sample in `test_geom_camber_cruise` (`000020.pt`)
  has +Inf in y. `data/scoring.py` is documented to skip non-finite samples,
  but `(pred - inf).abs() = inf` then `inf * 0.0 = NaN` propagates through
  the masked sum and contaminates the per-channel MAE accumulator. Running
  the train.py end-of-run test eval with the original code wrote
  `test_avg/mae_surf_p = NaN` to W&B for early runs.
  **Fix in `train.py:evaluate_split` and `ensemble_eval.py`/`model_soup.py`:**
  pre-filter the batch to drop non-finite-y samples before invoking the
  scoring helpers. `data/scoring.py` is left untouched per the read-only
  contract; the bad sample is correctly excluded from the metric (matches
  the documented "per-sample skipping for non-finite ground truth").

## Best results

(Updated periodically; final numbers in MLINTERN_RESULTS.jsonl.)

| Model / Ensemble | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| `r4b` (drop=0.05, --epochs 100, seed=5) — best single | 37.12 | 32.91 |
| `r4a` (drop=0.1, --epochs 100, seed=4) | 37.85 | 32.86 |
| `r4h` (drop=0.1, warm=0.10, --epochs 80, seed=11) | 39.65 | 35.44 |
| `r3a` (drop=0.1, --epochs 80, seed=42) | 40.42 | 35.65 (re-evaluated with fix) |
| 3‑model ensemble (r4b + r4a + r4h) | 34.53 | 30.33 |
| 5‑model ensemble (top 5 finished) | 34.55 | 30.11 |
| Round 5 winners (in progress) | tbd | tbd |

## Compute / GPU strategy

- **Hardware.** 8× NVIDIA RTX PRO 6000 Blackwell, 97 GB each.
- **Allocation.** One experiment per GPU pinned with `CUDA_VISIBLE_DEVICES`
  via `/workspace/runs/launch.sh`. PIDs tracked under `/workspace/runs/pids/`
  for clean kills.
- **Concurrency.** 5 rounds × ≈8 parallel 1‑GPU jobs. No GPU sharing during
  training; ensemble eval squeezed onto a GPU whose train job had finished.
- **Wall clock per experiment.** 90 min (round 1) → 180 min (round 2) →
  240–360 min (round 3) → 360–480 min (rounds 4–5).
- **Cost-equivalent budget.** ≈100 GPU-hours used across rounds 1–5 (of 192
  GPU-hours total available).

## Reproduce the best single run

```bash
cd /workspace/ml-intern-benchmark/target
TQDM_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 \
SENPAI_TIMEOUT_MINUTES=480 \
python ./train.py \
    --epochs 100 \
    --batch_size 4 \
    --loss_kind rel_l2 \
    --dropout 0.05 \
    --warmup_frac 0.05 \
    --grad_clip 1.0 \
    --seed 5 \
    --agent ml-intern-r3 \
    --wandb_group mlintern-pai2-24h-v3-r3 \
    --wandb_name "mlintern-pai2-24h-v3-r3/r4b_relL2_drop005_long_s5"
```

Reproduce the 3-model ensemble:

```bash
python ensemble_eval.py \
    --checkpoint_paths "models/model-<r4b-id>/checkpoint.pt,models/model-<r4a-id>/checkpoint.pt,models/model-<r4h-id>/checkpoint.pt" \
    --config_paths    "models/model-<r4b-id>/config.yaml,models/model-<r4a-id>/config.yaml,models/model-<r4h-id>/config.yaml" \
    --batch_size 4
```

## Files credited to this replicate

- `train.py` — added CLI flags for architecture, optimizer/schedule, loss
  kind / per-channel weights / input noise, mixed precision, and multiscale
  slice config; added pre-filter for non-finite y in `evaluate_split` to fix
  the test-eval NaN propagation when `data/scoring.py` would otherwise
  contaminate `mae_*` accumulators with `inf*0.0`.
- `ensemble_eval.py` — new script. Loads N checkpoints, averages predictions
  in normalized space, denormalizes, runs the same scoring helpers as
  `train.py` so val/test numbers are apples-to-apples comparable.
- `model_soup.py` — new script. Averages weights instead of predictions
  (tested for completeness; weight averaging across seeds was much worse than
  prediction ensembling here).
- `research/MLINTERN_RESULTS.jsonl` — one JSON record per W&B run.
- `research/MLINTERN_SUMMARY.md` — this document.

## Recommendations for next replicate

1. **Train longer.** The improvement curve flattens but doesn't plateau even at
   `--epochs 120`. A 12–24 h single-seed run with `--epochs 200` and a `Cosine
   eta_min ≈ 1e-5` floor (instead of 0) is worth one experiment.
2. **Architectural diversity for ensemble.** Cross-arch ensembles (e.g.
   `5L/128H` + `6L/192H` + `4L/96H`) might give larger gains than the
   same-arch seed ensembles I built here. The seed ensemble of identical
   architectures saturated around 34.5 val despite individual models at 37–42.
3. **Targeted data augmentation.** Reflection symmetry on negative-AoA
   single-foil samples (negate y-coords + Uy + surface-normal y-component) is
   physically valid and would roughly double the effective training set for
   that domain (which is by far the largest).
4. **Investigate `val_geom_camber_rc`.** It's the hardest split (rc surf_p ≈
   50 across all configs vs cruise ≈ 21 and re_rand ≈ 36). Specialized
   features for camber-extrapolation (e.g. higher-frequency geometry encoding)
   could move the needle there.
