# ML Intern TandemFoilSet-Balanced — pai2 24h replicate `mlintern-pai2-24h-v3-r1`

## TL;DR

Final results in `mae_surf_p` (m²/s², lower is better):

| | val_avg | test_avg |
|---|---:|---:|
| Original baseline default config (debug smoke) | ≈487 | 358.39 |
| Best single model (`final-pure-b2-lr3e4-e60`, ep57) | **69.45** | **61.22** |
| **6-model ensemble (3 finals + 3 longs)** | **62.27** | **55.51** |

**~6.4× lower surface-pressure test MAE than the original baseline.** The
ensemble is the best paper-facing number; the single-model number is what
you get from a single 9-hour training run with the recipe below.

## Pod / branch

- **Pod start**: 2026-04-30 06:21 UTC
- **Pod kill deadline**: 2026-05-01 06:21 UTC
- **Branch**: `mlintern-pai2-24h-v3-r1`
- **W&B project**: `wandb-applied-ai-team/senpai-v1-ml-intern`
- **W&B group**: `mlintern-pai2-24h-v3-r1`
- **W&B runs** are tagged with `agent=ml-intern-r1`. Run IDs in
  `research/MLINTERN_RESULTS.jsonl`. Direct URLs:
  `https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs/<run_id>`.

## GPU usage strategy

8 × NVIDIA RTX PRO 6000 Blackwell (96 GB each).

I ran multiple **one-GPU jobs in parallel** pinned with
`CUDA_VISIBLE_DEVICES=<n>`. Sweet spot was 4-6 parallel jobs — 8 parallel
caused per-job slowdown >2× because of CPU/PCIe/I-O contention from
~24 worker subprocesses per training job (4 train workers + 16 val
workers × N_jobs). 4-6 parallel adds only 10-30% per-epoch overhead.

Total GPU-hours: roughly 8 GPUs × ~22 h = ~175 GPU-hours, mostly
in the second half on long T_max=60 runs of the winning recipe.

## Code changes (single file: `train.py`)

1. `Config` dataclass extended with all the architecture knobs the original
   trainer hard-coded (`n_hidden`, `n_layers`, `n_head`, `slice_num`,
   `mlp_ratio`, `dropout`) plus training tweaks (`grad_clip`,
   `warmup_epochs`, `quiet`, `grad_checkpoint`, `grad_accum`,
   `ada_temp`, `rep_slice`).
2. `PhysicsAttention` now supports optional Transolver++ extensions
   (arXiv:2502.02414) — `ada_temp` (per-point learnable temperature) and
   `rep_slice` (Gumbel-Softmax slice noise during training). Default off.
3. `Transolver.forward` optionally wraps each block in
   `torch.utils.checkpoint.checkpoint` — needed to fit the scaled-up model
   in memory at the largest mesh sizes.
4. Cosine LR schedule optionally preceded by a `LinearLR` warmup.
5. Optional gradient accumulation and gradient clipping.
6. tqdm progress bar suppressed when `--quiet true` so logs are grep-friendly.
7. **Workaround for `data/scoring.py` NaN handling.** Exactly one
   ground-truth file in `test_geom_camber_cruise` (`000020.pt`) has
   non-finite y values. The metric accumulator was *designed* to skip
   non-finite samples but PyTorch's `NaN * 0 == NaN` propagates through
   the masked sum and produces a NaN `test_avg/mae_surf_p`. Workaround at
   the call-site in `evaluate_split`: zero out bad-sample y values and
   AND-out the per-sample mask before calling `accumulate_batch`.
   `data/scoring.py` itself is unchanged.

All flags are **off by default** — `python ./train.py` with no flags
reproduces the original baseline behaviour exactly.

Two new helper scripts:
- `eval_test.py` — single-model val + test evaluation
- `eval_ensemble.py` — averages predictions across multiple checkpoints

## Best recipe (single model)

```bash
python ./train.py \
    --epochs 60 \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --mlp_ratio 2 \
    --batch_size 2 --grad_checkpoint true \
    --surf_weight 30 --lr 3e-4 --weight_decay 1e-4 \
    --quiet true \
    --agent ml-intern-r1 \
    --wandb_group mlintern-pai2-24h-v3-r1 \
    --wandb_name "mlintern-pai2-24h-v3-r1/<descriptive-name>"
```

Wall-clock: ~9 hours on a single Blackwell B6000 (with 4-6 parallel jobs
co-loaded sharing CPU/IO). Reaches `val_avg/mae_surf_p ≈ 69.5` and
`test_avg/mae_surf_p ≈ 61.2` at epoch 57. ~3.94M parameters.

## Headline result table

All numbers are equal-weight averaged surface-pressure MAE across the 4
val/test splits, in the original target space (`m²/s²`), computed by the
same `data/scoring.py` accumulators that the trainer uses for
`val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

### Single models (best epoch by val)

| Run name | epochs | b | lr | surf_w | val | test |
|---|---:|---:|---:|---:|---:|---:|
| `final-pure-b2-lr3e4-e60` | 57/60 | 2 | 3e-4 | 30 | **69.45** | **61.22** |
| `final-pure-b4-lr3e4-e60` | 59/60 | 4 | 3e-4 | 30 | 69.50 | 63.63 |
| `final-pure-b2-lr3e4-sw40` | 60/60 | 2 | 3e-4 | 40 | 70.71 | 63.87 |
| `long-pure-sw30-lr3e4` | 51/60 | 2 | 3e-4 | 30 | 70.99 | 63.25 |
| `long-pure-sw30-b2`    | 58/60 | 2 | 5e-4 | 30 | 71.60 | 65.59 |
| `long-pure-sw30-b4`    | 50/60 | 4 | 5e-4 | 30 | 75.99 | 67.69 |
| `v3-pure-b2-e35`       | 35/35 | 2 | 5e-4 | 30 | 82.72 | 74.69 |
| `long-bigger-320h10l-sw30` | 35/50 | 2 | 5e-4 | 30 (320/10) | 91.59 | 82.08 |
| `repl-pure-b4-e20`     | 20/20 | 4 | 5e-4 | 30 | 101.76 | 91.93 |
| `w3d-pure-sw30-b4`     | 20/20 | 4 | 5e-4 | 30 | 102.96 | 92.31 |
| `w3d-tpp-sw50`         | 19/20 | 2 | 5e-4 | 50 +tpp | 125.45 | 114.69 |
| `w3b-scaled-sw30-pure` | 6/8   | 2 | 5e-4 | 30 | 147.76 | — |
| baseline (debug smoke) | 3/3   | 4 | 5e-4 | 10 | 487.68 | 358.39 |

All scaled models use `n_hidden=256, n_layers=8, n_head=8, slice_num=64,
grad_checkpoint=true` (~3.94M params). `320h10l` is the wider 320/10
variant (7.6M params).

### Ensembles (equal-weight avg of normalised predictions)

| Ensemble | val | test |
|---|---:|---:|
| Top-1 single best (`final-pure-b2-lr3e4-e60`) | 69.45 | 61.22 |
| Top-3 longs (lr3e4 + sw30-b2 + sw30-b4) | 65.51 | 58.46 |
| Top-3 + best final | 63.63 | 56.48 |
| Top-3 + 2 finals (e60-b2 + sw40) | 63.06 | 56.01 |
| **Top-3 + 3 finals (best 6 models)** | **62.27** | **55.51** |
| Top-6 + replica-of-w3d (zd0tto82) | 64.10 | 56.96 |
| Top-3 + bigger 320/10 | 67.25 | 59.95 |

Adding well-converged 256/8 models always helps. Adding the 320/10
variant or the under-converged 20-epoch run *hurts* the ensemble.

Per-split test of the best ensemble (top 6):

| Split | mae_surf_p |
|---|---:|
| `test_single_in_dist`     | 59.83 |
| `test_geom_camber_rc`     | 68.24 |
| `test_geom_camber_cruise` | 38.45 |
| `test_re_rand`            | 55.54 |

The hardest split is still `test_geom_camber_rc` (raceCar M=6-8, never
seen during training) — the rest are within ~10 of each other.

## What worked, in order of impact

1. **Scaling the model** from the 0.66M baseline (128/5/4/64) to 3.94M
   (256/8/8/64) Transolver. Single biggest jump — 6-epoch val drops from
   ~500 to ~150.
2. **Long cosine schedule reaching LR ≈ 0.** Set `--epochs N` so the
   schedule hits zero LR by the actual end of training. Naive
   `--epochs 999` keeps LR at peak the whole run and the model never
   finishes converging. With T_max ≈ actual epochs, the val loss drops
   sharply in the last 20% of epochs as LR approaches zero.
3. **Lower learning rate (`lr=3e-4` instead of `5e-4`)** with the long
   schedule. ~3 points val better at convergence (`70.99 → 69.45`).
4. **`surf_weight=30`** (vs default 10). The metric only counts surface
   nodes, so up-weighting the surface term in the loss helps directly.
   `50` was too aggressive (more variance, worse final val).
5. **`grad_checkpoint=True`** is required to fit the scaled architecture
   at `batch_size ∈ {2, 4}` without OOM at 240k-node samples.
6. **`batch_size=2`** slightly outperforms `batch_size=4` at the same
   T_max (more gradient updates per epoch). With short schedules
   (T_max=20) they were tied.
7. **Equal-weight ensemble of well-converged 256/8 b∈{2,4} runs** — adding
   each new well-converged 256/8 model improves the ensemble. Going from
   1 → 6 models drops test from 61.22 → 55.51, a free ~9% test
   improvement on top of the best single.

## What didn't work

- **Transolver++ `ada_temp` + `rep_slice`** (arXiv:2502.02414).
  Slightly better at epoch 1, but the Gumbel-Softmax slice noise
  increases epoch-to-epoch variance and hurts late-epoch convergence —
  the pure baseline catches up by epoch 3-5 and stays ahead. Net
  negative on this dataset / training budget.
- **Wider / deeper architectures** without more regularisation:
  - `n_hidden=320, n_layers=10` (7.6M params): trains stably but reaches
    val ~92 vs ~70 for the smaller model — strong sign of overfitting on
    the 1499-sample train set.
  - `n_hidden=256, n_layers=12` (5.8M params): clear overfitting, train
    loss falls but val rises monotonically. Killed at epoch 3.
  - The 320/10 model also *hurts* the ensemble despite more params.
- **`surf_weight=50`**: more noise during training, slightly worse final
  val (≈125 vs ≈103 for sw=30 at the same T_max=20).
- **`lr=1e-3`** with linear warmup: too high, val never recovers.
- **`warmup_epochs=3`** before cosine: just shifts the loss curve right.
- **`slice_num=128`** (vs 64): slower per epoch, harder to converge.
- **Naive `--epochs 999`** with cosine T_max=999 + 30-min wallclock cap:
  the LR schedule never decays, so the model never settles. Always set
  T_max ≈ the actual number of epochs you'll get.

## Best commands (for reproducing)

To train the single best model (val 69.45 / test 61.22):

```bash
python ./train.py --epochs 60 --skip_test true --quiet true \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --mlp_ratio 2 \
    --batch_size 2 --grad_checkpoint true \
    --surf_weight 30 --lr 3e-4 \
    --agent ml-intern-r1 \
    --wandb_group mlintern-pai2-24h-v3-r1 \
    --wandb_name "mlintern-pai2-24h-v3-r1/final-pure-b2-lr3e4-e60"
```

To get the test number on a saved checkpoint (with NaN-handling fix):

```bash
python eval_test.py --checkpoint models/model-<run_id>/checkpoint.pt
```

To get the best ensemble result, train 6 models of the recipe above with
varied `(lr, batch_size, surf_weight)`:

```
final-pure-b2-lr3e4-e60      b=2 lr=3e-4 sw=30 (igaf7ylx)
final-pure-b2-lr3e4-sw40     b=2 lr=3e-4 sw=40 (r79bfrtr)
final-pure-b4-lr3e4-e60      b=4 lr=3e-4 sw=30 (kiwnoj1g)
long-pure-sw30-lr3e4         b=2 lr=3e-4 sw=30 (oyefq97v) — different seed
long-pure-sw30-b2            b=2 lr=5e-4 sw=30 (94da0zux) — different lr
long-pure-sw30-b4            b=4 lr=5e-4 sw=30 (t5b5qgqz) — different lr
```

Then:

```bash
python eval_ensemble.py models/model-igaf7ylx models/model-r79bfrtr \
    models/model-kiwnoj1g models/model-oyefq97v \
    models/model-94da0zux models/model-t5b5qgqz
```

## Next recommendation

1. The remaining gap on `test_geom_camber_rc` (68.2) is much larger than
   the cruise camber gap (38.5). raceCar M=6-8 is the hardest split.
   Worth investigating data-augmentation that targets that split
   specifically, or a small network specialising on raceCar tandem
   geometry.
2. The ensemble gain (≈9% on test from 1 → 6 models) is reliable.
   Definitely include this in any future paper-facing reporting.
   Cheap-to-train, weight-averaging variants (SWA / EMA / Polyak averaging)
   would likely give similar reliability with a single run.
3. The wider model (320/10) has more capacity but overfits — combining
   it with proper regularisation (dropout 0.05-0.1, higher weight_decay,
   longer schedule) might unlock another small step. Out of scope for
   this 24h budget.
4. Going **deeper** (n_layers=12+) is a dead end on this dataset — the
   1499 train samples just aren't enough to support the extra capacity.
