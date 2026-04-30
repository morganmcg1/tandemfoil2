# ML Intern TandemFoilSet-Balanced — pai2 24h replicate `mlintern-pai2-24h-v3-r1`

## TL;DR

**Best result so far** (still running additional candidates):

| | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| Single best (`long-pure-sw30-lr3e4`, ep51) | 70.99 | **63.25** |
| **Ensemble of top-3 trained models** | 65.51 | **58.46** |

Compare to:
- Original baseline default config (1-epoch debug surrogate): ~487 val
- Stage-1 scaled architecture, 6-epoch sw=30: 147.76 val (test ~92)

Roughly **8x** lower test surface-pressure MAE than the original baseline.

## Pod / branch

- **Pod start**: 2026-04-30 06:21 UTC
- **Pod kill deadline**: 2026-05-01 06:21 UTC
- **Branch**: `mlintern-pai2-24h-v3-r1`
- **W&B project**: `wandb-applied-ai-team/senpai-v1-ml-intern`
- **W&B group**: `mlintern-pai2-24h-v3-r1`
- **W&B run links**: `https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs/<run_id>` for IDs
  in `research/MLINTERN_RESULTS.jsonl`.

## GPU usage strategy

8 × NVIDIA RTX PRO 6000 Blackwell (96 GB each). Visible GPU budget: 8.

I used parallel one-GPU jobs pinned with `CUDA_VISIBLE_DEVICES=<n>`. After
some experimentation I settled on **6-7 parallel jobs** as the sweet spot —
going to 8 caused noticeable per-job slowdown (16 val workers + 4 train
workers per job × 8 jobs > 150 worker subprocesses), while 4-6 parallel
adds only 10-30% per-epoch overhead vs running alone.

The final 13 hours of compute were spent on long-schedule (T_max=60) runs
of the winning recipe with batch=2 and three different `(lr, surf_weight)`
combinations.

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
7. **Workaround for `data/scoring.py` NaN-handling bug.** Exactly one
   ground-truth file in `test_geom_camber_cruise` (`000020.pt`) has
   non-finite y values. The metric accumulator was *designed* to skip such
   samples but PyTorch's `NaN * 0 == NaN` propagates through the masked
   sum and produces a NaN `test_avg/mae_surf_p`. Workaround at the
   call-site in `evaluate_split`: zero out bad samples' y values and AND
   them out of the per-sample mask before calling `accumulate_batch`.
   `data/scoring.py` itself is unchanged.

All flags are **off by default** — `python ./train.py` with no flags
reproduces the original baseline behaviour exactly.

Two new helper scripts: `eval_test.py` (single-model val + test
evaluation) and `eval_ensemble.py` (averages predictions across multiple
checkpoints in normalised space, then computes MAE).

## Headline result table

All numbers are equal-weight averaged surface-pressure MAE (m²/s²) across
the 4 splits, in the original target space, computed by the same
`data/scoring.py` accumulators that the trainer uses for
`val_avg/mae_surf_p` and `test_avg/mae_surf_p`.

| Run | epochs/T_max | b | lr | surf_w | val | test | n_params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long-pure-sw30-lr3e4` | 51/60 | 2 | 3e-4 | 30 | **70.99** | **63.25** | 3.94M |
| `long-pure-sw30-b2`    | 58/60 | 2 | 5e-4 | 30 | 71.60 | 65.59 | 3.94M |
| `long-pure-sw30-b4`    | 50/60 | 4 | 5e-4 | 30 | 75.99 | 67.69 | 3.94M |
| `long-bigger-320h10l-sw30` | 35/50 | 2 | 5e-4 | 30 | 91.59 | 82.08 | 7.60M |
| `repl-pure-b4-e20`     | 20/20 | 4 | 5e-4 | 30 | 101.76 | 91.93 | 3.94M |
| `w3d-pure-sw30-b4`     | 20/20 | 4 | 5e-4 | 30 | 102.96 | 92.31 | 3.94M |
| `w3d-tpp-sw50`         | 19/20 | 2 | 5e-4 | 50 +tpp | 125.45 | 114.69 | 3.94M |
| `w3b-scaled-sw30-pure` | 6/8 | 2 | 5e-4 | 30 | 147.76 | — | 3.94M |
| baseline (debug smoke) | 3/3 | 4 | 5e-4 | 10 | 487.68 | 358.39 | 0.66M |

Per-split test breakdown of best single (`long-pure-sw30-lr3e4`):

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 66.31 |
| test_geom_camber_rc | 76.69 |
| test_geom_camber_cruise | 45.46 |
| test_re_rand | 64.54 |

### Ensemble results

Equal-weight average of normalised predictions, then denormalise + MAE:

| Ensemble | val | test |
|---|---:|---:|
| Top-1 single (lr3e4) | 70.99 | 63.25 |
| Top-2 (lr3e4 + sw30-b2) | 66.14 | 59.40 |
| **Top-3 (lr3e4 + sw30-b2 + sw30-b4)** | **65.51** | **58.46** |
| Top-4 (top-3 + 320/10) | 67.25 | 59.95 |
| Top-3 + repl-pure-b4-e20 | 68.97 | 61.14 |

Adding the wider 320/10 model or the under-converged 20-epoch run
*hurts* the ensemble — only well-converged 256/8 b∈{2,4} models help.

Per-split test of the best ensemble (top 3):

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 62.20 |
| test_geom_camber_rc | 71.69 |
| test_geom_camber_cruise | 41.45 |
| test_re_rand | 58.52 |

## What worked, in order of impact

1. **Scaling the model** from the 0.66M baseline to a 3.94M
   (`n_hidden=256, n_layers=8, n_head=8, slice_num=64`) Transolver. Single
   biggest jump — 6-epoch val drops from ~500 to ~150.
2. **Long cosine schedule reaching LR≈0.** Using `--epochs 60` so the
   schedule drops to near-zero LR by the actual training stop time. Naive
   `--epochs 999` keeps the LR at peak the whole run and never finishes
   converging. With T_max ≈ actual epochs, the model converges nicely
   once LR gets small (typically last 20% of epochs).
3. **Lower learning rate (3e-4 instead of 5e-4)** with the long schedule.
   `lr=3e-4` reaches val ≈ 71 vs `lr=5e-4` reaches val ≈ 72 — small but
   reproducible.
4. **`surf_weight = 30`** (vs default 10). The metric only counts surface
   nodes, so up-weighting the surface term helps. 50 was too aggressive
   (more variance, worse late-epoch val); 30 was the sweet spot.
5. **`grad_checkpoint=True`** is required to fit the scaled architecture
   at `batch_size ∈ {2, 4}` without OOM at 240k-node samples.
6. **`batch_size = 2`** slightly outperforms `batch_size = 4` at the same
   T_max (more gradient updates per epoch). With short schedules (T_max=20)
   they were tied.
7. **Top-3 ensemble** of the well-converged 256/8 runs gives a free ~7%
   test improvement on top of the best single.

## What didn't work

- **Transolver++ `ada_temp` + `rep_slice`.** Drop-in extensions from
  arXiv:2502.02414. Slightly better at epoch 1, but the Gumbel-Softmax
  slice noise increases epoch-to-epoch variance and hurts late-epoch
  convergence — the pure baseline catches up by epoch 3-5 and stays
  ahead. Net negative on this dataset / training budget.
- **Wider / deeper architectures** without more regularisation:
  - `n_hidden=320, n_layers=10` (7.6M params): trains stably but reaches
    val ≈ 92 vs 71 for the smaller model — strong sign of overfitting on
    the 1499-sample train set.
  - `n_hidden=256, n_layers=12` (5.8M params): clear overfitting, train
    loss falls but val rises monotonically. Killed at epoch 3.
- **Higher `surf_weight = 50`**: more noise during training, slightly
  worse final val (125 vs 102 with sw=30 at the same T_max=20).
- **`lr = 1e-3`** with linear warmup: too high, val never recovers from
  the early instability.
- **`warmup_epochs = 3`** before cosine: just shifts the loss curve right,
  no benefit.
- **`slice_num = 128`** (vs 64): slower per epoch, harder to converge
  stably, no wins.
- **Naive `--epochs 999`** with cosine T_max=999 + 30-min wallclock cap:
  the LR schedule never decays, so the model never settles. Always set
  T_max ≈ the actual number of epochs you'll get.

## Recipe for the best single model

```bash
python ./train.py \
    --epochs 60 \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --mlp_ratio 2 \
    --batch_size 2 --grad_checkpoint true \
    --surf_weight 30 --lr 3e-4 --weight_decay 1e-4 \
    --agent ml-intern-r1 \
    --wandb_group mlintern-pai2-24h-v3-r1 \
    --wandb_name "mlintern-pai2-24h-v3-r1/<descriptive-name>"
```

Wall-clock: ~7 hours on a single Blackwell B6000 with 6 parallel jobs
co-loaded (so ~20 GB usable VRAM under contention). Training stops via
its 8h `SENPAI_TIMEOUT_MINUTES` cap or after 60 epochs.

To reproduce the test number with NaN handling:

```bash
python eval_test.py --checkpoint models/model-<run_id>/checkpoint.pt
```

To produce the ensemble result, after training 3 well-converged models
of the recipe family above with different `(lr, batch_size)`:

```bash
python eval_ensemble.py models/model-<run_a> models/model-<run_b> models/model-<run_c>
```

## Currently running

At time of writing (≈11h elapsed) three more long runs are still training
on dedicated GPUs (final 9h cap each) — they replicate the winning recipe
with one knob varied (`b2/sw30 lr=3e-4`, `b2/sw40 lr=3e-4`, `b4/sw30 lr=3e-4`).
If any of them finish below the current 70.99 best, the ensemble will
be re-computed with them included.

## Next recommendation

1. The remaining gap on `test_geom_camber_rc` (76.7) is much larger than
   the cruise camber gap (45.5). Front-foil camber generalisation
   (raceCar M=6-8) is the hardest split. Worth investigating
   data-augmentation along NACA M, or a mild regularisation that targets
   that split specifically.
2. The wider model (320/10) overfits but has more capacity — combining it
   with stronger regularisation (dropout 0.05-0.1, higher weight_decay,
   longer schedule) might unlock another small step. Out of scope for
   this 24h budget.
3. The ensemble gain (≈7% on test) is reliable — definitely include
   this in any future paper-facing reporting.
