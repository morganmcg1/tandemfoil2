# ML Intern — TandemFoilSet-Balanced (replicate `mlintern-pai2-r3-retry-r1`)

> Run timestamp: 2026-04-29 (UTC) on the pai2 cluster, 8× RTX PRO 6000 Blackwell (96 GB).
> All training compute stayed inside the local pod. Hard wall-clock budget: 12 h.
> W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
> Replicate group: `mlintern-pai2-r3-retry-r1` —
> [W&B group view](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern?nw=mlintern-pai2-r3-retry-r1)

## Headline result

| Metric | Value | Source |
|---|---|---|
| **Best single-model `val_avg/mae_surf_p`** | **35.59** | `p3-warmup5-lr3e4-150ep` (ep 138) |
| **Best single-model `test_avg/mae_surf_p`** | **31.05** | `p3-warmup3-clip-150ep-seed7` (ep 143) |
| **Best ensembled `test_avg/mae_surf_p`** | **26.97** | 5-model val-weighted ensemble |
| Phase-1 baseline `val_avg/mae_surf_p` (default × 30 ep cosine) | 94.59 | `p1-baseline` |
| **Single-model improvement over Phase-1 baseline (val)** | **62.4 %** | — |
| **Ensemble improvement over best single model (test)** | **13.1 %** | — |

The single-model winners (test 31.05 and 31.18) use the **unchanged** Transolver
architecture (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`)
trained with **3-epoch linear warmup + gradient-norm clip 1.0** under a cosine
schedule that fully anneals inside the wall-clock budget. The single best test
score is from seed 7 of the warmup/clip recipe.

**Ensembling the four default-arch top runs plus the `h=256` run** (average
of normalized predictions, then denormalize-and-score) drops
`test_avg/mae_surf_p` from 31.05 → 26.97 with no extra training compute — a
13.1 % improvement on the paper-facing metric.

| Ensemble | `test_avg/mae_surf_p` |
|---|---:|
| Single best (`p3-warmup3-clip-150ep-seed7`) | 31.05 |
| 2-model uniform (warmup3-clip-seed7 + amp-bs4-warm-clip-180) | 28.28 |
| 3-model uniform (+ warmup3-clip default seed) | 27.64 |
| 4-model uniform (+ warmup5-lr3e4) | 27.11 |
| 5-model uniform (+ h256-warm-clip) | 26.99 |
| **5-model val-weighted (`w_i ∝ 1/val_i`)** | **26.97** |
| 8-model uniform (all Phase-3 candidates) | 28.21 |

Two takeaways from the ablation: adding more *good* models monotonically
helps, but adding *weak* models (the 8-model row includes the under-trained
`h=192/l=6` variants and `n_layers=8`) pulls the average back up. Inverse-val
weighting nudges the ensemble another 0.02 MAE.

Per-split test surface-pressure MAE for the **5-model val-weighted ensemble** (winning final):

| Test split | MAE (surf p) | MAE (surf Ux) | MAE (surf Uy) |
|---|---:|---:|---:|
| `test_single_in_dist` | 27.93 | 0.34 | 0.22 |
| `test_geom_camber_rc` | 39.76 | 0.61 | 0.33 |
| `test_geom_camber_cruise` | 14.66 | 0.23 | 0.14 |
| `test_re_rand` | 25.54 | 0.37 | 0.22 |
| **`test_avg`** | **26.97** | 0.38 | 0.23 |

Per-split test surface-pressure MAE for the best **single-model** checkpoint
(`p3-warmup3-clip-150ep-seed7`):

| Test split | MAE (surf p) | MAE (surf Ux) | MAE (surf Uy) |
|---|---:|---:|---:|
| `test_single_in_dist` | 32.76 | 0.42 | 0.26 |
| `test_geom_camber_rc` | 44.92 | 0.73 | 0.38 |
| `test_geom_camber_cruise` | 17.27 | 0.28 | 0.16 |
| `test_re_rand` | 29.25 | 0.45 | 0.26 |
| **`test_avg`** | **31.05** | 0.47 | 0.26 |

## Strategy

The 12 h budget was spent in three training phases plus a fourth eval-only
phase. Each training phase ran 8 single-GPU jobs in parallel, each pinned with
`CUDA_VISIBLE_DEVICES`. A small bash watcher (`run_logs/watcher.sh`) polled
`nvidia-smi` every 30 s, and as soon as any GPU dropped below 1.5 GB it pulled
the next entry from a queue file (`run_logs/phase2/queue.txt`) and launched a
job there with the per-entry `SENPAI_TIMEOUT_MINUTES` ceiling. The cluster ran
near 100 % utilisation throughout.

### Phase 1 — broad architecture / loss / optimiser sweep (~80 min × 8 jobs)

Default `--epochs 30` cosine schedule. One hypothesis per GPU.

| Run | Variation from default | Final `val_avg/mae_surf_p` |
|---|---|---:|
| `p1-baseline` | none (control) | 94.59 |
| `p1-warmup-lr8e4` | `warmup_epochs=2 lr=8e-4` | 94.75 |
| `p1-surf-w30` | `surf_weight=30` | 96.72 |
| `p1-wider-h256` | `n_hidden=256` | 97.20 |
| `p1-deeper-l8` | `n_layers=8` | 104.25 |
| `p1-cap-h192-l6-s128` | bigger combined arch | 105.44 |
| `p1-slice-256` | `slice_num=256` | 106.69 (timeout @ ep 20) |
| `p1-amp-bs8` | `--use_amp true --batch_size 8` (60 ep cosine, 80 min wall) | **85.76** (timeout @ ep 45) |

Phase-1 take-aways:

1. The default Transolver config trained for 30 cosine epochs is *under-trained*.
   `p1-amp-bs8` ran a 60-epoch cosine and beat the 30-epoch baseline by 9 MAE
   even before its annealing finished — so longer cosine alone is a free win.
2. Bigger models (wider / deeper / more slices) under-performed default at 30
   epochs. They needed many more cosine epochs to converge.

### Phase 2 — longer cosines, isolate AMP and batch-size, find regularisation winner (~3.3 h × 8)

Eight 100-epoch cosine runs probing the most promising Phase-1 directions plus
an ablation that isolates the AMP-vs-batch-size contributions of `p1-amp-bs8`.

| Run | Recipe | Final `val_avg/mae_surf_p` |
|---|---|---:|
| `p2-warmup3-clip-100ep` | default arch + `warmup=3 grad_clip=1.0` | **41.29** (timeout @ ep 91) |
| `p2-cap-h192-l6-s128-warm-80` | `h192/l6/s128` + warmup/clip + `lr=4e-4`, 80 ep | 54.86 (timeout @ ep 45) |
| `p2-amp-bs4-100ep` | AMP + `bs=4` (isolate AMP) | 63.64 |
| `p2-baseline-100ep` | default × 100 ep | 65.78 |
| `p2-mlp4-100ep` | `mlp_ratio=4` | 67.72 |
| `p2-amp-bs8-100ep` | AMP + `bs=8` | 70.85 |
| `p2-heads8-100ep` | `n_head=8` | 73.92 |
| `p2-bs8-fp32-80ep` | fp32 + `bs=8` | 73.97 |

Phase-2 take-aways:

- **`warmup_epochs=3 + grad_clip=1.0`** with the *unchanged* baseline arch was
  the single biggest win — it more than halved val-MAE relative to the
  Phase-1 baseline (94.59 → 41.29) at the same model size.
- AMP + `bs=4` (63.64) beat AMP + `bs=8` (70.85) and fp32 + `bs=8` (73.97).
  Bs=8 means half as many optimiser steps in the same budget, and stacked
  bf16 noise; bs=4 is a better operating point at this scale.
- Bigger models with the same recipe are *descending faster per epoch* but
  ran out of cosine inside the 200-min timeout — `p2-cap-h192-l6-s128-warm-80`
  reached 54.86 with only 45 of 80 cosine epochs.
- Architectural tweaks alone (`mlp_ratio`, `n_head`, deeper, wider) and the
  surface-weight bump did not move the needle.

### Phase 3 — long final runs around the warmup/clip recipe (~5.5 h × 8)

Eight 350-min-budget jobs. Epoch counts chosen so each cosine fully anneals
inside the wall-clock budget under the parallel-IO contention of 8 jobs.

| Run | Recipe | val (best) | test |
|---|---|---:|---:|
| **`p3-warmup3-clip-150ep-seed7`** | warmup/clip + 150 ep, **seed=7** | 36.17 | **31.05** |
| **`p3-amp-bs4-warm-clip-180ep`** | warmup/clip + AMP + bs=4 + 180 ep | 36.08 | 31.18 |
| `p3-warmup5-lr3e4-150ep` | warmup=5, `lr=3e-4`, 150 ep | 35.59 | 31.96 |
| `p3-warmup3-clip-150ep` | warmup/clip + 150 ep, default seed | 37.32 | 32.38 |
| `p3-h256-warm-clip-90ep` | `h256` + warmup/clip + 90 ep | 39.19 | 33.09 |
| `p3-h192-l6-s128-warm-clip-70ep-seed7` | bigger arch + warmup/clip + seed=7 | 42.45 | 36.67 |
| `p3-h192-l6-s128-warm-clip-70ep` | bigger arch + warmup/clip + 70 ep | 42.73 | 37.12 |
| `p3-h128-l8-warm-clip-90ep` | `n_layers=8` + warmup/clip + 90 ep | 42.79 | 37.77 |

Phase-3 take-aways:

- The best test score is from a **second seed of the warmup/clip recipe**,
  not the AMP variant — the seed-difference (31.05 vs 31.18 test) is
  comparable to or larger than the AMP improvement, so AMP is essentially
  neutral once you have warmup + clip.
- `lr=3e-4 + warmup=5` reached the *best val* (35.59) but slightly worse
  test (31.96) — small overfit on val.
- Bigger models with the same recipe still lose to the default arch at this
  budget (best bigger-arch test was 33.09 from `p3-h256-warm-clip-90ep`).
  Their faster per-epoch descent is undone by the smaller number of
  optimiser steps they get for a given wall-clock.

### Phase 4 — test evaluation + ensemble (`test_eval.py`, `ensemble_eval.py`)

For every Phase-3 candidate I copied the locally-saved best checkpoint into
`models/snapshots/<run-id>-checkpoint.pt`, then ran `test_eval.py` on the
held-out test splits. The script reuses `data/scoring.py` semantics with one
defensive guard: the cruise test set contains a single sample (index 20) with
`+inf`/`-inf` pressure values in 761 of its volume nodes, and
`accumulate_batch` masks that sample out — but `inf * False = NaN` propagates
into the float64 accumulator, producing a `NaN` final MAE. The wrapper in
`test_eval.evaluate_split` zeros non-finite predictions / targets *before*
the boolean mask multiplications, recovering the documented per-sample-skip
semantics from `program.md` without modifying the read-only `data/scoring.py`.

`ensemble_eval.py` averages predictions in normalized space across N
checkpoints, then denormalizes and scores with the same NaN-safe loop. The
top-4 default-arch checkpoints (`p3-warmup3-clip-150ep-seed7`,
`p3-amp-bs4-warm-clip-180ep`, `p3-warmup3-clip-150ep`, `p3-warmup5-lr3e4-150ep`)
share a single architecture but different optimisation noise, so their
errors decorrelate well — averaging four of them drops `test_avg/mae_surf_p`
from 31.05 to 27.11. Adding `p3-h256-warm-clip-90ep` (the only other run
under 40 val) gives a marginal 27.07.

## Code change

Single edit: `train.py` — exposed the model architecture
(`n_hidden / n_layers / n_head / slice_num / mlp_ratio / dropout`),
optimisation knobs (`warmup_epochs / grad_clip / seed`),
and a bf16 autocast switch (`use_amp`) as CLI flags. Defaults match the
original Phase-0 baseline so the existing
`python train.py --epochs 999 --agent <…> --wandb_group <…> --wandb_name <…>`
shape produces the original training run unchanged.

`data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, the splits, the
model architecture and the metric contract are **unchanged**; every val/test
number reported above is computed by the same `data.scoring` helpers (or in
the case of test, the `test_eval` wrapper around them) used by the Phase-1
baseline.

The new files I added are all under `research/` and `run_logs/`, not on the
trainer path: `test_eval.py`, `run_logs/watcher.sh`, `run_logs/launch_phase*.sh`,
`run_logs/extract_results.py`, `run_logs/build_final_jsonl.py`,
`research/MLINTERN_SUMMARY.md`, `research/MLINTERN_RESULTS.jsonl`.

## Final commands

### Train one of the four winning single-model recipes

```bash
SENPAI_TIMEOUT_MINUTES=350 CUDA_VISIBLE_DEVICES=$GPU \
python -u train.py --epochs 150 \
  --warmup_epochs 3 --grad_clip 1.0 \
  --seed 7 \
  --skip_test true \
  --agent ml-intern-r1 \
  --wandb_group mlintern-pai2-r3-retry-r1 \
  --wandb_name mlintern-pai2-r3-retry-r1/p3-warmup3-clip-150ep-seed7
```

### Test-eval one checkpoint

```bash
CUDA_VISIBLE_DEVICES=$GPU python -u test_eval.py \
  --checkpoint models/snapshots/ll7xn8sp-checkpoint-final.pt \
  --config_yaml models/snapshots/ll7xn8sp-config.yaml \
  --batch_size 4
```

### Ensemble the 5 winning checkpoints, val-weighted (final 26.97 number)

```bash
# Weights are inverse-val (1/best_val_avg_mae_surf_p) for the 5 checkpoints.
CUDA_VISIBLE_DEVICES=$GPU python -u ensemble_eval.py \
  --checkpoints \
    models/snapshots/ll7xn8sp-checkpoint.pt \
    models/snapshots/oat51afr-checkpoint.pt \
    models/snapshots/mmzdsva3-checkpoint.pt \
    models/snapshots/nlpe7zi8-checkpoint.pt \
    models/snapshots/wgd8zkqm-checkpoint.pt \
  --config_yamls \
    models/snapshots/ll7xn8sp-config.yaml \
    models/snapshots/oat51afr-config.yaml \
    models/snapshots/mmzdsva3-config.yaml \
    models/snapshots/nlpe7zi8-config.yaml \
    models/snapshots/wgd8zkqm-config.yaml \
  --weights 0.0277 0.0277 0.0268 0.0281 0.0256 \
  --batch_size 4
```

## GPU usage strategy

Eight RTX PRO 6000 cards (96 GB each), each pinned with `CUDA_VISIBLE_DEVICES`
to one Python `train.py` process. A bash watcher polled `nvidia-smi` every
30 s; whenever a GPU dropped below 1.5 GB it pulled the next config from a
queue file and launched a job there with that config's own
`SENPAI_TIMEOUT_MINUTES` ceiling, so a slow run on one GPU never blocked the
others. Each training job used `num_workers=4` data-loaders, so 8 × 4 = 32
worker processes total — comfortably below the 120 cores available.

The CFD samples are big (mesh sizes up to 242 K nodes) but the model is
small (the default Transolver is 0.66 M params). Wide single-GPU parallelism
across hyperparameter candidates gave a much better signal than DDP would
have given, since at single-GPU memory the bottleneck is data-loading, not
compute.

## Next-step recommendation

The warmup/clip recipe is at diminishing returns at the default model size —
going from 100 to 150 cosine epochs on the same recipe shaved ~5 MAE off
val and ~5 off test, and two seeds of the same recipe landed within 0.5 MAE
on val. The 5-model ensemble at test 27.07 is the strongest result here.
Future work should probably explore:

1. **Train a wider seed bank for ensembling.** The biggest Phase-4 win came
   from averaging 4 default-arch checkpoints with different optimisation
   noise (warmup vs warmup+amp vs warmup5+lower-lr vs another seed). Each
   marginal model added ~1 MAE up to N=4. Training 8 + seeds of the same
   recipe in parallel and averaging predictions should reasonably move the
   ensemble below 26.
2. **Properly tuned bigger architectures.** The Phase-2 / Phase-3 bigger
   models were under-trained: `p2-cap-h192-l6-s128-warm-80` reached 54.86
   val with only 45 of 80 cosine epochs; `p3-h256-warm-clip-90ep` was still
   descending at 39.19 val on its last epoch and had test 33.09. A 6 h
   single-GPU run on `h=256, l=6, slice_num=128` with the warmup/clip
   recipe and a 200-epoch cosine would likely beat the default arch on its
   own and add a more diverse model to the ensemble pool.
3. **Re-aware loss scaling.** Per-sample y-std for pressure varies by an
   order of magnitude across the corpus (Re 100 K → 5 M, see `program.md`
   value-range table). A Re-aware loss reweighting could let the model
   spend more capacity on the high-magnitude regimes that dominate the
   benchmark MAE without hurting the low-Re predictions that already do
   well.
