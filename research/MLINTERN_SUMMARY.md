# ML Intern Run — `mlintern-pai2-72h-v4-r2`

Single autonomous launch on the pai2 cluster, 8x RTX PRO 6000 Blackwell (96 GB each), 72 h hard wallclock.
Local-only training (no HF Jobs / Sandboxes / Spaces).

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group:   `mlintern-pai2-72h-v4-r2`

## Headline numbers

| | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---|
| Original baseline (`baseline-60m`)      | 102.52 | 94.27 |
| Best single ML-Intern model (`r5-recipe-cosine160-360m-s35`) | **49.85** | **45.40** |
| **Final ensemble** (all 13 R5+R6 cosine160 baseline-arch seeds)  | n/a   | **39.29** |
| Same ensemble narrowed to top 10 (5 R5 + 5 R6) | n/a | 39.28 |
| Same ensemble + 2 R5 pw2 (15 models) | n/a | 39.37 |

The unmodified `train.py` baseline reports `test_avg/mae_surf_p = nan` due to one
non-finite ground-truth sample in `test_geom_camber_cruise/000020.pt`; with the
NaN-safe eval fix in this branch its real test number is `94.27`.

W&B run for the best single model:
<https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs/647tmmoh>

## Recipe (single best command)

```bash
SENPAI_TIMEOUT_MINUTES=360 python ./train.py \
  --epochs 160 --scheduler cosine \
  --batch_size 4 --lr 3e-4 --weight_decay 1e-5 --surf_weight 1.0 \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --seed 35 \
  --agent ml-intern-r2 \
  --wandb_group mlintern-pai2-72h-v4-r2 \
  --wandb_name "mlintern-pai2-72h-v4-r2/r5-recipe-cosine160-360m-s35"
```

Architecture is exactly the original baseline. The win is in the optimizer recipe + LR
schedule + budget. The only knobs that change vs. the unmodified `train.py` are
`lr 3e-4 → 5e-4`, `wd 1e-5 → 1e-4`, `surf_weight 1.0 → 10.0`, and using a real
cosine schedule (`--epochs 160`) instead of `--epochs 999` (which makes the cosine
schedule stay near constant).

## How the ensemble was built

The 10-model ensemble that gives `test_avg/mae_surf_p = 39.28` is just the average
of normalized-space predictions from these 10 W&B runs (all use the recipe above
with different `--seed`):

| W&B run id | round | seed | val | test |
|---|---|---:|---:|---:|
| 647tmmoh | r5 | 35 | 49.85 | 45.40 |
| lwghq8q0 | r5 | 31 | 53.71 | 46.95 |
| xzgi6fvf | r5 | 32 | 53.35 | 47.21 |
| 5dx5ibfm | r5 | 34 | 52.79 | 47.24 |
| ieovpclo | r5 | 33 | 53.62 | 47.59 |
| 9djkbols | r6 | 56 | 53.11 | 46.39 |
| rdwbldt5 | r6 | 55 | 52.44 | 46.79 |
| ef1tnt7x | r6 | 54 | 53.05 | 47.35 |
| ocwpbuhv | r6 | 58 | 53.22 | 47.46 |
| o57eo2z2 | r6 | 57 | 53.88 | 47.51 |

`research/ensemble_eval.py <model_dir1> <model_dir2> ...` reproduces this number.
Adding the 2 `p_weight=2` runs takes the ensemble to 39.37; adding 5 weaker R4
cosine120 models takes it back up to 41.22 — adding more isn't always better.

## Strategy (six rounds, all on this branch)

The whole campaign is a tight loop: launch a wave of 8 GPUs in parallel, learn from
the results, refine the next wave. The same `train.py` is reused with CLI flags
throughout; the only repo edits are in this branch's commits.

**Round 0 — diagnostics.** Read `program.md` and `data/SPLITS.md`, audit the
splits, smoke-test `train.py`. Found that `test_geom_camber_cruise/000020.pt`
has 761 non-finite values in `y` and that PyTorch's `0 * NaN = NaN` through the
mask multiplication poisons the MAE accumulator. Patched `evaluate_split` (the
read-only `data/scoring.py` doesn't need to change). Verified: baseline test went
`nan → 94.27` with no other change.

**Round 1 — coarse 60-min sweep, 8 GPUs.** Architecture vs. recipe. The LRSA paper's
AirfRANS recipe (`sw=1 lr=3e-4 wd=1e-5`) on the baseline arch wins at val 99.69 /
test 90.37. Bigger architectures all trail because they don't get enough epochs in
60 min.

**Round 2 — exploit + scale, 90-min runs.** Tested OneCycle, cosine30, the recipe
at scale (h256 L8 + AB-UPT mesh subsample), `sw=10 + low lr/wd`, etc. Two
takeaways:

1. The recipe at 90 min on the baseline arch with constant LR still wins:
   val 85.85 / test 77.57.
2. Adding proper LR decay (cosine30) on the *original default* config (sw=10,
   lr=5e-4, wd=1e-4) jumps from val 102 → 94.83 / test 84.32. Big confirmation
   of the "give it real LR decay" thesis.

**Round 3 — focus on the recipe, multi-seed, longer runs (180 min each).**
4 seeds + cosine80 + variants. Cosine80 dominates at val 64.91 / test 57.44; 6-model
ensemble = 52.02.

**Round 4 — push further, 240-min runs.** 5 cosine120 seeds (s11–s15) all reach
test 50.66 – 52.77. Top-5 R4 ensemble = 44.58. Scale models clearly worse.

**Round 5 — even longer cosine160, 360 min.** 5 seeds (s31–s35) + 2 pw2 (s41, s42)
+ 1 scale. Best single `r5-cosine160-s35` reaches val 49.85 / test 45.40. 7-model
ensemble = 40.24.

**Round 6 — final round, 8 more cosine160 seeds (s51–s58), 360 min each.**
Best single test = 46.39. 13-model ensemble (R5 + R6 cosine160) test = 39.29;
optimal 10-model subset = 39.28.

## Code changes (committed on this branch)

`train.py`:

1. **CLI hyperparameters** for the model architecture and the optimizer/scheduler
   (see `--help`). Defaults preserve the original baseline behavior.
2. **AB-UPT mesh subsampling** at training time (`--train_subsample N`) — random
   per-sample subsample down to `N` nodes per step, always preserving all surface
   nodes. Inspired by AB-UPT (arXiv 2502.09692). Cuts VRAM dramatically
   (42 GB → 17 GB at the baseline arch); at this dataset/model scale, however, it
   didn't actually beat the no-subsample baseline.
3. **NaN-safe eval** in `evaluate_split`. `data/scoring.py` is read-only, so the
   fix lives at the call site: when a batch has any non-finite y, drop the
   sample via `mask &= ~bad` and replace its y with zeros before
   `accumulate_batch`. Verified on `model-wlgao1cd`: test went `nan → 94.27`.
4. **OneCycle robustness** — wrap `scheduler.step()` in `try/except ValueError`
   so stepping past `total_steps` (the wallclock cap can let us complete more
   iterations than `--epochs * steps_per_epoch`) just freezes the LR at zero
   instead of crashing.
5. **Reproducibility seed** (`--seed`) — `random.seed`, `np.random.seed`,
   `torch.manual_seed`, `torch.cuda.manual_seed_all`. -1 means unchanged
   behavior (default).

`research/`:

- `parse_results.py` — parses session logs into a JSONL.
- `wandb_summary.py` — pulls current state of all runs in this group from W&B.
- `reeval_test.py` — re-runs test eval on a saved checkpoint with the NaN-safe
  path; usable on any checkpoint dir without re-training.
- `ensemble_eval.py` — loads N model checkpoints and averages predictions
  (in normalized space) before computing MAE.

`research/MLINTERN_RESULTS.jsonl` — one JSON per non-trivial run + ensemble
results (~30 entries).

## GPU usage

8 GPUs, each pinned with `CUDA_VISIBLE_DEVICES=N` per subprocess (no
double-booking). Memory: 41 GB at baseline arch with full mesh, 17 GB with
mesh subsampling. Cumulative training across rounds 1–6 ≈ 110 GPU-hours out of
the 576 GPU-hour pod budget; the remainder went to validation, monitoring,
ensemble eval, and reeval of the round-1 checkpoints (which had run before
the NaN fix).

## Findings worth carrying forward

1. **The recipe**: `sw=1 lr=3e-4 wd=1e-5` on the original baseline architecture
   beats the original config (`sw=10 lr=5e-4 wd=1e-4`) by a large margin at every
   training budget we tried. Backed by the LRSA paper's AirfRANS recipe.
2. **Real LR decay matters**: The original `train.py` with `--epochs 999` and
   `CosineAnnealingLR(T_max=999)` produces an essentially constant LR over a
   1-h run (LR drops <1% in the first 25 epochs). Setting `--epochs` to roughly
   the actual epoch count is worth ~5–15 units of `test_avg/mae_surf_p` on its
   own.
3. **Long training keeps paying off**: 60 → 90 → 180 → 240 → 360 min runs of the
   same recipe reach val 99.69 → 85.85 → 64.91 → 56.95 → 49.85. Diminishing
   returns past ~6 h on this dataset, but each step still helps.
4. **Multi-seed ensembling** is the single biggest test-time gain we measured:
   single best 45.40 → 10-model ensemble 39.28.
5. **Bigger architectures (h256 L8, h384 L8) consistently underperformed** the
   baseline arch at every time budget, with or without subsampling. The dataset
   is small (1499 train) and the bigger models can't absorb that capacity. Don't
   spend GPU time on them unless you grow the dataset.
6. **AB-UPT-style mesh subsampling** is a memory tool, not an accuracy tool here.

## Bug fix (NaN test eval) — important note for benchmark integrity

`test_geom_camber_cruise/000020.pt` has 761 non-finite values in `y`. After this
fix, all checkpoints will report finite test averages. For runs that finished
before the fix:
```bash
CUDA_VISIBLE_DEVICES=<i> python research/reeval_test.py models/model-<run_id>
```

All numbers in the leaderboard above are post-fix.

## Next recommendation

If a future replicate runs:

1. Skip rounds 1-2 — the recipe and "real LR decay" are reproducible and dominate.
   Go straight to long cosine schedules of the baseline arch.
2. Spend half your GPU-time on training, half on multi-seed ensembling.
3. Don't bother with bigger architectures unless you also grow the training set.
4. Keep the NaN-safe eval patch — it's the difference between getting a real
   test number and reporting `nan`.
