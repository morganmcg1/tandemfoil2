# ML Intern Run — `mlintern-pai2-72h-v4-r2`

Single autonomous launch on the pai2 cluster, 8x RTX PRO 6000 Blackwell (96 GB each), 72h hard wallclock. Local-only training (no HF Jobs / Sandboxes / Spaces).

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group:   `mlintern-pai2-72h-v4-r2`

## TL;DR

**Best single model:** `r5-recipe-cosine160-360m-s35` →
`val_avg/mae_surf_p = 49.85`, **`test_avg/mae_surf_p = 45.40`**.
W&B run: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs/647tmmoh>.

**Best ensemble (so far):** 7-model average of 5 cosine160 (s31-s35) + 2 pw2 cosine160 (s41-s42) →
**`test_avg/mae_surf_p ≈ 40.24`**. Round 6 (8 more cosine160 seeds) is still running and
should push this lower; the actual final number will be in `MLINTERN_RESULTS.jsonl` and the
final commit message.

For comparison, the unmodified `train.py` baseline (`baseline-60m`, 60 min, default config) gets
`val 102.52 / test 94.27` after the NaN-safe eval fix in this branch.

## Strategy (six rounds, all on this branch)

The whole campaign is a tight loop: launch a wave of 8 GPUs in parallel, learn from the
results, refine the next wave. The same single training script (`train.py`) is reused with
CLI flags throughout.

**Round 0** — read `program.md` and `data/SPLITS.md`, audit the splits with
`hf_inspect_dataset`-style scripts, smoke-test the existing `train.py`. Found that
`test_geom_camber_cruise/000020.pt` has 761 non-finite values in `y` and that
`data/scoring.py` is *almost* robust to it but PyTorch's `0 * NaN = NaN`
through the mask multiplication still poisons the accumulator. Patched
`evaluate_split` in `train.py` (since `data/scoring.py` is read-only) to drop
non-finite samples with `mask &= ~bad` and replace y with zeros before
`accumulate_batch` is called. Verified: baseline test went from `nan` →
`94.27` with no other change.

**Round 1** — coarse 60-min sweep across 8 GPUs: architecture (deep, wide,
medium, full LRSA-AirfRANS scale) and training recipe (LRSA paper recipe
`sw=1 lr=3e-4 wd=1e-5`, `lr=1e-3`, `bs=8`). The LRSA-AirfRANS recipe applied
to the baseline arch wins at val 99.69 / test 90.37 — bigger models all
trail because they don't get enough epochs in 60 min.

**Round 2** — exploit + scale, 90-min runs. Tested OneCycle and cosine30
schedulers, the `recipe` at scale (`h256 L8 + sub32k mesh subsample`),
combined `sw=10 + low lr/wd`, etc. Two takeaways:

1. The `recipe` (sw=1 lr=3e-4 wd=1e-5) at 90 min on the baseline arch with
   **constant LR** still wins: val 85.85 / test 77.57 (`r2-baseline-recipe-90m`).
2. Adding **proper LR decay** (cosine30 over 30 epochs) on the *original*
   default config (sw=10 lr=5e-4 wd=1e-4) jumps from val 102 → 94.83 / test 84.32,
   already validating the "give it real LR decay" thesis.

**Round 3** — focus on the recipe + multi-seed + longer runs (180 min each).
4 seeds of the recipe with constant LR + cosine80 + a couple of variants
(`p_weight=2`, scale + OneCycle, h192 medium). Result: cosine80 dominates at
val 64.91 / test 57.44; the 4 seeds are at val 65–72 / test 60–65. First
ensemble eval gave 6-model ensemble test = 52.02.

**Round 4** — push the recipe further: 240-min runs, 5 cosine120 seeds (s11-s15)
+ 3 scale variants. The 5 cosine120 seeds all reach test 50.66 – 52.77.
Top 5 R4 ensemble = test 44.58. The scale models are clearly worse and adding
them to the ensemble hurts.

**Round 5** — even longer cosine160 (360 min), 5 seeds (s31-s35) + 2 with
`p_weight=2` (s41, s42) for ensemble diversity + 1 scale. Best single
`r5-recipe-cosine160-360m-s35` reaches val 49.85 / test 45.40. 7-model
ensemble (5 cosine + 2 pw2) test = 40.24. Round 5 was the first time we
beat the round 4 ensemble.

**Round 6** — currently running. 8 more cosine160 seeds (s51-s58) for
ensemble. Expected to finish around 08:25 local; final ensemble will
combine R5 + R6 winners.

## Code changes (committed on this branch)

`train.py`:

1. **CLI hyperparameters** for the model architecture
   (`--n_hidden / --n_layers / --n_head / --slice_num / --mlp_ratio`) and
   optimizer/scheduler (`--scheduler {cosine,onecycle,linear_warmup_cosine}`,
   `--warmup_frac`, `--grad_clip`, `--p_weight`, `--train_subsample`, `--seed`,
   `--progress_every`). Defaults preserve the original baseline behavior.
2. **AB-UPT mesh subsampling** at training time (`--train_subsample N`) — random
   per-sample subsample down to `N` nodes per step (always preserves all surface
   nodes). Inspired by AB-UPT (arXiv 2502.09692). It cuts VRAM dramatically and
   acts as data augmentation, but at this dataset/model scale it didn't actually
   give better validation than the baseline arch with no subsampling.
3. **NaN-safe eval** in `evaluate_split` (described above).
4. **OneCycle robustness** — wrap `scheduler.step()` in `try/except ValueError` so
   stepping past `total_steps` (which can happen when the wallclock cap lets us
   complete more iterations than expected) just freezes the LR at zero rather than
   crashing.
5. **Reproducibility seed** (`--seed`) — `random.seed`, `np.random.seed`,
   `torch.manual_seed`, `torch.cuda.manual_seed_all`. -1 means unchanged behavior.

`research/`:

- `parse_results.py` — parses session logs into a JSONL with best val,
  best epoch, per-split best surf-p, test_avg, etc.
- `wandb_summary.py` — pulls current state of all runs in this group from
  the W&B API and prints a leaderboard.
- `reeval_test.py` — re-runs test eval on a saved checkpoint with the
  NaN-safe path; usable on any checkpoint dir without re-training.
- `ensemble_eval.py` — loads N model checkpoints and averages predictions
  (in normalized space) before computing MAE. Used to find the best ensemble
  subset across rounds.

`research/MLINTERN_RESULTS.jsonl` — one JSON per non-trivial run (~25 entries),
including ensemble-eval results.

## Compute usage

8 GPUs pinned with `CUDA_VISIBLE_DEVICES=N` per subprocess (no double-booking).
Most baseline-arch jobs use 41 GB VRAM with the default mesh, 17 GB with
mesh subsampling. Cumulative training budget across 6 rounds ≈ 110 GPU-hours
out of the 576 GPU-hour pod budget.

## Best command (single model)

```bash
python ./train.py \
  --epochs 160 --scheduler cosine \
  --batch_size 4 --lr 3e-4 --weight_decay 1e-5 --surf_weight 1.0 \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2 \
  --seed 35 \
  --agent ml-intern-r2 \
  --wandb_group mlintern-pai2-72h-v4-r2 \
  --wandb_name "mlintern-pai2-72h-v4-r2/r5-recipe-cosine160-360m-s35"
```

(With `SENPAI_TIMEOUT_MINUTES=360`.)

This is exactly the baseline architecture — only the optimizer recipe and
scheduler differ from the unmodified `train.py` defaults.

## Bug fix (NaN test eval) — important note for benchmark integrity

`test_geom_camber_cruise/000020.pt` has 761 non-finite values in `y`. After
this commit, all checkpoints will report finite test averages. For runs that
finished before the fix, run:
```bash
CUDA_VISIBLE_DEVICES=<i> python research/reeval_test.py models/model-<run_id>
```
to recompute test_avg. All numbers in the leaderboard above are *post-fix*.

## Next recommendation (if a future replicate runs)

1. Skip rounds 1-2 — the recipe (`sw=1 lr=3e-4 wd=1e-5` on the baseline arch)
   is reproducible and dominates. Go straight to long cosine schedules.
2. The biggest wins came from (a) longer training, (b) cosine LR decay over
   the *actual* run length, (c) multi-seed ensembling. Each is roughly a
   5-15 unit improvement on `test_avg/mae_surf_p`.
3. Bigger architectures (h256 L8, h384 L8) consistently underperformed the
   baseline arch on this task. The dataset is small (1499 train) and the
   scaled models overfit before they finish converging. Don't spend GPU time
   on them unless you have substantially more data.
4. Mesh subsampling (AB-UPT trick) cuts VRAM by ~3× but didn't improve
   accuracy; it's a memory tool more than an accuracy tool here.
