# ML Intern Run — `mlintern-pai2-72h-v4-r2`

Single autonomous launch on the pai2 cluster, 8x RTX PRO 6000 Blackwell (96 GB each), 72h hard wallclock.

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group:   `mlintern-pai2-72h-v4-r2`

## TL;DR

**Best validation so far:** `99.69` `val_avg/mae_surf_p` (run `r1-lrsa-train-recipe-bw1`).
**Best test so far:** `94.27` `test_avg/mae_surf_p` (baseline checkpoint `model-wlgao1cd`, recomputed
with NaN-safe eval — see "Bug fix" below). Round 2 still running, expecting to beat both.

## Strategy

Three rounds of experiments, each round informing the next.

**Round 0 — diagnostics.** Read `program.md`, `data/SPLITS.md`, audit the
splits with `hf_inspect_dataset`-style scripts. Discovered that
`val_geom_camber_cruise` has the lowest absolute pressure scale (median |p|≈134)
and so requires the most relative accuracy. Confirmed that `--epochs 999` with
`CosineAnnealingLR(T_max=999)` produces an essentially constant LR over a
60-min run (LR drops <1% across 25 epochs).

**Round 1 — coarse sweep.** 8 GPUs in parallel for 60 min each. Tested
architecture scaling (deep, wide, medium, full LRSA-AirfRANS-style scale) and
optimizer recipes (LRSA paper recipe `lr=3e-4 wd=1e-5 sw=1`, `lr=1e-3`, `bs=8`).

  Result: at 60 min, baseline arch wins because bigger models don't get enough
  epochs. Best was `r1-lrsa-train-recipe-bw1` at **99.69 val_avg/mae_surf_p**
  (sw=1 lr=3e-4 wd=1e-5 — the LRSA paper's AirfRANS recipe applied to our
  baseline arch). Two configs OOM'd on first launch; relaunched at smaller
  batch size.

**Round 2 — exploit + scale (90 min).** All running now. Combines the best
recipe from R1 (`lr=3e-4 wd=1e-5 sw=1`) with bigger architectures and the
AB-UPT mesh-subsample trick (random per-batch subsample to 32K nodes;
preserves all surface nodes). Runs and rationale:

  - GPU0  `r2-scale-h256l8-sub32k-recipe-90m` (3.94M, sw=1)
  - GPU1  `r2-baseline-onecycle30-recipe-80m` — OneCycle scheduler test
  - GPU2  `r2-deep-h128l8-recipe-90m` (1.03M, sw=1)
  - GPU3  `r2-med-h192l6-recipe-90m` (1.71M, sw=1)
  - GPU4  `r2-baseline-recipe-90m` — replicate R1 winner with longer time
  - GPU5  `r2-baseline-sw10-lowlr-90m` — sw=10 + low lr/wd combo
  - GPU6  `r2-baseline-cosine30-75m` — cosine scheduler over 30 epochs
  - GPU7  `r2-scale-h256l8-sub32k-sw10-recipe-90m` (3.94M, sw=10)

## Code changes (committed on this branch)

`train.py`:

1. **CLI hyperparameters** for the model architecture (`n_hidden`, `n_layers`,
   `n_head`, `slice_num`, `mlp_ratio`) and optimizer/scheduler choices
   (`scheduler`, `warmup_frac`, `grad_clip`, `p_weight`, `train_subsample`,
   `progress_every`). Defaults preserve the original baseline behavior.
2. **AB-UPT mesh subsampling** at training time — random per-sample
   subsample down to `train_subsample` nodes per step (always preserves all
   surface nodes). 0 = full mesh (default). Backed by AB-UPT
   (arXiv 2502.09692) which shows neural surrogates do not need full meshes
   during training and that random subsampling acts as data augmentation.
3. **NaN-safe eval** in `evaluate_split`. `test_geom_camber_cruise/000020.pt`
   contains 761 non-finite values (-inf) in y. `data/scoring.py` has a
   y_finite check, but PyTorch's `0 * NaN/Inf = NaN/Inf` propagation through
   the mask multiplication contaminates the accumulators. Fixed at the
   call site (`scoring.py` is read-only): mask out non-finite samples,
   replace their y with zeros, then call `accumulate_batch`. Verified on
   `model-wlgao1cd`: test_avg went from `nan` → `94.2694`.

`research/`:

- `parse_results.py` — parses session logs into a JSONL with best val,
  best epoch, per-split best surf-p, test_avg, etc.
- `wandb_summary.py` — pulls current state of all runs in this group from
  the W&B API and prints a leaderboard.
- `reeval_test.py` — re-runs test eval on a saved checkpoint with the
  NaN-safe path; usable on any checkpoint dir without re-training.

## Compute usage

8 GPUs pinned with `CUDA_VISIBLE_DEVICES`. Most jobs use 41–62 GB VRAM with
the baseline arch and 17 GB with mesh subsampling. Round 1 + Round 2
together = ~16 GPU-hours of training so far.

## Bug fix (NaN test eval)

`test_geom_camber_cruise/000020.pt` has 761 non-finite values in `y`. After
this commit, all checkpoints will report finite test averages. For runs that
finished before the fix, run:
```
CUDA_VISIBLE_DEVICES=<i> python research/reeval_test.py models/model-<run_id>
```
to recompute test_avg.

## Next steps (round 3)

Once round 2 finishes (~13:00 local), pick the top 1–2 configs and run them
for 3–4 hours each (or with multi-seed ensembling). Cosine vs OneCycle
scheduler comparison from round 2 will tell us whether to use proper LR
decay in round 3.
