# SENPAI Research State

- 2026-04-27 23:50 — round 1 in progress on `icml-appendix-willow-pai2d-r2`
- **Baseline anchored:** PR #328 (slice_num=128) merged. Current best
  `val_avg/mae_surf_p = 133.55` (W&B run `s1p2qs7l`, best epoch 11/50).
  Default config now: `n_hidden=128 n_layers=5 n_head=4 slice_num=128
  mlp_ratio=2 lr=5e-4 weight_decay=1e-4 batch_size=4 surf_weight=10.0
  epochs=50`.
- Primary metric: `val_avg/mae_surf_p`. Paper-facing:
  `test_avg/mae_surf_p` (currently NaN — bug-fix PR #367 in flight).
- W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2`.
- Wall-clock cap per run: `SENPAI_TIMEOUT_MINUTES=30`. Empirically this
  binds at **9–14 epochs** for round-1 configurations, not the planned
  50 — every finished run is undertrained.

## Round 1 status (after this cycle's actions)

| PR  | Student   | Axis                       | Status (now)   | Best val_avg/mae_surf_p |
|-----|-----------|----------------------------|----------------|-------------------------|
| 311 | alphonse  | width 128 → 192            | sent back → width-160 + AMP | 134.13 (epoch 10/50) |
| 325 | askeladd  | depth 5 → 8                | wip            | 150.06 (W&B obs)        |
| 326 | edward    | mlp_ratio 2 → 4            | sent back → mlp_ratio=3 | 137.83 (epoch 11/13) |
| 328 | fern      | slice_num 64 → 128         | **MERGED ★**   | **133.55 (new baseline)**|
| 330 | frieren   | MSE → Huber β=1            | wip            | (multiple crashes; debugging) |
| 332 | nezuko    | surf_weight 10 → 25 (sweep)| wip            | 137.42 surf-15 (sweep ongoing) |
| 335 | tanjiro   | warmup + cos, peak 1e-3    | sent back → cosine_t_max sweep | 154.57 (epoch 13/14) |
| 337 | thorfinn  | BS 4→8, lr 7e-4            | wip            | 139.39 (W&B obs)        |
| 367 | fern      | bug fix: cruise-NaN scoring| **wip (new)**  | n/a (bug fix, not experiment) |

PRs surfaced for advisor review this cycle: **#328, #326**. Actions:
**#328 merged** (round-1 winner), **#326 sent back** (FFN axis still
interesting at lower mlp_ratio).

## What we learned this cycle

1. **`slice_num` is the most pay-per-FLOP axis under the 30-min cap.**
   Doubling slice tokens (64 → 128) gave the cleanest win in round 1
   at minimal extra cost (~10–15% slowdown). Per-split signal matched
   the slice-bottleneck hypothesis cleanly: 7% improvement on
   `val_geom_camber_rc` vs the next-best run.
2. **Capacity-scaling axes (`n_hidden`, `n_layers`, `mlp_ratio`) are
   confounded by the 30-min cap.** They each chop the epoch budget
   by 30–80%, so the wall-clock-equal comparison favors cheaper
   changes even if the asymptotic answer might prefer wider models.
   Compute-equal middle grounds (width-160, mlp_ratio=3) are the
   right next step.
3. **The cruise NaN bug is a real blocker for paper-facing metrics.**
   Both edward and fern independently identified the same root cause
   (sample 20 of `test_geom_camber_cruise/.gt` has 761 NaN in pressure;
   `0.0 * NaN = NaN` in `accumulate_batch` defeats the `y_finite` mask).
   PR #367 puts up the 2-line `nan_to_num` fix.

## Round 2 candidate stacks (post round-1 settle, will compound on the new baseline)

- **Push `slice_num` further.** 128 worked; 192 or 256 likely also
  helps. Slice² scaling stays small relative to O(N · slice_num).
- **Stack slice-128 + width.** alphonse's width axis on top of the
  merged slice-128 baseline — directly tests the
  arch-stack hypothesis. Pending alphonse's width-160 result first.
- **AMP / mixed precision.** With width-192 hitting 92% peak memory,
  AMP is needed before any wider model can fit alongside slice-128.
  Likely a free unlock orthogonal to architecture.
- **Schedule that fits the budget.** OneCycleLR over `total_steps`
  (not epochs) is robust to 30-min wall-clock cuts. May supersede
  `T_max=epochs` cosine entirely. Pending tanjiro's iteration.
- **Per-channel loss weighting on `p`.** Metric only cares about
  pressure; loss currently weights all 3 channels equally. Pairs with
  surf_weight winner (nezuko's sweep).
- **Target-space reformulation.** `asinh` / per-sample-std
  normalization on the pressure channel — pairs with Huber if
  frieren's iteration finishes (currently crashing).
- **Geometry-preserving augmentation.** x-flip for the ground-effect
  raceCar domain (mirror y-coord and corresponding flow components).
- **Curriculum.** Sort batches by per-sample y std, warmup the
  model on low-magnitude samples first.

## Active blockers

1. **Frieren's huber-loss runs are crashing** (multiple
   crashed/failed runs visible in W&B). Suspect a NaN/Inf in the
   custom loss code path — should be addressed when frieren
   pushes a results comment or the entrypoint surfaces it for review.
2. **`test_avg/mae_surf_p = NaN` for every run** until PR #367 lands
   (in flight, fern). Round-1 winners are being selected on
   `val_avg/mae_surf_p` only — this is fine for the round-1 ranking
   but blocks paper-facing comparisons.

## Constraints respected

- `SENPAI_MAX_EPOCHS=999` and `SENPAI_TIMEOUT_MINUTES=30` not overridden.
- `data/scoring.py` and `train.py::evaluate_split` are normally
  read-only for experiment PRs — bug-fix PR #367 is the documented
  exception (metric-preserving 2-line patch).
- One hypothesis per PR. Send-backs ask for variants of the same
  axis or compute-equal restatements; not bundled changes.
