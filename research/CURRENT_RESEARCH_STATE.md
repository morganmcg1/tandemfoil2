# SENPAI Research State

- 2026-04-28 01:00 — round 1 in progress on `icml-appendix-willow-pai2d-r2`
- **Baseline anchored:** PR #328 (slice_num=128) merged. Current best
  `val_avg/mae_surf_p = 133.55` (W&B run `s1p2qs7l`, best epoch 11/50).
  Default config now: `n_hidden=128 n_layers=5 n_head=4 slice_num=128
  mlp_ratio=2 lr=5e-4 weight_decay=1e-4 batch_size=4 surf_weight=10.0
  epochs=50`.
- **Pending merge candidates** (all need on-baseline confirmation;
  all pre-#328, sent back for rebase):
  - **PR #330** (frieren, Huber β=1): `val_avg/mae_surf_p = 109.47`
    on slice_num=64. 18 % over baseline — outside ±10 % noise band,
    very likely a real signal.
  - **PR #311** (alphonse, width-160): `val_avg/mae_surf_p = 126.18`
    on slice_num=64. 5.5 % over baseline — inside ±10 % noise floor,
    needs on-baseline + multi-seed to disambiguate.
  - **PR #332** (nezuko, surf_weight=25): `val_avg/mae_surf_p = 133.19`
    on slice_num=64. 0.27 % over baseline — far inside noise floor,
    but sweep curve + val_vol_p both support the interior optimum
    qualitatively. Needs on-baseline + multi-seed.
- **Noise floor: ±10 % at single seed.** Thorfinn (PR #337) ran the same
  config twice (kon60q79=153.19, nphltrz9=139.39) — a ~10 % spread
  purely from random seed / data-order randomness. **This is a
  methodological constraint affecting every PR's number on this
  branch.** Single-seed deltas <10 % vs baseline (133.55) are not
  statistically distinguishable; merge decisions on borderline PRs
  should require multi-seed replication. Frieren's 109.47 sits outside
  this noise band, so it remains a real signal.
- Primary metric: `val_avg/mae_surf_p`. Paper-facing:
  `test_avg/mae_surf_p` (currently NaN — bug-fix PR #367 in flight).
- W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2`.
- Wall-clock cap per run: `SENPAI_TIMEOUT_MINUTES=30`. Empirically this
  binds at **9–14 epochs** for round-1 configurations, not the planned
  50 — every finished run is undertrained.

## Round 1 status (after this cycle's actions)

| PR  | Student   | Axis                       | Status (now)   | Best val_avg/mae_surf_p |
|-----|-----------|----------------------------|----------------|-------------------------|
| 311 | alphonse  | width 128 → 192 → 160       | sent back → rebase + on-baseline (+ multi-seed if borderline) | width-160=**126.18** (epoch 11/50; inside ±10 % noise of baseline) |
| 325 | askeladd  | depth 5 → 8                | wip            | 150.06 (W&B obs)        |
| 326 | edward    | mlp_ratio 2 → 4            | sent back → mlp_ratio=3 | 137.83 (epoch 11/13) |
| 328 | fern      | slice_num 64 → 128         | **MERGED ★**   | **133.55 (new baseline)**|
| 330 | frieren   | MSE → Huber β=1            | sent back → rebase + re-run | **109.47** (epoch 14/50, on slice_num=64; merge-candidate after rebase) |
| 332 | nezuko    | surf_weight 10 → 25 (sweep)| sent back → rebase + on-baseline + multi-seed | sweep done: surf-15=137.42, **surf-25=133.19**, surf-40=142.59 (clean interior optimum at 25; absolute level inside ±10 % noise of baseline) |
| 335 | tanjiro   | warmup + cos, peak 1e-3    | sent back → cosine_t_max sweep | 154.57 (epoch 13/14) |
| 337 | thorfinn  | BS 4→8, lr 7e-4            | sent back → rebase + BS=16/lr=1e-3 (+ multi-seed if budget) | 139.39 / 153.19 (2-seed mean 146.29; ~9.5 % worse than baseline on mean) |
| 367 | fern      | bug fix: cruise-NaN scoring| **wip (new)**  | n/a (bug fix, not experiment) |

PRs surfaced for advisor review this cycle: **#332**. Action:
**#332 sent back** — surf_weight sweep is qualitatively excellent
(clean interior optimum at 25 with val_vol_p as independent
secondary signal) but quantitatively inside-noise (0.27 % over
merged baseline). Branch pre-#328 (rebase needed). Asked for:
rebase + re-run surf_weight=25 on slice-128 + optional 2 more
seeds. Same decision rule as alphonse #311.

Earlier cycle actions (recap): #328 merged (round-1 winner, new
baseline 133.55); #326 + #335 + #330 + #337 + #311 sent back with
specific follow-up instructions; #367 NEW bug-fix PR assigned to
fern for cruise-NaN scoring (in flight).

## What we learned this cycle (and last)

0. **Single-seed noise floor on this branch is ≈ ±10 %.** Thorfinn's
   accidental same-config replicate (kon60q79 vs nphltrz9, both
   `bs=8 lr=7e-4`) showed a 13.8-point spread (153.19 vs 139.39)
   purely from random seed / data-order. This means:
   - Differences smaller than ~10 % between single-seed runs are not
     statistically meaningful.
   - The 134-138 cohort (alphonse, fern, edward, nezuko surf-15) is
     effectively tied within noise.
   - Merge decisions for results inside ±10 % of baseline must use
     multi-seed replication to disambiguate.
   - Frieren's 109.47 (18 % over baseline) is the only round-1
     result outside the noise band on a single seed; it's the only
     result we can be confident about without replication.

1. **Loss formulation is the highest-leverage axis we've found.**
   Frieren's Huber-β=1 result (109.47 on slice_num=64) is roughly 4×
   the magnitude of any architecture-axis gain seen in round 1.
   Likely combination of (a) Huber's L1-tail behavior matches the
   L1 metric we're ranked on better than MSE; (b) at high Re,
   normalized residuals exceed 1.0 enough that gradient clipping
   prevents tail samples from dominating. Direct evidence of (b):
   `val_re_rand` is best-of-cohort (100.85, 17 % over next-best).
2. **`slice_num` is the most pay-per-FLOP architecture axis.**
   Doubling slice tokens (64 → 128) gave the round-1 architecture
   winner at minimal extra cost (~10–15 % slowdown). Per-split
   signal matched the slice-bottleneck hypothesis cleanly: 7 %
   improvement on `val_geom_camber_rc` vs the next-best run.
3. **Capacity-scaling axes (`n_hidden`, `n_layers`, `mlp_ratio`)
   are confounded by the 30-min cap.** They each chop the epoch
   budget by 30–80 %, so the wall-clock-equal comparison favors
   cheaper changes even if the asymptotic answer might prefer
   wider models. Compute-equal middle grounds (width-160,
   mlp_ratio=3) are the right next step — both already in flight.
4. **The cruise NaN bug is a real blocker for paper-facing metrics.**
   Both edward and fern independently identified the same root cause
   (sample 20 of `test_geom_camber_cruise/.gt` has 761 NaN in pressure;
   `0.0 * NaN = NaN` in `accumulate_batch` defeats the `y_finite`
   mask). PR #367 puts up the 2-line `nan_to_num` fix.
5. **Branches need rebases as soon as the advisor branch moves.**
   Confirmed twice this cycle: frieren's #330 and thorfinn's #337
   were both created pre-#328 and both would silently revert
   slice_num=128 → 64 on direct squash-merge. The send-back-for-
   rebase pattern catches it. **All other in-flight round-1 PRs
   (alphonse, askeladd, edward, nezuko, tanjiro)** are pre-#328
   too and will need the same rebase before merge. Reviewers should
   diff against current advisor head before merging anything.

## Round 2 candidate stacks (post round-1 settle, will compound on the new baseline)

- **Push `slice_num` further.** 128 worked; 192 or 256 likely also
  helps. Slice² scaling stays small relative to O(N · slice_num).
- **Stack slice-128 + width.** alphonse's width axis on top of the
  merged slice-128 baseline — directly tests the
  arch-stack hypothesis. Pending alphonse's width-160 result first.
- **AMP / mixed precision.** Alphonse's fp16 attempt diverged at
  epoch 3 from `(pred-y_norm)² · surf_weight=10` overflowing fp16
  dynamic range during early high-loss epochs. Two clear fixes
  identified: (i) scoped autocast — model forward in fp16, loss in
  fp32; (ii) bfloat16 (Hopper-class GPUs, no GradScaler needed,
  fp32-equivalent dynamic range). Both unblock the AMP unlock for
  any wider/deeper round-2 stack.
- **Schedule that fits the budget.** OneCycleLR over `total_steps`
  (not epochs) is robust to 30-min wall-clock cuts. May supersede
  `T_max=epochs` cosine entirely. Pending tanjiro's iteration.
- **Per-channel loss weighting on `p`.** Metric only cares about
  pressure; loss currently weights all 3 channels equally. Pairs with
  surf_weight winner (nezuko's sweep).
- **Target-space reformulation.** `asinh` / per-sample-std
  normalization on the pressure channel — pairs naturally with Huber
  (now that Huber-β=1 has confirmed the high-Re-tail story).
- **Geometry-preserving augmentation.** x-flip for the ground-effect
  raceCar domain (mirror y-coord and corresponding flow components).
- **Curriculum.** Sort batches by per-sample y std, warmup the
  model on low-magnitude samples first.

## Active blockers

1. **`test_avg/mae_surf_p = NaN` for every run** until PR #367 lands
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
