# SENPAI Research State

- 2026-04-28 02:15 — round 1 mostly settled on `icml-appendix-willow-pai2d-r2`,
  round 2 building momentum
- **Baseline updated:** PR #330 (frieren, Huber β=1) merged on top of
  PR #328. Current best `val_avg/mae_surf_p = 115.61` (W&B run
  `uip4q05z`, best epoch 11/50). −13.4 % over the prior baseline,
  outside ±10 % noise floor. Default config now: `n_hidden=128
  n_layers=5 n_head=4 slice_num=128 mlp_ratio=2 lr=5e-4
  weight_decay=1e-4 batch_size=4 surf_weight=10.0 epochs=50` plus
  `loss=Huber(beta=1.0)` on normalized residuals.
- **Round 2 candidates in flight:**
  - **PR #415** (frieren, asinh on pressure target): NEW
    assignment. Pairs with merged Huber β=1 — orthogonal
    mechanism on the same high-Re-tail failure mode.
  - **PR #399** (askeladd, bf16 mixed precision): throughput unlock
    for the 30-min cap.
- **Pending round-1 merge candidates** (all need on-baseline
  confirmation; all pre-#330, on-baseline + multi-seed):
  - **PR #311** (alphonse, width-160): 126.18 on slice_num=64 (now
    11.7 % WORSE than the new 115.61 baseline — needs to land below
    115.61 to be a winner, well outside the ±10 % noise band).
  - **PR #332** (nezuko, surf_weight=25): 133.19 on slice_num=64
    (now 15.2 % WORSE than the new baseline). Bar is even higher
    after the Huber merge.
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
| 325 | askeladd  | depth 5 → 8                | **closed** (21 % regression at 30-min cap) | 150.06 / 162.05 (two seeds) |
| 399 | askeladd  | round 2: bf16 mixed precision | NEW assignment (status:wip) | n/a |
| 326 | edward    | mlp_ratio 2 → 4            | **closed** (FFN axis exhausted; 21 % worse than new baseline) | mlp-3=139.79, mlp-2-control=136.54 (slice-128, MSE; monotone trend smaller-is-better) |
| 429 | edward    | round 2: per-channel loss weighting on `p` | NEW assignment (status:wip) | n/a |
| 328 | fern      | slice_num 64 → 128         | **MERGED ★**   | **133.55 (new baseline)**|
| 330 | frieren   | MSE → Huber β=1            | **MERGED ★ (new baseline 115.61)** | rebased run = 115.61 (slice-128, epoch 11/50, run uip4q05z) |
| 399 | askeladd  | round 2: bf16 mixed precision | wip            | n/a (assigned this cycle) |
| 415 | frieren   | round 2: asinh on pressure target | NEW assignment | n/a (assigned this cycle) |
| 332 | nezuko    | surf_weight 10 → 25 (sweep)| sent back → rebase + on-baseline + multi-seed | sweep done: surf-15=137.42, **surf-25=133.19**, surf-40=142.59 (clean interior optimum at 25; absolute level inside ±10 % noise of baseline) |
| 335 | tanjiro   | warmup + cos, peak 1e-3    | sent back → rebase + on-baseline re-run | sweep done on slice-64+MSE: best **(b) 113.96** at lr=1e-3, T_max=15 (a tied at 115.15, c@135.24 likely seed-unlucky); needs on-baseline confirmation |
| 337 | thorfinn  | BS 4→8, lr 7e-4            | sent back → rebase + BS=16/lr=1e-3 (+ multi-seed if budget) | 139.39 / 153.19 (2-seed mean 146.29; ~9.5 % worse than baseline on mean) |
| 367 | fern      | bug fix: cruise-NaN scoring| sent back → rebase + Huber-aware reapplication | patch verified (test_avg: NaN → 125.05) but on pre-Huber base |

PRs surfaced for advisor review this cycle: **#326**. Action:
**#326 closed** — three FFN variants (mlp_ratio ∈ {2, 3, 4}) across
two architectures all show smaller-is-better at the 30-min cap.
Best variant 21 % worse than merged Huber baseline. Student
explicitly recommended closing in their own follow-up #1.
**Reassigned edward to round-2 axis #429 (per-channel loss
weighting on pressure)** — zero compute cost, direct push on the
metric we're ranked on (which only counts surface pressure but the
loss weights all 3 channels equally), stacks naturally with all
in-flight work. Sweep design: `p_weight ∈ {2, 3, 5}` with a
`_channel_weighted_huber` helper.

Bonus: edward's control re-run at identical config (136.54 vs
published baseline 133.55 = 3-point spread) is the second
independent corroboration of the ±10 % single-seed noise floor
(after thorfinn #337). Two convergent measurements now anchor the
methodology.

Earlier cycle actions (recap): #328 + #330 merged (slice-128 +
Huber β=1, current baseline 115.61); #311 + #332 + #335 + #337
sent back; #367 sent back for Huber-aware reapplication; #325 +
#326 closed (depth-8 + FFN axes exhausted at 30-min cap); #399 +
#415 + #429 assigned (askeladd bf16, frieren asinh-on-pressure,
edward p_weight — three round-2 axes in flight).

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
- **AMP / mixed precision.** ASSIGNED as PR #399 to askeladd:
  bf16 autocast around the model forward (loss kept in fp32),
  building on alphonse's fp16-failure-mode diagnosis. Expected to
  give 1.4–1.8× per-step speedup with no dynamic-range collapse.
  If clean, makes all heavier round-2 stacks viable inside the
  30-min cap.
- **Schedule that fits the budget.** OneCycleLR over `total_steps`
  (not epochs) is robust to 30-min wall-clock cuts. May supersede
  `T_max=epochs` cosine entirely. Pending tanjiro's iteration.
- **Per-channel loss weighting on `p`.** ASSIGNED as PR #429 to
  edward: `p_weight` CLI flag scaling the pressure column of the
  per-element Huber tensor before the spatial reduction. Sweep
  {2, 3, 5}. Zero compute cost, orthogonal to surf_weight (which
  works on the spatial axis) and to asinh (which works on the
  target-distribution axis). All three loss-axis levers can stack.
- **Target-space reformulation.** ASSIGNED as PR #415 to frieren:
  `asinh` on pressure channel of `y_norm` only (Ux/Uy unchanged).
  Pairs naturally with the merged Huber-β=1 — orthogonal mechanisms
  on the same high-Re-tail failure mode (distribution flattening vs
  gradient clipping). per-sample-std normalization is a separate
  follow-up if asinh doesn't capture the gain.
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
