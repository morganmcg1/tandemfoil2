# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2d-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`
- **Most recent human-team direction:** none received (fresh launch)

## Current research focus

**Four wins now merged: warmup, EMA, pure L1, OneCycleLR.** PR #320 (linear warmup), PR #410 (EMA), PR #294 (pure L1 surface loss), and PR #409 (OneCycleLR) compound to drop the single-seed val_avg/mae_surf_p from ~147 (default Transolver) → **87.74** (current merged baseline). Test_avg = **78.24**. The schedule effect from OneCycle shrunk vs the pre-EMA test (R2 was −33 MAE; R3 with EMA+L1 was −7.5 MAE) because OneCycle's cool-down and EMA+L1 partially overlap — both contribute to a smoother converged state. Still net-positive.

**Critical mechanistic finding (PR #409 diagnostic):** EMA-vs-live sign-flips across schedules. EMA helps noisy schedules (warmup-cosine, +7.26 MAE) but HURTS converged ones (OneCycle, −5.69 MAE). The mechanism is clean — EMA averages out training noise; useful when the schedule leaves the model still noisy at end-of-training, counterproductive when the schedule already converges via aggressive cool-down. Frieren's PR #671 is testing `use_ema=False` under OneCycle (predicted val_avg ≈ 82, another −5.7 MAE).

**⚠️ Seed-variance caveat (still relevant):** `train.py` has no seed control yet — runs with byte-identical configs differ by ~25 MAE. Thorfinn's PR #482 (multi-seed baseline + deterministic seeding) is in flight. Until it lands, within-sweep deltas remain the cleanest signal; absolute numbers vs. the merged baseline are noise-bound at single seed.

**Test-aggregate NaN bug RESOLVED.** Frieren independently diagnosed the same root cause as thorfinn (`0 * inf = NaN` from `-inf` values in `test_geom_camber_cruise/000020.pt`'s ground-truth pressure) and submitted a clean train-side safety net. Cherry-picked into advisor as commit `32b5b40`. All sibling Round-1 PRs now produce finite `test_avg/mae_surf_p` once they pull the rebased baseline.

**Rebase warning is still active for in-flight Round-1 PRs:** all PRs (#294, #315, #316, #317, #322) drafted against the OLD baseline must demonstrate their lever still wins on top of `peak_lr=1e-3, warmup_epochs=2`. PR #323 was first to be sent back; PR #319 (depth) was closed as compute-confounded at this budget; PR #397 (nezuko's original NaN follow-up) was closed because the safety-net work is now upstream.

**Round 2 already begun in the LR-schedule lane:** frieren reassigned to OneCycleLR (PR #409) and nezuko reassigned to EMA of weights (PR #410) — both compound on the merged warmup baseline along the training-recipe axis.

## Round 1 — orthogonal axes assigned (2026-04-27)

| Student | PR | Lever | Status | Result |
|---|---|---|---|---|
| alphonse | [#294](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/294) | **Loss alignment** — Huber surface loss (sweep δ ∈ {0.5, 1.0, 2.0}) | **rebase + re-run** | **In-sweep δ=0.5 hits 106.36 — already beats the new bar (115.84) at OLD LR.** Strongest Round-1 signal so far; monotonic trend toward L1; rebase to test interaction with warmup, extend to δ=0. |
| askeladd | [#315](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/315) | **Width** — `n_hidden` 128 → 192 (sweep 160/192/256) | wip | — |
| edward | [#316](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/316) | **Slice count** — `slice_num` 64 → 128 (sweep 96/128/192) | **CLOSED** | Negative result: slice_num=64 baseline wins; matches Transolver paper ablation. Compute-budget bound. |
| fern | [#317](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/317) | **Surface-vs-volume balance** — `surf_weight` sweep {5, 20, 40, 80} | **rebase + re-run** | In-sweep −10.1% (143.93→129.41 at sw=20); abs below new bar; clean U-shape, volume tradeoff visible. |
| frieren | [#319](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/319) | **Depth** — `n_layers` 5 → 8 (sweep 6/8/10) | **CLOSED** | Compute-confounded at 30-min budget (deeper = fewer epochs); n_layers=6 wins in-sweep at 143.33, below new bar. Bug fix cherry-picked. |
| nezuko | [#320](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/320) | **LR schedule** — linear warmup + peak LR ∈ {5e-4, 1e-3, 2e-3} | **MERGED** | **−21.5%** (147.55 → 115.84); peak_lr=1e-3 wins |
| tanjiro | [#322](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/322) | **Channel weighting** — upweight pressure in surface loss (sweep p_w ∈ {1, 2, 3, 5}) | **CLOSED** | Round 2 rebased: optimum shifted (3.0→5.0) and r1 winner regressed +29 MAE. Lever effect (~10 MAE) below seed-noise floor (~15–30 MAE). Direction/magnitude consistent across rounds — *parked, not failed*. Revisit post-multi-seed. |
| thorfinn | [#323](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/323) | **FFN expressivity** — `mlp_ratio` 2 → 4 (sweep 2/4/6) | **CLOSED** | Round-2 rebased sweep: ratio=2 control (140.70) > ratio=4 (145.36) > ratio=6 (154.40). Lever does not stack. **Surfaced critical seed-variance finding**: PR #320's 115.84 was single favorable seed — true baseline mean unknown until PR #482 (multi-seed) lands. |

These eight axes were chosen for **orthogonality** so that improvements compound when winners are merged sequentially: loss function, width, slice count, surface/volume balance, depth, LR schedule, channel weighting, and FFN expressivity touch nearly disjoint parts of the model and training stack.

## In-track baseline (post-PR #320 merge)

- `val_avg/mae_surf_p`: **115.84** (best epoch 14, run `w3mjq2ua`).
- `test_avg/mae_surf_p`: **NaN** — pre-existing `test_geom_camber_cruise` NaN-prediction bug. Mean of the 3 valid test splits: 112.78.
- Hyperparams: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`.
- All Round-1 sweep runs hit the 30-min timeout at ~epoch 14 of 50; cosine never fully annealed at this budget.

## Round 2 — extensions on the merged baseline (2026-04-28)

| Student | PR | Lever | Status |
|---|---|---|---|
| frieren | [#409](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/409) | **OneCycleLR** — onecycle_peak_lr=2e-3, pct_start=0.1, total_epochs=15 | **MERGED.** Within-sweep −7.51 MAE; vs prior merged baseline −7.15 MAE. New baseline 87.74. **Discovered EMA-vs-live sign-flip across schedules.** |
| nezuko | [#410](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/410) | **EMA of weights at eval time** — sweep decay ∈ {0.99, 0.999, 0.9995} | **MERGED** (within-sweep −21.7 MAE; live-vs-EMA −30.7 MAE in-run) |
| edward | [#420](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/420) | **Random Fourier features for spatial coords** — single-σ + multi-scale + concat_raw + EMA | **CLOSED** — Fourier+EMA substitute (both regularize spectral fit); stacking captures only ⅕ of solo effects. Multi-scale destabilized training. Direction real but doesn't compound enough vs new merged baseline 94.89. |
| alphonse | [#609](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/609) | **Focal-L1** — per-node residual reweighting on top of L1 | **CLOSED** — monotonic regression (γ=0→0.5→1→2 gives 98.19→102.71→114.37→212.08 val_avg). Mechanism cancellation: focal amplifies large residuals, L1 specifically chosen NOT to. **Reusable team insight: residual-magnitude reweighting is the wrong axis once loss is already the metric.** |
| alphonse | [#691](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/691) | **Saturated L1 (Lipschitz-clip)** — bound pathological residual contributions; opposite-sign of focal | wip |
| edward | [#618](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/618) | **UNet-style skip from preprocess into last block** — concat+linear vs. ReZero-gated | **iterate to ReZero-gated** — concat+linear regressed +1.68 MAE due to fusion init; clean diagnostic; α=0-init gate fixes the perturbation problem |
| frieren | [#671](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/671) | **`use_ema=False` under OneCycle** — confirm EMA-vs-live diagnostic; predicted val_avg ≈ 82 (−5.7 MAE) | wip — focused 2-run confirmation |
| thorfinn | [#482](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/482) | **Multi-seed baseline + deterministic seeding** (research infrastructure) | wip — redirected to post-EMA baseline |

## Round 3 — extensions on the post-EMA baseline (2026-04-28)

| Student | PR | Lever | Predicted Δ |
|---|---|---|---|
| nezuko | [#502](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/502) | **AdamW betas + weight_decay sweep** — β1∈{0.85, 0.9}, β2∈{0.99, 0.999, 0.9995}, wd∈{1e-4, 1e-3, 1e-2}; 1D-axis + 2D-corner design | −1 to −5% |
| tanjiro | [#508](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/508) | **Per-sample inverse-std weighting** | **CLOSED** — mechanism confirmed (cruise +, raceCar −) but magnitude bounded by dataset structure (within-sweep Δ=4.5 MAE < noise floor). Bigger reweighting makes aggregate worse, not better. |
| tanjiro | [#577](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/577) | **Surface-only auxiliary pressure head** — decouple capacity allocation from loss balancing | −5 to −15% |
| alphonse | [#609](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/609) | **Focal-L1** — per-node residual reweighting on top of L1 surface loss; γ ∈ {0, 0.5, 1.0, 2.0} | −3 to −8% (borderline at noise floor) |

## In-flight rebase PRs (Round 1, against new baseline)

| Student | PR | Lever | In-sweep Δ |
|---|---|---|---|
| **alphonse** | [#294](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/294) | **Pure L1 surface loss** (huber_delta=0) | **MERGED.** Within-sweep −15.72 MAE; vs prior merged baseline −26.55 MAE val, −24.72 MAE test. Compounding with warmup+EMA confirmed. New merged baseline 94.89. |
| fern | [#317](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/317) | `surf_weight=20` | −10.1% in-sweep at OLD LR (143.93→129.41). Within-sweep delta is robust because all runs share their seed environment. |

**Decision shift after the variance finding:** within-sweep deltas (where all runs in a group share their seed-noise environment) remain the cleanest signal — but only when the lever effect is *larger* than the seed-noise floor (~15–30 MAE at this budget). Channel-weighting (#322) was a counterexample: lever ~10 MAE < noise → couldn't confidently merge. Future lever-design should aim for predicted-effect-size > 25 MAE to be merge-able from a single-seed sweep before PR #482 lands.

## Potential next research directions

After Round 1 results land:

1. **Compound the winners.** Stack the top two or three orthogonal levers in a single PR (e.g. wider + deeper + Huber) to confirm gains add.
2. **Tackle the hardest split.** Whichever val split is the worst-performing across the round-1 winners gets a dedicated hypothesis next: the OOD-camber tracks (`val_geom_camber_rc`, `val_geom_camber_cruise`) are likely candidates because they require geometry extrapolation.
3. **Surface-only auxiliary head.** A dedicated MLP path that predicts pressure at surface nodes from the shared backbone, with its own loss term — addresses surface specialization without sacrificing volume regularization.
4. **Better data/feature engineering.** The 24-d input includes shape descriptors (`saf`, `dsdf`); a positional/feature-encoding experiment (Fourier features over `(x, z)`, learned positional embeddings, or RoPE-style encodings on the foil-relative coordinates) could be high-signal.
5. **Sample-difficulty-aware training.** High-Re samples drive the extreme y values; per-sample or per-region loss reweighting based on residual magnitude could help.
6. **Architectural alternatives.** If Transolver tweaks plateau early, candidates from the literature: GINO (graph-INO), OFormer, Geo-FNO, or PointNet++ with attention pooling.
7. **Test-time augmentation / ensembling.** Rotation/reflection symmetries of the airfoil setup are limited, but multi-seed ensemble of the best config is a near-free win on test metrics.
8. **Loss-landscape engineering.** EMA of weights, gradient clipping, AdamW betas, weight-decay sweep — small-but-real wins that are easy follow-ups once a strong base config is established.

## Operational notes

- Branching: all advisor work on `icml-appendix-willow-pai2d-r3`. PRs target it as base. Merges squash into it.
- Each round-1 PR contains a small grouped sweep (3–5 runs) so we get a curve, not a point. W&B groups: `huber-loss-surf-p`, `wider-n-hidden`, `slice-num-sweep`, `surface-weight-sweep`, `deeper-n-layers`, `lr-warmup-sweep`, `channel-weighted-loss`, `mlp-ratio-sweep`.
- Decision criteria from CLAUDE.md: merge if any sub-run beats baseline on `val_avg/mae_surf_p` (even small gains compound); request changes for promising but not-yet-winning directions; close only fundamentally broken approaches (>5% regression or crashes).
