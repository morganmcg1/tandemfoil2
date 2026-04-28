# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2d-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`
- **Most recent human-team direction:** none received (fresh launch)

## Current research focus

**⚠️ CRITICAL UPDATE (2026-04-28 02:55):** A seed-variance investigation triggered by PR #323 r2 found that `train.py` has **no seed control at all** — `torch.manual_seed`, `np.random.seed`, `random.seed` all absent. PR #320's "baseline" of `val_avg/mae_surf_p = 115.84` was a single favorable seed; thorfinn's rebased control with byte-identical config produced 140.70 (a 25-point, ~21% gap). The true baseline mean is likely ~130-145 with very wide variance. **All in-flight rebase decisions are operating against a single-seed bar that may be unrepresentative.** Thorfinn reassigned to PR #482 (multi-seed baseline + deterministic seeding); the multi-seed `mean ± std` will replace 115.84 as the merge bar within the next ~2.5h of GPU.

First Round-1 winner merged: **PR #320** (linear warmup + peak LR 1e-3) reportedly dropped `val_avg/mae_surf_p` from 147.55 → 115.84 (−21.5%, uniform across all 4 val splits) at a single seed. The 21.5% headline figure is now uncertain pending the multi-seed study.

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
| tanjiro | [#322](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/322) | **Channel weighting** — upweight pressure in surface loss (sweep p_w ∈ {1, 2, 3, 5}) | **rebase + re-run** | In-sweep −9.1% (138.87→126.18 at p_w=3); abs below new bar; clean U-shape with sharp optimum. |
| thorfinn | [#323](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/323) | **FFN expressivity** — `mlp_ratio` 2 → 4 (sweep 2/4/6) | **CLOSED** | Round-2 rebased sweep: ratio=2 control (140.70) > ratio=4 (145.36) > ratio=6 (154.40). Lever does not stack. **Surfaced critical seed-variance finding**: PR #320's 115.84 was single favorable seed — true baseline mean unknown until PR #482 (multi-seed) lands. |

These eight axes were chosen for **orthogonality** so that improvements compound when winners are merged sequentially: loss function, width, slice count, surface/volume balance, depth, LR schedule, channel weighting, and FFN expressivity touch nearly disjoint parts of the model and training stack.

## In-track baseline (post-PR #320 merge)

- `val_avg/mae_surf_p`: **115.84** (best epoch 14, run `w3mjq2ua`).
- `test_avg/mae_surf_p`: **NaN** — pre-existing `test_geom_camber_cruise` NaN-prediction bug. Mean of the 3 valid test splits: 112.78.
- Hyperparams: `peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`.
- All Round-1 sweep runs hit the 30-min timeout at ~epoch 14 of 50; cosine never fully annealed at this budget.

## Round 2 — extensions on the merged baseline (2026-04-28)

| Student | PR | Lever | Predicted Δ |
|---|---|---|---|
| frieren | [#409](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/409) | **OneCycleLR** — onecycle_peak_lr ∈ {1e-3, 2e-3, 3e-3}, pct_start=0.1 | −2 to −5% |
| nezuko | [#410](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/410) | **EMA of weights at eval time** — sweep decay ∈ {0.99, 0.999, 0.9995} | −1 to −5% |
| edward | [#420](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/420) | **Random Fourier features for spatial coords** — sigma sweep {0.5, 1, 2, 5}, m=64 | −3 to −10% |
| **thorfinn** | [#482](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/482) | **Multi-seed baseline + deterministic seeding** (research infrastructure) | 0% (characterizes baseline) |

Each axis is orthogonal to the others and to the in-flight rebase PRs.

## In-flight rebase PRs (Round 1, against new baseline)

| Student | PR | Lever | In-sweep Δ |
|---|---|---|---|
| **alphonse** | [#294](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/294) | **Huber surface loss δ=0.5** | In-sweep 106.36 at OLD LR (vs single-seed bar 115.84). With variance ~25 points known now, hard to call this a confident win pre-rebase, but it's the strongest signal we have. |
| fern | [#317](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/317) | `surf_weight=20` | −10.1% in-sweep at OLD LR (143.93→129.41). Within-sweep delta is robust because all runs share their seed environment. |
| tanjiro | [#322](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/322) | `surf_p_weight=3.0` | −9.1% in-sweep at OLD LR (138.87→126.18). Within-sweep delta is robust. |

**Decision shift after the variance finding:** within-sweep deltas (where all runs in a group share their seed-noise environment) remain the cleanest signal. Cross-PR comparisons against the 115.84 single-seed baseline are unreliable. Once PR #482 lands, in-flight rebase PRs will be evaluated against the multi-seed mean ± std.

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
