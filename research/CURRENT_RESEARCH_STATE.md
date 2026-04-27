# SENPAI Research State

- **Date:** 2026-04-27
- **Advisor branch:** `icml-appendix-willow-pai2d-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r3`
- **Most recent human-team direction:** none received (fresh launch)

## Current research focus

Establish a strong baseline above the default Transolver on TandemFoilSet by sweeping **eight orthogonal first-round levers** in parallel. Primary metric is `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the four val splits); paper-facing number is `test_avg/mae_surf_p` from the best-val checkpoint.

No in-track baseline established yet — the first wave of round-1 results will define the reference numbers for this advisor branch's BASELINE.md.

## Round 1 — orthogonal axes assigned (2026-04-27)

| Student | PR | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|---|---|---|---|
| alphonse | [#294](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/294) | **Loss alignment** — Huber surface loss (sweep δ ∈ {0.5, 1.0, 2.0}) | −3 to −8% |
| askeladd | [#315](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/315) | **Width** — `n_hidden` 128 → 192 (sweep 160/192/256) | −5 to −10% |
| edward | [#316](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/316) | **Slice count** — `slice_num` 64 → 128 (sweep 96/128/192) | −5 to −12% |
| fern | [#317](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/317) | **Surface-vs-volume balance** — `surf_weight` sweep {5, 20, 40, 80} | −3 to −8% |
| frieren | [#319](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/319) | **Depth** — `n_layers` 5 → 8 (sweep 6/8/10) | −3 to −8% |
| nezuko | [#320](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/320) | **LR schedule** — linear warmup + higher peak LR (sweep peak ∈ {5e-4, 1e-3, 2e-3}) | −3 to −7% |
| tanjiro | [#322](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/322) | **Channel weighting** — upweight pressure in surface loss (sweep p_w ∈ {1, 2, 3, 5}) | −3 to −6% |
| thorfinn | [#323](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/323) | **FFN expressivity** — `mlp_ratio` 2 → 4 (sweep 2/4/6) | −3 to −7% |

These eight axes were chosen for **orthogonality** so that improvements compound when winners are merged sequentially: loss function, width, slice count, surface/volume balance, depth, LR schedule, channel weighting, and FFN expressivity touch nearly disjoint parts of the model and training stack.

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
