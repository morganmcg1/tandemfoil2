# SENPAI Research State

- **Date:** 2026-04-27
- **Advisor branch:** icml-appendix-willow-pai2c-r2
- **W&B project:** wandb-applied-ai-team/senpai-charlie-wilson-willow-r2
- **Human researcher directives:** None received so far.

## Current Research Focus

**Round 2 — fresh launch.** All 8 students have been assigned first-round experiments covering 7 orthogonal improvement axes. No baseline has been established yet on this branch; PR #185 (alphonse) will produce the anchor.

The primary metric is `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits). Lower is better.

### Active Round-1 PRs

| PR | Student | Hypothesis | Key Change |
|----|---------|------------|------------|
| #185 | alphonse | Baseline anchor (3-seed control) | No code change; seed the RNG; run 3 seeds to measure variance |
| #186 | askeladd | Capacity: n_hidden=192, n_layers=7, n_head=8 | Wider + deeper Transolver |
| #187 | edward | surf_weight sweep: 5, 20, 50 | Loss weight on surface vs volume nodes |
| #188 | fern | Signed-log pressure transform (asinh-style) | Equalize Re-range gradients on heavy-tailed pressure |
| #189 | frieren | lr=1e-3 with 5-ep warmup + cosine | Optimizer schedule |
| #191 | nezuko | FiLM conditioning on log(Re)+NACA per block | Re-aware hidden state scaling |
| #192 | tanjiro | Weight EMA (decay=0.999) for checkpoint selection | Smoother model averaging |
| #194 | thorfinn | Huber loss (delta=1.0, 0.5) vs MSE | Robust regression for heavy-tailed p |

## Current Best Baseline

No baseline established yet on this branch. PR #185 will produce the anchor — expected `val_avg/mae_surf_p` near the upstream Transolver default.

## Key Research Hypotheses (Priority Order)

1. **Signed-log pressure (fern, PR #188)** — Researcher-agent's top pick. Heavy-tailed pressure field (~5 orders of magnitude across Re) means MSE gradients are dominated by high-Re raceCar samples. Asinh/signed-log-transform equalizes gradients across Re ranges. Predicted -5 to -12%.

2. **FiLM Re-conditioning (nezuko, PR #191)** — Reynolds number as a per-block affine modulation instead of just another input feature. Gives explicit "dial" for flow regime. Predicted -4 to -8%, especially on val_re_rand.

3. **Capacity scaling (askeladd, PR #186)** — The baseline model is deliberately small. 96 GB VRAM allows n_hidden=192, n_layers=7 comfortably. Predicted -5 to -10%.

4. **Weight EMA (tanjiro, PR #192)** — Cheap Kaggle-grade win, compounds with everything. Predicted -1 to -4%.

5. **surf_weight sweep (edward, PR #187)** — Primary metric is surface-only; higher surf_weight should focus the model better. Predicted -3 to -8% at surf_weight=50.

6. **lr warmup (frieren, PR #189)** — Well-known stabilization recipe. Predicted -3 to -6%.

7. **Huber loss (thorfinn, PR #194)** — L1-aligned training for MAE evaluation, robust to outliers. Predicted -2 to -6%.

## Next Research Directions (Round 2+)

See `/research/RESEARCH_IDEAS_2026-04-27_round2.md` for 12 detailed ideas. Top candidates for round 2 (once round-1 winners are merged):

1. **Sobolev / arc-length gradient surface loss** — penalize pressure ringing along surface via finite-differences on the `saf` coordinate
2. **Reynolds-conditioned normalization** — per-sample Re-adaptive y scaling before loss
3. **Model souping** — averaging checkpoints from multiple runs at different hyperparameters
4. **Stochastic depth / DropPath** — regularization for the deeper Transolver variants
5. **log-domain NACA feature encoding** — thickness/camber on a log scale to help with extreme geometries

## Physical Understanding

- The dataset spans 3 domains: raceCar single (~85K nodes/sample), raceCar tandem (~127K), cruise tandem (~210K)
- Pressure is by far the hardest channel: spans ±29K raw values vs velocity channels in the ±500 range
- The camber holdouts (val_geom_camber_rc, val_geom_camber_cruise) test interpolation to unseen front-foil shapes
- val_re_rand tests cross-regime Re generalization (stratified holdout across all Re in tandem domains)
- The balanced sampler ensures equal domain representation; raceCar single has 599 training samples vs 457/443 for the tandem domains
