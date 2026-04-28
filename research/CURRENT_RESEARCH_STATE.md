# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-28 19:30
- **Advisor branch:** `icml-appendix-willow-pai2e-r5`
- **Track tag:** `willow-pai2e-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r5`
- **Most recent direction from human team:** none yet (no human GitHub issues open).

## Research target

Beat the Transolver baseline on TandemFoilSet. Primary ranking metric is
`val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the four
validation splits) with `test_avg/mae_surf_p` as the paper-facing decision
metric. Baseline config is `n_hidden=128, n_layers=5, n_head=4, slice_num=64,
mlp_ratio=2`, AdamW + CosineAnnealingLR, `lr=5e-4`, `surf_weight=10.0`,
batch=4, 50 epochs, vol+10·surf MSE in normalized space.

The four validation tracks each probe a different generalization axis:
- `val_single_in_dist` — random holdout from single-foil (sanity).
- `val_geom_camber_rc` — unseen front-foil camber raceCar tandem M=6-8.
- `val_geom_camber_cruise` — unseen front-foil camber cruise tandem M=2-4.
- `val_re_rand` — stratified Re holdout across all tandem domains.

Per programme contract, surface pressure on the held-out camber/Re splits is
where the paper-facing numbers live. Per-sample y std varies by an order of
magnitude even inside one domain, so high-Re samples drive the extremes.

## Wave 1 — currently in flight (assigned, training)

| Student | PR | Hypothesis family | Why it's interesting |
|---------|----|-------------------|----------------------|
| alphonse | #732 | Larger model capacity (n_hidden 128→256, n_layers 5→7, n_head 4→8) | Capacity headroom — Transolver scaling. |
| askeladd | #733 | More slices (slice_num 64→256) | Finer physics-attention partitions for long meshes. |
| edward | #734 | Higher surf_weight (10→50) | Direct lever on the primary metric. |
| fern | #737 | LR warmup + cosine decay | Optimization stability, especially early epochs. |
| frieren | #739 | Huber loss replacing MSE | Robustness to high-Re extreme pressure spikes. |
| nezuko | #742 | Dropout regularization | Reduce overfit on small (~1.5k) train set. |
| tanjiro | #745 | Separate output heads (Ux, Uy, p) | Decouple pressure head from velocity dynamics. |
| thorfinn | #763 | Physics-informed distance features (dist-to-surface, log-dist, tandem indicator) | Inductive bias for boundary-layer / tandem interaction. |

This wave probes capacity, attention partitioning, loss reweighting, schedule,
robust loss, regularization, output decomposition, and feature engineering —
roughly orthogonal levers. Surface pressure metric will tell us which axes
matter most.

## Current research themes

1. **Get the Transolver baseline tuned.** Wave 1 covers all the obvious
   hyperparameter and loss levers. Whichever wins becomes part of the new
   baseline; the rest tell us where the head-room *isn't*.
2. **Surface-vs-volume balance.** `surf_weight` and tanjiro's separate-heads
   experiment will tell us whether pressure and velocity benefit from
   different optimization signal.
3. **Generalization-axis split disagreement.** The four val splits will likely
   disagree on which intervention helps. We treat that disagreement as
   information about which physics axis is the binding constraint.

## Potential next research directions (Wave 2+ candidates)

A `researcher-agent` is currently surveying the literature for Wave 2
hypotheses. Until that lands, the priors I'm carrying for next wave:

- **Architecture pivot.** If Transolver tuning saturates fast, swap in or
  hybridize with: GINOs / FNOs (spectral mixing), MeshGraphNets (explicit
  edges), neural-operator transformers, equivariant attention.
- **Multi-scale / hierarchical attention.** Meshes vary 74K–242K nodes and
  span three physical zones — coarse-to-fine attention or learned
  pooling/unpooling could unlock tandem zone-2/zone-3 interactions.
- **Pressure-aware output formulation.** Predict `p` in scale-aware
  space — log-magnitude, per-domain rescale, or decompose into
  Bernoulli + correction.
- **Re-conditional sampling / curriculum.** High-Re samples drive the loss;
  hard-example mining or Re-stratified curriculum may even out training.
- **Geometric augmentation.** Mirror, x-translate, modest camber jitter for
  augmentation that respects the physics symmetries.
- **Regularization of the right kind.** Stochastic depth, attention dropout
  on slice tokens, weight averaging (SWA / EMA) — variance reduction at
  evaluation, not just training.
- **Boundary-layer specialization.** Auxiliary loss on `is_surface` nodes
  (e.g., heteroscedastic head, gradient-reweighted loss).
- **Test-time adaptation.** TTA via per-sample fine-tuning on geometry
  features for the unseen-camber splits.

This list will be refreshed once the researcher-agent's
`RESEARCH_IDEAS_2026-04-28_19:30.md` lands.

## Open questions

- We have no measured baseline yet — Wave 1 is the first complete run.
  Whichever single-config training run lands fastest gives us a value to
  pin in `BASELINE.md`.
- Whether the four val tracks rank the same intervention identically (likely
  not — that disagreement will shape Wave 2 priorities).
- Whether 30-min/50-epoch budget is binding for the larger-capacity arm
  (alphonse). If it is, the `--epochs` budget knob isn't movable, but
  throughput optimizations (mixed precision, checkpointing) might be.
