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

## Wave 1 status

| Student | PR | Status | Result |
|---------|----|--------|--------|
| alphonse | #732 | Closed | val_avg=154.95 ref; NaN test; 6/50 epochs |
| alphonse | #796 | WIP | FiLM-Re conditioning |
| askeladd | #733 | Closed | val_avg=151.50; +18.5% regression; throughput cost decisive |
| askeladd | #811 | WIP | Mixed-precision BF16 — platform throughput improvement |
| edward | #734 | WIP | Higher surf_weight (10→50) |
| fern | #737 | **Merged** | val_avg=127.87 ← best; warmup+cosine |
| fern | #809 | WIP | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | WIP | Huber loss |
| nezuko | #742 | WIP | Dropout regularization |
| tanjiro | #745 | WIP | Separate output heads |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | WIP | EMA model checkpoint |

**Current best val_avg/mae_surf_p:** 127.872 (fern #737, run 5b22tecz) — merged baseline.
**test_avg/mae_surf_p:** 126.5598 (thorfinn #763, run 072wo9xb) — most recent clean test number.

**Key learnings from Wave-1 so far:**
- Throughput is binding: 0.93M-param baseline does ~20 epochs/30min; 3.01M-param does ~6 epochs. BF16/grad-checkpointing needed before scaling.
- Trajectory suggests capacity is NOT the bottleneck at first contact — need to establish baseline first.
- `test_geom_camber_cruise` NaN on under-trained large models; scoring.py NaN-pred gap confirmed (data/ read-only).

**Awaiting:** edward, nezuko, askeladd, fern, frieren, tanjiro, thorfinn results to establish baseline architecture val_avg benchmark.

## Current research themes

1. **Merged baseline now has two stacked wins:** physics distance features (#763) + warmup+cosine LR (#737). val_avg=127.87. Next generation of runs starts from here.
2. **Schedule sizing is a major lever.** T_max=50 with a 30-min (~14 epoch) wall-clock budget means the LR barely decays. PR #809 tests the obvious fix (epochs=14, warmup=2). Expect notable improvement.
3. **Critical dataset bug resolved.** `test_geom_camber_cruise` sample 20 has 761 NaN y[:, 2] entries. NaN-safe `evaluate_split` workaround merged in #763. All future runs from the new baseline will report finite test_avg.
4. **5 Wave-1 PRs still in-flight** (#733 #734 #739 #742 #745) — running against old baseline. When they submit, we evaluate whether their isolated intervention beats 127.87 after rebase.
5. **EMA (#810) and FiLM-Re (#796)** are the two highest-priority Wave-2 explorations: variance reduction and Re-regime conditioning respectively.

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

Full bank in `research/RESEARCH_IDEAS_2026-04-28_19:30.md`.

## Open questions

- We have no measured baseline yet — Wave 1 is the first complete run.
  Whichever single-config training run lands fastest gives us a value to
  pin in `BASELINE.md`.
- Whether the four val tracks rank the same intervention identically (likely
  not — that disagreement will shape Wave 2 priorities).
- Whether 30-min/50-epoch budget is binding for the larger-capacity arm
  (alphonse). If it is, the `--epochs` budget knob isn't movable, but
  throughput optimizations (mixed precision, checkpointing) might be.
