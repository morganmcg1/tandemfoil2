# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-28 21:35
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
| askeladd | #811 | **Merged** | val_avg=127.402 ← best; test_avg=116.211 (clean); 1.20× speedup; 33 GB VRAM (63 GB free) |
| askeladd | #848 | WIP | Larger batch size (batch_size=8→12) using 63 GB VRAM headroom |
| edward | #734 | WIP | Higher surf_weight (10→50) |
| fern | #737 | **Merged** | val_avg=127.87 ← best; warmup+cosine |
| fern | #809 | WIP | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | WIP | Huber loss |
| nezuko | #742 | WIP | Dropout regularization |
| tanjiro | #745 | WIP (rebase) | Heads on old code: Opt1=130.82 / Opt2=134.46. Sent back for Option 3 capacity-matched on rebased baseline. |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | WIP | EMA model checkpoint |

**Current best val_avg/mae_surf_p:** 127.402 (askeladd #811, run newqt8dd) — merged baseline.
**test_avg/mae_surf_p:** 116.211 (askeladd #811, run newqt8dd) — clean 4-split, current best paper-facing metric.

**Key learnings from Wave-1 so far:**
- BF16 merged (#811): 1.20× per-epoch speedup, 17 epochs in 30 min (was 14), 33 GB VRAM. Now the dominant non-matmul cost is the `add_derived_features` Python loop.
- 63 GB VRAM headroom unlocked — batch_size scaling is the highest-leverage next throughput lever.
- `test_geom_camber_cruise` NaN on under-trained large models; scoring.py NaN-pred gap confirmed (data/ read-only). NaN-safe workaround in train.py merged (#763).
- Three compounding baseline wins: distance features (#763) + warmup+cosine (#737) + BF16 (#811).

**Awaiting:** edward #734, nezuko #742, frieren #739 (Wave-1 tail; pre-merge code) + alphonse #796, fern #809, thorfinn #810 (Wave-2) + tanjiro #745 (Option 3 rebase) + askeladd NEW (batch_size scaling).

## Current research themes

1. **Three compounding baseline wins:** distance features (#763) + warmup+cosine (#737) + BF16 (#811). val_avg=127.402, test_avg=116.211 (clean 4-split). This is the platform to build on.
2. **BF16 platform improvement merged.** All future runs inherit 1.20× per-epoch speedup and 63 GB VRAM headroom. Non-matmul cost (`add_derived_features` Python loop) is the new bottleneck.
3. **Batch size scaling is the highest-leverage next throughput lever.** With 33 GB VRAM at batch_size=4, doubling to 8 or tripling to 12 should fit comfortably. Larger batches → smoother gradients + more nodes/step. Askeladd assigned to this.
4. **3 Wave-1 PRs still in-flight against pre-merge code** (#734 #739 #742). When they submit, evaluate whether their isolated intervention beats 127.40 after rebase.
5. **EMA (#810) and FiLM-Re (#796)** are in-flight Wave-2 explorations; schedule-sized (#809) will test the LR-budget hypothesis.

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

- Whether batch_size scaling delivers a big additional speedup, or if the Python feature-loop + padding cost eat the savings.
- Whether the four val tracks rank interventions consistently (early data suggests cruise/Re-rand improve faster than single_in_dist and rc — may reflect domain difficulty ordering).
- Whether capacity scaling (larger n_hidden/n_layers) becomes viable once BF16 + batch_size unlock sufficient throughput.
- val_avg at epoch 17 was only -0.47 vs baseline epoch 14. The LR schedule sizing (#809 test) may unlock more by matching T_max to actual budget.
