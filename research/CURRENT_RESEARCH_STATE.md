# SENPAI Research State
- 2026-04-28 19:30 UTC
- Most recent research direction from human researcher team: None received yet (no GitHub Issues found)
- Current research focus and themes: Round 1 — Baseline parameter sweeps and loss function experiments on the Transolver CFD surrogate for TandemFoilSet-Balanced

## Current Research Focus

The research target is `val_avg/mae_surf_p` and `test_avg/mae_surf_p` (surface pressure MAE, lower is better). The prior competition best is **40.927** (test), from a 3-layer Transolver with slice_num=16.

Round 1 is testing 8 independent hypotheses simultaneously, exploring:

1. **Loss alignment** — Huber loss (alphonse) and per-sample normalization (thorfinn) to better align training objective with evaluation MAE
2. **Surface focus** — Higher surf_weight 10→30 (edward) to drive more gradient through surface nodes
3. **Training stability** — Gradient clipping max_norm=1.0 (askeladd) and LR warmup 5 epochs (tanjiro) to stabilize early training
4. **Capacity** — Wider model n_hidden 256 (fern), deeper model n_layers=8 (frieren) to increase model expressivity
5. **Physics resolution** — slice_num 64→128 (nezuko) for finer partitioning

## Key Findings from Prior Competition

From the prior 12-hour competition (morganmcg1/tandemfoil2, 30 PRs):
- **Best config**: n_layers=3, slice_num=16 (counterintuitive — fewer layers won)
- n_head reduction and slice_num reduction (to 16) significantly outperformed the default baseline
- SwiGLU activation + Fourier sigma=0.7 gave a meaningful improvement over vanilla baseline
- Combining n_layers=3, sn=16 produced a triple compound improvement
- Capacity sweeps (more hidden width) did NOT help — overparameterization appears harmful
- Horizontal flip augmentation was worse

## Potential Next Research Directions

These themes should be explored in future rounds once Round 1 results are in:

1. **Reproduce the winning config from prior competition** — Confirm nl=3, sn=16, SwiGLU+Fourier combo as explicit baseline for this new track
2. **Pressure channel specialization** — Dual-head decoder: shared trunk for Ux/Uy, dedicated head for p (since p is what we optimize)
3. **Slice rebalancing** — The physics partitions (64 slices default) may not align with the surface nodes; adaptive slice placement
4. **High-Re weighting** — Per-sample weighting by Re magnitude to emphasize hard (high-Re) cases
5. **Mixed precision (AMP)** — Was used in prior competition but not yet this round; throughput gain → more epochs within timeout
6. **Cosine annealing** — Already in tanjiro's warmup PR; could test standalone
7. **Node subsampling** — Random subsampling of interior nodes during training to speed up iteration; keep all surface nodes
8. **Geometry-aware positional encoding** — Encoding foil chord length, gap and stagger directly into position encoding
9. **Ensemble / multi-seed** — Run same best config with 3 seeds and average predictions
10. **Separate surface/volume prediction streams** — Predict p on surface nodes with a specialized smaller MLP head
