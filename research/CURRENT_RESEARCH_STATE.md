# SENPAI Research State
- 2026-04-28 20:30 UTC
- Most recent research direction from human researcher team: None received yet (no GitHub Issues found)
- Current research focus and themes: Round 1 — Baseline parameter sweeps and loss function experiments on the Transolver CFD surrogate for TandemFoilSet-Balanced

## Current Research Focus

The research target is `val_avg/mae_surf_p` and `test_avg/mae_surf_p` (surface pressure MAE, lower is better). The prior competition best is **40.927** (test), from a 3-layer Transolver with slice_num=16.

Round 1 is testing 8 independent hypotheses simultaneously. Two PRs have already been reviewed and sent back to students:

### Active WIP PRs (in flight)
| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #788 | alphonse | Huber loss instead of MSE | Running |
| #789 | askeladd | Gradient clipping (max_norm=1.0) | Running |
| #790 | edward | Increase surf_weight 10→30 | Running |
| #792 | frieren | Deeper Transolver: n_layers 5→8, lr 5e-4→3e-4 | Reviewed: fix NaN + rerun at n_layers=6 |
| #793 | nezuko | Finer physics partitioning: slice_num 64→128 | Running |
| #794 | tanjiro | LR warmup (5 epochs) before cosine annealing | Running |
| #795 | thorfinn | Per-sample loss normalization | Running |
| #791 | fern | Wider model: n_hidden 128→256, n_head 4→8 | Reviewed: sent back for bf16 follow-up |
| #808 | fern | BF16 mixed precision for wider model (n_hidden=256, n_head=8) | New PR from review of #791 |

### Key Early Findings from Round 1

From PR #791 (fern, wider model, reviewed at epoch 7/50):
- **Infrastructure bug discovered**: `accumulate_batch` NaN propagation bug — `0 * NaN = NaN` in IEEE 754. Fixed in train.py with a sample filter in `evaluate_split`. This affects all experiments.
- The wider model (2.54M params) shows strong learning signal but 30-min timeout allows only 7 epochs vs ~25+ for the baseline.
- bf16 mixed precision (PR #808) should roughly halve epoch time, enabling ~14 epochs.

From PR #792 (frieren, n_layers=8, reviewed at epoch 9/50):
- NaN on `test_geom_camber_cruise` — p-channel numerical instability with 8 layers and extreme pressure magnitudes.
- 2.4x slowdown per epoch means only 9/50 epochs in budget — impractical.
- Recommendation: run n_layers=6 with gradient clipping to fix NaN.

## Key Findings from Prior Competition

From the prior 12-hour competition (morganmcg1/tandemfoil2, 30 PRs):
- **Best config**: n_layers=3, slice_num=16 (counterintuitive — fewer layers won)
- n_head reduction and slice_num reduction (to 16) significantly outperformed the default baseline
- SwiGLU activation + Fourier sigma=0.7 gave a meaningful improvement over vanilla baseline
- Combining n_layers=3, sn=16 produced a triple compound improvement
- Capacity sweeps (more hidden width) did NOT help — overparameterization appears harmful
- Horizontal flip augmentation was worse

## Potential Next Research Directions (Round 2+)

Once Round 1 results are in, prioritized directions:

1. **Reproduce the winning config from prior competition** — Confirm nl=3, sn=16, SwiGLU+Fourier combo as explicit baseline for this new track (HIGHEST PRIORITY)
2. **n_layers=6 with grad clipping** — Follow-up from frieren/PR #792; intermediate depth with NaN fix
3. **bf16 result from #808** — Verify if wider model + bf16 converges within budget; may need iso-epoch comparison
4. **Pressure channel specialization** — Dual-head decoder: shared trunk for Ux/Uy, dedicated head for p (since p is what we optimize)
5. **High-Re weighting** — Per-sample weighting by Re magnitude to emphasize hard (high-Re) cases
6. **Slice rebalancing** — The physics partitions may not align with surface nodes; adaptive or reduced slice_num (try sn=16 from prior best)
7. **Node subsampling** — Random subsampling of interior nodes during training to speed up iteration; keep all surface nodes
8. **Geometry-aware positional encoding** — Encoding foil chord length, gap and stagger directly into position encoding
9. **Ensemble / multi-seed** — Run same best config with 3 seeds and average predictions
10. **Separate surface/volume prediction streams** — Predict p on surface nodes with a specialized smaller MLP head
