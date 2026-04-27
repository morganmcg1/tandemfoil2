# SENPAI Research State — TandemFoilSet (icml-appendix-willow-pai2-r3)

- **Date:** 2026-04-27
- **Round:** 1 (cluster reset — pai-2)
- **Branch:** `icml-appendix-willow-pai2-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-pai2-r3`
- **Latest human directive:** none received yet on this round.

## Context

Prior willow rounds (`willow1`–`willow5`) ran on the wrong Kubernetes cluster
and were closed without merging. We are starting cleanly on `pai-2`. The
proven recipe from sibling work (`senpai-kagent-v-students`) drove
`val_avg/mae_surf_p` from ~88 → ~49 on a different cluster and is the obvious
first port to verify here.

## Current research focus

Round 1 has two simultaneous goals:

1. **Anchor the cluster.** Verify that the proven recipe (L1+sw=1, AMP+ga=4,
   Fourier PE+FiLM, SwiGLU, n_layers=3, slice_num=8) reproduces the prior
   ~49 val MAE on `pai-2`. One student (alphonse) is dedicated to this; their
   PR also includes a clean MSE/vanilla anchor run for the cluster.

2. **Probe orthogonal levers on the bare baseline.** Seven students each test
   a single high-leverage change *on top of* the vanilla baseline so the
   per-lever signal is comparable across the team. The expected delta is
   smaller than the recipe's compute-reduction win, but each lever is
   orthogonal to the recipe and a winner can be stacked on top in Round 2.

## Round-1 lever assignments

| Student | Hypothesis | Why |
|---------|-----------|-----|
| alphonse | Proven recipe + 2 seeds + MSE anchor | Cluster anchor / recipe verification |
| askeladd | Bare + Lookahead optimizer (k=5, α=0.5) | Outer optimization smoothing for noisy CFD batches |
| edward | Bare + EMA weights (decay=0.999, timm-style warmup) | Smoother validation minima at no train cost |
| fern | Bare + signed distance + wall-normal features | Geometry-aware features absent from x[24] |
| frieren | Bare + Huber loss δ ∈ {0.1, 0.2, 0.5, 1.0} | Heavy-tailed pressure → metric-aligned robust loss |
| nezuko | Bare + DropPath stochastic depth p ∈ {0.05, 0.1, 0.2} | OOD geometry generalisation regularizer |
| tanjiro | Bare + 2D RoPE on slice-token centroids | Spatial locality bias for physics-attention |
| thorfinn | Bare + per-channel loss balancing (Ux, Uy, p) | Prevent p (heavy-tail) from dominating MSE |

## Potential next research directions (Round 2+)

Each is contingent on the Round-1 outcomes:

- **Stack winners on the recipe.** Whichever bare-baseline lever beats vanilla
  will be retested combined with the recipe (recipe + lever).
- **Capacity scaling on the recipe.** With slice_num=8 and n_layers=3, there is
  unused VRAM headroom; sweep n_hidden ∈ {128, 192, 256} and n_layers up to 6
  on the recipe.
- **Loss reformulation beyond Huber.** Quantile loss (asymmetric for pressure
  extrema), Sobolev gradient loss on surface (PR #103 hypothesis), or
  uncertainty-weighted heteroscedastic loss.
- **Data-efficient training.** Mixup / Cutmix on mesh nodes, MAE pretraining
  on volume nodes followed by surface fine-tuning, or curriculum on Reynolds.
- **Equivariance.** SE(2) chord-line frame (rotate to AoA=0) — large-payoff
  candidate that was started but never measured (PR #104).
- **Test-time augmentation.** Reflect across symmetry axis at inference;
  average predictions for free generalisation gain.
- **Architectural alternatives.** GINO / ANO / Flow-Aware mesh transformer if
  Transolver plateaus.
