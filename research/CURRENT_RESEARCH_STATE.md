# SENPAI Research State
- 2026-04-29 (branch: icml-appendix-charlie-pai2f-r4)
- No human researcher team directives received yet.

## Current Research Focus

**Target:** TandemFoilSet CFD surrogate — predict (Ux, Uy, p) at every mesh node.
**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 val splits (lower is better).
**Model:** Transolver with physics-aware attention over irregular meshes.
**Status:** Round 4. First result reviewed and merged (PR #1112). Baseline established.

## Baseline

| Metric | Value | PR |
|--------|-------|----|
| **val_avg/mae_surf_p** | **129.531** | #1112 (attention dropout=0.1, epoch 13/50) |

Per-split val breakdown:
- val_single_in_dist: 159.429
- val_geom_camber_rc: 155.559
- val_geom_camber_cruise: 92.955
- val_re_rand: 110.181

Note: `test_avg/mae_surf_p = NaN` due to corrupt `test_geom_camber_cruise/000020.pt` (761 inf values in ground-truth). Valid test splits show 139.9 / 138.8 / 111.0 for single_in_dist, geom_camber_rc, re_rand respectively.

## Active Experiments (Round 4)

| PR   | Student    | Hypothesis |
|------|------------|-----------|
| #1137 | nezuko    | Scale Transolver to n_hidden=256, n_layers=8 for high-Re splits |
| #1128 | edward    | Per-sample Re-adaptive loss normalization for pressure scale |
| #1117 | thorfinn  | Re-conditioned output scale head for magnitude adaptation |
| #1116 | tanjiro   | OneCycleLR (max_lr=2e-3, pct_start=0.3) vs cosine annealing |
| #1114 | frieren   | Curriculum surf_weight ramp 1→20 over training |
| #1113 | fern      | Learnable 3-way domain embedding for multi-domain routing |
| #1111 | askeladd  | Layer-wise LR decay for geometry-stable representations |
| #1110 | alphonse  | Log-modulus transform on pressure channel loss |

All 8 students are currently WIP. No PRs ready for review.
Last checked: 2026-04-29. All pods healthy (1/1 ready).

## Research Themes Being Explored

1. **Loss formulation**: Curriculum surface weight ramp; log-modulus pressure transform; Re-adaptive loss normalization
2. **LR scheduling**: OneCycleLR vs cosine annealing; layer-wise LR decay
3. **Architecture additions**: Re-conditioned output scale head; learnable domain embedding; scaled model capacity (n_hidden=256, n_layers=8)
4. **Regularization**: Attention dropout for OOD robustness (merged — established baseline)

## Key Observations from Merged PR

- Attention dropout=0.1 achieves val_avg/mae_surf_p=129.531 at epoch 13/50 (training timed out ~14 epochs, ~133s/epoch)
- Cruise split (92.955) is much easier than raceCar splits (~155-159), suggesting multi-domain difficulty imbalance
- Single_in_dist val (159.429) is worse than geom_camber_rc (155.559) — unusual given it should be the easiest split
- VRAM usage peaked at 42.73 GB (well under 96 GB cap); there is room to increase model size or batch size
- Training timed out at epoch 14, best at epoch 13 — models likely under-trained; LR schedule / convergence speed is important

## Potential Next Research Directions

Once round 4 results are in, priority areas to explore:
- **Physics-informed losses**: Divergence-free velocity constraint (∇·u = 0 for incompressible); pressure-velocity coupling (Poisson-style penalty)
- **Model scaling**: 42 GB VRAM used of 96 GB available — increase n_hidden (256?), n_layers (6-8), or slice_num (128?) to improve capacity within timeout (being tested by nezuko, PR #1137)
- **Faster convergence**: Warm restarts, cyclical LR, gradient clipping to reach better minima within the epoch budget
- **Ensemble / multi-head outputs**: Separate prediction heads per domain (raceCar single, raceCar tandem, cruise)
- **Fourier/spectral features**: Augment node features with positional encodings based on geometry wavelengths
- **Higher resolution surface loss**: Weight loss proportional to local curvature (leading/trailing edge gets more weight)
- **Multi-task learning**: Joint Ux, Uy, p with task-specific loss weights tuned per domain
- **Graph neural message passing**: Augment transformer with local neighborhood message passing for better surface gradient propagation
- **Self-supervised pretraining**: Predict masked node features to prime geometry understanding
