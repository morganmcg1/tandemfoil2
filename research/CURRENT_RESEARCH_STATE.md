# SENPAI Research State
- 2026-04-29 (branch: icml-appendix-charlie-pai2f-r4)
- No human researcher team directives received yet.

## Current Research Focus

**Target:** TandemFoilSet CFD surrogate — predict (Ux, Uy, p) at every mesh node.
**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 val splits (lower is better).
**Model:** Transolver with physics-aware attention over irregular meshes.
**Status:** Round 4, first invocation. 8 experiments in flight. No results reviewed yet (no BASELINE.md established).

## Active Experiments (Round 4)

| PR   | Student   | Hypothesis |
|------|-----------|-----------|
| #1117 | thorfinn  | Re-conditioned output scale head for magnitude adaptation |
| #1116 | tanjiro   | OneCycleLR (max_lr=2e-3, pct_start=0.3) vs cosine annealing |
| #1115 | nezuko    | Oversample surface-node errors by inverse surface fraction |
| #1114 | frieren   | Curriculum surf_weight ramp 1→20 over training |
| #1113 | fern      | Learnable 3-way domain embedding for multi-domain routing |
| #1112 | edward    | Attention dropout=0.1 for OOD slice regularization |
| #1111 | askeladd  | Layer-wise LR decay for geometry-stable representations |
| #1110 | alphonse  | Log-modulus transform on pressure channel loss |

## Research Themes Being Explored

1. **Loss formulation**: Curriculum surface weight ramp; log-modulus pressure transform; surface-error oversampling
2. **LR scheduling**: OneCycleLR vs cosine annealing; layer-wise LR decay
3. **Architecture additions**: Re-conditioned output scale head; learnable domain embedding
4. **Regularization**: Attention dropout for OOD robustness

## Potential Next Research Directions

Once round 4 results are in, priority areas to explore:
- **Physics-informed losses**: Divergence-free velocity constraint (∇·u = 0 for incompressible); pressure-velocity coupling (Poisson-style penalty)
- **Ensemble / multi-head outputs**: Separate prediction heads per domain (raceCar single, raceCar tandem, cruise)
- **Fourier/spectral features**: Augment node features with positional encodings based on geometry wavelengths
- **Adaptive mesh refinement proxy**: Higher attention weight near leading/trailing edge and surface
- **Self-supervised pretraining**: Predict masked node features to prime geometry understanding
- **Graph neural message passing**: Augment transformer with local neighborhood message passing for better surface gradient propagation
- **Higher resolution surface loss**: Weight loss proportional to local curvature (leading/trailing edge gets more weight)
- **Multi-task learning**: Joint Ux, Uy, p with task-specific loss weights tuned per domain
