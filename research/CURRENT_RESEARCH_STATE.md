# SENPAI Research State

- 2026-04-27 18:50
- No recent research directives from the human researcher team
- **Current research focus:** Round 1 — implementing and verifying the full winning recipe from sibling track kagent_v_students, then pushing beyond it with new ideas
- **Strategy:** The 30-min wall-clock budget is the binding constraint (compute-constrained regime). Experiments that reduce per-step compute (smaller model, AMP, fewer layers) win because they fit more epochs. The winning recipe from kagent_v_students reduces val_avg/mae_surf_p from ~103 to ~48 through compounding improvements.

## Round 1 Hypotheses (assigned 2026-04-27)

1. **alphonse**: Full winning recipe (L1+AMP+Fourier+SwiGLU+nl3+sn16+nh1) — baseline establishment + sweep
2. **askeladd**: Batch size 1 + high grad accumulation (extreme sample diversity)
3. **edward**: Fourier PE with multi-scale positional encoding (m=160 + m=320)
4. **fern**: Channel-decoupled surface decoder (separate p head vs Ux/Uy)
5. **frieren**: EMA of model weights for more stable checkpoint selection
6. **nezuko**: Physics-informed Re-conditional normalization
7. **tanjiro**: Stochastic depth (DropPath) regularization on the winning recipe
8. **thorfinn**: Surface-only attention: dense slice attention on surface nodes

## Current Best
- val_avg/mae_surf_p: ~103 (stock baseline, unverified)
- Reference best from kagent_v_students: val ~48.1 / test ~40.9

## Potential Next Research Directions

### Architecture
- Deeper Fourier PE (add more frequencies, learn sigma)
- Cross-attention between surface and volume nodes
- Per-domain adaptation (raceCar single vs tandem vs cruise)
- Larger hidden dim with smaller n_layers/n_head (more width, less depth)

### Training
- Longer training budget (if SENPAI_TIMEOUT_MINUTES can be extended)
- Cyclical learning rate with warm restarts
- Gradient clipping for stability at high surf_weight
- OneCycleLR scheduler

### Loss / Objective
- Per-channel loss weighting (pressure matters most for ranking)
- Huber loss with different delta values
- Boundary-aware loss: extra weight on near-surface nodes vs far-field

### Physics-informed
- Reynolds number as an explicit conditioning signal (FiLM layers)
- Lift/drag coefficient supervision as an auxiliary loss
- Velocity divergence regularization (physical constraint: div(U)=0 in incompressible flow)

### Data
- Test-time augmentation (horizontal flip with Uy sign flip)
- Domain-specific normalization (separate stats per domain)
