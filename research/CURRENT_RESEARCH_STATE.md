# SENPAI Research State
- 2026-04-29 (round: icml-appendix-charlie-pai2f-r3)
- Most recent research direction from human researcher team: None (no GitHub Issues)

## Current Research Focus

We are in **charlie-pai2f round 3**, building on the compound baseline from charlie-pai2e-r5:
- **Baseline target**: `val_avg/mae_surf_p = 47.7385` (Lion+L1+EMA(0.995)+bf16, n_layers=1, surf_weight=28, cosine T_max=15, clip_grad=1.0)
- **Primary metric**: `val_avg/mae_surf_p` (lower is better) — equal-weight mean surface-pressure MAE across 4 val splits

## Active Experiments (8/8 students running)

| PR  | Student   | Hypothesis                                           |
|-----|-----------|------------------------------------------------------|
| #1093 | alphonse  | Compound baseline anchor re-implementation (Lion+L1+EMA+bf16, n_layers=1, sw=28) |
| #1103 | askeladd  | `slice_num` sweep {32, 64, 128}                      |
| #1104 | edward    | FiLM global conditioning (Re/AoA/NACA via scale+shift on hidden states) |
| #1105 | fern      | Per-channel pressure weight W_p sweep {2, 3, 5}      |
| #1106 | frieren   | Fourier positional encoding on (x,z)                 |
| #1107 | nezuko    | EMA decay sweep {0.99, 0.995, 0.999}                 |
| #1108 | tanjiro   | n_hidden width sweep {128, 192, 256}                 |
| #1109 | thorfinn  | Boundary-layer proxy feature: log(Re×|saf|+ε)        |

## Current Research Themes

1. **Architecture capacity** — slice_num, n_hidden width with n_layers=1 constraint
2. **Physics conditioning** — FiLM global conditioning, boundary-layer feature engineering
3. **Training dynamics** — EMA decay sweep, per-channel loss weighting
4. **Positional representation** — Fourier positional encodings for boundary-layer geometry

## Potential Next Research Directions

Once current round results are in:

1. **Multi-layer Transolver** — after compound baseline is stable, test n_layers=2 with n_hidden=128 vs n_layers=1 with n_hidden=256 (width vs depth tradeoff)
2. **Adaptive surf_weight** — dynamic weighting that increases surf_weight as training progresses, focusing later epochs on surface pressure accuracy
3. **Domain-aware attention** — explicit domain label (single/raceCar-tandem/cruise-tandem) injected as conditioning
4. **Graph-based local attention** — k-NN local attention over mesh nodes to capture boundary-layer gradients
5. **Separate decoders per output channel** — specialized decoders for (Ux, Uy, p) since pressure and velocity have different physics
6. **Curriculum learning** — order samples by difficulty (Re magnitude or domain complexity) to improve generalization
7. **Physics-informed loss terms** — soft continuity equation residual as auxiliary loss
8. **Stochastic depth / DropPath** — regularization for n_layers>1 configurations
9. **Mixed precision training improvements** — fp16 vs bf16 comparison, gradient scaling
10. **Test-time ensembling** — average predictions from EMA checkpoints at different epochs
