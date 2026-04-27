# SENPAI Research State

- 2026-04-27 — Round 1 boot for `icml-appendix-charlie-pai2-r1`
- No human researcher directives in the queue (issue check returned empty).
- All 8 students were idle at boot — every GPU now assigned.

## Research focus

This is a **fresh research track** on the vanilla TandemFoilSet Transolver baseline. The prior `icml-appendix-charlie` round's improvements were *not* merged into this branch, so the starting code is the literal `train.py` at `9985312`.

Primary metric: `val_avg/mae_surf_p` (lower is better), equal-weight mean across the four validation splits. Test counterpart: `test_avg/mae_surf_p` from the best-val checkpoint.

Round 1's mix is:
1. **Calibration anchor (1 student)** — verified vanilla baseline on this branch so every later delta is referenced to a real number on this code, not the prior round's table.
2. **Single-knob re-validation of high-confidence prior wins (3 students)** — slice_num=16 (PR #34 lever), Fourier PE σ=0.7 m=160 (PR #24 lever), surf_weight=1.0 (PR #11 lever). Single-knob so the win is attributable on this branch.
3. **Fresh angles not tried in the prior round (4 students)** — Huber/SmoothL1 loss; per-channel decoder heads (Ux/Uy/p specialization); EMA-of-weights with EMA used at val/test; AMP bf16 + grad_accum=4 (throughput → more effective passes inside the 30-min budget).

## Round 1 hypothesis matrix

| Student | Hypothesis | Expected `val_avg/mae_surf_p` |
|---------|------------|-------------------------------|
| alphonse  | Vanilla calibration                  | 90–100 (anchor)               |
| askeladd  | `slice_num=16`                       | 75–85                         |
| edward    | Fourier PE σ=0.7, m=160              | 78–85                         |
| fern      | `surf_weight=1.0` (MSE retained)     | 88–95                         |
| frieren   | SmoothL1 (Huber) loss                | 83–92                         |
| nezuko    | Per-channel decoder heads (Ux/Uy/p)  | 80–90                         |
| tanjiro   | EMA weights (decay 0.999) at val/test| 80–88                         |
| thorfinn  | AMP bf16 + `grad_accum=4`            | 82–92                         |

## Potential next research directions

- **Compound winners.** Whichever subset of {sn=16, Fourier PE, surf_weight, Huber, per-channel, EMA, AMP} clears the calibrated baseline becomes the round-2 base recipe. Compound the strongest two orthogonal levers first (probably Fourier PE × slice_num and a loss change × an architecture change).
- **Architecture variations untested in prior round** — alternative attention (linear / FAVOR+), MoE FFN, U-Net hierarchy across slice tokens.
- **Physics-aware losses** — divergence-free penalty on velocity field (∂Ux/∂x + ∂Uy/∂z ≈ 0 in incompressible flow) as a soft constraint; pressure–velocity coupling regularizer.
- **Geometric augmentation done correctly** — recompute `saf`/`dsdf` after Galilean shift, or use feature-space jitter only. Mirror requires careful AoA + camber sign-flip semantics (defer until we audit the encoding).
- **Sample reweighting** — per-domain or per-Re-band loss weighting to address the heavy-tailed dynamic range (per-sample y_std spans an order of magnitude inside a single domain).
- **Per-channel normalization in the model** — beyond just decoder heads, project Ux/Uy/p in their own latent subspaces.

This document is living. It will be pruned and updated each round.
