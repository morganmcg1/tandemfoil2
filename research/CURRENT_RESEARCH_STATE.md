# SENPAI Research State — `icml-appendix-willow-r3`

- **Updated:** 2026-04-27 (round 1 — opening matrix dispatched)
- **Advisor branch:** `icml-appendix-willow-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-r3`

---

## Most recent human-team direction
None received. Free hand to set the agenda.

## Current research focus and themes

This is a **fresh greenfield track** — no prior runs in the W&B project, no
merged PRs on the advisor branch. The opening `train.py` is the vanilla
Transolver baseline (MSE, no AMP, no Fourier PE, slice_num=64, n_layers=5,
surf_weight=10).

A sibling track (`kagent_v_students`) on the same dataset/model contract
already established that the following 8 components compound monotonically to
val ≈ 49.4 / test ≈ 42.5 from a similar starting point:

  L1 → surf_weight=1 → AMP+bf16 → grad_accum=4 → Fourier PE (σ=0.7, m=160)
  → SwiGLU FFN → slice_num=8 (down from 64) → n_layers=3 (down from 5).

The compute-reduction theme (AMP, slice_num↓, n_layers↓) was the dominant
signal — by lowering per-epoch cost, more epochs fit in the 30-min budget,
and the loss curve was still descending at the terminal epoch in many
configs. The **30-min wall-clock cap is the binding constraint, not
parameters**.

**Round 1 strategy:** dispatch 8 orthogonal hypotheses, each adding one of
those proven wins (or a closely related variation) on top of the vanilla
baseline. Round 2 will start merging winners and stacking. Each PR runs both
a vanilla anchor (1 seed) and the variant (1–2 seeds) inside the same 30-min
budget, so even if the variant fails we still get a clean baseline number for
this track's W&B project.

### Round 1 hypothesis matrix (one per student)

| Student | Hypothesis | Expected val gain |
|---------|-----------|-------------------|
| willow3-alphonse  | Vanilla anchor + L1 loss reformulation (closes MSE↔MAE gap) | −5 to −15% |
| willow3-askeladd  | AMP (bf16) + grad_accum=4 — throughput unlock for more epochs | −10 to −20% |
| willow3-edward    | Fourier PE on (x,z) with σ ∈ {0.7, 1.0}, m=160 — high-freq features | −3 to −10% |
| willow3-fern      | SwiGLU feedforward in TransolverBlock (replaces GELU MLP) | −5 to −13% |
| willow3-frieren   | surf_weight sweep ∈ {0.5, 1, 2, 5} — current `10` likely too high | −2 to −6% |
| willow3-nezuko    | slice_num sweep ∈ {16, 32, 48, 64} — lower may regularize + speed up | −3 to −8% |
| willow3-tanjiro   | n_layers depth sweep ∈ {3, 4, 5, 6} — 30 min favours shallower | −5 to −12% |
| willow3-thorfinn  | Cosine-with-warmup schedule + lr ∈ {5e-4, 1e-3} sweep | −1 to −5% |

All eight runs will additionally produce a **clean vanilla anchor** number on
this track's W&B project so that the BASELINE.md gets a real validated figure
ASAP.

## Potential next research directions (round 2+)

After winners merge, the obvious next moves stack the proven compounds:

- **Compound stacking:** L1 + surf_weight=1 + AMP + Fourier-PE + SwiGLU is the
  high-confidence first compound (5 wins co-merged). Test this directly in
  round 2.
- **Compute-reduction extension:** once AMP is in, sweep `slice_num` and
  `n_layers` jointly (lessons from the prior track: floors not yet found).
- **Within-recipe knobs:** `n_hidden ∈ {64, 96, 128, 160}`, `n_head ∈ {1, 2, 4, 8}`
  (prior monotonic trend nh=1 < nh=2 < nh=4 < nh=8 worth re-testing here),
  `mlp_ratio ∈ {1, 2, 4}`.
- **Long-standing unaddressed (round 3+):**
  - Physics-informed auxiliary losses (Kutta condition; divergence-free for Ux/Uy).
  - Per-channel surf weight (decouple p, Ux, Uy weighting).
  - EMA of weights for OOD splits (raceCar/cruise camber).
  - Geometric / horizontal-flip augmentation for the symmetric airfoil cases.
  - Cross-attention surface decoder (pulls surface predictions out of slice tokens).
  - Drop-path / stochastic depth (regularization at deeper recipes).
  - Mesh-aware / radius-graph local attention to complement Physics-Attention.
  - Better positional encoding for the boundary layer (signed-distance-aware Fourier).

## PR pipeline status

- Review-ready: 0
- WIP: 0 (8 to be dispatched in round 1)
- Idle students: 8 (all to be assigned)
