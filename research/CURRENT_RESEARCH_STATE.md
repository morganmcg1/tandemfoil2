# SENPAI Research State

- **Last updated**: 2026-04-28 UTC
- **Branch**: `icml-appendix-charlie-pai2e-r3`
- **Most recent direction from human researcher team**: none on file. Default contract from `target/program.md` stands — drive `val_avg/mae_surf_p` (and the matching `test_avg/mae_surf_p` from best val checkpoint) down on the equal-weight 4-split mean.

## Current best (post round-2 review)

- **`val_avg/mae_surf_p` = 94.387** (from PR #889, fern, Cosine T_max=15 + 1-epoch warmup; merged 2026-04-28).
- **`test_avg/mae_surf_p` = 92.232** (3-split mean, cruise excluded due to known 1-sample GT bug).

Config: Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2), AdamW(lr=5e-4, wd=1e-4), CosineAnnealingLR(T_max=15 + 1-epoch warmup), batch_size=4, surf_weight=10, epochs=50.

### Key insights from prior rounds

1. **MAE/L1 loss >> MSE** (PR #835): Biggest win. Large high-Re samples were over-penalized by MSE.
2. **Cosine schedule fix** (PR #889): T_max=50 was miscalibrated vs ~14 realized epochs. Fixing T_max=15 + 1-epoch warmup improved val_avg from 104.058 → 94.387 (9.2% gain). All 4 val splits improved 5–14%.
3. **Larger model (192/6/6) failed** (PR #887): bf16 autocast worked (no OOM), but per-epoch wall time increased 1.36× (125s → 170s), leaving only 11 epochs vs 14. Budget bottleneck is severe — bigger models hurt in the 30-min regime.
4. **Surface pressure is the bottleneck**: `val_single_in_dist` at 118.130 is hardest despite being in-distribution; camber-holdout splits (geom_camber_*) test unseen front-foil geometry.

## Round 4 in flight (8 of 8 GPUs active)

| PR  | Student   | Hypothesis                                                           | Status       |
|-----|-----------|----------------------------------------------------------------------|--------------|
| 916 | alphonse  | surf_weight=25 on MAE baseline (vs current 10)                      | WIP          |
| 919 | fern      | Per-sample Re-aware loss normalization (gradient equity)            | WIP          |
| 905 | askeladd  | Signed-log pressure target normalization for heavy-tail Re          | WIP          |
| 903 | edward    | slice_num=128 + bf16 AMP + NaN-safe eval on MAE baseline            | WIP          |
| 895 | thorfinn  | EMA of model weights (decay=0.999) for evaluation                   | WIP          |
| 892 | tanjiro   | OneCycleLR (max_lr=1e-3) for short-budget super-convergence         | WIP          |
| 890 | frieren   | Random Fourier features on (x,z) for camber generalization          | WIP          |
| 925 | nezuko    | FiLM conditioning on log(Re) for Re-aware representations           | WIP (new)    |

Closed this review cycle:
- PR #891 (nezuko, Smooth-L1 δ=1.0): CLOSED — val_avg 109.065, +15.5% vs baseline. L2-like quadratic regime dominates in normalized space; re-introduces MSE's high-Re tail pathology.

Round 4 spans 3 core themes:
1. **Loss function engineering**: surf_weight tuning (#916), per-sample Re normalization (#919), signed-log pressure transform (#905).
2. **LR scheduling and optimization**: OneCycleLR (#892), EMA weights (#895).
3. **Architecture/conditioning**: slice_num=128 with AMP (#903), Fourier coordinate features (#890), FiLM log(Re) conditioning (#925).

## Themes for round 5+

Hold these for after round-4 results:

1. **Stack winners.** If schedule + loss + Fourier features each give marginal gains, combine them — they're orthogonal.
2. **Camber-holdout-targeted methods.** If RFF (#890) helps on `val_geom_camber_rc`:
   - Geometry tokens / cross-attention to a compact camber descriptor.
   - Reflection symmetry augmentation (chord-axis flip → AoA negation).
   - Two-stage train: pretrain on rc-single, finetune on tandem.
3. **Adaptive surf_weight curriculum**: start at surf_weight=10, ramp to 25-50 as training converges.
4. **Gradient clipping**: with MAE loss, occasional extreme Re samples may still cause large gradient norms.
5. **Physics-informed losses**: divergence-free penalty on (Ux, Uy) for incompressible flow; pressure-Poisson residual.
6. **Multi-step LR warmup**: longer warmup (3-5 epochs) to stabilize training before aggressive cosine decay.
7. **Surface-only head**: dedicate separate capacity to surface node predictions vs. volume.
8. **Re-conditioned normalization**: layer norm or group norm conditioned on log(Re) feature.

## Open risks / things to watch

- **Test metric discipline**: `test_avg/mae_surf_p` from **best val checkpoint**, not last epoch.
- **The `data/scoring.py` NaN-poisoning bug** is worked around in `train.py` (merged via #835). A dedicated upstream PR to fix `data/scoring.py` itself would be cleaner — track as "code health."
- **Camber holdouts dominate the error**: `val_geom_camber_rc=100.284` is above the 94.387 average — methods that target generalization to unseen front-foil geometry (e.g., RFF in #890) have the largest potential lever.
- **30-min budget is binding**: per-epoch throughput constrains all experiments. Bf16 AMP (#903) is the main avenue to get more epochs; test carefully for quality regressions.
