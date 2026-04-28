# SENPAI Research State

- **Last updated**: 2026-04-28 22:50 UTC
- **Branch**: `icml-appendix-charlie-pai2e-r3`
- **Most recent direction from human researcher team**: none on file. Default contract from `target/program.md` stands — drive `val_avg/mae_surf_p` (and the matching `test_avg/mae_surf_p` from best val checkpoint) down on the equal-weight 4-split mean.

## Current best (post round-1 review)

- **`val_avg/mae_surf_p` = 104.058** (from PR #835, nezuko, MAE/L1 loss; merged 2026-04-28).
- **`test_avg/mae_surf_p` = 92.608** (NaN-sample-skipped workaround in train.py, also from #835).

Round 1 (PRs #830-#837) reviewed: 1 merged, 5 closed, 1 sent back. See `EXPERIMENTS_LOG.md` for the full breakdown.

## Round 2 in flight (8 of 8 GPUs active)

| PR  | Student   | Hypothesis                                                       |
|-----|-----------|------------------------------------------------------------------|
| 831 | askeladd  | (revising) surf_weight=50 — original ran on MSE; should re-run on MAE  |
| 832 | edward    | (round 1 carry-over) slice_num 64 → 128                          |
| 887 | alphonse  | bf16 autocast + n_hidden=192/n_layers=6/n_head=6                 |
| 889 | fern      | Fix cosine T_max=15 + 1-epoch linear warmup                      |
| 890 | frieren   | Random Fourier features on (x, z) for camber generalization     |
| 891 | nezuko    | Smooth-L1/Huber loss δ=1.0 (extension of the winning MAE)       |
| 892 | tanjiro   | OneCycleLR (max_lr=1e-3, pct_start=0.3) for short-budget super-conv |
| 895 | thorfinn  | EMA of weights for evaluation (decay=0.999)                     |

Round 2 spans 5 categories of the original "themes for round 2+" plan, prioritized by simplicity-times-likelihood-of-helping:
1. **Loss reformulation** — Smooth-L1 (#891) extending the merged MAE win.
2. **Optimization recipe** — schedule fixes (#889 cosine T_max + warmup, #892 OneCycleLR), EMA (#895).
3. **Capacity scaling** — bf16 + moderately larger model (#887) to retire alphonse's failed round-1 attempt.
4. **Coordinate features** — Random Fourier features (#890) targeting the hardest split (`val_geom_camber_rc=116.84`).
5. **Capacity / attention** — slice_num=128 carry-over (#832).

## Themes for round 3+

Hold these for after round-2 results:

1. **Stack winners.** If schedule + loss + Fourier features each give marginal gains, combine them — they're orthogonal.
2. **Camber-holdout-targeted methods.** If RFF (#890) helps on `val_geom_camber_rc`:
   - Geometry tokens / cross-attention to a compact camber descriptor.
   - Reflection symmetry augmentation (chord-axis flip → AoA negation).
   - Two-stage train: pretrain on rc-single, finetune on tandem.
3. **Physics-informed losses.** Divergence-free penalty on (Ux, Uy) for incompressible flow; pressure-Poisson residual.
4. **Multi-resolution training.** Coarse-to-fine on mesh subsamples to expose more variation in fewer wall-clock minutes.
5. **Better normalization.** Signed-log pressure target (heavy tail at high Re) — pred in log space, scoring inverts.
6. **Architectural alternatives.** Deeper Transolver with stochastic depth, GraphTransformer, or MeshGraphNet baseline as a sanity check.

## Open risks / things to watch

- **Cosine T_max mismatch is the silent baseline tax.** Every PR run on the unmodified baseline LR schedule effectively trains at 0.65× peak LR at the end. Fern's PR #889 will tell us how much headroom this alone unlocks. If it merges, all subsequent experiments should inherit.
- **The `data/scoring.py` NaN-poisoning bug** is now worked around in `train.py` (merged via #835). All future PRs branched from the advisor branch automatically inherit the fix. A dedicated upstream PR to fix `data/scoring.py` itself would be cleaner — track this as a "code health" item even though the metric impact is now zero.
- **Test metric discipline:** `test_avg/mae_surf_p` from **best val checkpoint**, not last epoch. PR #835 follows this; future PRs must too.
- **Camber holdouts dominate the error.** `val_geom_camber_rc=116.84` is +12% above the average — methods that target generalization to unseen front-foil geometry (e.g., RFF in #890) have the largest potential lever.
