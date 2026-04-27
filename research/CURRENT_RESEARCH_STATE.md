# SENPAI Research State — icml-appendix-charlie-pai2c-r2

- **Date**: 2026-04-27 (last update: 19:55 UTC)
- **Most recent human researcher direction**: None on this branch yet (no `for:advisor` issues; the only open issue from a sibling branch is #13 on the data-bug topic).
- **Empirical baseline**: `val_avg/mae_surf_p = 130.0568` set by PR #216 (wider-shallower-arch, charliepai2c2-tanjiro). Training timed out at epoch 11/50 — still improving; subsequent runs should consider matching `T_max` to realistic wall-clock budget.

## Current research focus

Round 1 is in flight. The first merged PR (#216) established the floor at `val_avg/mae_surf_p = 130.06` with the wider-shallower architecture (`n_hidden=192, n_layers=4, n_head=6`, ~1.18M params). Every subsequent PR is judged against this floor — even small improvements should be merged because they compound across rounds.

### Round 1 status by student

| Student | PR | Hypothesis | Status |
|---------|----|------------|--------|
| alphonse | #190 | surf-weight 10→30 | wip (training) |
| askeladd | #202 | asinh pressure target | wip (idle, label-index lag) |
| edward | #206 | EMA-of-weights | wip (training) |
| fern | #208 | LR warmup + cosine floor + peak 1e-3 | wip (training) |
| frieren | #210 | LayerScale + DropPath | wip (training) |
| nezuko | #213 | physical-space L1 surface loss | wip (training) |
| tanjiro | #216 → MERGED → #258 | (#216 wider-shallower-arch merged) → slice-num-doubled | wip (new) |
| thorfinn | #218 | Fourier-encoded position features | wip (idle, label-index lag) |

## Themes the round is meant to test

1. **Loss/metric alignment** — training MSE in normalized space ranks differently from
   physical-space L1 surface MAE. Tested by `surface-pressure-l1-loss` (#213) and
   `asinh-pressure-target` (#202).
2. **Surface-vs-volume balance** — does pushing `surf_weight` up move the right metric? `surf-weight-aggressive` (#190).
3. **Small-batch training noise** — batch_size=4 means high gradient variance; `ema-evaluation` (#206) and `lr-warmup-cosine-floor` (#208).
4. **Capacity allocation** — depth/width/slice trade-off. `wider-shallower-arch` (#216 merged) and now `slice-num-doubled` (#258, building on the new base).
5. **Geometry-interpolation generalization** — `val_geom_camber_*` punishes overfitting. `relative-position-features` (#218) and `layerscale-stochastic-depth` (#210).

## Open issues for the human team

- **`data/scoring.py` NaN bug** — `inf * 0 → NaN` in `accumulate_batch` silently NaN-contaminates `test_avg/mae_surf_p` whenever a single test sample has Inf GT. Affects `test_geom_camber_cruise/000020.pt` (761 Inf pressure values). One-line fix needed in the read-only `data/scoring.py`. Already documented for sibling branches in issue #13 and elsewhere; humans aware. **Until patched, rank PRs on `val_avg/mae_surf_p` only.**
- **GitHub label search index lag** — newly-created advisor-branch and student labels can take 30+ min to enter the search index, causing student pods to report "no work" and idle. Already documented in issue #257. System self-heals as the index settles.

## Per-split observations from the first baseline (PR #216)

The cruise tandem geometry-holdout split is the *easiest* (`mae_surf_p = 98.28`); single-foil in-distribution is *hardest* (`mae_surf_p = 161.15`). This is counterintuitive — the in-distribution sanity-check should be easiest. Hypothesized causes:
- Single-foil samples have larger pressure dynamic range (per program.md: y range up to ±29K vs cruise ±7.6K).
- Balanced sampler equally weights the three domains, so single-foil gets ~33% of training samples but is the highest-variance domain.

This per-split structure should inform Round 2 hypotheses — domain-id conditioning, per-Re-bin scaling, or single-foil-specific decoder heads may pay off. Also, the single-foil domain dominates pressure-MAE error magnitude in the val_avg, so any technique that disproportionately helps single-foil predictions should move the primary metric a lot.

## Potential next research directions (Round 2 candidates)

These are held until Round 1 results reveal whether to escalate or pivot.

- **Architecture replacement** (after first plateau): GINO/GeoFNO-style Fourier neural operator on irregular meshes; Set Transformer / Perceiver IO; GNN with kNN edges; OFormer with cross-attention.
- **Surface-pressure-specific decoder head**: split the final MLP into a surface decoder (surface-only context) and a volume decoder.
- **Re-conditioned normalization**: learnable per-Re-bin scaling of targets (FiLM, AdaIN).
- **Auxiliary physical losses**: divergence-free `(Ux, Uy)`, pressure-Laplacian smoothness, surface-tangent pressure-gradient consistency.
- **Mesh-aware features**: kNN-based neighborhood features (mean/std of nearby x), computed online inside `train.py`.
- **Asymmetric checkpoint selection**: pick best-checkpoint based on geometry-holdout splits alone — biases optimization toward the generalization metrics that the paper cares about.
- **Mixed-precision throughput** (bf16): frees VRAM for larger batches or longer slice attention; may also help fit more epochs in the wall-clock budget.
- **Budget-matched cosine schedule**: `T_max` = realistic epoch count, not 50. Should help every PR that's currently leaving LR on the table.

## What we are NOT doing on this branch

Per launch directive, we do not inspect, reuse, summarize, or compare against PRs, branches, commits, or W&B runs from sibling launches or older charlie/willow rounds. Every comparison is internal to this branch.
