# SENPAI Research State — icml-appendix-charlie-pai2c-r2

- **Date**: 2026-04-27 (last update: 20:35 UTC)
- **Most recent human researcher direction**: None on this branch yet. Issue #257 (cross-track GitHub label-index regression) is informational; humans aware via 36+ comments, system gradually self-healing.
- **Empirical baseline**: `val_avg/mae_surf_p = 102.71` (PR #213 merged, commit efa6e3c). 21% improvement over the previous baseline (130.06, PR #216) — physical-space L1 surface loss with all 4 val splits improving 18–23%.

## Current research focus

Round 1 has produced **two merged winners** so far:
- **PR #216** (tanjiro, wider-shallower-arch): `n_hidden=192, n_layers=4, n_head=6` — set the floor at 130.06.
- **PR #213** (nezuko, surface-pressure-l1-loss): physical-space L1 on surface term — drove the floor to 102.71. Single largest improvement of Round 1.

The L1-vs-MSE alignment was the dominant misalignment in the training recipe. Future PRs branch from this combined state (wider-shallower architecture + L1 surface loss).

### Round 1 status by student

| Student | PR | Hypothesis | Status |
|---------|----|------------|--------|
| alphonse | #190 | surf-weight 10→30 | wip (training) |
| askeladd | #202 | asinh pressure target | wip (training) |
| edward | #206 | EMA-of-weights | wip (training) |
| fern | #208 | LR warmup + cosine floor + peak 1e-3 | wip (training) |
| frieren | #210 | LayerScale + DropPath | wip (training) |
| nezuko | #213 → MERGED → #262 | (#213 L1 surface loss merged) → pressure-weighted-l1 | wip (new) |
| tanjiro | #216 → MERGED → #258 | (#216 wider-shallower merged) → slice-num-doubled | wip (training in background) |
| thorfinn | #218 | Fourier-encoded position features | wip (training, just picked up at iter 8) |

All 8 GPUs in use. No idle students.

## Themes the round is testing

1. **Loss/metric alignment** — `surface-pressure-l1-loss` (#213) **WIN**: –21% on val_avg by aligning gradient direction with the physical-space L1 metric. `asinh-pressure-target` (#202) tests an orthogonal recipe (target reshaping). Following up with `pressure-weighted-l1` (#262, channel-2 weighted 2× in L1).
2. **Surface-vs-volume balance** — `surf-weight-aggressive` (#190).
3. **Small-batch training noise** — `ema-evaluation` (#206), `lr-warmup-cosine-floor` (#208).
4. **Capacity allocation** — `wider-shallower-arch` (#216) **WIN**: established baseline. Following up with `slice-num-doubled` (#258, slice_num 64→128 on the wider base).
5. **Geometry-interpolation generalization** — `relative-position-features` (#218), `layerscale-stochastic-depth` (#210).

## Per-split observations (after PR #213 merge)

The per-split structure remained consistent: cruise tandem geom-holdout is *easiest* (`mae_surf_p = 80.47`), single-foil in-dist is *hardest* (`125.02`). The L1 transition closed about 30% of the single-foil disadvantage but the gap is still there. Single-foil samples have the largest pressure dynamic range (per program.md: y range up to ±29K vs cruise ±7.6K), so even L1-aligned gradient may be insufficient — the ratio of single-foil to cruise pressure-MAE is now ~1.55x (vs 1.64x at the previous baseline).

This per-split asymmetry strengthens the case for Round 2 candidates: domain-id conditioning, per-Re-bin scaling, or single-foil-specific decoder heads.

## Open issues for the human team

- **`data/scoring.py` NaN bug** — `NaN * 0 → NaN` in `accumulate_batch` (PyTorch semantics) silently NaN-contaminates `test_avg/mae_surf_p` whenever a single test sample has Inf/NaN GT. Affects `test_geom_camber_cruise/000020.pt` (761 Inf pressure values). One-line fix needed in the read-only `data/scoring.py`. Confirmed independently by tanjiro (#216) and nezuko (#213). Already documented in cross-track issue #13. **Until patched, rank PRs on `val_avg/mae_surf_p` only;** test numbers require manual NaN-safe re-eval (nezuko's #213 NaN-safe `test_avg = 91.52`).
- **GitHub label search index lag** — issue #257, system gradually self-healing.

## Potential next research directions (Round 2 candidates)

After Round 1 settles, the strong signal from #213 (L1-on-surface) suggests these directions:

- **Aggressive pressure-only loss**: `mae_surf_p` is the only metric, and #262 will test channel-2 weighting. The next step might be replacing the velocity surface terms entirely with their L2 to keep them mild but not load-bearing — or even a channel-2-only surface loss (with velocity supervised only via volume term).
- **Re-conditioned normalization**: learnable per-Re-bin scaling of targets (FiLM, AdaIN). The per-split observation that single-foil (largest Re range) is hardest strongly motivates this.
- **Single-foil-specific decoder head**: split the final MLP — surface decoder gets surface-only context with extra capacity; volume decoder gets the rest. Could disproportionately help single-foil where pressure dynamics are extreme.
- **Auxiliary physical losses**: divergence-free `(Ux, Uy)`, pressure-Laplacian smoothness, surface-tangent pressure-gradient consistency. Cheap to implement; may help geometry holdouts.
- **Budget-matched cosine schedule**: every #213/#216 result was timed out before LR decay. `T_max ≈ 12–14` would let the schedule actually anneal. (PR #208 partially addresses this with the 1e-5 floor.)
- **Mesh-aware features**: kNN-based neighborhood features (mean/std of nearby x), computed online inside `train.py`.
- **Architecture replacement** (after first plateau): GINO/GeoFNO, Set Transformer / Perceiver IO, GNN with kNN edges, OFormer.

## What we are NOT doing on this branch

Per launch directive, we do not inspect, reuse, summarize, or compare against PRs, branches, commits, or W&B runs from sibling launches or older charlie/willow rounds. Every comparison is internal to this branch.
