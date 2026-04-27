# SENPAI Research State — icml-appendix-charlie-pai2c-r2

- **Date**: 2026-04-27
- **Most recent human researcher direction**: None on this branch yet (no `for:advisor` issues; the only open issue, #13, is on a sibling branch).
- **Empirical baseline**: Not yet established. Round 1 in flight.

## Current research focus

Establishing the **empirical floor** for the Transolver baseline on TandemFoilSet. The
single ranking lens for every experiment on this branch is `val_avg/mae_surf_p` (and its
test-time counterpart `test_avg/mae_surf_p`) — surface-pressure MAE in physical units,
equal-weighted across the four val tracks.

The opening round is a **broad sweep across strategy tiers** so we get the most
information per GPU-hour:

| Tier | Round-1 PR (assigned to) |
|------|--------------------------|
| loss | `surf-weight-aggressive` (alphonse), `surface-pressure-l1-loss` (nezuko) |
| optimizer/schedule | `ema-evaluation` (edward), `lr-warmup-cosine-floor` (fern) |
| regularization | `layerscale-stochastic-depth` (frieren) |
| architecture-tweak | `wider-shallower-arch` (tanjiro) |
| feature-engineering | `relative-position-features` (thorfinn) |
| data-representation | `asinh-pressure-target` (askeladd) |

Six distinct tiers in eight slots. The bets are deliberately high-leverage,
single-line-to-single-block changes. No architecture replacements yet — those are Round 2
once we know where the empirical floor sits.

## Themes the round is meant to test

1. **Loss/metric alignment** — training MSE in normalized space ranks differently from
   physical-space L1 surface MAE. Tested by `surface-pressure-l1-loss` and
   `asinh-pressure-target` (re-shapes the target distribution so MSE-in-transform-space
   weighting is closer to L1-in-physical-space).
2. **Surface-vs-volume balance** — does pushing `surf_weight` up move the right metric, or
   does it just trade volume for surface? Tested by `surf-weight-aggressive`.
3. **Small-batch training noise** — batch_size=4 means high gradient variance; do EMA,
   warmup, and gradient clipping move the needle? Tested by `ema-evaluation` and
   `lr-warmup-cosine-floor`.
4. **Capacity allocation** — is the bottleneck depth, width, or slice count? Tested by
   `wider-shallower-arch`.
5. **Geometry-interpolation generalization** — `val_geom_camber_*` requires unseen-camber
   prediction. Does relative position encoding help? Tested by `relative-position-features`.
6. **Held-out generalization regularization** — does standard low-data transformer
   regularization (LayerScale + DropPath) hold across all four splits? Tested by
   `layerscale-stochastic-depth`.

## Potential next research directions (Round 2 candidates, not yet assigned)

These are held back so Round 1 results can inform whether to escalate or pivot.

- **Architecture replacement (after first plateau)**: GINO/GeoFNO-style Fourier neural
  operator on irregular meshes; Set Transformer / Perceiver IO with cross-attention to
  learned latents; GNN with kNN edges for local mesh adjacency; OFormer with
  cross-attention.
- **Surface-pressure-specific decoder head**: split the final MLP into a surface decoder
  (with surface-only context) and a volume decoder. The surface decoder gets a
  differently-conditioned input (e.g., concatenated surface-only attention output).
- **Re-conditioned normalization** — learnable per-Re-bin scaling of targets (FiLM or
  adaptive instance normalization) so the loss treats every Re regime on equal footing.
- **Auxiliary physical losses**: divergence-free constraint on (Ux, Uy), pressure
  Laplacian smoothness, or surface-tangent pressure-gradient consistency. Cheap, may help
  geometry interpolation by injecting a physics prior.
- **Mesh-aware features**: kNN-based neighborhood features (mean/std of nearby x), which
  give the network local-mesh structure without changing data loaders (compute kNN online
  inside `train.py`).
- **Asymmetric checkpoint selection**: pick best-checkpoint based on the geometry splits
  alone (the held-out tracks) rather than the average — biases optimization toward the
  generalization metrics that the paper cares about.
- **Mixed-precision throughput** (bf16) — frees VRAM for larger batches or longer slice
  attention.

## What we are NOT doing on this branch

Per launch directive, we do not inspect, reuse, summarize, or compare against PRs,
branches, commits, or W&B runs from sibling launches or older charlie/willow rounds.
Every comparison is internal to this branch.
