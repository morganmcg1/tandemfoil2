# SENPAI Research State

- **Date:** 2026-04-27
- **Track:** `icml-appendix-willow-pai2c-r5` (fresh research track, no prior baseline)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-r5`
- **Most recent direction from human researcher team:** None (no open issues on `willow-pai2c-r5` at boot)

## Current research focus

**Round 1 — orthogonal first-round sweep on the default Transolver baseline.**

This is a fresh research track with no prior baseline. The first action is to anchor a baseline (alphonse, default config) and in parallel run seven hypotheses spanning the major orthogonal axes of improvement. The goal is to learn which dimension(s) yield the largest gains so Round 2 can compound the winners.

Primary metric: `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four val splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Test-time companion: `test_avg/mae_surf_p`.

## Round 1 assignments (in flight)

| PR | Student | Hypothesis | Axis |
|----|---------|------------|------|
| #184 | alphonse | Baseline anchor: default Transolver config | reference |
| #221 | askeladd | `n_hidden` 128 → 256 | capacity (width) |
| #223 | edward | `slice_num` 64 → 128 | inductive bias (physics tokens) |
| #224 | fern | 5-epoch linear warmup + cosine | LR schedule |
| #225 | frieren | `surf_weight` 10 → 25 | loss weighting (toward primary metric) |
| #227 | nezuko | Smooth-L1 (Huber) on surface only | loss form (heavy tails) |
| #228 | tanjiro | `batch_size` 4 → 8, `lr` × √2 | batch + LR scaling |
| #229 | thorfinn | `n_layers` 5 → 7 | capacity (depth) |

## Potential next research directions (after Round 1 results)

Ranked by expected value, contingent on what wins in Round 1:

1. **Compound the winners.** If width and slice_num both improve metrics, run a combined `n_hidden=256 + slice_num=128` follow-up. If warmup + Huber both win, combine.
2. **Per-channel loss reweighting.** Targets (Ux, Uy, p) have very different magnitudes. Currently all weighted equally inside the squared error. Channel-specific weights — e.g. `w_p` larger to match the metric — is an obvious next lever.
3. **Boundary-layer-aware sampling.** Surface nodes near the leading edge of high-Re foils carry the largest pressure deviations and are likely the hardest. Surface-distance-weighted or curvature-weighted loss could focus capacity there.
4. **Symmetry data augmentation.** TandemFoilSet has no x-flip data augmentation. Vertical mirroring (around the chord line, with appropriate sign-flips on `Uy`, AoA, and camber) would roughly double the effective training set. Cheap and physics-respecting.
5. **Mixed-precision training (bf16 autocast).** Increases throughput → either more epochs or larger model in the same time budget. Particularly useful if capacity-scaling experiments win.
6. **Unified positional encoding (`unified_pos=True`, ref=8).** The Transolver has a built-in ref-grid encoding currently disabled. Worth a single-experiment test once the simpler levers are pulled.
7. **Surface-only attention head / late-layer surface bias.** If surface MAE is the metric, a small surface-specific output head fed by the last-layer features could dedicate capacity to surface predictions without sacrificing volume features.
8. **Mesh-size-aware learning rate per-batch.** Variable mesh sizes (74K → 242K) mean gradient norms vary by ~3× across batches. Adaptive per-batch LR or gradient clipping could stabilize.

## Plateau watch

Not applicable yet — Round 1 has not reported. Once 5+ consecutive experiments stop improving, escalate via the Plateau Protocol: change strategy tier, revisit first principles, try bolder ideas (new architectures, new loss formulations).
