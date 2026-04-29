# TandemFoilSet — Dataset analysis (round 1 reference)

Source: `target/program.md`, `target/data/SPLITS.md`, `target/data/loader.py`,
`target/data/scoring.py`. Values cross-checked against `meta.json` /
`stats.json` summaries documented in `program.md`.

## Sample counts

| Split | Count | Avg nodes | Notes |
|---|---|---|---|
| Train (3 domains, balanced) | 1499 | RC single 85K, RC tandem 127K, Cruise tandem 210K | Equal-weighted via `WeightedRandomSampler` |
| `val_single_in_dist` | 100 | ~85K | Random single-foil holdout |
| `val_geom_camber_rc` | 100 | ~129K | Front-foil M=6–8 (held out) |
| `val_geom_camber_cruise` | 100 | ~209K | Front-foil M=2–4 (held out) |
| `val_re_rand` | 100 | mixed tandem | Stratified Re holdout across all tandem domains |
| Each `test_*` | 200 | matching val | Hidden ground truth joined in via `.test_*_gt/` |

## Feature contract

24-channel input `x`:
- 0–1: node position (x, z)
- 2–3: signed arc-length features (`saf`)
- 4–11: distance-based shape descriptor (`dsdf`)
- 12: `is_surface` flag (also surfaced in collate as a separate tensor)
- 13: `log(Re)`
- 14: AoA foil 1 (rad)
- 15–17: NACA foil 1 (camber, position, thickness, normalized to [0,1])
- 18: AoA foil 2 (rad, 0 for single-foil)
- 19–21: NACA foil 2 (0,0,0 for single-foil)
- 22: gap (0 for single-foil)
- 23: stagger (0 for single-foil)

3-channel target `y`: `[Ux, Uy, p]`. Models predict in normalized space and the
trainer denormalizes with `pred * y_std + y_mean` before MAE.

## Magnitudes

Per-sample `y` std varies by ~10× within each split — high-Re samples drive the
extremes. Across splits:

| Source split | Re | y range | Avg per-sample y std | Max per-sample y std |
|---|---|---|---|---|
| `val_single_in_dist` (RC single) | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` (RC tandem P2, M=6–8) | 1.0M–5M | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` (Cruise tandem P2, M=2–4) | 122K–5M | (-7,648, +2,648) | 164 | 506 |

Implication: MSE on raw normalized targets may be dominated by high-Re samples,
motivating Huber / scale-aware losses, and explicit per-channel weighting (the
primary metric is surface pressure only).

## Geometry holdouts

Files 2 (raceCar tandem P2, M=6–8) and 5 (cruise tandem P2, M=2–4) are
fully held out — train sees front-foil cambers on either side of these slabs
and must interpolate. Per-foil camber distributions are non-overlapping by
design; all tandem files share the same 30 rear-foil shapes within each domain.

## Mesh / VRAM constraints

- Variable mesh size 74K–242K nodes (cruise tandem the largest).
- `pad_collate` zero-pads to per-batch max; the `mask` tensor is critical for
  loss + metric correctness on padded positions.
- 96 GB GPU is large relative to model size — the default Transolver is ~0.65M
  params and uses well under 10 GB at bs=4. Capacity scaling is essentially
  free from a memory perspective.

## Implications for round-1 design

- Surface pressure dominates the score, so per-channel and surface-weight
  experiments are first-order levers.
- Heavy-tailed magnitudes argue for Huber / scale-aware variants alongside
  pure-MSE.
- VRAM headroom + 30-min wall clock means we can grow the model meaningfully
  without OOM or timing out (must verify per-experiment).
- Geometry-camber holdouts vs Re-rand are different generalization axes; track
  per-split metrics in reviews — discrepancies are signal.
