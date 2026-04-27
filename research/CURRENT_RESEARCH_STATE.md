# SENPAI Research State

- 2026-04-27 22:30 — round 1 of `icml-appendix-willow-pai2d-r2` assigned
- Baseline: unmodified `train.py` (no concrete metric in `BASELINE.md` yet —
  first PR to beat baseline populates it).
- Primary metric: `val_avg/mae_surf_p` (equal-weight surface pressure MAE
  across 4 val tracks). Paper-facing: `test_avg/mae_surf_p`.
- W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2`.
- Wall-clock cap per run: `SENPAI_TIMEOUT_MINUTES=30`.

## Round 1 — eight orthogonal axes (one per student)

Goal: clean attribution of which standard Transolver lever helps. Round 2
will stack the strongest of {arch, optim, loss} winners.

| PR  | Student   | Axis                       | Change                                                                           |
|-----|-----------|----------------------------|----------------------------------------------------------------------------------|
| 311 | alphonse  | model width                | `n_hidden` 128 → 192                                                             |
| 325 | askeladd  | model depth                | `n_layers` 5 → 8                                                                 |
| 326 | edward    | FFN ratio                  | `mlp_ratio` 2 → 4                                                                |
| 328 | fern      | physics-token count        | `slice_num` 64 → 128                                                             |
| 330 | frieren   | loss formulation           | MSE → Huber (Smooth L1, β=1) in normalized space, both train + val "loss"        |
| 332 | nezuko    | surface-vs-volume weight   | `surf_weight` 10 → 25 (single + optional 15/25/40 sweep via `--wandb_group`)     |
| 335 | tanjiro   | LR schedule                | 5-epoch linear warmup + cosine, peak `lr` 5e-4 → 1e-3                            |
| 337 | thorfinn  | batch + LR scaling         | `batch_size` 4 → 8, `lr` 5e-4 → 7e-4 (sqrt scaling)                              |

All PRs are draft, labelled `status:wip` + `student:<name>` +
`icml-appendix-willow-pai2d-r2`, and target the advisor branch as base.

## What I expect to see

- **Architecture axes (alphonse / askeladd / edward / fern):** at least
  one of these should land a 2-5% gain. Width and FFN ratio are the most
  likely; depth often needs LR retuning; slice_num pays off most on
  geometry-OOD tracks. Watch per-split deltas, not just the average.
- **Loss / weighting axes (frieren / nezuko):** direct push on the metric.
  Huber is the more principled change; surf_weight is the most direct.
  Both should improve `val_avg/mae_surf_p`; nezuko at the cost of volume.
- **Optimizer axes (tanjiro / thorfinn):** these compound with everything
  else. If they win cleanly, round 2 always carries them.

## Round 2 candidate stacks (post-results)

- **Compound winner.** Pick the strongest arch axis × strongest optimizer
  axis × strongest loss axis and run the joint configuration. With four
  candidate buckets and 8 GPUs we can also explore 2-axis and 3-axis
  intersections in parallel.
- **Target-space reformulation.** The sample y-std spans 10× even within
  a domain. Try `asinh` or per-sample-std normalization on the pressure
  channel; this pairs especially well with Huber if Huber wins.
- **Per-channel loss weighting.** Up-weight the `p` channel (the metric
  cares only about pressure). Pair with a winning surf_weight.
- **Bigger models.** If width / depth / mlp / slice each give clean
  gains, push capacity further with EMA + gradient clipping.
- **Augmentation.** Geometry-preserving x-flip (mirror y-coord and
  corresponding flow components) for the ground-effect raceCar domain.
- **Curriculum.** Sort batches by per-sample y std, warmup the model on
  low-magnitude samples first.
- **Domain conditioning / explicit single-vs-tandem gate** using features
  18-23 (foil-2 AoA / NACA / gap / stagger).

## Constraints respected

- `SENPAI_MAX_EPOCHS=999` and `SENPAI_TIMEOUT_MINUTES=30` not overridden.
  Capacity-scaling axes (width-192, depth-8, mlp-4) may finish only
  ~30-40 epochs; best-checkpoint selection survives.
- VRAM 96 GB so BS=8 and wider models are safe even on 242K-node meshes.
- Loaders are read-only — all changes live in `train.py` only.
- One hypothesis per PR. No bundled changes.
