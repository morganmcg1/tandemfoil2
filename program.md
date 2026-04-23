<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet

Research target for CFD surrogate modelling on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Given tandem-airfoil geometry and flow conditions, predict the full velocity `(Ux, Uy)` and pressure `p` field at every mesh node.

The baseline is a [Transolver](https://arxiv.org/abs/2402.02366) with physics-aware attention over irregular meshes. Beat it.

## Codebase

- `train.py` — trainer. Transolver model, training loop, validation, end-of-run test evaluation. **Primary editable entrypoint.**
- `data/loader.py` — `SplitDataset`, `TestDataset`, `load_data`, `load_test_data`, `pad_collate`. **Read-only during normal experiment PRs.**
- `data/scoring.py` — MAE accumulation shared by val and test. **Read-only.**
- `data/prepare_splits.py` — materializes per-sample `.pt` files on the PVC from `data/split_manifest.json`. Organizer script, run once. **Read-only.**
- `data/generate_manifest.py`, `data/split_manifest.json`, `data/SPLITS.md` — split design and manifest. **Read-only.**
- `instructions/prompt-advisor.md`, `instructions/prompt-student.md` — senpai role prompts.

## Data

Pre-processed samples live on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`:

```
splits_v2/
├── train/000000.pt ...           {x: [N,24], y: [N,3], is_surface: [N]}
├── val_single_in_dist/...        random holdout from single-foil (sanity)
├── val_geom_camber_rc/...        unseen front-foil camber (raceCar)
├── val_geom_camber_cruise/...    unseen front-foil camber (cruise)
├── val_re_rand/...               stratified Re holdout across tandem domains
├── test_single_in_dist/...       {x, is_surface}  — hidden y in .test_*_gt/
├── test_geom_camber_rc/...
├── test_geom_camber_cruise/...
├── test_re_rand/...
├── .test_*_gt/                   {y, is_surface}  — hidden ground truth
├── stats.json                    x_mean, x_std, y_mean, y_std
└── meta.json                     split counts, domain groups
```

### Input features (x, 24 dims)

| Dims | Feature |
|------|---------|
| 0-1   | Node position (x, z) |
| 2-3   | Signed arc-length (`saf`) |
| 4-11  | Distance-based shape descriptor (`dsdf`) |
| 12    | Is-surface flag (0/1) |
| 13    | `log(Re)` |
| 14    | AoA foil 1 (radians) |
| 15-17 | NACA foil 1 (camber, position, thickness) |
| 18    | AoA foil 2 (radians, 0 for single-foil) |
| 19-21 | NACA foil 2 (0,0,0 for single-foil) |
| 22    | Gap between foils (0 for single-foil) |
| 23    | Stagger between foils (0 for single-foil) |

### Targets (y, 3 dims)

| Channel | Description |
|---------|-------------|
| 0 | `Ux` — velocity x-component |
| 1 | `Uy` — velocity z-component |
| 2 | `p`  — kinematic pressure (p / ρ, m²/s²) |

## Metrics

All metrics in `data/scoring.py` are computed in the original (denormalized)
target space, in float64, with per-sample skipping for non-finite ground truth.
The same helpers are used for validation during training and for the
end-of-run test evaluation so the numbers are apples-to-apples.

**Primary ranking metric.** Equal-weight mean surface-pressure MAE across the
four splits:

- val: `val_avg/mae_surf_p` — averaged across the four validation splits
- test: `test_avg/mae_surf_p` — averaged across the four test splits

Aggregation is global over all valid surface nodes in the split:

```
mae_surf_p(S) = Σ_{valid surface nodes in S} |p_pred - p_true| / n_surf_nodes
```

**Per-split diagnostics** (logged for every split):

- `mae_surf_{Ux, Uy, p}` — MAE over surface nodes
- `mae_vol_{Ux, Uy, p}`  — MAE over volume (non-surface) nodes
- `loss`, `vol_loss`, `surf_loss` — normalized-space losses used for training

**Checkpoint selection.** Best checkpoint = lowest `val_avg/mae_surf_p`. That
checkpoint is the one evaluated on the test splits at the end of training.

Lower is better. For paper-facing numbers, the decision-driving quantity is
`test_avg/mae_surf_p` on the held-out test splits.

## Constraints

- **VRAM**: GPUs have 96 GB.
- **Simplicity criterion**: all else being equal, simpler is better. A small
  improvement that adds ugly complexity is not worth it.
- **Timeout**: each training run is capped by `SENPAI_TIMEOUT_MINUTES` (wall
  clock) and `SENPAI_MAX_EPOCHS`. Do not override these.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model.
GitHub Issues are used for communication with the human researcher team.
See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
