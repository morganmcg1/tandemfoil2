<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet

Research target for CFD surrogate modelling on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Given tandem-airfoil geometry and flow conditions, predict the full velocity `(Ux, Uy)` and pressure `p` field at every mesh node.

The baseline is a [Transolver](https://arxiv.org/abs/2402.02366) with physics-aware attention over irregular meshes. Beat it.

## Codebase

- `train.py` — trainer. Transolver model, training loop, validation, end-of-run test evaluation, W&B model-artifact upload. **Primary editable entrypoint.**
- `data/loader.py` — `SplitDataset`, `TestDataset`, `load_data`, `load_test_data`, `pad_collate`. **Read-only during normal experiment PRs.**
- `data/scoring.py` — MAE accumulation shared by val and test. **Read-only.**
- `data/prepare_splits.py` — materializes per-sample `.pt` files on the PVC from `data/split_manifest.json`. Organizer script, run once. **Read-only.**
- `data/generate_manifest.py`, `data/split_manifest.json`, `data/SPLITS.md` — split design and manifest. **Read-only.**
- `instructions/prompt-advisor.md`, `instructions/prompt-student.md` — senpai role prompts.
- `pyproject.toml` — runtime deps. Add any new package you need in the same PR that uses it.

## Data

Pre-processed samples live on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`:

```
splits_v2/
├── train/000000.pt ...           Each: {x: [N,24], y: [N,3], is_surface: [N]}
├── val_single_in_dist/...        Random holdout from single-foil (sanity check)
├── val_geom_camber_rc/...        Unseen front foil camber M=6-8 (raceCar)
├── val_geom_camber_cruise/...    Unseen front foil camber M=2-4 (cruise)
├── val_re_rand/...               Stratified Re holdout across all tandem domains
├── test_*/...                    4 test splits (x + is_surface; y hidden)
├── .test_*_gt/                   Hidden {y, is_surface} joined in by `load_test_data`
├── stats.json                    Normalization stats (x_mean, x_std, y_mean, y_std)
└── meta.json                     Split counts, domain groups
```

### Input features (x, 24 dimensions)

| Dims | Feature |
|------|---------|
| 0-1   | Node position (x, z) |
| 2-3   | Signed arc-length (`saf`) |
| 4-11  | Distance-based shape descriptor (`dsdf`) |
| 12    | Is surface node (0/1) |
| 13    | `log(Re)` |
| 14    | AoA foil 1 (radians) |
| 15-17 | NACA foil 1 (camber, position, thickness) |
| 18    | AoA foil 2 (radians, 0 for single-foil) |
| 19-21 | NACA foil 2 (0,0,0 for single-foil) |
| 22    | Gap between foils (0 for single-foil) |
| 23    | Stagger between foils (0 for single-foil) |

### Targets (y, 3 dimensions)

| Channel | Description |
|---------|-------------|
| 0 | `Ux` — velocity x-component |
| 1 | `Uy` — velocity z-component |
| 2 | `p`  — kinematic pressure (p/ρ, m²/s²) |

### Loading data

```python
from data import load_data, load_test_data

train_ds, val_splits, stats, sample_weights = load_data()
# train_ds[i]                              → (x, y, is_surface)
# val_splits["val_single_in_dist"][i]      → (x, y, is_surface)
# stats                                    = {x_mean, x_std, y_mean, y_std}
# sample_weights                           → for balanced domain sampling

test_splits = load_test_data()
# test_splits["test_single_in_dist"][i]    → (x, y, is_surface)
# y is joined in from the hidden .test_*_gt/ dirs so the end-of-run test
# evaluation can compute MAE in the trainer without a separate submission step.
```

### Batching and padding

Samples have **variable mesh sizes** (74K to 242K nodes). The dataloader pads each batch to the largest sample using `pad_collate`, which returns:

```python
x, y, is_surface, mask = batch
# x:          [B, N_max, 24]  — padded with zeros
# y:          [B, N_max, 3]   — padded with zeros
# is_surface: [B, N_max]      — False for padding
# mask:       [B, N_max]      — True for real nodes, False for padding
```

**The `mask` tensor is critical.** Model output includes predictions for padding positions — these must be excluded from loss and metrics. The baseline `train.py` and `data/scoring.py` handle this correctly. If you write a custom loss or pooling, always use `mask` to ignore padding.

### Dataset domains

The training data spans three physical domains with different mesh sizes and flow regimes:

| Domain | Samples | Mesh nodes (mean) | Description |
|--------|---------|-------------------|-------------|
| RaceCar single | ~688 train | ~86K  | Single airfoil, Re ~700K–2M, AoA ±10° |
| RaceCar tandem | ~510 train | ~127K | Dual foils (Parts 1+3), Re ~700K–2M |
| Cruise         | ~408 train | ~208K | Tandem cruise foils (Parts 1+3), Re 802K–1.475M |

The three domains are **equally weighted** in training via a balanced sampler (`sample_weights` from `load_data`) — otherwise raceCar single would dominate.

### Physics context

Each sample is a 2D CFD simulation over an overset mesh with up to 3 zones:

```
┌─────────────────────────────────────────────────┐
│  Zone 0 — coarse background (full domain)       │
│                                                 │
│       ┌──────────────┐   ┌──────────────┐       │
│       │  Zone 1      │   │  Zone 2      │       │
│       │  (dense,     │   │  (dense,     │       │
│       │   foil 1)    │   │   foil 2)    │       │
│       └──────────────┘   └──────────────┘       │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Boundary types** in `is_surface`:
- IDs 5, 6 = foil 1 surface (upper/lower)
- ID 7 = foil 2 surface (tandem only)
- IDs 0–4 = interior, inlet, outlet, top/bottom walls

**Value ranges** vary dramatically across domains:

| Domain | Re range | y range (approx) | y std |
|--------|----------|------------------|-------|
| Cruise Part1 (Re=1.475M) | 1.475M  | [-1,278, 233]   | 55  |
| Cruise Part2 (Re=4.445M) | 4.445M  | [-2,360, 2,118] | 304 |
| Cruise Part3 (Re=802K)   | 802K    | [-300, 69]      | 17  |
| RaceCar single           | ~700K–2M | [-874, 467]    | 141 |
| RaceCar tandem           | ~700K–2M | [-4,277, 668]  | 235 |

The wide pressure range across Re numbers is a key challenge — the model must handle both low-Re (small values) and high-Re (extreme values) regimes.

### Parameter space

- **Reynolds number**: 802K to 4.445M (training sees up to ~1.5M; Cruise Part2 at 4.445M is OOD).
- **NACA profiles**: 4-digit codes encoding camber, position, thickness. Single-foil sweeps ~2205–2209; tandem has fixed front foils per Part (2412 / 6416 / 9412).
- **AoA**: ±8° (cruise) to ±10° (raceCar), per-foil for tandem.
- **Gap / stagger**: tandem geometry parameters — gap ~[-0.8, 1.3], stagger ~[0.7, 2.0].

### Splits

Four validation tracks test different generalization axes, each paired with a held-out test split on the same axis:

| Track | What it tests |
|-------|----------------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity: random holdout from single-foil |
| `val_geom_camber_rc` / `test_geom_camber_rc` | Unseen front foil camber M=6-8 — geometry interpolation (raceCar) |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Unseen front foil camber M=2-4 — geometry interpolation (cruise) |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem domains — cross-regime generalization |

Val is 4 × 100 samples; test is 4 × 200 samples. See `data/SPLITS.md` for the full split design rationale, per-file counts, and the reasoning behind each holdout.

## Model contract

The baseline `train.py` plugs a Transolver into this interface:

- **Input**: `{"x": tensor [B, N, 24]}` — batch of **normalized** node features.
- **Output**: `{"preds": tensor [B, N, 3]}` — predicted `[Ux, Uy, p]` in **normalized** space (the `(y - y_mean) / y_std` space).

Normalization is applied outside the model:

```python
# Normalize inputs
x_norm = (x - stats["x_mean"]) / stats["x_std"]
# Normalize targets for loss computation
y_norm = (y - stats["y_mean"]) / stats["y_std"]
# Model predicts in normalized space
pred = model({"x": x_norm})["preds"]  # [B, N, 3]
# Denormalize for MAE computation
pred_phys = pred * stats["y_std"] + stats["y_mean"]
```

Keep this contract intact — `data/scoring.py` assumes predictions are given in normalized space and denormalizes with `y_std * pred + y_mean` before MAE is taken.

## Metrics

All metrics in `data/scoring.py` are computed in the original (denormalized) target space, in float64, with per-sample skipping for non-finite ground truth. The same helpers are used for validation during training and for the end-of-run test evaluation so the numbers are apples-to-apples.

**Primary ranking metric.** Equal-weight mean surface-pressure MAE across the four splits:

- val: `val_avg/mae_surf_p` — averaged across the four validation splits
- test: `test_avg/mae_surf_p` — averaged across the four test splits

Aggregation is global over all valid surface nodes in the split:

```
mae_surf_p(S) = Σ_{valid surface nodes in S} |p_pred - p_true| / n_surf_nodes
```

**Per-split diagnostics** (logged every epoch for each val split, and once at the end for each test split):

- `{split}/mae_surf_{Ux, Uy, p}` — surface MAE per channel, physical units
- `{split}/mae_vol_{Ux, Uy, p}` — volume MAE per channel, physical units
- `{split}/loss`, `{split}/vol_loss`, `{split}/surf_loss` — normalized-space losses used for training

**Checkpoint selection.** Best checkpoint = lowest `val_avg/mae_surf_p`. That checkpoint is the one evaluated on the test splits at the end of training, and the one saved as a W&B model artifact (`model-<wandb_name-or-agent>-<run_id>` with aliases `best` and `epoch-N`).

Lower is better. For paper-facing numbers, the decision-driving quantity is `test_avg/mae_surf_p`. Surface pressure accuracy matters most.

## Constraints

- **VRAM**: GPUs have 96 GB. Don't OOM — meshes can reach 242K nodes.
- **Timeout**: each training run is capped by `SENPAI_TIMEOUT_MINUTES` (wall clock) and `--epochs`. Do not override these.
- **Simplicity**: all else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.
- **Data loaders are read-only.** Don't change the interface in `data/`. If you need a different sampler or feature transform, do it in `train.py`.
- **No new packages** outside of `pyproject.toml`. If you need one, add it in the same PR that uses it.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
