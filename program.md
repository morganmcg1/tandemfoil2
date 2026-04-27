<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet

Research target for CFD surrogate modelling on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Given tandem-airfoil geometry and flow conditions, predict the full velocity `(Ux, Uy)` and pressure `p` field at every mesh node.

The baseline is a [Transolver](https://arxiv.org/abs/2402.02366) with physics-aware attention over irregular meshes. Beat it.

## Codebase

- `train.py` — trainer. Transolver model, training loop, validation, final test evaluation, local checkpoint and metric summaries. **Primary editable entrypoint.**
- `data/loader.py` — `SplitDataset`, `TestDataset`, `load_data`, `load_test_data`, `pad_collate`. **Read-only during normal experiment PRs.**
- `data/scoring.py` — MAE accumulation shared by val and test. **Read-only.**
- `data/prepare_splits.py` — materializes per-sample `.pt` files on the PVC from `data/split_manifest.json`. Organizer script, execute once. **Read-only.**
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
# y is joined in from the hidden .test_*_gt/ dirs so the final test
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

The training data spans three physical domains with different mesh sizes and flow regimes (counts are exact from `meta.json`):

| Domain | Train samples | Mesh nodes (mean) | Description |
|--------|---------------|-------------------|-------------|
| RaceCar single | 599 | ~85K  | Single inverted airfoil with ground effect, Re 100K–5M, AoA -10° to 0° |
| RaceCar tandem | 457 | ~127K | Dual inverted foils (raceCar Parts 1+3), Re 1M–5M, AoA -10° to 0° |
| Cruise         | 443 | ~210K | Tandem freestream foils (cruise Parts 1+3), Re 110K–5M, AoA -5° to +6° |

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

The raw dataset contains per-node boundary IDs (foil surfaces vs. interior / inlet / outlet / walls); in these preprocessed samples those are collapsed to a single boolean `is_surface` (True on either foil surface, False everywhere else). The model does not see a foil 1 vs. foil 2 distinction — use dims 18–23 of `x` (foil 2 AoA, NACA, gap, stagger) to tell tandem from single.

**Value ranges.** Target magnitudes vary dramatically across domains (summary from the single-file val holdouts):

| Source split | Re range | y range (min, max) | Avg per-sample y std | Max per-sample y std |
|--------------|----------|--------------------|----------------------|----------------------|
| `val_single_in_dist` (raceCar single) | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| `val_geom_camber_rc` (raceCar tandem P2, M=6-8) | 1.0M–5M | (-10,312, +2,228) | 377 | 1,237 |
| `val_geom_camber_cruise` (cruise tandem P2, M=2-4) | 122K–5M | (-7,648, +2,648) | 164 | 506 |

Within every split, high-Re samples drive the extremes — per-sample y std varies by an order of magnitude even inside one domain. The model must handle both low-Re (small values) and high-Re (extreme values) regimes.

### Parameter space

- **Reynolds number**: ~100K to ~5M across the whole corpus. Training sees the full range; there is no intentional OOD Re slice.
- **NACA profiles**: 4-digit codes encoding camber (M), position (P), thickness (T). Features 15–17 are normalized to [0, 1]. Single-foil (file 0) sweeps M=2–9 plus a small "specials" cohort that encodes non-NACA foils as `(0, 0, 0)`. Tandem Parts partition the front-foil camber with no overlap:
  - raceCar tandem: P1 M=2–5, P2 M=6–8 (**held out**), P3 M=9 + 5 non-NACA specials
  - cruise tandem:  P1 M=0–2, P2 M=2–4 (**held out**), P3 M=4–6
- **AoA**: raceCar -10° to 0° (inverted, negative loading); cruise -5° to +6°. Per-foil for tandem.
- **Gap / stagger**: tandem geometry parameters — gap ~[-0.8, 1.6], stagger ~[0.0, 2.0]. Both are 0 for single-foil samples.

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

All metrics in `data/scoring.py` are computed in the original (denormalized) target space, in float64, with per-sample skipping for non-finite ground truth. The same helpers are used for validation during training and for the final test evaluation so the numbers are apples-to-apples.

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

**Checkpoint selection.** Best checkpoint = lowest `val_avg/mae_surf_p`. That checkpoint is the one evaluated on the test splits at the end of training and saved under `models/<experiment>/checkpoint.pt` with `metrics.yaml` and `metrics.jsonl`.

Lower is better. For paper-facing numbers, the decision-driving quantity is `test_avg/mae_surf_p`. Surface pressure accuracy matters most.

## Constraints

- **VRAM**: GPUs have 96 GB. Don't OOM — meshes can reach 242K nodes.
- **Timeout**: each training execution is capped by `SENPAI_TIMEOUT_MINUTES` (wall clock) and `--epochs`. Do not override these.
- **Simplicity**: all else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.
- **Data loaders are read-only.** Don't change the interface in `data/`. If you need a different sampler or feature transform, do it in `train.py`.
- **No new packages** outside of `pyproject.toml`. If you need one, add it in the same PR that uses it.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team. See `instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
