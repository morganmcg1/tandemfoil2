# TandemFoilSet-Balanced

**[TandemFoilSet Paper](https://openreview.net/forum?id=4Z0P4Nbosn)**

This repo contains a custom TandemFoilSet dataset split as well as a base set of files that the [senpai](https://github.com/wandb/senpai) autoresearch harness can use as input to try and improve performance.

## Splits
The goal behind the split is to be able to say:
- The model is good/bad at generalizing to unseen geometries (train on low + high camber, val/test on moderate)
- The same model works across different Reynolds numbers for both race car and cruise (train on low, med, high Re, val/test on low, moderate, high) - random split across all Re numbers
- General single-foil random split as a sanity check

See **[SPLITS.MD](https://github.com/morganmcg1/tandemfoil2/blob/main/data/SPLITS.md)** for a full description of the dataset splits

## Targets

CFD surrogate research target: predict the full velocity `(Ux, Uy)` and pressure `p` field on tandem-airfoil meshes from TandemFoilSet.

## Layout

```
.
├── README.md
├── program.md                 # research contract — read this first
├── train.py                   # Transolver trainer; primary editable entrypoint
├── instructions/
│   ├── prompt-advisor.md
│   └── prompt-student.md
└── data/
    ├── __init__.py
    ├── loader.py              # SplitDataset, TestDataset, load_data, load_test_data
    ├── scoring.py             # MAE accumulation shared by val and test
    ├── prepare_splits.py      # one-shot: materialize .pt samples onto the PVC
    ├── generate_manifest.py   # regenerate split_manifest.json from raw data
    ├── split_manifest.json    # pinned split definition (source of truth)
    └── SPLITS.md              # split design, holdout reasons, per-file counts
```

## Data

Pre-processed samples live on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`. Materialize them once from the manifest:

```
python data/prepare_splits.py
```

After that, `train.py` streams `.pt` samples directly — no re-preprocessing per run.

See `program.md` for feature layout (24 input dims), target channels, split design, and the full metric contract.

## Training

```
python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]
```

The trainer logs per-epoch metrics to W&B and prints a per-split MAE breakdown to the console every epoch. At the end of the run it loads the best checkpoint (selected on `val_avg/mae_surf_p`) and evaluates it on the four held-out test splits, logging `test_avg/mae_surf_p` plus per-split per-channel MAEs.

Environment:

- `SENPAI_TIMEOUT_MINUTES` — wall-clock cap (default 30)
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE` — W&B routing
