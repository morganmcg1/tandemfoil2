# TandemFoilSet-Balanced

This repo contains:
1) a custom TandemFoilSet dataset split called `TandemFoilSet-Balanced` based on the **[TandemFoilSet Paper](https://openreview.net/forum?id=4Z0P4Nbosn)**
2) a base set of data and training files that the [senpai](https://github.com/wandb/senpai) autoresearch harness can use as a baseline to try and improve performance on this dataset.

## Citation

If you use this balanced split design or benchmark package, please cite the **BibTeX for TandemFoilSet-Balanced**

```bibtex
@misc{mcguire2026tandemfoilsetbalanced,
  title        = {{TandemFoilSet-Balanced}: Balanced Split Design and CFD Surrogate Benchmark Package},
  author       = {McGuire, Morgan and Capelle, Thomas},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/morganmcg1/TandemFoilSet-Balanced},
  note         = {Accessed 2026-04-24}
}
```

## TandemFoilSet-Balanced Splits Rationale
The goal behind this split is to be able to say:
- The trained model is good/bad at generalizing to unseen geometries (train on low + high camber, val/test on moderate)
- AND the same model works across different Reynolds numbers for both race car and cruise (train on low, med, high Re, val/test on low, moderate, high) - random split across all Re numbers
- AND the model performs well on a single-foil random split as a sanity check

See **[SPLITS.md](data/SPLITS.md)** for a full description of the dataset splits.

## Training Files

Below are details of the training files that can be used in an autoresearch context. These provide an agent with a base train.py file as well as instructions of metrics to optimize, information about the dataset and file to preprocess the data and calculate results.

### Files Layout

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

### Data

Pre-processed samples live on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/` (private author-specific PVC, you'll have to download the dataset yourself to be able to run this code). Materialize them once from the manifest:

```
python data/prepare_splits.py
```

After that, `train.py` streams `.pt` samples directly — no re-preprocessing per run.

See `program.md` for input feature layout (24 input dims), target channels, split design, and the full metric contract.

### Training

```
python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]
```

The trainer logs per-epoch metrics to W&B and prints a per-split MAE breakdown to the console every epoch. At the end of the run it loads the best checkpoint (selected on `val_avg/mae_surf_p`) and evaluates it on the four held-out test splits, logging `test_avg/mae_surf_p` plus per-split per-channel MAEs.

Environment:

- `SENPAI_TIMEOUT_MINUTES` — wall-clock cap (default 30)
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE` — W&B routing


### BibTeX for the original TandemFoilSet paper

For the underlying dataset, cite the original TandemFoilSet paper:

```bibtex
@inproceedings{lim2026tandemfoilset,
  title     = {{TandemFoilSet}: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils},
  author    = {Lim, Wei Xian and Loh, Sher En Jessica and Li, Zenong and Oo, Thant Zin and Chan, Wai Lee and Kong, Adams Wai-Kin},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=4Z0P4Nbosn}
}
```
