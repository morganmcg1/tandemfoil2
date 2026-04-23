"""Dataset and loaders for the pre-materialized TandemFoil splits on PVC.

Run ``python data/prepare_splits.py`` once on the PVC to materialize
``splits_v2/`` from ``data/split_manifest.json``; this module then streams
per-sample ``.pt`` files on demand.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

X_DIM = 24

VAL_SPLIT_NAMES = [
    "val_single_in_dist",
    "val_geom_camber_rc",
    "val_geom_camber_cruise",
    "val_re_rand",
]

TEST_SPLIT_NAMES = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")


class SplitDataset(Dataset):
    """Dataset backed by individual ``{x, y, is_surface}`` .pt files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = torch.load(self.files[idx], weights_only=True)
        return s["x"], s["y"], s["is_surface"]


class TestDataset(Dataset):
    """Test split dataset: reads ``x`` from ``test_*/`` and ``y`` from ``.test_*_gt/``.

    Ground truth is held separately on PVC so prediction submissions can be
    scored blind. For in-repo training, we load both sides here.
    """

    def __init__(self, x_dir: str | Path, gt_dir: str | Path):
        self.x_files = sorted(Path(x_dir).glob("*.pt"))
        self.gt_files = sorted(Path(gt_dir).glob("*.pt"))
        assert len(self.x_files) == len(self.gt_files), (
            f"Test split file-count mismatch: {len(self.x_files)} x-files vs "
            f"{len(self.gt_files)} gt-files"
        )

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        xs = torch.load(self.x_files[idx], weights_only=True)
        gt = torch.load(self.gt_files[idx], weights_only=True)
        return xs["x"], gt["y"], gt["is_surface"]


def pad_collate(batch):
    """Pad variable-length mesh samples into a batch.

    Returns (x, y, is_surface, mask), each ``[B, N_max, ...]``.
    """
    xs, ys, surfs = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    return x_pad, y_pad, surf_pad, mask


def _load_stats(splits_dir: Path) -> dict[str, torch.Tensor]:
    with open(splits_dir / "stats.json") as f:
        raw = json.load(f)
    return {k: torch.tensor(raw[k], dtype=torch.float32) for k in ("x_mean", "x_std", "y_mean", "y_std")}


def load_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> tuple[SplitDataset, dict[str, SplitDataset], dict[str, torch.Tensor], torch.Tensor]:
    """Train + val datasets, normalization stats, and balanced-domain weights.

    Returns ``(train_ds, val_splits, stats, sample_weights)``.
    """
    splits_dir = Path(splits_dir)

    with open(splits_dir / "meta.json") as f:
        meta = json.load(f)
    stats = _load_stats(splits_dir)

    train_ds = SplitDataset(splits_dir / "train")
    val_splits = {name: SplitDataset(splits_dir / name) for name in VAL_SPLIT_NAMES}

    if debug:
        train_ds.files = train_ds.files[:6]
        for ds in val_splits.values():
            ds.files = ds.files[:2]

    domain_groups = meta["domain_groups"]
    group_sizes = {name: len(idxs) for name, idxs in domain_groups.items()}
    idx_to_group: dict[int, str] = {}
    for name, idxs in domain_groups.items():
        for i in idxs:
            idx_to_group[i] = name

    sample_weights = torch.tensor(
        [1.0 / group_sizes[idx_to_group[i]] for i in range(len(train_ds))],
        dtype=torch.float64,
    )

    print(
        f"Train: {len(train_ds)}, "
        + ", ".join(f"{k}: {len(v)}" for k, v in val_splits.items())
    )
    return train_ds, val_splits, stats, sample_weights


def load_test_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> dict[str, TestDataset]:
    """Test datasets keyed by split name (with joined hidden ground truth)."""
    splits_dir = Path(splits_dir)
    test_splits: dict[str, TestDataset] = {}
    for name in TEST_SPLIT_NAMES:
        ds = TestDataset(splits_dir / name, splits_dir / f".{name}_gt")
        if debug:
            ds.x_files = ds.x_files[:2]
            ds.gt_files = ds.gt_files[:2]
        test_splits[name] = ds
    print("Test: " + ", ".join(f"{k}: {len(v)}" for k, v in test_splits.items()))
    return test_splits
