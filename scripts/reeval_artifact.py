"""Re-evaluate a saved W&B model artifact against the (now NaN-safe) test scorer.

Usage:
  python scripts/reeval_artifact.py --artifact <full-artifact-name-with-version>

The script loads the artifact's checkpoint.pt + model_config metadata, rebuilds
the Transolver from train.py with that exact config, and runs evaluate_split
on the four test splits. Numbers are reported in the same shape as
``test_avg/mae_surf_p`` and ``test/<split>/mae_surf_p``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    TEST_SPLIT_NAMES,
    aggregate_splits,
    load_test_data,
    pad_collate,
)
from data.loader import _load_stats
from train import Transolver, evaluate_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True,
                        help="Full W&B artifact name, e.g. 'wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3/model-v1-zaqz12qi:best'")
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--surf_weight", type=float, default=10.0)
    args = parser.parse_args()

    api = wandb.Api()
    art = api.artifact(args.artifact, type="model")
    print(f"Artifact: {art.name}")
    print(f"  metadata.run_id: {art.metadata.get('run_id')}")
    print(f"  metadata.best_epoch: {art.metadata.get('best_epoch')}")
    print(f"  metadata.best_val_avg/mae_surf_p: {art.metadata.get('best_val_avg/mae_surf_p')}")

    art_dir = Path(art.download())
    ckpt_path = art_dir / "checkpoint.pt"
    print(f"Checkpoint: {ckpt_path}")

    model_config = art.metadata["model_config"]
    print(f"model_config: {json.dumps(model_config, indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stats = _load_stats(Path(args.splits_dir))
    stats = {k: v.to(device) for k, v in stats.items()}

    model = Transolver(**model_config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params/1e6:.2f}M params)")

    test_datasets = load_test_data(args.splits_dir)
    test_loaders = {
        name: DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=pad_collate, num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )
        for name, ds in test_datasets.items()
    }

    print("\nRe-evaluating test splits with NaN-safe scorer...")
    test_metrics = {}
    for name in TEST_SPLIT_NAMES:
        m = evaluate_split(model, test_loaders[name], stats, args.surf_weight, device)
        test_metrics[name] = m
        print(
            f"  {name:<26s} "
            f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
            f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
        )

    test_avg = aggregate_splits(test_metrics)
    print("\n=== TEST AVG (re-evaluated) ===")
    for k in sorted(test_avg.keys()):
        print(f"  {k}: {test_avg[k]:.6f}")


if __name__ == "__main__":
    main()
