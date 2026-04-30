"""Re-run test/val evaluation on a saved checkpoint in fp32.

Useful when training used bf16 autocast and the trainer's bf16 test pass
overflowed on the largest cruise meshes, leaving test_geom_camber_cruise
mae_surf_p = nan in the original run. The saved weights themselves are
unaffected — they're stored in fp32. This script just runs the eval pass
without autocast so the predicted pressure stays finite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_data,
    load_test_data,
    pad_collate,
)
from models import Transolver
from train import evaluate_split, print_split_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint.pt")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--splits_dir",
        default="/mnt/new-pvc/datasets/tandemfoil/splits_v2",
    )
    ap.add_argument("--out_json", default=None, help="Optional path to write metrics JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        model_config = yaml.safe_load(f)
    print(f"Model config: {model_config}")

    # Build model with same constructor signature as train.py
    constructor_kwargs = dict(
        space_dim=model_config.get("space_dim", 2),
        fun_dim=model_config.get("fun_dim", X_DIM - 2),
        out_dim=model_config.get("out_dim", 3),
        n_hidden=model_config["n_hidden"],
        n_layers=model_config["n_layers"],
        n_head=model_config["n_head"],
        slice_num=model_config["slice_num"],
        mlp_ratio=model_config["mlp_ratio"],
        dropout=model_config.get("dropout", 0.0),
        use_eidetic=model_config.get("use_eidetic", False),
        output_fields=model_config.get("output_fields", ["Ux", "Uy", "p"]),
        output_dims=model_config.get("output_dims", [1, 1, 1]),
    )
    model = Transolver(**constructor_kwargs).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params/1e6:.2f}M params")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(
        collate_fn=pad_collate, num_workers=2, pin_memory=True,
        persistent_workers=False, prefetch_factor=2,
    )

    # Val
    print("\n=== VAL (fp32) ===")
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    val_metrics = {
        name: evaluate_split(model, loader, stats, args.surf_weight, device, autocast_dtype=None)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(val_metrics)
    print(f"VAL avg_surf_p={val_avg['avg/mae_surf_p']:.4f}")
    for n in VAL_SPLIT_NAMES:
        print_split_metrics(n, val_metrics[n])

    # Test
    print("\n=== TEST (fp32) ===")
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }
    test_metrics = {
        name: evaluate_split(model, loader, stats, args.surf_weight, device, autocast_dtype=None)
        for name, loader in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    print(f"TEST avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for n in TEST_SPLIT_NAMES:
        print_split_metrics(n, test_metrics[n])

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "config": model_config,
                "n_params": n_params,
                "val_per_split": val_metrics,
                "val_avg": val_avg,
                "test_per_split": test_metrics,
                "test_avg": test_avg,
            }, f, indent=2)
        print(f"\nWrote metrics to {args.out_json}")


if __name__ == "__main__":
    main()
