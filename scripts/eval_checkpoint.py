"""Re-evaluate a saved checkpoint on val + test splits.

Useful for:
- Producing test_avg/mae_surf_p for finalists that originally trained with
  --skip_test true.
- Sanity-checking that an EMA checkpoint, when loaded back, gives the val
  numbers we logged during training.

Usage:
  python scripts/eval_checkpoint.py \\
    --checkpoint models/model-<run_id>/checkpoint.pt \\
    [--config models/model-<run_id>/config.yaml]   # optional, falls back to defaults
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Allow `from data import ...` and `from train import ...`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    aggregate_splits,
    load_data,
    load_test_data,
    pad_collate,
)
from train import Transolver, evaluate_split, print_split_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="Optional config.yaml; if missing, baseline defaults used")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 autocast during eval (matches training).")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = args.config
    if cfg_path is None:
        # Fall back to sibling config.yaml of the checkpoint
        sibling = Path(args.checkpoint).with_name("config.yaml")
        if sibling.exists():
            cfg_path = str(sibling)
    if cfg_path is None:
        model_config = dict(
            space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
            n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
            output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
        )
        print("[INFO] No config.yaml found; using baseline defaults.")
    else:
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)
        print(f"[INFO] Loaded model_config from {cfg_path}")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(
        collate_fn=pad_collate, num_workers=4, pin_memory=True,
        persistent_workers=False, prefetch_factor=2,
    )
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = Transolver(**model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={list(missing)[:3]}... unexpected={list(unexpected)[:3]}...")
    model.eval()
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    print("\n=== Validation ===")
    val_metrics = {
        name: evaluate_split(
            model, loader, stats, args.surf_weight, device,
            autocast_dtype=autocast_dtype,
        )
        for name, loader in val_loaders.items()
    }
    val_agg = aggregate_splits(val_metrics)
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, val_metrics[name])
    print(f"  VAL  avg_surf_p={val_agg['avg/mae_surf_p']:.4f}")

    test_metrics = None
    test_agg = None
    if not args.skip_test:
        print("\n=== Test ===")
        test_datasets = load_test_data(args.splits_dir)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(
                model, loader, stats, args.surf_weight, device,
                autocast_dtype=autocast_dtype,
            )
            for name, loader in test_loaders.items()
        }
        test_agg = aggregate_splits(test_metrics)
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        print(f"  TEST  avg_surf_p={test_agg['avg/mae_surf_p']:.4f}")

    if args.out_json:
        out = {
            "checkpoint": args.checkpoint,
            "config": cfg_path,
            "val_per_split": val_metrics,
            "val_avg": val_agg,
            "test_per_split": test_metrics,
            "test_avg": test_agg,
        }
        # Convert any tensors to plain floats
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
