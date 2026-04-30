"""Ensemble evaluation: load multiple checkpoints, average predictions per
node, then compute val + test MAE through ``data.scoring`` so the metrics are
identical to the trainer's.

Usage:
  python scripts/ensemble_eval.py \\
    --checkpoints models/model-A/checkpoint.pt models/model-B/checkpoint.pt \\
    [--configs models/model-A/config.yaml models/model-B/config.yaml] \\
    [--bf16] [--skip_test]

Models in the ensemble do NOT need to share architecture — each is loaded with
its own config.yaml (or baseline defaults if the file is missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)
from train import Transolver, print_split_metrics  # noqa: E402


def load_one(checkpoint: Path, config_path: Path | None, device) -> torch.nn.Module:
    if config_path is None:
        sibling = checkpoint.with_name("config.yaml")
        if sibling.exists():
            config_path = sibling
    if config_path is None:
        model_config = dict(
            space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
            n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
            output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
        )
        print(f"  [WARN] No config.yaml for {checkpoint}; using baseline defaults")
    else:
        with open(config_path) as f:
            model_config = yaml.safe_load(f)
    model = Transolver(**model_config).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [WARN] {checkpoint}: missing/unexpected keys={list(missing)[:2]} {list(unexpected)[:2]}")
    return model


def evaluate_ensemble(models: list, loader, stats, surf_weight: float, device,
                      autocast_dtype=None) -> dict[str, float]:
    """Same semantics as ``train.evaluate_split`` but predictions are averaged
    across models in the ensemble before denorm + MAE computation."""
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            preds = []
            for model in models:
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        p = model({"x": x_norm})["preds"]
                    p = p.float()
                else:
                    p = model({"x": x_norm})["preds"]
                preds.append(p)
            pred = torch.stack(preds).mean(dim=0)

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (sq_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    out = {
        "vol_loss": vol_loss_sum / max(n_batches, 1),
        "surf_loss": surf_loss_sum / max(n_batches, 1),
    }
    out["loss"] = out["vol_loss"] + surf_weight * out["surf_loss"]
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Optional list of config.yaml paths matching checkpoints")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    if args.configs and len(args.configs) != len(args.checkpoints):
        raise ValueError("--configs must match --checkpoints in length")

    print(f"Loading {len(args.checkpoints)} models for ensemble...")
    models = []
    for i, ckpt in enumerate(args.checkpoints):
        cfg = args.configs[i] if args.configs else None
        models.append(load_one(Path(ckpt), Path(cfg) if cfg else None, device))
        n_params = sum(p.numel() for p in models[-1].parameters())
        print(f"  [{i}] {ckpt}: {n_params/1e6:.2f}M params")

    # Load val (and optionally test) data
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

    print("\n=== Validation (ensemble) ===")
    val_metrics = {
        name: evaluate_ensemble(models, loader, stats, args.surf_weight, device,
                                autocast_dtype=autocast_dtype)
        for name, loader in val_loaders.items()
    }
    val_agg = aggregate_splits(val_metrics)
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, val_metrics[name])
    print(f"  VAL ENSEMBLE  avg_surf_p={val_agg['avg/mae_surf_p']:.4f}")

    test_metrics = None
    test_agg = None
    if not args.skip_test:
        print("\n=== Test (ensemble) ===")
        test_datasets = load_test_data(args.splits_dir)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_ensemble(models, loader, stats, args.surf_weight, device,
                                    autocast_dtype=autocast_dtype)
            for name, loader in test_loaders.items()
        }
        test_agg = aggregate_splits(test_metrics)
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        print(f"  TEST ENSEMBLE  avg_surf_p={test_agg['avg/mae_surf_p']:.4f}")

    if args.out_json:
        out = {
            "checkpoints": args.checkpoints,
            "val_per_split": val_metrics,
            "val_avg": val_agg,
            "test_per_split": test_metrics,
            "test_avg": test_agg,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
