"""Ensemble several saved checkpoints by averaging their normalized predictions.

Usage:
  python eval_ensemble.py --models models/model-A models/model-B ... --out_json research/eval_ensemble.json
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
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)
from models import Transolver
from train import print_split_metrics


def build_model(model_dir: Path, device: torch.device) -> Transolver:
    with open(model_dir / "config.yaml") as f:
        mc = yaml.safe_load(f)
    model = Transolver(
        space_dim=mc.get("space_dim", 2),
        fun_dim=mc.get("fun_dim", X_DIM - 2),
        out_dim=mc.get("out_dim", 3),
        n_hidden=mc["n_hidden"],
        n_layers=mc["n_layers"],
        n_head=mc["n_head"],
        slice_num=mc["slice_num"],
        mlp_ratio=mc["mlp_ratio"],
        dropout=mc.get("dropout", 0.0),
        use_eidetic=mc.get("use_eidetic", False),
        output_fields=mc.get("output_fields", ["Ux", "Uy", "p"]),
        output_dims=mc.get("output_dims", [1, 1, 1]),
    ).to(device)
    state = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_split_ensemble(
    models, loader, stats, surf_weight, device,
):
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

            B = y.shape[0]
            sample_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not sample_finite.all():
                mask = mask & sample_finite[:, None]
                y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            preds = []
            for m in models:
                p = m({"x": x_norm})["preds"]
                preds.append(p)
            pred = torch.stack(preds, 0).mean(0)

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--skip_val", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [build_model(Path(m), device) for m in args.models]
    print(f"Loaded {len(models)} models for ensemble")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(
        collate_fn=pad_collate, num_workers=2, pin_memory=True,
        persistent_workers=False, prefetch_factor=2,
    )

    out_metrics = {"models": args.models}

    if not args.skip_val:
        print("\n=== ENSEMBLE VAL (fp32) ===")
        val_loaders = {
            n: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for n, ds in val_splits.items()
        }
        val_metrics = {
            n: evaluate_split_ensemble(models, l, stats, args.surf_weight, device)
            for n, l in val_loaders.items()
        }
        val_avg = aggregate_splits(val_metrics)
        print(f"VAL avg_surf_p={val_avg['avg/mae_surf_p']:.4f}")
        for n in VAL_SPLIT_NAMES:
            print_split_metrics(n, val_metrics[n])
        out_metrics["val_per_split"] = val_metrics
        out_metrics["val_avg"] = val_avg

    print("\n=== ENSEMBLE TEST (fp32) ===")
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {
        n: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for n, ds in test_datasets.items()
    }
    test_metrics = {
        n: evaluate_split_ensemble(models, l, stats, args.surf_weight, device)
        for n, l in test_loaders.items()
    }
    test_avg = aggregate_splits(test_metrics)
    print(f"TEST avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for n in TEST_SPLIT_NAMES:
        print_split_metrics(n, test_metrics[n])
    out_metrics["test_per_split"] = test_metrics
    out_metrics["test_avg"] = test_avg

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out_metrics, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
