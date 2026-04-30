"""Ensemble evaluator.

Averages predictions across multiple checkpoints (in normalized space, the
same space the model outputs) and computes val + test ``avg/mae_surf_p`` using
the same accumulator semantics as ``data/scoring.py``.

Usage:
  python eval_ensemble.py models/model-A models/model-B models/model-C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Reuse model classes + helpers from eval_test.py
sys.path.insert(0, str(Path(__file__).parent))
from eval_test import (
    Transolver,
    print_split_metrics,
)
from data import (
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)


def build_model(cfg_path, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = Transolver(**cfg).to(device)
    return model, cfg


def evaluate_ensemble(models, loader, stats, device):
    """Run inference with each model, average normalized predictions, compute MAE."""
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            preds_norm_sum = None
            for m in models:
                p = m({"x": x_norm})["preds"]
                preds_norm_sum = p if preds_norm_sum is None else preds_norm_sum + p
            pred_norm_avg = preds_norm_sum / len(models)
            pred_orig = pred_norm_avg * stats["y_std"] + stats["y_mean"]
            # Same NaN workaround as eval_test.py
            y_finite_b = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not y_finite_b.all():
                y_clean = torch.where(
                    y_finite_b.view(-1, 1, 1).expand_as(y), y, torch.zeros_like(y)
                )
                mask_clean = mask & y_finite_b.unsqueeze(-1)
            else:
                y_clean, mask_clean = y, mask
            ds, dv = accumulate_batch(pred_orig, y_clean, is_surface, mask_clean,
                                       mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dirs", nargs="+", help="One or more models/model-<id>/ dirs")
    p.add_argument("--splits_dir", type=str,
                   default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    models = []
    for d in args.model_dirs:
        d = Path(d)
        cfg_path = d / "config.yaml"
        ckpt = d / "checkpoint.pt"
        m, cfg = build_model(cfg_path, device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        n = sum(p.numel() for p in m.parameters())
        print(f"Loaded {d.name}: {n/1e6:.2f}M params, cfg={cfg}")
    print(f"\nEnsembling {len(models)} models")

    _, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_splits = load_test_data(args.splits_dir, debug=False)

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)

    print("\n=== Val ===")
    val_metrics = {}
    for name in VAL_SPLIT_NAMES:
        ds = val_splits[name]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_ensemble(models, loader, stats, device)
        val_metrics[name] = m
        m["loss"] = 0.0  # not computed for ensemble
        m["vol_loss"] = 0.0
        m["surf_loss"] = 0.0
        print_split_metrics(name, m)
    val_avg = aggregate_splits(val_metrics)
    print(f"\n  ENSEMBLE VAL  avg_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

    print("\n=== Test ===")
    test_metrics = {}
    for name in TEST_SPLIT_NAMES:
        ds = test_splits[name]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_ensemble(models, loader, stats, device)
        test_metrics[name] = m
        m["loss"] = 0.0; m["vol_loss"] = 0.0; m["surf_loss"] = 0.0
        print_split_metrics(name, m)
    test_avg = aggregate_splits(test_metrics)
    print(f"\n  ENSEMBLE TEST avg_surf_p = {test_avg['avg/mae_surf_p']:.4f}")


if __name__ == "__main__":
    main()
