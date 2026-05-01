#!/usr/bin/env python3
"""Weighted ensemble eval (inverse-val weighting).

Same as ensemble_eval.py but each model gets a weight ~ 1/val_score**alpha,
where lower val => higher weight.

Usage:
  python launchers/weighted_ensemble.py \
    --checkpoints ... \
    --val_scores 29.0 29.02 29.08 29.23 ... \
    --alpha 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Reuse model + data + scoring from ensemble_eval
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval import Transolver  # noqa: E402

from data import (
    SPLITS_DIR, TEST_SPLIT_NAMES, VAL_SPLIT_NAMES,
    accumulate_batch, aggregate_splits, finalize_split,
    load_data, load_test_data, pad_collate,
)


def evaluate_weighted_ensemble(models, weights, loader, stats, device):
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()  # normalize

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred = None
            for m, w in zip(models, weights):
                p = m({"x": x_norm})["preds"]
                pred = p * w if pred is None else pred + p * w
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--val_scores", nargs="+", type=float, required=True,
                    help="val score for each checkpoint (in same order)")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="weight = 1/val^alpha. Higher => more weight on best.")
    ap.add_argument("--config_yaml", required=False)
    ap.add_argument("--splits_dir", default=str(SPLITS_DIR))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    assert len(args.checkpoints) == len(args.val_scores), (
        f"checkpoints ({len(args.checkpoints)}) and val_scores "
        f"({len(args.val_scores)}) must match")

    device = torch.device(args.device)
    if args.config_yaml is None:
        first = Path(args.checkpoints[0]).parent / "config.yaml"
        args.config_yaml = str(first)
    with open(args.config_yaml) as f:
        model_config = yaml.safe_load(f)

    # Compute weights ~ 1/val^alpha
    weights = [1.0 / (v ** args.alpha) for v in args.val_scores]
    s = sum(weights)
    weights_norm = [w / s for w in weights]
    print(f"alpha={args.alpha}")
    for ck, v, w in zip(args.checkpoints, args.val_scores, weights_norm):
        print(f"  {Path(ck).parent.name}  val={v:.3f}  w={w:.4f}")

    print(f"Loading {len(args.checkpoints)} checkpoints…")
    models = []
    for ck in args.checkpoints:
        m = Transolver(**model_config).to(device).eval()
        sd = torch.load(ck, map_location=device, weights_only=True)
        m.load_state_dict(sd, strict=True)
        models.append(m)

    splits_dir = Path(args.splits_dir)
    train_ds, val_splits, stats, _ = load_data(splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)

    print("\n=== Weighted ensemble validation ===")
    val_metrics = {}
    for name in VAL_SPLIT_NAMES:
        loader = DataLoader(val_splits[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_weighted_ensemble(models, weights, loader, stats, device)
        val_metrics[name] = m
        print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f}]")
    val_avg = aggregate_splits(val_metrics)
    print(f"\n  VAL  avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

    test_metrics = None; test_avg = None
    if not args.skip_test:
        print("\n=== Weighted ensemble test ===")
        test_datasets = load_test_data(splits_dir)
        test_metrics = {}
        for name in TEST_SPLIT_NAMES:
            loader = DataLoader(test_datasets[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            m = evaluate_weighted_ensemble(models, weights, loader, stats, device)
            test_metrics[name] = m
            print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f}]")
        test_avg = aggregate_splits(test_metrics)
        print(f"  TEST avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")

    out = {
        "alpha": args.alpha,
        "n_members": len(models),
        "checkpoints": list(args.checkpoints),
        "val_scores": list(args.val_scores),
        "weights_norm": weights_norm,
        "val_per_split": val_metrics,
        "val_avg": val_avg,
    }
    if test_metrics is not None:
        out["test_per_split"] = test_metrics
        out["test_avg"] = test_avg
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
