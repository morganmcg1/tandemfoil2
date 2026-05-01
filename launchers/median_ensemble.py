#!/usr/bin/env python3
"""Median ensemble eval.

Computes per-node median across models in normalized space, then
denormalizes and accumulates MAE the same way as the organizer scorer.

Usage:
  python launchers/median_ensemble.py --checkpoints ck1.pt ck2.pt ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_eval import Transolver  # noqa: E402

from data import (
    SPLITS_DIR, TEST_SPLIT_NAMES, VAL_SPLIT_NAMES,
    accumulate_batch, aggregate_splits, finalize_split,
    load_data, load_test_data, pad_collate,
)


def evaluate_median_ensemble(models, loader, stats, device):
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
            preds = []
            for m in models:
                preds.append(m({"x": x_norm})["preds"])
            stack = torch.stack(preds, dim=0)  # [M, B, N, 3]
            pred = stack.median(dim=0).values  # [B, N, 3]
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--config_yaml", required=False)
    ap.add_argument("--splits_dir", default=str(SPLITS_DIR))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.config_yaml is None:
        first = Path(args.checkpoints[0]).parent / "config.yaml"
        args.config_yaml = str(first)
    with open(args.config_yaml) as f:
        model_config = yaml.safe_load(f)

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

    print("\n=== Median ensemble validation ===")
    val_metrics = {}
    for name in VAL_SPLIT_NAMES:
        loader = DataLoader(val_splits[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_median_ensemble(models, loader, stats, device)
        val_metrics[name] = m
        print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f}]")
    val_avg = aggregate_splits(val_metrics)
    print(f"\n  VAL  avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

    test_metrics = None; test_avg = None
    if not args.skip_test:
        print("\n=== Median ensemble test ===")
        test_datasets = load_test_data(splits_dir)
        test_metrics = {}
        for name in TEST_SPLIT_NAMES:
            loader = DataLoader(test_datasets[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            m = evaluate_median_ensemble(models, loader, stats, device)
            test_metrics[name] = m
            print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f}]")
        test_avg = aggregate_splits(test_metrics)
        print(f"  TEST avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")

    out = {
        "kind": "median_ensemble",
        "n_members": len(models),
        "checkpoints": list(args.checkpoints),
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
