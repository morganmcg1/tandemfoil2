"""Retro-evaluate a saved Transolver checkpoint on val + test in fp32.

Standalone — independent of train.py's at-import-time training loop.
Useful when a training run's autocast forward pass produced NaN test
predictions on the larger cruise meshes.

Usage:
  python eval_checkpoint.py --model_dir models/model-<run_id> \
      [--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 \
       --ada_temp false --gumbel false --unified_pos false --dropout 0.0 \
       --ref 8] \
      [--splits_dir /mnt/new-pvc/datasets/tandemfoil/splits_v2] \
      [--batch_size 4] [--out_json research/eval_<name>.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
from model import Transolver  # extracted to model.py to avoid running train.py
                              # top-level training at import time.


def evaluate(model, loader, stats, surf_weight, device, label):
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
            pred = model({"x": x_norm})["preds"].float()
            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum()
                             / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum()
                              / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1
            # Same NaN-y guard as train.py — see docstring there.
            B = y.shape[0]
            y_finite_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not bool(y_finite_sample.all()):
                bad_extend = (~y_finite_sample).view(B, 1).expand_as(mask)
                mask_for_acc = mask & ~bad_extend
                y_for_acc = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                mask_for_acc = mask
                y_for_acc = y
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(
                pred_orig, y_for_acc, is_surface, mask_for_acc, mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv
    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True,
                   help="Directory with checkpoint.pt and config.yaml.")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Override checkpoint path (default: model_dir/checkpoint.pt).")
    p.add_argument("--config_yaml", type=str, default=None,
                   help="Override config.yaml (default: model_dir/config.yaml).")

    # Model overrides (used only if config.yaml is missing keys)
    p.add_argument("--n_hidden", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--slice_num", type=int, default=None)
    p.add_argument("--mlp_ratio", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--ada_temp", action="store_true")
    p.add_argument("--gumbel", action="store_true")
    p.add_argument("--unified_pos", action="store_true")
    p.add_argument("--ref", type=int, default=None)

    p.add_argument("--splits_dir", type=str,
                   default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--surf_weight", type=float, default=10.0)
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else model_dir / "checkpoint.pt"
    config_yaml = Path(args.config_yaml) if args.config_yaml else model_dir / "config.yaml"

    # Load model config
    cfg = {}
    if config_yaml.exists():
        with open(config_yaml) as fh:
            cfg = yaml.safe_load(fh) or {}
        print(f"Loaded model config from {config_yaml}: {cfg}")

    # CLI overrides
    overrides = {
        k: getattr(args, k) for k in
        ("n_hidden", "n_layers", "n_head", "slice_num", "mlp_ratio",
         "dropout", "ada_temp", "gumbel", "unified_pos", "ref")
        if getattr(args, k, None) is not None and getattr(args, k) is not False
    }
    cfg.update(overrides)

    cfg.setdefault("space_dim", 2)
    cfg.setdefault("fun_dim", X_DIM - 2)
    cfg.setdefault("out_dim", 3)
    cfg.setdefault("output_fields", ["Ux", "Uy", "p"])
    cfg.setdefault("output_dims", [1, 1, 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transolver(**cfg).to(device).float()

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARNING: missing={missing} unexpected={unexpected}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params/1e6:.2f}M params) from {ckpt_path}")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    print("\nVal eval:")
    val_metrics = {
        name: evaluate(model, loader, stats, args.surf_weight, device, name)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(val_metrics)
    print(f"  val_avg/mae_surf_p = {val_avg.get('avg/mae_surf_p', float('nan')):.4f}")
    for name in VAL_SPLIT_NAMES:
        m = val_metrics[name]
        print(f"    {name:<26s} surf_p={m['mae_surf_p']:.4f} "
              f"surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    test_metrics = test_avg = None
    if not args.skip_test:
        print("\nTest eval:")
        test_datasets = load_test_data(args.splits_dir, debug=False)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate(model, loader, stats, args.surf_weight, device, name)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"  test_avg/mae_surf_p = {test_avg.get('avg/mae_surf_p', float('nan')):.4f}")
        for name in TEST_SPLIT_NAMES:
            m = test_metrics[name]
            print(f"    {name:<26s} surf_p={m['mae_surf_p']:.4f} "
                  f"surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    out = {
        "model_dir": str(model_dir),
        "ckpt_path": str(ckpt_path),
        "n_params": n_params,
        "model_config": {k: v for k, v in cfg.items()
                         if k not in ("output_fields", "output_dims")},
        "val_per_split": {n: {k: float(v) for k, v in m.items()}
                          for n, m in val_metrics.items()},
        "val_avg": {k: float(v) for k, v in val_avg.items()},
        "test_per_split": {n: {k: float(v) for k, v in m.items()}
                           for n, m in (test_metrics or {}).items()} if test_metrics else None,
        "test_avg": {k: float(v) for k, v in (test_avg or {}).items()} if test_avg else None,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote eval JSON to {out_path}")


if __name__ == "__main__":
    main()
