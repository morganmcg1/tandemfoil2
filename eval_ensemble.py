"""Ensemble evaluation of multiple Transolver checkpoints.

Loads N saved checkpoints, runs each through val + test in fp32, and reports:
  * per-model val/test metrics (matching ``eval_checkpoint.py``)
  * ensemble metrics where the per-node prediction is the per-model
    mean before denormalization. Variance reduction across independent runs
    typically lowers MAE by 5-15 % on field-prediction tasks.

Includes the same NaN-y guard as ``train.py:evaluate_split`` so the
``test_geom_camber_cruise/000020.pt`` non-finite-p sample doesn't poison the
test_avg float64 accumulator.

Usage:
  python eval_ensemble.py \
      --model_dirs models/model-A models/model-B models/model-C \
      [--names a b c] [--batch_size 4] [--out_json research/ensemble.json]
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
from model import Transolver


def _build_model(config_yaml: Path, ckpt_path: Path, device) -> torch.nn.Module:
    cfg = {}
    if config_yaml.exists():
        with open(config_yaml) as fh:
            cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("space_dim", 2)
    cfg.setdefault("fun_dim", X_DIM - 2)
    cfg.setdefault("out_dim", 3)
    cfg.setdefault("output_fields", ["Ux", "Uy", "p"])
    cfg.setdefault("output_dims", [1, 1, 1])
    model = Transolver(**cfg).to(device).float()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _accumulate_split(
    pred_orig_fn,  # callable: (x, y, is_surface, mask) -> pred_orig (avg over models)
    loader,
    stats,
    surf_weight,
    device,
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
            pred_orig, pred_norm, y_norm = pred_orig_fn(x, y, is_surface, mask)
            sq_err = (pred_norm - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum()
                             / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum()
                              / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1
            B = y.shape[0]
            y_finite_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if not bool(y_finite_sample.all()):
                bad_extend = (~y_finite_sample).view(B, 1).expand_as(mask)
                mask_for_acc = mask & ~bad_extend
                y_for_acc = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                mask_for_acc = mask
                y_for_acc = y
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
    p.add_argument("--model_dirs", nargs="+", required=True,
                   help="Each must contain checkpoint.pt and config.yaml.")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--splits_dir", type=str,
                   default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--surf_weight", type=float, default=10.0)
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--skip_individual", action="store_true",
                   help="Skip per-model eval, only run ensemble.")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dirs = [Path(d) for d in args.model_dirs]
    names = args.names or [d.name for d in model_dirs]
    assert len(names) == len(model_dirs)

    # Load all models eagerly — they're small (<100MB each).
    models = []
    for d, n in zip(model_dirs, names):
        ckpt = d / "checkpoint.pt"
        yml = d / "config.yaml"
        m = _build_model(yml, ckpt, device)
        params = sum(p.numel() for p in m.parameters())
        print(f"  loaded {n}: {params/1e6:.2f}M params from {ckpt}")
        models.append(m)

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    out = {"per_model": {}, "ensemble": {}, "names": names,
           "model_dirs": [str(d) for d in model_dirs]}

    def make_single_predictor(model):
        def f(x, y, is_surface, mask):
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred_norm = model({"x": x_norm})["preds"].float()
            pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
            return pred_orig, pred_norm, y_norm
        return f

    def make_ensemble_predictor(models):
        def f(x, y, is_surface, mask):
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred_norm = None
            for m in models:
                p_n = m({"x": x_norm})["preds"].float()
                pred_norm = p_n if pred_norm is None else pred_norm + p_n
            pred_norm = pred_norm / len(models)
            pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
            return pred_orig, pred_norm, y_norm
        return f

    if not args.skip_individual:
        for m, n in zip(models, names):
            print(f"\n=== {n} (single model) ===")
            val_loaders = {
                name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                for name, ds in val_splits.items()
            }
            val_per = {
                vname: _accumulate_split(make_single_predictor(m), loader,
                                         stats, args.surf_weight, device)
                for vname, loader in val_loaders.items()
            }
            val_avg = aggregate_splits(val_per)
            print(f"  val_avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

            test_per = test_avg = None
            if not args.skip_test:
                test_datasets = load_test_data(args.splits_dir, debug=False)
                test_loaders = {
                    name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                    for name, ds in test_datasets.items()
                }
                test_per = {
                    tname: _accumulate_split(make_single_predictor(m), loader,
                                             stats, args.surf_weight, device)
                    for tname, loader in test_loaders.items()
                }
                test_avg = aggregate_splits(test_per)
                print(f"  test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")

            out["per_model"][n] = {
                "val_per_split": {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in val_per.items()},
                "val_avg": {k: float(v) for k, v in val_avg.items()},
                "test_per_split": ({k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in test_per.items()} if test_per else None),
                "test_avg": ({k: float(v) for k, v in test_avg.items()} if test_avg else None),
            }

    print(f"\n=== ENSEMBLE ({len(models)} models) ===")
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }
    val_per = {
        vname: _accumulate_split(make_ensemble_predictor(models), loader,
                                 stats, args.surf_weight, device)
        for vname, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(val_per)
    print(f"  val_avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for n in VAL_SPLIT_NAMES:
        m = val_per[n]
        print(f"    {n:<26s} surf_p={m['mae_surf_p']:.4f} "
              f"surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    test_per = test_avg = None
    if not args.skip_test:
        test_datasets = load_test_data(args.splits_dir, debug=False)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_per = {
            tname: _accumulate_split(make_ensemble_predictor(models), loader,
                                     stats, args.surf_weight, device)
            for tname, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_per)
        print(f"  test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
        for n in TEST_SPLIT_NAMES:
            m = test_per[n]
            print(f"    {n:<26s} surf_p={m['mae_surf_p']:.4f} "
                  f"surf_Ux={m['mae_surf_Ux']:.4f} surf_Uy={m['mae_surf_Uy']:.4f}")

    out["ensemble"] = {
        "val_per_split": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in val_per.items()},
        "val_avg": {k: float(v) for k, v in val_avg.items()},
        "test_per_split": ({k: {kk: float(vv) for kk, vv in v.items()}
                            for k, v in test_per.items()} if test_per else None),
        "test_avg": ({k: float(v) for k, v in test_avg.items()} if test_avg else None),
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote ensemble JSON to {out_path}")


if __name__ == "__main__":
    main()
