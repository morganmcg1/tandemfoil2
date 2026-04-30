"""Model soup: average weights of multiple checkpoints (vs ensemble of predictions).

Loads multiple model checkpoints, averages their state dicts, evaluates the
single resulting model on val/test splits. Models must have the same architecture.

Usage:
  python model_soup.py \
    --checkpoint_paths model_dir1/checkpoint.pt,model_dir2/checkpoint.pt,... \
    --config_paths model_dir1/config.yaml,model_dir2/config.yaml,... \
    [--splits_dir /mnt/new-pvc/datasets/tandemfoil/splits_v2] \
    [--batch_size 4]
"""

from __future__ import annotations
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Reuse model definition from ensemble_eval.py
import importlib.util
_path = Path(__file__).resolve().parent / "ensemble_eval.py"
_spec = importlib.util.spec_from_file_location("_ens_module", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Transolver = _mod.Transolver


def main():
    from data import (
        VAL_SPLIT_NAMES,
        TEST_SPLIT_NAMES,
        load_data,
        load_test_data,
        pad_collate,
        accumulate_batch,
        finalize_split,
        aggregate_splits,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_paths", required=True, type=str)
    parser.add_argument("--config_paths", required=True, type=str)
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    ckpts = args.checkpoint_paths.split(",")
    cfgs = args.config_paths.split(",")
    assert len(ckpts) == len(cfgs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, soup size: {len(ckpts)}")

    # Load first model as template
    with open(cfgs[0]) as f:
        mc = yaml.safe_load(f)
    model = Transolver(**mc).to(device)

    # Load all state dicts and average them
    avg_sd = None
    for ckpt_path in ckpts:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        if avg_sd is None:
            avg_sd = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k in avg_sd:
                avg_sd[k] += sd[k].float()
    for k in avg_sd:
        avg_sd[k] /= len(ckpts)

    model.load_state_dict(avg_sd)
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters())/1e6:.2f}M param soup")

    # Load data
    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    def evaluate_split_one(loader):
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
                if not y_finite.any():
                    continue
                if not y_finite.all():
                    good = torch.where(y_finite)[0]
                    x = x[good]; y = y[good]; is_surface = is_surface[good]; mask = mask[good]
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                pred = model({"x": x_norm})["preds"]
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
                n_surf += ds; n_vol += dv
        return finalize_split(mae_surf, mae_vol, n_surf, n_vol)

    print("\n=== Validation soup ===")
    val_metrics = {name: evaluate_split_one(loader) for name, loader in val_loaders.items()}
    val_avg = aggregate_splits(val_metrics)
    print(f"  val_avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for name, m in val_metrics.items():
        print(f"  {name:30s} surf_p={m['mae_surf_p']:.2f} vol_p={m['mae_vol_p']:.2f}")

    if not args.skip_test:
        print("\n=== Test soup ===")
        test_datasets = load_test_data(args.splits_dir, debug=False)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {name: evaluate_split_one(loader) for name, loader in test_loaders.items()}
        test_avg = aggregate_splits(test_metrics)
        print(f"  test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
        for name, m in test_metrics.items():
            print(f"  {name:30s} surf_p={m['mae_surf_p']:.2f} vol_p={m['mae_vol_p']:.2f}")


if __name__ == "__main__":
    main()
