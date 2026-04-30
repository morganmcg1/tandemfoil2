"""Re-evaluate a saved checkpoint on the test splits using the fixed
NaN-safe evaluation. Useful to recover test_avg/mae_surf_p for runs that
were finished before the NaN-propagation fix landed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    TEST_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)
from train import COND_DIM, Transolver, extract_cond


def reeval(model_dir: Path, splits_dir: str, batch_size: int = 4):
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "checkpoint.pt"
    if not config_path.exists() or not ckpt_path.exists():
        print(f"missing checkpoint files in {model_dir}")
        return None

    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    use_global_cond = model_cfg.get("cond_dim", 0) > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_splits, stats, sample_weights = load_data(splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    model = Transolver(**model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ckpt_path} ({n_params/1e6:.2f}M params)")
    print(f"Config: use_eidetic={model_cfg.get('use_eidetic', False)} "
          f"cond_dim={model_cfg.get('cond_dim', 0)} n_hidden={model_cfg['n_hidden']} "
          f"n_layers={model_cfg['n_layers']} slice_num={model_cfg['slice_num']}")

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    test_datasets = load_test_data(splits_dir, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    test_metrics = {}
    for name, loader in test_loaders.items():
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
                inputs = {"x": x_norm}
                if use_global_cond:
                    inputs["cond"] = extract_cond(x_norm, mask)
                pred = model(inputs)["preds"]
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                # NaN-safe: zero out non-finite y, exclude bad samples
                y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
                y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
                mask_safe = mask & y_finite.unsqueeze(-1)
                ds_, dv_ = accumulate_batch(pred_orig, y_safe, is_surface, mask_safe,
                                            mae_surf, mae_vol)
                n_surf += ds_
                n_vol += dv_
        m = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
        test_metrics[name] = m

    test_avg = aggregate_splits(test_metrics)
    print(f"\nTEST avg_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        m = test_metrics[name]
        print(f"    {name:<26s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} "
              f"Uy={m['mae_surf_Uy']:.4f}]  "
              f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]")

    return {"test_metrics": test_metrics, "test_avg": test_avg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    result = reeval(args.model_dir, args.splits_dir, args.batch_size)
    if args.out_json and result:
        with open(args.out_json, "w") as f:
            json.dump({k: dict(v) if isinstance(v, dict) else v for k, v in result["test_avg"].items()},
                      f, indent=2)
