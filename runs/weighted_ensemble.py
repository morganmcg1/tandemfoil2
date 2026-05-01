"""Weighted ensemble of multiple checkpoints.
Weights are inverse to val_avg/mae_surf_p (lower val = higher weight)."""

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
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)


def _import_train_module():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--debug", "--skip_test", "--max_minutes", "0.001"]
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_module",
            Path(__file__).resolve().parent.parent / "train.py")
        src = (Path(__file__).resolve().parent.parent / "train.py").read_text()
        cutoff = src.find("# ---------------------------------------------------------------------------\n# Training")
        if cutoff < 0:
            raise RuntimeError("Could not find Training section marker in train.py")
        head = src[:cutoff]
        ns: dict = {"__name__": "train_module"}
        exec(compile(head, str(spec.origin), "exec"), ns)
        return ns
    finally:
        sys.argv = saved_argv


_t = _import_train_module()
Transolver = _t["Transolver"]
extract_cond = _t["extract_cond"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Per-model weight (any positives, normalised). Default uniform.")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.weights is None:
        args.weights = [1.0] * len(args.model_dirs)
    if len(args.weights) != len(args.model_dirs):
        raise ValueError("len(weights) must equal len(model_dirs)")

    w_total = sum(args.weights)
    w = [x / w_total for x in args.weights]
    print(f"Loading {len(args.model_dirs)} checkpoints, weights={[f'{x:.4f}' for x in w]}")

    models = []
    use_global_conds = []
    for d in args.model_dirs:
        with open(d / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        m = Transolver(**cfg).to(device)
        state = torch.load(d / "checkpoint.pt", map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        use_global_conds.append(cfg.get("cond_dim", 0) > 0)
        print(f"  {d.name}: cond={use_global_conds[-1]}")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
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
                pred_sum = None
                for mi, model in enumerate(models):
                    inputs = {"x": x_norm}
                    if use_global_conds[mi]:
                        inputs["cond"] = extract_cond(x_norm, mask)
                    p = model(inputs)["preds"] * w[mi]
                    pred_sum = p if pred_sum is None else pred_sum + p
                pred = pred_sum
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
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
    print(f"\nWeighted ensemble TEST avg_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")

    if args.out_json:
        out = {
            "weights": list(args.weights),
            "ensemble_test_avg": dict(test_avg),
            "ensemble_test_per_split": {k: dict(v) for k, v in test_metrics.items()},
            "model_dirs": [str(d) for d in args.model_dirs],
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
