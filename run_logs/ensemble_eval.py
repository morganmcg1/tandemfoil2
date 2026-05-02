"""Evaluate an ensemble of best checkpoints by averaging predictions.

Loads N model checkpoints, averages their per-sample predictions on val/test,
and reports the resulting MAE per split + averaged. Supports both raw and EMA
checkpoints (whatever was saved by train.py as best). Uses the same NaN-safe
scoring path as train.py.

Usage:
  python ensemble_eval.py \
    --ckpts models/model-mlintern-pai2-72h-v4-r3-s6-best-6h-seed0-XXXX/checkpoint.pt \
            models/model-mlintern-pai2-72h-v4-r3-s6-best-6h-seed1-YYYY/checkpoint.pt \
    --model_config '{"n_layers":3,"n_hidden":128,"n_head":1,"slice_num":16,"mlp_ratio":2,"space_dim":2,"fun_dim":22,"out_dim":3,"output_fields":["Ux","Uy","p"],"output_dims":[1,1,1]}' \
    --out research/ensemble_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    TEST_SPLIT_NAMES, VAL_SPLIT_NAMES, X_DIM,
    aggregate_splits, finalize_split,
    load_data, load_test_data, pad_collate,
)
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", str(Path(__file__).parent.parent / "train.py"))


def load_train_module():
    """Lazy-import train.py code without triggering its top-level execution.

    train.py runs argparsing + W&B + training at module scope. We can't simply
    `import train`. Instead we read its source and exec only the class
    definitions (Transolver, FourierFeatures, etc.) and helper functions.
    """
    import ast
    src = (Path(__file__).parent.parent / "train.py").read_text()
    tree = ast.parse(src)
    keep = {"MLP", "PhysicsAttention", "TransolverBlock", "Transolver",
            "FourierFeatures", "_accumulate_batch_safe"}
    new_body = [n for n in tree.body if (
        (isinstance(n, ast.ClassDef) and n.name in keep) or
        (isinstance(n, ast.FunctionDef) and n.name in keep) or
        # Keep imports
        isinstance(n, (ast.Import, ast.ImportFrom)) or
        # ACTIVATION dict
        (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ACTIVATION" for t in n.targets
        ))
    )]
    new_tree = ast.Module(body=new_body, type_ignores=[])
    code = compile(new_tree, str(Path(__file__).parent.parent / "train.py"), "exec")
    ns = {"__name__": "_train_subset"}
    exec(code, ns)
    return ns


def eval_ensemble(models, loader, stats, device, _accumulate_batch_safe):
    """Average predictions from the list of models, accumulate MAE."""
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
            preds_norm = []
            for m in models:
                preds_norm.append(m({"x": x_norm})["preds"])
            pred_norm_avg = torch.stack(preds_norm, dim=0).mean(dim=0)
            pred_orig = pred_norm_avg * stats["y_std"] + stats["y_mean"]
            ds, dv = _accumulate_batch_safe(
                pred_orig, y, is_surface, mask, mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv
    out = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--model_config", default=None,
                   help='JSON string for shared Transolver kwargs. If omitted, '
                        'each checkpoint dir must contain a config.yaml.')
    p.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ns = load_train_module()
    Transolver = train_ns["Transolver"]
    _accumulate_batch_safe = train_ns["_accumulate_batch_safe"]

    import yaml
    shared_cfg = json.loads(args.model_config) if args.model_config else None

    models = []
    for ck in args.ckpts:
        ck_path = Path(ck)
        ck_dir = ck_path.parent
        if shared_cfg is not None:
            cfg = dict(shared_cfg)
        else:
            cfg_path = ck_dir / "config.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(f"No config.yaml in {ck_dir} and no --model_config")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        m = Transolver(**cfg).to(device)
        sd = torch.load(ck, map_location=device, weights_only=True)
        sd = {k: v for k, v in sd.items() if k != "n_averaged"}
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
        n_p = sum(p.numel() for p in m.parameters()) / 1e6
        n_layers = cfg.get("n_layers", "?")
        n_head = cfg.get("n_head", "?")
        slice_num = cfg.get("slice_num", "?")
        mlp_ratio = cfg.get("mlp_ratio", "?")
        print(f"Loaded {ck_dir.name}: {n_p:.2f}M (n_layers={n_layers}, n_head={n_head}, slice={slice_num}, mlp_ratio={mlp_ratio})")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_splits = load_test_data(args.splits_dir, debug=False)

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)

    val_metrics = {
        name: eval_ensemble(
            models,
            DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs),
            stats, device, _accumulate_batch_safe,
        )
        for name, ds in val_splits.items()
    }
    test_metrics = {
        name: eval_ensemble(
            models,
            DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs),
            stats, device, _accumulate_batch_safe,
        )
        for name, ds in test_splits.items()
    }

    val_avg = aggregate_splits(val_metrics)
    test_avg = aggregate_splits(test_metrics)

    print("\n=== ENSEMBLE VAL ===")
    for k, v in val_avg.items():
        print(f"  val/{k}: {v:.6f}")
    print("\n=== ENSEMBLE TEST ===")
    for k, v in test_avg.items():
        print(f"  test/{k}: {v:.6f}")

    print("\n=== Per-split ===")
    for name, m in val_metrics.items():
        print(f"  val/{name}/mae_surf_p = {m['mae_surf_p']:.4f}")
    for name, m in test_metrics.items():
        print(f"  test/{name}/mae_surf_p = {m['mae_surf_p']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ckpts": args.ckpts,
            "model_config": cfg,
            "val_avg": val_avg,
            "test_avg": test_avg,
            "val_per_split": val_metrics,
            "test_per_split": test_metrics,
        }, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
