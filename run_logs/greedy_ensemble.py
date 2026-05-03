"""Greedy forward selection for ensemble: starting from best single, add the
model that most reduces val MAE at each step. Stops when adding a model no
longer improves val MAE.

This caches per-model predictions on val/test once, then evaluates ensembles
by computing means + accumulating the same NaN-safe metric used by train.py.
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
    TEST_SPLIT_NAMES, VAL_SPLIT_NAMES,
    aggregate_splits, finalize_split,
    load_data, load_test_data, pad_collate,
)


def load_train_module():
    import ast
    src = (Path(__file__).parent.parent / "train.py").read_text()
    tree = ast.parse(src)
    keep = {"MLP", "PhysicsAttention", "TransolverBlock", "Transolver",
            "FourierFeatures", "_accumulate_batch_safe"}
    new_body = [n for n in tree.body if (
        (isinstance(n, ast.ClassDef) and n.name in keep) or
        (isinstance(n, ast.FunctionDef) and n.name in keep) or
        isinstance(n, (ast.Import, ast.ImportFrom)) or
        (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ACTIVATION" for t in n.targets
        ))
    )]
    new_tree = ast.Module(body=new_body, type_ignores=[])
    code = compile(new_tree, str(Path(__file__).parent.parent / "train.py"), "exec")
    ns = {"__name__": "_train_subset"}
    exec(code, ns)
    return ns


@torch.no_grad()
def get_normalized_preds(model, loader, stats, device):
    """Run model and return list of normalized predictions per batch."""
    out = []
    for x, y, is_surface, mask in loader:
        x = x.to(device, non_blocking=True)
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        pred = model({"x": x_norm})["preds"]
        out.append(pred.detach().cpu())
    return out


def aggregate_split_with_preds(preds_per_model, loader, stats, device, _accumulate_batch_safe):
    """preds_per_model: list[N_models][N_batches] tensor in normalized space.

    Average the predictions across models, denormalize, accumulate MAE.
    """
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    n_batches = len(preds_per_model[0])
    for batch_idx in range(n_batches):
        # Average normalized preds across models
        stacked = torch.stack(
            [preds_per_model[i][batch_idx] for i in range(len(preds_per_model))],
            dim=0
        )
        pred_norm_avg = stacked.mean(dim=0).to(device)
        # Need the corresponding y, is_surface, mask from loader
        # Re-iterate loader once for this batch (lazy approach)
        pass
    # simpler: re-iterate loader, recompute averaging
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Pool of N checkpoints to greedy-select from")
    p.add_argument("--max_K", type=int, default=20,
                   help="Stop after this many models added")
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
    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_splits = load_test_data(args.splits_dir, debug=False)

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=2, pin_memory=True)
    val_loaders = {n: DataLoader(d, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                   for n, d in val_splits.items()}
    test_loaders = {n: DataLoader(d, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
                    for n, d in test_splits.items()}

    # Cache per-model predictions on each val + test split (kept on CPU)
    print(f"\nCaching predictions for {len(args.ckpts)} models...")
    cached_val = {n: [] for n in val_splits}
    cached_test = {n: [] for n in test_splits}
    cached_y = {}
    cached_is_surface = {}
    cached_mask = {}

    # First pass: cache the labels (same across models)
    print("Caching labels...")
    for n, loader in val_loaders.items():
        cached_y.setdefault(("val", n), [])
        cached_is_surface.setdefault(("val", n), [])
        cached_mask.setdefault(("val", n), [])
        for x, y, is_surface, mask in loader:
            cached_y[("val", n)].append(y)
            cached_is_surface[("val", n)].append(is_surface)
            cached_mask[("val", n)].append(mask)
    for n, loader in test_loaders.items():
        cached_y.setdefault(("test", n), [])
        cached_is_surface.setdefault(("test", n), [])
        cached_mask.setdefault(("test", n), [])
        for x, y, is_surface, mask in loader:
            cached_y[("test", n)].append(y)
            cached_is_surface[("test", n)].append(is_surface)
            cached_mask[("test", n)].append(mask)
    print("Labels cached")

    # Per-model predictions
    for ck in args.ckpts:
        ck_dir = Path(ck).parent
        cfg_path = ck_dir / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        m = Transolver(**cfg).to(device)
        sd = torch.load(ck, map_location=device, weights_only=True)
        sd = {k: v for k, v in sd.items() if k != "n_averaged"}
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        m.load_state_dict(sd)
        m.eval()
        # Compute predictions on each split
        for split_loaders, dst in [(val_loaders, cached_val), (test_loaders, cached_test)]:
            for split_name, loader in split_loaders.items():
                batch_preds = []
                for x, y, is_surface, mask in loader:
                    x = x.to(device, non_blocking=True)
                    x_norm = (x - stats["x_mean"]) / stats["x_std"]
                    pred = m({"x": x_norm})["preds"].detach().cpu()
                    batch_preds.append(pred)
                dst[split_name].append(batch_preds)
        del m
        torch.cuda.empty_cache()
        print(f"  cached {ck_dir.name}")

    # Helper: compute val MAE for a list of model indices.
    # Uses correct per-split-mean aggregation matching aggregate_splits().
    def ensemble_metric(model_indices, on_test=False):
        cache = cached_test if on_test else cached_val
        labels_split = "test" if on_test else "val"
        per_split_p = []
        for split_name in cache:
            mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
            mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
            n_surf = n_vol = 0
            for batch_idx in range(len(cache[split_name][0])):
                stacked = torch.stack(
                    [cache[split_name][i][batch_idx] for i in model_indices],
                    dim=0
                )
                pred_norm_avg = stacked.mean(dim=0).to(device)
                pred_orig = pred_norm_avg * stats["y_std"] + stats["y_mean"]
                y = cached_y[(labels_split, split_name)][batch_idx].to(device)
                is_surface = cached_is_surface[(labels_split, split_name)][batch_idx].to(device)
                mask = cached_mask[(labels_split, split_name)][batch_idx].to(device)
                ds, dv = _accumulate_batch_safe(
                    pred_orig, y, is_surface, mask, mae_surf, mae_vol
                )
                n_surf += ds
                n_vol += dv
            s = mae_surf / max(n_surf, 1)
            per_split_p.append(float(s[2].item()))
        # Mean across splits — same as aggregate_splits in data/scoring.py
        return sum(per_split_p) / len(per_split_p)

    # Greedy selection
    pool = list(range(len(args.ckpts)))
    selected = []
    history = []

    # Start with single best on val
    best_idx, best_val = None, float("inf")
    for i in pool:
        v = ensemble_metric([i])
        if v < best_val:
            best_idx, best_val = i, v
    selected.append(best_idx)
    pool.remove(best_idx)
    test_init = ensemble_metric(selected, on_test=True)
    history.append({"K": 1, "added_idx": best_idx, "added_ckpt": args.ckpts[best_idx],
                    "val": best_val, "test": test_init})
    print(f"\n[1/{args.max_K}] start: idx={best_idx} ({Path(args.ckpts[best_idx]).parent.name}) val={best_val:.4f} test={test_init:.4f}")

    while len(selected) < args.max_K and pool:
        best_add, best_v = None, float("inf")
        for i in pool:
            v = ensemble_metric(selected + [i])
            if v < best_v:
                best_v, best_add = v, i
        # Always add the best candidate, even if it doesn't improve val.
        # Stops only when pool is exhausted or max_K is reached.
        selected.append(best_add)
        pool.remove(best_add)
        best_val = best_v
        test_at_K = ensemble_metric(selected, on_test=True)
        improvement = " *" if best_val < history[-1]["val"] else ""
        history.append({"K": len(selected), "added_idx": best_add, "added_ckpt": args.ckpts[best_add],
                        "val": best_val, "test": test_at_K})
        print(f"[{len(selected)}/{args.max_K}] add idx={best_add} ({Path(args.ckpts[best_add]).parent.name}) val={best_val:.4f} test={test_at_K:.4f}{improvement}")

    final_val = ensemble_metric(selected)
    final_test = ensemble_metric(selected, on_test=True)
    print(f"\nFinal greedy ensemble (K={len(selected)}): val={final_val:.4f}, test={final_test:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "selected_indices": selected,
            "selected_ckpts": [args.ckpts[i] for i in selected],
            "val_avg_mae_surf_p": final_val,
            "test_avg_mae_surf_p": final_test,
            "history": history,
        }, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
