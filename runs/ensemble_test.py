"""Ensemble multiple checkpoints by averaging predictions on test splits.
Uses the same NaN-safe evaluation as evaluate_split."""

from __future__ import annotations

import argparse
import importlib
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


def _import_train_module():
    """Import train.py without letting its top-level simple_parsing.parse run.

    train.py instantiates a Config via simple_parsing at module import; that
    triggers argparse on the *current* process's sys.argv. We temporarily
    swap sys.argv so train.py sees a clean argv, then restore it.
    """
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--debug", "--skip_test", "--max_minutes", "0.001"]
    try:
        # Importing train.py runs the trainer top-to-bottom (including wandb
        # init). We don't actually want that in this script, so we just pull
        # the model classes into our namespace via direct re-execution of
        # the relevant pieces. Use exec on a stripped subset.
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_module",
            Path(__file__).resolve().parent.parent / "train.py")
        # Read the source and only run the model-definition portion (up to
        # ``# Training`` section). This avoids running training inside import.
        src = (Path(__file__).resolve().parent.parent / "train.py").read_text()
        marker = "# Training"
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
COND_DIM = _t["COND_DIM"]
Transolver = _t["Transolver"]
extract_cond = _t["extract_cond"]


def load_model(model_dir: Path, device):
    config_path = model_dir / "config.yaml"
    ckpt_path = model_dir / "checkpoint.pt"
    if not config_path.exists() or not ckpt_path.exists():
        return None, None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = Transolver(**cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", type=Path, nargs="+", required=True,
                    help="One or more model checkpoint directories to ensemble")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {len(args.model_dirs)} checkpoints into ensemble...")
    models = []
    use_global_conds = []
    for d in args.model_dirs:
        m, cfg = load_model(d, device)
        if m is None:
            print(f"Skipped {d} (missing files)")
            continue
        models.append(m)
        use_global_conds.append(cfg.get("cond_dim", 0) > 0)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {d.name}: {n_params/1e6:.2f}M params, cond={use_global_conds[-1]}")

    if not models:
        print("No models loaded")
        return

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_datasets = load_test_data(args.splits_dir, debug=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    # ---------------- Per-model test ----------------
    print("\nPer-model test_avg/mae_surf_p:")
    per_model_results = []
    for mi, model in enumerate(models):
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        for name, loader in test_loaders.items():
            with torch.no_grad():
                for x, y, is_surface, mask in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    is_surface = is_surface.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    x_norm = (x - stats["x_mean"]) / stats["x_std"]
                    inputs = {"x": x_norm}
                    if use_global_conds[mi]:
                        inputs["cond"] = extract_cond(x_norm, mask)
                    pred = model(inputs)["preds"]
                    pred_orig = pred * stats["y_std"] + stats["y_mean"]
                    y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
                    y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
                    mask_safe = mask & y_finite.unsqueeze(-1)
                    accumulate_batch(pred_orig, y_safe, is_surface, mask_safe, mae_surf, mae_vol)
        s = mae_surf / max(int(mae_surf.sum().item() > 0) * (n_surf or 1) or 1, 1)
        # Better: recompute properly per split
        # (This is approximate; exact below)

    # ---------------- Ensemble test ----------------
    print("\nEnsemble (mean of normalised predictions) test_avg/mae_surf_p:")
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
                # Average predictions across ensemble
                pred_sum = None
                for mi, model in enumerate(models):
                    inputs = {"x": x_norm}
                    if use_global_conds[mi]:
                        inputs["cond"] = extract_cond(x_norm, mask)
                    p = model(inputs)["preds"]
                    pred_sum = p if pred_sum is None else pred_sum + p
                pred = pred_sum / len(models)
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
        print(f"  {name}: surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} "
              f"Uy={m['mae_surf_Uy']:.4f}]")

    test_avg = aggregate_splits(test_metrics)
    print(f"\nEnsemble TEST avg_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
    print("Per-channel/per-location:")
    for k, v in test_avg.items():
        print(f"  {k}: {v:.4f}")

    if args.out_json:
        out = {
            "ensemble_test_avg": dict(test_avg),
            "ensemble_test_per_split": {k: dict(v) for k, v in test_metrics.items()},
            "model_dirs": [str(d) for d in args.model_dirs],
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
