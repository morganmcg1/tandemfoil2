"""Re-evaluate a checkpoint on test splits with NaN-safe handling.

Usage: python research/reeval_test.py <model_dir>
"""
import sys
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/ml-intern-benchmark/target")
from data import load_test_data, accumulate_batch, aggregate_splits, finalize_split, pad_collate
from data.loader import _load_stats

import importlib.util
# Import Transolver class from train.py
spec = importlib.util.spec_from_file_location("train", "/workspace/ml-intern-benchmark/target/train.py")


def main(model_dir: str):
    md = Path(model_dir)
    cfg_yaml = md / "config.yaml"
    ckpt = md / "checkpoint.pt"
    if not (cfg_yaml.exists() and ckpt.exists()):
        print(f"Missing files in {md}")
        return

    model_cfg = yaml.safe_load(cfg_yaml.read_text())
    print(f"Model config: {model_cfg}")

    # Just import Transolver class directly
    sys.path.insert(0, "/workspace/ml-intern-benchmark/target")
    # Hack: avoid running train.py as script
    import types
    train_mod = types.ModuleType("train_module")
    exec_globals = {"__name__": "_no_main_train"}
    # Read class definitions only (lines 1-216 contain Transolver class)
    src = Path("/workspace/ml-intern-benchmark/target/train.py").read_text()
    # Take everything up through the end of class Transolver (search for end token)
    end_marker = "# Evaluation helpers"
    idx = src.find(end_marker)
    src_top = src[:idx]
    exec(compile(src_top, "train_classes.py", "exec"), exec_globals)
    Transolver = exec_globals["Transolver"]

    device = torch.device("cuda")
    model = Transolver(**model_cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    splits_dir = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    test_splits = load_test_data(splits_dir)

    stats_raw = _load_stats(Path(splits_dir))
    stats = {k: v.to(device) for k, v in stats_raw.items()}

    test_metrics = {}
    for name, ds in test_splits.items():
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=pad_collate, num_workers=4)
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # NaN-safe
                if not torch.isfinite(y).all():
                    B = y.shape[0]
                    sf = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
                    bad = ~sf
                    if bad.any():
                        mask = mask & ~bad[:, None]
                    y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                pred = model({"x": x_norm})["preds"]
                pred_orig = pred * stats["y_std"] + stats["y_mean"]

                ds_n, dv_n = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
                n_surf += ds_n
                n_vol += dv_n
        m = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
        test_metrics[name] = m
        print(f"  {name:30s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  vol[p={m['mae_vol_p']:.4f}]")

    avg = aggregate_splits(test_metrics)
    print(f"\n  TEST  avg_surf_p={avg['avg/mae_surf_p']:.4f}")
    return test_metrics, avg


if __name__ == "__main__":
    main(sys.argv[1])
