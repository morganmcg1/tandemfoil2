"""Ensemble eval: load N model checkpoints and average their predictions.

Usage: python research/ensemble_eval.py models/model-A models/model-B [...]

For each test split, runs all models in sequence and averages the predicted
fields BEFORE accumulating MAE. The ensemble prediction is in normalized
space, then denormalized as usual.
"""
import sys
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/ml-intern-benchmark/target")
from data import load_test_data, accumulate_batch, aggregate_splits, finalize_split, pad_collate
from data.loader import _load_stats


def load_transolver(model_dir: str, device):
    md = Path(model_dir)
    cfg_yaml = md / "config.yaml"
    ckpt = md / "checkpoint.pt"
    model_cfg = yaml.safe_load(cfg_yaml.read_text())

    src = Path("/workspace/ml-intern-benchmark/target/train.py").read_text()
    end_marker = "# Evaluation helpers"
    src_top = src[:src.find(end_marker)]
    exec_globals = {"__name__": "_no_main_train"}
    exec(compile(src_top, "train_classes.py", "exec"), exec_globals)
    Transolver = exec_globals["Transolver"]
    model = Transolver(**model_cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main(model_dirs: list[str]):
    device = torch.device("cuda")
    print(f"Loading {len(model_dirs)} models...")
    models = [load_transolver(d, device) for d in model_dirs]
    for d, m in zip(model_dirs, models):
        np = sum(p.numel() for p in m.parameters())/1e6
        print(f"  {d}: {np:.2f}M params")

    splits_dir = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    test_splits = load_test_data(splits_dir)

    stats_raw = _load_stats(Path(splits_dir))
    stats = {k: v.to(device) for k, v in stats_raw.items()}

    test_metrics = {}
    for name, ds in test_splits.items():
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=pad_collate, num_workers=4)
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                if not torch.isfinite(y).all():
                    B = y.shape[0]
                    sf = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
                    bad = ~sf
                    if bad.any():
                        mask = mask & ~bad[:, None]
                    y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                preds = []
                for m in models:
                    p = m({"x": x_norm})["preds"]
                    preds.append(p)
                pred = torch.stack(preds, dim=0).mean(dim=0)
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                ds_n, dv_n = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
                n_surf += ds_n
                n_vol += dv_n
        m_dict = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
        test_metrics[name] = m_dict
        print(f"  {name:30s} surf[p={m_dict['mae_surf_p']:.4f}]")

    avg = aggregate_splits(test_metrics)
    print(f"\nENSEMBLE TEST avg_surf_p={avg['avg/mae_surf_p']:.4f}")
    return test_metrics, avg


if __name__ == "__main__":
    main(sys.argv[1:])
