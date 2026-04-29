"""Ensemble test evaluation: average predictions across multiple Transolver
checkpoints and report `test_avg/mae_surf_p` plus per-split MAEs.

Each checkpoint must have a sibling `config.yaml` (the file train.py writes
into ``models/model-<id>/``). All checkpoints must share the same
architecture (we only average preds in normalized space, so different archs
would still work — but mixing AMP / fp32 trained weights is fine because we
denormalize after averaging).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from contextlib import ExitStack

import torch
import yaml
from torch.utils.data import DataLoader

from data import (
    TEST_SPLIT_NAMES,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_test_data,
    pad_collate,
)
from test_eval import Transolver, evaluate_split, print_split_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='List of checkpoint paths. Each must have a sibling .yaml '
                         '(or pass --config_yamls explicitly).')
    ap.add_argument('--config_yamls', nargs='*', default=None)
    ap.add_argument('--splits_dir', default='/mnt/new-pvc/datasets/tandemfoil/splits_v2')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--surf_weight', type=float, default=10.0)
    ap.add_argument('--weights', nargs='*', type=float, default=None,
                    help='Optional weights for each checkpoint. Default = uniform.')
    args = ap.parse_args()

    cfg_paths = args.config_yamls
    if cfg_paths is None:
        cfg_paths = []
        for ckpt in args.checkpoints:
            yml = ckpt.replace('-checkpoint.pt', '-config.yaml').replace('checkpoint.pt', 'config.yaml').replace('checkpoint-final.pt', 'config.yaml')
            if not os.path.exists(yml):
                # fallback: same dir
                p = Path(ckpt).parent / 'config.yaml'
                yml = str(p)
            cfg_paths.append(yml)
    assert len(cfg_paths) == len(args.checkpoints)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load stats
    with open(Path(args.splits_dir) / 'stats.json') as f:
        raw = json.load(f)
    stats = {k: torch.tensor(raw[k], dtype=torch.float32, device=device)
             for k in ('x_mean', 'x_std', 'y_mean', 'y_std')}

    # Build models
    models = []
    for ckpt, yml in zip(args.checkpoints, cfg_paths):
        with open(yml) as f:
            mc = yaml.safe_load(f)
        m = Transolver(**mc).to(device).eval()
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        models.append(m)
        print(f'Loaded {ckpt}  arch={mc}')

    # Ensemble eval — same semantics as test_eval.evaluate_split but averages
    # predictions across models.
    test_datasets = load_test_data(args.splits_dir)
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in test_datasets.items()
    }

    weights = None
    if args.weights:
        if len(args.weights) != len(models):
            raise SystemExit(f'--weights count ({len(args.weights)}) != n models ({len(models)})')
        s = sum(args.weights)
        weights = [w / s for w in args.weights]
        print(f'Using weights: {weights}')

    test_metrics = {}
    for name, loader in test_loaders.items():
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        vol_loss_sum = surf_loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                y_finite_per_sample = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
                y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

                x_norm = (x - stats['x_mean']) / stats['x_std']
                y_norm = (y_safe - stats['y_mean']) / stats['y_std']

                # (Weighted) average predictions
                preds = []
                for m in models:
                    p = m({'x': x_norm})['preds']
                    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
                    preds.append(p)
                if weights is None:
                    pred = sum(preds) / len(preds)
                else:
                    pred = sum(w * p for w, p in zip(weights, preds))

                sq_err = (pred - y_norm) ** 2
                sample_keep = y_finite_per_sample.view(-1, 1).expand(-1, mask.shape[-1])
                vol_mask = mask & ~is_surface & sample_keep
                surf_mask = mask & is_surface & sample_keep
                vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)).item()
                surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)).item()
                n_batches += 1

                pred_orig = pred * stats['y_std'] + stats['y_mean']
                ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask & sample_keep,
                                          mae_surf, mae_vol)
                n_surf += ds; n_vol += dv

        vol_loss = vol_loss_sum / max(n_batches, 1)
        surf_loss = surf_loss_sum / max(n_batches, 1)
        out = {'vol_loss': vol_loss, 'surf_loss': surf_loss,
               'loss': vol_loss + args.surf_weight * surf_loss}
        out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
        test_metrics[name] = out

    test_avg = aggregate_splits(test_metrics)
    print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
    for name in TEST_SPLIT_NAMES:
        print_split_metrics(name, test_metrics[name])

    return test_avg, test_metrics


if __name__ == '__main__':
    main()
