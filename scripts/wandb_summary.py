"""Pull current best val_avg/mae_surf_p from each run in mlintern-pai2-r1 W&B group.

Usage:
  python scripts/wandb_summary.py [--group mlintern-pai2-r1] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import wandb


def fetch_runs(entity: str, project: str, group: str, limit: int = 50):
    api = wandb.Api()
    runs = api.runs(
        path=f"{entity}/{project}",
        filters={"group": group},
        order="-created_at",
    )
    out = []
    for i, r in enumerate(runs):
        if i >= limit:
            break
        s = r.summary._json_dict if hasattr(r.summary, "_json_dict") else dict(r.summary)
        cfg = dict(r.config or {})
        out.append({
            "id": r.id,
            "name": r.name,
            "state": r.state,
            "best_epoch": s.get("best_epoch"),
            "best_val_avg/mae_surf_p": s.get("best_val_avg/mae_surf_p"),
            "test_avg/mae_surf_p": s.get("test_avg/mae_surf_p"),
            "test/test_single_in_dist/mae_surf_p": s.get("test/test_single_in_dist/mae_surf_p"),
            "test/test_geom_camber_rc/mae_surf_p": s.get("test/test_geom_camber_rc/mae_surf_p"),
            "test/test_geom_camber_cruise/mae_surf_p": s.get("test/test_geom_camber_cruise/mae_surf_p"),
            "test/test_re_rand/mae_surf_p": s.get("test/test_re_rand/mae_surf_p"),
            "n_params": s.get("n_params") or cfg.get("n_params"),
            "epochs_configured": cfg.get("epochs"),
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "batch_size": cfg.get("batch_size"),
            "surf_weight": cfg.get("surf_weight"),
            "ema_decay": cfg.get("ema_decay"),
            "amp_dtype": cfg.get("amp_dtype"),
            "n_layers": cfg.get("n_layers"),
            "n_hidden": cfg.get("n_hidden"),
            "n_head": cfg.get("n_head"),
            "slice_num": cfg.get("slice_num"),
            "mlp_ratio": cfg.get("mlp_ratio"),
            "p_loss_weight": cfg.get("p_loss_weight"),
            "grad_clip": cfg.get("grad_clip"),
            "warmup_frac": cfg.get("warmup_frac"),
            "_runtime": s.get("_runtime"),
            "url": r.url,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "senpai-v1-ml-intern"))
    ap.add_argument("--group", default="mlintern-pai2-r1")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--jsonl", action="store_true", help="Output JSONL instead of pretty table")
    args = ap.parse_args()

    runs = fetch_runs(args.entity, args.project, args.group, args.limit)

    if args.jsonl:
        for r in runs:
            print(json.dumps(r))
        return

    # Sort by best_val_avg/mae_surf_p (None at end)
    def key(r):
        v = r.get("best_val_avg/mae_surf_p")
        return (v is None, v if v is not None else float("inf"))

    runs.sort(key=key)
    print(f"Group: {args.group}  |  {len(runs)} runs")
    print(f"{'name':<40s} {'state':<8s} {'best_val':<10s} {'test_avg':<10s} {'best_ep':<8s} {'arch':<24s} {'misc':<32s}")
    print("-" * 140)
    for r in runs:
        name = (r.get("name") or r["id"])[:40]
        state = (r.get("state") or "?")[:8]
        bv = r.get("best_val_avg/mae_surf_p")
        ta = r.get("test_avg/mae_surf_p")
        be = r.get("best_epoch")
        arch = (
            f"L{r.get('n_layers')}/W{r.get('n_hidden')}/H{r.get('n_head')}/"
            f"S{r.get('slice_num')}"
        )[:24]
        misc = (
            f"sw={r.get('surf_weight')} pw={r.get('p_loss_weight')} "
            f"ema={r.get('ema_decay')} amp={r.get('amp_dtype')}"
        )[:32]
        print(
            f"{name:<40s} {state:<8s} "
            f"{(f'{bv:.3f}' if bv is not None else '—'):<10s} "
            f"{(f'{ta:.3f}' if ta is not None else '—'):<10s} "
            f"{(str(be) if be is not None else '—'):<8s} "
            f"{arch:<24s} {misc:<32s}"
        )


if __name__ == "__main__":
    main()
