"""Compile per-run results into research/MLINTERN_RESULTS.jsonl.

Pulls W&B runs in the mlintern-pai2-r1 group and writes one JSON object per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import wandb


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    return api.runs(
        path=f"{entity}/{project}",
        filters={"group": group},
        order="-created_at",
    )


def best_val_history(run) -> dict:
    """Get final best val_avg/mae_surf_p value and per-split values from history."""
    out = {"val_history_min": None}
    try:
        hist = run.history(keys=["val_avg/mae_surf_p"], pandas=False, samples=10000)
        vals = [h.get("val_avg/mae_surf_p") for h in hist if h.get("val_avg/mae_surf_p") is not None]
        if vals:
            out["val_history_min"] = min(vals)
            out["val_history_count"] = len(vals)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "senpai-v1-ml-intern"))
    ap.add_argument("--group", default="mlintern-pai2-r1")
    ap.add_argument("--out", default="research/MLINTERN_RESULTS.jsonl")
    ap.add_argument("--include-history", action="store_true", help="Also pull min from history")
    args = ap.parse_args()

    runs = fetch_runs(args.entity, args.project, args.group)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in runs:
        s = r.summary._json_dict if hasattr(r.summary, "_json_dict") else dict(r.summary)
        cfg = dict(r.config or {})
        row = {
            "run_id": r.id,
            "run_name": r.name,
            "state": r.state,
            "url": r.url,
            "created_at": getattr(r, "created_at", None) and str(r.created_at),
            "best_epoch": s.get("best_epoch"),
            "best_val_avg_mae_surf_p": s.get("best_val_avg/mae_surf_p"),
            "test_avg_mae_surf_p": s.get("test_avg/mae_surf_p"),
            "test_per_split_mae_surf_p": {
                "test_single_in_dist": s.get("test/test_single_in_dist/mae_surf_p"),
                "test_geom_camber_rc": s.get("test/test_geom_camber_rc/mae_surf_p"),
                "test_geom_camber_cruise": s.get("test/test_geom_camber_cruise/mae_surf_p"),
                "test_re_rand": s.get("test/test_re_rand/mae_surf_p"),
            },
            "n_params": s.get("n_params") or cfg.get("n_params"),
            "config": {
                "epochs": cfg.get("epochs"),
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
            },
            "total_train_minutes": s.get("total_train_minutes"),
            "_runtime_s": s.get("_runtime"),
        }
        if args.include_history:
            row.update(best_val_history(r))
        rows.append(row)

    # Sort by best_val_avg_mae_surf_p (None last)
    def k(r):
        v = r.get("best_val_avg_mae_surf_p")
        return (v is None, v if v is not None else float("inf"))
    rows.sort(key=k)

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
