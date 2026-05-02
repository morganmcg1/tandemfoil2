"""Automatically discover all completed runs and run the best ensemble.

Reads MLINTERN_RESULTS.jsonl, filters TESTED runs above a quality threshold,
sorts by val MAE, and runs ensemble_eval over the top N. Useful between sweeps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="research/MLINTERN_RESULTS.jsonl")
    p.add_argument("--top", type=int, default=20,
                   help="Take top N by val MAE (default 20)")
    p.add_argument("--max_test", type=float, default=None,
                   help="Drop runs with test MAE above this (default no filter)")
    p.add_argument("--max_val", type=float, default=None,
                   help="Drop runs with val MAE above this (default no filter)")
    p.add_argument("--include_prefix", default="s",
                   help="Only include runs whose name starts with this (default 's')")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    rows = []
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "TESTED":
                continue
            if not r.get("name", "").startswith(args.include_prefix):
                continue
            v = r.get("best_val_mae_surf_p")
            t = r.get("test_avg_mae_surf_p")
            wid = r.get("wandb_run_id")
            if v is None or wid is None:
                continue
            if args.max_test is not None and (t is None or t > args.max_test):
                continue
            if args.max_val is not None and v > args.max_val:
                continue
            ckpt = Path(f"models/model-{wid}/checkpoint.pt")
            if not ckpt.exists():
                continue
            rows.append((v, t, r["name"], str(ckpt)))

    rows.sort(key=lambda x: x[0])
    rows = rows[: args.top]
    if not rows:
        print("No matching runs found", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(rows)} runs (top {args.top} by val MAE):")
    for v, t, n, c in rows:
        print(f"  {n:<35s} val={v:7.3f} test={t!s:>7s}  ({c})")

    ckpts = [c for _, _, _, c in rows]
    cmd = ["python", "run_logs/ensemble_eval.py",
           "--ckpts", *ckpts,
           "--batch_size", "4",
           "--out", args.out]
    print(f"\nCommand: CUDA_VISIBLE_DEVICES={args.gpu} " + " ".join(cmd))
    if args.dry_run:
        return
    import os
    env = dict(**os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    subprocess.run(cmd, env=env, check=True)
    with open(args.out) as f:
        d = json.load(f)
    print(f"\n=== ENSEMBLE RESULTS ({args.out}) ===")
    print(f"  val/avg/mae_surf_p:  {d['val_avg']['avg/mae_surf_p']:.4f}")
    print(f"  test/avg/mae_surf_p: {d['test_avg']['avg/mae_surf_p']:.4f}")


if __name__ == "__main__":
    main()
