#!/usr/bin/env python3
"""Extract structured results from training log files.

Outputs JSON-lines with: name, status, best_val_avg_mae_surf_p, best_epoch,
total_epochs, n_params, test_avg_mae_surf_p (if available), test_per_split (dict).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def parse_log(path: str) -> dict:
    name = os.path.basename(path).replace(".log", "")
    out = {
        "name": name,
        "log_path": path,
        "status": "RUN",
        "best_val_mae_surf_p": None,
        "best_epoch": None,
        "total_epochs": 0,
        "n_params_M": None,
        "test_avg_mae_surf_p": None,
        "test_avg_mae_surf_Ux": None,
        "test_avg_mae_surf_Uy": None,
        "test_per_split": {},
        "wandb_run_id": None,
    }
    with open(path, "rb") as f:
        # Read raw bytes, replace \r -> \n
        data = f.read().replace(b"\r", b"\n").decode("utf-8", errors="replace")

    # Status
    if "Training done" in data:
        out["status"] = "DONE"
    if "TEST  avg_surf_p=" in data:
        out["status"] = "TESTED"
    if any(s in data for s in ["Traceback", "OutOfMemoryError"]):
        out["status"] = "ERROR"

    # n_params
    m = re.search(r"Transolver \(([\d.]+)M params\)", data)
    if m:
        out["n_params_M"] = float(m.group(1))

    # Per-epoch best (lines like "Epoch 7 (56s) ... val_avg_surf_p=158.3677 *")
    # Best is marked with " *" at end
    best = float("inf")
    best_epoch = None
    total = 0
    for line in data.splitlines():
        # Allow optional " *" or " * [EMA]" markers at end
        m = re.match(r"^Epoch +(\d+) +\(.*val_avg_surf_p=([\d.]+)", line.strip())
        if m:
            ep = int(m.group(1))
            val = float(m.group(2))
            total = max(total, ep)
            if val < best:
                best = val
                best_epoch = ep
    if best_epoch is not None:
        out["best_val_mae_surf_p"] = best
        out["best_epoch"] = best_epoch
        out["total_epochs"] = total

    # test_avg
    m = re.search(r"TEST  avg_surf_p=([\d.]+)", data)
    if m:
        out["test_avg_mae_surf_p"] = float(m.group(1))
    # per-split test
    for sname in ["test_single_in_dist", "test_geom_camber_rc",
                  "test_geom_camber_cruise", "test_re_rand"]:
        m = re.search(rf"{sname}\s+loss=[\d.]+\s+surf\[p=([\d.]+) Ux=([\d.]+) Uy=([\d.]+)\]\s+vol\[p=([\d.]+) Ux=([\d.]+) Uy=([\d.]+)\]", data)
        if m:
            out["test_per_split"][sname] = {
                "mae_surf_p": float(m.group(1)),
                "mae_surf_Ux": float(m.group(2)),
                "mae_surf_Uy": float(m.group(3)),
                "mae_vol_p": float(m.group(4)),
                "mae_vol_Ux": float(m.group(5)),
                "mae_vol_Uy": float(m.group(6)),
            }

    # wandb run id — match the directory naming pattern run-YYYYMMDD_HHMMSS-<id>
    m = re.search(r"Run data is saved locally in .*?run-\d{8}_\d{6}-([a-z0-9]+)", data)
    if m:
        out["wandb_run_id"] = m.group(1)
    else:
        m = re.search(r"senpai-v1-ml-intern/runs/([a-z0-9]+)", data)
        if m:
            out["wandb_run_id"] = m.group(1)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", default=None)
    p.add_argument("--summary", action="store_true", help="Print summary table")
    args = p.parse_args()
    paths = args.logs if args.logs else sorted(glob.glob("run_logs/*.log"))
    rows = [parse_log(p) for p in paths if os.path.isfile(p)]

    if args.summary:
        print(f"{'name':<35s} {'status':<8s} {'#par':>5s} {'epoch':>5s} {'best_val':>9s} {'test':>9s}")
        for r in sorted(rows, key=lambda r: (r.get("best_val_mae_surf_p") or 1e9)):
            tval = r["test_avg_mae_surf_p"]
            ts = f"{tval:9.3f}" if tval is not None else "        -"
            bval = r["best_val_mae_surf_p"]
            bs = f"{bval:9.3f}" if bval is not None else "        -"
            np = r["n_params_M"]
            ns = f"{np:5.2f}" if np is not None else "    -"
            print(f"{r['name']:<35s} {r['status']:<8s} {ns:>5s} {str(r['best_epoch'] or '-'):>5s} {bs:>9s} {ts:>9s}")
        return

    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
