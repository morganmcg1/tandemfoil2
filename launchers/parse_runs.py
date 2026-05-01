#!/usr/bin/env python3
"""Parse all run logs and emit a JSONL summary + a markdown leaderboard.

Each log is parsed for:
  - Final per-epoch summary lines
  - Best `val_avg/mae_surf_p` and the epoch achieved
  - Per-split MAE at the best epoch
  - End-of-run TEST split metrics (if present)
  - Run config from W&B-printed banner lines (loss, arch)
  - W&B run id (from `Run data is saved locally in ...`)

Usage:
  python launchers/parse_runs.py [logs_dir1 logs_dir2 ...] > out.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\((\d+)s\)\s+\[([\d.]+)GB\]\s+train\[vol=([\d.eE+-]+)\s+surf=([\d.eE+-]+)\]\s+val_avg_surf_p=([\d.eE+-]+)(\s+\*)?",
    re.MULTILINE,
)
SPLIT_RE = re.compile(
    r"^\s+(val_\w+|test_\w+)\s+loss=([\d.eE+-]+)\s+surf\[p=([\d.eE+-]+)\s+Ux=([\d.eE+-]+)\s+Uy=([\d.eE+-]+)\]\s+vol\[p=([\d.eE+-]+)\s+Ux=([\d.eE+-]+)\s+Uy=([\d.eE+-]+)\]",
    re.MULTILINE,
)
BANNER_RE = re.compile(
    r"Model: Transolver \(([\d.]+)M params\)\s+cfg.n_hidden=(\d+)\s+n_layers=(\d+)\s+n_head=(\d+)\s+slice_num=(\d+)\s+mlp_ratio=(\d+)"
)
LOSS_RE = re.compile(
    r"Loss: (\w+) \(huber_beta=([\d.eE+-]+)\)\s+surf_weight=([\d.eE+-]+)\s+p_extra_weight=([\d.eE+-]+)"
)
OPTIM_RE = re.compile(
    r"Optim: (\w+)\s+lr=([\d.eE+-]+)\s+min_lr=([\d.eE+-]+)\s+wd=([\d.eE+-]+)\s+warmup=(\d+)\s+grad_clip=([\d.eE+-]+)\s+bs=(\d+)"
)
RUNID_RE = re.compile(r"Run data is saved locally in .*?run-\d+_\d+-(\w+)")
TEST_AVG_RE = re.compile(r"TEST\s+avg_surf_p=([\d.eE+-]+)")
BEST_VAL_RE = re.compile(r"Best val: epoch (\d+), val_avg/mae_surf_p = ([\d.eE+-]+)")
TIMEOUT_RE = re.compile(r"Timeout \(([\d.]+) min\)\. Stopping\.")
DONE_RE = re.compile(r"Training done in ([\d.]+) min")


def parse_one(path: Path) -> dict | None:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return None
    if not text:
        return None

    out = {"name": path.stem, "log_path": str(path)}

    # Architecture
    m = BANNER_RE.search(text)
    if m:
        out["params_M"] = float(m.group(1))
        out["n_hidden"] = int(m.group(2))
        out["n_layers"] = int(m.group(3))
        out["n_head"] = int(m.group(4))
        out["slice_num"] = int(m.group(5))
        out["mlp_ratio"] = int(m.group(6))

    m = LOSS_RE.search(text)
    if m:
        out["loss_type"] = m.group(1)
        out["huber_beta"] = float(m.group(2))
        out["surf_weight"] = float(m.group(3))
        out["p_extra_weight"] = float(m.group(4))

    m = OPTIM_RE.search(text)
    if m:
        out["optimizer"] = m.group(1)
        out["lr"] = float(m.group(2))
        out["min_lr"] = float(m.group(3))
        out["weight_decay"] = float(m.group(4))
        out["warmup_epochs"] = int(m.group(5))
        out["grad_clip"] = float(m.group(6))
        out["batch_size"] = int(m.group(7))

    m = RUNID_RE.search(text)
    if m:
        out["wandb_run_id"] = m.group(1)

    # Best epoch summary
    epochs = []
    best_val = float("inf")
    best_ep = None
    for m in EPOCH_RE.finditer(text):
        ep = int(m.group(1))
        val = float(m.group(6))
        is_best = bool(m.group(7))
        epochs.append({"epoch": ep, "epoch_time_s": int(m.group(2)),
                       "peak_gb": float(m.group(3)),
                       "train_vol": float(m.group(4)),
                       "train_surf": float(m.group(5)),
                       "val_avg_mae_surf_p": val})
        if is_best and val < best_val:
            best_val = val
            best_ep = ep
    if best_ep is None and epochs:
        # No `*` mark — pick min seen
        idx = min(range(len(epochs)), key=lambda i: epochs[i]["val_avg_mae_surf_p"])
        best_ep = epochs[idx]["epoch"]
        best_val = epochs[idx]["val_avg_mae_surf_p"]

    out["epochs_completed"] = len(epochs)
    out["best_epoch"] = best_ep
    out["best_val_avg_mae_surf_p"] = best_val if best_val != float("inf") else None
    out["last_epoch_time_s"] = epochs[-1]["epoch_time_s"] if epochs else None
    out["avg_epoch_time_s"] = (sum(e["epoch_time_s"] for e in epochs) / len(epochs)) if epochs else None
    out["last_val_avg_mae_surf_p"] = epochs[-1]["val_avg_mae_surf_p"] if epochs else None

    # Best val line was followed by per-split details. Find best epoch's val_*
    # We use the LAST occurrence of each split name in the log: that corresponds
    # to either the most recent val or the test eval.
    val_splits = {}
    test_splits = {}
    for m in SPLIT_RE.finditer(text):
        name = m.group(1)
        d = {"loss": float(m.group(2)),
             "mae_surf_p": float(m.group(3)),
             "mae_surf_Ux": float(m.group(4)),
             "mae_surf_Uy": float(m.group(5)),
             "mae_vol_p": float(m.group(6)),
             "mae_vol_Ux": float(m.group(7)),
             "mae_vol_Uy": float(m.group(8))}
        if name.startswith("test_"):
            test_splits[name] = d
        else:
            val_splits[name] = d

    if val_splits:
        out["last_val_per_split"] = val_splits
    if test_splits:
        out["test_per_split"] = test_splits
        out["test_avg_mae_surf_p"] = sum(s["mae_surf_p"] for s in test_splits.values()) / len(test_splits)

    # End-of-training markers
    m = TEST_AVG_RE.search(text)
    if m:
        out["printed_test_avg_surf_p"] = float(m.group(1))
    m = BEST_VAL_RE.search(text)
    if m:
        out["printed_best_epoch"] = int(m.group(1))
        out["printed_best_val_avg_mae_surf_p"] = float(m.group(2))

    out["finished"] = bool(DONE_RE.search(text))
    out["timed_out"] = bool(TIMEOUT_RE.search(text))

    return out


def main():
    if len(sys.argv) < 2:
        log_dirs = [Path("logs/phase1"), Path("logs/phase2")]
    else:
        log_dirs = [Path(p) for p in sys.argv[1:]]

    rows = []
    for d in log_dirs:
        if not d.exists():
            continue
        for log in sorted(d.glob("*.log")):
            r = parse_one(log)
            if r:
                r["phase"] = d.name
                rows.append(r)

    # Sort by best val
    rows.sort(key=lambda r: r.get("best_val_avg_mae_surf_p") or float("inf"))

    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
