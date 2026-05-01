#!/usr/bin/env python3
"""Quick status of all phase1/phase2 training runs from log files."""
from __future__ import annotations

import os
import re
import sys
import subprocess
from pathlib import Path

LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/phase1")
PIDFILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("logs/phase1.pids")

# Each ANSI tqdm carriage line ends with `\r`, but final summary lines are `\n`-terminated.
# Final per-epoch summary line format:
#   Epoch  12 (133s) [42.1GB]  train[vol=0.27 surf=0.16]  val_avg_surf_p=241.5113 *
EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\((\d+)s\)\s+\[([\d.]+)GB\]\s+train\[vol=([\d.eE+-]+)\s+surf=([\d.eE+-]+)\]\s+val_avg_surf_p=([\d.eE+-]+)(\s+\*)?",
    re.MULTILINE,
)
TQDM_RE = re.compile(r"^Epoch\s+(\d+)/(\d+):\s*(\d+)%")


def parse_log(path: Path):
    """Return dict with summary fields. Skip if no epoch summary yet."""
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return None

    summaries = list(EPOCH_RE.finditer(text))
    cur_ep = None
    cur_pct = None
    for m in TQDM_RE.finditer(text):
        cur_ep = int(m.group(1))
        cur_pct = int(m.group(3))

    if not summaries and cur_ep is None:
        return {"name": path.stem, "status": "starting"}

    total_epochs_seen = len(summaries)
    epoch_times = [int(m.group(2)) for m in summaries]
    avg_epoch = sum(epoch_times) / max(len(epoch_times), 1)

    best_val = float("inf")
    best_ep = None
    last_val = None
    last_ep = None
    for m in summaries:
        ep = int(m.group(1))
        v = float(m.group(6))
        is_best = bool(m.group(7))
        last_val = v
        last_ep = ep
        if is_best:
            if v < best_val:
                best_val = v
                best_ep = ep

    if best_ep is None and summaries:
        # No epoch was marked best by training; pick min seen
        for m in summaries:
            v = float(m.group(6))
            if v < best_val:
                best_val = v
                best_ep = int(m.group(1))

    return {
        "name": path.stem,
        "status": "training",
        "epochs_done": total_epochs_seen,
        "current_epoch": cur_ep,
        "current_pct": cur_pct,
        "best_val": best_val,
        "best_epoch": best_ep,
        "last_val": last_val,
        "last_epoch": last_ep,
        "avg_epoch_sec": avg_epoch,
        "epoch_times": epoch_times,
    }


def parse_pidfile():
    pid_to_name = {}
    if PIDFILE.exists():
        for line in PIDFILE.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                gpu, name, pid = parts[0], parts[1], parts[2]
                pid_to_name[name] = (gpu, pid)
    return pid_to_name


def is_alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def main():
    pid_map = parse_pidfile()
    rows = []
    for log in sorted(LOG_DIR.glob("*.log")):
        info = parse_log(log)
        if info is None:
            continue
        gpu, pid = pid_map.get(info["name"], ("-", "-"))
        alive = is_alive(pid) if pid != "-" else False
        info.update(gpu=gpu, pid=pid, alive=alive)
        rows.append(info)

    # Sort by best_val asc; "starting" status (no best) goes last
    rows.sort(key=lambda r: r.get("best_val") or float("inf"))

    # Print table
    print(f"{'Name':<24} {'GPU':>3} {'PID':>6} {'St':>4} {'Ep':>3} {'CurEp':>5}/{'%%':>3} {'BestVal':>9} @{'BEp':>3} {'LastVal':>9} {'avgs':>5}")
    print("-" * 100)
    for r in rows:
        if r["status"] == "starting":
            print(f"{r['name']:<24} {r.get('gpu','-'):>3} {r.get('pid','-'):>6} {'STR':>4}")
            continue
        alive = "RUN" if r["alive"] else "END"
        print(
            f"{r['name']:<24} "
            f"{r.get('gpu','-'):>3} {r.get('pid','-'):>6} {alive:>4} "
            f"{r['epochs_done']:>3} "
            f"{r.get('current_epoch') or '?':>5}/{r.get('current_pct') or '?':>3} "
            f"{r['best_val']:>9.3f} @{r['best_epoch']:>3} "
            f"{r['last_val']:>9.3f} "
            f"{r['avg_epoch_sec']:>5.0f}"
        )


if __name__ == "__main__":
    main()
