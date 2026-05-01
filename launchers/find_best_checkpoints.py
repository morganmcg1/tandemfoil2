#!/usr/bin/env python3
"""Find the best-K checkpoints across all phases by parsing log files.

Output is a list of (val_score, log_name, run_id, ckpt_path) tuples in JSON
format that can be fed to ensemble_eval.py.

Usage:
  python launchers/find_best_checkpoints.py --top 5 --arch baseline > best.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\((\d+)s\)\s+\[([\d.]+)GB\]\s+train\[vol=([\d.eE+-]+)\s+surf=([\d.eE+-]+)\]\s+val_avg_surf_p=([\d.eE+-]+)(\s+\*)?",
    re.MULTILINE,
)
RUNID_RE = re.compile(r"Run data is saved locally in .*?run-\d+_\d+-(\w+)")
ARCH_RE = re.compile(
    r"Model: Transolver \(([\d.]+)M params\)\s+cfg.n_hidden=(\d+)\s+n_layers=(\d+)\s+n_head=(\d+)\s+slice_num=(\d+)\s+mlp_ratio=(\d+)"
)


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    if not text:
        return None
    m = RUNID_RE.search(text)
    rid = m.group(1) if m else None
    a = ARCH_RE.search(text)
    arch = None
    if a:
        arch = {
            "n_hidden": int(a.group(2)),
            "n_layers": int(a.group(3)),
            "n_head": int(a.group(4)),
            "slice_num": int(a.group(5)),
            "mlp_ratio": int(a.group(6)),
        }
    best_val = float("inf")
    best_ep = None
    for m in EPOCH_RE.finditer(text):
        if m.group(7):  # marked best
            v = float(m.group(6))
            if v < best_val:
                best_val = v
                best_ep = int(m.group(1))
    if best_ep is None:
        # fall back: pick min seen
        for m in EPOCH_RE.finditer(text):
            v = float(m.group(6))
            if v < best_val:
                best_val = v
                best_ep = int(m.group(1))
    if best_ep is None:
        return None
    return {
        "name": path.stem,
        "log_path": str(path),
        "run_id": rid,
        "best_val_avg_mae_surf_p": best_val,
        "best_epoch": best_ep,
        "arch": arch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5,
                    help="number of best checkpoints to return")
    ap.add_argument("--arch", default="any",
                    choices=["any", "baseline", "medium"],
                    help="filter by architecture family. baseline=128/5/4/64/2; medium=192/8/8/32/4")
    ap.add_argument("--logs", nargs="*",
                    default=["logs/phase1", "logs/phase2", "logs/phase3"])
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--require_ckpt", action="store_true", default=True)
    args = ap.parse_args()

    rows = []
    for d in args.logs:
        p = Path(d)
        if not p.exists():
            continue
        for log in sorted(p.glob("*.log")):
            r = parse_log(log)
            if not r:
                continue
            r["phase"] = p.name
            rows.append(r)

    # Filter by arch
    def match_arch(r):
        if args.arch == "any":
            return True
        a = r.get("arch") or {}
        if args.arch == "baseline":
            return (a.get("n_hidden") == 128 and a.get("n_layers") == 5 and
                    a.get("n_head") == 4 and a.get("slice_num") == 64 and
                    a.get("mlp_ratio") == 2)
        if args.arch == "medium":
            return (a.get("n_hidden") == 192 and a.get("n_layers") == 8 and
                    a.get("n_head") == 8 and a.get("slice_num") == 32 and
                    a.get("mlp_ratio") == 4)
        return True

    rows = [r for r in rows if match_arch(r)]
    rows.sort(key=lambda r: r["best_val_avg_mae_surf_p"])

    # Resolve checkpoint paths
    out = []
    for r in rows:
        ck = Path(args.models_dir) / f"model-{r['run_id']}" / "checkpoint.pt"
        if args.require_ckpt and not ck.exists():
            continue
        r["ckpt"] = str(ck)
        r["config_yaml"] = str(Path(args.models_dir) / f"model-{r['run_id']}" / "config.yaml")
        out.append(r)
        if len(out) >= args.top:
            break

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
