"""Parse session_logs/*.log files to extract per-run results into JSONL.

Reads Epoch summary lines and TEST avg lines, extracting:
  - run name (from filename or wandb_name found in log)
  - best val_avg/mae_surf_p across epochs
  - last epoch number
  - test_avg/mae_surf_p (if present)
  - per-split surf p MAE for best epoch (rough proxy: last best line)
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

SESSION_LOGS = Path("/workspace/ml-intern-benchmark/target/session_logs")
OUT = Path("/workspace/ml-intern-benchmark/target/research/MLINTERN_RESULTS.jsonl")

EPOCH_RE = re.compile(r"^Epoch +(\d+) \((\d+)s\) \[([\d.]+)GB\]\s+train\[vol=([\d.]+) surf=([\d.]+)\]\s+val_avg_surf_p=([\d.]+)( \*)?")
SPLIT_RE = re.compile(r"^    (\S+)\s+loss=([\d.]+)\s+surf\[p=([\d.]+) Ux=([\d.]+) Uy=([\d.]+)\]\s+vol\[p=([\d.]+) Ux=([\d.]+) Uy=([\d.]+)\]")
TEST_AVG_RE = re.compile(r"^  TEST\s+avg_surf_p=([\d.]+)")
NAME_RE = re.compile(r"--wandb_name [\"']?([^\s\"']+)[\"']?", re.MULTILINE)
WANDB_RUN_URL = re.compile(r"View run at: (https://wandb\.ai/\S+)")


def parse_log(p: Path) -> dict | None:
    text = p.read_text(errors="ignore")
    text = text.replace("\r", "\n")
    # Extract config from the wandb_name found in the log.
    name_m = re.search(r"Syncing run (\S+)", text) or NAME_RE.search(text)
    name = name_m.group(1) if name_m else p.stem
    rid_m = re.search(r"runs/([0-9a-z]+)", text)
    run_id = rid_m.group(1) if rid_m else None
    url_m = WANDB_RUN_URL.search(text)
    url = url_m.group(1) if url_m else None
    n_params_m = re.search(r"Transolver \(([\d.]+)M params\)", text)
    n_params = float(n_params_m.group(1)) if n_params_m else None

    # Per-epoch metrics
    epochs = []
    best_val = None
    best_epoch_n = None
    best_per_split: dict | None = None
    cur_per_split = {}

    lines = text.split("\n")
    last_epoch_lines: list[str] = []
    cur_epoch = None
    for line in lines:
        m = EPOCH_RE.match(line)
        if m:
            ep = int(m.group(1))
            t = int(m.group(2))
            gb = float(m.group(3))
            tvol = float(m.group(4))
            tsurf = float(m.group(5))
            val_avg = float(m.group(6))
            best_marker = bool(m.group(7))
            cur_epoch = {"epoch": ep, "epoch_time_s": t, "peak_gb": gb,
                         "train_vol": tvol, "train_surf": tsurf,
                         "val_avg_surf_p": val_avg, "is_best": best_marker}
            if best_marker or best_val is None:
                best_val = val_avg
                best_epoch_n = ep
            cur_per_split = {}
            continue
        sm = SPLIT_RE.match(line)
        if sm and cur_epoch is not None:
            split = sm.group(1)
            cur_per_split[split] = {
                "loss": float(sm.group(2)),
                "mae_surf_p": float(sm.group(3)),
                "mae_surf_Ux": float(sm.group(4)),
                "mae_surf_Uy": float(sm.group(5)),
                "mae_vol_p": float(sm.group(6)),
                "mae_vol_Ux": float(sm.group(7)),
                "mae_vol_Uy": float(sm.group(8)),
            }
            # Once we see all 4 splits + cur_epoch is_best, set best_per_split
            if cur_epoch.get("is_best") and len(cur_per_split) == 4:
                best_per_split = dict(cur_per_split)
                cur_epoch["per_split"] = dict(cur_per_split)
                epochs.append(cur_epoch)
                cur_epoch = None
                cur_per_split = {}
            elif len(cur_per_split) == 4:
                # Non-best epoch — finalize and continue
                cur_epoch["per_split"] = dict(cur_per_split)
                epochs.append(cur_epoch)
                cur_epoch = None
                cur_per_split = {}
            continue

    test_avg_m = TEST_AVG_RE.search(text)
    test_avg = float(test_avg_m.group(1)) if test_avg_m else None
    test_per_split = None
    if test_avg is not None:
        # Look for test split lines after TEST line
        idx = text.find("TEST  avg_surf_p")
        post = text[idx:]
        test_per_split = {}
        for line in post.split("\n")[1:5]:
            sm = SPLIT_RE.match(line)
            if sm:
                test_per_split[sm.group(1)] = {
                    "loss": float(sm.group(2)),
                    "mae_surf_p": float(sm.group(3)),
                    "mae_surf_Ux": float(sm.group(4)),
                    "mae_surf_Uy": float(sm.group(5)),
                    "mae_vol_p": float(sm.group(6)),
                    "mae_vol_Ux": float(sm.group(7)),
                    "mae_vol_Uy": float(sm.group(8)),
                }

    has_oom = "OutOfMemoryError" in text
    has_traceback = "Traceback (most recent call last)" in text

    if best_val is None and not has_oom and not has_traceback:
        return None

    return {
        "log_file": p.name,
        "wandb_name": name,
        "wandb_run_id": run_id,
        "wandb_url": url,
        "n_params_M": n_params,
        "best_val_avg_surf_p": best_val,
        "best_epoch": best_epoch_n,
        "best_per_split_surf_p": {s: v["mae_surf_p"] for s, v in (best_per_split or {}).items()} if best_per_split else None,
        "n_epochs_completed": max((e["epoch"] for e in epochs), default=0),
        "test_avg_surf_p": test_avg,
        "test_per_split": test_per_split,
        "had_oom": has_oom,
        "failed": has_oom or (has_traceback and best_val is None),
        "n_logged_epochs": len(epochs),
    }


def main():
    results = []
    for log_path in sorted(SESSION_LOGS.glob("*.log")):
        try:
            r = parse_log(log_path)
        except Exception as e:
            print(f"!! parse failed {log_path}: {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        results.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print short summary
    finished = [r for r in results if not r["failed"]]
    finished.sort(key=lambda r: (r["best_val_avg_surf_p"] or 1e9))
    print(f"\nParsed {len(results)} runs ({len(finished)} successful):")
    for r in finished[:30]:
        bv = r["best_val_avg_surf_p"] or 0.0
        ta = r["test_avg_surf_p"] or 0.0
        params = r["n_params_M"] or 0.0
        ne = r["n_epochs_completed"] or 0
        print(f"  {r['wandb_name'][:65]:<65s}  "
              f"params={params:.2f}M  "
              f"best_val={bv:7.2f}  "
              f"test_avg={ta:7.2f}  "
              f"n_epochs={ne}")
    failed = [r for r in results if r["failed"]]
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for r in failed:
            print(f"  {r['wandb_name'][:60]}  oom={r['had_oom']}")
    return results


if __name__ == "__main__":
    main()
