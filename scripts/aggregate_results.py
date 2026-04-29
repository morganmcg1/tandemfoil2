"""Parse logs/*.log and produce research/MLINTERN_RESULTS.jsonl.

Each line: {run, log, group, epochs_done, best_epoch, best_val_avg_mae_surf_p,
            final_val, mem_gb, sec_per_epoch, params, n_params, ...}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EPOCH_PAT = re.compile(
    r"Epoch\s+(\d+)\s*\((\d+)s\)\s*\[([\d.]+)GB\]\s*"
    r"train\[vol=([\d.]+)\s+surf=([\d.]+)\]\s*val_avg_surf_p=([\d.]+)"
)
PARAMS_PAT = re.compile(r"Model:\s+Transolver\s+\(([\d.]+)M\s+params\)")


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="ignore")

    rows = []
    for line in re.split(r"[\r\n]+", text):
        m = EPOCH_PAT.search(line.strip())
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "epoch_time_s": int(m.group(2)),
                "peak_mem_gb": float(m.group(3)),
                "train_vol": float(m.group(4)),
                "train_surf": float(m.group(5)),
                "val_avg_mae_surf_p": float(m.group(6)),
            })
    if not rows:
        return None

    pm = PARAMS_PAT.search(text)
    n_params_M = float(pm.group(1)) if pm else None

    best = min(rows, key=lambda r: r["val_avg_mae_surf_p"])
    final = rows[-1]

    name = path.stem
    if name.startswith("r1-"):
        round_id = "round1"
    elif name.startswith("r2-"):
        round_id = "round2"
    elif name.startswith("r3-"):
        round_id = "round3"
    elif name.startswith("r4-"):
        round_id = "round4"
    elif name == "baseline":
        round_id = "baseline"
    else:
        round_id = "other"

    avg_epoch_time = sum(r["epoch_time_s"] for r in rows) / len(rows)
    return {
        "run_log": str(path.name),
        "round": round_id,
        "n_params_M": n_params_M,
        "epochs_done": len(rows),
        "best_epoch": best["epoch"],
        "best_val_avg_mae_surf_p": best["val_avg_mae_surf_p"],
        "final_val_avg_mae_surf_p": final["val_avg_mae_surf_p"],
        "peak_mem_gb": best["peak_mem_gb"],
        "avg_epoch_time_s": round(avg_epoch_time, 1),
        "best_train_vol": best["train_vol"],
        "best_train_surf": best["train_surf"],
        "trajectory": [
            {"epoch": r["epoch"], "val_avg_mae_surf_p": r["val_avg_mae_surf_p"]}
            for r in rows
        ],
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    logs_dir = repo / "logs"
    out = repo / "research" / "MLINTERN_RESULTS.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for log in sorted(logs_dir.glob("*.log")):
        info = parse_log(log)
        if info:
            runs.append(info)

    runs.sort(key=lambda r: r["best_val_avg_mae_surf_p"])
    with open(out, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(runs)} runs to {out}")
    print()
    for r in runs[:15]:
        print(
            f"  {r['best_val_avg_mae_surf_p']:7.3f}"
            f"  ep={r['epochs_done']:>2d} (best e{r['best_epoch']:>2d}){r['n_params_M']:>5.2f}M"
            f"  {r['run_log']}"
        )


if __name__ == "__main__":
    main()
