"""Parse all training logs into research/MLINTERN_RESULTS.jsonl.

One JSONL row per non-debug training run we launched in this replicate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent / "logs"
OUT_PATH = Path(__file__).parent / "MLINTERN_RESULTS.jsonl"


def parse_log(path: Path) -> dict | None:
    text = path.read_text()
    if len(text) < 500:
        return None
    name_m = re.search(r"mlintern-pai2-72h-v4-r5/(\S+)", text)
    if not name_m:
        return None
    name = name_m.group(1)
    run_id_m = re.search(r"wandb/run-[0-9_-]+-([a-z0-9]{8})", text)
    run_id = run_id_m.group(1) if run_id_m else None
    cfg_m = re.search(r"Model: Transolver \(([\d.]+)M params, eidetic=(\w+)\)", text)
    n_params_M = float(cfg_m.group(1)) if cfg_m else None
    eidetic = cfg_m.group(2) == "True" if cfg_m else None

    # Extract per-epoch summaries
    pat = (
        r"^Epoch +(\d+) \((\d+)s\) \[([\d.]+)GB\] +"
        r"train\[vol=([\d.]+) surf=([\d.]+)\] +"
        r"val_avg_surf_p=([\d.]+)"
    )
    epochs = []
    for m in re.finditer(pat, text, re.M):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "time_s": int(m.group(2)),
                "mem_gb": float(m.group(3)),
                "train_vol": float(m.group(4)),
                "train_surf": float(m.group(5)),
                "val_avg_surf_p": float(m.group(6)),
            }
        )
    best_v = min((e["val_avg_surf_p"] for e in epochs), default=None)
    best_e = next(
        (e["epoch"] for e in epochs if e["val_avg_surf_p"] == best_v), None
    ) if best_v else None

    test_avg_m = re.search(r"TEST  avg_surf_p=([\d.]+|nan)", text)
    test_avg = test_avg_m.group(1) if test_avg_m else None
    if test_avg in ("nan", None):
        test_avg = None
    else:
        test_avg = float(test_avg)

    # Per-split test (if present and finite)
    test_per_split = {}
    for split in (
        "test_single_in_dist",
        "test_geom_camber_rc",
        "test_geom_camber_cruise",
        "test_re_rand",
    ):
        m = re.search(rf"{split} +loss=\S+ +surf\[p=([\d.]+|nan)", text)
        if m:
            v = m.group(1)
            test_per_split[split] = None if v == "nan" else float(v)

    # CLI args extracted from the wandb run name
    return {
        "log": str(path),
        "name": name,
        "run_id": run_id,
        "n_params_M": n_params_M,
        "eidetic": eidetic,
        "n_epochs_logged": len(epochs),
        "best_val_avg_mae_surf_p": best_v,
        "best_epoch": best_e,
        "test_avg_mae_surf_p": test_avg,
        "test_per_split": test_per_split or None,
    }


def main() -> None:
    rows = []
    for log in sorted(LOG_DIR.glob("p*_g*.log")):
        r = parse_log(log)
        if r and r["n_epochs_logged"] >= 1:
            rows.append(r)

    # Augment from research/eval_*.json overrides (post-hoc fp32 eval)
    research = Path(__file__).parent
    for ev in sorted(research.glob("eval_*.json")):
        with open(ev) as f:
            data = json.load(f)
        # Find row by run_id in checkpoint path
        m = re.search(r"models/model-([a-z0-9]+)/", data.get("ckpt", ""))
        if not m:
            continue
        run_id = m.group(1)
        for r in rows:
            if r["run_id"] == run_id:
                r["fp32_test_avg_mae_surf_p"] = data["test_avg"]["avg/mae_surf_p"]
                r["fp32_val_avg_mae_surf_p"] = data["val_avg"]["avg/mae_surf_p"]
                r["fp32_test_per_split"] = {
                    k: v["mae_surf_p"] for k, v in data["test_per_split"].items()
                }
                break

    rows.sort(
        key=lambda r: (
            r.get("fp32_test_avg_mae_surf_p")
            or r.get("test_avg_mae_surf_p")
            or r.get("best_val_avg_mae_surf_p")
            or 1e9
        )
    )

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows -> {OUT_PATH}")
    print("\nTop 10 by best metric:")
    for r in rows[:10]:
        ranked = (
            r.get("fp32_test_avg_mae_surf_p")
            or r.get("test_avg_mae_surf_p")
            or r.get("best_val_avg_mae_surf_p")
        )
        print(
            f"  {r['name']:<55s} "
            f"val={r['best_val_avg_mae_surf_p']!s:<8} "
            f"test={r.get('fp32_test_avg_mae_surf_p') or r.get('test_avg_mae_surf_p')} "
            f"params={r['n_params_M']}M epochs={r['n_epochs_logged']} ranked={ranked}"
        )


if __name__ == "__main__":
    main()
