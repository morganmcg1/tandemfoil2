"""Push recomputed val/test scores back into existing W&B run summaries.

Why: training was launched with ``--skip_test`` so the W&B run summaries don't
have ``test_avg/mae_surf_p`` set. ``scripts/eval_test.py`` (NaN-safe) computed
the test scores for each saved checkpoint locally; this script writes them
into the corresponding W&B run summary so the leader run shows the
paper-facing test number on the W&B dashboard.

Idempotent: re-running just overwrites the same keys.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "senpai-v1-ml-intern"))
    p.add_argument("--evals_dir", default="research")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Specific wandb run ids to patch. If omitted, patch all "
                        "runs that have a matching eval_<run_id>.json.")
    args = p.parse_args()

    api = wandb.Api()
    evals_dir = Path(args.evals_dir)

    if args.runs:
        targets = list(args.runs)
    else:
        targets = []
        for f in sorted(evals_dir.glob("eval_*.json")):
            stem = f.stem.replace("eval_", "")
            if stem.startswith("ensemble_"):
                continue
            if stem == "r4_leader":
                continue
            targets.append(stem)

    for run_id in targets:
        eval_path = evals_dir / f"eval_{run_id}.json"
        if not eval_path.exists():
            print(f"  skip {run_id}: {eval_path} missing")
            continue
        ev = json.load(open(eval_path))
        try:
            run = api.run(f"{args.entity}/{args.project}/{run_id}")
        except Exception as e:
            print(f"  skip {run_id}: {e}")
            continue

        test_log = {f"test_{k}": v for k, v in ev["test_avg"].items()}
        for split_name, m in ev["test_per_split"].items():
            for kk, vv in m.items():
                test_log[f"test/{split_name}/{kk}"] = vv
        val_log = {f"recheck_val_{k}": v for k, v in ev["val_avg"].items()}

        run.summary.update({**val_log, **test_log})
        print(f"  patched {run_id}: test_avg/mae_surf_p={ev['test_avg']['avg/mae_surf_p']:.4f}")


if __name__ == "__main__":
    main()
