"""Query wandb runs in our project group and summarize.

Usage: python research/wandb_summary.py [<group_name>]
"""
import os
import sys
import wandb

api = wandb.Api()
group = sys.argv[1] if len(sys.argv) > 1 else "mlintern-pai2-72h-v4-r2"
project = "wandb-applied-ai-team/senpai-v1-ml-intern"

print(f"Querying {project}, group={group}")
runs = api.runs(project, filters={"group": group})

# Format: name, state, n_params, best_val, test_avg, n_epochs
out = []
for run in runs:
    cfg = run.config
    summary = run.summary
    out.append({
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "params_M": (cfg.get("n_params") or 0) / 1e6,
        "best_val": summary.get("best_val_avg/mae_surf_p", float("inf")),
        "test_avg": summary.get("test_avg/mae_surf_p", None) or summary.get("test_avg/mae_surf_p ", None),
        "best_epoch": summary.get("best_epoch", None),
        "lr": cfg.get("lr"),
        "wd": cfg.get("weight_decay"),
        "sw": cfg.get("surf_weight"),
        "bs": cfg.get("batch_size"),
        "n_hidden": cfg.get("n_hidden"),
        "n_layers": cfg.get("n_layers"),
        "n_head": cfg.get("n_head"),
        "subsample": cfg.get("train_subsample") or 0,
        "scheduler": cfg.get("scheduler", "cosine"),
        "url": run.url,
    })

# Sort by best_val
out.sort(key=lambda x: x["best_val"] if x["best_val"] is not None else 1e9)

# Summary
print(f"\n{'Name':<60s} {'State':<10s} {'BestVal':>8s} {'TestAvg':>8s} {'Ep':>3s} {'Params':>6s} {'lr':>6s} {'wd':>6s} {'sw':>5s} {'bs':>3s} {'arch':>16s}")
print("-" * 160)
for r in out:
    arch = f"h{r['n_hidden']}L{r['n_layers']}h{r['n_head']}"
    if r['subsample'] > 0:
        arch += f"_s{r['subsample']//1000}k"
    state = r["state"][:9]
    bv = r["best_val"] if r["best_val"] is not None else float("inf")
    ta = r["test_avg"] if r["test_avg"] is not None else float("nan")
    ep = r["best_epoch"] or 0
    print(f"{r['name'][:60]:<60s} {state:<10s} {bv:>8.2f} {ta:>8.2f} {ep:>3d} "
          f"{r['params_M']:>5.2f}M {r['lr']:>6.0e} {r['wd']:>6.0e} {r['sw']:>5.1f} {r['bs']:>3d} {arch:>16s}")
