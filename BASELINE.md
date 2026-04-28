# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2e-r1)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (PR #788 — Huber loss, epoch 10) |
| `test_avg/mae_surf_p` | **40.927** (prior competition best) |

**Source:** README.md prior competition results — PR #32 (morganmcg1/tandemfoil2): "Single-head nl3/sn16 triple compound"
- W&B run: [ip8hn4tx](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ip8hn4tx)
- test_single_in_dist: 46.569
- test_geom_camber_rc: 52.859
- test_geom_camber_cruise: 24.717
- test_re_rand: 39.561

**Config (best known):** n_layers=3, slice_num=16, n_hidden tuned

## Round 1 — First Winner

### PR #788 — Huber loss instead of MSE (2026-04-28 20:49)
**Student:** charliepai2e1-alphonse | **Branch:** charliepai2e1-alphonse/l1-huber-loss

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (epoch 10/14) |
| `val_avg/mae_surf_Ux` | 1.6785 |
| `val_avg/mae_surf_Uy` | 0.7747 |
| `val_single_in_dist/mae_surf_p` | 148.4833 |
| `val_geom_camber_rc/mae_surf_p` | 120.0717 |
| `val_geom_camber_cruise/mae_surf_p` | 91.6644 |
| `val_re_rand/mae_surf_p` | 102.3790 |
| `test_avg/mae_surf_p` | NaN (cruise split pre-existing bug) |

**vs MSE baseline:** 115.65 vs 126.88 → **-8.85% improvement**
**Metric summary:** `target/metrics/charliepai2e1-alphonse-huber-delta1.0-gtc81aav.jsonl`
**Reproduce:** `cd target/ && python train.py --loss huber --huber_delta 1.0`

## Round 1 — Remaining WIPs

8 WIP PRs active. PRs #789–#795 + #808.

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #789 | askeladd | Gradient clipping (max_norm=1.0) | Running |
| #790 | edward | Increase surf_weight 10→30 | Running |
| #791 | fern | Wider model: n_hidden 128→256, n_head 4→8 | Sent back; follow-up #808 in progress |
| #792 | frieren | Deeper Transolver: n_layers 5→8→6, lr 5e-4→3e-4 | NaN fix + rerun at n_layers=6 |
| #793 | nezuko | Finer physics partitioning: slice_num 64→128 | Running |
| #794 | tanjiro | LR warmup (5 epochs) before cosine annealing | Running |
| #795 | thorfinn | Per-sample loss normalization | Running |
| #808 | fern | bf16 mixed precision for wider model (n_hidden=256, n_head=8) | Running |

### Key Infrastructure Fix (PR #791)
- `accumulate_batch` NaN propagation bug fixed — `0 * NaN = NaN` in `evaluate_split`. All subsequent experiments should include this fix.

## Update History

- 2026-04-28: Round 1 launched. 8 experiments in flight.
- 2026-04-28: PR #808 added (fern bf16 follow-up). 9 active WIPs total.
- 2026-04-28 20:49: PR #788 merged. Huber loss sets new best val_avg/mae_surf_p = 115.6496 (-8.85% vs MSE baseline 126.88).
