# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2e-r1)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | TBD (Round 1 in progress) |
| `test_avg/mae_surf_p` | **40.927** |

**Source:** README.md prior competition results — PR #32 (morganmcg1/tandemfoil2): "Single-head nl3/sn16 triple compound"
- W&B run: [ip8hn4tx](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ip8hn4tx)
- test_single_in_dist: 46.569
- test_geom_camber_rc: 52.859
- test_geom_camber_cruise: 24.717
- test_re_rand: 39.561

**Config (best known):** n_layers=3, slice_num=16, n_hidden tuned

## Round 1 — In Progress

9 WIP PRs active. PRs #788–#795 + #808.

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #788 | alphonse | Huber loss instead of MSE | Running |
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
