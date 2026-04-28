# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 92.63** (W&B run `tirux1y1`, tanjiro L1 surface MAE v1-rebased, merged PR #761, 2026-04-28)
- **test_avg/mae_surf_p = 82.83**
- Beat-threshold for new PRs: **val_avg < 92.63**

### Prior best (for reference)
- val_avg/mae_surf_p = 103.13 (askeladd Huber surf loss, merged PR #814) — superseded by L1

## Founding baseline (round 1 reference)

- val_avg/mae_surf_p = 122.15 (W&B run `8cvp4x6r`, unmodified Transolver)
- test_avg/mae_surf_p = 130.90 (W&B run `zaqz12qi`, re-eval via #807)
- Round-1 noise band: 122–146 (single seed, 14-epoch budget)
- PR #807 (NaN-safe masked accumulation) merged — all future runs produce finite `test_avg`

## Progress summary

| PR | Title | Outcome | val_avg |
|----|-------|---------|---------|
| #807 | NaN-safe scoring fix | **MERGED** (infra) | — |
| #814 | Huber surface loss (delta=1.0) | **MERGED** | 103.13 |
| **#761** | **L1 surface MAE loss** | **MERGED — current best** | **92.63** |
| #748 | Transolver 2x capacity | Closed (under-trained) | 203.16 |
| #762 | Boundary-layer features | Closed (−13.3%) | 138.43 |
| #759 | EMA model weights | Closed (wrong-regime) | 124.51 |
| #743 | Channel-weighted 3xp | v2 sent back for L1 rebase | 99.21 |
| #751 | Dropout + stoch-depth | v1 sent back (halve reg + rebase) | 138.81 |
| #750 | LR warmup + cosine | v2 winner, rebase pending | 111.12 |
| #756 | Fourier Re-encoding | v2 winner, rebase pending | 120.22 |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| askeladd | #847 | Huber delta=0.5 sweep | WIP (informational vs L1 win — delta→0 confirms L1 ordering) |
| thorfinn | #815 | FiLM conditioning per-block on log(Re) | WIP — v1b: −3.0% vs founding baseline (val=118.50), rebase onto L1 pending |
| nezuko | #858 | Focal surface loss gamma=1.0 (L1 base) | WIP |
| fern | #751 | Dropout 0.05 + drop_path 0.05 (v2, rebase pending) | WIP |
| edward | #750 | LR warmup + cosine v2 (rebase pending) | WIP |
| frieren | #756 | Fourier Re-encoding v2 (rebase pending) | WIP |
| alphonse | #743 | Channel-weighted L1 v3 (rebase onto post-#761) | WIP |
| tanjiro | #869 | surf_weight sweep (3.0 and 5.0) | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments include `--epochs 14` so cosine annealing completes.
- **NaN test poisoning FIXED** via PR #807. All future runs produce finite `test_avg/mae_surf_p`.
- **L1 > Huber(1.0) > MSE on surface pressure.** PR #761 confirms pure L1 (val=92.63) beats Huber(delta=1.0) (103.13) by 10.2%. Heavy-tailed pressure residuals respond better to always-linear gradient. Huber delta=0.5 (#847 WIP) will complete the ordering.
- **Channel weighting stacks with Huber** (alphonse #743 v2: −3.8% on top of Huber). Pending whether it also stacks with L1 (v3 in progress).
- **Surface dominates volume ~7:1 at L1 convergence** (tanjiro diagnosis). surf_weight=3.0 (#869) tests whether rebalancing frees volume capacity.
- **Boundary-layer features falsified.** log(Re·|saf|) is redundant; volume-node saf mismatch hurts in-dist.

## Potential next research directions

1. **Stack L1 + FiLM** — if thorfinn #815 beats baseline, combining with L1 is the natural round-3 stack (orthogonal mechanisms: loss-shape vs hidden-state regime modulation). High EV.
2. **RevIN output normalization** — per-sample amplitude normalization of y before loss (targets 10× intra-split y_std variation across Re). Unassigned.
3. **Re-stratified oversampling** — within-domain oversample top Re-quintile; addresses high-Re gradient under-coverage. Unassigned.
4. **Per-channel L1 on p only, MSE on Ux/Uy** — tanjiro follow-up #2; assign after #869 result.
5. **Budget-matched capacity scaling** — revisit 2× capacity with `--epochs 4`. Deferred from askeladd #748.
6. **Low-rank slice attention (LRSA)** — replace S×S slice-token self-attention with rank-16 factored. High EV, higher complexity.
7. **Compound: L1 + channel weighting + surf_weight rebalancing** — if all three win independently, round-3 stack.
