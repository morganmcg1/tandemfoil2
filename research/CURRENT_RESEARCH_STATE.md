# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 103.13** (W&B run `at52zeu5`, askeladd Huber surf loss v1, merged PR #814, 2026-04-28)
- **test_avg/mae_surf_p = 92.99**
- Beat-threshold for new PRs: **val_avg < 103.13**

## Founding baseline (round 1 reference)

- val_avg/mae_surf_p = 122.15 (W&B run `8cvp4x6r`, unmodified Transolver)
- test_avg/mae_surf_p = 130.90 (W&B run `zaqz12qi`, re-eval via #807)
- Round-1 noise band: 122–146 (single seed, 14-epoch budget)
- PR #807 (NaN-safe masked accumulation) merged — all future runs produce finite `test_avg`

## Round-1 summary (closed)

| Student | PR | Outcome | val_avg |
|---------|-----|---------|---------|
| askeladd | #748 transolver-2x | Closed (under-trained, 4/50 epochs) | 203.16 |
| askeladd | #807 scoring-fix | **MERGED** | — |
| thorfinn | #762 boundary-layer-features | Closed (−13.3% WORSE: 138.43 vs 122.15) | 138.43 |
| alphonse | #743 channel-weighted-3xp | Sent back (v2 pending) | 146.10 |
| **tanjiro** | **#761 l1-surface-mae-loss** | **v1 WINNER → rebase pending (val_avg=109.53, test_avg=98.44)** | **109.53** |
| **edward** | **#750 lr-warmup-cosine** | **v2 WINNER → rebase pending (was 135.89, now 111.12)** | **111.12** |
| **frieren** | **#756 fourier-re-encoding** | **v2 WINNER → rebase pending (was 141.25, now 120.22)** | **120.22** |
| nezuko | #759 ema-model-weights | **CLOSED** — EMA wrong-regime at 14-ep budget (val=124.51, +20.7% vs best) | — |
| fern | #751 dropout-stochastic-depth | In progress (status:wip) | — |

## Round-2 assignments (active)

| Student | PR | Hypothesis | Angle | Status |
|---------|-----|-----------|-------|--------|
| askeladd | #814 | huber-surf-loss (delta=1.0) | Loss alignment | **MERGED** (val=103.13, test=92.99) |
| askeladd | #847 | huber-delta-sweep (delta=0.5) | Loss alignment follow-up | WIP |
| thorfinn | #815 | film-re-conditioning (per-block log(Re) FiLM) | Architecture: regime adaptation | WIP |
| nezuko | #858 | focal-surface-loss (gamma=1.0) | Loss: concentrate gradient on high-error nodes | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments now include `--epochs 14` so cosine annealing completes rather than truncating mid-curve.
- **NaN test poisoning FIXED** via PR #807 (torch.where pattern). All future runs produce finite `test_avg/mae_surf_p`.
- **Round-1 noise band: 122–146.** Beat-threshold: `val_avg/mae_surf_p < 122.15`. Single-seed <5% gains are inconclusive; flag for multi-seed confirmation.
- **Boundary-layer features falsified.** log(Re·|saf|) is redundant with existing dims 13+2:3; volume-node saf mismatch hurts in-dist. Surface-gated BL variants deferred to round 3.

## Potential round-2+ research directions

1. **RevIN output normalization** — per-sample amplitude normalization of y before loss (targets 10× intra-split y_std variation across Re). Unassigned.
2. **Focal-surface-loss** — top-20%-error-node up-weighting (concentrates gradient on stagnation/suction peak). **Assigned → nezuko PR #858**.
3. **Re-stratified oversampling** — within-domain oversample top Re-quintile; addresses high-Re gradient under-coverage. Unassigned.
4. **Stack round-2 winners** — if askeladd Huber + thorfinn FiLM both win, combine them in round 3 (orthogonal mechanisms).
5. **Budget-matched capacity scaling** — revisit 2× capacity with `--epochs 4` (matching 30-min ceiling for larger model). Deferred from askeladd #748.
6. **Low-rank slice attention (LRSA)** — replace S×S slice-token self-attention with rank-16 factored; reportedly +17% on PDE benchmarks. High EV, higher complexity.
7. **Mixed-precision (bf16/fp16)** — opens headroom for capacity scaling. More relevant for charlie branch; applicable here if VRAM becomes constraint.
