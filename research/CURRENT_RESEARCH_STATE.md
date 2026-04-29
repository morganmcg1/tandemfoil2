# SENPAI Research State

- **Date:** 2026-04-29
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 79.54** (W&B run `wakfw4uy`, nezuko Re-stratified batch sampling, merged PR #910, 2026-04-29)
- **test_avg/mae_surf_p = 70.26**
- Per-split val: single_in_dist=84.70, geom_camber_rc=92.95, geom_camber_cruise=63.49, re_rand=77.02
- Beat-threshold for new PRs: **val_avg < 79.54**

### Prior bests (for reference)
- val_avg/mae_surf_p = 81.55 (thorfinn pre-block FiLM, merged PR #909) — superseded by Re-stratify
- val_avg/mae_surf_p = 82.77 (thorfinn FiLM v2-on-l1, merged PR #815) — superseded by pre-block FiLM
- val_avg/mae_surf_p = 92.63 (tanjiro L1 surface MAE, merged PR #761) — superseded by FiLM+L1
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
| #761 | L1 surface MAE loss | **MERGED** | 92.63 |
| #815 | FiLM+L1 (per-block Re conditioning, post-block) | **MERGED** | 82.77 |
| **#909** | **Pre-block FiLM (condition attention input on Re)** | **MERGED** | **81.55** |
| **#910** | **Re-stratified batch sampling** | **MERGED — current best** | **79.54** |
| #748 | Transolver 2x capacity | Closed (under-trained) | 203.16 |
| #762 | Boundary-layer features | Closed (−13.3%) | 138.43 |
| #759 | EMA model weights | Closed (wrong-regime) | 124.51 |
| #847 | Huber delta sweep (0.5, 2.0) | Closed — flat; L1 dominates | 102.97 |
| #751 v2 | Dropout 0.05 + drop_path 0.05 on L1 | Closed — within noise | 93.16 |
| #858 | Focal surface loss gamma=0.5/1.0 on L1 | Closed — γ=1.0 +13.4% worse | 92.13 |
| #884 | RevIN — per-sample y normalization | Closed — structural mismatch (+65%) | 152.64 |
| #750 v2-rebased | LR warmup + cosine on FiLM+L1 | Closed — mechanism baked in (+2.78%) | 85.07 |
| #902 | Volume L1 (mirror surface L1 on vol side) | Closed — gradient rebalancing hurts surf_p (+4.2%) | 96.52 |
| #743 v3 | Channel-weighted L1 [1.0,0.5,2.0] on FiLM+L1 | Closed — mechanism falsified on FiLM+L1 (+1.1%) | 83.69 |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| askeladd | #917 | Re-input noise augmentation (smooth FiLM via training-time log(Re) perturbation) | WIP |
| thorfinn | #934 | Layer-targeted FiLM: pre-block conditioning on last 2 blocks only | WIP — new 2026-04-29 |
| nezuko | #937 | Dual FiLM: pre-block + post-block Re conditioning per block | WIP — new 2026-04-29 |
| fern | #927 | Per-channel volume loss (L1 on p only, MSE on Ux/Uy) | WIP |
| edward | #924 | Per-channel output heads (Ux/Uy/p) — decouple decoder pathways | WIP |
| frieren | #756 | Fourier Re-encoding v3 (rebase onto FiLM+pre-block+Re-stratify stack) | WIP — sent back 2026-04-29 |
| alphonse | #936 | Depth scaling: n_layers=7 on full FiLM+L1+Re-stratify stack | WIP — new 2026-04-29 |
| tanjiro | #869 | surf_weight sweep (sw=5 wins on L1 base); v2 rebase onto FiLM+L1 pending | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments include `--epochs 14` so cosine annealing completes.
- **NaN test poisoning FIXED** via PR #807. All future runs produce finite `test_avg/mae_surf_p`.
- **L1 dominates the loss-shape sensitivity curve.** Full ordering confirmed (PRs #761, #814, #847): L1 (92.63) << Huber(0.5) (102.97) ≈ Huber(1.0) (103.13) < Huber(2.0) (106.78). Big lever is Huber→L1 (−9.9%).
- **FiLM stacks cleanly with L1** (PR #815 v2-on-l1: −10.6%). Orthogonal mechanisms confirmed. FiLM gains biggest on Re-stratified and widest-Re-range splits.
- **Pre-block FiLM marginally better than post-block** (PR #909: −1.5% val vs post-block baseline). Mixed per-split: Re-targeted splits (re_rand, cruise) improved, in-dist/rc slightly regressed. Mechanism: pre-block modulates Q/K/V attention computation (regime-aware attention patterns) vs post-block modulation which only scales outputs.
- **Re-stratified batch sampling stacks with FiLM+pre-block** (PR #910: −2.5% val). Largest surprise: single_in_dist −9.6% val (not re_rand as predicted). Gradient equalizes high-Re bias under L1. `--re_stratify` now defaults to True.
- **`geom_camber_rc` is the hardest split** (92.95 val at current best) — consistently the most resistant to improvement. Potential next target.
- **Channel weighting falsified on FiLM+L1 stack** (PR #743 v3: +1.1% worse). FiLM's hidden-state modulation already captures the per-channel gradient lever. Channel weighting was genuine at Huber stage (−3.8%) but FiLM makes it redundant.
- **Focal loss falsified on L1 base** (PR #858): high-error nodes are convergence-bottlenecked, not gradient-bottlenecked.
- **RevIN structurally mismatched** (PR #884): per-sample loss normalization decouples gradient from absolute-MAE metric.
- **LR warmup mechanism baked into baseline** (PR #750): schedule-budget alignment principle survives as a convention.
- **Full vol-L1 falsified** (PR #902): volume bulk is Gaussian-ish far-field where MSE is theoretically optimal.
- **IMPORTANT:** PRs #756 (Fourier), #869 (surf_weight) need to beat **79.54** now. When sent back, always rebase onto current HEAD.

## Potential next research directions

1. **Layer-targeted FiLM (last 2 blocks only)** — reduces over-conditioning on early geometry features. **Assigned → thorfinn PR #934.**
2. **Dual FiLM (pre-block + post-block per block)** — pre-block adapts attention patterns, post-block adapts output magnitudes. **Assigned → nezuko PR #937.**
3. **Depth scaling (n_layers=7)** — 2 extra Transolver blocks within 96GB budget (~62GB est). **Assigned → alphonse PR #936.**
4. **Re-input noise augmentation** — sigma=0.05 Gaussian noise on log(Re). **Assigned → askeladd PR #917.**
5. **Per-channel volume loss (L1 on p only, MSE on Ux/Uy)** — refined vol-L1. **Assigned → fern PR #927.**
6. **surf_weight rebalancing** — test surf_weight=5 on FiLM+L1 (tanjiro PR #869 v2 rebase pending).
7. **Per-channel output heads** — decouple Ux/Uy/p decoder pathways. **Assigned → edward PR #924.**
8. **Fourier Re-encoding v3** — concatenate scalar + 12 Fourier features, rebase onto current stack. **Assigned → frieren PR #756.**
9. **Width scaling (n_hidden=192)** — if depth (#936) stalls, try 1.5× width (~66GB est). Slightly riskier than depth.
10. **Low-rank slice attention (LRSA)** — replace S×S (64×64) slice-token self-attention with rank-16 factored. Reduces compute, possibly improves regularization.
11. **Compound: FiLM + Re-stratify + depth + Re-noise** — if 3+ win independently, round-4 stack.
