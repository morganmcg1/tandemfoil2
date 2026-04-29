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
| #924 | Per-channel output heads (3 independent decoders) | Closed — slows convergence, loses 1 epoch to timeout (+5.8% vs current best) | 84.16 |
| #936 | Depth scaling n_layers=7 | Closed — wall-clock incompatible (10/14 epochs, +15.3%) | 91.74 |
| #756 v3 | Fourier Re-encoding on FiLM+Re-stratify stack | Closed — mechanism redundant with FiLM (+3.0%) | 81.96 |
| #934 | Layer-targeted FiLM (last 2 blocks only) | Closed — pruning early-block FiLM removes useful capacity (+2.8%) | 81.74 |
| #937 | Dual FiLM (pre-block + post-block per block) | Closed — Re-axis lever saturated, capacity overlap (+3.5%) | 82.36 |
| #952 | Wider single output head (128→256→3) | Closed — decoder capacity not the lever (+2.5%); two probes now falsified | 81.52 |
| #917 | Re-input noise σ sweep {0.02, 0.05, 0.10} | Closed — mechanism real but small (+0.4% on old baseline); Re-axis saturated by Re-stratify on current stack | 83.08 |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| askeladd | #976 | **AoA-FiLM: extend FiLM input from 1-d log_Re to 3-d (log_Re, AoA1, AoA2)** | WIP — new 2026-04-29 |
| thorfinn | #970 | Shared FiLM head: one head reused at all 5 blocks (rank-reduction probe) | WIP — new 2026-04-29 |
| nezuko | #969 | Vertical-flip data augmentation (geometric symmetry: y → -y, Uy → -Uy) | WIP — new 2026-04-29 |
| fern | #927 | Per-channel vol loss v2 (rebase onto FiLM+pre-block+Re-stratify; paired A/B) | WIP — sent back 2026-04-29 |
| edward | #975 | **DropPath rate sweep {0.05, 0.10, 0.15} on FiLM+L1+Re-stratify** | WIP — new 2026-04-29 |
| frieren | #962 | EMA model weights on FiLM+L1+Re-stratify (revisit #759 in new regime) | WIP — new 2026-04-29 |
| alphonse | #961 | SwiGLU MLP: replace GELU MLP with Swish-gated linear unit | WIP — new 2026-04-29 |
| tanjiro | #869 | surf_weight sweep (sw=5 wins on L1 base); v2 rebase onto FiLM+L1 pending | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments include `--epochs 14` so cosine annealing completes.
- **NaN test poisoning FIXED** via PR #807. All future runs produce finite `test_avg/mae_surf_p`.
- **L1 dominates the loss-shape sensitivity curve.** Full ordering confirmed (PRs #761, #814, #847): L1 (92.63) << Huber(0.5) (102.97) ≈ Huber(1.0) (103.13) < Huber(2.0) (106.78). Big lever is Huber→L1 (−9.9%).
- **FiLM stacks cleanly with L1** (PR #815 v2-on-l1: −10.6%). Orthogonal mechanisms confirmed. FiLM gains biggest on Re-stratified and widest-Re-range splits.
- **Pre-block FiLM marginally better than post-block** (PR #909: −1.5% val vs post-block baseline). Mixed per-split: Re-targeted splits (re_rand, cruise) improved, in-dist/rc slightly regressed. Mechanism: pre-block modulates Q/K/V attention computation (regime-aware attention patterns) vs post-block modulation which only scales outputs.
- **Re-stratified batch sampling stacks with FiLM+pre-block** (PR #910: −2.5% val). Largest surprise: single_in_dist −9.6% val (not re_rand as predicted). Gradient equalizes high-Re bias under L1. `--re_stratify` now defaults to True.
- **`geom_camber_rc` is the hardest split** (92.95 val at current best) — consistently the most resistant to improvement. Potential next target.
- **Per-channel vol-L1 (p only) works on vol_p** (PR #927 v1: −9% val_vol_p / −9.4% test_vol_p on FiLM+L1 baseline). Surf_p flat — mechanism orthogonal to surface improvements. v2 rebase onto current stack pending.
- **Depth scaling wall-clock incompatible** (PR #936: n_layers=7 → 185s/epoch +41%, only 10/14 epochs fit in 30-min timeout, +15.3% regression). Not a capacity verdict; would need bf16 or torch.compile to test fairly.
- **Fourier encoding redundant with FiLM** (PR #756 v3: +3.0% on full stack). FiLM is a strict generalization of fixed-frequency input encoding. The Re-axis lever is saturated architecturally; no benefit from input-side encoding.
- **Channel weighting falsified on FiLM+L1 stack** (PR #743 v3: +1.1% worse). FiLM's hidden-state modulation already captures the per-channel gradient lever. Channel weighting was genuine at Huber stage (−3.8%) but FiLM makes it redundant.
- **FiLM redistribution attempts saturated** — three independent redistribution probes (PR #934 last-2 FiLM +2.8%, PR #937 dual FiLM +3.5%, PR #756 Fourier +3.0%) all regress vs the current pre-block-FiLM design. **The Re-conditioning lever is architecturally saturated** at this design; future FiLM-axis work should be **rank-reduction** (shared head, PR #970) rather than redistribution.
- **Re-input-noise saturated by Re-stratify** (PR #917 σ-sweep). Mechanism confirmed (val_re_rand −2.1% at σ=0.05) but small absolute effect; Re-stratify already achieves val_re_rand=77.02 < σ=0.05's 77.58. The Re-axis input-side smoothing lever is gone.
- **Decoder capacity is not a lever** — two falsifications: PR #924 per-channel heads (+5.8%) and PR #952 wider single head (+2.5%). All 3 channels regress uniformly, decoder isn't bottlenecked at this depth/width budget. **One preserved signal: `geom_camber_rc` improved on PR #952 (−4.2% val).** Suggests rc-bottleneck is representational, not capacity-uniform — open question for future rc-targeted intervention.
- **FiLM input axis is single-variable now; AoA-FiLM probe (PR #976) opens multi-variable conditioning** as a fresh axis. AoA is a primary flow parameter the model has zero conditioning-awareness of. If AoA-FiLM wins, opens 4-d (Re, AoA1, AoA2, gap) and beyond.
- **Off-Re axes underway:** geometric symmetry (vflip #969), MLP architecture (SwiGLU #961), regularization (DropPath #975), conditioning multi-variable (AoA-FiLM #976), evaluation smoothing (EMA #962), volume-side per-channel loss (#927), rank-reduction (shared FiLM #970). These should compose more cleanly than within-Re-axis variants.
- **Focal loss falsified on L1 base** (PR #858): high-error nodes are convergence-bottlenecked, not gradient-bottlenecked.
- **RevIN structurally mismatched** (PR #884): per-sample loss normalization decouples gradient from absolute-MAE metric.
- **LR warmup mechanism baked into baseline** (PR #750): schedule-budget alignment principle survives as a convention.
- **Full vol-L1 falsified** (PR #902): volume bulk is Gaussian-ish far-field where MSE is theoretically optimal.
- **IMPORTANT:** PRs #756 (Fourier), #869 (surf_weight) need to beat **79.54** now. When sent back, always rebase onto current HEAD.

## Potential next research directions

1. **Vertical-flip data augmentation (geometric symmetry)** — y → -y, Uy → -Uy at p=0.5. Opens a fresh axis (geometric symmetry) orthogonal to FiLM/Re-stratify. **Assigned → nezuko PR #969.**
2. **Shared FiLM head** — single FiLMLayer reused at all 5 blocks. Rank-reduction probe; tests whether per-block FiLM specialization carries information. **Assigned → thorfinn PR #970.**
3. **AoA-FiLM (multi-variable conditioning)** — extend FiLM input from 1-d (log_Re) to 3-d (log_Re, AoA1, AoA2). Tests whether multi-variable conditioning compounds. **Assigned → askeladd PR #976.**
4. **DropPath rate sweep** — stochastic depth on FiLM+L1+Re-stratify stack {0.05, 0.10, 0.15}. Regularization axis. **Assigned → edward PR #975.**
5. **SwiGLU MLP** — replace GELU MLP with Swish-gated linear unit; strict expressivity gain at near-same FLOPs. **Assigned → alphonse PR #961.**
6. **Per-channel volume loss (L1 on p only, MSE on Ux/Uy)** — refined vol-L1. **Assigned → fern PR #927.**
7. **EMA model weights** — exponential moving average for evaluation; revisit prior close (#759) in correct regime. **Assigned → frieren PR #962.**
8. **surf_weight rebalancing** — test surf_weight=5 on FiLM+L1 (tanjiro PR #869 v2 rebase pending).

### Off-the-Re-axis ideas (Re-conditioning lever saturated; pivot to fresh axes)
9. **Horizontal-flip + rotation augmentation (conditional on vflip #969 result)** — if vflip wins, test horizontal-flip (x → -x, Ux → -Ux) and small-angle rotation. Domain-aware augmentation pipeline.
10. **rc-targeted intervention** — `geom_camber_rc` improved isolated on PR #952 (−4.2%). Probe: rc-aware loss reweighting OR a small geometry-domain conditioning head. Currently the hardest split (92.95).
11. **Slice-token reduction / LRSA** — replace S×S (64×64) slice-token self-attention with rank-16 factored. Reduces compute, possibly improves regularization, frees budget for more width.
12. **Width scaling with bf16** — n_hidden=192 plus mixed precision to fit the wall-clock budget that depth-scaling broke.
13. **Geometric pre-training / SSL** — pretrain on vflip-augmented data with masked-node reconstruction, fine-tune on pressure prediction.
14. **4-d FiLM (Re, AoA1, AoA2, gap)** — conditional on PR #976 (AoA-FiLM) result. If multi-variable wins, extend to 4-d.
15. **Test-time augmentation (TTA) with vflip** — at val/test, run forward on input AND vertical-flipped input, average predictions. Inference-only, zero training cost.
16. **Compound 4-way stack** — FiLM-pre + Re-stratify + vflip (if #969 wins) + best-of: SwiGLU, EMA, shared-FiLM, DropPath, AoA-FiLM. Round-4 candidate.
