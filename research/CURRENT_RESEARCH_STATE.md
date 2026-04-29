<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date**: 2026-04-28 (round 6 assignments complete; 8 PRs active WIP; 0 students idle)
- **Most recent research direction from human researcher team**: None (no GitHub Issues open)
- **Current baseline**: `val_avg/mae_surf_p = 93.1083` (PR #1001, n_head=2 wider heads on top of T_max=15 + max_norm=5.0 + Re-weighting, epoch 16/50)

## Current Research Focus and Themes

The research has stacked four compound improvements:
1. **Gradient clipping** (PR #778): clip_grad_norm max_norm=1.0 — 24% improvement
2. **Schedule + clip relaxation** (PR #911): T_max=15 + max_norm=5.0 — additional 6.9% improvement
3. **Per-sample Re-weighting** (PR #931): w_i∝1/(log_re−min+1) normalized per batch — additional 2.8% improvement
4. **n_head=2 wider attention heads** (PR #1001): head_dim=64 instead of 32 — additional 1.76% improvement

The model is still **timeout-cut at ~14-16/50 epochs** — all loss curves descending at cutoff. Every technique that improves convergence speed or provides better signal in early epochs has amplified value.

**Architecture note**: The `TransolverBlock` is already pre-norm (LayerNorm applied BEFORE attention and MLP in each residual path). The "pre-norm variant" direction from earlier rounds is therefore invalid — the architecture is already pre-norm.

Key findings from rounds 3-5:
- **Re-regime imbalance is still a dominant signal** — per-sample weighting (PR #931) produces meaningful accuracy gains
- **Wider attention heads improve OOD generalization** (PR #1001, n_head=2 → head_dim=64)
- **Depth (n_layers) is exhausted at n_hidden=128** — both n_layers=6 (#1037) and n_layers=8 (#1010) regressed due to wall-clock cost eating budget epochs
- **slice_num tuning** is timeout-confounded — slice_num=96 (#1039) regressed; slice_num=48 (#1051 WIP) and 128 (#1012 closed-regression) tested
- **Loss objective alignment matters** — Huber (#1040) and relative MAE (#1009) both regressed; focal node weighting (#1052 WIP) is the active next step

### Round 6 Active WIP (8 total, 0 idle students)

- **PR #1034** (charliepai2e2-edward): n_head=1 — single widest-head attention (head_dim=128).
- **PR #1036** (charliepai2e2-alphonse): mlp_ratio=3 — wider MLP feedforward per Transolver block.
- **PR #1042** (charliepai2e2-askeladd): Multi-task Re auxiliary head (auxiliary log(Re) prediction).
- **PR #1044** (charliepai2e2-fern): AdamW beta2=0.99 (faster gradient variance decay for tight epoch budget).
- **PR #1046** (charliepai2e2-alphonse): Geometry feature dropout p=0.2.
- **PR #1049** (charliepai2e2-nezuko): mlp_ratio=4 — wider FFN at fixed n_layers=5 (512 hidden in FFN).
- **PR #1051** (charliepai2e2-frieren): slice_num=48 — fewer physics slices for potential epoch budget gain.
- **PR #1052** (charliepai2e2-tanjiro): Focal-style node-level pressure weighting — w_j ∝ |p_j| / mean(|p_j|) per sample.
- **PR #1055** (charliepai2e2-thorfinn): Learned Re-embedding — log(Re) as Linear(1→n_hidden) additive bias injected after preprocess MLP. Explicit Re-regime conditioning at feature level to complement per-sample Re-weighting at loss level.

### Round 5 Closed Dead Ends

- **PR #1041** (charliepai2e2-thorfinn): SGDR warm restarts T_0=5, T_mult=2 — regressed. Baseline CosineAnnealing T_max=15 with smooth single cycle is optimal for the ~14-16 epoch budget.
- **PR #1037** (charliepai2e2-nezuko): n_layers=6 — +4.7% regression. Wall-clock cost of extra block eats 2+ budget epochs. Depth axis exhausted.
- **PR #1039** (charliepai2e2-frieren): slice_num=96 — +3.7% regression. Timeout confound: +8-10s/epoch = 1 fewer epoch in budget.
- **PR #1040** (charliepai2e2-tanjiro): Huber loss delta=1000 — +2.9% regression. High-Re gradients carry signal, not noise.

### Recently Closed / Merged (rounds 3-4)

- **PR #1001** (charliepai2e2-edward, n_head=2): MERGED → new baseline 93.1083
- **PR #966** (charliepai2e2-fern, n_hidden=256): Closed — 85% epoch slowdown, dead end
- **PR #974** (charliepai2e2-nezuko, n_head=8): Closed — head_dim=16 too narrow, 36% slower
- **PR #992** (charliepai2e2-tanjiro, Re-weighting alpha sweep): Closed — alpha=1 optimal
- **PR #985** (charliepai2e2-thorfinn, WD sweep): Closed — WD=1e-5 is near sweet spot
- **PR #996** (charliepai2e2-alphonse, curriculum): Closed — curriculum vs tight epoch budget conflicted
- **PR #978** (charliepai2e2-askeladd, T_max=18): Closed — T_max=15 remains optimal
- **PR #1010** (charliepai2e2-nezuko, n_layers=8): Closed — depth too slow for budget

## Key Insights

1. **Gradient explosion from high-Re samples was the dominant failure mode** — pre-clip norms 40–900× above threshold. Fixed by clipping (max_norm=1.0 → 5.0).
2. **LR schedule mismatch was structural** — T_max=50 with ~14 epoch budget means LR stays at 84% of initial throughout. Aligned to T_max=15 in PR #911.
3. **Per-sample Re-weighting is additive to clipping** — the 1/(log_re−min+1) weighting reduces grad norms by 2.6× without suppressing gradient direction as strongly.
4. **Checkpoint averaging (ckpt_avg K=3) is free improvement** — zero training cost. Standard on all future experiments.
5. **EMA is a weaker substitute for ckpt_avg** — not worth the complexity.
6. **eta_min=1e-6 is correct** — raising to 1e-5 made results worse (PR #930).
7. **OOD split asymmetry is systematic**: Regularization/smoothing consistently helps `val_geom_camber_cruise` and `val_re_rand` while hurting `val_single_in_dist` and `val_geom_camber_rc`. Any improvement must be net-positive across all 4.
8. **Grad norm monitoring is a proxy for Re-regime health** — watch mean pre-clip norms as a training diagnostic.
9. **Architecture is already pre-norm** — TransolverBlock applies LayerNorm before attention AND MLP. This direction is exhausted.

## Potential Next Research Directions (post round 6)

### If round 6 yields winners

- **n_hidden=192 with budget-aligned T_max** — compromise between 128 and 256. ~25% epoch slowdown → ~11-12 epochs in budget. head_dim stays 96 with n_head=2.
- **Stochastic depth (DropPath)** — randomly drop entire Transolver blocks during training with probability p=0.1. Acts as regularizer and ensemble over sub-models.
- **Combine winning techniques** — if both mlp_ratio=4 and focal weighting win, test them together.

### Plateau Protocol

Round 5 produced three consecutive dead ends (n_layers=6, slice_num=96, Huber loss) plus SGDR = four consecutive dead ends now. Combined with earlier dead ends, the depth and slice_num axes are exhausted. Round 6 covers the remaining high-priority directions (mlp_ratio=4, focal weighting, slice_num=48) plus medium-priority directions (learned Re-embedding, AdamW beta2, geometry dropout, n_head=1, mlp_ratio=3, SGDR-closed). If round 6 also yields no winners, escalate to:
- Bold architectural changes (n_hidden=192, U-Net skip connections)
- Different attention mechanisms (linear attention, FNO-style, graph attention)
- Data representation changes (Re as explicit input feature via learned embedding — partially covered by PR #1055)
- Loss reformulation (spectral loss on pressure field, physics-informed constraints)
