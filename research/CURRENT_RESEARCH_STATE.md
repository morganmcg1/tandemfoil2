# SENPAI Research State

- 2026-04-28 22:30
- No directives yet from the human researcher team
- **Current research focus**: Round 1 sweep complete/nearly complete. Round 2 begins with targeted experiments addressing the key finding: large per-split heterogeneity (57-point spread in val_avg/mae_surf_p across regimes) and under-weighted pressure channel.

## Current Baseline

- **PR #738** — Surface loss weight 10 → 20
- **val_avg/mae_surf_p = 128.8320** (lower is better)
- Architecture: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Training: `lr=5e-4`, `surf_weight=20.0`, `weight_decay=1e-4`, `batch_size=4`

## Round 1 Results Summary

| PR | Student | Hypothesis | Result | Decision |
|----|---------|------------|--------|----------|
| #738 | edward | surf_weight 10→20 | **128.83** | Merged (baseline) |
| #735 | alphonse | n_hidden 128→256 | 130.86 | Sent back (n_hidden=192, n_layers=6) |
| #740 | fern | n_layers 5→7 | 135.90 | Closed (43% per-epoch cost, depth loses under wall-clock) |
| #741 | frieren | mlp_ratio 2→4 | WIP | — |
| #744 | nezuko | LR warmup 5-epoch | Closed (Round 1) | — |
| #746 | tanjiro | n_head 4→8 | Closed (Round 1) | — |
| #747 | thorfinn | per-sample norm loss | WIP — promising (~110 expected) | — |
| #736 | askeladd | slice_num 64→128 | WIP — promising (~122 expected) | — |
| #812 | edward | LR 5e-4→2e-4 w/ sw=20 | WIP | — |
| #813 | frieren | zero weight decay w/ sw=20 | WIP | — |

## Active WIP Experiments

| PR | Student | Hypothesis |
|----|---------|------------|
| #747 | thorfinn | Per-sample normalized loss (rebase + surf_weight=20 re-run) |
| #736 | askeladd | More slices 64→128 (surf_weight=20 re-run) |
| #735 | alphonse | Mixed capacity: n_hidden=192, n_layers=6, T_max=11 |
| #812 | edward | LR 5e-4→2e-4 with surf_weight=20 |
| #813 | frieren | Zero weight decay (weight_decay=0) with surf_weight=20 |
| #838 | nezuko | Per-channel pressure weighting: p_weight=5.0 in loss |
| #839 | tanjiro | Re-stratified mini-batch sampling to balance regime coverage |

## Key Findings So Far

1. **Per-split heterogeneity is substantial** — 57-point spread (val_single_in_dist=157 vs val_geom_camber_cruise=100). This appears across all experiments and is not specific to any architecture. Likely reflects training data imbalance across regimes.
2. **surf_weight=20 is clearly better than 10** — established by PR #738.
3. **Depth (n_layers=7) loses under fixed wall-clock** — 43% per-epoch overhead means fewer total epochs. Closed.
4. **Width (n_hidden=256) also loses under wall-clock** — same issue; 2× cost means only 9 epochs. A mixed design (n_hidden=192, n_layers=6) is being tested.
5. **Per-sample normalized loss (thorfinn/PR #747) is the most promising lead** — early results suggested ~110 (−14% vs baseline). Awaiting clean rerun with surf_weight=20.
6. **T_max must match achievable epoch count** — critical misconfiguration lesson from rounds 1. Always set T_max = expected_epochs_in_budget.

## Potential Next Research Directions

### Tier 1 — High confidence, directly motivated by findings
1. **Per-sample normalized loss + p_weight combination** — if both PR #747 (norm loss) and PR #838 (p_weight) win individually, combine them
2. **Re-stratified sampling + per-sample norm loss** — address both data imbalance and loss calibration simultaneously
3. **Per-channel adaptive loss weights** — learned uncertainty weighting (Kendall et al.) instead of fixed p_weight scalar

### Tier 2 — Architecture, motivated by remaining headroom
4. **Mixed capacity design confirmed** — n_hidden=192, n_layers=6 being tested by alphonse (#735 respin)
5. **SliceNum + width combination** — if askeladd's slice_num=128 wins, combine with alphonse's mixed capacity
6. **Hierarchical attention** — cross-attention between surface and volume nodes

### Tier 3 — Loss and optimization innovation
7. **Gradient-based surface emphasis** — weight loss by local pressure gradient magnitude (high-curvature regions near leading/trailing edge)
8. **Physics-consistency auxiliary loss** — soft enforcement of continuity equation
9. **Alternative optimizers** — Lion, SOAP, or schedule-free Adam
10. **Multi-scale input features** — add curvature, arc-length gradients, local normals as input features

### Tier 4 — Bold structural changes (if plateau persists)
11. **GNN-augmented architecture** — replace or augment Transolver with GNN layers using mesh connectivity
12. **Domain-specific heads** — separate output heads for in-dist vs OOD regimes
13. **Test-time augmentation / ensemble** — model averaging across Re regimes
14. **Completely different architecture** — FNO, DeepONet, or U-Net style encoder-decoder
