# SENPAI Research State

- 2026-04-28 22:44
- No new directives from the human researcher team
- **Current research focus**: Round 4 — baseline is now PR #845 (per-sample norm loss + surf_weight=20, val_avg/mae_surf_p=105.9649). Round 4 explores: combining per-sample norm with lower LR (2e-4), wider/deeper architecture with per-sample norm, Re-stratified sampling, per-channel pressure weighting, surf_weight=40, and gradient clipping. **Target: break sub-100 and push toward sub-90.**

## Current Baseline

- **PR #845** — Per-sample normalized loss + surf_weight=20 (fern)
- **val_avg/mae_surf_p = 105.9649** (lower is better)
- Architecture: `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Training: `lr=5e-4`, `surf_weight=20.0`, `weight_decay=1e-4`, `loss_kind=per_sample_norm_mse`, `--epochs 14`

## Experiment History Summary

| PR | Student | Hypothesis | Result | Decision |
|----|---------|------------|--------|----------|
| #738 | edward | surf_weight 10→20 | **128.83** | MERGED (original baseline) |
| #735 | alphonse | n_hidden 192, n_layers=6, sw=20 | 128.38 | MERGED (cherry-picked into baseline) |
| #740 | fern | n_layers 5→7 | 135.90 | CLOSED — too slow per epoch |
| #741 | frieren | mlp_ratio 2→4 | 141.54 | CLOSED — too slow per epoch |
| #736 | askeladd | slice_num 64→128 | 135.96 | CLOSED |
| #746 | tanjiro | n_head 4→8 | 128.96 (tied) | CLOSED |
| #747 | thorfinn | per-sample norm loss (sw=10) | 110.37 | Sent back for rebase onto advisor branch |
| #802 | edward | bf16 + batch_size=8 | 129.14 | CLOSED (r5 track) |
| #812 | edward | LR 5e-4→2e-4 w/ sw=20 | **112.94** | MERGED |
| #813 | frieren | zero weight decay w/ sw=20 | WIP (~91 min, may be stalled) | — |
| #838 | nezuko | per-channel p weighting: p_weight=5.0 | CLOSED — superseded | — |
| #839 | tanjiro | Re-stratified mini-batch sampling | CLOSED — superseded | — |
| #845 | fern | per-sample norm loss + sw=20 | **105.9649** | MERGED (current best) |
| #849 | askeladd | surf_weight 20→40 + T_max=15 | CLOSED — superseded | — |

## Active WIP Experiments (Round 4)

| PR | Student | Hypothesis | Assigned |
|----|---------|------------|----------|
| #747 | thorfinn | Per-sample norm loss rebase onto advisor branch (sw=10 → sw=20 combo) | ~22:00 |
| #813 | frieren | Zero weight decay (weight_decay=0.0) with sw=20, epochs=14 | ~21:13 |
| #868 | edward | Per-sample norm + lower LR (2e-4) + sw=20 | 22:27 |
| #870 | alphonse | Wider arch (n_hidden=192, n_layers=6) + per-sample norm + sw=20 | 22:30 |
| #871 | fern | Per-sample norm + surf_weight=40 | 22:31 |
| #874 | nezuko | Per-channel p weighting + per-sample norm + sw=20 | 22:33 |
| #876 | tanjiro | Re-stratified sampling + per-sample norm + sw=20 | 22:34 |
| #877 | askeladd | Gradient clipping (max_norm=1.0) + per-sample norm + sw=20 | 22:34 |

## Key Technical Findings

1. **Per-sample normalized loss is now the gold standard** — divides each sample's MSE by per-sample target variance; equalizes Re-regime contributions. +14.3% win (sw=10); combining with sw=20 gave 105.96 (−17.7% from original baseline 128.83). All new experiments MUST include `loss_kind=per_sample_norm_mse`.
2. **surf_weight=20 is clearly better than 10** — established by PR #738. All new experiments must include `--surf_weight 20.0`.
3. **Lower LR (2e-4) gives +12.3%** on surface pressure vs default 5e-4 when using surf_weight=20 (PR #812). Likely stacks with per-sample norm.
4. **T_max must match achievable epoch count** — always set T_max = expected_epochs_in_budget (set via `--epochs N`).
5. **Depth/width hurt under wall-clock budget** — n_layers=7 and n_hidden=256 both lose in 30-min budget by reducing achievable epochs. Mixed design (n_hidden=192, n_layers=6) borderline — per-epoch cost must be profiled.
6. **Per-split heterogeneity is substantial** — 52-point spread in PR #845 (val_geom_camber_cruise=79.12 vs val_single_in_dist=130.99). Regime imbalance is a key lever.
7. **NaN in test_geom_camber_cruise** — sample 20 has -inf ground truth; val splits unaffected. Pre-existing pipeline issue.

## Potential Next Research Directions

### Tier 1 — Direct follow-ons to current best (per-sample norm + sw=20 = 105.97)
1. **Per-sample norm + lower LR (2e-4)** — in-flight PR #868 (edward). LR 5e-4→2e-4 gave +12.3% at sw=20 (PR #812); should stack with per-sample norm.
2. **Wider arch + per-sample norm** — in-flight PR #870 (alphonse). Wider model had acceptable per-epoch time; per-sample norm may help it converge faster.
3. **Per-sample norm + surf_weight=40** — in-flight PR #871 (fern). Extrapolating sw=10→20 gain.
4. **Gradient clipping + per-sample norm** — in-flight PR #877 (askeladd). Reduce oscillation on surface-pressure loss spikes.

### Tier 2 — Regime diversity and sampling
5. **Re-stratified sampling + per-sample norm** — in-flight PR #876 (tanjiro). 52-point per-split spread suggests regime imbalance; stratified sampling + per-sample norm may together eliminate it.
6. **Per-channel p weighting + per-sample norm** — in-flight PR #874 (nezuko). Extra p-channel emphasis within sample-normalized loss.
7. **CosineAnnealingWarmRestarts** — cyclic LR restarts within 30-min budget may escape local minima.

### Tier 3 — Architecture innovation
8. **n_hidden=192, n_layers=6 at lr=2e-4** — mixed-capacity design with lower LR; in-flight PR #870 tests part of this.
9. **Learned uncertainty weighting (Kendall et al.)** — adaptive channel weights instead of fixed scalars.
10. **Input feature enrichment** — curvature, arc-length gradients, local normals as extra node features.

### Tier 4 — Bold structural changes (if plateau persists near 95-100)
11. **GNN-augmented architecture** — replace or augment Transolver with GNN layers using mesh connectivity.
12. **Ensemble / checkpoint averaging** — average final K checkpoints (free post-training gain).
13. **Physics-consistency auxiliary loss** — soft enforcement of continuity equation.
14. **Completely different architecture** — FNO, DeepONet, or U-Net style encoder-decoder.
