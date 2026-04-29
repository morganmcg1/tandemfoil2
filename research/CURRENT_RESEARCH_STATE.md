# SENPAI Research State
- 2026-04-28 (updated post-PR#984-merge)
- No recent directives from human researcher team
- Branch: icml-appendix-charlie-pai2e-r3

## Current Baseline
- **Source**: Compound Multi-scale RFF (σ₁=1.0, σ₂=5.0, num_freq=32) + OneCycleLR (max_lr=1e-3, ONECYCLE_EPOCHS=15, pct_start=0.3) + EMA decay=0.99 (PR #984, thorfinn)
- **val_avg/mae_surf_p = 73.159** (lower is better)
- **test_avg/mae_surf_p = 72.058** (3-split mean, cruise excluded due to known 1-sample GT bug)
- Per-split val: single_in_dist=82.191, geom_camber_rc=85.919, geom_camber_cruise=53.760, re_rand=70.765
- Config: Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2), AdamW(lr=5e-4, wd=1e-4), Multi-scale RFF (σ₁=1.0, σ₂=5.0, num_freq=32), OneCycleLR (max_lr=1e-3, ONECYCLE_EPOCHS=15, pct_start=0.3), EMA decay=0.99, per-sample Re-aware RMS normalization, batch_size=4, surf_weight=10, epochs=50
- Reproduce: `cd target/ && CUDA_VISIBLE_DEVICES=0 EMA_DECAY=0.99 WANDB_MODE=offline python train.py --lr 5e-4 --surf_weight 10 --batch_size 4 --epochs 50`

## Merged Baseline History
| PR  | Student   | Technique                              | val_avg/mae_surf_p |
|-----|-----------|----------------------------------------|-------------------|
| #835| nezuko    | MAE/L1 loss (baseline from MSE)        | 104.058            |
| #889| fern      | CosineAnnealingLR T_max=15 + 1-ep warmup | 94.387           |
| #919| fern      | Per-sample Re-aware RMS loss norm      | 87.614             |
| #895| thorfinn  | EMA weights decay=0.99                 | 87.233             |
| #928| frieren   | Multi-scale RFF σ=(1,5) normalized coords | 83.338          |
| **#984**| **thorfinn** | **Compound RFF + OneCycleLR**    | **73.159** ← current best |

## Confirmed Winners Pending Rebase
These beat earlier baselines but must now beat **73.159** to be worth merging:

| PR  | Student   | Technique                              | val_avg result (old baseline) | Status |
|-----|-----------|----------------------------------------|-------------------------------|--------|
| **#892** | tanjiro | OneCycleLR max_lr=1e-3 standalone | ~81 (on 87.233 baseline) | Sent back for rebase — likely below 73.159 now |
| **#947** | alphonse | OneCycleLR schedule tuning (16ep, pct_start=0.3) | 79.331 | Sent back for rebase — likely below 73.159 now |

Note: Both #892 and #947 tested OneCycleLR alone, which is already in PR #984. On the new baseline (73.159), they may not improve further — review carefully when submitted.

## Current WIP Experiments

| PR  | Student    | Hypothesis                                                       | Status |
|-----|------------|------------------------------------------------------------------|--------|
| #982 | askeladd  | Increase RFF num_freq 32→64 for higher spectral resolution      | WIP    |
| #950 | fern      | OneCycleLR + batch_size=8 with sqrt-scaled max_lr=1.41e-3       | WIP    |
| #947 | alphonse  | OneCycleLR schedule tuning (sent back for rebase)               | WIP (rebase) |
| #925 | nezuko    | FiLM conditioning on log(Re) for Re-aware representations        | WIP    |
| #903 | edward    | slice_num=128 + bf16 AMP                                        | WIP    |
| #892 | tanjiro   | OneCycleLR (sent back for 2nd rebase)                           | WIP (rebase) |

## Current Research Themes

1. **LR scheduling** (most impactful direction so far): OneCycleLR confirmed winner on 3 independent experiments (PRs #892, #947), schedule tuning confirms cycle_length = realized_epochs+1 is the key knob
2. **Positional encoding / input representation**: Multi-scale RFF confirmed winner (#928), RFF num_freq scaling (#982), FiLM log(Re) conditioning (#925)
3. **Loss function engineering**: Per-sample Re normalization (merged #919), EMA weights (merged #895) — confirmed orthogonal wins
4. **Compound bets**: Multi-scale RFF + OneCycleLR together (#984, thorfinn)

## Key Research Insights
1. MAE loss > MSE for high-Re pressure surrogate (merged #835)
2. CosineAnnealingLR T_max=15 >> T_max=50 for 14-epoch budget (merged #889)
3. Per-sample Re-aware loss normalization improves cross-Re generalization (merged #919)
4. EMA weights decay=0.99: implicit ensembling, orthogonal gain (merged #895)
5. Multi-scale RFF encoding at σ=1,5: spectral bias inductive bias, ~8% improvement (pending #928)
6. OneCycleLR max_lr=1e-3, 16ep cycle: ~9% improvement via deeper annealing phase (pending #947)
7. The OneCycleLR mechanism: cycle_length = realized_budget+2 keeps final epoch in steep cosine descent
8. Camber-holdout splits (geom_camber_rc) are hardest — biggest room for improvement

## Potential Next Research Directions

### High priority (once rebase winners merge)
1. **OneCycleLR cycle=17, pct_start=0.3**: Alphonse's own suggestion — epoch-14 LR was still 7.7e-5, one more epoch pushes it further down the slope. Low risk, cheap.
2. **max_lr sweep around 1e-3**: Variants: {8e-4, 1.2e-3, 1.5e-3} with the confirmed 16ep cycle shape.
3. **Compound: RFF + OneCycle + EMA all together** (once #928 and #947 merged individually).

### Near-term
4. **surf_weight re-sweep with OneCycleLR**: surf_weight=10 tuned for cosine; OneCycleLR's higher peak LR may shift optimum. Sweep {5, 10, 15, 20}.
5. **Geometry-aware NACA features + RFF encoding**: NACA parameters (dims 15-21) could benefit from RFF encoding alongside (x,z). Critical for camber-holdout generalization.
6. **Three-scale RFF (σ=0.3, 1.0, 5.0)**: Add very-low-frequency global structure encoding.
7. **Learnable RFF basis (SAFE-NET style)**: Make B1, B2 learnable rather than fixed.
8. **Adaptive surf_weight curriculum**: Ramp surf_weight from 5→20 as training converges.
9. **EMA decay sweep with OneCycleLR**: Optimal EMA decay may shift under OneCycleLR (shorter effective training horizon).
10. **Pressure-specific architecture**: separate MLP head for pressure channel.
