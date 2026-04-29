# SENPAI Research State
- 2026-04-29 (updated 2026-04-29 01:20) (branch: icml-appendix-charlie-pai2e-r3)
- No recent directives from human researcher team (checked 2026-04-29 01:20)

## Current Baseline
- **Source**: Per-sample Re-aware loss normalization (PR #919, fern)
- **val_avg/mae_surf_p = 87.614** (lower is better)
- **test_avg/mae_surf_p = 84.461** (3-split mean, cruise excluded due to known 1-sample GT bug)
- Per-split val: single_in_dist=104.985, geom_camber_rc=95.516, geom_camber_cruise=66.346, re_rand=83.608
- Config: Transolver(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2), AdamW(lr=5e-4, wd=1e-4), CosineAnnealingLR(T_max=15 + 1-epoch warmup), batch_size=4, surf_weight=10, epochs=50, per-sample Re-aware RMS normalization

## Key Pending Results (confirmed winners awaiting rebase)
- **PR #892 (tanjiro, OneCycleLR, max_lr=1e-3)**: val_avg=82.019 on OLD baseline (94.387); sent back for rebase. Merge immediately on resubmission.
- **PR #928 (frieren, Multi-scale RFF σ=1,5 on normalized coords)**: val_avg=84.105 on OLD baseline (94.387); still beats new baseline (87.614). Sent back for rebase 2026-04-29. Merge immediately on resubmission.

## Current WIP Experiments

| PR  | Student    | Hypothesis                                                      | Status |
|-----|------------|-----------------------------------------------------------------|--------|
| #950 | fern      | OneCycleLR + batch_size=8 with sqrt-scaled max_lr=1.41e-3      | WIP (just assigned) |
| #947 | alphonse  | OneCycleLR schedule tuning: ONECYCLE_TOTAL_EPOCHS=16 and pct_start=0.2 variants | WIP (just assigned) |
| #928 | frieren   | Multi-scale RFF (σ=1,5) on normalized coords                    | WIP (rebase pending) |
| #925 | nezuko    | FiLM conditioning on log(Re) for Re-aware representations       | WIP    |
| #905 | askeladd  | Signed-log pressure target normalization for heavy-tail Re      | WIP    |
| #903 | edward    | slice_num=128 + bf16 AMP + NaN-safe eval on MAE baseline        | WIP    |
| #895 | thorfinn  | EMA of model weights (decay=0.999) for evaluation               | WIP    |
| #892 | tanjiro   | OneCycleLR (max_lr=1e-3) — WINNER, pending rebase              | WIP (rebase) |

## Current Research Themes

1. **LR scheduling** (most impactful direction): OneCycleLR (PR #892, confirmed winner), OneCycleLR schedule tuning (PR #947), EMA weights (PR #895), OneCycleLR + batch=8 (PR #950)
2. **Positional encoding / input representation**: Multi-scale RFF (PR #928, confirmed winner on old baseline, pending rebase), FiLM log(Re) conditioning (PR #925)
3. **Loss function engineering**: Per-sample Re normalization (merged PR #919), signed-log pressure transform (PR #905)
4. **Architecture/capacity**: slice_num=128 with AMP (PR #903)

## Compound Wins Expected
Once PR #892 (OneCycleLR) and PR #928 (Multi-scale RFF) rebase and merge sequentially, the expected new baseline will be in the low-80s or below. Both techniques are orthogonal (schedule vs input representation) and should compound.

## Potential Next Research Directions

1. **OneCycleLR schedule tuning on current baseline**: pct_start=0.2 (shorter warmup, more annealing time), ONECYCLE_TOTAL_EPOCHS=16/17 — assigned PR #947
2. **OneCycleLR + batch_size=8 with sqrt-scaled max_lr=1.41e-3**: More samples per step, higher peak LR — assigned PR #950
3. **Multi-scale RFF + OneCycleLR combination**: Once both are individually merged, combine for compound gain
4. **Increase RFF num_freq from 32→64**: Frieren's own suggestion; Aero-Nef uses 64-128, cheap to try
5. **Three-scale RFF (σ=0.3, 1.0, 5.0)**: Add very-low-frequency global structure encoding
6. **surf_weight re-sweep with OneCycleLR**: surf_weight=10 was optimal for cosine; deeper cool-down may shift optimum. Sweep {5, 10, 20, 30}
7. **Geometry encoding**: Dedicated encoding of NACA parameters (camber M, position P, thickness T) as positional features — relevant for camber-holdout generalization
8. **Adaptive surf_weight curriculum**: Ramp surf_weight from 10→30 as training converges; synergizes with OneCycleLR's deep annealing phase
9. **KNN/physics-aware attention bias**: Bias Transolver attention by mesh distance to concentrate capacity near foil surface
10. **Learnable RFF basis (SAFE-NET style)**: Make B1, B2 learnable rather than fixed; frieren's suggestion from PR #928
