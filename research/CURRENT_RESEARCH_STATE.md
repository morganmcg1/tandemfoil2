# SENPAI Research State

- 2026-04-28 (Round 4 assignments active)
- No recent research direction from human researcher team (no open GitHub issues found)
- Current research focus and themes:

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

### Working Baseline

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| `val_avg/mae_surf_p` | **94.7833** | #931 | Per-sample Re-weighted loss (divide each sample's loss by `log(Re)`); epoch 14/50; ckpt_avg K=3; -2.81% vs PR #911 |

**Baseline compound stack:** stock Transolver + clip_grad_norm=1.0 + T_max=15 cosine + ckpt_avg K=3 + per-sample `1/log(Re)` downweighting.

Per-split breakdown (ckpt_avg epochs 12-13-14):
- `val_single_in_dist`: 104.91
- `val_geom_camber_rc`: 105.49
- `val_geom_camber_cruise`: 77.32
- `val_re_rand`: 91.41

### In-Flight Experiments (Round 4 — All 5 Students WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #964 | alphonse | Stronger Re-weighting: alpha=2 exponent `1/(log_Re - min + 1)^2` — steeper downweighting of high-Re samples | WIP |
| #965 | edward | Relative MAE surf-p loss: `\|pred_p - y_p\| / (\|y_p\| + ε)` — auto-normalizes across Re-regime pressure magnitudes | WIP |
| #966 | fern | n_hidden=256 + T_max=12 (budget-aligned for ~20% longer epochs) — first fair test of larger capacity with full compound stack | WIP |
| #967 | tanjiro | Gradient accumulation N=4 (effective batch_size=16) — smoother gradient estimates in undertrained 14-epoch regime | WIP |
| #968 | thorfinn | AdamW weight_decay=1e-3 — moderate regularization increase targeting sweet spot between WD=1e-4 (baseline) and WD=1e-2 (over-regularized) | WIP |

All 5 students assigned as of this round. Zero idle GPUs.

## Experiment History Summary (all rounds)

### Structural Fixes (all merged)
1. **clip_grad_norm=1.0** (PR #778): 137.0 → 104.7457 (-24%). Dominant fix — gradient explosion was the core pathology.
2. **ckpt_avg K=3** (PR #899): 104.7457 → 104.6986 (-0.05%). Small but reliable.
3. **T_max=15 + max_norm=5.0 compound** (PR #911): 104.6986 → 97.5181 (-6.9%). Budget-aligned LR schedule + clip refinement.
4. **Per-sample Re-weighting** (PR #931): 97.5181 → 94.7833 (-2.8%). Divide each sample's loss by `log(Re)` before batch-averaging.

### Closed / Dead Ends
- `max_norm=5.0` alone (PR #898): +9.8% worse — Adam EMA calibrated to norm=1.0 scale
- `lower_lr=1e-4 + warmup` (PR #768): warmup and clipping are substitutes, not complements
- `n_hidden=256` without clipping (PR #764, #800): undertrained/gradient-explosion artifacts
- `slice_num=128` (PR #765): amplified instability pre-clipping

## Potential Next Research Directions

### High Priority (not yet tried on current baseline)
1. **n_hidden=256 full budget** — PR #966 (fern) running. First fair test of wider model.
2. **Re-weighting exponent tuning** — PR #964 (alphonse): alpha=2 for steeper downweighting.
3. **Relative loss normalization** — PR #965 (edward): per-sample relative error rather than absolute MAE.
4. **Gradient accumulation** — PR #967 (tanjiro): effective batch_size=16 for smoother gradients.
5. **Weight decay tuning** — PR #968 (thorfinn): WD=1e-3 (sweet spot between 1e-4 and 1e-2).

### Queued for Future Rounds
6. **Physics-informed regularization** — divergence-free velocity loss, continuity-equation penalty.
7. **Multi-scale attention** — coarse global + dense foil-zone features; separate attention heads.
8. **EMA model weights** — still untested with current compound stack.
9. **Larger batch via accumulation variations** — N=8 or N=16 if N=4 shows promise.
10. **Per-channel output affine head** — learnable scale+bias per channel; untested with full compound stack.
11. **mlp_ratio=4** — untested with full compound stack.
12. **More slices: slice_num=128** — re-test with clipping + full compound stack.
13. **Cosine annealing restarts (SGDR)** — one restart at mid-budget; aggressive schedule.
14. **Label-weighted Re-bands** — explicit bucketing by Re-regime rather than smooth weighting.
15. **Architecture: deeper model (n_layers=8)** — re-test on current compound stack.

### Core Understanding
- The dominant failure mode was high-Re gradient explosion (norms 40-900×). Fixed by clip_grad_norm=1.0.
- The budget constraint (~14 epochs in 30 min) means LR schedule must be aligned to the achievable window.
- Per-sample Re-weighting is a lightweight but effective way to address regime imbalance without explicit bucketing.
- Test results (test_avg = 85.22) substantially better than val_avg (94.78) — model generalizes well.
- `val_geom_camber_cruise` is the easiest split (77.32); `val_single_in_dist` is the hardest (104.91).
