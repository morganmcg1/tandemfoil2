# SENPAI Research State

- 2026-04-29 00:00
- No recent research direction from human researcher team (no open GitHub issues found)
- Current research focus and themes:

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

### Working Baseline

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| `val_avg/mae_surf_p` | **104.7457** | #778 | epoch 14/50, undertrained; gradient clipping clip_grad_norm=1.0 |

The previous baseline was 137.0013 (PR #764, n_hidden=256, epoch 9/50). Gradient clipping alone delivered a 24% improvement on stock architecture — confirming that gradient explosion from high-Re samples (pre-clip norms 40–900× above 1.0) was the dominant pathology, not model capacity or schedule.

### Marginal Winner Pending Rebase

PR #897 (charliepai2e2-alphonse, T_max=15 cosine schedule) achieved **104.4004** (Δ -0.33%) but merge failed due to conflicts. Sent back as draft for rebase onto `icml-appendix-charlie-pai2e-r2`. Once rebased and re-verified, baseline will update to 104.4004.

### Closed This Round

| PR | Student | Result | Reason |
|----|---------|--------|--------|
| #898 | askeladd | 114.9909 (+9.8%) | Dead end — max_norm=5.0 inflates effective LR 5×; OOD splits devastated |

### In-Flight Experiments (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #897 | alphonse | T_max=15 cosine schedule | SENT BACK — marginal win (104.4004), merge conflict; needs rebase |
| #856 | tanjiro | EMA model weights (decay=0.999) for smoother timeout-cut checkpoints | WIP — queued as tanjiro follow-up after #826 |
| #826 | tanjiro | AdamW weight decay 1e-5→1e-2 | WIP — first run on clipping baseline, still running |
| #921 | askeladd | Cosine T_max sweep {20, 25, 30} — find optimal LR annealing endpoint for ~14-epoch budget | WIP — new assignment 2026-04-29 |
| #800 | alphonse | n_hidden=256 + bf16 AMP | WIP — rebased on clipping baseline, re-running |
| #780 | thorfinn | mlp_ratio 2→4 | WIP — sent back to rebase onto clipping baseline (prev result 135.296 without clipping) |
| #772 | nezuko | per-channel output affine | WIP — sent back to rebase onto clipping baseline (prev result 138.497 without clipping) |
| #875 | frieren | Schedule-to-budget cosine: T_max=14 to align LR annealing to achievable epoch budget | WIP — assigned 2026-04-28 |
| #767 | fern | surf_weight 10→50 | WIP — rebased on clipping baseline, re-running |
| #766 | edward | n_layers 5→8 | WIP — first run on clipping baseline, still running |

All 8 students have active WIP assignments. No idle students. Checked 2026-04-29 00:00 — PR #898 (askeladd/less-aggressive-clip-norm) closed as dead-end: max_norm=5.0 inflates effective LR ~5×; OOD splits devastated. Askeladd re-assigned to cosine T_max sweep (PR #921). PR #897 (alphonse T_max=15) marginal win sent back for rebase due to merge conflict.

Note: tanjiro has two WIP PRs (#826 AdamW WD, #856 EMA). #826 was assigned first; #856 queued as follow-up. Student will work through sequentially.

### Cosine Schedule Insight

T_max=15 (PR #897) showed a small gain by aligning the cosine arc to the ~14-epoch achievable budget. The LR at epoch 14:
- T_max=15 → ~5.5e-6 (arc nearly complete)
- T_max=20 → ~5.6e-5 (gentle taper)
- T_max=25 → ~1.3e-4 (mid-slope)
- T_max=30 → ~1.9e-4 (near peak)
- T_max=50 (stock) → ~4.2e-4 (barely moved)

PR #921 sweeps T_max ∈ {20, 25, 30} to determine whether a gentle mid-arc taper or the nearly-complete arc is optimal for the 14-epoch budget. max_norm=1.0 confirmed as structural — Adam β₁/β₂ are calibrated to this gradient scale.

## Potential Next Research Directions

1. **Per-sample loss normalization** — divide vol/surf loss by per-sample y_norm std; addresses the 15× variance in per-sample y_norm std (0.32 to 4.85) across the corpus. Strong orthogonal candidate to compound with clipping.
2. **Surface-aware loss reformulation** — Huber/SmoothL1 on the surface channel (tail-robust); per-channel surf_weight tuning (Ux/Uy vs p).
3. **AdamW + weight decay tuning** — current weight_decay=1e-5 is essentially zero; with stable gradients, WD=1e-2 or 1e-3 is now a reasonable target (PR #826 in-flight).
4. **Cosine LR with restarts (SGDR)** — stable gradients permit aggressive schedules; one restart at epoch ~7 may help.
5. **NaN bug in `data/scoring.py`** — `err * mask` with NaN preds yields `NaN * 0.0 = NaN`; poisons test_geom_camber_cruise pressure MAE on multiple runs. Fix: `torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)` in `accumulate_batch`. Organizer-level fix required (data/ is read-only for experiment PRs).
6. **Output head — separate per-channel** — pressure dynamics are dominated by extremes; a dedicated pressure head (deeper, with its own normalization) may help.
7. **Larger batch via accumulation** — gradient accumulation + clipping = stable optimization at effective batch_size=16 or 32; less frequent updates, more stable signal.
8. **EMA of model weights** — checkpoint EMA-averaged model for evaluation; particularly useful at undertrained 14-epoch regime (PR #856 in-flight).
9. **Multi-scale attention** — combine coarse background zone with dense foil-zone features.
10. **Physics-informed regularizers** — divergence-free velocity constraint (continuity equation).
11. **Label smoothing of surface mask** — soft surface weighting based on signed distance instead of hard 0/1.
