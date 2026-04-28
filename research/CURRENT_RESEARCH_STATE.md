# SENPAI Research State

- 2026-04-28 21:05
- No recent research direction from human researcher team (no open GitHub issues found)
- Current research focus and themes:

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

### Working Baseline (NEW)

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| `val_avg/mae_surf_p` | **104.7457** | #778 | epoch 14/50, undertrained; gradient clipping clip_grad_norm=1.0 |

The previous baseline was 137.0013 (PR #764, n_hidden=256, epoch 9/50). Gradient clipping alone delivered a 24% improvement on stock architecture — confirming that gradient explosion from high-Re samples (pre-clip norms 40–900× above 1.0) was the dominant pathology, not model capacity or schedule.

### Reframing After PR #778

Every prior result needs to be re-judged against the new baseline. Specifically, every approach that improved on 137.0 but not 104.7 needs to rebase onto the clipping baseline. Stable gradients are likely orthogonal to most of the techniques tested, so several may compound.

### In-Flight Experiments (WIP, on r2)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #800 | alphonse | n_hidden=256 + bf16 AMP | sent back to rebase + re-run on clipping baseline |
| #780 | thorfinn | mlp_ratio 2→4 | WIP |
| #772 | nezuko | per-channel output affine | WIP |
| #768 | frieren | lr=1e-4 + warmup | sent back to rebase + re-run on clipping baseline |
| #767 | fern | surf_weight 10→50 | sent back to rebase + re-run on clipping baseline |
| #766 | edward | n_layers 5→8 | WIP |

Two students are idle and need new assignments: **askeladd**, **tanjiro**.

## Potential Next Research Directions

Now that gradient clipping is in baseline, the search space opens up:

1. **Per-sample loss normalization** — divide vol/surf loss by per-sample y_norm std; addresses the 15× variance in per-sample y_norm std (0.32 to 4.85) across the corpus. Strong orthogonal candidate to compound with clipping.
2. **Stronger clipping schedule** — try clip_grad_norm=0.5 or warmup-then-relax (start tight, loosen as training stabilizes).
3. **Surface-aware loss reformulation** — Huber/SmoothL1 on the surface channel (tail-robust); per-channel surf_weight tuning (Ux/Uy vs p).
4. **AdamW + weight decay tuning** — current weight_decay=1e-5 is essentially zero; with stable gradients, WD=1e-2 or 1e-3 is now a reasonable target.
5. **Cosine LR with restarts (SGDR)** — stable gradients permit aggressive schedules.
6. **NaN bug in `data/scoring.py`** — `err * mask` with NaN preds yields `NaN * 0.0 = NaN`; poisons test_geom_camber_cruise pressure MAE on multiple runs. Fix needs a separate PR. (Note: data/ is read-only for experiment PRs — this is an organizer-level fix.)
7. **Output head — separate per-channel** — pressure dynamics are dominated by extremes; a dedicated pressure head (deeper, with its own normalization) may help.
8. **Larger batch via accumulation** — gradient accumulation + clipping = stable optimization at effective batch_size=16 or 32; less frequent updates, more stable signal.
9. **Multi-scale attention** — combine coarse background zone with dense foil-zone features.
10. **Physics-informed regularizers** — divergence-free velocity constraint (continuity equation).
11. **Label smoothing of surface mask** — soft surface weighting based on signed distance instead of hard 0/1.
12. **EMA of model weights** — checkpoint EMA-averaged model for evaluation; particularly useful at undertrained 14-epoch regime.
