# SENPAI Research State
- 2026-04-29 ~02:00 UTC (last updated)
- Most recent research direction from human researcher team: None received yet (no GitHub Issues found)
- Current research focus and themes: Round 1 — Compound baseline refinement + promising winners awaiting final rebase. Three potential breakthrough PRs (#792 frieren 90.78, #795 thorfinn 93.40, #789 askeladd) blocked on final rebase+re-run onto post-#882 compound stack.

## Current Baseline

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **103.2182** | PR #882 — EMA(0.999) + bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12 + grad_clip=1.0 |
| `test_avg/mae_surf_p` | **92.4867** | PR #882 |

**Best reproduce command:**
```bash
cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999
```

## Active WIP PRs (8 students, all running or awaiting rebase)

| PR | Student | Hypothesis | Status | Last best val |
|----|---------|------------|--------|---------------|
| #960 | alphonse | surf_weight sweep (20/30/50) on compound + grad_clip baseline | Running (just assigned) | — |
| #942 | nezuko | EMA decay sweep: 0.99/0.995 vs 0.999 on compound baseline | Running | — |
| #904 | fern | Huber delta sweep: 0.25/0.5/1.0/2.0 on wider-model baseline | Running | — |
| #828 | edward | AdamW weight_decay=1e-4 on compound baseline | Awaiting rebase+re-run | 106.9078 (pre-compound) |
| #795 | thorfinn | Per-sample loss norm + Huber on compound stack | **Awaiting final rebase+re-run** | **93.3991** (potential winner) |
| #794 | tanjiro | LR warmup (2 epochs) + Huber on compound stack | Awaiting re-run | 136.25 (pre-compound) |
| #792 | frieren | Huber + grad_clip 1.0 on compound stack | **Awaiting final rebase+re-run** | **90.7796** (potential winner -12.0%) |
| #789 | askeladd | Gradient clipping (max_norm=1.0) on compound stack | **Awaiting v4 experiment run** | 109.67 (pre-compound) |

## Merged Winners (Round 1)

| PR | Hypothesis | val_avg/mae_surf_p | vs prior | 
|----|------------|-------------------|----------|
| #882 | EMA(0.999) on compound baseline | **103.2182** | -0.86% |
| #808 | bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12 | 104.1120 | -4.97% |
| #827 | Huber loss + surf_weight=30 | 109.5716 | -5.26% |
| #788 | Huber loss (delta=1.0) | 115.6496 | -8.85% |

## Closed Dead Ends

| PR | Hypothesis | Why Closed |
|----|------------|------------|
| #886 | Pressure-channel loss weighting (p_weight=2/3/5) | All variants 6-22% worse than baseline; surf_weight already captures pressure focus |
| #790 | surf_weight 10→30/50 on MSE | MSE results 11.5% above Huber baseline |

## Key Infrastructure Fixes Applied

1. **NaN propagation fix** (PR #791): `accumulate_batch` 0*NaN=NaN bug fixed in `evaluate_split`
2. **eval sanitization** (PR #792): `--grad_clip 1.0` + upstream pred/GT sanitization resolves NaN propagation
3. **Test split data bug** (PR #792): `test_geom_camber_cruise/000020.pt` has 761 Inf values in p — correctly skipped
4. All new PRs must include both fixes

## Key Research Insights

1. **Compound improvements stack well** — Huber → surf_weight=30 → wider model → EMA each added 0.86–8.85%
2. **surf_weight=30 on narrow model gave -5.26%** — has NEVER been tested on wide model (n_hidden=256). PR #954 closes this gap
3. **Channel-level pressure weighting (p_weight) is harmful** — surf_weight=30 at node level already routes gradient to pressure; stacking channel reweighting is redundant and degrades performance
4. **Deeper models (n_layers>5) are impractical** within epoch budget — 2.4x slowdown prevents convergence
5. **bf16 doubles throughput** — enables more epochs for wider models in 30-min budget
6. **EMA(0.999) gives marginal but consistent benefit** — small smoothing gain at zero cost
7. **Huber loss is the foundation** — all experiments should build on it

## Potential Next Research Directions (Post-Round-1 completion)

Priority order (based on orthogonality and prior evidence):

1. **SwiGLU activation** (from prior competition win) — test on current compound baseline, especially if orthogonal to loss/weight changes
2. **Fourier positional encoding (sigma=0.7)** — contributed to prior competition winner; test on Huber baseline
3. **n_layers=3 + slice_num=16** (prior competition winning config) — explicit test with Huber loss on this track
4. **Cosine annealing without warmup** — vs current warmup+cosine schedule; clean comparison on compound baseline
5. **Longer training if budget allows** — epochs=12 may still be epoch-starved; test epochs=15-20 with early stopping
6. **Geometry-aware positional encoding** — encode chord, gap, stagger into node features before attention
7. **Reynolds number conditioning** — add Re as a global conditioning variable (film conditioning or prefix token)
8. **Node subsampling during training** — keep all surface nodes, randomly subsample interior volume nodes to speed training
9. **Learning rate schedule tuning** — 1cycle vs cosine vs warmup+cosine; optimizer momentum/beta sweeps
10. **Label noise / augmentation** — simulate Reynolds number perturbation, small geometric jitter
11. **Ensemble of EMA + final checkpoint** — free metric gain if inference budget permits
12. **Higher EMA decay** — try 0.9999 on compound baseline if 0.999 was marginal
13. **Separate surface/volume prediction heads** — shared trunk, specialized head for surface p vs volume fields
14. **Sparse attention** over geometry graph — prune long-range edges using physical distance thresholds
