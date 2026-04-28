# SENPAI Research State
- 2026-04-28 22:30 UTC
- Most recent research direction from human researcher team: None received yet (no GitHub Issues found)
- Current research focus and themes: Round 1 — Baseline parameter sweeps and loss function experiments on the Transolver CFD surrogate for TandemFoilSet-Balanced

## Current Baseline

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **115.6496** | PR #788 — Huber loss, epoch 10 |
| `test_avg/mae_surf_p` | **40.927** | Prior competition best (nl=3, sn=16) |

The val baseline (115.65) is what current WIP experiments must beat.

## Active WIP PRs (9 students, all running)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #789 | askeladd | Gradient clipping (max_norm=1.0) | Running |
| #792 | frieren | n_layers=6 Transolver + grad clipping (NaN fix) | Running |
| #793 | nezuko | Finer physics partitioning: slice_num 64→128 | Running |
| #794 | tanjiro | LR warmup (2 epochs) + Huber loss | Revision of prior result |
| #795 | thorfinn | Per-sample norm + Huber loss combined | Revision of prior result |
| #808 | fern | bf16 mixed precision for wider model (n_hidden=256, n_head=8) | Running |
| #827 | alphonse | surf_weight sweep (20, 30, 50) on Huber baseline | Running |
| #828 | edward | AdamW weight_decay sweep (1e-4, 1e-3, 1e-2) on Huber baseline | Running |

Note: #791 (fern wider model) was sent back; follow-up is #808.

## Key Infrastructure Fix Applied

- `accumulate_batch` NaN bug: `0 * NaN = NaN` in IEEE 754 in `evaluate_split` — **fixed** in all subsequent PRs following PR #791.
- Known pre-existing bug: `test_geom_camber_cruise/mae_surf_p` returns NaN for some models. Under investigation — `data/scoring.py` doesn't guard non-finite predictions.

## Round 1 Summary

### Merged Winners
1. **PR #788** — Huber loss (delta=1.0): val_avg/mae_surf_p = 115.65 vs MSE baseline 126.88 (-8.85%)

### Sent Back for Revision
- **PR #795** (thorfinn): Per-sample normalization alone — above Huber baseline. Re-running with Huber+norm combined.
- **PR #794** (tanjiro): LR warmup 5 epochs + cosine — above Huber baseline. Re-running with 2-epoch warmup + Huber.

### Closed Dead Ends
- **PR #790** (edward): surf_weight 10→30/50 on MSE — 128.98, above Huber baseline. Re-assigned as #828 using Huber foundation.

## Key Research Insights So Far

1. **Huber loss is the new foundation** — 8.85% improvement. All future experiments should build on Huber loss.
2. **Wider models may overfit or be epoch-starved within timeout** — #791 showed limited epochs due to budget constraints.
3. **bf16 doubles throughput** — should allow ~2x epochs for wider models within the 30-min budget.
4. **Deeper models (n_layers>5) are impractical** — 2.4x slowdown per epoch makes convergence impossible in budget.
5. **From prior competition**: n_layers=3 + slice_num=16 was the winning config — should be explicitly tested here with Huber.

## Potential Next Research Directions (Post-Round-1)

Priority order:

1. **Reproduce prior competition winner explicitly**: n_layers=3, slice_num=16, SwiGLU+Fourier, Huber loss — HIGHEST PRIORITY
2. **SwiGLU activation** — contributed to prior competition win; test on Huber baseline
3. **Fourier positional encoding (sigma=0.7)** — prior competition showed benefit
4. **n_layers=3 + slice_num=16 combo** on this new track with Huber loss as foundation
5. **Cosine annealing without warmup** (vs current schedule) — clean comparison
6. **Pressure-only output head** — dual head: shared trunk for Ux/Uy, specialized head for p
7. **High-Re weighting** — per-sample weight by Re to stress-test high-Re regimes
8. **Separate surface/volume loss** — stronger weight on surface nodes in the combined loss
9. **Node subsampling during training** — keep all surface nodes, randomly subsample interior to speed training
10. **Geometry-aware positional encoding** — directly encode chord, gap, stagger into node features before attention
