# SENPAI Research State

- 2026-04-29 16:10
- Most recent research direction from human researcher team: None (no open GitHub Issues)
- Current research focus: Compounding optimization-side wins. New baseline (PR #1191, adamw_beta2=0.98) has dropped val_avg/mae_surf_p to **49.0719**, test_avg to **42.8204** — compete gap to Transolver target now **1.89 (4.6%)**, down from 3.77. Wave-6 produced the best result yet. **Currently 8 students all assigned** (8 WIP, 0 idle), all pods READY 1/1.

## Current Research Focus and Themes

### The Problem
TandemFoilSet CFD surrogate: predict Ux, Uy, pressure at every mesh node given geometry + flow conditions. Primary metric: `val_avg/mae_surf_p` (surface pressure MAE, lower is better — equal-weight mean over 4 val splits). Compete target: `test_avg/mae_surf_p` ~ 40.93 (Transolver paper). Current best: 44.7020 test / 51.0626 val. Gap to close: ~9.2%.

### Current Compound Baseline (PR #1149 — merged)
- Architecture: n_hidden=256, n_head=8, **n_layers=2**, slice_num=16, mlp_ratio=2 (1.14M params)
- Training: Huber loss with **delta=0.1**, epochs=30, **weight_decay=5e-4**, **warmup_epochs=3** (NEW from PR #1149), grad_clip=1.0, ema_decay=0.999, per_sample_norm
- lr=5e-4, batch_size=4, surf_weight=10.0, bf16 mixed precision
- Peak VRAM 22.22 GB

**Reproduce:**
```
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3
```
*(n_layers=2, slice_num=16 hardcoded in model_config dict in train.py)*

### Per-split Val/Test Breakdown (PR #1149 best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | 55.97 | 51.36 |
| geom_camber_rc | **66.02** | **59.84** |
| geom_camber_cruise | 31.00 | 25.64 |
| re_rand | 51.25 | 41.97 |
| **avg** | **51.0626** | **44.7020** |

`geom_camber_rc` remains worst — persistently the hardest OOD split. `geom_camber_cruise` is strong at 25.64 (below Transolver 40.93 avg). Compete gap on easiest split just 25.64 vs 40.93 Transolver-on-avg.

### What We Know Works (compounded baseline ingredients)
- **`warmup_epochs=3`** (PR #1149, -1.75% val, -3.09% test): 3-epoch linear LR warmup smooths early-epoch gradient noise. Best epoch = 30 (still improving at termination).
- **`weight_decay=5e-4`** (PR #1136, -6.16% val, -5.46% test): stronger L2 helped all OOD splits more than in-dist.
- **`epochs=26`** cosine-aligned (PR #1134, -1.66% val, -1.62% test): full cosine completion. Note: PR #1136+#1149 use 30 epochs.
- **`n_layers=2`** (PR #1120, -3.51% val, -3.38% test): shallower → faster epochs → more cosine decay completed.
- **`--huber_delta 0.1`** (PR #1121, -5.04% val, -5.52% test). Tighter Huber clamp on PSN-normalized residuals.
- `--per_sample_norm`: equalizes 15× Re-driven gradient-magnitude spread across samples.
- EMA weight averaging (decay=0.999): marginal but consistent.
- Huber loss + surf_weight=10 baseline.
- `n_hidden=256, n_head=8, slice_num=16, mlp_ratio=2`: best architecture.
- `grad_clip=1.0`: stabilizes against extreme pressure values.
- BF16 mixed precision, cosine T_max=epochs.

### Diagnostic Conclusion: Optimization-bound, not Capacity-bound
Wave-2/3/4 results confirmed: every "more capacity" hypothesis regressed; every optimizer/schedule improvement won. **Future hypotheses should target optimization (LR schedule, optimizer betas, regularization, warmup, loss formulation) rather than capacity.**

### What Just Failed (closed, on r5 advisor branch)
- `adamw betas=(0.9, 0.95)` (PR #1157): +4.19% regression — too fast second moment adaptation.
- `epochs=40` (PR #1154): +1.47% regression — T_max=40 stretches cosine, LR still ~1.2e-4 at typical completion.
- `weight_decay=1e-3` (PR #1153): +8.09% regression — past optimum at 5e-4.
- `droppath p=0.1` (PR #1151): +1.47% regression — stochastic depth hurt optimization stability.
- `L1-surf + Huber-vol` (PR #1150): +1.28% regression — channel-asymmetric loss marginal/negative.
- `surf_weight=5.0` (PR #1171): +1.55% regression — both sides exhausted (5, 10, 20 all tried).
- `lr=1e-3` (PR #1122): +14% regression — without warmup, early epochs noisy.
- `n_hidden=320` (PR #1123): +12% regression — compute-bound.
- `weight_decay=0` (PR #1124): +2.5% regression. Pointed UP → led to wd=5e-4 win.
- `huber_delta=0.05` (PR #1130): regression — both sides of optimum explored.
- `mlp_ratio=4` (PR #1131): regression — compute-bound.
- `surf_weight=20` (PR #1132): regression — both sides exhausted.
- `slice_num=24` (PR #1133): regression — compute-bound.
- `cosine eta_min=5e-5` (PR #1119): +22.9% regression — LR floor too high.
- SGDR cosine warm restarts: closed (PR on r5 wave).
- ema_decay=0.9999: closed.
- DropPath: closed.

### Round 6 Experiments In-Flight (2026-04-29 14:20)

**On PR #1149 (warmup_epochs=3) compound baseline:**
| PR | Student | Experiment | Predicted Δ vs PR #1149 |
|----|---------|------------|--------------------------|
| #1145 | alphonse (legacy) | dropout p=0.1 | awaiting results |
| #1146 | edward (legacy) | base lr=2e-4 | awaiting results |
| #1188 | askeladd | Per-channel Huber delta: delta_p=0.1, delta_uv=0.5 | -1 to -3% |
| #1189 | fern | n_layers=1 (throughput push) | -1 to -3% |
| #1190 | frieren | batch_size=8 (gradient quality) | -1 to -3% |
| #1191 | nezuko | AdamW betas=(0.9, 0.98) LLaMA-style | -1 to -2% |
| #1192 | thorfinn | Annealed huber_delta 1.0→0.1 | -1 to -3% |
| #1194 | tanjiro | slice_num=8 (throughput: fewer attention slices) | -1 to -2% |

**Key bets:**
- **Loss formulation**: askeladd (per-channel delta), thorfinn (annealed delta) — two orthogonal loss-shape angles.
- **Throughput**: fern (n_layers=1), tanjiro (slice_num=8) — push the optimization-bound diagnosis further; faster epochs = more gradient steps.
- **Optimizer**: nezuko (betas=0.98), frieren (batch_size=8) — optimizer quality improvements.
- **Awaiting**: edward (lr=2e-4), alphonse (dropout=0.1) — legacy PRs still running.

## Potential Next Research Directions

### High Priority — pursue after wave 6 results
1. **Compound winners** — once any wave-6 experiments win, compound with PR #1149 base.
2. **n_layers=1 + epochs=40** — if fern's n_layers=1 wins, test with more epochs (throughput gain enables budget extension).
3. **Per-channel huber_delta with more exploration** — if askeladd wins, try delta_uv=0.3 or 1.0.
4. **AdamW betas=(0.9, 0.999) sweep** — if nezuko's 0.98 wins, try 0.995 to refine the optimum.
5. **Targeted geom_camber_rc improvement** — that split lags hardest (test=59.84). If wave-6 wins don't help this split, dedicate a hypothesis to it: heavier training weight on rc-like samples, or split-aware curriculum.

### Medium Priority
6. **batch_size=8 + lr=1e-3 with warmup** — if frieren's conservative batch_size=8 wins, the LR-scaled variant may add more.
7. **Re-conditioned normalization** — separate norm stats per Re bucket instead of per_sample_norm.
8. **Annealed huber_delta with non-linear schedule** — cosine delta anneal instead of linear.
9. **Lookahead optimizer wrapper** — k-step inner-outer optimization.
10. **Multi-scale slice_num** — different slice_num per layer (coarse-to-fine attention hierarchy).

### Lower Priority / Bold Ideas
11. **Physics-informed regularization** — continuity equation residual or Bernoulli constraint as auxiliary loss.
12. **Ensemble of 3-5 seeds** — average predictions; ~free gain if inference allows.
13. **Curriculum learning** — easy (low-Re, single-foil) → hard (high-Re, tandem).
14. **Augmentation** — geometric (mirror, scale) or learnable input perturbation.
15. **Focal-style surface weighting** — weight hardest surface nodes more.

## Notes on the geom_camber_rc gap

`geom_camber_rc` is consistently the hardest split (test=59.84, vs cruise 25.64 — over 2× worse). It's the split that gains least from regularization and optimizer tweaks. This confirms it's representation-bottlenecked, not overfitting. After wave-6 completes, if rc still doesn't improve, dedicate a hypothesis specifically to it: heavier sampling weight on rc-like training samples, or a domain embedding approach.
