# SENPAI Research State

- 2026-04-29 11:35
- Most recent research direction from human researcher team: None (no open GitHub Issues)
- Current research focus: Compounding wins. Round-r5 reviews so far: **2 winners** (PR #1121 huber_delta=0.1, PR #1120 n_layers=2) and **5 closes** (#1118 epochs=50, #1122 lr=1e-3, #1123 n_hidden=320, #1124 weight_decay=0, #1125 surf_weight=5). New baseline (PR #1120): val=56.4257, test=49.6211. All 8 students assigned (PRs #1119, #1130-#1136).

## Current Research Focus and Themes

### The Problem
TandemFoilSet CFD surrogate: predict Ux, Uy, pressure at every mesh node given geometry + flow conditions. Primary metric: `val_avg/mae_surf_p` (surface pressure MAE, lower is better — equal-weight mean over 4 val splits). Compete target: `test_avg/mae_surf_p` ~ 40.93 (Transolver paper). Current best: 51.3554 test / 58.4790 val. Gap to close: ~20%.

### NEW Compound Baseline (PR #1120 — merged 2026-04-29)
- Architecture: n_hidden=256, n_head=8, **n_layers=2** (NEW from PR #1120), slice_num=16, mlp_ratio=2
- Training: Huber loss (**delta=0.1** recommended via CLI from PR #1121), epochs=30, grad_clip=1.0, ema_decay=0.999, per_sample_norm=True
- lr=5e-4, batch_size=4, surf_weight=10.0, bf16 mixed precision
- 26/30 epochs in 30-min budget (vs 22/30 for n_layers=3); cosine fully spent (LR=2.16e-5 at termination)
- 1,141,299 params (-29% vs n_layers=3) | Peak VRAM 22.22 GB (-27%)

**Compounding caveat:** PR #1120 was measured with huber_delta=1.0 (not 0.1). The recommended baseline command compounds n_layers=2 + huber_delta=0.1 — we expect val<56.4257 from the compound but haven't yet measured it. Wave-2 + edward's run will test compound effects.

### Per-split Val Breakdown (PR #1120 best — equal-weight mean)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | 59.68 | 53.47 |
| geom_camber_rc | **70.42** | **64.11** |
| geom_camber_cruise | 38.61 | 32.81 |
| re_rand | 57.00 | 48.10 |
| **avg** | **56.4257** | **49.6211** |

`geom_camber_cruise` remains easiest. `geom_camber_rc` is still the hardest split (val 70, test 64) — narrow improvement vs PR #1121's 71/65. Closing this split would unlock big aggregate gains.

### What We Know Works
- **`n_layers=2`** (NEW from PR #1120, -3.51% val, -3.38% test): shallower → faster epochs → more cosine decay completed → low-LR fine-tuning epochs unlocked. -27% VRAM, -29% params.
- **`--huber_delta 0.1`** (NEW from PR #1121, -5.04% val, -5.52% test). Tighter Huber clamp on PSN-normalized residuals. Cruise (-14.4% test) and re_rand (-9.4% test) gained most.
- `--per_sample_norm`: equalizes 15× Re-driven gradient-magnitude spread across samples.
- EMA weight averaging (decay=0.999): marginal but consistent.
- Huber loss + surf_weight=10 baseline.
- `n_hidden=256, n_head=8, slice_num=16, mlp_ratio=2`: best architecture found (depth simplified to 2).
- `grad_clip=1.0`: stabilizes against extreme pressure values.
- BF16 mixed precision.
- Cosine annealing T_max=epochs.
- epochs=30 with n_layers=2 reaches 26/30 in budget (cosine almost fully spent, LR=2e-5).

### What Just Failed (closed 2026-04-29)
- `lr=1e-3` (PR #1122): +14% regression. Without warmup, early epochs are noisy (val=321 at ep1) and the model never catches up in 30 min.
- `n_hidden=320` (PR #1123): +12% regression. Compute-bound — only 18/30 epochs in budget.
- `surf_weight=5` (PR #1125): +0.7% regression. Volume-p improved -8% but didn't transfer to surface — Pareto trade.
- `epochs=50` (PR #1118): +5.8% regression. T_max=50 stretches cosine; run terminates at LR=2.97e-4 (3.6× higher than baseline). Confirms that "match T_max to budget" is the right principle.
- `weight_decay=0` (PR #1124): +2.5% regression. OOD-heavy (geom_camber_rc test +4.5%). Confirms wd=1e-4 is right-sized — going UP may help (askeladd reassigned to wd=5e-4).

### Round 5 Experiments In-Flight (2026-04-29 11:30)

**Wave 2 (assigned, on n_layers=3 + huber_delta=0.1 fork — predates PR #1120):**
| PR | Student | Experiment | Predicted Δ vs PR #1121 baseline |
|----|---------|------------|----------------------------------|
| #1130 | fern | huber_delta=0.05 (even tighter) | -1 to -3% |
| #1131 | alphonse | mlp_ratio=4 (cheaper capacity) | -2 to -4% |
| #1132 | frieren | surf_weight=20 (other direction) | -1 to -3% |
| #1133 | tanjiro | slice_num=24 (attention capacity) | -1 to -3% |

**Wave 3 (newly assigned on PR #1120 + huber_delta=0.1 compound baseline):**
| PR | Student | Experiment | Predicted Δ vs PR #1120 |
|----|---------|------------|--------------------------|
| #1134 | edward | epochs=26 (cosine-aligned to budget) | -1 to -3% |
| #1135 | nezuko | batch_size=8 (use freed VRAM) | -1 to -3% |
| #1136 | askeladd | weight_decay=5e-4 (sweep up — OOD focus) | -1 to -2% |

**Wave 1 still running (predates PR #1120):**
| PR | Student | Experiment | Notes |
|----|---------|------------|-------|
| #1119 | thorfinn | cosine eta_min=5e-5 | LR floor — n_layers=3 stack |

**Important — comparison frame for wave-2 results:** All wave-2 PRs were branched before PR #1120 merged, so they run with `n_layers=3` and measure against PR #1121's val=58.4790 / test=51.3554. **A wave-2 result needs to beat val=56.4257 (current PR #1120 baseline) to merge directly.** Otherwise it's informative but we'd compound it with n_layers=2 in a fresh PR.

## Potential Next Research Directions

### High Priority — pursue after wave 2 / 3 results
1. **Compound winners** — once wave-2 finishes, take any single-knob winners and compound with n_layers=2 in fresh PRs.
2. **huber_delta=0.03** if PR #1130 wins; otherwise stop tightening.
3. **surf_weight=30** (revisit PR #827 finding) if PR #1132 wins at sw=20.
4. **slice_num=32** if PR #1133 wins at slice_num=24.
5. **mlp_ratio=6 or 8** if PR #1131 wins.
6. **n_layers=1** (push throughput further) — nezuko suggested this. Risk: too little capacity. If wins, we get even more epochs.
7. **batch_size=8** with n_layers=2 — VRAM dropped to 22 GB so headroom exists. May improve optimization stability.
8. **Linear warmup + cosine** — PR #1122's failure suggests warmup would unlock higher LR. Needs custom scheduler.
9. **Per-channel huber_delta** — fern suggested decoupling: huber_delta_p=0.1 (tight, primary) and huber_delta_Uxy=0.5 (looser). Needs train.py change.
10. **Annealed huber_delta** schedule (1.0 → 0.1) — combines warmup robustness with late-stage precision.

### Medium Priority
11. **Cosine warm restarts (SGDR)** — periodic LR resets to escape shallow optima. Edward suggested in #1118 close-out.
12. **Focal-style surface weighting** — weight hardest surface nodes more. Frieren suggested. Needs custom loss code.
13. **Re-conditioned normalization** — separate norm stats per Re bucket instead of per_sample_norm.
14. **Multi-scale slice_num** — different slice_num per layer (coarse-to-fine attention hierarchy).
15. **Targeted geom_camber_rc improvement** — that split lags hardest. Heavier sampling weight on rc-like training samples or split-aware curriculum.

### Lower Priority / Bold Ideas
13. **Physics-informed regularization** — continuity equation residual or Bernoulli constraint as auxiliary loss.
14. **Ensemble of 3-5 seeds** — average predictions; ~free gain if inference allows.
15. **Curriculum learning** — easy (low-Re, single-foil) → hard (high-Re, tandem).
16. **SGDR / warm restarts** — periodic cosine restarts to escape shallow optima (mentioned but never tested).
17. **Augmentation** — geometric (mirror, scale) or learnable input perturbation to grow effective training set beyond 1500 samples.

## Notes on the geom_camber_rc gap

`geom_camber_rc` is consistently the hardest split (test 64.96, vs cruise 32.35 — almost 2× worse). It's also the split that *regressed* +1.05% with huber_delta=0.1. If the next 1-2 winners don't help this split, dedicate a hypothesis to it specifically — e.g. heavier sampling weight on rc-like training samples, or a split-aware curriculum.
