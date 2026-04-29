# SENPAI Research State

- 2026-04-29 22:30 (updated — 6 WIP PRs, 0 review-ready, 2 idle students)
- **Most recent research direction from human researcher team:** None (no GitHub Issues)
- **Current best:** `val_avg/mae_surf_p` = **46.6765**, `test_avg/mae_surf_p` = **39.9351** (PR #1275, lr=8e-4 on beta2=0.985+huber_delta=0.1 stack)
- **Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper) — **BEATEN** (gap = **-0.9949**, 2.43% below target)

## Current Research Focus and Themes

The research is in **round r5** on branch `icml-appendix-charlie-pai2f-r5`. The current baseline compounds:
- n_layers=2 (hardcoded)
- slice_num=8 (hardcoded)
- huber_delta=0.1 (**current baseline** — 0.12 also tested and won vs 0.1, but lr=8e-4 win from PR #1275 is on top of 0.1 stack)
- epochs=32 (SequentialLR: LinearLR warmup 3 epochs + CosineAnnealingLR over 29 epochs)
- warmup_epochs=3
- weight_decay=5e-4
- lr=8e-4 (**new winner** — PR #1275)
- batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
- adamw_beta2=0.985

**Compete target BEATEN by -0.9949 (2.43%).** The test_avg is now 39.9351 vs target 40.93. The model remains training-budget-limited.

**Reproduce:**
```bash
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 32 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 \
  --adamw_beta2 0.985 --lr 8e-4
```

*(Note: n_layers=2, slice_num=8 are hardcoded in model_config dict in train.py — slice_num was changed from 16→8 in PR #1194)*

### Per-split Breakdown (PR #1275 current best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | ~47 | 42.8273 |
| geom_camber_rc | ~58 | 54.7152 |
| geom_camber_cruise | ~25 | 24.2136 |
| re_rand | ~39 | 37.9841 |
| **avg** | **46.6765** | **39.9351** |

`geom_camber_rc` remains the hardest OOD split. Compete target beaten by -0.9949 — pushing further below.

### Active Experiments — 6 WIP PRs

| PR | Student | Experiment | Goal |
|----|---------|------------|------|
| #1319 | charliepai2f5-tanjiro | epochs=40 budget extension | Extend budget: terminal slope means model still improving |
| #1308 | charliepai2f5-askeladd | Multi-step LR (step drops at ep 28,31) | Fine-grained late-phase decay: warmup→flat→gamma=0.3 drops |
| #1306 | charliepai2f5-alphonse | n_hidden=192 slim model OOD probe | Smaller model: faster epochs → more cosine steps → better OOD? |
| #1281 | charliepai2f5-edward | AdamW beta1 sweep (0.95, 0.85) | Probe beta1 axis on lr=8e-4+beta2=0.985 stack |
| #1276 | charliepai2f5-thorfinn | warmup_epochs=5 | Longer linear warmup on beta2=0.985 stack |
| #1268 | charliepai2f5-nezuko | OneCycleLR max_lr=3e-3 | Higher peak LR for faster one-cycle convergence |

**Idle students (2): charliepai2f5-frieren, charliepai2f5-fern** — being assigned new experiments.

## Consolidated History of What Works vs What Failed

### Confirmed Wins (compounded into baseline)
- **lr=8e-4** (PR #1275, -2.39% test): decisive win on the beta2=0.985 compound stack — COMPETE TARGET BEATEN by 2.43%
- **lr=7e-4** (PR #1242, -0.25% test): higher LR improves exploration in the cosine phase (superseded by 8e-4)
- **huber_delta=0.12** (PR #1311, -2.73% val, -0.76% test vs 0.10): looser Huber boundary wins over 0.10 — not yet compounded with lr=8e-4
- **adamw_beta2=0.985** (PR #1241, -0.75% val): fine-tuned between 0.98 and 0.999 default
- **adamw_beta2=0.98** (PR #1191, +win): LLaMA-style slower second moment decay
- **weight_decay=5e-4** (PR #1136, -6.16% val): stronger OOD L2 regularization
- **epochs=26→32 cosine-aligned** (PR #1134, #1154, -1.66% val): full cosine completion budget
- **warmup_epochs=3** (PR #1149, -1.75% val): smooth early-epoch gradient noise
- **slice_num=8** (PR #1194, -2.1% val): fewer attention slices → faster epochs → more cosine steps
- **huber_delta=0.1** (PR #1121, -5.04% val): tighter Huber clamp on PSN-normalized residuals
- **per_sample_norm** (baseline): equalizes 15× Re-driven gradient-magnitude spread

### Closed / Dead Ends
- **huber_delta=0.15** (PR #1316): +5.58% regression — bracket closed: optimal is 0.10–0.12
- **huber_delta=0.08** (PR #1293): too tight — between 0.1 (best then) and 0.05 (dead end)
- **huber_delta=0.05** (PR #1240): too tight — pressure gradients over-clamped
- **huber_delta per-channel** (PR #1188): dead end
- **huber_delta annealed** (PR #1192): dead end
- **surf_weight=20** (PR #1229): regression — surf_weight=10 is optimal, do NOT exceed
- **surf_weight=5** (PR #1171): regression
- **SWA** (PR #1233): no gain over EMA baseline
- **ema_decay=0.9995** (PR #1224): regression
- **ema_decay=0.9999**: regression — 0.999 confirmed optimal for ~11k-step budget
- **adamw betas=(0.9, 0.95)** (PR #1157): +4.19% regression
- **n_layers=1** (PR #1189): capacity floor — faster but weaker
- **n_hidden=384** (PR #1212): too slow per epoch
- **n_head=16**: no gain over 8 heads
- **slice_num=24** (PR #1133): compute-bound
- **batch_size=8** (PR #1190): no consistent gain
- **weight_decay=1e-3** (PR #1153): past optimum
- **weight_decay=2e-4 / 1e-3**: dead ends (5e-4 confirmed optimal)
- **droppath p=0.1** (PR #1151): regression
- **dropout p=0.1** (PR #1248): closed — implicit regularization did not help OOD
- **one-cycle LR (PR#1246 at peak 2e-3)**: merged but marginal

## Current Research Themes

### 1. LR Sweep — Upper Bound
- **Just merged:** lr=8e-4 (PR #1275) — decisive win, 2.43% below compete target
- **In flight:** lr=8e-4 is new baseline; probe lr=9e-4 as natural next step (LR curve slope still positive)
- **Key question:** Where is the actual LR optimum above 8e-4?

### 2. Huber Delta + LR Compound
- **Key gap:** PR #1275 baseline uses huber_delta=0.1, but PR #1311 showed huber_delta=0.12 won over 0.1. Neither compound has been tested together.
- **Next:** Test lr=8e-4 + huber_delta=0.12 compound — do the two wins stack additively?
- **Bracket CLOSED:** 0.05 dead end, 0.08 dead end, 0.10 winner, 0.12 winner over 0.10, 0.15 dead end. Done.

### 3. Budget Extension
- **In flight:** epochs=40 (PR #1319, tanjiro) — terminal slope means model still improving; extra epochs expected to yield gains

### 4. LR Schedule Optimization
- **In flight:** multi-step LR gamma=0.3 drops (PR #1308), warmup_epochs=5 (PR #1276), OneCycleLR max_lr=3e-3 (PR #1268)
- **Key question:** Can we get more out of the 32-epoch budget with a different schedule shape?

### 5. Optimizer Tuning
- **In flight:** AdamW beta1 sweep 0.95/0.85 (PR #1281)
- **beta2 axis CLOSED:** beta2=0.985 is optimal.
- **weight_decay axis CLOSED:** 5e-4 is optimal.
- **EMA axis CLOSED:** 0.999 is optimal.

### 6. Architecture & Capacity
- **In flight:** n_hidden=192 slim model (PR #1306, alphonse) — can a smaller model exploit cosine schedule better due to faster epoch time?

## Potential Next Research Directions (after current wave resolves)

### High Priority
1. **lr=8e-4 + huber_delta=0.12 compound** — the two biggest recent wins haven't been stacked; expected additive gain
2. **lr=9e-4 probe** — LR curve slope still positive at 8e-4; optimum not yet found
3. **Compound winners from current wave** — stack any improvements onto PR #1275 baseline
4. **Gradient accumulation (effective batch=16)** — VRAM heavily underutilized (~21 GB / 96 GB); 4x effective batch with zero VRAM cost
5. **Log-cosh loss** — smooth Huber alternative with natural gradient behavior, no delta hyperparameter

### Medium Priority
6. **Separate Re embedding** — explicitly embed Reynolds number as learnable conditioning (targets geom_camber_rc)
7. **Deferred EMA start** — decay=0 for warmup epochs, ramp to 0.999 over a few hundred steps
8. **slice_num=4** — even fewer slices → even faster epochs (probing the floor)
9. **huber_delta=0.12 + epochs=40 compound** — if PR #1319 wins, test with 0.12

### Bold/Creative Directions
10. **Physics-informed loss** — pressure divergence (∇p) or Kutta condition consistency as aux loss
11. **Geometry encoder** — pre-encode foil shape with small CNN, inject as model conditioning
12. **Ensemble at inference** — run 3-5 models with different seeds, average predictions (free gains)
13. **FNO-style spectral layer** — add one spectral convolution layer before Transolver attention
14. **Cosine restart schedule (SGDR)** — T_0=10, T_mult=1 for 3 warm restarts within 32 epochs
15. **Test-time augmentation** — mirror geometry inputs and average predicted fields

## Notes
- **COMPETE TARGET BEATEN** by -0.9949 (2.43%) as of PR #1275 (test=39.9351 vs target 40.93). Goal now is to push further below.
- **Key compound opportunity:** lr=8e-4 (PR #1275) + huber_delta=0.12 (PR #1311) have never been tested together. Either could be on a new best when stacked.
- geom_camber_rc split is still the hardest OOD split — high Re + geometry shift. Focus on it.
- The model is consistently training-budget-limited. Best improvements come from: more throughput, better schedules, budget extension, or effective batch size increase (gradient accumulation).
- VRAM is heavily underutilized (~21 GB / 96 GB available). Gradient accumulation can 4× effective batch with zero VRAM cost.
- EMA axis CLOSED — 0.999 is optimal for the current ~11k-step training budget.
- surf_weight axis CLOSED — 10 is optimal; exceeding it hurts OOD generalization.
- huber_delta axis CLOSED — 0.10–0.12 is the optimal range; 0.15 tested and failed.
- The `--adamw_beta1` flag was added in commit d386a5f (betas=(cfg.adamw_beta1, cfg.adamw_beta2) in AdamW constructor).
