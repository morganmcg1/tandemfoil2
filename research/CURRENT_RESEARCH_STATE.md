# SENPAI Research State

- 2026-04-29 (updated — 11 WIP PRs, 0 review-ready, 0 idle students; current baseline val=47.4955 test=41.2226 from PR #1242; compete gap=0.2926 / +0.71% above target)
- **Most recent research direction from human researcher team:** None (no GitHub Issues)
- **Current best:** `val_avg/mae_surf_p` = 47.4955, `test_avg/mae_surf_p` = 41.2226 (PR #1242, lr=7e-4 on beta2=0.985 compound stack)
- **Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper) — currently +0.71% above target (gap = 0.2926)

## Current Research Focus and Themes

The research is in **round r5** on branch `icml-appendix-charlie-pai2f-r5`. The current baseline compounds:
- n_layers=2 (hardcoded)
- slice_num=8 (hardcoded)
- huber_delta=0.1
- epochs=32 (SequentialLR: LinearLR warmup 3 epochs + CosineAnnealingLR over 29 epochs)
- warmup_epochs=3
- weight_decay=5e-4
- lr=7e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
- adamw_beta2=0.985

**Reproduce:**
```bash
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 32 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 \
  --adamw_beta2 0.985 --lr 7e-4
```

The model is **training-budget-limited** — it still improves at epoch 32 every run. The compete gap is very small: 0.2926 (0.71%). We are extremely close to the Transolver paper target.

### Per-split Breakdown (PR #1242 current best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | ~46 | 44.0112 |
| geom_camber_rc | ~65 | 56.3252 |
| geom_camber_cruise | ~27 | 25.1105 |
| re_rand | ~52 | 39.4435 |
| **avg** | **47.4955** | **41.2226** |

`geom_camber_rc` remains the hardest OOD split. Compete gap is just 0.2926 on test.

### Active Experiments — 11 WIP PRs (Wave 10 on PR#1242 baseline)

| PR | Student | Experiment | Goal |
|----|---------|------------|------|
| #1293 | charliepai2f5-askeladd | huber_delta=0.08 | Sweet spot probe between 0.1 (best) and 0.05 (dead end) |
| #1281 | charliepai2f5-edward | AdamW beta1 sweep (0.95, 0.85) | Probe beta1 axis on lr=7e-4+beta2=0.985 stack |
| #1279 | charliepai2f5-frieren | warmup_epochs=2 | Shorter warmup, more cosine epochs on lr=7e-4 stack |
| #1276 | charliepai2f5-thorfinn | warmup_epochs=5 | Longer linear warmup on beta2=0.985 stack |
| #1275 | charliepai2f5-fern | lr=8e-4 | Continue LR probe above 7e-4 on beta2=0.985 stack |
| #1268 | charliepai2f5-nezuko | OneCycleLR max_lr=3e-3 | Higher peak LR for faster one-cycle convergence |
| #1245 | charliepai2f5-tanjiro | Cosine LR floor eta_min=5e-5 | Prevent optimizer stall at epoch end |
| #1222 | charliepai2f5-alphonse | AdamW beta2=0.97 | Faster adaptation between 0.95 fail and 0.98 win |
| #1220 | nezuko | n_head=16 | More attention heads on full PR#1191 compound stack |
| #1219 | edward | Lower base LR 3e-4 | Probe below current optimal on PR#1191 stack |
| #1217 | alphonse | epochs=40 | Budget extension on PR#1191 compound stack |

## Consolidated History of What Works vs What Failed

### Confirmed Wins (compounded into baseline)
- **lr=7e-4** (PR #1242, -0.25% test): higher LR improves exploration in the cosine phase
- **adamw_beta2=0.985** (PR #1241, -0.75% val): fine-tuned between 0.98 and 0.999 default
- **adamw_beta2=0.98** (PR #1191, +win): LLaMA-style slower second moment decay
- **weight_decay=5e-4** (PR #1136, -6.16% val): stronger OOD L2 regularization
- **epochs=26→32 cosine-aligned** (PR #1134, #1154, -1.66% val): full cosine completion budget
- **warmup_epochs=3** (PR #1149, -1.75% val): smooth early-epoch gradient noise
- **slice_num=8** (PR #1194, -2.1% val): fewer attention slices → faster epochs → more cosine steps
- **huber_delta=0.1** (PR #1121, -5.04% val): tighter Huber clamp on PSN-normalized residuals
- **per_sample_norm** (baseline): equalizes 15× Re-driven gradient-magnitude spread

### Closed / Dead Ends
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
- **one-cycle LR (PR#1246 at peak 2e-3)**: merged but marginal — higher peak PR#1268 in flight

## Current Research Themes

### 1. LR Schedule Optimization
- **In flight:** cosine LR floor eta_min=5e-5 (PR #1245), one-cycle LR peak 3e-3 (PR #1268), warmup_epochs=2 (PR #1279), warmup_epochs=5 (PR #1276), lr=8e-4 (PR #1275)
- **Key question:** Can we get more out of the 32-epoch cosine schedule? Optimal lr=7e-4 found — can we go higher? Can warmup duration be tuned?

### 2. Optimizer Tuning
- **In flight:** AdamW beta2=0.97 (PR #1222), AdamW beta1 sweep 0.95/0.85 (PR #1281)
- **beta2 axis:** beta2=0.985 won. PR #1222 probes 0.97 (between known fail 0.95 and known win 0.98).
- **beta1 axis:** newly opened — commit d386a5f added `--adamw_beta1` CLI flag. Default 0.9; try 0.95 (less gradient memory) and 0.85.
- **weight_decay axis CLOSED:** 5e-4 is optimal.
- **EMA axis CLOSED:** 0.999 is optimal.

### 3. Loss Function Engineering
- **CLOSED — surf_weight axis:** surf_weight=10 is optimal. Do NOT increase above 10.
- **In flight:** huber_delta=0.08 (PR #1293) — between known-good 0.1 and dead-end 0.05

### 4. Architecture & Capacity
- **In flight:** n_head=16 (PR #1220)
- **Dead ends:** n_hidden=384 (too slow), n_layers=1 (capacity floor)

## Potential Next Research Directions (after current wave resolves)

### High Priority
1. **Compound winners from wave 10** — stack any improvements onto the PR #1242 baseline
2. **lr=6e-4** — fill in the gap between 5e-4 (prior) and 7e-4 (current best) for curve-fitting
3. **Gradient accumulation (effective batch=16)** — VRAM heavily underutilized (~21 GB / 96 GB); 4x effective batch with zero VRAM cost
4. **Log-cosh loss** — smooth Huber alternative with natural gradient behavior

### Medium Priority
5. **Separate Re embedding** — explicitly embed Reynolds number as learnable conditioning
6. **Multi-step LR** — step at epochs 24, 28, 31 for fine-grained late decay
7. **Deferred EMA start** — decay=0 for warmup epochs, ramp to 0.999 over a few hundred steps
8. **slice_num=32** — more expressive attention with more slices (cost: slower epochs)

### Bold/Creative Directions
9. **Physics-informed loss** — pressure divergence (∇p) or Kutta condition consistency as aux loss
10. **Geometry encoder** — pre-encode foil shape with small CNN, inject as model conditioning
11. **Ensemble at inference** — run 3-5 models with different seeds, average predictions
12. **FNO-style spectral layer** — add one spectral convolution layer before Transolver attention
13. **Cosine restart schedule (SGDR)** — T_0=10, T_mult=1 for 3 warm restarts within 32 epochs
14. **Test-time augmentation** — mirror geometry inputs and average predicted fields

## Notes
- The compete gap is now only 0.2926 (0.71% above target). This is within striking distance.
- geom_camber_rc split is the hardest OOD split — high Re + geometry shift. Focus on it.
- The model is consistently training-budget-limited. Best improvements come from: more throughput, better schedules, or effective batch size increase (gradient accumulation).
- VRAM is heavily underutilized (~21 GB / 96 GB available). Gradient accumulation can 4× effective batch with zero VRAM cost.
- EMA axis is CLOSED — 0.999 is optimal for the current ~11k-step training budget.
- surf_weight axis is CLOSED — 10 is optimal; exceeding it hurts OOD generalization.
- The `--adamw_beta1` flag was added in commit d386a5f (betas=(cfg.adamw_beta1, cfg.adamw_beta2) in AdamW constructor).
