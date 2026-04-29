# SENPAI Research State

- 2026-04-29 (updated — 8 students busy, 0 idle; all GPUs occupied; current baseline val=47.6501 test=41.4598 from PR #1241; compete gap=0.5298 / 1.29%)
- **Most recent research direction from human researcher team:** None (no GitHub Issues)
- **Current best:** `val_avg/mae_surf_p` = 47.6501, `test_avg/mae_surf_p` = 41.4598 (PR #1241, adamw_beta2=0.985 on full compound stack)
- **Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper) — currently +1.29% above target (gap = 0.5298)

## Current Research Focus and Themes

The research is in **round r5** on branch `icml-appendix-charlie-pai2f-r5`. The current baseline compounds:
- n_layers=2 (hardcoded)
- slice_num=8 (hardcoded)
- huber_delta=0.1
- epochs=32 (cosine T_max=32, LR→0 at completion), warmup_epochs=3
- weight_decay=5e-4
- lr=5e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
- adamw_beta2=0.985

**Reproduce:**
```bash
python train.py --loss huber --huber_delta 0.1 --epochs 32 --warmup_epochs 3 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 \
  --lr 5e-4 --batch_size 4 --surf_weight 10 --adamw_beta2 0.985
```

The model is **training-budget-limited** — it still improves at epoch 32 every run. The core challenge is to extract more learning within the 30-min wall-clock budget. The compete gap is very small: 0.5298 (1.29%). Closing it requires ~1.29% more improvement.

### Per-split Breakdown (PR #1241 current best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | ~46 | ~40 |
| geom_camber_rc | ~65 | ~57 |
| geom_camber_cruise | ~27 | ~22 |
| re_rand | ~52 | ~45 |
| **avg** | **47.6501** | **41.4598** |

`geom_camber_rc` remains the hardest OOD split. Compete gap is just 0.5298 on test.

### Active Experiments — 8 WIP PRs (Wave 9 on PR#1241 baseline)

| PR | Student | Experiment | Goal |
|----|---------|------------|------|
| #1222 | alphonse | AdamW beta2=0.97 | Probe optimizer momentum between 0.95 (fail) and 0.98 |
| #1242 | fern | lr=7e-4 on beta2=0.985 stack | Higher LR on optimal beta2 stack |
| #1245 | tanjiro | Cosine LR floor eta_min=5e-5 | Prevent optimizer stall at epoch end |
| #1246 | nezuko | One-cycle LR policy, peak 2e-3 | Aggressive LR schedule for training-budget regime |
| #1247 | askeladd | Tighter huber_delta=0.08 | Between 0.1 (win) and 0.05 (dead end) |
| #1248 | thorfinn | Dropout regularization p=0.1 | Implicit regularization for OOD generalization |
| #1251 | frieren | Gradient accumulation (effective batch=16) | Better gradient quality without VRAM cost |
| #1253 | edward | Log-cosh loss | Smooth Huber alternative with natural gradient behavior |

## Consolidated History of What Works vs What Failed

### Confirmed Wins (compounded into baseline)
- **weight_decay=5e-4** (PR #1136, -6.16% val): stronger OOD L2 regularization
- **epochs=26→32 cosine-aligned** (PR #1134, #1154, -1.66% val): full cosine completion budget
- **warmup_epochs=3** (PR #1149, -1.75% val): smooth early-epoch gradient noise
- **slice_num=8** (PR #1194, -2.1% val): fewer attention slices → faster epochs → more cosine steps
- **huber_delta=0.1** (PR #1121, -5.04% val): tighter Huber clamp on PSN-normalized residuals
- **per_sample_norm** (baseline): equalizes 15× Re-driven gradient-magnitude spread
- **adamw_beta2=0.98** (PR #1191, +win): LLaMA-style slower second moment decay
- **adamw_beta2=0.985** (PR #1241, -0.75% val): fine-tuned between 0.98 and 0.999 default

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

## Current Research Themes

### 1. Throughput / Schedule Optimization
- **In flight:** cosine LR floor eta_min=5e-5 (PR #1245), one-cycle LR peak 2e-3 (PR #1246), gradient accumulation batch 16 (PR #1251)
- **Key question:** Can we pack more gradient steps or better gradient quality into 30 min?
- **VRAM heavily underutilized:** ~21 GB / 96 GB. Gradient accumulation is the key lever.

### 2. Loss Function Engineering
- **CLOSED — surf_weight axis:** surf_weight=10 is optimal. Do NOT increase above 10.
- **CLOSED — huber_delta < 0.05:** dead end (huber_delta=0.1 is the optimal floor found so far)
- **In flight:** huber_delta=0.08 (PR #1247), log-cosh loss (PR #1253)

### 3. Optimizer & Gradient Quality
- **In flight:** AdamW beta2=0.97 (PR #1222), gradient accumulation batch 16 (PR #1251), lr=7e-4 on beta2=0.985 stack (PR #1242)
- **EMA axis CLOSED:** ema_decay=0.999 is confirmed optimal.
- **beta2 axis:** beta2=0.985 won (PR #1241). PR #1222 probes 0.97.
- **weight_decay axis CLOSED:** 5e-4 is optimal.

### 4. Architecture & Capacity
- **In flight:** Dropout regularization p=0.1 (PR #1248)
- **Dead ends:** n_hidden=384 (too slow), n_layers=1 (capacity floor)

## Potential Next Research Directions (after current wave resolves)

### High Priority
1. **Compound winners from wave 9** — stack any improvements onto the PR #1241 baseline
2. **warmup_epochs=2** — interpolation between 1 and 3
3. **AdamW beta1 sweep** — default 0.9; try 0.95 (less gradient memory, faster adaptation)
4. **lr=6e-4 or 8e-4** — probe between 5e-4 and 7e-4 to narrow optimum

### Medium Priority
5. **Separate Re embedding** — explicitly embed Reynolds number as learnable conditioning
6. **slice_num=32** — more expressive attention with more slices (cost: slower epochs)
7. **Multi-step LR** — step at epochs 24, 28, 31 for fine-grained late decay
8. **Deferred EMA start** — decay=0 for warmup epochs, ramp to 0.999 over a few hundred steps

### Bold/Creative Directions
9. **Physics-informed loss** — pressure divergence (∇p) or Kutta condition consistency as aux loss
10. **Geometry encoder** — pre-encode foil shape with small CNN, inject as model conditioning
11. **Ensemble at inference** — run 3-5 models with different seeds, average predictions
12. **FNO-style spectral layer** — add one spectral convolution layer before Transolver attention
13. **Cosine restart schedule (SGDR)** — T_0=10, T_mult=1 for 3 warm restarts within 32 epochs
14. **Test-time augmentation** — mirror geometry inputs and average predicted fields

## Notes
- The compete gap is now only 0.5298 (1.29% above target). This is within striking distance.
- geom_camber_rc split is the hardest OOD split — high Re + geometry shift. Focus on it.
- The model is consistently training-budget-limited. Best improvements come from: more throughput, better schedules, or effective batch size increase (gradient accumulation).
- VRAM is heavily underutilized (~21 GB / 96 GB available). Gradient accumulation can 4× effective batch with zero VRAM cost.
- EMA axis is CLOSED — 0.999 is optimal for the current ~11k-step training budget.
- surf_weight axis is CLOSED — 10 is optimal; exceeding it hurts OOD generalization.
