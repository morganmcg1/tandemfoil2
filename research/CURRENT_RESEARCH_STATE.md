# SENPAI Research State

- 2026-04-29 23:50 (updated — 8 active students, all WIP; current baseline val=45.5945 test=39.7038 from PR #1319; compete target BEATEN by 3.0%, gap=-1.2262)
- **Most recent research direction from human researcher team:** None (no GitHub Issues)
- **Current best:** `val_avg/mae_surf_p` = 45.5945, `test_avg/mae_surf_p` = 39.7038 (PR #1319, epochs=40 budget extension on full compound stack, stopped at epoch 33/40 by 30-min timeout)
- **Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper) — **BEATEN** (gap = -1.2262, -3.0%)

## Current Research Focus and Themes

The research is in **round r5** on branch `icml-appendix-charlie-pai2f-r5`. The current baseline (PR #1319) compounds:
- n_layers=2 (hardcoded), slice_num=8 (hardcoded)
- loss=huber, huber_delta=0.12
- epochs=40 (completed 33/40 before 30-min timeout) — val slope ~-0.49/epoch at termination
- warmup_epochs=3, SequentialLR: 3-epoch linear warmup + CosineAnnealingLR T_max=37
- weight_decay=5e-4
- lr=7e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
- adamw_beta2=0.985
- Peak VRAM: 20.98 GB / 96 GB available

The model remains **strongly training-budget-limited** — val slope was -0.49/epoch at epoch 33/40 when the 30-min timeout hit. The single biggest lever is getting more training epochs into the time budget.

Key insight: The wall-clock limit (30 min) is the binding constraint, not the epoch count. Approaches that improve per-epoch speed or per-step quality are critical.

### Active Experiments — 8 WIP PRs

**All on PR#1319 compound baseline (lr=7e-4 + beta2=0.985 + huber_delta=0.12 + epochs=40 + warmup=3 + wd=5e-4):**

1. **PR #1346** (charliepai2f5-tanjiro): Lookahead(AdamW) optimizer — slow-weight interpolation (k=5, alpha=0.5) targeting flatter minima for OOD generalization. First optimizer family change on this track.
2. **PR #1336** (charliepai2f5-frieren): warmup_epochs=4 on PR#1319 stack — filling warmup gap between 3 (baseline) and 5 (thorfinn)
3. **PR #1334** (charliepai2f5-askeladd): Gradient accumulation effective batch=16 — zero VRAM cost via grad_accum=4 (WARNING: reduces optimizer steps 4×)
4. **PR #1330** (charliepai2f5-fern): lr=9e-4 on PR#1319 stack — sent back to re-run on PR#1319 stack (huber_delta=0.12, epochs=40)
5. **PR #1306** (charliepai2f5-alphonse): n_hidden=192 — slim model OOD probe (lighter model → more epochs in budget?)
6. **PR #1281** (charliepai2f5-edward): AdamW beta1=0.95 on PR#1319 stack — sent back to re-run on PR#1319 stack (huber_delta=0.12, epochs=40)
7. **PR #1276** (charliepai2f5-thorfinn): warmup_epochs=5 — longer linear warmup on beta2=0.985 stack
8. **PR #1268** (charliepai2f5-nezuko): OneCycleLR max_lr=3e-3 — one-cycle schedule with higher peak LR

## Current Research Themes

### 1. Training Budget Extension (HIGHEST PRIORITY)
- **Key finding:** PR #1319 was still improving at -0.49/epoch when the 30-min timeout hit at epoch 33/40. If we can squeeze more epochs within 30 min, we get essentially free improvement.
- **Key question:** Can we reduce per-epoch wall-clock time? The model uses only ~21 GB of 96 GB VRAM — there may be room to increase throughput via larger batch or data loading optimizations.
- **Next priority after current wave:** epochs=50 or epochs=60 with cosine T_max aligned (if we can solve the timeout), OR mixed-precision (AMP/bf16) to speed up per-epoch training.

### 2. LR Sweep (ACTIVE)
- **In flight:** lr=9e-4 (PR #1330)
- **Completed wins:** lr=7e-4 (PR #1242 over 5e-4). lr=8e-4 tested on older baseline (PR #1275 closed/sent back).
- **Key question:** Is 7e-4 still the peak on the new epochs=40 budget, or does more training capacity allow a higher LR?
- **Caution:** LR >8e-4 showed instability on older stacks (OneCycleLR max_lr=3.5e-3 crashed).

### 3. Loss Function Engineering
- **CLOSED — surf_weight axis:** surf_weight=10 is optimal. Do NOT vary.
- **CLOSED — tighter huber_delta:** 0.05 (PR #1240) and 0.08 (PR #1293) both dead ends.
- **CLOSED — special losses:** log-cosh (PR #1253) dead end. Per-channel (PR #1188) dead end. Annealed (PR #1192) dead end.
- **WON — PR #1311:** huber_delta=0.12 → beat compete target. WON — PR #1319: compounded into full baseline.
- **Open question:** huber_delta=0.15 or 0.20 — could loosening further help? PR #1316 was closed/superseded. Worth revisiting on new baseline.

### 4. Optimizer & Gradient Quality
- **In flight:** beta1=0.95 sweep (PR #1281), OneCycleLR max_lr=3e-3 (PR #1268), Lookahead(AdamW) (PR #1346)
- **CLOSED:** EMA axis (0.999 optimal), weight_decay axis (5e-4 optimal), beta2 axis (0.985 optimal)
- **Next probe:** After beta1 resolves, consider Lion optimizer or AdamW with decoupled weight decay schedule.

### 5. Architecture & Capacity
- **In flight:** n_hidden=192 (PR #1306 — lighter model, runs faster → more epochs in budget?)
- **Dead ends:** n_hidden=384 (too slow), n_layers=1 (capacity floor)
- **VRAM severely underutilized:** ~21 GB / 96 GB. This is the key opportunity: can we find a model that uses more VRAM while training faster per epoch?
- **Next idea:** slice_num=16 or 32 — more expressive attention (uses more VRAM without increasing n_hidden)

### 6. LR Schedule Optimization
- **In flight:** warmup_epochs=4 (PR #1336), warmup_epochs=5 (PR #1276), OneCycleLR max_lr=3e-3 (PR #1268), gradient accumulation effective batch=16 (PR #1334)
- **Key question:** Can we get more out of the 32/40-epoch budget with a different schedule shape?

## Potential Next Research Directions (post-current wave)

### High Priority (attack the timeout bottleneck)
1. **AMP/bf16 training** — mixed precision to halve per-step wall-clock → fit 50-60 epochs in 30 min
2. **epochs=50 or 60 + cosine aligned** — if we get AMP working, extend the budget further
3. **Larger batch_size** — batch=8 or 16 increases GPU utilization and may speed up training (gradient accumulation ruled out — kills step count, but native larger batch might work)
4. **beta1=0.95 compound on PR#1319 stack** — edward's prior result (test=40.9696) was on old baseline; new deeper-trained stack likely yields further gain

### Medium Priority
5. **huber_delta=0.15 or 0.20 probe** — loosening delta further on new baseline
6. **slice_num=16 or 32** — more attention heads per slice (uses more VRAM, may improve expressivity)
7. **Reynolds number embedding** — explicit conditioning on Re as learnable embedding injected into attention
8. **Deferred EMA start** — EMA decay=0 for warmup, ramp to 0.999 — protects early warmup from EMA averaging noise
9. **warmup_epochs=4** — filling gap between 3 (baseline) and 5 (PR #1276 in flight)

### Bold/Creative Directions
10. **Physics-informed loss** — pressure divergence (∇p) or Kutta condition consistency as auxiliary loss
11. **FNO-style spectral layer** — add spectral convolution feature extractor before Transolver attention
12. **Geometry encoder** — pre-encode airfoil shape with small CNN, inject as model conditioning
13. **SGDR cosine restarts** — T_0=8, T_mult=1 for ~4 warm restarts within 40-epoch budget
14. **Ensemble inference** — 3-5 seeds, average predictions (post-training, no GPU cost per seed)
15. **Data augmentation** — random foil perturbation or Reynolds number jitter during training for OOD robustness

## Notes
- **Compete target BEATEN** by wide margin — PR #1319 test=39.7038, gap=-1.2262 (-3.0%). Focus: push val further down and improve OOD splits.
- geom_camber_rc split remains hardest OOD challenge — high Re + geometry shift simultaneously.
- The BINDING constraint is the 30-minute wall-clock timeout. Val was still -0.49/epoch when we hit the timeout. Speed improvements compound.
- VRAM is severely underutilized (~21/96 GB). This represents latent throughput capacity.
- Gradient accumulation (effective batch=16 via grad_accum=4) is risky — PR #1251 showed severe under-convergence from 12k→3k steps. PR #1334 re-testing on new baseline.
- Dead ends this round: PR #1188 (per-channel huber), PR #1192 (annealed huber), PR #1224 (ema_decay=0.9995), PR #1229 (surf_weight=20), PR #1233 (SWA old baseline), PR #1240 (huber_delta=0.05), PR #1248 (dropout p=0.1), PR #1251 (grad_accum batch 16 old baseline), PR #1253 (log-cosh loss), PR #1274 (OneCycleLR max_lr=3.5e-3), PR #1279 (warmup_epochs=2), PR #1293 (huber_delta=0.08), PR #1316 (huber_delta=0.15).
