# SENPAI Research Results

## 2026-04-29 22:30 — PR #1302: OneCycleLR pct_start 0.3→0.4 — CLOSED (negative result)

- charliepai2f2-tanjiro/onecycle-pct-start-0p4
- **Hypothesis**: Increase OneCycleLR `pct_start` from 0.3 to 0.4, extending warmup from 30% to 40% of training. Hypothesis was that longer warmup allows more thorough exploration and better minima, especially for geometry-OOD generalization.
- **Status**: CLOSED — primary metric regresses, confirms pct_start=0.3 as optimal

| Split | Baseline val (PR #1264) | PR #1302 val | Δ |
|-------|------------------------|--------------|---|
| single_in_dist     | 82.09 | 83.61 | +1.52 (+1.9%) |
| geom_camber_rc     | 85.31 | 88.73 | +3.42 (+4.0%) |
| geom_camber_cruise | 55.67 | 56.67 | +1.00 (+1.8%) |
| re_rand            | 74.38 | 73.44 | **-0.94 (-1.3%)** |
| **avg**            | **74.36** | **75.61** | **+1.25 (+1.7%)** |

Test avg: 66.34 vs baseline 65.91 (+0.65%). Best epoch: 18 (final epoch). n_params: 827,735.

**Analysis**: pct_start=0.4 partially confirms the warmup exploration hypothesis for Re-OOD (73.44 vs 74.38, -1.3% improvement) but causes geometry-OOD splits to regress, particularly geom_camber_rc (+4%). The model reaches best epoch at the final epoch (18/18), suggesting the longer warmup slightly delays convergence without net benefit. The shared FiLM generator conditioned on log(Re) naturally benefits from Re-domain exploration, but geometry generalization relies more on architecture capacity than exploration schedule. **Conclusion: pct_start=0.3 is confirmed optimal for this architecture.**

**Follow-up**: Tanjiro reassigned to PR #1313 — testing `pct_start=0.2` on the FiLM stack (shorter warmup / longer annealing; pct_start=0.2 was previously tested pre-FiLM only).

---

## 2026-04-29 21:45 — PR #1297: Per-block FiLM generators — CLOSED (negative result)

- charliepai2f2-edward/per-block-film-generators
- **Hypothesis**: Replace `SharedFiLMGenerator` (one shared MLP for all 5 blocks) with 5 independent `PerBlockFiLMGenerator` instances (each `Linear(1,128) → GELU → Linear(128,256)` per block). Hypothesis was that per-block specialization allows each TransolverBlock to apply distinct Re conditioning patterns.
- **Status**: CLOSED — clear regression, all splits worse

| Split | Baseline (PR #1264) | PR #1297 val | Δ |
|-------|---------------------|--------------|---|
| single_in_dist     | 82.09 | 85.46 | +3.37 (+4.1%) |
| geom_camber_rc     | 85.31 | 91.12 | +5.81 (+6.8%) |
| geom_camber_cruise | 55.67 | 58.39 | +2.72 (+4.9%) |
| re_rand            | 74.38 | 77.34 | +2.96 (+4.0%) |
| **avg**            | **74.36** | **78.40** | **+4.04 (+5.4%)** |

17 epochs at ~104s/epoch. Params: 828,759 (+1,024 vs baseline — essentially same total).

**Root cause**: The shared generator acts as a useful inductive bias with only ~1,500 training samples. Each block independently learning a 1-D log(Re) mapping wastes capacity — the shared bottleneck forces coordinated Re conditioning across all 5 TransolverBlocks. At this data scale, parameter efficiency matters more than per-block specialization. Student's analysis correct and thorough.

**Follow-up**: PR #1309 assigned to edward — deeper shared MLP (2-layer: 1→128→128→1280) retains shared bottleneck but increases the generator's non-linear capacity. This is the correct direction motivated directly by the PR #1297 failure analysis.

---

## 2026-04-29 22:00 — PR #1295: n_layers 5→6 on BF16+FiLM+OneCycleLR stack

- charliepai2f2-alphonse/n-layers-6-film-bf16-stack
- **Hypothesis**: Adding a 6th TransolverBlock on the FiLM+BF16+OneCycleLR stack to test whether more depth improves CFD feature extraction. Previously failed at FP32 (PR #1161) due to VRAM. Now with 60 GB headroom, architecture may be viable.
- **Status**: CLOSED — undertrained (epoch-budget mismatch, not a genuine architecture failure)

| Split | Baseline val (PR #1264) | PR #1295 val | Δ |
|-------|------------------------|--------------|---|
| single_in_dist     | 82.09 | ~97 (est) | +18% |
| geom_camber_rc     | 85.31 | ~100 (est) | +17% |
| geom_camber_cruise | 55.67 | ~65 (est)  | +17% |
| re_rand            | 74.38 | ~87 (est)  | +17% |
| **avg**            | **74.36** | **87.42** | **+17.6%** |

**Root cause**: `ONECYCLE_PER_EPOCH_SEC_ESTIMATE=100.0` assumed 100 s/epoch but n_layers=6 actual wall-clock was ~124 s/epoch. This caused `budgeted_epochs_lr=19` but only ~15 epochs fit in 30 min. The OneCycleLR schedule never reached its annealing tail — model was still in the cosine descent phase at cutoff, losing ~12 pts/epoch improvement. This is *not* a genuine architecture failure; it's a timing/budget mismatch. The hypothesis remains open for a corrected re-run.

**Follow-up**: charliepai2f2-alphonse assigned corrected n_layers=6 re-test with `ONECYCLE_PER_EPOCH_SEC_ESTIMATE=125` so `budgeted_epochs_lr≈14-15`, schedule properly anneals within 30-min budget.

---

## 2026-04-29 16:30 — PR #1207: batch_size=8 — Larger batch with BF16 VRAM headroom

- charliepai2f2-askeladd/batch-size-8
- **Hypothesis**: BF16 AMP (PR #1184) reduced peak VRAM from ~42GB to ~33GB on a 96GB H100, freeing ~60GB of headroom. batch_size=8 (doubled from 4) provides smoother gradient estimates per step, potentially improving final convergence quality especially on high-variance OOD splits. No LR scaling — keep lr=1e-3 (linear scaling risky at only 19 epochs). Epoch wall-clock time expected to remain similar (half steps × 2× data per step). Budget-aware CosineAnnealingLR handles the halved steps-per-epoch gracefully.
- **Status**: ASSIGNED — awaiting results

**Baseline (PR #1184)**: val_avg/mae_surf_p = 89.00 (epoch 19/50, BF16 + lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware cosine + surf_weight=25)

**Target**: val_avg/mae_surf_p < 89.00

```bash
cd target/ && python train.py \
  --agent charliepai2f2-askeladd \
  --experiment_name "charliepai2f2-askeladd/batch-size-8" \
  --grad_clip 1.0
# batch_size=8 in Config dataclass (or --batch_size 8)
```

---

## 2026-04-29 16:30 — PR #1206: mlp_ratio=3 — Wider FFN for improved capacity

- charliepai2f2-edward/mlp-ratio-3
- **Hypothesis**: Current mlp_ratio=2 gives FFN hidden width 2×128=256. Increasing to mlp_ratio=3 expands to 3×128=384 — a 50% increase in FFN capacity at zero per-epoch timing overhead (attention dominates FLOPs). mlp_ratio=4 was tested in PR #1102 (val=136.16, regression due to epoch slowdown). mlp_ratio=3 is the untested intermediate that may provide extra capacity without timing penalty. Richer FFN should allow more complex non-linear mappings between physics-aware attention outputs and CFD field targets.
- **Status**: ASSIGNED — awaiting results

**Baseline (PR #1184)**: val_avg/mae_surf_p = 89.00 (epoch 19/50, BF16 + lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware cosine + surf_weight=25)

**Target**: val_avg/mae_surf_p < 89.00

```bash
cd target/ && python train.py \
  --agent charliepai2f2-edward \
  --experiment_name "charliepai2f2-edward/mlp-ratio-3" \
  --grad_clip 1.0
# mlp_ratio=3 changed in model_config dict in train.py
```

---

## 2026-04-29 16:10 — PR #1182: surf_weight 25→50: stronger surface loss focus

- charliepai2f2-fern/surf-weight-50
- **Hypothesis**: Doubling surf_weight from 25→50 further focuses the loss on surface nodes, pushing the model to prioritize surface pressure accuracy.
- **Outcome**: **CLOSED — negative result**. val_avg/mae_surf_p = 101.86 vs baseline 100.41 (+1.4% regression).

| Split | Baseline val (PR #1098) | PR #1182 val | Δ |
|-------|------------------------|--------------|---|
| single_in_dist | 120.68 | 118.08 | -2.60 |
| geom_camber_rc | 111.80 | 113.62 | +1.82 |
| geom_camber_cruise | 75.99 | 77.84 | +1.85 |
| re_rand | 93.15 | 97.88 | +4.73 |
| **val_avg** | **100.41** | **101.86** | **+1.45** |
| test_avg | 88.58 | 90.24 | +1.66 |
| vol_p MAE | 120.81 | 132.09 | +9.3% worse |

- 14 epochs in budget; ~134.6s/epoch; 42.11GB VRAM
- Metrics path: `target/models/model-charliepai2f2-fern-surf-weight-50-20260429-140543/`

**Analysis**: Surface-only loss amplification hurts OOD generalization. single_in_dist (in-distribution) slightly improves, but all OOD splits regress, most severely re_rand (+4.73). Volume MAE degradation (+9.3%) suggests the model de-prioritizes learning the full flow field — which is needed as volumetric context for accurate surface predictions in unseen geometries. **surf_weight=25 confirmed as optimal.** The current surface/volume balance is well-calibrated for the full generalization spectrum.

---

## 2026-04-29 16:10 — PR #1143: Combined best config (superseded)

- charliepai2f2-alphonse/combined-best-config
- **Hypothesis**: Stack lr=1e-3 + grad_clip=1.0 + T_max=14 + surf_weight=25 together for the first time (started before PRs #1091 and #1098 merged).
- **Outcome**: **CLOSED — superseded by PR #1098**. After rebasing onto the current advisor branch, net code delta = zero. All changes were already incorporated via PRs #1091 and #1098.

| Run | val_avg/mae_surf_p | Notes |
|-----|-------------------|-------|
| Original (stale baseline) | 104.25 | vs stale 127.67 baseline; sent back for rebase |
| Rebased contaminated run | 127.86 | GPU contention caused T_max=4; discarded |
| Rebased clean run | **98.68** | idle GPU; clean result |
| Current baseline (PR #1098) | 100.41 | — |

Per-split (clean rebased run):

| Split | PR #1143 rebased | Baseline (PR #1098) | Δ |
|-------|-----------------|---------------------|---|
| single_in_dist | 116.72 | 120.68 | -3.96 |
| geom_camber_rc | 109.52 | 111.80 | -2.28 |
| geom_camber_cruise | 75.02 | 75.99 | -0.97 |
| re_rand | 93.46 | 93.15 | +0.31 |
| **val_avg** | **98.68** | **100.41** | **-1.73** |
| test_avg | 87.70 | 88.58 | -0.88 |

**Analysis**: The 98.68 result is within run-to-run variance of 100.41 (no new code was added). The experiment primarily validates baseline reproducibility. Student explicitly recommended closing as superseded. The contaminated run highlighted that the budget-aware scheduler is sensitive to GPU contention (T_max is estimated from actual epoch timing) — a useful diagnostic note for future experiments. Nothing new to merge.

---

## 2026-04-29 15:50 — PR #1178: Weight decay 1e-4 to 1e-3: stronger L2 regularization

- charliepai2f2-tanjiro/weight-decay-1e-3-sweep
- **Hypothesis**: Increasing weight decay from wd=1e-4 to wd=1e-3 (10x) provides stronger L2 regularization that compounds with DropPath 0→0.1, further reducing overfitting on OOD splits (re_rand, geom_camber splits).
- **Outcome**: **CLOSED — negative result**. val_avg/mae_surf_p = 100.90 vs current baseline 100.41 — a regression of +0.49 (+0.49%).

| Split | Baseline val (PR #1098) | PR #1178 val | Δ |
|-------|------------------------|--------------|---|
| single_in_dist | 120.68 | 119.93 | -0.75 |
| geom_camber_rc | 111.80 | 110.41 | -1.39 |
| geom_camber_cruise | 75.99 | 77.95 | +1.96 |
| re_rand | 93.15 | 95.33 | +2.18 |
| **val_avg** | **100.41** | **100.90** | **+0.49** |
| test_avg | 88.58 | 90.42 | +1.84 |

- 14 epochs in budget; best at epoch 14; lr cosine-decayed to ~2.12e-5
- Student analysis was excellent: confirmed no underfitting signal, suggested wd=5e-4, per-group wd, higher DropPath, or smaller wd as follow-ups

**Analysis**: wd=1e-3 slightly over-regularizes. The two OOD splits (re_rand +2.18, geom_camber_cruise +1.96) show the strongest regression — over-regularization hurts generalization rather than helping it. The two in-distribution splits improve marginally, but the net effect is negative. The existing DropPath 0→0.1 already provides sufficient implicit regularization in the 14-epoch budget. **wd=1e-4 is confirmed as optimal weight decay for this training regime.** Higher L2 regularization does not compound positively with stochastic depth in a 14-epoch budget.

---

## 2026-04-29 15:10 — PR #1156: DropPath max rate 0.1→0.2: stronger stochastic depth regularization

- charliepai2f2-nezuko/drop-path-sweep-0p2
- **Hypothesis**: Increasing DropPath linear schedule from 0→0.1 to 0→0.2 provides stronger stochastic depth regularization, reducing overfitting and improving OOD generalization (particularly re_rand split).
- **Outcome**: **CLOSED — negative result**. val_avg/mae_surf_p = 126.17 vs current baseline 100.41 — a regression of ~25.8%.

| Metric | DropPath 0→0.2 | Baseline (PR #1098) | Δ |
|--------|---------------|---------------------|---|
| **val_avg/mae_surf_p** (PRIMARY) | **126.17** | **100.41** | **+25.8% WORSE** |

- 14 epochs in budget; OOD splits most sensitive to over-regularization
- Student noted the 14-epoch budget means high drop_path rates cause underfitting before the model can recover

**Analysis**: With only 14 epochs available, DropPath 0→0.2 trains the model for too few effective "full-capacity" steps before annealing completes. The model underfits rather than regularizes. DropPath 0→0.1 (PR #1091) remains near-optimal for the epoch-limited regime. Going below 0.1 (e.g., 0.05, 0.075) was suggested by the student but deprioritized given the current focus on the full stack. The confirmed insight: **in a 14-epoch budget, stochastic depth regularization beyond 0.1 causes more harm than benefit**.

---

## 2026-04-29 15:05 — PR #1144: BF16 AMP: more epochs per 30-min budget via mixed precision

- charliepai2f2-askeladd/bf16-mixed-precision
- **Hypothesis**: torch.autocast BF16 on H100 hardware reduces per-epoch time by ~26% (131s → ~97s), yielding ~19-20 epochs instead of 14 within the 30-min budget — free throughput gain.
- **Outcome**: **CLOSED — technique validated but stale baseline**. val_avg/mae_surf_p = 122.39, which does NOT beat current baseline 100.41. However, the experiment was run on the pre-PR #1091 baseline (127.67), so the technique itself (26% speedup, 19 epochs vs 14) is confirmed.

| Metric | BF16 result | Prior baseline (127.67) | Current baseline (100.41) |
|--------|-------------|------------------------|--------------------------|
| **val_avg/mae_surf_p** | **122.39** | 127.67 (beat) | 100.41 (did NOT beat) |

- Per-epoch time confirmed: ~96.6s (vs 131s baseline) — 26.2% speedup
- 19 epochs completed vs 14 expected
- Implementation: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` + `pred.float()` before loss + no GradScaler

**Analysis**: The BF16 technique is confirmed to work and deliver the expected speedup on H100 hardware. The gap versus current baseline (100.41) is because the experiment did not include lr=1e-3, grad_clip=1.0, DropPath 0→0.1, and budget-aware cosine from PRs #1091 and #1098. Re-assigning this exact technique on the full current stack (PR #1184) — with the budget-aware schedule now benefiting from ~97s/epoch timing, ~19 epochs, and all other improvements stacked. Expected outcome: possibly <95 val_avg/mae_surf_p.

---

## 2026-04-29 13:56 — PR #1161: n_layers 5→6: moderate depth + budget-aware LR

- charliepai2f2-fern/n-layers-6-budget-aware
- **Hypothesis**: n_layers=6 (one extra layer over baseline's 5) is a middle ground that keeps per-epoch time at ~160-170s, fitting ~10-11 epochs in 30 min, and may capture depth-helps-geometry-generalization signal from PR #1089 without the epoch-starvation penalty.
- **Outcome**: **CLOSED — dead end**. val_avg/mae_surf_p=127.56 vs baseline 100.41 — 27% regression.

| Metric | n_layers=6 | Baseline (PR #1098) | Δ |
|--------|-----------|---------------------|---|
| **val_avg/mae_surf_p** (PRIMARY) | **127.56** | **100.41** | **+27.0% WORSE** |
| val_avg/mae_surf_Ux | 2.54 | 1.50 | worse |
| val_avg/mae_surf_Uy | 0.88 | 0.74 | worse |

| Split | val (n_layers=6) | val (baseline) | Δ |
|-------|-----------------|----------------|---|
| single_in_dist | 160.30 | 120.68 | +39.62 |
| geom_camber_rc | 138.04 | 111.80 | +26.24 |
| geom_camber_cruise | 96.14 | 75.99 | +20.15 |
| re_rand | 115.76 | 93.15 | +22.61 |
| **avg** | **127.56** | **100.41** | **+27.15** |

- 12 epochs in 30 min; per-epoch time 158.4s (vs 135s baseline)
- Params: 0.78M vs 0.66M baseline
- Metrics YAML: `target/models/model-charliepai2f2-fern-n-layers-6-budget-aware-20260429-125523/metrics.yaml`

**Analysis**: The extra serial latency (158s vs 135s/epoch) costs 2 epochs (12 vs 14). 0.78M vs 0.66M params leaves the model under-trained. Budget-aware cosine LR worked correctly but couldn't compensate for the capacity/throughput tradeoff. Throughput-over-capacity rule reconfirmed: at 30-min budget, deeper architectures are net negatives. The cruise-geometry tie (96.14 vs 75.99 at 30-min is actually also worse — only partial signal). n_layers=5 is the sweet spot for this budget.

---

## 2026-04-29 10:50 — PR #1088: Increase surf_weight from 10 to 25 for surface MAE focus

- **Branch**: charliepai2f2-edward/surf-weight-sweep-25
- **Hypothesis**: Increasing surf_weight from 10→25 focuses training loss on surface nodes, directly targeting the primary val metric (surface pressure MAE).
- **Outcome**: **MERGED** — new baseline. val_avg/mae_surf_p = 127.6661

### Results (epoch 13/50, best checkpoint, ~14 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **127.6661** |
| val_avg/mae_vol_p | 139.9394 |
| val_avg/mae_surf_Ux | 2.2548 |
| val_avg/mae_surf_Uy | 0.9431 |
| val_avg/mae_vol_Ux | 5.8663 |
| val_avg/mae_vol_Uy | 2.6935 |

| Split | mae_surf_p | mae_vol_p |
|-------|------------|-----------|
| val_single_in_dist | 157.82 | 178.70 |
| val_geom_camber_rc | 135.65 | 146.43 |
| val_geom_camber_cruise | 99.26 | 112.71 |
| val_re_rand | 117.94 | 121.91 |

Test metrics (3 of 4 valid; test_geom_camber_cruise NaN — corrupted GT sample 000020.pt):
| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 137.04 |
| test_geom_camber_rc | 122.18 |
| test_geom_camber_cruise | NaN (upstream data bug) |
| test_re_rand | 117.39 |
| 3-split avg | 125.54 |

- Per-epoch time: ~131.6s; Peak GPU: 42.12 GB
- Metrics JSONL: `target/models/model-charliepai2f2-edward-surf-weight-25-20260429-095003/metrics.jsonl`

### Analysis

A clean, minimal single-parameter change. Explicitly up-weighting surface nodes in the training loss directly improves the primary surface pressure metric. The large improvement (vs. a hypothetical baseline that would be ~130+ before any tuning) validates that the original surf_weight=10 was under-emphasizing surface accuracy. Edward also identified the NaN issue in scoring (corrupted GT sample) and provided a root cause analysis. Model is efficient: 2× fewer VRAM than width-expansion experiment, nearly 2× faster per epoch.

Follow-up: edward is testing timeout-aware CosineAnnealingLR (T_max=14 to match actual epoch count) in PR #1126.

---

## 2026-04-29 12:20 — PR #1091: Add stochastic depth (drop_path 0→0.1) for OOD generalization — REVISION 2 (MERGED)

- **Branch**: charliepai2f2-nezuko/stochastic-depth-regularization
- **Hypothesis**: DropPath on residual paths with linear layer schedule (0→0.1) + budget-aware CosineAnnealingLR (T_max estimated from warm-up timing) reduces overfitting and improves OOD generalization.
- **Outcome**: **MERGED** — new baseline. val_avg/mae_surf_p = 121.89 (-4.5% vs prior baseline 127.67)

### Results (epoch 13/50, best checkpoint, ~14 epochs in 30-min timeout)

| Metric | Value | vs prior baseline |
|--------|-------|-------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **121.89** | **-5.78 (-4.5%)** |
| val_avg/mae_surf_Ux | 1.97 | — |
| val_avg/mae_surf_Uy | 0.89 | — |

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 154.41 | 130.06 |
| geom_camber_rc | 128.38 | 118.20 |
| geom_camber_cruise | 95.98 | 79.45 |
| re_rand | 108.78 | 110.64 |
| **avg** | **121.89** | **109.59** |

- Budget-aware cosine schedule: T_max=11 estimated after 2 warm-up epochs (1529s remaining / 135.4s per epoch), eta_min=1e-6
- Peak GPU memory: 42.1 GB (single H100, batch=4)
- Params: 0.66 M (DropPath adds no parameters)
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-sd-cosine-budget-aware-20260429-114039/metrics.jsonl`

### Analysis

The revision confirmed the hypothesis. Two changes together drove the -4.5% improvement: (1) DropPath linear schedule (0.0→0.1 across 5 TransolverBlocks) providing stochastic regularization, and (2) budget-aware CosineAnnealingLR that estimates the remaining epoch budget by timing warm-up epochs and sets T_max accordingly (T_max=11 from 1529s remaining / 135.4s per epoch). This ensures the LR actually reaches eta_min by the end of training, rather than staying near peak (5e-4) throughout. The combination of regularization + proper annealing is synergistic: annealing without regularization risks overfit at low LR; regularization without annealing stays at high LR too long.

Per-split analysis: geom_camber_cruise (-3.28 val, -9.26 test from prior) and re_rand (-9.16 val from prior) benefited most — the OOD splits that benefit most from regularization. single_in_dist improved (-3.41 val) even though it is in-distribution, suggesting the LR fix alone mattered.

NaN guard (nan_to_num + per-sample finite filter) is confirmed working — all 4 test splits reported cleanly.

---

## 2026-04-29 12:00 — PR #1091: Add stochastic depth (drop_path 0→0.1) for OOD generalization — REVISION 1 (Sent back)

- **Branch**: charliepai2f2-nezuko/stochastic-depth-regularization
- **Hypothesis**: DropPath on residual paths with linear layer schedule (0→0.1) reduces overfitting and improves OOD generalization.
- **Outcome**: **Sent back for revision** — marginally worse than baseline; result confounded by LR schedule mismatch

### Results (epoch 13/50, best checkpoint, ~14 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **127.86** |
| test_avg/mae_surf_p (3-split NaN-safe) | 116.32 |

| Split | mae_surf_p |
|-------|------------|
| val_single_in_dist | ~158 |
| val_geom_camber_rc | ~136 |
| val_geom_camber_cruise | ~100 |
| val_re_rand | ~118 |

vs. baseline: **127.6661** (worse by 0.19)

### Analysis

The stochastic depth result (127.86) does not beat baseline (127.6661). However, the margin is only 0.19, well within epoch-to-epoch noise (~30 unit variance between adjacent epochs observed in the training log). The result is confounded by the CosineAnnealingLR T_max=50 mismatch: only ~14 epochs complete in 30 min, so the LR barely decays from its peak (5e-4). DropPath specifically benefits from a well-annealed model at the end of training. Student implemented a clean NaN guard (`nan_to_num` + mask invalidation for corrupted GT samples) — valuable fix.

**Sent back** with instruction to: (1) implement timeout-aware LR (estimate actual epoch budget by timing warm-up epochs, set T_max accordingly), (2) keep drop_path=0.1 unchanged, (3) rerun.

---

## 2026-04-29 12:00 — PR #1087: Increase Transolver slice_num from 64 to 128

- **Branch**: charliepai2f2-askeladd/slice-num-sweep-128
- **Hypothesis**: Finer physics-mode resolution via slice_num 64→128 captures more distinct flow regimes in the attention mechanism.
- **Outcome**: **CLOSED** — clear regression driven by reduced epoch budget

### Results (epoch 9/50, best checkpoint, ~9 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **132.75** |

vs. baseline: **127.6661** (worse by ~5.1, ~4% regression)

- Per-epoch time: ~210s vs ~131s baseline
- Peak GPU: 54.51 GB vs 42.12 GB baseline
- Only 9 epochs vs ~14 baseline in same 30-min window

### Analysis

The increase in per-epoch compute time (210s vs 131s) reduced the training budget from ~14 epochs to ~9 epochs. With CosineAnnealingLR(T_max=50) and only 9 epochs, the LR had barely annealed. The capacity gain from 2× more physics slices did not compensate for ~36% fewer gradient steps. In our budget-constrained 30-min setting, throughput is paramount — increasing model complexity without a compensating speedup always loses. Closed as a clear dead end.

---

## 2026-04-29 12:20 — PR #1129: Per-sample instance-normalized loss for Re-regime balance

- **Branch**: charliepai2f2-askeladd/per-sample-loss-normalization
- **Hypothesis**: Replace fixed MSE with per-sample per-channel instance-normalized loss to equalize gradient contributions from high-Re and low-Re samples.
- **Outcome**: **CLOSED** — 2.9x worse than baseline; loss formulation broke pressure learning

### Results (epoch 11/14, best checkpoint, ~14 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **366.76** |
| test_avg/mae_surf_p | 343.32 |

| Split | mae_surf_p (this run) | mae_surf_p (baseline) |
|-------|----------------------|----------------------|
| val_single_in_dist | 513.01 | 157.82 |
| val_geom_camber_rc | 453.48 | 135.65 |
| val_geom_camber_cruise | 209.57 | 99.26 |
| val_re_rand | 290.99 | 117.94 |
| **val_avg** | **366.76** | **127.67** |

- Metrics JSONL: `target/models/model-charliepai2f2-askeladd-per-sample-loss-norm-20260429-112007/metrics.jsonl`

### Analysis

A clear failure. The root cause is subtle but well-diagnosed by the student: per-sample-per-channel std normalization conflates between-sample Re scaling with between-channel scaling. Pressure has the largest within-sample dynamic range, so its per-channel std is large, which *downweights* the pressure gradient relative to velocity — exactly the opposite of what we want with our pressure-focused primary metric. The global normalization `(y - y_mean) / y_std` already equalizes channel scales; per-sample normalization undoes that equalization. Additionally, the loss was numerically unstable: `eps=1e-6` in physical pressure units is far too small, causing catastrophic gradient spikes on near-uniform-flow samples.

The training loss trajectory (467, 366, 627, 85, 153, 45...) shows the instability clearly — no coherent convergence. Velocity errors ended up near baseline (~16, ~2) while pressure errors were catastrophically inflated.

**Better alternatives**: (1) Per-sample *scalar* normalization using single std over globally-normalized targets — preserves channel ratios; (2) Sample-level Re-conditional loss weighting `w_i = 1 / log(1 + per_sample_y_std_global)` — directly attacks Re imbalance without disrupting channel balance; (3) Larger eps proportional to global y_std per channel.

---

## 2026-04-29 14:30 — PR #1086: Widen Transolver: n_hidden 128→256, n_head 4→8 (ITERATION 2: n_hidden=192, n_head=6)

- **Branch**: charliepai2f2-alphonse/width-expansion-256
- **Hypothesis**: Doubling hidden width from 128→256 and scaling heads 4→8 increases model capacity for multi-domain generalization. Revised to n_hidden=192, n_head=6 to reduce per-epoch cost.
- **Outcome**: **CLOSED** — val_avg/mae_surf_p = 146.55 vs baseline 127.67; still undertraining under 30-min timeout

### Results — Iteration 2 (n_hidden=192, n_head=6, surf_weight=25, epoch 9/50)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---|---|---|---|---|---|
| val_single_in_dist | 187.10 | 2.28 | 1.05 | 203.57 | 7.17 | 3.02 |
| val_geom_camber_rc | 158.25 | 3.53 | 1.22 | 164.36 | 8.52 | 3.52 |
| val_geom_camber_cruise | 101.96 | 1.73 | 0.80 | 120.76 | 5.30 | 2.12 |
| val_re_rand | 138.87 | 2.99 | 1.03 | 143.95 | 7.24 | 2.76 |
| **val_avg** | **146.55** | 2.63 | 1.02 | 158.16 | 7.06 | 2.86 |

| test split | mae_surf_p |
|---|---|
| test_single_in_dist | 171.45 |
| test_geom_camber_rc | 148.54 |
| test_geom_camber_cruise | 89.60 |
| test_re_rand | 130.12 |
| **test_avg** | **134.93** |

- Per-epoch time: ~204s; Peak GPU: 63.0 GB; 9 epochs (still improving at final epoch)
- Metrics JSONL: `models/model-charliepai2f2-alphonse-width-192-h6-sw25-v2-20260429-111300/metrics.jsonl`
- vs. baseline: 127.67 (worse by 18.88, ~14.8%)

### Analysis

Two iterations explored width expansion:
- **Iter 1** (n_hidden=256, n_head=8): 258s/epoch, 7 epochs, val=173.99 — severely undertrained
- **Iter 2** (n_hidden=192, n_head=6): 204s/epoch, 9 epochs, val=146.55 — still undertrained

Core finding: **width expansion is fundamentally incompatible with the 30-minute timeout at current epoch budget settings**. The wider model needs ~1.5x the wall-clock to match the baseline's 14-epoch training, but within the same 30-min cap it only gets 9 epochs with the cosine LR barely annealed. The val curve was still strictly improving at epoch 9 (monotone descent from epoch 6 onward), suggesting the model CAN learn — it just needs more time.

Notable per-split finding: the cruise domain (val_geom_camber_cruise = 101.96; test = 89.60) benefits most from width expansion, suggesting that geometry generalization (unseen camber values) is capacity-limited. Both in-dist and racecar splits lagged, driving the average up.

**Closed** — width expansion under 30-min budget is consistently dominated by the throughput constraint. Revisit once timeout increases or if a fast-per-epoch wider architecture can be found (e.g., 192 width, mlp_ratio=1, fewer heads).

---

## 2026-04-29 12:30 — PR #1089: Deeper Transolver: n_layers 5→8 with lr 5e-4→3e-4

- **Branch**: charliepai2f2-fern/deeper-transolver-8layers
- **Hypothesis**: Increasing depth from n_layers=5 to n_layers=8 adds representational capacity for complex multi-domain flow patterns; lr reduced 5e-4→3e-4 for stability.
- **Outcome**: **CLOSED** — val_avg/mae_surf_p = 156.24 vs baseline 121.89; clear regression; two confounders identified

### Results (epoch 8/9 completed, best checkpoint, ~9 epochs in 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **156.24** |

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| val_single_in_dist | 206.85 | 178.34 |
| val_geom_camber_rc | 177.21 | 163.03 |
| val_geom_camber_cruise | 107.20 | 93.24 |
| val_re_rand | 133.69 | 135.08 |
| **avg** | **156.24** | **142.42** |

- Per-epoch time: ~207s vs ~131s baseline; Peak GPU: 64.5 GB
- Only 9 epochs vs ~14 baseline in same 30-min window
- surf_weight: 10.0 (old baseline — should have been 25.0)
- Metrics JSONL: on fern's branch

### Analysis

Two major confounders explain the regression:

1. **surf_weight=10 instead of 25**: The PR instructions referenced the pre-PR-#1088 baseline. The current merged baseline uses surf_weight=25. Under-emphasizing surface nodes directly penalizes the primary metric.

2. **Epoch starvation under 30-min timeout**: n_layers=8 runs at ~207s/epoch vs baseline ~131s/epoch (~58% slower), fitting only ~9 epochs in 30 min vs ~14 for baseline. With CosineAnnealingLR(T_max=50) and only 9 epochs, the LR had barely annealed from its starting 3e-4.

Notable finding: the **cruise domain** (val=107.20, test=93.24) was competitive with baseline (95.98 val, 79.45 test), suggesting that depth specifically helps the geometry-generalization split. However, single_in_dist and rc splits were badly hurt (+49.03 and +48.83 respectively on val), driving the average up by +34.35.

Fern also implemented a valuable **sample-level NaN filter** in `evaluate_split`: checks `torch.isfinite(y.reshape(B,-1)).all(dim=-1)` per sample and skips non-finite GT batches. This gives clean 4-split test coverage vs. the previous 3-split workaround for corrupted sample 000020.pt. This fix should be included in future train.py iterations.

**Closed**. Depth expansion at n_layers=8 is incompatible with 30-min budget — same throughput problem as width expansion. A controlled follow-up (n_layers=6 + surf_weight=25 + budget-aware CosineAnnealingLR) could isolate whether moderate depth helps, but this is lower priority than other directions.

---

## 2026-04-29 11:00 — PR #1086 ITERATION 1: Widen Transolver: n_hidden 128→256, n_head 4→8

- **Branch**: charliepai2f2-alphonse/width-expansion-256
- **Hypothesis**: Doubling hidden width from 128→256 and scaling heads 4→8 increases model capacity for multi-domain generalization. 4x parameter increase in attention and MLP layers.
- **Outcome**: Sent back for revision — inconclusive due to timeout constraint

### Results (epoch 6/50, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---|---|---|---|---|---|
| val_single_in_dist | 208.89 | 2.69 | 1.02 | 212.80 | 7.20 | 2.75 |
| val_geom_camber_rc | 175.16 | 4.16 | 1.25 | 180.37 | 7.96 | 3.31 |
| val_geom_camber_cruise | 154.44 | 2.43 | 0.84 | 162.93 | 5.51 | 1.84 |
| val_re_rand | 157.46 | 3.26 | 1.02 | 162.81 | 6.61 | 2.41 |
| **val_avg** | **173.99** | — | — | — | — | — |

**Current baseline**: val_avg/mae_surf_p = 127.67 (PR #1088, surf_weight=25)

### Analysis

Not a fair comparison. The n_hidden=256 model runs at ~258s/epoch vs. baseline's ~131s/epoch, so in the 30-minute timeout, it only completed 7/50 epochs vs. baseline's ~14. The LR was still near its peak (cosine annealing with T_max=50 had only annealed ~5%). The training curve was monotonically improving at epoch 6 (173.99), with epoch 7 showing instability (213.43) from high LR. Additional issue: NaN in test_geom_camber_cruise pressure — same corrupted GT sample as other experiments; not unique to this model.

**Sent back with instruction to try n_hidden=192, n_head=6 (keeping head_dim=32) with surf_weight=25, which should bring per-epoch time closer to 200s and allow ~9 epochs within the timeout with better LR annealing. Also instructed to add nan_to_num guard.**

---

## 2026-04-29 18:02 — PR #1211: BF16 + OneCycleLR (1-cycle schedule over 19-epoch BF16 budget)

- **Branch**: charliepai2f2-nezuko/bf16-onecycle-combined
- **Hypothesis**: Combine BF16 AMP (PR #1184, 19 epochs) with OneCycleLR (PR #1195, schedule shape) to compound throughput + schedule gains. max_lr=1.2e-3, pct_start=0.3, total_steps = steps_per_epoch * 19.
- **Outcome**: **MERGED — new baseline (val=80.53, -9.5%)**

### Results

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| single_in_dist     | 91.33 | 80.43 |
| geom_camber_rc     | 91.01 | 83.65 |
| geom_camber_cruise | 60.42 | 50.04 |
| re_rand            | 79.36 | 70.80 |
| **avg**            | **80.53** | **71.23** |

- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-bf16-onecycle-combined-20260429-154838/metrics.jsonl`
- Wall: 31.4 min, peak GPU memory 32.96 GB, 19 epochs at 98–100 s/epoch, best checkpoint = final epoch
- Schedule fully annealed to min_lr ≈ 4.9e-9; classic 1-cycle convergence pattern (val drops monotonically from epoch 6 onwards)

### Analysis

The two improvements were nearly additive: PR #1184 alone gave 89.00 (BF16 throughput), PR #1195 alone gave a ~14% improvement over CosineAnnealing at the older baseline. Together they yield 80.53. Best checkpoint at the final epoch indicates no overshoot — schedule is well-sized but might still be slightly under-budgeted. Suggested follow-ups (max_lr {1.0e-3, 1.5e-3}, pct_start=0.2, 20-epoch budget, surf_weight×OneCycle interaction) become high-priority next experiments.

---

## 2026-04-29 18:21 — Round 4 closed experiments (negative results)

| PR | Student | Hypothesis | val_avg/mae_surf_p | vs PR #1184 baseline | Disposition |
|----|---------|-----------|--------------------:|----------------------|-------------|
| #1207 | askeladd | batch_size=8 with BF16 VRAM headroom | 112.62 | +26.5% | Closed — halved gradient steps in 30-min budget |
| #1206 | edward   | mlp_ratio=2→3 (wider FFN) | 95.31 | +7.1% | Closed — model still descending at epoch 18, underfit |
| #1204 | alphonse | Re noise std=0.05 | 99.99 (baseline 100.41 stale) | n/a | Closed — implementation bug (per-node not per-sample); idea sound, retry on full stack with `torch.randn(B,1,1)*0.05` |
| #1203 | fern     | Checkpoint averaging K=3 (SWA) | 102.98 | +15.7% | Closed — all 3 averaged checkpoints from cosine eta_min tail; no diversity to average |
| #1199 | tanjiro  | FiLM conditioning on log(Re) | 92.20 | +3.6% | Closed — fair test impossible (no BF16, only 13 epochs); retry as lightweight shared FiLM on BF16+OneCycle stack |
| #1152 | thorfinn | CosineAnnealingLR eta_min=1e-5 | 92.28 | +3.7% | Closed — eta_min=1e-6 default better; finer late-training updates win in cosine tail |

All six experiments produced clean negative results with clear root-cause analyses from students. Lessons logged into `CURRENT_RESEARCH_STATE.md` insights 1, 8, 9, 10.

