# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2f-r5)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **52.0698** (PR #1136 — weight_decay=5e-4 + huber_delta=0.1 + n_layers=2, epoch 30/30) |
| `test_avg/mae_surf_p` | **46.1497** (PR #1136) |

**Source:** PR #1136 — Stronger L2: weight_decay=5e-4 to improve OOD generalization — all 30 epochs completed (LR=0.0 at epoch 30), 30.07 min wall-clock.
- Branch: `charliepai2f5-askeladd/weight-decay-5e-4`
- Config: n_layers=2 (hardcoded), slice_num=16, n_hidden=256, n_head=8, loss=huber, huber_delta=0.1, ema_decay=0.999, grad_clip=1.0, per_sample_norm, epochs=30, lr=5e-4, batch_size=4, weight_decay=5e-4
- Run completed all 30/30 epochs; LR=0.0 at final epoch (full cosine completion). Best epoch = 30 (model still improving at termination — training-budget-limited)
- 1,141,299 params, Peak VRAM 22.22 GB, Run ID: l6zjon8u

**Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper reference) — currently +12.7% above target (gap closed from 8.69 to 5.22 vs prior best).

## Round r5 — Recommended Working Baseline (compound n_layers=2 + huber_delta=0.1 + weight_decay=5e-4 + epochs=30)

```
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4
```
*(Note: n_layers=2, slice_num=16 are hardcoded in model_config dict in train.py)*

## Round r5 — Merged Winners

### PR #1136 — Stronger L2: weight_decay=5e-4 to improve OOD generalization (2026-04-29)
**Student:** charliepai2f5-askeladd | **Branch:** charliepai2f5-askeladd/weight-decay-5e-4

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **52.0698** (epoch 30/30 — full cosine completion, LR=0.0) |
| `val_single_in_dist/mae_surf_p` | 56.5175 |
| `val_geom_camber_rc/mae_surf_p` | 66.5405 |
| `val_geom_camber_cruise/mae_surf_p` | 34.0280 |
| `val_re_rand/mae_surf_p` | 51.1932 |
| `test_avg/mae_surf_p` | **46.1497** |
| `test_single_in_dist/mae_surf_p` | 51.7768 |
| `test_geom_camber_rc/mae_surf_p` | 61.4231 |
| `test_geom_camber_cruise/mae_surf_p` | 28.3556 |
| `test_re_rand/mae_surf_p` | 43.0433 |

**vs prior baseline (PR #1134):** 52.0698 vs 55.4877 → **-6.16% val improvement**
**Test improvement:** 46.1497 vs 48.8156 → **-5.46% test improvement**
**vs PR #1120 (instructions baseline):** val -7.72%, test -7.00%
**Run config:** n_layers=2 (hardcoded), huber_delta=0.1, weight_decay=5e-4, epochs=30, lr=5e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
**Best epoch:** 30/30 — model still improving at termination (is_best=True every epoch from 24 onward) — training-budget-limited result
**OOD analysis:** All OOD splits beat in-dist on relative gain (geom_camber_cruise: -13.57%, re_rand: -10.52%, vs in-dist: -3.16%). geom_camber_rc gained least (-4.19%) — bottlenecked by representation, not pure overfitting. Compete gap: 5.22 (from 8.69 prior round)
**Peak VRAM:** 22.22 GB | **Wall-clock:** 30.07 min | **Run ID:** l6zjon8u
**Metrics JSONL:** `metrics/charliepai2f5-askeladd-weight-decay-5e-4-l6zjon8u.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4`

### PR #1134 — Cosine-aligned epochs=26 on n_layers=2 + huber_delta=0.1 stack (2026-04-29)
**Student:** charliepai2f5-edward | **Branch:** charliepai2f5-edward/epochs-26-cosine-aligned

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **55.4877** (epoch 26/26 — full cosine completion, LR=0.0) |
| `val_single_in_dist/mae_surf_p` | 60.5296 |
| `val_geom_camber_rc/mae_surf_p` | 70.5915 |
| `val_geom_camber_cruise/mae_surf_p` | 35.7350 |
| `val_re_rand/mae_surf_p` | 55.0944 |
| `test_avg/mae_surf_p` | **48.8156** |
| `test_single_in_dist/mae_surf_p` | 54.9459 |
| `test_geom_camber_rc/mae_surf_p` | 64.3997 |
| `test_geom_camber_cruise/mae_surf_p` | 29.8793 |
| `test_re_rand/mae_surf_p` | 46.0375 |

**vs prior baseline (PR #1120):** 55.4877 vs 56.4257 → **-1.66% val improvement**
**Test improvement:** 48.8156 vs 49.6211 → **-1.62% test improvement**
**Run config:** n_layers=2 (hardcoded), huber_delta=0.1, epochs=26, cosine T_max=26 (fully aligned to budget)
**Schedule alignment:** LR at epoch 26 = 0.0 (full cosine completion). Monotonically improving every epoch.
**Peak VRAM:** 22.22 GB | **Wall-clock:** 26.11 min (well within 30-min budget)
**Win concentrated in:** geom_camber_cruise (-7.45% val / -8.92% test) and re_rand (-3.34% val / -4.29% test). Minor regression on single_in_dist (+1.43%) and geom_camber_rc (+0.25%).
**Metrics JSONL:** `metrics/charliepai2f5-edward-epochs-26-cosine-aligned-9a9ve4zq.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 26 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`

### PR #1120 — Shallower model: n_layers=2 (2026-04-29)
**Student:** charliepai2f5-nezuko | **Branch:** charlie5-nezuko/n-layers-2

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **56.4257** (epoch 26/30 — terminated by 30-min timeout, still falling) |
| `val_single_in_dist/mae_surf_p` | 59.6760 |
| `val_geom_camber_rc/mae_surf_p` | 70.4189 |
| `val_geom_camber_cruise/mae_surf_p` | 38.6126 |
| `val_re_rand/mae_surf_p` | 56.9952 |
| `test_avg/mae_surf_p` | **49.6211** |
| `test_single_in_dist/mae_surf_p` | 53.4660 |
| `test_geom_camber_rc/mae_surf_p` | 64.1098 |
| `test_geom_camber_cruise/mae_surf_p` | 32.8067 |
| `test_re_rand/mae_surf_p` | 48.1021 |
| `test_avg/mae_surf_Ux` | 0.7912 |
| `test_avg/mae_surf_Uy` | 0.3831 |
| `test_avg/mae_vol_p` | 55.7155 |

**vs prior baseline (PR #1121):** 56.4257 vs 58.4790 → **-3.51% val improvement**
**Test improvement:** 49.6211 vs 51.3554 → **-3.38% test improvement**
**Run config:** n_layers=2, but huber_delta=1.0 (NOT 0.1 — student branched from #1050 era).
**Model parameters:** 1,141,299 (-29% vs #1121) | **Peak VRAM:** 22.22 GB (-27%) | **Train time:** 30.81 min (timeout)
**Note:** Throughput win — 26 epochs in 30 min vs 22 for n_layers=3. Val monotone-decreasing every epoch, still falling at termination. Consider `--huber_delta 0.1` to compound with PR #1121.
**Metrics JSONL:** `metrics/charliepai2f5-nezuko-n-layers-2-93bfb7ek.jsonl`
**Reproduce (run-as-merged):** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`
**Reproduce (recommended compound):** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`

### PR #1121 — Tighter Huber loss: huber_delta=0.1 (2026-04-29)
**Student:** charliepai2f5-fern | **Branch:** charlie5-fern/huber-delta-0.1

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **58.4790** (epoch 22/30 — terminated by 30-min timeout) |
| `val_single_in_dist/mae_surf_p` | 66.2861 |
| `val_geom_camber_rc/mae_surf_p` | 71.2084 |
| `val_geom_camber_cruise/mae_surf_p` | 39.3226 |
| `val_re_rand/mae_surf_p` | 57.0991 |
| `val_avg/mae_surf_Ux` | (per-channel improved -14.81% on test) |
| `test_avg/mae_surf_p` | **51.3554** |
| `test_single_in_dist/mae_surf_p` | 59.5717 |
| `test_geom_camber_rc/mae_surf_p` | 64.9563 |
| `test_geom_camber_cruise/mae_surf_p` | 32.3451 |
| `test_re_rand/mae_surf_p` | 48.5484 |
| `test_avg/mae_surf_Ux` | 0.7276 |
| `test_avg/mae_surf_Uy` | 0.3657 |
| `test_avg/mae_vol_p` | 56.5247 |

**vs prior baseline (PR #1050):** 58.4790 vs 61.5855 → **-5.04% val improvement**
**Test improvement:** 51.3554 vs 54.3573 → **-5.52% test improvement**
**Model parameters:** 1,606,219 | **Peak VRAM:** 30.45 GB | **Train time:** 30.91 min (timeout)
**Note:** Cruise (-14.4%) and re_rand (-9.4%) gained most — exactly the splits with extreme Re where tighter Huber clamp was predicted to help. geom_camber_rc test +1.05% (small regression on uniform high-Re split).
**Metrics JSONL:** `metrics/charliepai2f5-fern-huber-delta-0.1-jzaml14l.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`

## Prior Round Winners (History)

### PR #1050 — PSN + epochs=30 on compound stack (2026-04-29)
**Student:** charliepai2e1-edward | **Branch:** charliepai2e1-edward/psn-plus-epochs-30

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **61.5855** (epoch 22/30 — terminated by 30-min timeout) |
| `val_single_in_dist/mae_surf_p` | 68.3069 |
| `val_geom_camber_rc/mae_surf_p` | 72.6498 |
| `val_geom_camber_cruise/mae_surf_p` | 44.8940 |
| `val_re_rand/mae_surf_p` | 60.4914 |
| `val_avg/mae_surf_Ux` | 0.9179 |
| `val_avg/mae_surf_Uy` | 0.4509 |
| `val_avg/mae_vol_p` | 67.5720 |
| `test_avg/mae_surf_p` | **54.3573** |
| `test_single_in_dist/mae_surf_p` | 61.7523 |
| `test_geom_camber_rc/mae_surf_p` | 64.2811 |
| `test_geom_camber_cruise/mae_surf_p` | 37.8047 |
| `test_re_rand/mae_surf_p` | 53.5912 |
| `test_avg/mae_surf_Ux` | 0.8541 |
| `test_avg/mae_surf_Uy` | 0.4187 |
| `test_avg/mae_vol_p` | 60.2983 |

**vs prior baseline (PR #1015):** 61.5855 vs 66.8085 → **-7.8% val improvement**
**Test improvement:** 54.3573 vs 58.7266 → **-7.4% test improvement**
**Model parameters:** 1,606,219 | **Peak VRAM:** ~30.44 GB | **Train time:** ~30 min (hit timeout)
**Note:** Val still falling ~2.8%/epoch at epoch 22 when 30-min timeout hit (LR=8.27e-5). More epochs likely to yield further gains.
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`

### PR #1015 — Longer training: epochs=24 on compound stack (2026-04-28)
**Student:** charliepai2e1-edward | **Branch:** charliepai2e1-edward/longer-training-epochs-24

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **66.8085** (epoch 22/24 — terminated by 30-min timeout) |
| `val_single_in_dist/mae_surf_p` | 73.9641 |
| `val_geom_camber_rc/mae_surf_p` | 79.1014 |
| `val_geom_camber_cruise/mae_surf_p` | 48.9877 |
| `val_re_rand/mae_surf_p` | 65.1809 |
| `val_avg/mae_surf_Ux` | 0.9857 |
| `val_avg/mae_surf_Uy` | 0.4870 |
| `val_avg/mae_vol_p` | 73.0050 |
| `test_avg/mae_surf_p` | **58.7266** |
| `test_single_in_dist/mae_surf_p` | 67.5104 |
| `test_geom_camber_rc/mae_surf_p` | 70.2042 |
| `test_geom_camber_cruise/mae_surf_p` | 40.5897 |
| `test_re_rand/mae_surf_p` | 56.6022 |
| `test_avg/mae_surf_Ux` | 0.9206 |
| `test_avg/mae_surf_Uy` | 0.4514 |
| `test_avg/mae_vol_p` | 65.1125 |

**vs prior baseline (PR #795):** 66.8085 vs 90.4014 → **-26.1% val improvement**
**Test improvement:** 58.7266 vs 80.3748 → **-27.0% test improvement**
**Model parameters:** 1,606,219 | **Peak VRAM:** 30.45 GB | **Train time:** 30.42 min (hit 30-min timeout)
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 24 --grad_clip 1.0 --ema_decay 0.999`

### PR #795 — Per-sample loss normalization on compound stack (2026-04-28)
**Student:** charliepai2e1-thorfinn | **Branch:** charliepai2e1-thorfinn/per-sample-loss-norm

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **90.4014** (epoch 12/12) |
| `val_single_in_dist/mae_surf_p` | 108.5561 |
| `val_geom_camber_rc/mae_surf_p` | 101.4393 |
| `val_geom_camber_cruise/mae_surf_p` | 66.9027 |
| `val_re_rand/mae_surf_p` | 84.7074 |
| `test_avg/mae_surf_p` | **80.3748** |

**vs prior baseline (PR #1005):** 90.4014 vs 94.6541 → **-4.50% improvement**
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm`

### PR #1005 — n_layers=3, slice_num=16 reference architecture (2026-04-29)
**Student:** charliepai2e1-edward

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **94.6541** (epoch 12/12) |
| `test_avg/mae_surf_p` | **83.7608** |

**vs prior baseline (PR #882):** 94.6541 vs 103.2182 → **-8.31% improvement**

### PR #882 — EMA model weights (decay=0.999) on compound baseline (2026-04-29)
**Student:** charliepai2e1-nezuko

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **103.2182** (epoch 10/12) |
| `test_avg/mae_surf_p` | **92.4867** |

**vs prior baseline (PR #808):** 103.22 vs 104.11 → **-0.86% improvement**

### PR #808 — bf16 mixed precision + wider model (n_hidden=256, n_head=8) + Huber + epochs=12 (2026-04-28)
**Student:** charliepai2e1-fern

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **104.1120** (epoch 10/12) |
| `test_avg/mae_surf_p` | **94.7010** |

**vs prior baseline (PR #827):** 104.11 vs 109.57 → **-4.97% improvement**

### PR #827 — Huber loss + surf_weight=30 (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.5716** (epoch 13/14) |

**vs Huber baseline (PR #788):** 109.57 vs 115.65 → **-5.26% improvement**

### PR #788 — Huber loss instead of MSE (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (epoch 10/14) |

**vs MSE baseline:** 115.65 vs 126.88 → **-8.85% improvement**

## Key Infrastructure Fixes

### NaN guard (PR #792)
- `--grad_clip 1.0` + upstream pred/GT sanitization in `evaluate_split` resolves NaN propagation.
- Root cause: IEEE 754 `Inf * False (==0.0) = NaN` — `(pred - y).abs()` computed before masking.
- `test_geom_camber_cruise/000020.pt` has 761 Inf values in p channel — correctly skipped (n_skipped_nonfinite=1).

### accumulate_batch NaN bug fix (PR #791)
- `0 * NaN = NaN` in `evaluate_split` — fixed in PR #791. All subsequent experiments include this fix.

## Update History

- 2026-04-28: Round 1 launched. 8 experiments in flight.
- 2026-04-28: PR #788 merged. Huber loss: val_avg=115.6496 (-8.85% vs MSE baseline 126.88).
- 2026-04-28: PR #827 merged. Huber+surf_weight=30: val_avg=109.5716 (-5.26%).
- 2026-04-28: PR #808 merged. bf16+n_hidden=256+n_head=8+Huber+epochs=12: val_avg=104.1120 (-4.97%).
- 2026-04-29: PR #882 merged. EMA decay=0.999: val_avg=103.2182 (-0.86%).
- 2026-04-29: PR #1005 merged. n_layers=3, slice_num=16: val_avg=94.6541 (-8.31%).
- 2026-04-29: PR #795 merged. PSN: val_avg=90.4014 (-4.50%).
- 2026-04-28: PR #1015 merged. Epochs=24: val_avg=66.8085 (-26.1%).
- 2026-04-29: PR #1050 merged. PSN+epochs=30: val_avg=61.5855 (-7.8%).
- 2026-04-29: Round r5 launched on icml-appendix-charlie-pai2f-r5.
- 2026-04-29: PR #1121 merged. huber_delta=0.1: val_avg=58.4790 (-5.04%), test_avg=51.3554 (-5.52%).
- 2026-04-29: PR #1120 merged. n_layers=2: val_avg=56.4257 (-3.51%), test_avg=49.6211 (-3.38%).
- 2026-04-29: PR #1134 merged. epochs=26 cosine-aligned (n_layers=2 + huber_delta=0.1 compound): val_avg=55.4877 (-1.66%), test_avg=48.8156 (-1.62%).
- 2026-04-29: PR #1136 merged. weight_decay=5e-4 (n_layers=2 + huber_delta=0.1 + epochs=30): val_avg=52.0698 (-6.16% vs PR #1134), test_avg=46.1497 (-5.46%) — **Current best.** Compete gap: 5.22.
