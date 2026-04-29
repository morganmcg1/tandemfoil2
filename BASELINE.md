# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2f-r5)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **58.4790** (PR #1121 — huber_delta=0.1, epoch 22/30) |
| `test_avg/mae_surf_p` | **51.3554** (PR #1121) |

**Source:** PR #1121 — Tighter Huber loss (huber_delta=0.1) on compound PSN stack
- Branch: `charlie5-fern/huber-delta-0.1`
- Config: n_layers=3, slice_num=16, n_hidden=256, n_head=8, loss=huber, huber_delta=0.1, ema_decay=0.999, grad_clip=1.0, per_sample_norm, epochs=30, lr=5e-4, batch_size=4, surf_weight=10.0
- Val still falling ~2.9%/epoch at epoch 22 when 30-min timeout hit (LR=8.27e-5)

**Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper reference)

## Round r5 — Starting Point

All students start from the compound charlie config with huber_delta=0.1 (PR #1121 winner):

```
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm
```
*(Note: n_layers=3, slice_num=16 are hardcoded in model_config dict in train.py)*

## Round r5 — Merged Winners

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
- 2026-04-29: PR #1121 merged. huber_delta=0.1: val_avg=58.4790 (-5.04%), test_avg=51.3554 (-5.52%). **Current best.**
