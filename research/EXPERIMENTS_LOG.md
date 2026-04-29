# SENPAI Research Results

<!-- This log is maintained on the icml-appendix-charlie-pai2f-r5 advisor branch. -->
<!-- Each entry records a reviewed PR with metrics and analysis. -->

## Round 5 Experiments — Launched 2026-04-29

| PR | Student | Experiment | Status | Result |
|----|---------|------------|--------|--------|
| #1118 | edward | epochs=50 extended training | CLOSED 04-29 | val=65.16 (+5.8% vs PR #1050) — T_max stretch wasted annealing |
| #1119 | thorfinn | cosine eta_min=5e-5 | WIP | — |
| #1120 | nezuko | **n_layers=2 shallower model** | **MERGED 04-29** | **val=56.4257 (-3.51% vs PR #1121), test=49.6211 (-3.38%)** |
| #1121 | fern | **huber_delta=0.1 tighter loss** | **MERGED 04-29** | **val=58.4790 (-5.04%), test=51.3554 (-5.52%)** |
| #1122 | alphonse | lr=1e-3 higher LR | CLOSED 04-29 | val=70.23 (+14.0%) — early-phase noise |
| #1123 | tanjiro | n_hidden=320 wider model | CLOSED 04-29 | val=68.83 (+11.8%) — compute-bound, only 18 epochs |
| #1124 | askeladd | weight_decay=0 no L2 | WIP | — |
| #1125 | frieren | surf_weight=5 reduced surface emphasis | CLOSED 04-29 | val=62.03 (+0.72%); vol_p improved -8% but didn't transfer |

## 2026-04-29 — Round 5 Reviews (4 PRs)

### 2026-04-29 11:00 — PR #1121 (MERGED, NEW BASELINE): Tighter Huber loss huber_delta=0.1
- Student: charliepai2f5-fern | Branch: `charlie5-fern/huber-delta-0.1`
- **Hypothesis:** With per-sample-norm shrinking residuals to ~O(1), huber_delta=1.0 means the entire residual range is in the L2 bowl. Tightening to delta=0.1 puts only the smallest residuals in L2 (where gradient is sharpest) and outliers in L1 (capped magnitude). Predicted: better calibration on cruise/re_rand where extreme Re drives outliers.
- **Result:**

| Metric | Value | vs PR #1050 |
|---|---|---|
| `val_avg/mae_surf_p` | 58.4790 | **-5.04%** |
| `test_avg/mae_surf_p` | 51.3554 | **-5.52%** |
| `val_single_in_dist/mae_surf_p` | 66.2861 | -2.96% |
| `val_geom_camber_rc/mae_surf_p` | 71.2084 | -1.98% |
| `val_geom_camber_cruise/mae_surf_p` | 39.3226 | **-12.41%** |
| `val_re_rand/mae_surf_p` | 57.0991 | **-5.61%** |
| `test_geom_camber_cruise/mae_surf_p` | 32.3451 | **-14.44%** |
| `test_re_rand/mae_surf_p` | 48.5484 | **-9.41%** |
| `test_geom_camber_rc/mae_surf_p` | 64.9563 | +1.05% (small regression) |
| `test_avg/mae_surf_Ux` | 0.7276 | -14.81% |
| `test_avg/mae_surf_Uy` | 0.3657 | -12.66% |

- **Metrics JSONL:** `metrics/charliepai2f5-fern-huber-delta-0.1-jzaml14l.jsonl`
- **Analysis:** Hypothesis confirmed exactly as predicted. Cruise (-14.4%) and re_rand (-9.4%) gained most — those are the splits with most extreme Re where tighter clamp helps. The single-camber-rc test +1.05% is acceptable; that split has uniform high-Re distribution so L1 clipping costs a hair. All velocity channels improved -12 to -15%. Training was stable, monotonically improving every epoch, still falling ~3%/epoch when 30-min timeout hit (LR=8.27e-5). Same wall-clock and VRAM as baseline. **MERGED — new baseline.** Strong follow-ups: huber_delta=0.05/0.03 (even tighter), per-channel huber_delta, anneal delta down 1.0→0.1.

### 2026-04-29 11:00 — PR #1122 (CLOSED): Higher learning rate lr=1e-3
- Student: charliepai2f5-alphonse | Branch: `charlie5-alphonse/lr-1e-3`
- **Hypothesis:** lr=5e-4 may be under-stepping under timeout; lr=1e-3 might cover more parameter space per epoch. Risk: instability mitigated by grad_clip=1.0.
- **Result:** val=70.2309 (+14.0%), test=60.5497 (+11.4%). Stable training, no NaN/spikes. Best epoch 21/30 (timeout). At every epoch the lr=1e-3 trajectory was behind baseline. Epoch 1 val=321 vs baseline's lower start.
- **Analysis:** Without warmup, the first ~5-8 epochs are lost climbing out of a worse loss surface region. AdamW's bias-correction transient + a noisy loss surface eat the early budget. Per-split regression uniform → undertrained, not over-stepped. **CLOSED.** Follow-up worth pursuing: linear warmup (1-2 ep) + cosine to lr=1e-3.

### 2026-04-29 11:00 — PR #1123 (CLOSED): Wider model n_hidden=320
- Student: charliepai2f5-tanjiro | Branch: `charlie5-tanjiro/n-hidden-320`
- **Hypothesis:** +25% capacity per layer for tandem flow patterns. VRAM budget ample.
- **Result:** val=68.8261 (+11.8%), test=60.4215 (+11.2%). Params 2.50M (1.56× baseline), VRAM 37.0 GB. Epoch time +31% slower → only 18/30 epochs in budget (vs 22/30 baseline). At cutoff, val still falling ~3.5%/epoch with LR=2.45e-4 (vs baseline's 8.27e-5).
- **Analysis:** Capacity isn't the bottleneck under 30-min wall-clock timeout. Wider model is compute-dominated. Per-split regression uniform → undertrained, not overfit. **CLOSED.** Follow-ups: mlp_ratio=4 (cheaper capacity boost), slice_num=24 (attention capacity), or revisit wider with longer cosine T_max.

### 2026-04-29 11:20 — PR #1120 (MERGED, NEW BASELINE): Shallower model n_layers=2
- Student: charliepai2f5-nezuko | Branch: `charlie5-nezuko/n-layers-2`
- **Hypothesis:** Shallower → faster epochs → more gradient steps and deeper-into-cosine annealing within 30-min budget. Capacity loss should be compensated by extra epochs.
- **Result:**

| Metric | Value | vs PR #1121 (n_layers=3, huber_delta=0.1) | vs PR #1050 (n_layers=3, huber_delta=1.0) |
|---|---|---|---|
| `val_avg/mae_surf_p` | 56.4257 | **-3.51%** | -8.4% |
| `test_avg/mae_surf_p` | 49.6211 | **-3.38%** | -8.7% |
| `val_single_in_dist/mae_surf_p` | 59.6760 | (n/a — different stack) | -12.6% |
| `val_geom_camber_rc/mae_surf_p` | 70.4189 | | -3.1% |
| `val_geom_camber_cruise/mae_surf_p` | 38.6126 | | -14.0% |
| `val_re_rand/mae_surf_p` | 56.9952 | | -5.8% |
| `test_avg/mae_surf_Ux` | 0.7912 | | -7.4% |
| `test_avg/mae_surf_Uy` | 0.3831 | | -8.5% |

- **Run config:** n_layers=2, but huber_delta=1.0 in this run (NOT 0.1) — student branched from PR #1050 era.
- **Throughput:** 26/30 epochs in budget (vs 22 for n_layers=3); peak VRAM 22.22 GB (-27%); params 1,141,299 (-29%).
- **Metrics JSONL:** `metrics/charliepai2f5-nezuko-n-layers-2-93bfb7ek.jsonl`
- **Analysis:** Strong throughput-driven win. Cosine fully spent (final LR=2.16e-5, deeper than PR #1050's 8.27e-5) so the model entered the low-LR fine-tuning regime — exactly the regime PR #1118 (epochs=50) demonstrated was missing for that experiment. Val still falling at termination, so further headroom exists. **MERGED.** All four splits improved on both val and test, no regression. **CRITICAL COMPOUND OPPORTUNITY:** This run did not include huber_delta=0.1 (PR #1121's win). Combining n_layers=2 + huber_delta=0.1 should compound to even better numbers — recommended baseline command going forward.

### 2026-04-29 11:20 — PR #1118 (CLOSED): Extend training budget epochs=50
- Student: charliepai2f5-edward | Branch: `charlie5-edward/epochs-50`
- **Hypothesis:** Longer T_max=50 keeps LR in useful range longer. Mitigates the timeout-truncation seen in PR #1050.
- **Result:** val=65.1571 (+5.8% vs PR #1050), test=56.7468 (+4.4%). Hit 30-min timeout at epoch 22 — same as PR #1050. LR at termination 2.97e-4 vs PR #1050's 8.27e-5.
- **Analysis:** Hypothesis cleanly disproved. T_max=50 stretches cosine so the run terminates at LR 3.6× higher than PR #1050 — the model never enters the low-LR fine-tuning regime. The gain in PR #1050 wasn't from more steps; it was from those steps happening at lower LR. Excellent diagnosis from student. **CLOSED.** Follow-up assigned to edward: `epochs=26` (T_max-aligned to actual budget under n_layers=2 throughput).

### 2026-04-29 11:00 — PR #1125 (CLOSED): Reduced surface weight surf_weight=5
- Student: charliepai2f5-frieren | Branch: `charlie5-frieren/surf-weight-5`
- **Hypothesis:** Lower surface emphasis lets volume gradients improve flow physics, indirectly helping surface metric. Uncertain prediction.
- **Result:** val=62.0307 (+0.72%), test=54.5126 (+0.29%). Volume-p improved -8% (val and test). Per-split mixed: cruise & test_single_in_dist improved 1-3%, but rc and re_rand regressed +3-4%.
- **Analysis:** The internal effect predicted (better volume) did happen but didn't transfer to surface — model simply traded along the Pareto front along surf_weight. Surface and volume MAE are coupled by loss-weight curve, and we were already at (or past) the surface optimum at sw=10. **CLOSED.** Follow-up: surf_weight=15 or 20 (other direction) on the new huber_delta=0.1 baseline; focal-style weighting on hard surface nodes.

## Prior Round Winners (Full History, most recent first)

### PR #1050 — PSN + epochs=30 on compound stack (2026-04-29) — CURRENT BEST
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

- **vs prior baseline (PR #1015):** 61.5855 vs 66.8085 → -7.8% val improvement
- **Test improvement:** 54.3573 vs 58.7266 → -7.4% test improvement
- **Model parameters:** 1,606,219 | **Peak VRAM:** ~30.44 GB | **Train time:** ~30 min (hit timeout)
- **Note:** Val still falling ~2.8%/epoch at epoch 22 when 30-min timeout hit (LR=8.27e-5). More epochs likely to yield further gains.

### PR #1015 — Longer training: epochs=24 on compound stack (2026-04-28)
**Student:** charliepai2e1-edward | **Branch:** charliepai2e1-edward/longer-training-epochs-24

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **66.8085** (epoch 22/24 — timeout at 30 min) |
| `val_single_in_dist/mae_surf_p` | 73.9641 |
| `val_geom_camber_rc/mae_surf_p` | 79.1014 |
| `val_geom_camber_cruise/mae_surf_p` | 48.9877 |
| `val_re_rand/mae_surf_p` | 65.1809 |
| `test_avg/mae_surf_p` | **58.7266** |
| `test_single_in_dist/mae_surf_p` | 67.5104 |
| `test_geom_camber_rc/mae_surf_p` | 70.2042 |
| `test_geom_camber_cruise/mae_surf_p` | 40.5897 |
| `test_re_rand/mae_surf_p` | 56.6022 |

- **vs prior baseline (PR #795):** 66.8085 vs 90.4014 → -26.1% val improvement
- **Test improvement:** 58.7266 vs 80.3748 → -27.0% test improvement
- **Peak VRAM:** 30.45 GB | **Train time:** 30.42 min (hit 30-min timeout)

### PR #795 — Per-sample loss normalization (PSN) on compound stack (2026-04-28)
**Student:** charliepai2e1-thorfinn | **Branch:** charliepai2e1-thorfinn/per-sample-loss-norm

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **90.4014** (epoch 12/12) |
| `val_single_in_dist/mae_surf_p` | 108.5561 |
| `val_geom_camber_rc/mae_surf_p` | 101.4393 |
| `val_geom_camber_cruise/mae_surf_p` | 66.9027 |
| `val_re_rand/mae_surf_p` | 84.7074 |
| `test_avg/mae_surf_p` | **80.3748** |

- **vs prior baseline (PR #1005):** 90.4014 vs 94.6541 → -4.50% improvement

### PR #1005 — n_layers=3, slice_num=16 reference architecture (2026-04-29)
**Student:** charliepai2e1-edward

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **94.6541** (epoch 12/12) |
| `test_avg/mae_surf_p` | **83.7608** |

- **vs prior baseline (PR #882):** 94.6541 vs 103.2182 → -8.31% improvement

### PR #882 — EMA model weights (decay=0.999) on compound baseline (2026-04-29)
**Student:** charliepai2e1-nezuko

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **103.2182** (epoch 10/12) |
| `test_avg/mae_surf_p` | **92.4867** |

- **vs prior baseline (PR #808):** 103.22 vs 104.11 → -0.86% improvement

### PR #808 — bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12 (2026-04-28)
**Student:** charliepai2e1-fern

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **104.1120** (epoch 10/12) |
| `test_avg/mae_surf_p` | **94.7010** |

- **vs prior baseline (PR #827):** 104.11 vs 109.57 → -4.97% improvement

### PR #827 — Huber loss + surf_weight=30 (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.5716** (epoch 13/14) |

- **vs Huber baseline (PR #788):** 109.57 vs 115.65 → -5.26% improvement

### PR #788 — Huber loss instead of MSE (2026-04-28)
**Student:** charliepai2e1-alphonse

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (epoch 10/14) |

- **vs MSE baseline:** 115.65 vs 126.88 → -8.85% improvement

## Infrastructure Fixes

### NaN guard (PR #792)
- `--grad_clip 1.0` + upstream pred/GT sanitization in `evaluate_split` resolves NaN propagation.
- Root cause: IEEE 754 `Inf * False (==0.0) = NaN` — `(pred - y).abs()` computed before masking.
- `test_geom_camber_cruise/000020.pt` has 761 Inf values in p channel — correctly skipped (n_skipped_nonfinite=1).

### accumulate_batch NaN bug fix (PR #791)
- `0 * NaN = NaN` in `evaluate_split` — fixed in PR #791. All subsequent experiments include this fix.
