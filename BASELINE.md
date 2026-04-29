# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2f-r5)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **47.5231** (PR #1246 — OneCycleLR: peak 2e-3, 30% ramp, cosine decay) |
| `test_avg/mae_surf_p` | **41.3253** (PR #1246) |
| `test_single_in_dist/mae_surf_p` | 44.3980 |
| `test_geom_camber_rc/mae_surf_p` | 55.9591 |
| `test_geom_camber_cruise/mae_surf_p` | 25.0234 |
| `test_re_rand/mae_surf_p` | 39.9207 |

**Source:** PR #1246 — OneCycleLR replaces SequentialLR (LinearLR warmup + CosineAnnealingLR). max_lr=2e-3 (4× base), pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4, total_steps=32×375=12000. Scheduler is per-step (called after each batch).
- Branch: `charliepai2f5-nezuko/one-cycle-lr-policy`
- Config: n_layers=2 (hardcoded), slice_num=8, n_hidden=256, n_head=8, loss=huber, huber_delta=0.1, ema_decay=0.999, grad_clip=1.0, per_sample_norm, epochs=32, lr=5e-4 (config), batch_size=4, weight_decay=5e-4, adamw_beta2=0.98
- Best epoch = 32/32 (final epoch, val curve still descending — training-budget-limited)
- Peak VRAM: 20.97 GB, Wall-clock: 29.94 min, Run ID: z3y61gtc

**Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper reference) — currently +0.96% above target (gap = **0.3953**, down from 0.5298 — 47% of remaining gap closed by OneCycleLR).

## Round r5 — Recommended Working Baseline (compound n_layers=2 + huber_delta=0.1 + weight_decay=5e-4 + OneCycleLR max_lr=2e-3 + epochs=32 + adamw_beta2=0.98 + slice_num=8)

```
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 32 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 \
  --adamw_beta2 0.985
```
*(Note: n_layers=2, slice_num=8 are hardcoded in model_config dict in train.py — slice_num was changed from 16→8 in PR #1194)*
*(Note: warmup_epochs=3 activates SequentialLR: LinearLR 3 epochs ramp + CosineAnnealingLR over remaining 29 epochs)*

## Round r5 — Merged Winners

### PR #1241 — AdamW beta2=0.985 fine-tune (interpolation between 0.98 and 0.999) (2026-04-29)
**Student:** charliepai2f5-askeladd | **Branch:** charliepai2f5-askeladd/adamw-beta2-0.985

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **47.6501** (epoch 32/32 — final epoch, val curve still descending) |
| `test_avg/mae_surf_p` | **41.4598** |
| `test_single_in_dist/mae_surf_p` | 46.5426 |
| `test_geom_camber_rc/mae_surf_p` | 55.7505 |
| `test_geom_camber_cruise/mae_surf_p` | 24.1577 |
| `test_re_rand/mae_surf_p` | 39.3885 |

**vs prior baseline (PR #1194):** val 47.6501 vs 48.0121 → **-0.75% val improvement**
**Test improvement:** 41.4598 vs 41.6806 → **-0.53% test improvement**
**Compete gap:** 0.5298 (was 0.7506 — 29% of remaining gap closed)
**Mechanism:** beta2=0.985 gives ~67-step effective second-moment memory window (vs ~50-step at 0.98). Slightly smoother adaptation helps in late cosine tail where LR is very small. OOD splits benefited most: geom_camber_rc -2.61%, geom_camber_cruise -1.83%.
**Split analysis:** geom_camber_rc improved (-2.61% test) — this was the hardest-bottlenecked split. re_rand also improved (-1.89%). single_in_dist minimal change (-0.68%). geom_camber_cruise slight improvement (-1.83%).
**Budget-limited:** is_best=True at final epoch 32, val curve still descending — model still improving at termination.
**Peak VRAM:** 20.98 GB | **Wall-clock:** 29.9 min | **Run ID:** gz5kow96
**Metrics JSONL:** `metrics/charliepai2f5-askeladd-adamw-beta2-0.985-gz5kow96.jsonl`
**Reproduce:** `cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 32 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 --adamw_beta2 0.985`

### PR #1194 — slice_num=8 throughput gain + compound retest (adamw_beta2=0.98 + epochs=32) (2026-04-29)
**Student:** charliepai2f5-tanjiro | **Branch:** tanjiro/slice-num-8-throughput

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **48.0121** (epoch 32/32 — final epoch, val curve still descending) |
| `test_avg/mae_surf_p` | **41.6806** |
| `test_single_in_dist/mae_surf_p` | 46.2342 |
| `test_geom_camber_rc/mae_surf_p` | 57.2466 |
| `test_geom_camber_cruise/mae_surf_p` | 24.6087 |
| `test_re_rand/mae_surf_p` | 38.6329 |

**vs prior baseline (PR #1191):** val 48.0121 vs 49.0719 → **-2.16% val improvement**
**Test improvement:** 41.6806 vs 42.8204 → **-2.66% test improvement**
**Compete gap:** 0.7506 (was 1.89 — closed 60% of remaining gap in one compound step)
**Mechanism:** slice_num: 16→8 reduces FLOPs per attention head, enabling ~6.5% faster epochs (56.26 s/epoch vs 60.2 s). With 30-min budget, this frees 2 extra epochs (32 vs 30). The throughput gain directly translates to more gradient steps at low LR in the cosine tail — where descending LR has maximum benefit. VRAM also drops from 22.22 GB to 20.97 GB.
**Split analysis:** Biggest gains on re_rand (-4.41%), single_in_dist (-3.99%), geom_camber_cruise (-3.72%). geom_camber_rc essentially flat (+0.17% test) — this split may be a harder bottleneck.
**Budget-limited:** is_best=True at final epoch 32, loss curve still descending — model still improving at termination.
**Peak VRAM:** 20.97 GB | **Wall-clock:** 30.01 min | **Run ID:** 2itg2geu
**Metrics JSONL:** `metrics/charliepai2f5-tanjiro-slice-num-8-adamw-beta2-0.98-epochs-32-2itg2geu.jsonl`
**Reproduce:** `cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 32 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 --adamw_beta2 0.98` *(plus slice_num: 16→8 in model_config dict in train.py)*

### PR #1191 — AdamW betas=(0.9, 0.98): LLaMA-style faster second moment adaptation (2026-04-29)
**Student:** charliepai2f5-nezuko | **Branch:** nezuko/adamw-betas-0.9-0.98

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **49.0719** (epoch 30/30 — final epoch, val curve still descending) |
| `val_single_in_dist/mae_surf_p` | 51.8270 |
| `val_geom_camber_rc/mae_surf_p` | 63.0721 |
| `val_geom_camber_cruise/mae_surf_p` | 30.7061 |
| `val_re_rand/mae_surf_p` | 50.6823 |
| `test_avg/mae_surf_p` | **42.8204** |
| `test_single_in_dist/mae_surf_p` | 48.1549 |
| `test_geom_camber_rc/mae_surf_p` | 57.1495 |
| `test_geom_camber_cruise/mae_surf_p` | 25.5606 |
| `test_re_rand/mae_surf_p` | 40.4164 |

**vs prior baseline (PR #1149):** 49.0719 vs 51.0626 → **-3.90% val improvement**
**Test improvement:** 42.8204 vs 44.7020 → **-4.21% test improvement**
**vs paper target:** compete gap 1.89 (from 3.77 prior best) — test_avg now only +4.6% above Transolver reference
**Run config:** n_layers=2 (hardcoded), huber_delta=0.1, weight_decay=5e-4, warmup_epochs=3, epochs=30 (full completion), lr=5e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm, adamw_beta2=0.98
**Mechanism:** Faster second moment adaptation (β2=0.98 vs default 0.999) helps optimizer respond more quickly to gradient magnitude changes in short 30-epoch runs. Previous β2=0.95 (PR #1157) caused +4.19% regression; 0.98 is the sweet spot.
**Split analysis:** Biggest gains on single_in_dist (-7.40% val) and geom_camber_rc (-4.47% val). geom_camber_cruise near-saturated, minor gain. re_rand: small val gain (-1.11%) but larger test gain (-3.70%).
**Budget-limited:** Best val at final epoch 30, loss curve still descending — more epochs may yield further gains.
**Peak VRAM:** 22.22 GB | **Wall-clock:** 30.1 min | **Run ID:** 1ie79roq
**Metrics JSONL:** `metrics/charliepai2f5-nezuko-adamw-beta2-0.98-1ie79roq.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 --adamw_beta2 0.98`

### PR #1149 — warmup-lr-schedule: 3-epoch linear LR warmup before cosine decay (2026-04-29)
**Student:** charliepai2f5-fern | **Branch:** charliepai2f5-fern/warmup-lr-schedule

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **51.0626** (epoch 30/30 — full cosine completion, LR=0.0) |
| `val_single_in_dist/mae_surf_p` | 55.9743 |
| `val_geom_camber_rc/mae_surf_p` | 66.0244 |
| `val_geom_camber_cruise/mae_surf_p` | 31.0014 |
| `val_re_rand/mae_surf_p` | 51.2500 |
| `test_avg/mae_surf_p` | **44.7020** |
| `test_single_in_dist/mae_surf_p` | 51.3552 |
| `test_geom_camber_rc/mae_surf_p` | 59.8370 |
| `test_geom_camber_cruise/mae_surf_p` | 25.6420 |
| `test_re_rand/mae_surf_p` | 41.9738 |

**vs prior baseline (PR #1154):** 51.0626 vs 51.8005 → **-1.43% val improvement**
**Test improvement:** 44.7020 vs 45.3699 → **-1.47% test improvement**
**vs paper target:** compete gap 3.77 (from 4.44 prior best)
**Run config:** n_layers=2 (hardcoded), huber_delta=0.1, weight_decay=5e-4, warmup_epochs=3, epochs=30 (full completion), lr=5e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
**Key win:** warmup stacks cleanly on wd=5e-4 — biggest gain on geom_camber_cruise (val -8.89%, test -9.57%). Warmup provides complementary mechanism (gentler initialization) to weight_decay (regularization).
**Best epoch:** 30/30 — model still monotonically improving at final epoch, cosine-decayed to LR=0.0. Still training-budget-limited.
**Peak VRAM:** 22.22 GB | **Wall-clock:** 30.00 min | **Run ID:** vvp4agqj
**Metrics JSONL:** `metrics/charliepai2f5-fern-warmup-lr-schedule-wd5e-4-vvp4agqj.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3`

### PR #1154 — Extended training epochs=35 (budget-limited to epoch 27) (2026-04-29)
**Student:** charliepai2f5-nezuko | **Branch:** charliepai2f5-nezuko/epochs-35-budget-extension

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **51.8005** (epoch 27/35 — cut by timeout, LR=6.2e-5) |
| `val_single_in_dist/mae_surf_p` | ~55.5 (estimated) |
| `val_geom_camber_rc/mae_surf_p` | ~67.0 (estimated) |
| `val_geom_camber_cruise/mae_surf_p` | ~32.3 (estimated) |
| `val_re_rand/mae_surf_p` | ~52.3 (estimated) |
| `test_avg/mae_surf_p` | **45.3699** |
| `test_single_in_dist/mae_surf_p` | 49.8059 |
| `test_geom_camber_rc/mae_surf_p` | 60.8141 |
| `test_geom_camber_cruise/mae_surf_p` | 27.6087 |
| `test_re_rand/mae_surf_p` | 43.2508 |

**vs prior baseline (PR #1136):** 51.8005 vs 52.0698 → **-0.52% val improvement**
**Test improvement:** 45.3699 vs 46.1497 → **-1.69% test improvement**
**vs paper target:** compete gap 4.44 (from 5.22 prior best)
**Run config:** n_layers=2 (hardcoded), huber_delta=0.1, weight_decay=5e-4, epochs=35 (cut at 27/35 by timeout), lr=5e-4, batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
**Budget analysis:** Slower cosine decay (T_max=35 vs T_max=30) → at timeout epoch 27, LR=6.2e-5 instead of 0.0. Model still improving at cut — would likely continue to improve with more time/epochs.
**Split analysis:** Biggest gains on in_dist (-3.81%) and geom_camber_cruise (-2.63%). re_rand slightly worse (+0.48%).
**Peak VRAM:** 22.225 GB | **Wall-clock:** 30.48 min | **Run ID:** l5g7r4k9
**Metrics JSONL:** `metrics/charliepai2f5-nezuko-epochs-35-budget-extension-l5g7r4k9.jsonl`
**Reproduce:** `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 35 --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4`

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
- 2026-04-29: PR #1136 merged. weight_decay=5e-4 (n_layers=2 + huber_delta=0.1 + epochs=30): val_avg=52.0698 (-6.16% vs PR #1134), test_avg=46.1497 (-5.46%). Compete gap: 5.22.
- 2026-04-29: PR #1149 merged. warmup_epochs=3 (linear LR ramp + cosine): val_avg=51.0626 (-1.43%), test_avg=44.7020 (-1.47%). Compete gap: 3.77.
- 2026-04-29: PR #1191 merged. adamw_beta2=0.98 (faster second moment adaptation): val_avg=49.0719 (-3.90%), test_avg=42.8204 (-4.21%). Compete gap: 1.89.
- 2026-04-29: PR #1194 merged. slice_num=8 (throughput) + adamw_beta2=0.98 + epochs=32 compound: val_avg=48.0121 (-2.16%), test_avg=41.6806 (-2.66%). Compete gap: 0.7506 (60% of remaining gap closed in one step).
- 2026-04-29: PR #1241 merged. adamw_beta2=0.985 fine-tune: val_avg=47.6501 (-0.75%), test_avg=41.4598 (-0.53%) — **Current best.** Compete gap: 0.5298 (29% of remaining gap closed).
