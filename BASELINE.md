# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2e-r1)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **103.2182** (PR #882 — EMA(decay=0.999) + bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12, epoch 10) |
| `test_avg/mae_surf_p` | **92.4867** (PR #882) |

**Source:** README.md prior competition results — PR #32 (morganmcg1/tandemfoil2): "Single-head nl3/sn16 triple compound"
- W&B run: [ip8hn4tx](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ip8hn4tx)
- test_single_in_dist: 46.569
- test_geom_camber_rc: 52.859
- test_geom_camber_cruise: 24.717
- test_re_rand: 39.561

**Config (best known):** n_layers=3, slice_num=16, n_hidden tuned

## Round 1 — Merged Winners

### PR #882 — EMA model weights (decay=0.999) on compound baseline (2026-04-29)
**Student:** charliepai2e1-nezuko | **Branch:** charliepai2e1-nezuko/ema-model-weights

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **103.2182** (epoch 10/12) |
| `val_single_in_dist/mae_surf_p` | 125.0237 |
| `val_geom_camber_rc/mae_surf_p` | 117.0167 |
| `val_geom_camber_cruise/mae_surf_p` | 78.0963 |
| `val_re_rand/mae_surf_p` | 92.6361 |
| `test_avg/mae_surf_p` | 92.4867 |
| `test_single_in_dist/mae_surf_p` | 112.5660 |
| `test_geom_camber_rc/mae_surf_p` | 105.3162 |
| `test_geom_camber_cruise/mae_surf_p` | 61.7016 |
| `test_re_rand/mae_surf_p` | 90.3631 |

**vs prior baseline (PR #808):** 103.22 vs 104.11 → **-0.86% improvement**
**Metric summary:** `target/metrics/charliepai2e1-nezuko-ema-rebased-i2fjdqe3.jsonl`
**Reproduce:** `cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999`

### PR #808 — bf16 mixed precision + wider model (n_hidden=256, n_head=8) + Huber + epochs=12 (2026-04-28)
**Student:** charliepai2e1-fern | **Branch:** charliepai2e1-fern/bf16-wider-model

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **104.1120** (epoch 10/12) |
| `val_single_in_dist/mae_surf_p` | 121.3574 |
| `val_geom_camber_rc/mae_surf_p` | 110.5814 |
| `val_geom_camber_cruise/mae_surf_p` | 88.3729 |
| `val_re_rand/mae_surf_p` | 96.1362 |
| `test_avg/mae_surf_p` | 94.7010 |
| `test_single_in_dist/mae_surf_p` | 109.1814 |
| `test_geom_camber_rc/mae_surf_p` | 98.6706 |
| `test_geom_camber_cruise/mae_surf_p` | 75.3729 |
| `test_re_rand/mae_surf_p` | 95.5792 |

**vs prior baseline (PR #827):** 104.11 vs 109.57 → **-4.97% improvement**
**Metric summary:** `target/metrics/charliepai2e1-fern-bf16-wider-huber-ep12-rebased-jo0imdkm.jsonl`
**Reproduce:** `cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12`

### PR #827 — Huber loss + surf_weight=30 (2026-04-28 22:57)
**Student:** charliepai2e1-alphonse | **Branch:** charliepai2e1-alphonse/huber-surf-weight-sweep

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **109.5716** (epoch 13/14) |
| `val_single_in_dist/mae_surf_p` | 125.7142 |
| `val_geom_camber_rc/mae_surf_p` | 128.1718 |
| `val_geom_camber_cruise/mae_surf_p` | 84.5458 |
| `val_re_rand/mae_surf_p` | 99.8546 |
| `test_single_in_dist/mae_surf_p` | 114.7811 |
| `test_geom_camber_rc/mae_surf_p` | 111.9464 |
| `test_re_rand/mae_surf_p` | 96.7687 |

**vs Huber baseline (PR #788):** 109.57 vs 115.65 → **-5.26% improvement**
**Metric summaries:**
- `target/metrics/charliepai2e1-alphonse-huber-surf30-cy8mg2si.jsonl`
- `target/metrics/charliepai2e1-alphonse-huber-surf20-n6axj9cr.jsonl`
- `target/metrics/charliepai2e1-alphonse-huber-surf50-5i468c21.jsonl`
**Reproduce:** `cd target/ && python train.py --loss huber --huber_delta 1.0 --surf_weight 30`

### PR #788 — Huber loss instead of MSE (2026-04-28 20:49)
**Student:** charliepai2e1-alphonse | **Branch:** charliepai2e1-alphonse/l1-huber-loss

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (epoch 10/14) |
| `val_avg/mae_surf_Ux` | 1.6785 |
| `val_avg/mae_surf_Uy` | 0.7747 |
| `val_single_in_dist/mae_surf_p` | 148.4833 |
| `val_geom_camber_rc/mae_surf_p` | 120.0717 |
| `val_geom_camber_cruise/mae_surf_p` | 91.6644 |
| `val_re_rand/mae_surf_p` | 102.3790 |
| `test_avg/mae_surf_p` | NaN (cruise split pre-existing bug) |

**vs MSE baseline:** 115.65 vs 126.88 → **-8.85% improvement**
**Metric summary:** `target/metrics/charliepai2e1-alphonse-huber-delta1.0-gtc81aav.jsonl`
**Reproduce:** `cd target/ && python train.py --loss huber --huber_delta 1.0`

## Round 1 — Sent Back for Revision / Awaiting Rebase

| PR | Student | Hypothesis | Decision | Notes |
|----|---------|------------|----------|-------|
| #795 R3 | thorfinn | Huber + per-sample loss norm combined | **Rebase needed** | Winner: 93.3991 (-9.5% vs PR #808). Merge conflict after PR #882. Rebase onto advisor branch + re-run. |
| #795 R2 | thorfinn | Huber + per-sample loss norm combined | **Rebase needed** | Winner: 104.2271 (-9.9% vs Huber baseline). Merge conflict after PR #788. Rebase onto advisor branch + re-run. |
| #795 R1 | thorfinn | Per-sample loss normalization (MSE) | Revision requested | -11.5% vs MSE but above Huber baseline 115.65. Re-run with Huber+norm combined (done, see R2). |
| #794 | tanjiro | LR warmup 5 epochs + cosine | Revision requested | -4.87% vs no-warmup but above Huber baseline. Shorten warmup to 2 epochs + stack on Huber. |
| #789 | askeladd | Gradient clipping (max_norm=1.0) | **Rebase needed** | Winner: 114.3451. Merge conflict on advisor branch. Rebase + re-run. |
| #808 | fern | bf16 mixed precision + wider model (n_hidden=256, n_head=8) | **MERGED** | val=104.1120 (PR #808 Round 3 with Huber+epochs=12). |
| #882 | nezuko | EMA model weights (decay=0.999) on compound | **MERGED** | val=103.2182. Current best baseline. |

## Round 1 — Closed

| PR | Student | Hypothesis | Decision | Notes |
|----|---------|------------|----------|-------|
| #792 | frieren | Deeper Transolver n_layers=8, grad_clip (5 rounds) | Closed | R5 rebase produced 107.54 — above current baseline (103.22). Core contribution (grad_clip) was already absorbed into compound baseline via PR #882. Train-loss NaN guard never fires with grad_clip active. Branch deleted. |
| #960 | alphonse | surf_weight sweep (20/30/50) on compound + grad_clip baseline | Closed | Clean monotone degradation: sw=20 (105.89, +2.59%), sw=30 (108.43, +5.04%), sw=50 (108.69, +5.30%) — all worse than default sw=10. Re-assigned as PR #1011 to explore sub-10. |
| #790 | edward | surf_weight 10→30/50 (MSE) | Closed | Best 128.98 — 11.5% above Huber baseline. Re-assigned: surf_weight on Huber (PR #827). |

## Round 1 — Active WIPs (Running)

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #998 | frieren | slice_num 64→128 on compound baseline (n_hidden=256, wider PhysicsAttn) | New assignment |
| #1011 | alphonse | surf_weight sub-10 sweep (1/3/5/7) on compound baseline | New assignment (follow-up to PR #960 monotone signal) |
| #942 | nezuko | EMA decay sweep: 0.99/0.995 vs 0.999 on compound | Running |
| #904 | fern | Huber delta sweep: 0.25/0.5/1.0/2.0 on wider-model baseline | Running |
| #1005 | edward | n_layers=3, slice_num=16 stacked on compound baseline (reference arch) | New assignment |
| #794 | tanjiro | LR warmup + Huber | Revision in progress |
| #795 | thorfinn | Huber + per-sample norm — rebase + re-run | Awaiting rebase (R3 winner 93.40) |
| #789 | askeladd | Gradient clipping (max_norm=1.0) | Awaiting rebase (winner 114.35) |

### Key Infrastructure Fix (PR #792 Round 2)
- `--grad_clip 1.0` + upstream pred/GT sanitization in `evaluate_split` resolves NaN propagation on `test_geom_camber_cruise`.
- Root cause: IEEE 754 `Inf * False (==0.0) = NaN` — existing `accumulate_batch` y_finite check was insufficient because `(pred - y).abs()` is computed before masking. Fix is upstream sanitization before the masked arithmetic.
- Corrupted GT: `test_geom_camber_cruise/000020.pt` has 761 Inf values in p channel — correctly skipped by fix (n_skipped_nonfinite=1).
- All future PRs should include this fix.

### Key Infrastructure Fix (PR #791)
- `accumulate_batch` NaN propagation bug fixed — `0 * NaN = NaN` in `evaluate_split`. All subsequent experiments should include this fix.

### Known Pre-existing Bug (RESOLVED in PR #792)
- `test_geom_camber_cruise/mae_surf_p` NaN across all student runs — now fixed by PR #792 NaN guard infrastructure (--grad_clip + eval sanitization). All new PRs should include this fix.

## Update History

- 2026-04-28: Round 1 launched. 8 experiments in flight.
- 2026-04-28: PR #808 added (fern bf16 follow-up). 9 active WIPs total.
- 2026-04-28 20:49: PR #788 merged. Huber loss sets new best val_avg/mae_surf_p = 115.6496 (-8.85% vs MSE baseline 126.88).
- 2026-04-28 22:00: Round 1 review cycle completed. PRs #794 and #795 sent back for revision; PR #790 closed. New assignments: PR #827 (alphonse, surf_weight+Huber), PR #828 (edward, weight_decay).
- 2026-04-28 23:15: PR #795 R2 (Huber+PSN) reviewed — winner at 104.2271 but sent back for rebase after merge conflict.
- 2026-04-28 23:30: PR #792 R2 reviewed — depth hypothesis inconclusive; hidden gem: n=5 + grad_clip achieves 109.62 (below Huber baseline); sent back for Huber+grad_clip focused retest. NaN fix infrastructure confirmed working.
- 2026-04-28 22:57: PR #827 merged. Huber+surf_weight=30 sets new best val_avg/mae_surf_p = 109.5716 (-5.26% vs Huber baseline 115.6496). New compound baseline: --loss huber --huber_delta 1.0 --surf_weight 30.
- 2026-04-28: PR #808 merged. bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12 sets new best val_avg/mae_surf_p = 104.1120 (-4.97% vs PR #827 baseline 109.5716). New compound baseline: --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12.
- 2026-04-29: PR #882 merged. EMA decay=0.999 on compound sets new best val_avg/mae_surf_p = 103.2182 (-0.86% vs PR #808). Current baseline.
- 2026-04-29: PR #792 R4 reviewed — outstanding result (val_avg=90.7796, -12.0% vs #882, wins on every val+test split) but merge-conflicted; sent back for rebase onto post-#882 advisor branch.
- 2026-04-29: PR #954 (alphonse, surf_weight=30 on compound) prematurely closed without running results; re-assigned as PR #960 (surf_weight 20/30/50 sweep on compound + grad_clip).
- 2026-04-29: Fixed label mismatch on PR #942 (`student:nezuko` → `student:charliepai2e1-nezuko`).
- 2026-04-29: PR #792 closed after 5 rounds. R5 rebase run produced val_avg=107.54 — above current baseline (103.22). grad_clip already in compound baseline via #882; unique delta (train-loss NaN guard) provides no measurable gain with grad_clip active.
- 2026-04-29: PR #960 (alphonse, surf_weight sweep 20/30/50) closed — clean monotone degradation: sw=20 (+2.59%), sw=30 (+5.04%), sw=50 (+5.30%). Optimal surf_weight has shifted below default 10 on compound stack. Re-assigned as PR #1011 (sub-10 sweep: sw=1/3/5/7).
