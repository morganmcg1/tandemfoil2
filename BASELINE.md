# Baseline — TandemFoilSet-Balanced (icml-appendix-charlie-pai2e-r1)

## Current Best

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` | **115.6496** (PR #788 — Huber loss, epoch 10) |
| `test_avg/mae_surf_p` | **40.927** (prior competition best) |

**Source:** README.md prior competition results — PR #32 (morganmcg1/tandemfoil2): "Single-head nl3/sn16 triple compound"
- W&B run: [ip8hn4tx](https://wandb.ai/wandb-applied-ai-team/senpai-kagent-v-students/runs/ip8hn4tx)
- test_single_in_dist: 46.569
- test_geom_camber_rc: 52.859
- test_geom_camber_cruise: 24.717
- test_re_rand: 39.561

**Config (best known):** n_layers=3, slice_num=16, n_hidden tuned

## Round 1 — Merged Winners

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
| #795 R2 | thorfinn | Huber + per-sample loss norm combined | **Rebase needed** | Winner: 104.2271 (-9.9% vs Huber baseline). Merge conflict after PR #788. Rebase onto advisor branch + re-run. |
| #795 R1 | thorfinn | Per-sample loss normalization (MSE) | Revision requested | -11.5% vs MSE but above Huber baseline 115.65. Re-run with Huber+norm combined (done, see R2). |
| #794 | tanjiro | LR warmup 5 epochs + cosine | Revision requested | -4.87% vs no-warmup but above Huber baseline. Shorten warmup to 2 epochs + stack on Huber. |
| #792 | frieren | Deeper Transolver: n_layers=6, lr=3e-4 | Sent back | Depth inconclusive at budget (n=5: 109.62, n=6: 113.83 at equal wall-clock). Resubmit as Huber + grad_clip 1.0 (n=5 default). |
| #789 | askeladd | Gradient clipping (max_norm=1.0) | **Rebase needed** | Winner: 114.3451. Merge conflict on advisor branch. Rebase + re-run. |
| #808 | fern | bf16 mixed precision + wider model (n_hidden=256, n_head=8) | Revision requested | 128.5863 (MSE, missing Huber). Re-run with Huber + clean architecture + epochs=12. |

## Round 1 — Closed

| PR | Student | Hypothesis | Decision | Notes |
|----|---------|------------|----------|-------|
| #790 | edward | surf_weight 10→30/50 (MSE) | Closed | Best 128.98 — 11.5% above Huber baseline. Re-assigned: surf_weight on Huber (PR #827). |

## Round 1 — Active WIPs (Running)

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #882 | nezuko | EMA model weights (decay=0.999) on Huber baseline | Running (first time) |
| #794 | tanjiro | LR warmup (2 epochs) + Huber | Revision in progress |
| #795 | thorfinn | Huber + per-sample norm — rebase + re-run | Awaiting rebase |
| #827 | alphonse | surf_weight sweep (20/30/50) on Huber baseline | Running (first time) |
| #828 | edward | AdamW weight_decay sweep (1e-4/1e-3/1e-2) on Huber baseline | Running (first time) |

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
