# SENPAI Research State
- 2026-04-28 01:55 — round 1.5 active; **five big wins merged**: #356 (EMA, −3.1 %), #374 (grad-clip(1.0), −14.45 %), #402 (grad-clip(0.5), −2.07 %), #408 (lr=1e-3, −2.59 %), **#417 (EMA(0.99), −8.69 %)**
- Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val splits); ranking final metric is `test_avg/mae_surf_p`

## Current best (post-PR-#417)
- **`val_avg/mae_surf_p` = 98.581** (EMA, ep13/50 timeout-cut)
- **`test_avg/mae_surf_p` = 87.881**
- See `BASELINE.md` for the full per-split breakdown.
- **Pending winners** (both rebasing onto post-#374):
  - **PR #352 (smoothl1-surface)**: raw run measured val=105.56, test=95.39 (−20.2 % / −19.2 % vs prior #356). Projected post-rebase: val ≈ 90, test ≈ 80 if SmoothL1 composes with EMA + grad-clip.
  - **PR #394 (torch.compile)**: confirmed −23.1 % per-epoch (17 vs 13 epochs in 30 min). Metric vs current #374 was +0.79 % / +2.13 % (run pre-dated grad-clip). Projected post-rebase: val ~108–110, test ~95–97 (compile + grad-clip + 17 epochs).

## Resolved: scoring NaN bug
- **Root cause** (independently flagged by tanjiro on #356 and askeladd on #351): one sample (`test_geom_camber_cruise` idx 20) has non-finite `y[p]`. `data/scoring.py:accumulate_batch` builds the right per-sample mask but does `err = |pred − y|` *before* the masked sum, so IEEE-754 `NaN*0 = NaN` (and `inf*0 = NaN`) defeats it and poisons the float64 accumulator → `mae_surf_p`/`mae_vol_p` go NaN for the whole split.
- **Fix:** NaN-safe pre-pass in `train.py:evaluate_split` (drops bad-y samples from `mask`, zeros their `y`). Now in baseline. `data/scoring.py` left untouched per program contract.
- **Action item for in-flight PRs:** any returning PR that pre-dates the post-#356 baseline must rebase to pick up the fix; otherwise their `test_avg/mae_surf_p` will be NaN.

## Round 1 hypothesis portfolio status

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #350 | alphonse  | bigger-transolver-bf16   | Architecture (n_hidden 128→256, n_head 4→8) + bf16 | wip |
| ~~#351~~ | ~~askeladd~~  | ~~surf-weight-50~~           | ~~Loss balance (10→50)~~ | **CLOSED 04-28 01:02**: re-run on post-#356 base gave val=131.02 (−0.95 % vs #356 / +15.8 % vs current #374), test=117.90 (−0.12 % vs #356 / +18.7 % vs #374). Within-noise gain on rebased baseline; grad-clip merged after rebase started. EMA absorbs most of the surface-signal gain. Reassigned to #417. |
| #352 | edward    | smoothl1-surface         | Loss form (SmoothL1 β=1 on surface) | **sent back 04-28 00:10** for rebase + re-run: val=105.56 raw / test=95.39 (−20.2% / −19.2% vs EMA baseline; raw-vs-raw −22.7%). Decisive winner; conflicts with merged #356 in `evaluate_split`. Will merge as new baseline once post-rebase numbers land. |
| ~~#353~~ | ~~fern~~      | ~~warmup-cosine-1e3~~        | ~~LR schedule (5-ep warmup + cosine to 1e-5, peak 1e-3)~~ | **CLOSED 04-28 00:52**: pre-EMA/pre-grad-clip base; val=139.91 raw, +2.5% vs #356 raw best, +14.7% vs #374 raw. Schedule was degenerate (cosine T_max=50, only 7 decay epochs, LR barely fell). Peak lr=1e-3 trained stably but val oscillated — exactly the noise grad-clip damps. Reassigned to #408. |
| ~~#354~~ | ~~frieren~~   | ~~slice-128-heads-8~~        | ~~Slice/head count (slice 64→128, n_head 4→8)~~ | **CLOSED 23:51**: val=156.48 (+18%), test=144.10 (+22%); throughput-bound (250 s/ep, 8/50 epochs) |
| ~~#355~~ | ~~nezuko~~    | ~~mlp-ratio-4~~              | ~~MLP capacity (mlp_ratio 2→4)~~ | **CLOSED 04-28 00:30**: re-run on EMA baseline gave val=132.96 (+0.52 %), test=118.09 (+0.04 %) — wash. Real raw-vs-raw gain (−5.2 %) hidden by EMA at 12-ep budget. In-dist −2.7 % vs OOD +0.3 % to +2.1 % suggests capacity → in-dist memorization. Reassigned to #398 SwiGLU at matched params. |
| #356 | tanjiro   | ema-eval                 | EMA(0.999) shadow for val + checkpoint | **MERGED 23:42** as first round-1 baseline (val=132.276, test=118.041) |
| ~~#357~~ | ~~thorfinn~~  | ~~channel-weighted-loss~~    | ~~Per-channel surface weights ([1,1,5] for Ux,Uy,p)~~ | **CLOSED 04-28 00:18**: val=150.91 (+14.1 %), test=143.07 (+21.2 %); raw-vs-raw +10.5 %. Severe per-epoch oscillation; loss-shape lever (#352) dominates this direction by ~30 %. |

## Round 1.5 follow-up assignments (post-#356, all targeting `icml-appendix-charlie-pai2d-r1` baseline)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| ~~#373~~ | ~~frieren~~ | ~~mixed-slice-last-layer~~ | ~~Last-layer-only `slice_num=128`~~ | **CLOSED 04-28 00:48**: val=133.49 (+0.92 %), test=120.85 (+2.38 %); same in-dist-helps/OOD-regresses pattern as closed #355. Reassigned to #403. |
| #374 | tanjiro | grad-clip-1p0 | Gradient clipping at `max_norm=1.0` between backward and step | **MERGED 00:43** as second round-1 baseline (val=113.157, test=99.322) — −14.45 % val / −15.86 % test vs #356. Pre-clip norm 50–100× max_norm → effective LR cap. |
| #394 | thorfinn | torch-compile-throughput | `torch.compile(model, ema_model)` mode=default, dynamic=True | **Throughput delivery confirmed −23.1 % per-epoch (108.4 s/ep, 17 epochs in 30 min). Sent back 04-28 01:18** for rebase onto post-#374 (was on pre-grad-clip base). Predicted post-rebase: val ~108–110, test ~95–97 → new baseline + throughput multiplier (every future PR fits 17 epochs). |
| #398 | nezuko | swiglu-mlp-matched | SwiGLU MLP `(W_g(x)⊙silu(W_v(x)))W_o` at `swiglu_inner=168`, matched to baseline param count | **Strong win on prior baseline** (val=111.795 vs #356 132.276 = −15.48 %; per-split fixes the in-dist-vs-OOD trade-off from closed #355). **Sent back 01:42** for rebase onto post-#408 — vs current baseline is +0.88 % val / +2.83 % test (within noise). Projected post-rebase: val ~94–98, test ~83–87. |
| #402 | tanjiro | grad-clip-0p5 | Aggressive grad-clip: `max_norm=1.0 → 0.5` | **MERGED 01:29** as new baseline (val=110.822 / test=97.955; −2.07 % / −1.38 % vs #374). Diminishing-returns curve on clipping lever now mapped. |
| ~~#403~~ | ~~frieren~~ | ~~batch8-lr-sqrt2~~ | ~~`batch_size=4 → 8`, `lr=5e-4 → 7e-4` (√2 scaling)~~ | **CLOSED 04-28 02:11**: val +75 % / test +74 % vs current baseline. Step-count starvation dominates (b=8 halves steps/ep, √2 LR under-compensates). Variance-reduction lever real (grad_norm −16 %), but eaten by missing late-training updates. Reassigned frieren to #458. |
| #408 | fern | higher-lr-1e3 | `Config.lr = 5e-4 → 1e-3` on merged grad-clip baseline | **MERGED 01:41** as new baseline (val=107.957 / test=95.675; −2.59 % / −2.33 % vs #402). Pre-clip grad norm halved at lr=1e-3 — AdamW preconditioner adapts; clip envelope dominates per-step magnitude. |
| #417 | askeladd | ema-decay-0p99 | `ema_decay = 0.999 → 0.99` | **MERGED 01:54** as new baseline (val=98.581 / test=87.881; −8.69 % / −8.15 % vs #408). Mechanism confirmed: under-converged iterate is improving fast, shorter EMA window captures recent (better) iterate. Raw at ep13 essentially unchanged — gain came from better shadow extraction. |
| #430 | tanjiro | lion-optimizer | Lion (sign-of-momentum) replacing AdamW; `lr=1.7e-4`, `wd=3e-4`, betas=(0.9, 0.99) | Fresh axis after three merged variance-reduction wins (#356/#374/#402). Reported 1–3 % gains on transformer-shaped problems; sign-update naturally bounds per-param step magnitude |
| #438 | fern | lr-2e-3 | `Config.lr = 1e-3 → 2e-3` on merged #408 baseline | Fern's own follow-up #1; tests how far the LR-scaling-under-clip envelope extends. Single-knob continuation. |
| #445 | askeladd | ema-decay-0p95 | `ema_decay = 0.99 → 0.95` on merged #417 baseline | Askeladd's own follow-up #1; tests where the EMA-decay responsiveness curve bottoms out. Honest predicted band −1 % to +5 %. |
| #458 | frieren | weight-decay-5e-4 | `Config.weight_decay = 1e-4 → 5e-4` on merged #417 baseline | Replaces closed #403; standard regularization sweep. Plausibly helps OOD splits where capacity bumps (#355/#373) showed in-dist-helps / OOD-regresses. Honest band −1 % to +2 %. |

## Updated picture from round-1 returns
- **#356 (EMA) merged** at val=132.276 (−3.1 % vs same-run best raw).
- **#374 (grad-clip(1.0)) merged** at val=113.157 (−14.45 % val, −15.86 % test). Pre-clip grad norms 50–100× max_norm → clip is acting as effective LR cap.
- **#402 (grad-clip(0.5)) merged** at val=110.822 (−2.07 %), test=97.955 (−1.38 %). **Diminishing-returns curve on clipping lever now mapped**: any-clip = −14 %, 1.0 → 0.5 = −2 %.
- **#408 (lr=1e-3) merged** at val=107.957 (−2.59 %), test=95.675 (−2.33 %). Pre-clip grad norm halved at lr=1e-3 (mean ~44 vs ~73). AdamW preconditioner adapts; clip envelope dominates per-step magnitude. "Higher LR safe under clip" hypothesis confirmed.
- **#417 (EMA decay 0.999 → 0.99) merged** at val=98.581 (−8.69 % vs #408), test=87.881 (−8.15 %). Mechanism: at 13-epoch under-converged budget, shorter EMA window captures recent (better) iterate before old (worse) iterate drags the shadow back. Raw at ep13 essentially unchanged — all gain from better shadow extraction.
- **#352 (SmoothL1) raw run** beats prior #356 baseline by −20.2 % / −19.2 % — strongest single-lever delta seen. Pending rebase onto post-#417.
- **#394 (torch.compile) confirmed −23.1 % per-epoch wall clock**, 17 epochs in 30 min. Pending rebase onto post-#417.
- **#398 (SwiGLU at matched params) confirmed −15.48 % vs #356**, with per-split breakdown showing it fixes the in-dist-vs-OOD trade-off from closed #355. Pending rebase onto post-#417.
- **Variance reduction is the dominant winning direction so far**:
  - iterate-level: EMA (merged)
  - step-magnitude-level: grad-clip(1.0 then 0.5) (both merged, diminishing returns mapped)
  - LR-scaling under clip envelope: lr=1e-3 (merged); lr=2e-3 in flight (#438)
  - aggregation-level: larger batch (PR #403, in flight)
- **Loss-form direction** strongly winning: SmoothL1 (PR #352, pending rebase, projected val ~88–92).
- **Per-node-nonlinearity direction** decoupled from capacity: SwiGLU (#398, pending rebase, projected val ~94–98).
- **Closed levers**: more capacity at this epoch budget (#355 mlp_ratio=4, #373 last-layer slice_num=128) showed an in-dist-helps / OOD-regresses pattern. SwiGLU is the "fix" for that pathology — same hidden_dim, just different activation/gating shape, and OOD splits all gain 11–14 % on it.

## Round 2 candidates (queued)
Once round 1 finishes (best-of-merged-and-still-WIP) and we have a few merged compounders, the next round will pull from:

- **L1-only surface loss + β-sweep** for SmoothL1 (β ∈ {0.25, 0.5, 1.0, 2.0}) — directly follows from #352's gain. β=1 may be too generous once mid-training residuals shrink below 1σ.
- **SmoothL1 on volume too** — if surface SmoothL1 holds, volume is the natural propagation; pure-L1 surface is a worthwhile end-of-spectrum point.
- **SmoothL1 + channel weighting + surf_weight** — three loss-shape levers that may compound (#352, #357, #351).

- **EMA decay sweep** at fixed budget — try `ema_decay ∈ {0.9999}` (slow) and a warmup-EMA variant (skip first 1–2 epochs of EMA to avoid the random-init drag tanjiro observed at ep1). Polyak-Ruppert bias correction (`/ (1 - decay^t)`) is cheap and tightens early epochs.
- **SwiGLU MLP** at matched param count vs. plain GELU `mlp_ratio=4` — modern transformer recipe; nezuko flagged it.
- **Width × MLP ratio sweep** (`n_hidden ∈ {160, 192}` × `mlp_ratio=4`) once mlp_ratio=4 is confirmed.
- **`surf_weight ∈ {25, 100}`** to bracket the optimum once round 1 establishes whether 50 helps under EMA.
- **Optimizer changes** (Lion, Adan, SOAP).
- **Mesh/sample augmentation** (rotation, sub-sampling for larger effective batch).
- **Physics-informed regularization** (divergence-free / mass conservation auxiliary loss).
- **Multi-scale slice attention** (mix slice_num=32, 64, 128 across layers).
- **Re-engineering of input features** (log-Re bucketing, Fourier position features, distance-to-leading-edge).
- **Per-domain conditioning** (single vs raceCar tandem vs cruise tandem).
- **Throughput levers**: 30-min timeout currently fits ~12–13 epochs of 50; gradient accumulation, smaller batch with memory-light forward, slice subsampling, or torch.compile could buy more epochs in the same wall clock.
- **Train/val mismatch diagnostics**: `val_single_in_dist` is consistently the *worst* split (170.491 at baseline) despite being labelled "in-distribution sanity". Worth investigating — likely the high-Re raceCar single tail.
