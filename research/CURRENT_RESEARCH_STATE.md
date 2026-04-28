# SENPAI Research State
- 2026-04-27 23:42 — round 1 in flight; **first merged baseline established by PR #356**
- Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across the four val splits); ranking final metric is `test_avg/mae_surf_p`

## Current best (post-PR-#356, baseline of round 1)
- **`val_avg/mae_surf_p` = 132.276** (EMA, ep13/50 timeout-cut)
- **`test_avg/mae_surf_p` = 118.041**
- See `BASELINE.md` for the full per-split breakdown.
- **Pending winner**: PR #352 (smoothl1-surface) raw run measured val=105.56, test=95.39 (−20.2% / −19.2% vs current). Sent back for rebase onto post-#356; will merge as new baseline after re-run.

## Resolved: scoring NaN bug
- **Root cause** (independently flagged by tanjiro on #356 and askeladd on #351): one sample (`test_geom_camber_cruise` idx 20) has non-finite `y[p]`. `data/scoring.py:accumulate_batch` builds the right per-sample mask but does `err = |pred − y|` *before* the masked sum, so IEEE-754 `NaN*0 = NaN` (and `inf*0 = NaN`) defeats it and poisons the float64 accumulator → `mae_surf_p`/`mae_vol_p` go NaN for the whole split.
- **Fix:** NaN-safe pre-pass in `train.py:evaluate_split` (drops bad-y samples from `mask`, zeros their `y`). Now in baseline. `data/scoring.py` left untouched per program contract.
- **Action item for in-flight PRs:** any returning PR that pre-dates the post-#356 baseline must rebase to pick up the fix; otherwise their `test_avg/mae_surf_p` will be NaN.

## Round 1 hypothesis portfolio status

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #350 | alphonse  | bigger-transolver-bf16   | Architecture (n_hidden 128→256, n_head 4→8) + bf16 | wip |
| #351 | askeladd  | surf-weight-50           | Loss balance (10→50) | **sent back 23:42**: val=135.19 raw, test NaN; rebase onto post-#356 + retain surf_weight=50; test compounding with EMA |
| #352 | edward    | smoothl1-surface         | Loss form (SmoothL1 β=1 on surface) | **sent back 04-28 00:10** for rebase + re-run: val=105.56 raw / test=95.39 (−20.2% / −19.2% vs EMA baseline; raw-vs-raw −22.7%). Decisive winner; conflicts with merged #356 in `evaluate_split`. Will merge as new baseline once post-rebase numbers land. |
| #353 | fern      | warmup-cosine-1e3        | LR schedule (5-ep warmup + cosine to 1e-5, peak 1e-3) | wip |
| ~~#354~~ | ~~frieren~~   | ~~slice-128-heads-8~~        | ~~Slice/head count (slice 64→128, n_head 4→8)~~ | **CLOSED 23:51**: val=156.48 (+18%), test=144.10 (+22%); throughput-bound (250 s/ep, 8/50 epochs) |
| ~~#355~~ | ~~nezuko~~    | ~~mlp-ratio-4~~              | ~~MLP capacity (mlp_ratio 2→4)~~ | **CLOSED 04-28 00:30**: re-run on EMA baseline gave val=132.96 (+0.52 %), test=118.09 (+0.04 %) — wash. Real raw-vs-raw gain (−5.2 %) hidden by EMA at 12-ep budget. In-dist −2.7 % vs OOD +0.3 % to +2.1 % suggests capacity → in-dist memorization. Reassigned to #398 SwiGLU at matched params. |
| #356 | tanjiro   | ema-eval                 | EMA(0.999) shadow for val + checkpoint | **MERGED 23:42** as new baseline (val=132.276, test=118.041) |
| ~~#357~~ | ~~thorfinn~~  | ~~channel-weighted-loss~~    | ~~Per-channel surface weights ([1,1,5] for Ux,Uy,p)~~ | **CLOSED 04-28 00:18**: val=150.91 (+14.1 %), test=143.07 (+21.2 %); raw-vs-raw +10.5 %. Severe per-epoch oscillation; loss-shape lever (#352) dominates this direction by ~30 %. |

## Round 1.5 follow-up assignments (post-#356, all targeting `icml-appendix-charlie-pai2d-r1` baseline)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #373 | frieren | mixed-slice-last-layer | Last-layer-only `slice_num=128` (mixed slicing) | Replaces closed #354; pays slice cost only at the regression head — fits in 30-min budget |
| #374 | tanjiro | grad-clip-1p0 | Gradient clipping at `max_norm=1.0` between backward and step | Variance-reduction lever complementary to EMA; pre-clip grad norm logged as diagnostic |
| #394 | thorfinn | torch-compile-throughput | `torch.compile(model, ema_model)` mode=reduce-overhead, dynamic=True | Replaces closed #357; structural throughput improvement — every subsequent PR gets more epochs in the 30-min timeout |
| #398 | nezuko | swiglu-mlp-matched | SwiGLU MLP `(W_g(x)⊙silu(W_v(x)))W_o` at `swiglu_inner=168`, matched to baseline param count | Replaces closed #355; cleaner per-node-nonlinearity test (no capacity/wall-clock confound vs `mlp_ratio=4 GELU`) |

## Updated picture from round-1 partial returns
- **#356 (EMA) merged** as round-1 baseline at val=132.276 (−3.1% vs same-run best raw).
- **#352 (SmoothL1) raw run** beats baseline by −20.2% / −19.2% — by far the strongest single-lever delta of round 1. Pending rebase + re-run.
- **#354 (slice_num=128 + heads=8)** closed: throughput-bound at 250 s/epoch.
- The biggest signal so far: **loss form (MSE→SmoothL1) is more impactful than checkpoint smoothing or any other lever measured to date**. Round 2 priorities should re-rank to put loss-form variants high.

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
