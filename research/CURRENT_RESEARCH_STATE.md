# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-28 00:45
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus

**Current best:** PR #308 (nezuko, EMA decay=0.999 + grad clip max_norm=1.0), merged commit 5bdb284. `val_avg/mae_surf_p = 106.40` (EMA-evaluated), `test_avg/mae_surf_p = 93.99`. **-16.2% over PR #287** (the prior baseline at surf_weight=25 → 126.67).

**Critical attribution caveat:** nezuko's grad clip at `max_norm=1.0` fired on 100% of batches (pre-clip gn_mean ≈ 50-100 vs threshold 1.0), so it acted as implicit unit-norm SGD on top of AdamW rather than as outlier protection. The 16% gain is shared between EMA's late-epoch smoothing and the implicit-lr-dampener effect in unknown proportion. **Ablation queued (PR #381): EMA decay=0.995 + clip=10.0** to isolate.

**Round 1 status:** 8 PRs assigned across 4 axes (loss formulation, loss weighting, architecture, optimization). 5 closed/merged so far. Round 1 is a **14-epoch ranking exercise** — the 50-epoch cosine schedule's tail is unreached for every PR at the 30-min cap.

**Throughput is a first-class research axis** going forward — alphonse (#372: bf16 autocast) and frieren (#382: larger batch) are testing the two main throughput levers in parallel. Both should compound with the existing wins from #287 and #308.

**Resolved infrastructure issue:** PR #358 (edward, commit 010235e) fixed `data/scoring.py` `inf * 0 = NaN` propagation via `torch.where`. New `data/test_scoring.py` has 4 surgical regression tests. PRs branched before that merge may still report NaN on `test_geom_camber_cruise/mae_surf_p`; we evaluate them on val_avg/mae_surf_p instead.

## Round 1 PR roster

| Student | PR | Slug | Axis | Predicted Δ | Status |
|---|---|---|---|---|---|
| alphonse | #287 | surf-weight-up | Loss weighting (10→25) | -3% to -7% | **MERGED** e4a0c18 → val_avg=126.67 |
| alphonse | #372 | bf16-autocast | Throughput (bf16 autocast) | -10% to -20% | WIP |
| askeladd | #289 | huber-loss | Loss formulation (MSE→SmoothL1) | -5% to -10% | WIP |
| edward   | #300 | wider-model | Width (192/96) | -5% to -10% | **CLOSED** — under-trained 9/50 |
| edward   | #358 | fix-scoring-nan-mask | Maintenance | n/a | **MERGED** 010235e |
| edward   | #368 | fourier-pos-encoding | Input (8-freq Fourier on (x,z)) | -3% to -8% | WIP |
| fern     | #304 | deeper-model-droppath | Depth (5→8 + DropPath 0.1) | -3% to -8% | **CLOSED** — 210 s/epoch, 9/50 epochs, equal-epoch worse |
| fern     | #388 | arcsinh-pressure | Heavy-tail (arcsinh on p target only) | -5% to -15% | WIP |
| frieren  | #307 | warmup-cosine-1e3 | Optim (warmup + lr 1e-3) | -2% to -6% | **CLOSED** — 134.58, 26% worse than #308 |
| frieren  | #382 | batch8-lr7e-4 | Throughput (larger batch + sqrt-lr) | -5% to -15% | WIP |
| nezuko   | #308 | ema-grad-clip | Optim (EMA 0.999 + clip 1.0) | -3% to -8% | **MERGED** 5bdb284 → val_avg=106.40 (NEW BEST) |
| nezuko   | #381 | ema995-gradclip10 | Ablation (EMA 0.995 + clip 10) | -3% to -10% | WIP |
| tanjiro  | #309 | more-slices | Architecture (128/8) | -3% to -7% | **CLOSED** — 2× slower, not better |
| tanjiro  | #378 | per-sample-relmse | Heavy-tail (per-sample y-var) | -3% to -7% | WIP |
| thorfinn | #310 | per-channel-surf-weights | Loss weighting (3× p) | -3% to -8% | **CLOSED** — +13% regression |
| thorfinn | #379 | surface-aware-decoder | Architecture (aux surface MLP head) | -3% to -7% | WIP |

## Lessons from round 1 so far

- **The 30-min cap is binding** at ~14 epochs for the published Transolver. Compute, not memory, is the bottleneck (peak 42-82 GB / 96 across all runs).
- **Round 1 is a 14-epoch ranking exercise** — the cosine schedule's tail is unreached. Round-1 winners may need re-validation under longer training.
- **EMA late-epoch smoothing is high-value** in this regime (PR #308 hit best on every one of 13 epochs). Decay=0.999 has a slow warmup; decay=0.995 should pay off earlier in the budget.
- **Aggressive grad clipping is implicit lr dampening** — useful side-effect, but worth attributing cleanly.
- **Architectural-scale changes need throughput-friendliness baked in** — wider (#300), more-slices (#309), and deeper (#304) all lost epochs to per-step compute. We've now closed three PRs on the same axis pattern; the lesson is firmly priced in.
- **Independent diagnoses converged on the same scoring NaN bug** (4 students), now fixed (#358).

## Potential next research directions

- **Compound the wins so far** in a single PR (after #381 attribution returns): surf_weight=25 + EMA(0.995) + clip(10) + bf16 (when #372 lands) + larger batch (when #382 lands). Independent changes should stack.
- **Further throughput levers after bf16 + larger batch:** `torch.compile(mode="reduce-overhead")`, channels-last memory format, mixed batch sizes via gradient accumulation across mesh-size buckets.
- **Heavy-tail pressure handling beyond per-sample y-norm (#378):** arcsinh / log-pressure target reparametrization, focal-style weighting on extreme |p|, per-Re-bin loss balancing.
- **Conservative widening that fits the budget**: revisit n_hidden=144 / slice_num=80 once throughput levers land.
- **Domain-conditional FiLM** modulation across single / raceCar-tandem / cruise-tandem regimes (orders of magnitude difference in y-std).
- **Better val-track-aware checkpoint selection**: weight `val_geom_camber_*` higher since they're the harder splits and currently lift `val_avg` the most.
- **Test-time augmentation**: average predictions on x↔−x flipped meshes (with corresponding AoA / stagger flips).
