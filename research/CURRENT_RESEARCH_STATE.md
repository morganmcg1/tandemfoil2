# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-28 02:45
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus

**Current best:** PR #401 (alphonse, torch.compile + bf16 + EMA + clip), merged commit 5f2edca. `val_avg/mae_surf_p = 66.89` (EMA-evaluated), `test_avg/mae_surf_p = 57.86`. **-37.1% over PR #308**, **-32.3% over PR #381**. Cumulative **-47% from the published-baseline-equivalent**.

**Mechanism is throughput-driven.** Compile dropped per-epoch from 141 s to 54.6 s (2.58×), pushing the budget from 13 epochs to 33. The cosine schedule's tail finally reachable + EMA gets 33 useful epochs instead of ~3. Per-split gains uniform 30-43% across all 4 val and 4 test splits — this is the cosine-tail + EMA combo finally getting room to operate, not a bias toward any particular split.

**Round 1 character changed:** "14-epoch ranking exercise" → "33-epoch ranking exercise". Architectural-scale PRs that previously closed for blowing per-epoch budget (wider #300, more-slices #309, deeper #304) should be revisited under the new regime. **PR #435 (alphonse, deeper-8 + DropPath)** is the first such revisit.

**Round 1 status:** 8 PRs assigned across 4 axes (loss formulation, loss weighting, architecture, optimization). PR #372 (bf16) merged as infrastructure → **the 14-epoch ranking exercise is now a 19-epoch ranking exercise** in the same wall-clock budget (1.36× speedup). torch.compile (PR #401) targets another 1.2-1.5× on top, potentially pushing to 25-30 epochs.

**Throughput is a first-class research axis** going forward — alphonse (#372: bf16 autocast) and frieren (#382: larger batch) are testing the two main throughput levers in parallel. Both should compound with the existing wins from #287 and #308.

**Resolved infrastructure issue:** PR #358 (edward, commit 010235e) fixed `data/scoring.py` `inf * 0 = NaN` propagation via `torch.where`. New `data/test_scoring.py` has 4 surgical regression tests. PRs branched before that merge may still report NaN on `test_geom_camber_cruise/mae_surf_p`; we evaluate them on val_avg/mae_surf_p instead.

## Round 1 PR roster

| Student | PR | Slug | Axis | Predicted Δ | Status |
|---|---|---|---|---|---|
| alphonse | #287 | surf-weight-up | Loss weighting (10→25) | -3% to -7% | **MERGED** e4a0c18 → val_avg=126.67 |
| alphonse | #372 | bf16-autocast | Throughput (bf16 autocast) | -10% to -20% | **MERGED** 91d8a4e (infra) → 1.36× speedup, 19/50 epochs |
| alphonse | #401 | compile-bf16-emaclip | Throughput (torch.compile reduce-overhead, dynamic) | -5% to -15% | **MERGED** 5f2edca → val_avg=**66.89** (NEW BEST, -37.1%) |
| alphonse | #435 | deeper8-droppath01-compile | Architecture (n_layers 5→8 + DropPath 0.1, revisit fern's #304 under compile) | -5% to -10% | WIP |
| askeladd | #289 | huber-loss | Loss formulation (MSE→SmoothL1) | -5% to -10% | **SENT BACK** — clean -9.9% on loss axis but pre-#308; rebase + re-run with EMA |
| edward   | #300 | wider-model | Width (192/96) | -5% to -10% | **CLOSED** — under-trained 9/50 |
| edward   | #358 | fix-scoring-nan-mask | Maintenance | n/a | **MERGED** 010235e |
| edward   | #368 | fourier-pos-encoding | Input (8-freq Fourier on (x,z)) | -3% to -8% | **SENT BACK AGAIN** — rebase to post-#401 + re-run; equal-epoch shows -11% compounding signal but lost 3 epochs to GPU contention |
| fern     | #304 | deeper-model-droppath | Depth (5→8 + DropPath 0.1) | -3% to -8% | **CLOSED** — 210 s/epoch, 9/50 epochs, equal-epoch worse |
| fern     | #388 | arcsinh-pressure | Heavy-tail (arcsinh on p target) | -5% to -15% | **CLOSED** — +15.1% regression; sinh decode amplifies tail errors |
| fern     | #422 | pchannel-p-w05 | Loss weighting (per-channel w_p=0.5 to free velocity gradient) | -2% to -7% | WIP |
| frieren  | #307 | warmup-cosine-1e3 | Optim (warmup + lr 1e-3) | -2% to -6% | **CLOSED** — 134.58, 26% worse than #308 |
| frieren  | #382 | batch8-lr7e-4 | Throughput (larger batch + sqrt-lr) | -5% to -15% | **CLOSED** — GPU compute-saturated at bs=4; +52% EMA artifact |
| frieren  | #431 | wider160-bf16 | Architecture (n_hidden 128→160 under bf16) | -2% to -7% | WIP |
| nezuko   | #308 | ema-grad-clip | Optim (EMA 0.999 + clip 1.0) | -3% to -8% | **MERGED** 5bdb284 → val_avg=106.40 |
| nezuko   | #381 | ema995-gradclip10 | Ablation (EMA 0.995 + clip 10) | -3% to -10% | **MERGED** a620ba1 → val_avg=**98.85** (NEW BEST, -7.1%) |
| nezuko   | #421 | ema995-noclip | Ablation (EMA 0.995, NO clip — clean isolation) | ±5% | **CLOSED** — val_avg=109.99 (+11.3% vs #381), clip is load-bearing as dampener |
| nezuko   | #449 | surf25-emaclip-compile | Compound (--surf_weight 25 on top of #401 compile+EMA+clip) | -2% to -7% | WIP |
| tanjiro  | #309 | more-slices | Architecture (128/8) | -3% to -7% | **CLOSED** — 2× slower, not better |
| tanjiro  | #378 | per-sample-relmse | Heavy-tail (per-sample y-var) | -3% to -7% | WIP |
| thorfinn | #310 | per-channel-surf-weights | Loss weighting (3× p) | -3% to -8% | **CLOSED** — +13% regression |
| thorfinn | #379 | surface-aware-decoder | Architecture (substitutive surface MLP head) | -3% to -7% | **CLOSED** — within noise vs own baseline-ref; substitutive design wastes vol head signal |
| thorfinn | #436 | additive-surf-head | Architecture (additive surface head: preds_vol + is_surface * preds_surf) | -2% to -7% | WIP |

## Lessons from round 1 so far

- **The 30-min cap is binding** at ~14 epochs for the published Transolver. Compute, not memory, is the bottleneck (peak 42-82 GB / 96 across all runs).
- **Round 1 is a 14-epoch ranking exercise** — the cosine schedule's tail is unreached. Round-1 winners may need re-validation under longer training.
- **EMA late-epoch smoothing is high-value** in this regime. PR #381 confirmed: decay=0.995 crosses online at epoch 2 vs ~10 at decay=0.999 (PR #308); 11 useful EMA epochs in the budget instead of 3.
- **Aggressive grad clipping is implicit lr dampening** — at max_norm=1 (#308) 100% of batches fire; at max_norm=10 (#381) still 87-100% fire because gn_mean is 44-107. We don't yet have a clean EMA-only number; PR #421 (nezuko, no clip) is the isolation test.
- **Heavy-tail target reparametrization (arcsinh)** — fundamentally wrong direction for an equal-weight physical-space MAE metric. Decode through nonlinear functions amplifies tail errors disproportionately. Per-channel scalar weights are the right lever.
- **Architectural-scale changes need throughput-friendliness baked in** — wider (#300), more-slices (#309), and deeper (#304) all lost epochs to per-step compute. With bf16 (#372) AND torch.compile (#401) now in the merged baseline, per-epoch dropped 141s→55s. **Architectural-scale PRs that previously closed for budget reasons should be revisited** under the new regime — alphonse #435 (deeper-8) and frieren #431 (wider-160) are the two retests in flight.
- **Memory headroom does NOT translate to throughput on this hardware.** Frieren #382 confirmed: GPU is compute-saturated at bs=4. Doubling batch ≈ doubles per-step time → net flat. Memory headroom should go to capacity (n_hidden, n_layers) or be cashed via bf16/compile (which actually drop per-step compute), not via bigger batches.
- **The compile-driven epoch-budget recovery is the dominant mechanism behind the -37% jump in #401.** With cosine-tail finally reachable, EMA gets enough useful epochs to do its job. Round-1 ranking is now a 33-epoch exercise, not a 13-epoch one — past results may need re-evaluation.
- **Clip is load-bearing as a per-batch dampener** (PR #421 attribution). At max_norm=10 and our AdamW/lr settings, the clip isn't catching runaway gradients (gn_max stays in 344-767 band with or without clipping); it's per-batch dampening that materially helps generalization, especially on the smallest-magnitude split (cruise camber, +22% regression without clip). Round-2 compound: surf_weight=25 + EMA(0.995) + clip(10.0), all established as load-bearing.
- **Independent diagnoses converged on the same scoring NaN bug** (6 students), now fixed (#358).
- **Variance floor: ~5pp** between two Huber seeds (askeladd #289). Round-1 winners by less than ~5% on val_avg are within run-to-run noise.

## Potential next research directions

- **Compound the wins so far** in a single PR (after #381 attribution returns): surf_weight=25 + EMA(0.995) + clip(10) + bf16 (when #372 lands) + larger batch (when #382 lands). Independent changes should stack.
- **Further throughput levers after bf16 + larger batch:** `torch.compile(mode="reduce-overhead")`, channels-last memory format, mixed batch sizes via gradient accumulation across mesh-size buckets.
- **Heavy-tail pressure handling beyond per-sample y-norm (#378):** arcsinh / log-pressure target reparametrization, focal-style weighting on extreme |p|, per-Re-bin loss balancing.
- **Conservative widening that fits the budget**: revisit n_hidden=144 / slice_num=80 once throughput levers land.
- **Domain-conditional FiLM** modulation across single / raceCar-tandem / cruise-tandem regimes (orders of magnitude difference in y-std).
- **Better val-track-aware checkpoint selection**: weight `val_geom_camber_*` higher since they're the harder splits and currently lift `val_avg` the most.
- **Test-time augmentation**: average predictions on x↔−x flipped meshes (with corresponding AoA / stagger flips).
