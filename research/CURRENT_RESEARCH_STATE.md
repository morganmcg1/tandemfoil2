# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-28 06:20
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus

**Current best:** PR #484 (thorfinn, surface-conditional FiLM + Huber β=0.5 + Fourier + EMA + clip + bf16 + compile + cudagraph_skip), merged commit dc9e0e5. `val_avg/mae_surf_p = 57.37` (EMA-evaluated), `test_avg/mae_surf_p = 48.96`. **-0.23% over PR #467 on val, -3.06% on test_avg.** Paired comparison shows -3.05% val, -2.92% test, **all 8 splits gain**. Cumulative **-54.7% from PR #287's first baseline**, **-57.6% from the published-baseline-equivalent**.

**Compounding evidence accumulating across 8 stacked levers**: FiLM (PR #484) + Huber β=0.5 (PR #467) + Fourier (PR #368) + Huber base (PR #289) + EMA + clip (PR #381) + compile (PR #401) + bf16 (PR #372) — eight orthogonal levers all positive without observable interference. FiLM mechanism: shared trunk → tiny domain-conditional decoder (γ_surf, β_surf) vs (γ_vol, β_vol) at last LayerNorm output before mlp2 → no parallel pathway, no trunk interference. **Volume MAE also improves** on all 4 val splits, replicating the trunk/decoder-separation story.

**Open infrastructure note**: Config defaults still `huber_beta=1.0` and `film=False`. Reproducing the merged baseline requires explicit `--huber_beta 0.5 --film`. PR #539 (askeladd's β finer sweep) flips `huber_beta` default to 0.5; thorfinn's next PR (FiLM at all blocks) may flip `film` default to True.

**Open infrastructure issue:** 2 of 4 launches at the rebased compile + EMA + clip + bf16 stack crash with CUDAGraph private-pool blowup at variable mesh sizes. Alphonse's depth experiment (#435) hit the same OOM at depth=8, requiring `mode="default"` workaround (~10-15% throughput cost). PR #466 (alphonse) bundles the fix: `cudagraph_skip_dynamic_graphs=True` flag + cosine T_max retune to actually-reachable epoch count.

**Round 1 status:** 8 PRs assigned across 4 axes (loss formulation, loss weighting, architecture, optimization). PR #372 (bf16) merged as infrastructure → **the 14-epoch ranking exercise is now a 19-epoch ranking exercise** in the same wall-clock budget (1.36× speedup). torch.compile (PR #401) targets another 1.2-1.5× on top, potentially pushing to 25-30 epochs.

**Throughput is a first-class research axis** going forward — alphonse (#372: bf16 autocast) and frieren (#382: larger batch) are testing the two main throughput levers in parallel. Both should compound with the existing wins from #287 and #308.

**Resolved infrastructure issue:** PR #358 (edward, commit 010235e) fixed `data/scoring.py` `inf * 0 = NaN` propagation via `torch.where`. New `data/test_scoring.py` has 4 surgical regression tests. PRs branched before that merge may still report NaN on `test_geom_camber_cruise/mae_surf_p`; we evaluate them on val_avg/mae_surf_p instead.

## Round 1 PR roster

| Student | PR | Slug | Axis | Predicted Δ | Status |
|---|---|---|---|---|---|
| alphonse | #287 | surf-weight-up | Loss weighting (10→25) | -3% to -7% | **MERGED** e4a0c18 → val_avg=126.67 |
| alphonse | #372 | bf16-autocast | Throughput (bf16 autocast) | -10% to -20% | **MERGED** 91d8a4e (infra) → 1.36× speedup, 19/50 epochs |
| alphonse | #401 | compile-bf16-emaclip | Throughput (torch.compile reduce-overhead, dynamic) | -5% to -15% | **MERGED** 5f2edca → val_avg=**66.89** (NEW BEST, -37.1%) |
| alphonse | #435 | deeper8-droppath01-compile | Architecture (n_layers 5→8 + DropPath 0.1) | -5% to -10% | **CLOSED** — +30% (cosine T_max=50 mismatched with 22 reached epochs) |
| alphonse | #466 | tmax32-cudagraph-skip | Infra (cosine_epochs flag + cudagraph_skip robustness) | -1% to -5% | **MERGED** e0a902b (revision) — cosine_epochs flag plumbed at default 50 (no behavior change); cudagraph_skip auto-deduped with #467 |
| alphonse | #549 | warmup-cosine-sweep | Schedule (linear warmup ∈ {2, 3, 5} + cosine to existing endpoint) | -1% to -3% | WIP |
| askeladd | #289 | huber-loss | Loss formulation (MSE→SmoothL1) | -5% to -10% | **MERGED** 906a2c1 → val_avg=**63.33** (NEW BEST, -5.31%) |
| askeladd | #467 | huber-beta-sweep | Loss formulation (β ∈ {0.5, 1.0, 2.0} sweep) | β=0.5 predicted -1% to -4% | **MERGED** eb5168f → val_avg=**57.50** (NEW BEST, -8.65% vs #368) |
| askeladd | #539 | huber-beta-finer | Loss formulation (β ∈ {0.3, 0.5, 0.7} + flip Config default to 0.5) | -1% to -3% | WIP |
| edward   | #300 | wider-model | Width (192/96) | -5% to -10% | **CLOSED** — under-trained 9/50 |
| edward   | #358 | fix-scoring-nan-mask | Maintenance | n/a | **MERGED** 010235e |
| edward   | #368 | fourier-pos-encoding | Input (8-freq Fourier on (x,z)) | -3% to -8% | **MERGED** 430cd62 → val_avg=**62.94** (NEW BEST, -0.62% val / -1.30% test) |
| edward   | #512 | fourier-nfreqs-sweep | Input (n_freqs ∈ {4, 6, 12} sweep on top of #368) | -1% to -3% | WIP |
| fern     | #304 | deeper-model-droppath | Depth (5→8 + DropPath 0.1) | -3% to -8% | **CLOSED** — 210 s/epoch, 9/50 epochs, equal-epoch worse |
| fern     | #388 | arcsinh-pressure | Heavy-tail (arcsinh on p target) | -5% to -15% | **CLOSED** — +15.1% regression; sinh decode amplifies tail errors |
| fern     | #422 | pchannel-p-w05 | Loss weighting (per-channel w_p=0.5 to free velocity gradient) | -2% to -7% | **CLOSED** — same-epoch -1.85% (within noise); velocity -7 to -10% (mechanism supported) |
| fern     | #453 | pchannel-p-ramp05-10 | Loss weighting (linear ramp w_p 0.5→1.0 over training) | -2% to -7% | **SENT BACK** — strong mechanism (-3.49% same-pod, velocity 4× pressure) but conflicts with #289; rebase + re-run with Huber |
| frieren  | #307 | warmup-cosine-1e3 | Optim (warmup + lr 1e-3) | -2% to -6% | **CLOSED** — 134.58, 26% worse than #308 |
| frieren  | #382 | batch8-lr7e-4 | Throughput (larger batch + sqrt-lr) | -5% to -15% | **CLOSED** — GPU compute-saturated at bs=4; +52% EMA artifact |
| frieren  | #431 | wider160-bf16 | Architecture (n_hidden 128→160 under bf16) | -2% to -7% | **CLOSED** — within-noise +2.5% (same-epoch capacity supported but +14% per-epoch tax) |
| frieren  | #477 | wider144-compile | Architecture (n_hidden 128→144 under post-#289) | -2% to -7% | **CLOSED** — +6.4% val (second paired confirmation that capacity loses to budget) |
| frieren  | #528 | cosine-eta-min-sweep | Schedule (cosine eta_min ∈ {1e-5, 1e-4, 5e-4}) | -1% to -4% | WIP |
| nezuko   | #308 | ema-grad-clip | Optim (EMA 0.999 + clip 1.0) | -3% to -8% | **MERGED** 5bdb284 → val_avg=106.40 |
| nezuko   | #381 | ema995-gradclip10 | Ablation (EMA 0.995 + clip 10) | -3% to -10% | **MERGED** a620ba1 → val_avg=**98.85** (NEW BEST, -7.1%) |
| nezuko   | #421 | ema995-noclip | Ablation (EMA 0.995, NO clip — clean isolation) | ±5% | **CLOSED** — val_avg=109.99 (+11.3% vs #381), clip is load-bearing as dampener |
| nezuko   | #449 | surf25-emaclip-compile | Compound (--surf_weight 25 on top of #401 compile+EMA+clip) | -2% to -7% | WIP |
| tanjiro  | #309 | more-slices | Architecture (128/8) | -3% to -7% | **CLOSED** — 2× slower, not better |
| tanjiro  | #378 | per-sample-relmse | Heavy-tail (per-sample y-var) | -3% to -7% | WIP |
| thorfinn | #310 | per-channel-surf-weights | Loss weighting (3× p) | -3% to -8% | **CLOSED** — +13% regression |
| thorfinn | #379 | surface-aware-decoder | Architecture (substitutive surface MLP head) | -3% to -7% | **CLOSED** — within noise vs own baseline-ref; substitutive design wastes vol head signal |
| thorfinn | #436 | additive-surf-head | Architecture (additive surface head) | -2% to -7% | **CLOSED** — +3.47% (trunk interference is deeper bottleneck) |
| thorfinn | #484 | surface-film | Architecture (surface-conditional FiLM in last block) | -1% to -4% | **MERGED** dc9e0e5 → val_avg=**57.37** (NEW BEST, -0.23% val / -3.06% test); paired -3.05%/-2.92% all 8 splits gain |
| thorfinn | #594 | film-all-blocks | Architecture (FiLM at all 5 block boundaries, mid-network specialization) | -1% to -3% | WIP |

## Lessons from round 1 so far

- **The 30-min cap is binding** at ~14 epochs for the published Transolver. Compute, not memory, is the bottleneck (peak 42-82 GB / 96 across all runs).
- **Round 1 is a 14-epoch ranking exercise** — the cosine schedule's tail is unreached. Round-1 winners may need re-validation under longer training.
- **EMA late-epoch smoothing is high-value** in this regime. PR #381 confirmed: decay=0.995 crosses online at epoch 2 vs ~10 at decay=0.999 (PR #308); 11 useful EMA epochs in the budget instead of 3.
- **Aggressive grad clipping is implicit lr dampening** — at max_norm=1 (#308) 100% of batches fire; at max_norm=10 (#381) still 87-100% fire because gn_mean is 44-107. We don't yet have a clean EMA-only number; PR #421 (nezuko, no clip) is the isolation test.
- **Heavy-tail target reparametrization (arcsinh)** — fundamentally wrong direction for an equal-weight physical-space MAE metric. Decode through nonlinear functions amplifies tail errors disproportionately. Per-channel scalar weights are the right lever.
- **Architectural-scale changes need throughput-friendliness baked in** — wider (#300), more-slices (#309), and deeper (#304) all lost epochs to per-step compute. With bf16 (#372) AND torch.compile (#401) now in the merged baseline, per-epoch dropped 141s→55s. **Architectural-scale PRs that previously closed for budget reasons should be revisited** under the new regime — alphonse #435 (deeper-8) and frieren #431 (wider-160) are the two retests in flight.
- **Memory headroom does NOT translate to throughput on this hardware.** Frieren #382 confirmed: GPU is compute-saturated at bs=4. Doubling batch ≈ doubles per-step time → net flat. Memory headroom should go to capacity (n_hidden, n_layers) or be cashed via bf16/compile (which actually drop per-step compute), not via bigger batches.
- **The compile-driven epoch-budget recovery is the dominant mechanism behind the -37% jump in #401.** With cosine-tail finally reachable, EMA gets enough useful epochs to do its job. Round-1 ranking is now a 33-epoch exercise, not a 13-epoch one — past results may need re-evaluation.
- **Clip is load-bearing as a per-batch dampener** (PR #421 attribution). At max_norm=10 and our AdamW/lr settings, the clip isn't catching runaway gradients (gn_max stays in 344-767 band with or without clipping); it's per-batch dampening that materially helps generalization, especially on the smallest-magnitude split (cruise camber, +22% regression without clip).
- **Schedule mismatch hypothesis was wrong.** PR #466's clean A/B (held cudagraph flag constant) showed cosine_epochs=32 *regresses* val_avg by +6.7% — the model is in the bulk-learning regime at epoch 33 (lr~1.3e-4, dropping ~0.9 mae/epoch), NOT in a fine-tuning phase being cut off. T_max=50 default is correct for n_layers=5. Future levers: **eta_min > 0** (lift the LR floor instead of truncating the schedule).
- **Compile flakiness needs addressing.** 2 of 4 launches at the rebased stack crashed with CUDAGraph private-pool blowup. The fix (`cudagraph_skip_dynamic_graphs=True`) drops ~10-15% throughput but eliminates the failure mode — net positive given how much architectural-side work remains.
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
