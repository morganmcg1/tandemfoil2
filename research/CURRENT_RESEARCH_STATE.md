# SENPAI Research State

- **Date:** 2026-04-28 09:30 UTC
- **Advisor branch:** `icml-appendix-willow-pai2d-r4`
- **Most recent human-team direction:** none received yet on this advisor branch
- **Current best:** PR #343 (askeladd H6 bf16+compile × FiLM × EMA) merged. `val_avg/mae_surf_p=80.91`, `test_avg/mae_surf_p=72.73`. See BASELINE.md for full details and recommended config (`--batch_size 4 --amp_dtype bf16 --compile True --film_re True --use_ema True --ema_decay 0.99 --ema_eval_every 2 --epochs 37 --lr 7e-4 --weight_decay 5e-4 --seed 123`).

## Active research focus

**Round 0 ongoing — four winners merged.** PR #344 (warmup+cosine+NaN fix), PR #404 (FiLM-on-Re + wd=5e-4), PR #442 (EMA decay=0.99 × FiLM), and now **PR #343 (bf16+torch.compile × FiLM × EMA)** — the largest single-PR effect of the round at −25.7% / −26.1% on top of the prior baseline. Cumulative round-0 progress vs original baseline: **−33% on val, −33% on test.**

The bf16+compile mechanism is the dominant compounding lever. With 2.4× throughput we now train 34 epochs (vs 14 before) in the same 30-min budget; cosine reaches lr=0 at epoch 36 for the first time on this branch. Every in-flight PR should rebase to inherit this throughput uplift.

**Notable side finding from PR #343:** EMA's marginal value drops to ~0 at full-convergence regime (per-epoch EMA-vs-raw gap narrowed from ~16 pts at epoch 1 to 0.13 pts at epoch 33; raw was selected as active checkpoint). EMA still useful as defensive measure for noisy training; kept on by default but don't expect it to add much when underlying training is already converged.

## Current themes

1. **Scale-aware losses** — H1 (alphonse, on rebase) and H16 (nezuko, on rebase) attack heavy-tailed pressure from loss-space and target-space angles respectively.
2. **Throughput as a lever** — H6 (askeladd, **MERGED**) is the dominant compounding lever. Now active by default for every future PR.
3. **Geometry-OOD generalization** — H9 (fern, surface-arc gradient penalty) targets geom-OOD specifically.
4. **Re-conditioning** — H11 merged with 1-D log(Re) FiLM. Richer conditioning (H14 cond5) closed as a wash at single-seed precision.
5. **Loss alignment with metric** — H3 (tanjiro, Huber on surface) targets the MSE-vs-MAE mismatch on heavy-tailed pressure.
6. **Optimization** — H17 (edward, layer-wise lr decay) and H19 (askeladd, **NEW**, Lion optimizer) test optimization-bucket hypotheses orthogonal to everything else.
7. **Architectural scaling** — H18 (thorfinn, wider Transolver) tests whether the now-stable training unlocks more capacity.
8. **Inference-time** — H15 (frieren, TTA) tests whether the trained model has learned z-symmetry on cruise samples (PR #561 entered review).
9. **Robustness** — defensive `nan_to_num` (PR #344), `--seed` flag (PR #404), bf16 fp32-fallback (PR #343), `_raw_module()` for compile-aware state_dict (PR #343).

## Currently in flight

| PR | Student | Hypothesis | Bucket | Predicted Δ | Status |
|----|---------|------------|--------|-------------|--------|
| #342 | alphonse | H1: per-sample y-std loss normalization | Loss reformulation | -8% to -18% | wip (sent back for rebase + sw sweep on merged schedule; first round Run B at sw=5 gave clean −7.9% val on apples-to-apples pre-merge baseline, cross-split signature matched prediction precisely) |
| #348 | tanjiro | H3: Smooth L1 (Huber) on surface pressure | Loss reformulation | -2% to -6% | wip |
| #468 | fern | H9: surface-arc pressure-gradient penalty | Physics-aware | -2% to -5% | wip |
| #654 | frieren | H20: random Re-jitter augmentation on log(Re) input | Regularization | -2% to -5% | wip |
| #576 | nezuko | H16: arcsinh-compressed pressure target | Target transform | -2% to -6% | wip (sent back round 2 — needs rebase onto #343 for full compound test; round 2 Run D at val_ema=86.12 / test=75.87 is -21.1% / -23.0% on post-#442 path, EMA × arcsinh confirmed clean compound but +6.4% above current post-#343 merged baseline) |
| #662 | edward | H21: exclude 1-D parameters from weight decay | Optimization | -0.5% to -2% | wip |
| #693 | thorfinn | H22: torch.compile mode=reduce-overhead with fixed N_max padding | Throughput | -2% to -5% | wip |
| #650 | askeladd | H19: Lion optimizer | Optimization | -1% to -5% | wip |

## Resolved this round

| PR | Student | Hypothesis | Outcome | val_avg/mae_surf_p |
|----|---------|------------|---------|--------------------|
| #344 | edward | H2: warmup + corrected cosine | **merged** | 120.97 (Run C, –3.4% vs Run A) |
| #346 | frieren | H7: z-mirror augmentation | closed (strict regression) | +231% at p=1.0 |
| #349 | thorfinn | H8: slice_num scaling matrix | closed (regression vs baseline) | 148.65 (+23% vs baseline) |
| #345 | fern | H4: surface-only norm + distance feature | closed (cruise OOD structural regression) | 129.13 (+6.7% vs baseline) |
| #406 | frieren | H10: surf_weight ramp curriculum | closed (effect below seed-variance floor) | 122.90 (+1.6% vs baseline) |
| #490 | frieren | H13: stochastic depth (DropPath) | closed (B-vs-A signature matched prediction strongly but absolute effect below noise floor) | 120.57 (+1.0% vs baseline) |
| #347 | nezuko | H5: Fourier features (× FiLM in round 3) | closed (Fourier × FiLM antagonistic) | 129.49 (+8.5% vs PR #404 baseline) |
| #523 | edward | H14: 5-D FiLM conditioner | closed (B vs C seed-pair spread > 8%; Mean(B,C) − A ≈ 0) | mean 120.76 (+1.2% vs PR #404 baseline) |
| #404 | edward | H11: Re-conditional FiLM modulation | **merged** (after disentanglement) | **119.36** (Run E, −1.3% vs PR #344 baseline) |
| #442 | thorfinn | H12: EMA decay=0.99 × FiLM | **merged** (after compound test) | **val_ema=109.19** (Run F, −8.5% vs PR #404 baseline) |
| #343 | askeladd | H6: bf16+compile × FiLM × EMA | **merged** (after compound test) | **val=80.91** (Run G, **−25.7%** vs PR #442 baseline; largest single-PR effect of round 0) |
| #561 | frieren | H15: test-time z-mirror augmentation (TTA) | closed (TTA decisively rejected at +137% / +154% regression; model becomes MORE z-asymmetric over training; cross-split ordering matches BC argument but every split regresses) | 283.11 (+137% vs PR #404 baseline) |
| #602 | edward | H17: layer-wise lr decay | closed (Mean(B,C) − A = +0.24% val / +1.25% test, in [−2%, +2%] band; B-C seed-pair spread tight at 0.77%/1.21% confirming genuine mild regression not noise) | 119.65 (mean B,C; +0.24% vs PR #404 baseline) |
| #611 | thorfinn | H18: wider Transolver (n_hidden=192, n_head=6) | closed (Mean(B,C) − A = +21.2% val_ema, far past +2% threshold; under-convergence not capacity regression — wider runs hit timeout at epoch 9 vs baseline's 13) | mean 132.30 (+21.2% vs PR #442 baseline) |

## Held in reserve / promising follow-ups

- **3-seed nail-down of the merged baseline (Run G config)** — would give a real error bar on the val=80.91 / test=72.73 floor. Useful for paper-confidence numbers.
- **Re-tune EMA for longer-budget regime.** With raw converging well under bf16+compile, decay=0.99 (~100-step horizon) tracks too tightly. Larger decay (e.g. 0.998) might give EMA a lead late. Lower priority since raw already wins at full budget.
- **`torch.compile(mode="reduce-overhead")` with padded N_max.** Could unlock another 20-30% throughput via CUDAGraphs — would require wrapping pad_collate to pad to fixed N_max.
- **Re-tuned bs=8 with proper LR.** Still untested.
- **Edward's `--epochs 14` lr-frontier follow-up** — outdated given longer training now possible.
- **Thorfinn's slice_num=96 follow-up** — original H8 was on pre-merge; worth retesting on merged baseline if H18 wider underperforms.
- **Frieren's domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip; small but clean physics.
- **Fern's C2-Lite ablation + multi-scale distance feature** — milder loss-rebalancing variants that *might* dodge the cruise structural regression.
- **Per-domain `surf_weight`** — closely related to H10's failure mode; revisit if H10 mechanism can be salvaged.
- **FiLM hidden=32** — halve the FiLM head's 83K params, tightening the +12.6% params objection.
- **Concat-Re instead of FiLM** — cheaper alternative.
- **Stack the round-0 winners + an in-flight winner** — once H1 (alphonse) or H16 (nezuko) lands, run a combined-best PR to verify the round-0 winners are not at the upper bound of compounding.

## Open methodological note

Single-run noise floor on this branch is **~6–8% peak-to-peak**, well-calibrated across multiple PRs (#404, #406, #442, #490, #523). PR #523's seed pair was the cleanest demonstration: same code, same config, only seed differs, val swings from -2.93% to +5.27%. **Predicted effect sizes <5% are below this floor by design** and require multi-seed confirmation up front.

The `--seed 123` reproducibility protocol has been demonstrated four times now: PR #442 Run F, PR #523 Run A, PR #576 Run A, PR #343 Run G all reproduce their respective baselines to ~4 decimals. This is the strongest evidence we have that the seed-controlled comparison protocol works at the program level.

## Potential next research directions (post round 0)

- **Once H1 (alphonse) or H16 (nezuko) lands,** combine with the merged baseline for a "stack the round-0 winners" PR to verify additivity.
- **If H18 (thorfinn, wider) lands cleanly,** revisit deeper architectures (n_layers=6 or 7).
- **If geometry-OOD splits (camber_rc, cruise) remain stubborn,** escalate to graph/edge-aware mesh modules or coordinate-network heads.
- **If Re-OOD splits remain stubborn,** explore Re-conditional separate models or hierarchical heads.
- **Best-checkpoint multi-seed reproducibility** — at some point, lock in the best-known recipe and run a 5-seed reproducer to estimate variance for paper-ready numbers.
