# SENPAI Research State

- **Date:** 2026-04-28 04:10 UTC
- **Advisor branch:** `icml-appendix-willow-pai2d-r4`
- **Most recent human-team direction:** none received yet on this advisor branch
- **Current best:** PR #344 (edward H2) merged. `val_avg/mae_surf_p=120.97`, `test_avg/mae_surf_p=109.92`. See BASELINE.md for full details and recommended config (`--epochs 25 --lr 7e-4`).

## Active research focus

**Round 0 in flight; round-0 follow-ups starting.** Eight non-overlapping hypotheses originally fielded; H2 (warmup + per-step cosine + NaN fix) merged as the first winner; H7 (z-mirror augmentation) closed as a strict regression with strong diagnostic write-up. Edward and frieren reassigned to H11 and H10 from the reserve list — both should compound with the merged baseline.

## Current themes

1. **Scale-aware losses** — per-sample y-std varies ~40x; tested actively via H1 (alphonse).
2. **Training-recipe debt** — partially closed by H2 merge. Open follow-up: shorter `--epochs 14` to test cosine-actually-reaching-zero in the run-time budget. Deferred for a later round.
3. **Throughput as a lever** — H6 (askeladd) tests bf16/compile/larger batch; once landed, every other hypothesis benefits from more epochs in the budget.
4. **Geometry-OOD generalization** — H4 (fern, surface-only norm + distance feature), H5 (nezuko, Fourier features), H8 (thorfinn, slice_num) all hit this from different angles.
5. **Re-conditioning** — H11 (edward, FiLM on log(Re)) directly attacks the Re-OOD gap.
6. **Loss/curriculum tuning** — H10 (frieren, surf_weight ramp) on top of merged baseline.
7. **Robustness** — defensive `nan_to_num` shipped in the H2 merge; future runs no longer get NaN-poisoned by `test_geom_camber_cruise` sample 20's `-inf` GT.

## Currently in flight

| PR | Student | Hypothesis | Bucket | Predicted Δ | Status |
|----|---------|------------|--------|-------------|--------|
| #342 | alphonse | H1: per-sample y-std loss normalization | Loss reformulation | -8% to -18% | wip (sent back for rebase + sw sweep on merged schedule; first round Run B at sw=5 gave clean −7.9% val on apples-to-apples pre-merge baseline, cross-split signature matched prediction precisely) |
| #343 | askeladd | H6: bf16 + torch.compile + larger batch | Throughput | -3% to -9% | wip |
| #347 | nezuko | H5: random Fourier features on (x, z) | Position | -2% to -8% | wip (rebase + re-run on merged schedule; first round σ-sweep ID'd σ=4 as winner) |
| #468 | fern | H9: surface-arc pressure-gradient penalty | Physics-aware | -2% to -5% | wip |
| #348 | tanjiro | H3: Smooth L1 (Huber) on surface pressure | Loss reformulation | -2% to -6% | wip |
| #404 | edward | H11: Re-conditional FiLM modulation | Feature engineering | -3% to -7% | wip (sent back for FiLM-vs-wd disentanglement + seed repeat; first round Run C gave +0.7% test improvement but within noise) |
| #442 | thorfinn | H12: EMA of model weights for evaluation | Optimization | -1% to -4% | wip (sent back for decay=0.99 + every-other-epoch EMA eval; first round confirmed EMA mechanism within-run but absolute test didn't beat baseline) |
| #490 | frieren | H13: stochastic depth (DropPath) on Transolver blocks | Architectural regularization | -1% to -4% | wip |

## Resolved this round

| PR | Student | Hypothesis | Outcome | val_avg/mae_surf_p |
|----|---------|------------|---------|--------------------|
| #344 | edward | H2: warmup + corrected cosine | **merged** | 120.97 (Run C, –3.4% vs Run A) |
| #346 | frieren | H7: z-mirror augmentation | closed (strict regression) | +231% at p=1.0 |
| #349 | thorfinn | H8: slice_num scaling matrix | closed (regression vs baseline) | 148.65 (+23% vs baseline) |
| #345 | fern | H4: surface-only norm + distance feature | closed (cruise OOD structural regression) | 129.13 (+6.7% vs baseline) |
| #406 | frieren | H10: surf_weight ramp curriculum | closed (within-experiment +4.3% vs A but +1.6% regression vs baseline; effect below seed-variance floor) | 122.90 (+1.6% vs baseline) |

## Held in reserve / promising follow-ups

- **Edward's `--epochs 14` lr-frontier follow-up** — finish testing the schedule frontier on the merged code.
- **Thorfinn's slice_num=96 follow-up** — never tested below the slice 128 winner; the 128→256 curve goes sharply up which hints 64 may already be slightly over-partitioned.
- **Frieren's TTA (test-time augmentation) study** — evaluate predictions on (x, mirror(x)) at test time. Tests whether the symmetry exists in the trained model, divorced from training-time corruption.
- **Frieren's domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip; small but clean physics.
- **Fern's C2-Lite ablation + multi-scale distance feature** — milder loss-rebalancing variants that *might* dodge the cruise structural regression.
- **Per-domain `surf_weight`** — if frieren's H10 (surf_weight ramp) lands, this is the natural next step.
- **Compounding the round-0 winners** — once 2–3 separate ideas merge, run a combined-best PR to ensure their gains are additive.

## Open methodological note

We have now seen single-run noise of **~6% peak-to-peak** across three PRs:
- #404 (FiLM): Run A control 7% *worse* than baseline on equivalent code
- #406 (surf_weight ramp): Run A control 6% *worse* than baseline on equivalent code
- #442 (EMA): Run A control 2.6% *better* than baseline on equivalent code

The variance is bidirectional and well-calibrated. **Predicted effect sizes <5% are below this noise floor by design.** Future small-effect hypotheses should plan multi-seed confirmation up front, OR ride on top of larger-effect winners (e.g. H1) where the headline number is already moving by 5-10%. This is also a strong argument for landing H6 (askeladd, throughput) so we can run more epochs per training and reduce training-time variance.

## Potential next research directions (post round 0)

- **Once throughput lands (H6),** revisit larger architectures with H8 follow-ups and a wider `n_hidden`.
- **If H1 (per-sample y-std) and H3 (Huber) both land,** combine them in one PR — they target different aspects of the heavy-tailed surface pressure regime.
- **If geometry-OOD splits remain stubborn** after rounds 0/1, escalate to graph/edge-aware mesh modules or coordinate-network heads.
- **If Re-OOD splits remain stubborn,** explore Re-conditional separate models or hierarchical heads beyond what FiLM provides.
- **Best-checkpoint reproducibility** — at some point, lock in the best-known recipe and run a multi-seed reproducer to estimate variance for paper-ready numbers.
