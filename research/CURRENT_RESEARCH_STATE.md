# SENPAI Research State

- **Date:** 2026-04-28 05:25 UTC
- **Advisor branch:** `icml-appendix-willow-pai2d-r4`
- **Most recent human-team direction:** none received yet on this advisor branch
- **Current best:** PR #404 (edward H11) merged. `val_avg/mae_surf_p=119.36`, `test_avg/mae_surf_p=107.54`. See BASELINE.md for full details and recommended config (`--film_re True --epochs 25 --lr 7e-4 --weight_decay 5e-4 --seed 123`).

## Active research focus

**Round 0 ongoing — two winners merged so far.** PR #344 (warmup + per-step cosine + NaN fix) merged first; PR #404 (Re-conditional FiLM + wd=5e-4) merged second after rigorous disentanglement that confirmed FiLM × wd interaction is the load-bearing mechanism. Six in-flight hypotheses are testing other directions, three of which have been sent back for rebases or focused re-runs because their first-round results need to be measured against the moving baseline.

## Current themes

1. **Scale-aware losses** — per-sample y-std varies ~40x; tested actively via H1 (alphonse, on rebase). Strong mechanistic signal in first round.
2. **Throughput as a lever** — H6 (askeladd) tests bf16/compile/larger batch; once landed, every other hypothesis benefits from more epochs in the budget. **Highest-leverage in-flight item.**
3. **Geometry-OOD generalization** — H5 (nezuko, Fourier features, on rebase), H9 (fern, surface-arc gradient penalty) hit this from different angles.
4. **Re-conditioning** — H11 merged with 1-D log(Re) FiLM. H14 (edward, **NEW**) extends to a 5-D conditioner `[log(Re), AoA1, AoA2, gap, stagger]` to test whether per-split signature sharpens.
5. **Loss alignment with metric** — H3 (tanjiro, Huber on surface) targets the MSE-vs-MAE mismatch on heavy-tailed pressure.
6. **Optimization regularization** — H12 (thorfinn, EMA, sent back for decay=0.99) and H13 (frieren, stochastic depth) compound with everything.
7. **Robustness** — defensive `nan_to_num` in `evaluate_split` shipped via #344; `--seed` CLI flag shipped via #404 for variance checks.

## Currently in flight

| PR | Student | Hypothesis | Bucket | Predicted Δ | Status |
|----|---------|------------|--------|-------------|--------|
| #342 | alphonse | H1: per-sample y-std loss normalization | Loss reformulation | -8% to -18% | wip (sent back for rebase + sw sweep on merged schedule; first round Run B at sw=5 gave clean −7.9% val on apples-to-apples pre-merge baseline, cross-split signature matched prediction precisely) |
| #343 | askeladd | H6: bf16 + torch.compile + larger batch | Throughput | -3% to -9% | wip |
| #347 | nezuko | H5: random Fourier features on (x, z) | Position | -2% to -8% | wip (sent back round 2 — needs rebase onto #404 + Run H to verify Fourier × FiLM compounds; round 2 σ=5 alone gave clean −3.14% vs #344 baseline with predicted geom-OOD signature) |
| #348 | tanjiro | H3: Smooth L1 (Huber) on surface pressure | Loss reformulation | -2% to -6% | wip |
| #442 | thorfinn | H12: EMA of model weights for evaluation | Optimization | -1% to -4% | wip (sent back for decay=0.99 + every-other-epoch EMA eval; first round confirmed EMA mechanism within-run but absolute test didn't beat baseline) |
| #468 | fern | H9: surface-arc pressure-gradient penalty | Physics-aware | -2% to -5% | wip |
| #561 | frieren | H15: test-time z-mirror augmentation (TTA) | Inference-time regularization | -1% to -3% | wip |
| #523 | edward | H14: 5-D conditioning vector for FiLM | Feature engineering | -1% to -4% | wip |

## Resolved this round

| PR | Student | Hypothesis | Outcome | val_avg/mae_surf_p |
|----|---------|------------|---------|--------------------|
| #344 | edward | H2: warmup + corrected cosine | **merged** | 120.97 (Run C, –3.4% vs Run A) |
| #346 | frieren | H7: z-mirror augmentation | closed (strict regression) | +231% at p=1.0 |
| #349 | thorfinn | H8: slice_num scaling matrix | closed (regression vs baseline) | 148.65 (+23% vs baseline) |
| #345 | fern | H4: surface-only norm + distance feature | closed (cruise OOD structural regression) | 129.13 (+6.7% vs baseline) |
| #406 | frieren | H10: surf_weight ramp curriculum | closed (within-experiment +4.3% vs A but +1.6% regression vs baseline; effect below seed-variance floor) | 122.90 (+1.6% vs baseline) |
| #490 | frieren | H13: stochastic depth (DropPath) | closed (B-vs-A signature matched prediction strongly but absolute effect below noise floor) | 120.57 (+1.0% vs current baseline) |
| #404 | edward | H11: Re-conditional FiLM modulation | **merged** (after disentanglement) | **119.36** (Run E, −1.3% vs prior baseline) |

## Held in reserve / promising follow-ups

- **Edward's `--epochs 14` lr-frontier follow-up** — finish testing the schedule frontier on the merged code.
- **Thorfinn's slice_num=96 follow-up** — never tested below the slice 128 winner; the 128→256 curve goes sharply up which hints 64 may already be slightly over-partitioned.
- ~~**Frieren's TTA (test-time augmentation) study**~~ — now in flight as PR #561 (H15).
- **Frieren's domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip; small but clean physics.
- **Fern's C2-Lite ablation + multi-scale distance feature** — milder loss-rebalancing variants that *might* dodge the cruise structural regression.
- **Per-domain `surf_weight`** — closely related to H10's failure mode; revisit if H10 mechanism can be salvaged.
- **FiLM hidden=32** — halve the FiLM head's 83K params from PR #404, tightening the +12.6% params objection.
- **Concat-Re instead of FiLM** — cheaper alternative to FiLM if it captures most of the gain.
- **3-seed nail-down of Run E (PR #404)** — would give a real error bar on the −2.2% test headline.
- **Compounding the round-0 winners** — once 2–3 separate ideas merge, run a combined-best PR to ensure their gains are additive.

## Open methodological note

We have now seen single-run noise of **~6% peak-to-peak** across multiple PRs:
- #404 (FiLM, round 1): Run A control 7% *worse* than baseline on equivalent code
- #406 (surf_weight ramp): Run A control 6% *worse* than baseline on equivalent code
- #442 (EMA): Run A control 2.6% *better* than baseline on equivalent code
- #490 (DropPath): Run A control 7.7% *worse* than baseline on equivalent code

The variance is bidirectional and well-calibrated at ~6% peak-to-peak. **Predicted effect sizes <5% are below this noise floor by design.** Going forward:
- New small-effect hypotheses should plan multi-seed confirmation up front (PR #404 round 2 was the model — Run D for disentanglement + Run E with `--seed 123` for variance check).
- The `--seed` CLI flag shipped via PR #404 makes seed-controlled comparisons cheap; future PRs that depend on small effects should reuse it.
- Landing H6 (askeladd, throughput) so we can run more epochs per training is a strong argument for compounding effect-vs-noise improvements.

## Potential next research directions (post round 0)

- **Once throughput lands (H6),** revisit larger architectures with H8 follow-ups and a wider `n_hidden`.
- **If H1 (per-sample y-std) and H3 (Huber) both land,** combine them in one PR — they target different aspects of the heavy-tailed surface pressure regime.
- **If geometry-OOD splits remain stubborn** after rounds 0/1, escalate to graph/edge-aware mesh modules or coordinate-network heads.
- **If Re-OOD splits remain stubborn,** explore Re-conditional separate models or hierarchical heads beyond what FiLM provides.
- **Best-checkpoint reproducibility** — at some point, lock in the best-known recipe and run a multi-seed reproducer to estimate variance for paper-ready numbers.
