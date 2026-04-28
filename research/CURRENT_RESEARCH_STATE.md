# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

Round baseline is **PR #327 (tanjiro, FF K=8 on bf16): val_avg=106.92, test_avg=96.82** — cumulative **−25.9%** vs the original PR #312 reference (144.21).

**Big incoming**: edward's PR #314 (SmoothL1/Huber loss on bf16, no FF) landed val_avg=104.27 — *better than the FF baseline*. Sent back to rebase onto FF and re-run; if Huber stacks with FF (mechanisms look orthogonal), val_avg ≈ 90 is the predicted next baseline. **Huber emerges as the next biggest lever after FF.**

Themes in play:

1. **Spatial frequency representation** — FF K=8 merged. Tanjiro testing **Gaussian RFF** as the followup (#443).
2. **Loss formulation** — Huber sent back to rebase onto FF (#314). askeladd's pressure-weighting falsified post-bf16 (closed #313); refined surface-only variant assigned (#451).
3. **Throughput / schedule** — fern on cosine T_max alignment (#407), alphonse on `torch.compile` pilot (#416).
4. **Stability / regularization, schedule shape** — nezuko (#324, EMA + grad-clip), thorfinn (#333, surf_weight sweep), frieren (#321 sent back, peak=7e-4) — all assigned vs the 144.21 baseline. Will likely need rebase + re-run on FF (or post-Huber) baseline when results come in.

Important falsified hypotheses now ruled out:
- Larger batch alone (PR #360): trainer not launch-bound, padding scales with B.
- Domain-bucketed sampler (PR #384): allocator fragmentation, pipeline mismatch.
- Pressure-weighting (1,1,5) post-bf16 (PR #313 rebase): not orthogonal to bf16; starves Ux/Uy.
- Halfstep capacity at T_max=50 (PR #393): schedule mismatch, parked.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #314 | edward | Loss formulation | **SmoothL1/Huber β=1.0** — sent back to rebase onto FF baseline (high-priority) |
| #321 | frieren | Optimization & schedule | warmup + cosine peak=7e-4 (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #407 | fern | Schedule (on bf16+FF) | Cosine T_max alignment via `--epochs 20` |
| #416 | alphonse | Throughput (on bf16+FF) | `torch.compile(dynamic=True)` pilot |
| #443 | tanjiro | Spatial features (on bf16+FF) | Gaussian RFF K=16 σ=10 |
| **#451** | **askeladd** | **Loss formulation** | **Surface-only pressure weighting (1,1,5) on surf_loss only** |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded twice | Initial baseline: val_avg=144.21. + `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. Variation: peak=7e-4 (in flight). |
| #360 | fern | Closed | +3.12%. bsz=8 alone didn't help — trainer not launch-bound. |
| #359 | alphonse | Merged → superseded by #327 | bf16 autocast: val_avg=121.85 (−15.5%). |
| #313 v1 | askeladd | Sent back → v2 closed | Pre-bf16 −4.2% didn't transfer post-bf16. Falsified post-bf16 (closed v2). |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max=50 / 14-epoch mismatch. Parked. |
| **#327** | **tanjiro** | **Merged (CURRENT BASELINE)** | **FF K=8: val_avg=106.92 (−12.2% vs bf16, −25.9% cumulative). Largest single win.** |
| #313 v2 | askeladd | **Closed** | **+0.56% on bf16 / +14.6% vs FF.** Pressure weighting and bf16 not orthogonal as I'd assumed. Refined hypothesis (#451) assigned. |
| #314 | edward | **Sent back** | **−14.4% vs bf16 / −2.5% vs FF on the wrong base. Stacking with FF predicted to give ~−15% on top.** |

## Throughput levers status

- bf16 autocast: **MERGED**
- Sinusoidal Fourier features (x,z) K=8: **MERGED**
- SmoothL1/Huber loss: **HIGHLY PROMISING** (#314 sent back to test on FF)
- Larger batch size: **RULED OUT**
- Domain-bucketed sampler: **RULED OUT**
- Pressure weighting (uniform): **RULED OUT post-bf16**
- Cosine T_max alignment: in flight (#407)
- `torch.compile` pilot: in flight (#416)

## Potential next directions

- **Stack winners.** When #314 (Huber) lands on FF baseline, that becomes
  the new baseline for everything else. The existing in-flight cohort
  (#321/#324/#333) will likely all need rebases.
- **Targeted rc-camber experiment.** From #327 per-split signal: rc-camber
  M=6-8 holdout gained only −3.3% from FF (vs 17-20% on cruise / single).
  OOD geometry generalisation is bottlenecked by camber→pressure mapping.
  Candidate experiments (queued for round 2): NACA-camber-aware feature
  embedding, per-camber stratified loss reweighting, camber-conditional
  layer norm.
- **FF on saf/dsdf** (tanjiro followup #4) — extend FF to dist-based
  shape descriptor (dims 2-11). High-cost but plausible.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` — flagged 5+
  times now.
- **Round 2 ideas (kept warm):**
  - Test-time augmentation (limited by cruise AoA asymmetry).
  - Per-Re weighting in sampler.
  - Surface-only auxiliary head.
  - Mesh-aware encoders (kNN/GAT/PointNet) before slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf|.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) **still binding** at
  current FF baseline (19/50 epochs). Cosine T_max alignment may release
  a few more % from the schedule tail.
- Memory note: SmoothL1 has higher transient peak memory (~95 GB on
  epochs 1-2) than MSE due to extra autograd intermediates. Means
  Huber + larger-batch combinations are unsafe without instrumentation.
- `data/scoring.py` patched (b78f404) — `test_avg/mae_surf_p` is finite.
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
