# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

Round 1 produced a merged baseline (PR #312, alphonse, **val_avg/mae_surf_p = 144.2118**); two throughput attempts have been resolved with a key finding: **the trainer is HBM-bandwidth + padding-cost bound, not kernel-launch-bound**. Doubling batch size (PR #360) gave essentially no per-epoch speedup because `pad_collate` pads every batch to the largest sample's mesh size, and the three domains differ ~2.5× in mesh size. This redirects the throughput effort toward (a) bf16 autocast (still in flight, alphonse #359) and (b) **domain-bucketed batch sampling** (just-assigned, fern #384) to cut padding waste.

The original round-1 hypothesis themes (loss/metric alignment, regularization, spatial features) remain active across five in-flight PRs from the initial cohort plus frieren's sent-back PR #321.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-channel-weighted MSE (5x p) |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | 5-epoch warmup + cosine, **peak=7e-4** (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #327 | tanjiro | Spatial inductive bias | Fourier features for (x, z), K=8 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #359 | alphonse | Throughput | bf16 autocast on forward + loss |
| **#384** | **fern** | **Throughput** | **Domain-bucketed batch sampler (cut padding waste)** |

## Reviewed (round 1)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | **Merged** | Baseline: val_avg=144.2118 / test_avg=131.1823. Cherry-picked the `data/scoring.py` `0*Inf=NaN` fix into b78f404. |
| #318 | fern | **Closed** | +22% regression (val_avg=175.85). Bigger config untestable at 30-min cap. |
| #321 | frieren | **Sent back** | +2.9% with peak=1e-3; epochs 6-7 instability after warmup. Variation: peak=7e-4. |
| #360 | fern | **Closed** | +3.12% (val_avg=148.72). Trainer not launch-bound — bsz alone doesn't help. Diagnosis points at padding waste. Redirected to bucketed-sampler (#384). |

## Potential next directions

- **Compose throughput wins.** bf16 (#359) + bucketed sampler (#384) are
  orthogonal (compute time per step vs padded-node count per step). If both
  win, stack them in round 2 — likely 2-3× total throughput.
- **Re-issue the half-step scale-up** (`h=160, L=5, heads=5, slices=80`)
  once throughput is unblocked. The bigger model is parked, not abandoned.
- **Stack winning loss/metric-alignment signals.** If pressure-weighted MSE
  (#313), Huber (#314), and surf_weight tuning (#333) all win independently,
  combine the best of each into one PR.
- **Cosine T_max realism.** The schedule was spec'd for 50 epochs but only
  14 run. After throughput unlocks, either (a) raise SENPAI_MAX_EPOCHS, or
  (b) keep `--epochs 50` and let cosine actually decay; either way the
  current half-decayed schedule is a confounder for every comparison.
- **Beyond round 1** ideas, kept warm:
  - Test-time augmentation: mirror-flip x for cruise foils (raceCar is
    asymmetric in z so this is split-specific).
  - Per-Re weighting in the sampler (high-Re samples drive the pressure
    metric tail).
  - Surface-only auxiliary head trained on surface nodes only.
  - Mesh-aware encoders: kNN/GAT/PointNet local message passing before
    slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf| as extra input channels.
  - Bigger architectural swings if simpler levers plateau: GNO/GNOT,
    Galerkin Transformer, multi-scale slice transformer.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) is **binding** —
  every reviewed run so far stopped at 14 epochs. The cosine LR schedule
  for 50 epochs barely decays in 14, so all comparisons are
  early-training, not converged.
- `data/scoring.py` patched (b78f404) — `test_avg/mae_surf_p` is finite for
  all sibling PRs.
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known
  and not blocking; MAE rankings are correct.
- One hypothesis per PR. Sweeps allowed when localized (e.g. surf_weight ∈
  {15, 25, 40}) under a single `--wandb_group`.
