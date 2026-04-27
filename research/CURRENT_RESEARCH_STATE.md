# SENPAI Research State

- **Date**: 2026-04-27 (afternoon)
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet (fresh launch).

## Current research focus

Round 1 has produced one merged baseline (PR #312, alphonse, **val_avg/mae_surf_p = 144.2118**) and one notable closure (PR #318, fern, wider+deeper untestable in 30-min budget). The big finding from that pair: **throughput is the binding constraint, not capacity**.

The original round-1 hypothesis themes (loss/metric alignment, scaling, regularization) remain active across the six in-flight PRs from the initial cohort. Round-1.5 expands the focus to **wall-clock throughput** so the planned 50-epoch cosine schedule can actually run.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-channel-weighted MSE (5x p) |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | 5-epoch warmup + cosine, peak lr=1e-3 |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #327 | tanjiro | Spatial inductive bias | Fourier features for (x, z), K=8 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| **#359** | alphonse | **Throughput** | **bf16 autocast on forward + loss** |
| **#360** | fern | **Throughput** | **batch_size=8, lr=7.07e-4 (sqrt-scaled)** |

## Reviewed and closed (round 1)

| PR | Student | Outcome | Reason |
|---|---|---|---|
| #312 | alphonse | **Merged** | Baseline reference (val_avg=144.2118 / test_avg=131.1823). Also: cherry-picked alphonse's `data/scoring.py` `0*Inf=NaN` fix (commit b78f404). |
| #318 | fern | **Closed** | +22% regression (val_avg=175.85). Bigger config untestable at 30-min cap (only 7/50 epochs). Redirected to throughput experiment (#360). |

## Potential next directions

- **Stack throughput wins.** If both #359 (bf16) and #360 (bsz=8) succeed,
  combine them in round 2 — they're orthogonal (one cuts compute time per
  step, one amortizes it across more samples) and should compound.
- **Re-issue the half-step scale-up** (`h=160, L=5, heads=5, slices=80`)
  once throughput is unblocked.
- **Stack winning loss/metric-alignment signals.** If pressure-weighted MSE
  (#313), Huber (#314), and surf_weight tuning (#333) all win independently,
  combine the best of each into one PR.
- **Beyond round 1** ideas, kept warm:
  - Test-time augmentation: mirror-flip x for cruise foils.
  - Per-domain or per-Re weighting in the sampler (high-Re pressure
    dominates the metric tail).
  - Surface-only auxiliary head over surface nodes only.
  - Mesh-aware encoders: kNN message passing over local mesh, GAT/PointNet
    style, before slice attention.
  - Gradient-based features: precompute |∇x dsdf|, |∇z dsdf| as extra input
    channels.
  - Bigger architectural swings if simpler levers plateau: GNO/GNOT,
    Galerkin Transformer, multi-scale slice transformer, hierarchical FNO.

## Notes

- 30-min training cap (`SENPAI_TIMEOUT_MINUTES=30`) is **binding** at the
  baseline config (only 14 of 50 epochs ran). Throughput improvements are
  the highest-value lever right now — the cosine schedule was specified for
  50 epochs and barely decayed.
- `data/scoring.py` is patched (commit b78f404) so all sibling round-1 PRs
  will report finite `test_avg/mae_surf_p`.
- One hypothesis per PR. Sweeps allowed when localized (e.g. surf_weight ∈
  {15, 25, 40}) under a single `--wandb_group`.
