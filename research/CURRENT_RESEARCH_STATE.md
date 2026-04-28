# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

bf16 autocast (PR #359) is the new baseline: **val_avg/mae_surf_p = 121.85**, **test_avg = 111.15** — a 15.5% improvement over the original PR #312 baseline (144.21). The win came entirely from running 5 more epochs of the cosine schedule inside the 30-min cap (per-epoch wall −26%, peak GPU −22% to 32.9 GB).

That changes the focus: **throughput is mostly unblocked, capacity is now testable**. Key open questions:

1. Does a half-step capacity scale-up (h=160, L=5, heads=5, slices=80) win on top of bf16, now that fern's earlier wider+deeper had a chance to converge? — *just-assigned, alphonse PR #393*
2. Does the domain-bucketed sampler (fern PR #384) compose with bf16 to push throughput further?
3. Does cosine T_max alignment (`--epochs 20-ish` instead of 50) help — currently the schedule barely decays in 19 epochs.
4. Do the original round-1 hypotheses (loss/regularization/features) still win against the new, much stronger 121.85 baseline?

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #313 | askeladd | Loss / metric alignment | Pressure-channel-weighted MSE (5x p) |
| #314 | edward | Loss / metric alignment | SmoothL1 / Huber (β=1.0) |
| #321 | frieren | Optimization & schedule | 5-epoch warmup + cosine, **peak=7e-4** (sent back) |
| #324 | nezuko | Stability / regularization | EMA(0.9999) + grad-clip 1.0 |
| #327 | tanjiro | Spatial inductive bias | Fourier features for (x, z), K=8 |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #384 | fern | Throughput | Domain-bucketed batch sampler |
| **#393** | **alphonse** | **Capacity** | **Half-step scale-up on bf16 (h=160/L=5/heads=5/slices=80)** |

## Reviewed (round 1)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded by #359 | Initial baseline: val_avg=144.21. Cherry-picked the `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% regression. Wider+deeper untestable at old throughput. Redirected to #360 throughput. |
| #321 | frieren | Sent back | +2.9% with peak=1e-3. Variation requested: peak=7e-4. |
| #360 | fern | Closed | +3.12%. Trainer not launch-bound. Diagnosis: padding waste. Redirected to #384 bucketed sampler. |
| **#359** | **alphonse** | **Merged (NEW BASELINE)** | **bf16 autocast: val_avg=121.85 (−15.5%), test_avg=111.15. 26% faster, 22% less memory.** |

## Note on round-1 baseline shift

The round-1 cohort (#313, #314, #324, #327, #333) was assigned against the pre-bf16 baseline (val_avg=144.21). When their results land, comparison is now against **121.85** — a much harder target. Expected effect: more of the round-1 cohort will land near or just above the new baseline rather than below it. Not closing them preemptively — small positive deltas on different dimensions still compound usefully when stacked, and the new baseline reveals which interventions transfer to the bf16 regime.

## Potential next directions

- **Compose the throughput wins.** Once #384 (bucketed sampler) lands, if it
  wins on top of bf16, the round-2 baseline is bf16 + bucketed.
- **Cosine T_max alignment.** Set `--epochs 20-ish` so cosine actually
  decays in the achievable epoch budget. Cheap, isolated, likely a small
  but reliable gain. Queued for next idle slot.
- **Stack capacity (if #393 wins) + bf16** as the next baseline, then
  re-test the round-1 hypotheses against that baseline.
- **`torch.compile`** on top of bf16, ideally in `dynamic=True` mode for
  variable mesh sizes. Risky (graph breaks) but could give another
  meaningful throughput nudge.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` loss accumulator —
  flagged twice now (alphonse and fern). Worth fixing in a passing PR.
- **Beyond round 1** ideas, kept warm:
  - Test-time augmentation: mirror-flip x for cruise foils.
  - Per-Re weighting in the sampler.
  - Surface-only auxiliary head on surface nodes only.
  - Mesh-aware encoders: kNN/GAT/PointNet local message passing before
    slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf| as extra input channels.
  - Bigger architectural swings if simpler levers plateau: GNO/GNOT,
    Galerkin Transformer, multi-scale slice transformer.

## Notes

- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) is **still binding**
  even with bf16 — 19/50 epochs at the new baseline.
- `data/scoring.py` patched (b78f404).
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
  MAE rankings are correct.
- One hypothesis per PR. Sweeps allowed when localized under a single
  `--wandb_group`.
