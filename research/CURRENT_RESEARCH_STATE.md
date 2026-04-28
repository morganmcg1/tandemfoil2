# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

**Round baseline is PR #416 (alphonse, compile + bf16 + FF): val_avg = 80.85, test_avg = 73.41** — cumulative **−44.0%** vs the original PR #312 reference (144.21 → 80.85). The biggest jumps came from compile (−24.4%), FF (−12.2% earlier), and bf16 (−15.5% earlier).

**Big update on rc-camber**: FF on its own only relieved the OOD `geom_camber_rc` split by −3.3%, hinting at a "camber→pressure mapping" representation bottleneck. Compile + FF takes it to **−25.8%** on the same split — so the rc-camber gap was *schedule-truncation-bound*, not a representation bottleneck. The "camber-aware feature embedding" round-2 experiment has been **dequeued**.

Themes in play:

1. **Throughput / schedule** — compile merged. Followups: `mode="reduce-overhead"` (alphonse #481, just-assigned), cosine T_max alignment (fern #407, in flight). With 37 epochs reachable, T_max=50 is increasingly misaligned.
2. **Loss formulation** — Huber+FF clean win (edward #314, sent back to rebase onto compile+FF — predicted post-rebase val_avg 65-75). Pressure-weighting falsified post-bf16 (askeladd #313 closed); refined surface-only variant in flight (#451).
3. **Spatial features** — FF merged. Tanjiro testing Gaussian RFF (#443) as the representation followup.
4. **Stability / regularization** — nezuko EMA decay=0.999 rebased (#324). EMA-warmup pathology diagnosed and corrected.
5. **Capacity** — parked at PR #393. Now feasible again with compile speedup + 70 GB VRAM headroom; queued.

## Important falsified hypotheses

- Larger batch alone (PR #360): trainer not launch-bound, padding scales with B.
- Domain-bucketed sampler (PR #384): allocator fragmentation, pipeline mismatch.
- Pressure-weighting (1,1,5) post-bf16 (PR #313): not orthogonal to bf16; starves Ux/Uy.
- Halfstep capacity at T_max=50 (PR #393): schedule mismatch, parked.
- EMA decay=0.9999 at 7K-step budget (PR #324 v1): warmup pathology, decay=0.999 rebased.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #314 | edward | Loss formulation | **SmoothL1/Huber β=1.0** — sent back AGAIN to rebase onto compile+FF baseline (was on FF: val_avg=92.32) |
| #321 | frieren | Optimization & schedule | warmup + cosine peak=7e-4 (sent back from peak=1e-3) |
| #324 | nezuko | Stability / regularization | EMA-only **decay=0.999** (sent back from 0.9999, dropped grad_clip — bundled hypothesis) |
| #333 | thorfinn | Loss / metric alignment | surf_weight ∈ {15, 25, 40} sweep |
| #407 | fern | Schedule | Cosine T_max alignment via `--epochs 20` (was for FF baseline; will likely need adjustment to match new 37-epoch budget) |
| #443 | tanjiro | Spatial features (on bf16+FF) | Gaussian RFF K=16 σ=10 |
| #451 | askeladd | Loss formulation | Surface-only pressure weighting (1,1,5) on surf_loss only |
| **#481** | **alphonse** | **Throughput** | **`torch.compile(mode="reduce-overhead")` pilot** |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded ×3 | Initial baseline: val_avg=144.21. Cherry-picked `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. Variation: peak=7e-4 (in flight). |
| #360 | fern | Closed | +3.12%. bsz=8 alone didn't help. |
| #359 | alphonse | Merged → superseded by #327 | bf16 autocast: val_avg=121.85 (−15.5%). |
| #313 v2 | askeladd | Closed | +0.56% on bf16. Pressure weighting and bf16 not orthogonal. Refined hypothesis (#451) assigned. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max mismatch. Parked, retest queued. |
| #327 | tanjiro | Merged → superseded by #416 | FF K=8: val_avg=106.92 (−12.2% vs bf16). |
| #324 v1 | nezuko | Sent back | +148% — EMA decay 0.9999 too slow for 7K-step budget. Rebase + decay=0.999. |
| #314 v1 | edward | Sent back | −14.4% on bf16 alone. Sent back to rebase onto FF (then compile+FF). |
| #314 v2 | edward | **Sent back AGAIN** | **−13.65% on FF (val_avg=92.32). compile+FF merged simultaneously.** |
| **#416** | **alphonse** | **Merged (CURRENT BASELINE)** | **`torch.compile`+FF: val_avg=80.85 (−24.4% vs FF, cumulative −44.0%). 37 epochs in 30 min.** |

## Throughput levers status

- bf16 autocast: **MERGED**
- Sinusoidal Fourier features (x,z) K=8: **MERGED**
- `torch.compile(dynamic=True)`: **MERGED**
- Larger batch size: **RULED OUT**
- Domain-bucketed sampler: **RULED OUT**
- Pressure weighting (uniform): **RULED OUT post-bf16**
- `mode="reduce-overhead"`: in flight (#481)
- Cosine T_max alignment: in flight (#407, may need re-spec post-compile)
- Capacity scale-up (re-test): queued (would benefit from compile + 70 GB headroom)

## Potential next directions

- **Once edward #314 v3 lands** (Huber + compile + FF): predicted val_avg
  65-75. Would unlock the "all four orthogonal levers stacked" config.
- **Capacity scale-up revisited.** With 70 GB VRAM free + 49 s/epoch, the
  half-step (h=160, L=5, heads=5, slices=80) PR #393 should now finish
  ~25-30 epochs in 30 min. Schedule decay still misaligned at T_max=50
  but much better-positioned than the 14 epochs of the original attempt.
- **Cosine T_max=37** to match the achievable budget on compile+FF
  (fern's #407 with `--epochs 20` was matched to the *pre-compile* 19-
  epoch budget). May need to re-spec.
- **β sweep on Huber** (edward followup #1, queued).
- **Pure L1 on FF baseline** (edward followup #2, queued).
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
  current compile+FF baseline (37/50 epochs).
- Memory note: SmoothL1 has higher transient peak (~98 GB on FF baseline);
  Compile drops peak by ~9 GB. Expected post-compile-rebase Huber peak:
  ~88-90 GB. Monitor on edward's next run.
- `data/scoring.py` patched (b78f404) — `test_avg/mae_surf_p` is finite.
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
