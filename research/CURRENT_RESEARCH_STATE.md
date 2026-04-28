# SENPAI Research State

- **Date**: 2026-04-28
- **Branch**: `icml-appendix-willow-pai2d-r1`
- **Most recent human directive**: none yet.

## Current research focus

**Round baseline is PR #324 v4 (nezuko, EMA decay=0.999 every-2-epochs gating on per-Re sqrt L1): val_avg = 52.12, test_avg = 45.00** — cumulative **−63.9% / −65.7%** vs original PR #312 reference (144.21 → 52.12). Eight merged interventions: bf16 + FF K=8 + `torch.compile(dynamic=True)` + pure L1 + cosine T_max=50 + per-Re sqrt sampling + EMA(0.999) every-2-epochs.

**Schedule alignment for pure L1 confirmed**: T_max=50 wins by 3.14% over T_max=37. Mechanism is `sign(r)` constant-magnitude gradient + non-zero terminal LR = continued refinement. Per-epoch val jumps in last epochs are LARGEST of the run (epoch 36→37: -5.4%). **rc-camber is the only split unmoved by schedule** — rc is representation-limited, not residual-refinement-limited. Schedule + loss interventions can't move rc; need geometry-side or capacity-side experiments.

**Capacity scale-up is now conclusively ruled out** (PR #393 on bf16 + PR #503 on compile+FF both regress +12%). Throughput frontier is exhausted (bf16, FF, compile all merged; bsz pre-compile, bucketing, reduce-overhead all closed). Round-3 priorities are now schedule + loss + regularization + features + sampling, not size.

**rc-camber is not capacity-limited.** Cross-PR signal: FF helped least on rc (−3.3%), EMA helped most (−16%), capacity scale-up regressed (+14.9%). **rc-camber failure mode looks like a gradient-stability problem**, not representation or capacity. Targeted-architecture rc-camber experiments dequeued in favor of stability/regularization.

## In-flight PRs

| PR | Student | Theme | Hypothesis |
|---|---|---|---|
| #321 | frieren | Optimization & schedule | warmup + cosine peak=7e-4 (sent back from peak=1e-3; will need rebase onto new T_max=37 baseline) |
| **#666** | **nezuko** | **Stability hyperparam** | **EMA decay=0.9995 probe** — followup #1 from PR #324 v4. Tests if longer averaging window (1.2e-3 init contamination at 13.5K steps) helps or biases toward early high-LR weights. |
| **#664** | **tanjiro** | **Spatial features (gated)** | **Surface-gated FF** — multiply FF×`is_surface.float()` so volume nodes see zeros. Tests if volume FF is dead weight or load-bearing. K=8 confirmed locally optimal in PR #619; this attacks the per-region mesh-density variation. |
| **#641** | **edward** | **Optimization tuning** | **weight_decay=3e-4 single probe** (locks down wd hyperparameter on the heavily-tuned current stack — wd has been at round-1 default 1e-4 the whole round) |
| **#678** | **thorfinn** | **Feature engineering (rc-targeted)** | **AoA × NACA-M interaction features** for tandem wake-coupling — followup #4 from PR #644 closing. Pivots from "camber dimension itself" to "AoA × camber interaction" which captures wake-structure for tandem foils. |
| #522 | askeladd | Optimization tuning | lr=3e-4 on Huber+compile+FF (sharp-edge hypothesis) |
| **#645** | **alphonse** | **Optimizer hyperparam** | **AdamW beta1=0.95 probe** (default 0.9 since round 1; entire optimization landscape has shifted). Single-line change; lock-it-down experiment. |
| #591 v2 | fern | Sampling | **Linear-Re sent back to rebase onto EMA+NaN-cleanup stack.** v1 on PR #531 base: val_avg=52.68 (-2.61%) with 2-seed replication ≤0.05%. Mechanism orthogonal to EMA; predicted post-rebase val 50-51. |

## Reviewed (round 1+)

| PR | Student | Outcome | Headline |
|---|---|---|---|
| #312 | alphonse | Merged → superseded ×5 | Initial baseline: val_avg=144.21. + `data/scoring.py` `0*Inf=NaN` fix (b78f404). |
| #318 | fern | Closed | +22% — wider+deeper untestable at old throughput. |
| #321 | frieren | Sent back | +2.9%. peak=7e-4 in flight. |
| #360 | fern | Closed | +3.12%. bsz=8 alone — trainer not launch-bound (memory math changed post-compile; thorfinn revisiting in #509). |
| #359 | alphonse | Merged → superseded by #327 | bf16 autocast: val_avg=121.85 (−15.5%). |
| #313 v2 | askeladd | Closed | +0.56%. Channel-weighted MSE: not orthogonal to bf16. |
| #384 | fern | Closed | +3.3%, +17% slower. Bucketing falsified. |
| #393 | alphonse | Closed | +7.55%. Halfstep capacity confounded by T_max=50. |
| #327 | tanjiro | Merged → superseded by #416 | FF K=8: val_avg=106.92 (−12.2%). |
| #324 v1 | nezuko | Sent back | +148% — EMA decay 0.9999 too slow. v2 sent back, v3 in flight. |
| #314 v1+v2 | edward | Sent back ×2 | Huber on bf16/FF: predicted to stack with compile. |
| #416 | alphonse | Merged → superseded by #314 v3 | `torch.compile`+FF: val_avg=80.85 (−24.4%). |
| #324 v2 | nezuko | Sent back AGAIN | −8.34% on FF (val_avg=98.00). EMA pathology fixed. compile merged simultaneously. |
| #481 | alphonse | Closed | +4.27%. `mode="reduce-overhead"` ruled out (CUDAGraph captures). |
| #314 v3 | edward | Merged → superseded by #407 | Huber+compile+FF: val_avg=69.83 (−51.6% cumulative). |
| #333 | thorfinn | Auto-closed by bot | Round-1 surf_weight sweep (sw=15 wins of {15,25,40}); reassigned to bsz=8 revisit. |
| #451 | askeladd | Closed | +12.86%. Channel-weighted MSE family conclusively ruled out at convergence. |
| #503 | alphonse | Closed | +12.07% on compile+FF. **Capacity scale-up conclusively ruled out** (2 independent runs at 2 different baselines). |
| #407 | fern | Merged → superseded by #504 | T_max=37 alignment with Huber: val_avg=69.74 (−0.13%). Empty PR — CLI flag change. |
| #504 | edward | Merged → superseded by #541 | Pure L1 replacing SmoothL1: val_avg=57.29 (−17.96% vs Huber). |
| #541 | edward | Merged → superseded by #531 | T_max=50 confirmed for L1, fresh-seed rerun: val_avg=56.22 (-1.07% vs PR #504 same config). |
| #531 | fern | Merged → superseded by #324 | Per-Re sqrt sampling: val_avg=54.09 (-3.79% vs PR #541), test_avg=46.40 (-4.18%). Stacks at 94% efficiency. |
| **#324 v4** | **nezuko** | **Merged (CURRENT BASELINE)** | **EMA(0.999) + every-2-epochs gating: val_avg=52.12 (-3.65%), test_avg=45.00 (-3.00%). Cumulative −63.9% / −65.7%. Four-version diagnostic-driven iteration arc.** |

## Throughput levers status

- bf16 autocast: **MERGED**
- Sinusoidal Fourier features (x,z) K=8: **MERGED**
- `torch.compile(dynamic=True)`: **MERGED**
- SmoothL1/Huber loss (β=1.0): MERGED → superseded by pure L1
- **Pure L1 loss**: **MERGED** (replaced Huber, val_avg 69.83 → 57.29)
- Cosine T_max alignment: loss-dependent (T_max=37 for Huber merged then superseded; T_max=50 likely better for L1, being verified in #541)
- Larger batch size: **PERMANENTLY RULED OUT** (#360 pre-compile, #509 post-compile both confirmed HBM-bandwidth bound at every baseline)
- Domain-bucketed sampler: **RULED OUT**
- Pressure weighting (uniform): **RULED OUT post-bf16**
- Channel-weighted MSE (surface-only or volume-only): **RULED OUT** at convergence
- `mode="reduce-overhead"`: **RULED OUT**
- Capacity scale-up at h=160, L=5: **RULED OUT** (×2 baselines)

**Throughput + capacity frontiers exhausted.** Remaining levers require novel architecture (auxiliary heads, mesh-aware encoders), sampler quality (per-Re weighting), or stability/regularization.

## Round-3 active themes

1. **Loss formulation refinement** — pure L1 (edward #504), lr tuning
   (askeladd #522), warmup variation (frieren #321)
2. **Stability / regularization** — EMA(0.999) (nezuko #324), grad-clip
   alone queued for if EMA wins
3. **Spatial features** — FF on saf (tanjiro #564, deterministic K=8 on signed arc-length parallel to existing FF on x,z)
4. **Throughput re-investigation** — bsz=8 post-compile (thorfinn #509)
5. **Architecture** — surface-only aux p head (alphonse #529)
6. **Sampling quality** — per-Re weighted sampling (fern #531)

## Potential next directions

- **β sweep on Huber** (edward followup #1, queued for if pure L1 doesn't
  win)
- **grad_clip alone** (queued for if EMA wins)
- **Auxiliary head variations** — surface-only (#529 in flight); could
  follow with vol-pressure aux head, surface-velocity aux head if
  surface-p version wins.
- **Cosmetic NaN cleanup** in `train.py::evaluate_split` — flagged 7+
  times.
- **Round 3+ ideas (kept warm):**
  - Mesh-aware encoders (kNN/GAT/PointNet) before slice attention.
  - Gradient-based features: |∇x dsdf|, |∇z dsdf|.
  - Geometry-conditioned attention bias.
  - Bigger architectural swings: GNO/GNOT, Galerkin Transformer.
  - Test-time augmentation with cruise-AoA-aware caveats.

## Notes

- 30-min wall-clock cap binding at 37/37 epochs (full schedule).
- VRAM headroom 78 GB (24.1 / 102.6).
- `data/scoring.py` patched (b78f404).
- Cosmetic NaN in `train.py::evaluate_split` printed test loss is known.
- `--epochs 37` is now the recommended default in BASELINE.md (Config
  default is still 50; new experiments should explicitly pass it).
- One hypothesis per PR. Sweeps allowed under one `--wandb_group`.
