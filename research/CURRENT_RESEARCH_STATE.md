# SENPAI Research State — icml-appendix-charlie-pai2d-r4

- **Date:** 2026-04-28 00:05
- **Track:** charlie-pai2d-r4 (TandemFoilSet — Transolver CFD surrogate)
- **Primary metric:** `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- **Test metric:** `test_avg/mae_surf_p` (same 4-axis structure)

## Current research focus

**Round 1 status (in flight):** PR #287 (alphonse, surf_weight=25) merged as the first baseline → `val_avg/mae_surf_p = 126.67`, `test_avg/mae_surf_p = 114.88` at epoch 14/50 timeout-capped. Round 1 is effectively a **14-epoch ranking exercise** — the cosine schedule's tail is unreached for every PR. This makes throughput a first-class research lever, not just a hyperparameter knob.

**Round 1 covers four orthogonal axes:** loss formulation, loss weighting, architecture capacity, and optimization. The published Transolver in `train.py` is a small model (128 hidden, 5 layers) trained with MSE on normalized targets even though we evaluate on MAE — both gaps were obvious low-cost wins.

**New strategic axis (assigned to alphonse, PR #372):** training throughput. bf16 autocast is the first lever; downstream candidates are `torch.compile`, channels-last memory format, and larger batch sizes. Predicted Δ from autocast alone: -10% to -20% on val_avg/mae_surf_p simply from finishing more of the cosine schedule's tail. Critical because the gain compounds with every other in-flight experiment.

**Resolved infrastructure issue:** PR #358 (edward) merged 2026-04-27 — `data/scoring.py` now uses `torch.where` masking instead of float-mask multiplication, so `inf * 0 = NaN` no longer poisons the float64 accumulator. New `data/test_scoring.py` has 4 surgical regression tests. Existing in-flight PRs branched **before** the merge will still produce NaN on `test_geom_camber_cruise/mae_surf_p`; we'll rebase or cherry-pick on a per-PR basis at review time. Future assignments branch from the post-fix advisor branch.

## Round 1 hypotheses
| Student | PR | Slug | Axis | Predicted Δ | Status |
|---|---|---|---|---|---|
| alphonse | #287 | surf-weight-up | Loss weighting (surf_weight 10→25) | -3% to -7% | **MERGED** e4a0c18 → val_avg=126.67 (14/50 epochs) |
| alphonse | #372 | bf16-autocast | Throughput (autocast forward in bf16) | -10% to -20% | WIP |
| askeladd | #289 | huber-loss | Loss formulation (MSE→SmoothL1/Huber) | -5% to -10% | WIP |
| edward   | #300 | wider-model | Architecture width (192/96) | -5% to -10% | **CLOSED** — under-trained 9/50 epochs at 30-min cap |
| edward   | #358 | fix-scoring-nan-mask | Maintenance fix to data/scoring.py | n/a (unblocks test_avg) | **MERGED** 010235e |
| edward   | #368 | fourier-pos-encoding | Architecture/input (8-freq Fourier features on (x,z), fun_dim 22→54) | -3% to -8% | WIP |
| fern     | #304 | deeper-model-droppath | Architecture depth (n_layers 5→8 + DropPath 0.1) | -3% to -8% | WIP |
| frieren  | #307 | warmup-cosine-1e3 | Optimization (linear warmup + peak lr 1e-3) | -2% to -6% | WIP |
| nezuko   | #308 | ema-grad-clip | Optimization (EMA decay 0.999 + grad clip 1.0) | -3% to -8% | WIP |
| tanjiro  | #309 | more-slices | Architecture (slice_num 64→128, n_head 4→8) | -3% to -7% | **CLOSED** — 2x slower per epoch and not better at equal-epoch |
| tanjiro  | #378 | per-sample-relmse | Heavy-tail (per-sample y-variance normalization in loss) | -3% to -7% | WIP |
| thorfinn | #310 | per-channel-surf-weights | Loss weighting (3× surface pressure) | -3% to -8% | WIP |

## Lessons from round 1 so far
- **The 30-min cap is binding for everyone.** alphonse fits 14 epochs (small/baseline model) at 131 s/epoch; edward's wider hit only 9 epochs at 205 s/epoch. Compute, not memory, is the bottleneck (peak ~42-63 GB / 96).
- **Round 1 is a 14-epoch ranking exercise**, not a 50-epoch one. The cosine schedule's tail is unreached. Decisions made in this round inherit that caveat — winners may need re-validation once we recover budget.
- **Throughput is now a first-class research axis.** bf16 autocast (PR #372) is the first lever; success there compounds with every architecture/loss change we've assigned.
- Two students independently diagnosed the `inf * 0 = NaN` mask trap in `data/scoring.py`; fixed in #358 (merged).

## Potential next research directions (post-round 1)
- **Compound round-1 winners** in a single PR after PR #372 lands — surf_weight=25 + bf16 autocast + (Huber if askeladd wins) + (EMA if nezuko wins) + (warmup if frieren wins). Each is independent and they should stack.
- **Further throughput wins after bf16:** `torch.compile(model, mode="reduce-overhead")`, channels-last memory format, larger batch size enabled by halved activations.
- Conservative widening that fits the 30-min budget: e.g. n_hidden=144, slice_num=80 — only meaningful once throughput is recovered.
- Heavy-tail-aware pressure handling: per-sample y-std normalization, log-pressure target, or focal weighting on extreme |p|.
- Fourier / RFF positional encoding on (x, z) to give the model multi-scale spatial frequency info — currently only raw position + signed-arc-length.
- Surface-aware decoder: separate surface-only head with extra capacity, since `mae_surf_p` is what we're scored on.
- Domain-conditional FiLM modulation of attention slices (single vs. raceCar tandem vs. cruise tandem) — the three regimes differ by orders of magnitude in y-std.
- Gradient-aware loss scaling (GradNorm or DWA) across surface and volume to stop one branch dominating.
- Test-time augmentation: average predictions from x↔−x flipped meshes (after re-orienting AoA / stagger).
- Better val-track-aware checkpoint selection: weight `val_geom_camber_*` higher since they're the harder generalization tracks.
