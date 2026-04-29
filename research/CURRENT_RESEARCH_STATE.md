# SENPAI Research State

- 2026-04-29 10:55 (round 1 in flight, branch `icml-appendix-charlie-pai2f-r1`)
- No human researcher directives yet for this branch.
- Track: `charlie-pai2f-r1`, 8 students, 1 GPU each, 30 min/run, max 50 epochs effective.

## Round 1 status

| PR | Student | Hypothesis | Status | best val_avg/mae_surf_p |
|---|---|---|---|---|
| #1092 | alphonse | capacity-scale-up | wip | — |
| #1094 | askeladd | surf-weight-25 | wip | — |
| #1095 | edward | pressure-channel-weight | sent back (formula) | 133.892 |
| #1096 | fern | huber-vol | wip | — |
| #1097 | frieren | slice-num-128 | sent back (bs↑, clamp) | 162.562 |
| #1099 | nezuko | lr1e-3-warmup5 | wip | — |
| #1100 | tanjiro | wider-bs8 (fallback bs=5) | sent back (mlp_ratio↓, clamp) | 165.304 |
| #1101 | thorfinn | warmup-cosine-floor | wip | — |

## Cross-experiment learnings so far

1. **30-min budget is the binding constraint.** All 3 finished runs hit timeout: edward 14/50, frieren 11/50, tanjiro 8/50. None reached the cosine LR low-LR phase. Per-epoch wall clock ranges from ~130s (baseline shape) to ~227s (n_hidden=256, bs=5). **Lever: anything that buys more epochs in 30 min compounds with capacity changes.**
2. **VRAM utilization varies wildly.** Frieren used 54 GB of 95 GB at slice_num=128 (massive headroom for bs↑), while tanjiro hit 92 GB at n_hidden=256+bs=5 (no headroom). Per-experiment VRAM-aware bs tuning is the next throughput lever.
3. **Test pressure NaN is a multi-failure mode.**
   - **Mode A (data):** `test_geom_camber_cruise/000020.pt` has +Inf in p ground truth — exposed by `data/scoring.py` mask-multiply propagating NaN. **Branch-side fix applied** via `torch.where`-based masking (round-1 metrics already report finite values for runs that finish post-fix).
   - **Mode B (model):** wider tanjiro produced fp32 overflow in pred_p on a cruise inference sample, blowing up vol_loss to +Inf. Output-side pressure clamping is the right fix at the model side; sent that to tanjiro.
4. **Best so far: 133.89.** Provisional, edward only — and it's a confounded run (loss formula softened aggregate surface signal ~3×). True round-1 winner not yet decided.

## Branch-side fixes

- **`data/scoring.py` NaN-propagation bug.** Multiply-mask let NaN ground-truth p
  values bleed past the sample-level filter, producing NaN
  `test_avg/mae_surf_p` whenever any test sample has non-finite y. Fixed via
  `torch.where`-based masking on the advisor branch (committed alongside this
  state update). In-flight student runs that finish before they can rebase will
  still report NaN test pressure on `test_geom_camber_cruise`; their val
  numbers are unaffected. Merge winners on val_avg, treat test_avg as paper
  number that will need rerun if NaN. **Confirmed independently by edward (PR
  #1095) and frieren (PR #1097).**

## Current research focus

Round 1 establishes a balanced sweep across the main optimization levers for the
default Transolver baseline on TandemFoilSet. The eight assignments cover three
families:

1. **Capacity scaling** — `alphonse` (n_hidden 192, layers 6, mlp_ratio 4),
   `tanjiro` (n_hidden 256 + bs 8), `frieren` (slice_num 128).
2. **Loss / metric alignment** — `askeladd` (`surf_weight 25`), `edward`
   (per-channel pressure-weighted surf loss), `fern` (Huber on volume).
3. **Optimization discipline** — `nezuko` (lr 1e-3 + warmup), `thorfinn`
   (warmup + non-zero cosine floor at default lr).

The intent is to pin down which lever moves `val_avg/mae_surf_p` most, then in
later rounds stack the winning levers and explore architecturally bolder
follow-ups (Fourier features, neural operator hybrids, attention variants,
physics-informed losses, EMA / SWA averaging, etc.).

## Next research directions (post-round-1 candidates)

- **Stack winners.** Whichever capacity, loss, and schedule changes win get
  combined into a single recipe and re-tested.
- **Per-domain specialization.** Inspect per-split metrics — if camber-rc /
  camber-cruise behave differently from re_rand, explore conditioning or domain
  embedding (e.g. learned domain token concatenated to features).
- **Geometry-aware augmentation.** Random AoA reflection (sign flip + y-axis
  flip), light positional jitter, or NACA cambered-thickness perturbation could
  expand effective sample count without changing the data contract.
- **Spectral / Fourier features.** Random Fourier features on x[:, 0:2] (node
  positions) often boost mesh-based surrogates in CFD.
- **Loss reformulation.** Sobolev-style loss (gradient matching), per-sample
  scale-aware losses (divide errors by sample y_std), pressure-only auxiliary
  head with a stronger weight.
- **Optimizer swap.** Lion or AdEMAMix as alternatives to AdamW, especially if
  capacity-scaling wins because larger models often respond better to
  alternative optimizers.
- **Sampler tweaks.** Sampling weighted by per-sample y_std (high-variance
  samples seen more often) to attack heavy-tailed errors.
- **Model averaging.** EMA of weights with decay 0.999, evaluating EMA at val
  time for noise-robust generalization.

## Constraints reminder

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary metric: `val_avg/mae_surf_p`; test metric: `test_avg/mae_surf_p`.
