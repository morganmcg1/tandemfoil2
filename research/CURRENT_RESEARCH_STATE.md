# SENPAI Research State

- 2026-04-29 (round 1 kickoff, branch `icml-appendix-charlie-pai2f-r1`)
- No human researcher directives yet for this branch.
- Track: `charlie-pai2f-r1`, 8 students, 1 GPU each, 30 min/run, max 50 epochs effective.

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
