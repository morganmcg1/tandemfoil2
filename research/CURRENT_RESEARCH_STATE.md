# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-28 (round 1 kickoff)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`

## Current research focus

Establish a working baseline on `val_avg/mae_surf_p` and `test_avg/mae_surf_p`
by exploring the **stock-Transolver knobs that are most likely to help out of
the box** before committing to architectural changes. The stock config is
visibly under-tuned along several axes:

- The Transolver is small (~512K params) while VRAM is ~96 GB — capacity is
  available.
- Loss is MSE while the metric is MAE — there is a known loss/metric mismatch.
- `surf_weight` and per-channel weights have not been swept; surface pressure
  is the only thing the headline metric measures.
- Default `lr=5e-4` with no warmup and `batch_size=4` are conservative.

Round 1 covers eight orthogonal levers so we can read the gradient of the
landscape after a single round of merges.

## Round 1 hypotheses (one per student)

| Student | Lever | Predicted edge |
|---------|-------|----------------|
| alphonse | Capacity scale-up (`n_hidden=256, n_layers=8`) | Headroom from underused VRAM |
| askeladd | L1 loss in normalized space | Loss/metric alignment |
| edward | `surf_weight` sweep up (10 → 30) | Direct upweight of headline metric |
| fern | Per-channel loss weight on `p` (3×) | Pressure is the only ranked channel |
| frieren | More physics slices (`slice_num=64 → 128`) | Finer slice decomposition |
| nezuko | LR warmup (5% linear) + cosine | Adam stability early in training |
| tanjiro | Higher peak LR (1e-3) with 10% warmup | Default 5e-4 is conservative |
| thorfinn | Larger batch (`batch_size=8`) | Better gradient estimates, VRAM available |

## Round 2 hypotheses ranked and ready

Researcher-agent literature pass landed in
[`RESEARCH_IDEAS_2026-04-28_round2.md`](RESEARCH_IDEAS_2026-04-28_round2.md)
on 2026-04-28. Top 10 in expected-impact order, queued for assignment as
soon as round 1 PRs return for review:

1. **Relative L2 loss** — sample-normalized MSE; the high-Re samples are
   dominating the gradient. (-5–15%)
2. **FiLM conditioning of LayerNorm** — global scalars (log(Re), AoAs,
   geometry) get direct modulation authority over every block. (-5–12%)
3. **Fourier positional encoding** on `(x, z)` — captures the 1000:1
   wavelength range from boundary layer to background. (-4–10%)
4. **POD/PCA output reparameterization** — surface pressure lives in a
   low-dim modal subspace; predict K=32 PCA coefficients. (-5–15%, won
   ML4CFD 2024 1st place via MMGP)
5. **Huber loss δ=1.0** — outlier-robust variant of MSE on the same
   error tensor. (-3–8%)
6. **LinearNO ELU linear attention** — drops slice-token attention from
   O(S²) to O(S·d), often beats Transolver at lower compute. (-3–8%)
7. **Surface-node geometric oversampling** — keep only 15% of volume
   nodes per step; raises surface share to ~25%. (-3–8%)
8. **Domain-ID embedding** — derive 3-class domain ID from gap/AoA1 and
   embed; resolves the regime discontinuity baked into the dataset. (-2–6%)
9. **SGDR warm restarts** — three cosine basins inside the 50-epoch cap. (-2–5%)
10. **Test-time augmentation** (vertical mirror) — 2× inference cost,
    no training change. (-1–4%)

Best two-by-two interactions to test once round 1 winners are merged:
- Relative L2 × FiLM conditioning (likely the largest compounded gain)
- PCA output reparameterization × any winning loss
- Surface oversampling × winning surf_weight from round 1

## Open uncertainties (from researcher-agent)

- Which lever contributes more: relative L2 loss vs. FiLM conditioning. They
  likely compound but the interaction is unknown.
- Whether a PCA basis fit on training data transfers to OOD camber splits
  (`val_geom_camber_rc`, `val_geom_camber_cruise`).
- Whether surface-node oversampling interacts adversely with the existing
  balanced-domain sampler.

## Notes / constraints

- `SENPAI_MAX_EPOCHS=50` and `SENPAI_TIMEOUT_MINUTES=30` are hard caps. Don't
  override them — design experiments that fit in 30 minutes wall-clock.
- All four val splits are weighted equally in the headline metric. Prefer
  changes that travel across splits over hacks that only help one.
- Test metrics are computed at the end of every run from the best-val
  checkpoint — they are the ranking quantity for the paper.
