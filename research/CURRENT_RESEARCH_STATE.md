# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-28 ~21:00 (round 1 closing, round 2 in flight)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`
- **Current best:** `val_avg/mae_surf_p = 101.93` (PR #752, L1 loss merged)

## Current research focus

Round 1 has resolved into a clear picture: **L1 loss merged** as the new
baseline (101.93). Other levers either failed to compound on top of L1
(closed) or are still pending L1-retest results. The highest-impact
round-2 ideas from the literature pass have now been launched.

### Round 1 outcomes (tags + decisions)

| PR | Student | Lever | Outcome |
|---|---|---|---|
| #752 | askeladd | L1 loss | **merged** (baseline 101.93) |
| #758 | tanjiro | lr=1e-3 + 10% warmup | **closed** (L1 retest +9.7% worse) |
| #749 | alphonse | Capacity 256×8 | **closed** (no convergence in 30-min budget; infra reusable) |
| #755 | frieren | slice_num 128 | **closed** (+33% epoch cost cancels per-epoch gain) |
| #760 | thorfinn | batch_size 8 | **closed** (during send-back cycle; BS=8 viable but no L1 retest landed) |
| #754 | fern | Per-channel pressure 3× | wip (L1 retest pending) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #757 | nezuko | 5% warmup + cosine | wip |
| #797 | askeladd | NaN/Inf guard (model + GT) | wip |

### Round 2 in flight (orthogonal, on top of L1)

| PR | Student | Round-2 idea | Predicted impact |
|---|---|---|---|
| #816 | alphonse | FiLM conditioning of LayerNorm (#2) | -5 to -12% |
| #818 | tanjiro | SGDR Cosine Warm Restarts (#9) | -2 to -5% |
| #819 | frieren | Relative L2 loss (per-sample norm) (#1) | -5 to -15% |
| #820 | thorfinn | Fourier PE on (x, z) coords (#3) | -4 to -10% |

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
