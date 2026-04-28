# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-28 ~22:10 (round 2 mid-flight; FiLM #816 + Fourier PE #820 both sent back for rebase — both strong winners)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`
- **Current best:** `val_avg/mae_surf_p = 99.226` (PR #754, L1 + ch=[1,1,3] merged)

## Current research focus

Round 1 has fully resolved. Two compounding wins reached the baseline:
**L1 loss** (#752, 101.93) → **L1 + per-channel p×3** (#754, 99.23, −2.65%).
The other round-1 levers either did not compound on L1 (closed) or have
been sent back for L1-retest with the new baseline. All idle students are
either on round-2 hypotheses or on a continuation of the merged channel-
weight signal.

### Round 1 outcomes (final)

| PR | Student | Lever | Outcome |
|---|---|---|---|
| #752 | askeladd | L1 loss | **merged** (101.93, prior baseline) |
| #754 | fern | Per-channel `p` ×3 on L1 | **MERGED — new baseline 99.23** |
| #758 | tanjiro | lr=1e-3 + 10% warmup | **closed** (L1 retest +9.7% worse) |
| #749 | alphonse | Capacity 256×8 | **closed** (no convergence in 30-min budget) |
| #755 | frieren | slice_num 128 | **closed** (+33% epoch cost cancels gain) |
| #760 | thorfinn | batch_size 8 | **closed** (BS=8 viable but no L1 retest) |
| #753 | edward | surf_weight 20/30/50 | wip (round 1, L1+ch retest) |
| #757 | nezuko | 5% warmup + cosine on L1+ch=[1,1,3] | wip (sent back to retest on new baseline) |
| #797 | askeladd | NaN/Inf guard (model + GT) | wip (canonical fix; expanded scope) |

### Round 2 in flight (orthogonal, on top of L1 + ch=[1,1,3] = 99.23)

| PR | Student | Round-2 idea | Predicted impact | Status |
|---|---|---|---|---|
| #816 | alphonse | FiLM conditioning of LayerNorm (#2) | -5 to -12% | **rebase + rerun (96.61 on L1-only → -2.6% on new baseline expected)** |
| #820 | thorfinn | Fourier PE on (x, z) coords (#3) | -4 to -10% | **rebase + rerun (91.15 on L1-only → -8.2% on new baseline expected — strongest single signal)** |
| #851 | tanjiro | Huber loss δ=1.0 (#5) — replaces #818 SGDR (closed) | -3 to -8% | wip |
| #819 | frieren | Relative L2 loss (per-sample norm) (#1) | -5 to -15% | wip |
| #861 | fern | Volume subsampling (15%) (#7) — replaces #829 (closed) | -3 to -8% | wip |

**Closed in round 2:**
- #818 tanjiro SGDR T_0=10 → +6% worse, structural budget mismatch
  (restart fires at natural convergence epoch). Schedule lever family
  exhausted at this budget — two negatives in a row (#758 + #818).
- #829 fern p-channel 5× → +3.6% worse. Optimum in (3×, 5×). Channel-
  weight scalar lever saturated at 3×; split-aware weighting would be
  the next step but is a larger redesign.

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

## Open uncertainties (from researcher-agent + round-2 outcomes)

- Which lever contributes more: relative L2 loss vs. FiLM conditioning. They
  likely compound but the interaction is unknown.
- **Channel-weight optimum is settled** at 3× — fern's sweep showed 5× is
  past the inflection. Split-level heterogeneity (camber_rc still gains
  from more p-weight, others hurt) suggests split-aware weighting as a
  future round-3 redesign, not a round-2 lever.
- Whether a PCA basis fit on training data transfers to OOD camber splits
  (`val_geom_camber_rc`, `val_geom_camber_cruise`).
- Whether 15% volume subsampling preserves enough Re signal for
  `val_re_rand` (where the volume Re field is the regime signal).
- Whether the FiLM `val_geom_camber_rc` regression (+9.9% on alphonse's
  L1-only run) is from spurious foil-2 conditioning in raceCar samples,
  and whether it persists on the rebased baseline.

## Notes / constraints

- `SENPAI_MAX_EPOCHS=50` and `SENPAI_TIMEOUT_MINUTES=30` are hard caps. Don't
  override them — design experiments that fit in 30 minutes wall-clock.
- All four val splits are weighted equally in the headline metric. Prefer
  changes that travel across splits over hacks that only help one.
- Test metrics are computed at the end of every run from the best-val
  checkpoint — they are the ranking quantity for the paper.
