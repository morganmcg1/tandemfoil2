# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-28 ~23:38 (Fourier PE K=4 merged at 89.71; #816 alphonse FiLM and #819 frieren rel-L2 mix both sent back for rebase #2 onto post-#820 baseline)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`
- **Current best:** `val_avg/mae_surf_p = 89.714` (#820) and `3-split test mean = 88.16` (run `w9xbc0wl`)

## Current research focus

**Three compounding wins landed:** L1 (101.93) → +ch=[1,1,3] (99.23, −2.65%)
→ +Fourier PE K=4 (89.71, −9.59%) = **−12.0% cumulative**. This is round 2,
and the Fourier PE was the largest single-lever gain. New baseline is 89.71.
All in-flight PRs need to beat 89.71 to merge.

Key insight: Fourier PE + channel weighting are orthogonal and additive.
The ch=[1,1,3] win made pressure gradient dominant → Fourier PE's spectral
basis earned its keep on every split (including val_re_rand which was flat
on L1-only baseline). The compounding mechanism is well-understood.

### Round 1 outcomes (final)

| PR | Student | Lever | Outcome |
|---|---|---|---|
| #752 | askeladd | L1 loss | **merged** (101.93, prior baseline) |
| #754 | fern | Per-channel `p` ×3 on L1 | **MERGED — val baseline 99.23** |
| #797 | askeladd | NaN/Inf guards in evaluate_split | **MERGED — test_avg unblocked (92.61)** |
| #758 | tanjiro | lr=1e-3 + 10% warmup | **closed** (L1 retest +9.7% worse) |
| #749 | alphonse | Capacity 256×8 | **closed** (no convergence in 30-min budget) |
| #755 | frieren | slice_num 128 | **closed** (+33% epoch cost cancels gain) |
| #760 | thorfinn | batch_size 8 | **closed** (BS=8 viable but no L1 retest) |
| #753 | edward | surf_weight 20/30/50 | **closed** (sw=30 best at 125.80; superseded by ch=[1,1,3] merge) |
| #757 | nezuko | 5% warmup + cosine on L1+ch=[1,1,3] | **closed** (+3.81% worse; schedule family exhausted) |

### Round 2 in flight (orthogonal, on top of L1 + ch=[1,1,3] + Fourier K=4 = 89.71)

| PR | Student | Round-2 idea | Predicted impact | Status |
|---|---|---|---|---|
| #816 | alphonse | FiLM conditioning of LayerNorm (#2) | -5 to -12% | **rebase #1 returned val=91.82 (-7.47%) on PRE-#820 baseline; sent back for rebase #2 onto post-Fourier-PE 89.71** |
| #820 | thorfinn | Fourier PE on (x, z) coords (#3) | **−9.59%** | **MERGED — val baseline 89.71 (new best)** |
| #863 | askeladd | Seed determinism PR (infra) — replaces merged #797 | infra: variance ↓ | wip |
| #819 | frieren | Relative L2 mix α=0.5 (rebase #1 returned val=98.01 / test=88.78 on PRE-#820 base) | -1 to -4% target | **rebase #2 onto post-#820; predicted val ≤88.6 to merge** |
| #888 | fern | Stratified vol subsample by distance-to-surface — replaces #861 (closed mask-only DropConnect) | -1 to -4% | wip |
| #872 | nezuko | Domain-ID embedding 3-class (#8) — replaces #757 (closed) | -2 to -6% | wip |
| #873 | edward | EMA model weights (Polyak) — replaces #753 (closed) | -1 to -3% | wip |
| #880 | tanjiro | LinearNO ELU+1 linear attention (#6) — replaces #851 Huber (closed) | -3 to -8% | wip (needs rebase to 89.71) |
| #883 | thorfinn | Fourier bands sweep K∈{3,6,8} — follow-up to merged #820 | mapping optimum | wip |

**Closed in round 2:**
- #818 tanjiro SGDR T_0=10 → +6% worse, structural budget mismatch
  (restart fires at natural convergence epoch).
- #829 fern p-channel 5× → +3.6% worse. Optimum in (3×, 5×). Channel-
  weight scalar lever saturated at 3×; split-aware weighting would be
  the next step but is a larger redesign.
- #757 nezuko warmup retest → +3.81% worse on the new baseline.
  **Schedule lever family is now exhausted at this 30-min budget** —
  three consecutive negatives (#758 + #818 + #757).
- #753 edward surf_weight sweep → superseded by L1+ch=[1,1,3] merge.
  Best (sw=30 at 125.80) sits 27% above current baseline because runs
  predate the channel-weight merge. **`surf_weight × channel_weights`
  lever family is exhausted** — they're multiplicative, not orthogonal.
- #851 tanjiro Huber δ=1.0 → +12.6% worse (val), +10.3% worse (test).
  **Loss-shape lever family exhausted** — z-normalized residuals
  concentrate in the |err|<=1 regime where Huber's gradient shrinks,
  giving optimizer 3-10× weaker pressure on the residuals we want
  cleaned up. Direct mechanistic clash with L1+ch=[1,1,3] (whose merit
  is constant-magnitude gradient × per-channel weight).
- #861 fern uniform vol subsample keep_frac=0.15 → val +0.37% (wash),
  test −1.73% (DropConnect-style regularizer effect). Superseded by
  #820 merge (89.71 baseline) — runs against 99.23 are no longer
  competitive. Fern's per-split analysis surfaced the boundary-layer
  density hypothesis → promoted to stratified variant (#888).

**Note on rebasing after #820 merge:** All in-flight PRs (#816, #819, #861,
#863, #872, #873, #880) are based on the old 99.23 baseline. They now need
to beat **89.71** to qualify for merge. Students should rebase onto the
updated advisor branch HEAD when their run returns. Those that cannot beat
89.71 will need to be combined or redesigned for round 3.

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
- **FiLM (continuous) vs Domain-ID (categorical) conditioning** — both touch
  the regime-information story but with different mathematical mechanisms
  (multiplicative LayerNorm scale vs additive token-stream embedding). They
  may compound or one may dominate. nezuko's #872 vs alphonse's #816
  rebase will resolve this.
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
  **PARTIAL ANSWER (#816 rebase #1):** Reversed cleanly under ch=[1,1,3]
  → camber_rc became the LARGEST gain at −13.84%. Mechanistic read:
  the regression on L1-only was a loss-landscape imbalance, not a
  conditioning-amplification problem. With pressure up-weighted ×3,
  FiLM modulation correctly exploits the pressure-dominant gradient
  signal. Open question: whether this persists once Fourier PE's
  spectral basis is also in the model (rebase #2 will resolve).
- **NEW: FiLM × Fourier PE compounding** — rebase #2 of #816 will tell
  us whether the two levers are additive (predicted: yes, mechanism
  is orthogonal — input-feature spectral basis vs in-block LN scale)
  or whether Fourier PE absorbs FiLM's effective capacity. If they
  compound, val should land at ≤87. If they wash, ≈89.71. Either
  outcome is a clean appendix-worthy result.

## Notes / constraints

- `SENPAI_MAX_EPOCHS=50` and `SENPAI_TIMEOUT_MINUTES=30` are hard caps. Don't
  override them — design experiments that fit in 30 minutes wall-clock.
- All four val splits are weighted equally in the headline metric. Prefer
  changes that travel across splits over hacks that only help one.
- Test metrics are computed at the end of every run from the best-val
  checkpoint — they are the ranking quantity for the paper.
