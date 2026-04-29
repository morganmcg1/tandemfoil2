# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-29 ~00:25 (Fourier PE K=4 merged at 89.71; 4 PRs in rebase queue; #880, #888, #872 closed; tanjiro→#914 SwiGLU, fern→#920 coord-skip, nezuko→#929 DropPath rate=0.1; **#873 EMA is the predicted next big winner** at val=88.85/test=78.67 on pre-#820 base, awaiting compound-test on post-#820)
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
| #863 | askeladd | Seed determinism PR (infra) — bit-perfect proved (0.0000 drift) | infra: variance ↓ | **rebase + 1 canonical-baseline run on post-#820, then merge** |
| #819 | frieren | Relative L2 mix α=0.5 (rebase #1 returned val=98.01 / test=88.78 on PRE-#820 base) | -1 to -4% target | **rebase #2 onto post-#820; predicted val ≤88.6 to merge** |
| #888 | fern | Stratified vol subsample by distance-to-surface | -1 to -4% predicted; +3.77% actual | **CLOSED — VOLUME-MASK-SUBSAMPLING lever family exhausted** |
| #920 | fern | Per-block coordinate skip-connection (Fourier feat re-injection at each block) | -1 to -3% | wip (just assigned) |
| #872 | nezuko | Domain-ID embedding 3-class (#8) — replaces #757 (closed) | -2 to -6% predicted; +3.1% actual | **CLOSED — additive categorical fights LN; categorical-FiLM as round-3 follow-up** |
| #929 | nezuko | DropPath / Stochastic Depth at rate 0.1 linear (replaces #872 close) | -1 to -3% | wip (just assigned) |
| #873 | edward | EMA model weights (Polyak decay=0.99) — predicted -1 to -3%, ACTUAL **-10.46% val / -15.06% test** on pre-#820 baseline | -1 to -3% predicted; **4-5× actual** | **rebase to post-#820; predicted compounded val ≈ 80-83 / test ≈ 70-73 if compounds with Fourier PE** |
| #880 | tanjiro | LinearNO ELU+1 linear attention | -3 to -8% predicted; +6.92% actual | **CLOSED — attention-kernel-substitution lever family exhausted at S=64** |
| #914 | tanjiro | SwiGLU MLP swap (replaces #880 close) | -1 to -3% | wip (just assigned) |
| #883 | thorfinn | Fourier bands sweep K∈{3,6,8} — follow-up to merged #820 | mapping optimum | wip |

**Round 3 candidates (queued, contingent on round 2 outcomes):**

- **EMA decay sweep at 0.995 and 0.97** — once #873 merges, the
  optimum decay value is unknown. Edward's analysis: 0.995 likely
  best (longer effective horizon ~200 steps); 0.97 likely worse
  but maps the slope. Cheap follow-up.
- **EMA × FiLM stack** — once both #873 and #816 (FiLM) merge,
  stack them. Mechanisms are fully orthogonal (snapshot averaging
  × in-block LN modulation). Predicted target: val ≈ 75-80.
- **SwiGLU × EMA** — once #914 (tanjiro SwiGLU) and #873 (EMA)
  both land, stack them. Both small-impact regularizers in
  different parts of the pipeline.
- **Multi-seed mean for borderline ablations** — once #863 merges,
  use `--seed {0, 1, 2}` for ablations with predicted ≤1% effect.
- **Per-channel relative L2 norm** (frieren's #2 follow-up) —
  compute `y_rms_per_sample` per channel instead of jointly.
  Pressure dominates the joint computation due to channel weight
  3×; per-channel norm could be a different lever from α-mix.

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
- #880 tanjiro LinearNO ELU+1 → +6.92% val, +3.23% test. **Attention-
  kernel-substitution lever family exhausted at S=64.** ELU+1 features
  become near-uniform at small S, collapsing slice-pair information
  flow. Cao 2021 Galerkin precedent applies to full-token (large N)
  attention, not slice-token (small S=64). cruise/single splits where
  pressure dynamics span small numeric ranges regressed +11-15%.
  Useful negative result: peak focus is load-bearing for slice
  attention.
- #861 fern uniform vol subsample keep_frac=0.15 → val +0.37% (wash),
  test −1.73% (DropConnect-style regularizer effect). Superseded by
  #820 merge (89.71 baseline) — runs against 99.23 are no longer
  competitive. Fern's per-split analysis surfaced the boundary-layer
  density hypothesis → promoted to stratified variant (#888).
- #888 fern stratified vol subsample (BL Gaussian σ=0.05) → val
  +3.77%, test +6.20%. 3 of 4 val splits regressed; only camber_rc
  improved (−9.2%, high-Re raceCar — BL story holds for that one
  regime). BL-canary failed in opposite direction (single_in_dist
  +11.6%). **Far-field volume nodes carry real supervision** beyond
  DropConnect regularization (mean-field anchor, regularization,
  Re/AoA context). CFD-mesh BL refinement made σ=0.05 milder than
  predicted (~29% vol drop, not ~85%). **Lever family
  VOLUME-MASK-SUBSAMPLING exhausted** — future surface-focus
  experiments should use loss-side levers, not mask-side surgery.
- #872 nezuko Domain-ID embedding 3-class additive → val +3.1%,
  test_avg tied. `val_re_rand` +9.87% is the cleanest regression
  signal — the split where regime info should help MOST, hurt
  MOST. Mechanism: additive shift on `fx` propagates through 5
  blocks of attention; LayerNorm only partially undoes it after
  first attention pass already disturbed. Single-class embedding
  norm 0.20 vs tandem 0.52 → model didn't find single-foil to
  need a special channel (the `x[:, :, 18:24]` zeros already
  encode "no foil 2"). **Partial-signal:** test_single_in_dist
  −10.4% suggests regime info CAN help in the right form.
  **Categorical-FiLM (multiplicative) is the round-3 follow-up**
  if alphonse's #816 (continuous FiLM) lands. Lever family
  ADDITIVE-CATEGORICAL-CONDITIONING exhausted; multiplicative
  variants remain candidates.

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
