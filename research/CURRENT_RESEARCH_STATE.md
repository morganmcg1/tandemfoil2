# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-29 ~02:55 (**SwiGLU unseeded best 81.81 / test 73.04** holds; **#863 MERGED — canonical seeded baseline `--seed 0 = 85.14`, bit-perfect determinism on single-GPU**; #938 sent back for σ∈{2,5} sweep (RFF σ=10 +2.4%, mechanism diagnosed: no low-freq channel); #939 closed (frieren n_layers=6 +4.7% under-trained), frieren→#963 schedule-to-budget T_max=13; #883 closed (Fourier K-axis settled), thorfinn→#955 per-channel heads; #920 closed (coord-skip), fern→#949 LayerScale; **#873 EMA is the predicted next BIG compounded winner**, awaiting rebase; askeladd→3-seed mean follow-up assigned. **NEW THEME: schedule-budget mismatch from #939 affects ALL pre-#914 PRs that were "still descending" at timeout. NEW INFRA: borderline-ablation contract — PRs with <2% predicted Δ must run `--seed {0,1,2}` mean ± std.**)
- **Most recent human direction:** none yet for this track
- **Branch:** `icml-appendix-willow-pai2e-r4`
- **Current best (unseeded):** `val_avg/mae_surf_p = 81.81` (#914), `test_avg = 73.04` (4-split, finite), `3-split test mean = 81.28` (run `2akpdg9t`)
- **Seeded canonical (`--seed 0` PR-to-PR ranking):** `val_avg = 85.14`, `test_avg = 78.97`, `3-split = 86.78` (run `j1r5y758`, post-#914+#863)

## Current research focus

**Four compounding wins:** L1 (101.93) → +ch=[1,1,3] (99.23, −2.65%)
→ +Fourier PE K=4 (89.71, −9.59%) → +SwiGLU MLP (81.81, **−8.81%**)
= **−19.7% cumulative**. SwiGLU is the second-largest single-lever gain.
New baseline: **val=81.81 / test=73.04 (4-split, first finite)**.
All in-flight PRs need to beat **81.81** to merge.

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

### Round 2 in flight (on top of L1 + ch=[1,1,3] + Fourier K=4 + SwiGLU = 81.81)

**All PRs must beat val=81.81 / test=73.04 to merge.**

| PR | Student | Idea | Predicted impact | Status |
|---|---|---|---|---|
| #816 | alphonse | FiLM conditioning of LayerNorm | -5 to -12% | **needs rebase onto post-#914 (81.81)** |
| #820 | thorfinn | Fourier PE K=4 | −9.59% | **MERGED — val 89.71** |
| #863 | askeladd | Seed determinism (infra) | variance ↓ | **MERGED — bit-perfect; canonical `--seed 0 = 85.14`; askeladd→3-seed mean follow-up** |
| #819 | frieren | Relative L2 α=0.5 (rebase #2) | washed +0.27% | **CLOSED — Fourier PE absorbs equalization; camber_rc −4.59% orthogonal** |
| #888 | fern | Stratified vol subsample | +3.77% actual | **CLOSED — VOLUME-MASK-SUBSAMPLING exhausted** |
| #920 | fern | Per-block coord skip-connection | -1 to -3% predicted | **CLOSED +4.1% val (vs pre-#914) / +14.4% vs post-#914; mechanism active but cruise/re_rand regress sharply; COORD-INJECTION exhausted** |
| #872 | nezuko | Domain-ID 3-class additive | +3.1% actual | **CLOSED — additive categorical fights LN** |
| #929 | nezuko | DropPath rate=0.1 linear | -1 to -3% | **wip (needs rebase onto #914)** |
| #873 | edward | EMA Polyak decay=0.99 | −10.46% val on pre-#820 | **needs rebase onto post-#914; predicted compound val ≈ 70-75** |
| #880 | tanjiro | LinearNO ELU+1 | +6.92% actual | **CLOSED — attention-kernel-substitution exhausted** |
| #914 | tanjiro | SwiGLU MLP swap | **−8.81% val, test 73.04** | **MERGED — new baseline 81.81** |
| #938 | tanjiro | Random Fourier Features σ=10 | -1 to -3% predicted | **rebase #1 +2.4% val (83.76); SENT BACK for σ∈{2,5} sweep — student diagnosed coord-range mismatch (σ=10 has no low-freq channel)** |
| #939 | frieren | n_layers=6 (one extra block) | -1 to -3% predicted | **CLOSED +4.7% val (85.65); under-trained at timeout (11/11 epochs), not under-capacity; CAPACITY-WITHIN-BUDGET-DEPTH-AXIS exhausted at this budget** |
| #883 | thorfinn | Fourier bands sweep K∈{3,6,8} | mapping optimum | **CLOSED — K=8 −0.43% noise-bounded; K=4 settled; FOURIER-BAND-SWEEP-AXIS-ALIGNED exhausted** |
| #949 | fern | LayerScale γ_init=1e-4 (CaiT) | -1 to -3% | wip |
| #955 | thorfinn | Per-channel output heads (Ux/Uy/p) | -2 to -4% | wip |
| #963 | frieren | Schedule-to-budget T_max=13 (cosine match horizon) | -1 to -3% | wip (just assigned, post-#914) |
| #972 | askeladd | 3-seed mean canonical (--seed {0,1,2}) | replace single-seed noise → tight ranking quantity | wip (just assigned, post-#863+#914) |

**Round 3 candidates (queued, contingent on round 2 outcomes):**

- **EMA × SwiGLU compound** (#873 rebased onto post-#914 baseline 81.81).
  EMA gave −10.46% on pre-#820; SwiGLU gave −8.81%. Predicted compound
  val ≈ 70-75. **Highest-priority pending experiment.**
- **FiLM × SwiGLU compound** (#816 rebased onto post-#914). Predicted
  val ≈ 72-76 if orthogonal. Bar is 81.81.
- **EMA + FiLM + SwiGLU triple stack** — if both #873 and #816 land,
  combine. Three orthogonal levers. Predicted val ≈ 63-70.
- **LayerScale × SwiGLU** (#949 in flight). Per-channel residual gate
  vs per-feature FFN gate — should be orthogonal. Predicted val ≈ 79-81
  if it works alone. Round-3 candidate to stack with EMA/FiLM if all land.
- **LayerScale × DropPath** stack (#949 + #929) — deterministic γ-soft-start
  + stochastic block dropping. CaiT-XL precedent. Round-3 only if both work
  alone first.
- **EMA decay sweep {0.995, 0.97}** — once #873 merges on SwiGLU.
- **Categorical-FiLM (multiplicative)** — round-3 follow-up once #816 lands.
- **n_layers=4** (the opposite direction from #939) — at ~115 s/epoch
  → ~15 epochs in budget. Tests whether depth-budget tradeoff favors
  shallow side. Counterfactual to #939.
- **n_layers=6 with extended 60-min budget** (round-3 retest if budget
  envelope lifts) — disentangle "depth doesn't help" from "depth needs
  more steps." #939's result strongly suggests this would land negative-Δ.
- **Width-scaling: n_hidden=144 at n_layers=5** — iso-param to n_layers=6
  (~+170K params) but with no extra block depth. Tests width vs depth at
  same capacity. Round-3 candidate.
- **RFF σ sweep {2, 5}** — in flight as #938 rebase #2 (skipping σ=20 per
  student's coord-range diagnosis). σ=2 matches baseline's k=0 octave;
  σ=5 fills the gap between baseline's π and 4π bands. If both lose,
  RFF-as-replacement is settled and we move to hybrid axis-aligned+RFF
  for round 3.
- **Hybrid axis-aligned + RFF concatenation** (round-3 candidate; tanjiro's
  follow-up #2 from #938 rebase #1). Keep the 4 axis-aligned octaves AND
  add 8 random freqs at σ=10 on top. Tests whether RFF can *augment*
  rather than *replace*. Only if RFF-as-replacement fails on σ∈{2,5}.
- **Per-channel output heads** (Ux/Uy/p separate decoder MLP). Decouples
  decoder pathways for the channel-weighted [1,1,3] target distribution.
  Predicted -2 to -4%. **In flight: thorfinn #955 (post-#914 baseline).**
- **Gated coord-skip with σ=0 init** (round-3 follow-up to closed #920) —
  sigmoid gate on per-block coord re-injection. Lets the model self-suppress
  on splits where the skip hurts. Likely redundant with SwiGLU's per-feature
  gating; deprioritize unless SwiGLU's gates show coord-relevant features.
- **GeGLU ablation** (F.silu → F.gelu in gate). ±0.5%, confirms paper
  citation choice. Low-priority.
- **Multi-seed mean** — once #863 merges, seeds {0,1,2} for ≤1% ablations.
  Askeladd's queued post-merge assignment.

**Closed in round 2:**
- #939 frieren n_layers=6 → val +4.7% (85.65), test +4.0% (75.99). The
  +18% wall-clock per epoch cost LR-schedule progress: model ran 11
  epochs vs baseline 13 and was still aggressively descending (epoch-10
  → 11 dropped val by 11.7 points, far above mean per-epoch improvement).
  Pattern: **under-trained, not under-capacity.** All-splits-regress is
  consistent with mid-training, not wrong-inductive-bias. Only signal
  in correct direction: test_geom_camber_rc −0.5% (cross-foil reasoning
  marginal gain). **Lever family CAPACITY-WITHIN-BUDGET-DEPTH-AXIS
  exhausted at 30-min budget** — extended-budget retest deferred to
  round 3.

  **Surfaced theme: schedule-budget mismatch.** Cosine T_max=50 with
  13-epoch budget means LR never decays below 65% of peak — affects
  ALL current PRs that hit timeout. frieren reassigned PR #963 to fix.
- #883 thorfinn Fourier bands sweep K∈{3,6,8} → K=8 −0.43% val (89.32),
  −1.40% test 3-split (86.93); K=3, K=6 both regress vs K=4. K=8 win
  is single-epoch variance at the timeout cliff (val 94→116→89 in
  epochs 12-14). All runs were on pre-#914 baseline; predicted
  carry-forward to post-#914 lands within seed noise of current 81.28
  test 3-split — not a baseline-update candidate. Useful negative side
  result: **K=8's ~452 cycles/domain at highest band did NOT destabilize
  training**, evidence that high-σ random Fourier features (tanjiro
  #938) won't explode. Cruise val/test divergence recurred across all
  K values — split-design flag for round-3 retrospective. **Lever family
  FOURIER-BAND-SWEEP-AXIS-ALIGNED settled at K=4.**
- #920 fern per-block coord skip-connection → val +4.1% (vs pre-#914) /
  +14.4% (vs post-#914). Mechanism active: block norms grew uniformly to
  1.12-1.25 across all 5 blocks (refutes the "depth dilutes spatial signal"
  framing). Per-split asymmetry: high-Re raceCar splits (single_in_dist
  −2.3%, camber_rc −1.1%) marginally improved; low-Re/global-flow splits
  (cruise +15.5%, re_rand +10.8%) regressed sharply. Re-injecting Fourier
  features at every block over-emphasizes spatial-frequency at the cost of
  low-frequency global features. Same split asymmetry as #888 (vol-mask).
  Post-#914, redundant with SwiGLU's per-feature gating. **Lever family
  COORD-INJECTION exhausted.**
- #819 frieren Relative L2 α=0.5 (rebase #2) → val +0.27% (washed), test
  +2.2%. Fourier PE absorbs the per-sample gradient equalization on heavy-tail
  samples. camber_rc exception: −4.59% (absolute-loss gradient on
  geometric-outlier samples is orthogonal to Fourier PE). **Lever family
  LOSS-FUNCTION-EQUALIZATION exhausted.**
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

**Note on rebasing after #914 merge:** Round-2 in-flight PRs based on
pre-SwiGLU baselines (#816 FiLM, #863 seed determinism, #873 EMA,
#929 DropPath) need rebase onto post-#914 advisor HEAD — they all need
to beat **81.81** to qualify for merge. PRs assigned post-#914 (#938
RFF, #939 n_layers=6, #949 LayerScale, #955 per-channel heads) are
already on the current baseline. Closed in round 2 from
weak-but-mechanistically-clean negative results: #920 fern coord-skip
(+4.1% on pre-#914; mechanism actively biased away from low-frequency
splits) and #883 thorfinn Fourier sweep (K=8 −0.43% noise-bounded by
timeout cliff; K-axis settled at K=4).

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
