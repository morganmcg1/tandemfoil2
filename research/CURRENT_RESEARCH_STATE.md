# SENPAI Research State — willow-pai2e-r4

- **As of:** 2026-04-29 ~09:00
- **#963 MERGED** — T_max=13 schedule fix: val 81.81 → **64.91** (−20.66%), test 73.04 → **57.25** (−21.62%); largest single-PR win; run `j8yi780z`
- **#1007 CLOSED** — RFF hybrid concat clean negative (+3.14% val); projection-bandwidth-dilution; RFF-ENCODING family fully exhausted; tanjiro→#1032 mlp_ratio sweep
- **#873 CLOSED** — EMA decay=0.99 wash (+1.11% val, −0.53% test); #963 stole EMA's lunch; SNAPSHOT-AVERAGING family exhausted at T_max=13; edward→#1033 wd sweep
- **#1006 CLOSED** — n_layers=6 retest at T_max=11 clean negative (+9.16% val, +8.26% test); 30-min budget binds depth choice
- **#972 CLOSED** — 3-seed mean at T_max=50 methodology valid but obsolete regime; seed=2 beats unseeded best 4.3%; askeladd→#1022 retest at T_max=13
- **#949 CLOSED** — LayerScale γ_init=1e-4 regime-flip: −4.58% val at T_max=50 → **+9.56% val at T_max=13**; soft-start competes with cosine annealing; SOFT-START-REGULARIZATION exhausted; fern→#1026 n_head sweep
- **#929 CLOSED** — DropPath@0.1 clean negative (+9.5–19.6% val regression, budget-binding, 3 seeds)
- **Most recent human direction:** none
- **Branch:** `icml-appendix-willow-pai2e-r4`
- **Current best (unseeded):** `val_avg = 64.91`, `test_avg = 57.25` (run `j8yi780z`, PR #963, T_max=13)
- **Seeded canonical at T_max=13:** TBD — frieren #1000 (T_max sweep {10,12,13,16} at seed=0) in flight
- **Prior seeded canonical (T_max=50, obsolete for ranking):** val=85.14 (run `j1r5y758`, PR #863) — same architecture, wrong schedule regime

## Current research focus

**REFRAMING EVENT:** T_max=13 (one-line schedule fix) delivered −20.66% val /
−21.62% test in a single PR. The "still descending at timeout" pattern across
all of round 2 was a schedule artifact. **Prior architectural wins (Fourier PE
−9.6%, ch-weights −2.7%, SwiGLU −8.8%) were ALL real but measured under constant
~4.25e-4 LR, not fully annealed. At T_max=13, these stacks will compound even
further.** New projected floor (once stacks are re-tested at T_max=13): val < 50.

**Five compounding wins:** L1 (101.93) → +ch=[1,1,3] (99.23, −2.65%)
→ +Fourier PE K=4 (89.71, −9.59%) → +SwiGLU MLP (81.81, **−8.81%**)
→ +T_max=13 (64.91, **−20.66%**) = **−36.3% cumulative**.
T_max=13 is the **largest single-PR gain in the project**. Zero parameter
cost. Zero wall-clock cost. Pure schedule right-sizing.

**All in-flight PRs should use `--t_max 13` and `--seed 0` for fair comparison.
New target to beat: val=64.91 (unseeded) or val=(TBD, seeded T_max=13 from
frieren's next run).**

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

### Round 2 in flight (on top of L1 + ch=[1,1,3] + Fourier K=4 + SwiGLU + T_max=13 = **64.91**)

**All PRs must beat val=64.91 / test=57.25 to merge. All new runs must include `--t_max 13`.**

| PR | Student | Idea | Predicted impact | Status |
|---|---|---|---|---|
| #816 | alphonse | FiLM conditioning of LayerNorm | -5 to -12% | **needs rebase onto post-#963 (64.91)** |
| #820 | thorfinn | Fourier PE K=4 | −9.59% | **MERGED — val 89.71** |
| #863 | askeladd | Seed determinism (infra) | variance ↓ | **MERGED — bit-perfect; canonical `--seed 0 = 85.14` (T_max=50 era, superseded)** |
| #819 | frieren | Relative L2 α=0.5 (rebase #2) | washed +0.27% | **CLOSED — Fourier PE absorbs equalization** |
| #888 | fern | Stratified vol subsample | +3.77% actual | **CLOSED — VOLUME-MASK-SUBSAMPLING exhausted** |
| #920 | fern | Per-block coord skip-connection | +4.1% actual | **CLOSED — COORD-INJECTION exhausted** |
| #872 | nezuko | Domain-ID 3-class additive | +3.1% actual | **CLOSED — additive categorical fights LN** |
| #929 | nezuko | DropPath rate=0.1 linear | +9.5–19.6% actual | **CLOSED — budget-binding; nezuko→#1002 DropPath@0.05+T_max=13** |
| #873 | edward | EMA Polyak decay=0.99 | +1.11% val at T_max=13 (wash) | **CLOSED — #963 stole EMA's lunch; SNAPSHOT-AVERAGING exhausted; edward→#1033** |
| #880 | tanjiro | LinearNO ELU+1 | +6.92% actual | **CLOSED — attention-kernel-substitution exhausted** |
| #914 | tanjiro | SwiGLU MLP swap | **−8.81% val** | **MERGED — val 81.81** |
| #938 | tanjiro | RFF σ sweep {2,5,10} | +2.4–7.3% actual | **CLOSED — RFF-as-replacement settled negative; tanjiro→#1007 hybrid concat** |
| #939 | frieren | n_layers=6 | +4.7% actual | **CLOSED — under-trained; thorfinn→#1006 retest at T_max=11** |
| #883 | thorfinn | Fourier bands sweep K∈{3,6,8} | K=4 settled | **CLOSED** |
| #949 | fern | LayerScale γ_init=1e-4 (CaiT) | +9.56% actual at T_max=13 | **CLOSED — regime-flip; soft-start competes with cosine annealing; fern→#1026** |
| #955 | thorfinn | Per-channel output heads (Ux/Uy/p) | +2.7% actual | **CLOSED — cross-channel coupling load-bearing; DECODER-DECOUPLING exhausted** |
| #963 | frieren | Schedule-to-budget T_max=13 | **−20.66% val / −21.62% test** | **MERGED — new baseline 64.91** |
| #979 | thorfinn | Pressure-only head (decouple only p) | +0.61% val (seed noise); −2.45% 3-split test | **CLOSED — new baseline supersedes; DECODER-DECOUPLING exhausted; thorfinn→#1006** |
| #972 | askeladd | 3-seed mean canonical (T_max=50) | variance floor | **CLOSED — methodology valid, regime obsolete; askeladd→#1022** |
| #1000 | frieren | T_max sweep {10,12,13,16} at seed=0 | map LR optimum + seeded canonical | **WIP** |
| #1002 | nezuko | DropPath@0.05 + T_max=13 | +2.0% actual | **CLOSED — convergence drag; DROPPATH exhausted; nezuko→#1017** |
| #1017 | nezuko | Channel-weight re-tune: p-weight sweep {2,4,6} at T_max=13 | 0 to −3% | **WIP** |
| #1006 | thorfinn | n_layers=6 retest at T_max=11 | +9.16% actual | **CLOSED — compute-starved at 30-min; DEPTH-VIA-MORE-BLOCKS exhausted; thorfinn→#1023** |
| #1007 | tanjiro | RFF hybrid: axis-aligned + σ=10 RFF concat at T_max=13 | +3.14% actual | **CLOSED — projection-bandwidth-dilution; RFF-ENCODING exhausted; tanjiro→#1032** |
| #1022 | askeladd | 3-seed mean at T_max=13 (paper-facing canonical, current regime) | std ↓ vs T_max=50 | **WIP (just assigned)** |
| #1023 | thorfinn | Peak LR sweep at T_max=13: lr ∈ {3e-4, 5e-4, 7.5e-4, 1e-3} | 0 to −7% | **WIP (just assigned)** |
| #1026 | fern | n_head sweep at T_max=13: n_head ∈ {2, 4, 8} | 0 to −3% | **WIP** |
| #1032 | tanjiro | mlp_ratio sweep at T_max=13: ratio ∈ {1, 2, 3, 4} | 0 to −7% | **WIP (just assigned)** |
| #1033 | edward | weight_decay sweep at T_max=13: wd ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2} | 0 to −5% | **WIP (just assigned)** |

**Round 3 candidates (queued):**

⚠️ **ALL round-3 experiments MUST include `--t_max 13` and beat val=64.91.**

- **T_max sweep {10,12,13,16} at seed=0** — IN FLIGHT as frieren #1000. Maps
  LR-schedule optimum AND establishes seeded T_max=13 canonical. Predicted:
  T_max ∈ {12, 13} near-optimal; T_max=10 cuts too early; T_max=16 under-anneals.
- **n_layers=6 retest at T_max=11** — **CLOSED #1006 +9.16% val / +8.26% test.**
  Schedule fix worked (LR landed 0 at E11); model still descending at E11 with
  steeper-than-expected slope → compute-starved. 30-min budget binds depth.
  DEPTH-VIA-MORE-BLOCKS exhausted under current compute envelope.
- **Peak LR sweep at T_max=13** — **IN FLIGHT as thorfinn #1023.** Sweep
  lr ∈ {3e-4, 5e-4, 7.5e-4, 1e-3} at fixed T_max=13, seed=0. Tests if peak LR
  was tuned to T_max=50 era (lr=5e-4 inherited from pre-#820). Predicted:
  lr=7.5e-4 likely winner via "schedule-LR product is constant" heuristic
  (lr × √T_max conserves cumulative-step magnitude); lr=1e-3 speculative
  high-end. Orthogonal to frieren #1000 (varies T_max at fixed lr=5e-4).
- **3-seed mean at T_max=13** — **IN FLIGHT as askeladd #1022.** Reissue of
  closed #972 under post-#963 regime. Predicted val mean ≈ 63-67, std < 3.74
  (proper convergence tightens basin selection). Produces actual paper-facing
  canonical for current era; replaces "unseeded best 64.91" in BASELINE.md.
- **EMA × T_max=13 compound** (#873 needs rebase onto post-#963). EMA −10.46%
  on pre-#820; T_max=13 −20.66%. If orthogonal, compound val ≈ 44-50. **Highest-
  priority architectural stack candidate.**
- **FiLM × T_max=13** (#816 needs rebase onto post-#963). Bar is 64.91.
- **EMA + FiLM + T_max=13 triple stack** — if both land, combine.
- **LayerScale × T_max=13** — **CLOSED #949 +9.56% val.** Regime-flip: won
  under T_max=50 (−4.58%) but soft-start γ_init=1e-4 competes with cosine
  annealing under T_max=13 (double-dampened early epochs). SOFT-START-
  REGULARIZATION lever family exhausted. Key finding: γ_mlp is regime-robust
  and uniform across blocks; γ_attn depth-grading dissolves without long
  lukewarm-LR runway. Fern reassigned to n_head sweep (#1026).
- **n_head sweep at T_max=13** — **IN FLIGHT as fern #1026.** n_head ∈ {2, 4, 8}
  at seed=0, T_max=13. Zero param change. Tests "fewer wider vs more narrower
  heads" under post-#963 schedule. n_head=4 baseline sanity-check included.
  Predicted: ±0-3% on val_avg. Orthogonal to thorfinn #1023 (LR sweep) and
  frieren #1000 (T_max sweep).
- **DropPath@0.05 × T_max=13** — CLOSED (#1002, +2.0% regression). Lever family DROPPATH-AT-N-LAYERS-5 exhausted. n_layers=6 retest also closed (#1006), so no headroom to revisit.
- **Channel-weight re-tune at T_max=13** — IN FLIGHT as nezuko #1017. ch=[1,1,3] was tuned under T_max=50; p-weight sweep {2,4,6} at T_max=13 re-validates.
- **RFF hybrid concat (axis-aligned + σ=10 RFF)** — **CLOSED #1007 +3.14% val / +1.81% test.** Projection-bandwidth-dilution: cruise/re_rand replacement gain didn't transfer to concat; RFF channels inject high-freq noise model can't gate away in 13 epochs. **RFF-ENCODING family fully exhausted** (replacement + concat).
- **mlp_ratio sweep at T_max=13** — **IN FLIGHT as tanjiro #1032.** ratio ∈ {1, 2, 3, 4} at seed=0. mlp_ratio=2 (current) may be under-capacity vs typical transformer (4). Fern's #949 γ_mlp uniformity finding suggests FFN saturation. Predicted winner: mlp_ratio=3.
- **weight_decay sweep at T_max=13** — **IN FLIGHT as edward #1033.** wd ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}. Generic default 1e-4 never tuned. Under T_max=13, cosine decay reduces cumulative WD contribution → higher WD may compensate. Predicted winner: wd=1e-3.
- **EMA decay sweep** at T_max=13 — **CANCELLED.** EMA decay=0.99 washed (#873 v2); regime absorption mechanism rules out all decay values.
- **Categorical-FiLM (multiplicative)** at T_max=13 — round-3 follow-up once #816 lands.
- **RFF σ sweep {2, 5}** — in flight as #938 rebase #2 (compare vs T_max=13 baseline).
- **Hybrid axis-aligned + RFF concatenation** (round-3 candidate; only if
  RFF-as-replacement fails on σ∈{2,5}).
- **Pressure-only head** at T_max=13 (rebase #979 once #963 merges). Per-#955
  analysis, mechanism may work better with proper convergence — a positive
  result at T_max=13 would settle the DECODER-DECOUPLING family.
- **Per-channel output heads (full Ux/Uy/p decoupling)** — **CLOSED #955
  +2.7% val / +4.1% test**. Mechanism active (head_p specialized ×2.22 vs
  Ux ×1.57) but cross-channel coupling was load-bearing for single_in_dist
  generalization. Lever family DECODER-PER-CHANNEL-FULL exhausted.
- **Pressure-only head decoupling** — **NEW round-3 candidate, IN FLIGHT
  as #979 (thorfinn)**. Decouple ONLY head_p; keep Ux+Uy on shared
  2-channel head. Tests whether pressure-specific capacity benefit can
  be preserved without Ux/Uy over-specialization. +16K params (half of
  #955's footprint). If this also fails, broader DECODER-DECOUPLING
  family is settled.
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
