# SENPAI Research Results — willow-pai2e-r4

## 2026-04-29 00:25 — PR #872: Domain-ID embedding (3-class additive) — **CLOSED (regression on 3 of 4 val splits)**

- Branch: `willowpai2e4-nezuko/domain-id-embedding`
- Student: willowpai2e4-nezuko
- W&B run: per PR comments

**Hypothesis.** Derive a 3-class domain ID at runtime from the
global scalars (single / tandem-RC / tandem-cruise), embed via a
zero-init `[3, n_hidden]` lookup table, and add to the per-node
feature stream after preprocess. Predicted −2 to −6%; mechanism
distinct from FiLM (continuous multiplicative) — categorical
additive shift to encode regime discontinuity.

**Implementation deviation handled correctly.** The student found
the spec'd 5.0-rad threshold (≈286°) was degenerate post-z-norm
and self-corrected to a data-grounded discriminator
`abs(aoa1-aoa2) < 1e-4`. Diagnostic counts confirmed sane class
balance across batches. Bug was caught before any wasted training.

**Results vs current baseline 89.71/88.16 (post-#820 Fourier PE):**

| Metric | Baseline | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 89.71 | ~92.5 | **+3.1%** ✗ |
| `test_avg` (3-split) | 88.16 | ~88.0 | tied |

**Per-split val:**

| Split | Δ |
|---|---:|
| `val_re_rand` | **+9.87% ✗** (largest regression — exactly where regime info should help most) |
| `val_single_in_dist` | mild regression |
| `val_geom_camber_rc` | mild regression |
| `val_geom_camber_cruise` | flat-ish |

**Per-split test (best signal):**
- `test_single_in_dist`: **−10.4% ✓** (regime info CAN help in right form)
- Other test splits: mixed-to-negative

**Mechanistic post-mortem:**

The additive shift on `fx` injects a bias that the rest of the
network can't easily down-weight when the regime label is unhelpful.
LayerNorm inside each TransolverBlock partially undoes it but only
after the first attention pass has already been disturbed. Across
5 blocks the additive shift propagates and amplifies in unhelpful
directions on splits where the embedding doesn't carry useful
information. The single-foil class barely got used (norm 0.20 vs
~0.52 for tandem) — the model learns "no foil 2" from
`x[:, :, 18:24]` (zeroed foil-2 globals) directly through the
preprocess MLP. The categorical embedding was redundant in a way
the additive design couldn't take advantage of.

**Three takeaways for the appendix:**
1. **Additive categorical conditioning fights LayerNorm.**
   Additive shifts propagated through 5 blocks of attention amplify
   in unhelpful directions. `val_re_rand` +9.87% is the cleanest
   evidence: the split where regime info should help most, hurt most.
2. **Single class barely got used (norm 0.20 vs ~0.52 for tandem).**
   Negative signal that the model didn't find single-foil to need a
   special channel — the `x[:, :, 18:24]` zeros already do that work.
3. **`test_single_in_dist` improved −10.4%** — partial signal that
   regime info CAN help in the right form. Hypothesis isn't dead;
   delivery mechanism was wrong. **Categorical-FiLM as round-3
   stack** is the natural follow-up if alphonse's #816 (continuous
   FiLM) lands: same regime-anchoring effect with a multiplicative
   form that the model can down-weight when not helpful.

**nezuko reassigned PR #929: DropPath / Stochastic Depth (rate 0.1
linear schedule across 5 blocks).** Implicit ensembling regularizer,
fully orthogonal to EMA (#873) and FiLM (#816). Zero parameter cost,
zero inference cost. Predicted −1 to −3%, with largest gains on the
OOD camber splits where overfitting is the failure mode.

## 2026-04-29 00:05 — PR #888: Stratified volume subsample (BL Gaussian, σ=0.05) — **CLOSED (regression on 3 of 4 splits)**

- Branch: `willowpai2e4-fern/stratified-vol-subsample`
- Student: willowpai2e4-fern
- W&B run: [`gcectu4h`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/gcectu4h)

**Hypothesis.** Stratify volume keep-probability by Gaussian
distance-to-surface (BL nodes ~1.0 keep, far-field 0.10 floor).
Predicted −1 to −4% via concentrating supervision on the
boundary-layer nodes that drive surface pressure.

**Results vs current baseline 89.71/88.16 (post-#820 Fourier PE):**

| Metric | Baseline | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 89.71 | 93.09 | **+3.77%** ✗ |
| `test_avg` (4-split, finite!) | NaN→88.16 (3-split) | 85.18 / 93.62 (3-split) | **+6.20%** ✗ |
| Best epoch | 14 | 13 (timeout, +8% epoch cost) | -1 |

**Per-split val:**

| Split | Δ |
|---|---:|
| `val_single_in_dist` | **+11.6% ✗** (BL canary FAILED in opposite direction) |
| `val_geom_camber_rc` | **−9.2% ✓** (only winner; high-Re raceCar) |
| `val_geom_camber_cruise` | +11.9% ✗ |
| `val_re_rand` | +4.3% ✗ |

**Diagnostics** (intervention working as designed):
- BL keep frac (d<0.1): ~0.86 (predicted ~1.0) ✓
- Far keep frac (d>0.5): ~0.10 (predicted floor 0.10) ✓
- `effective_surf_frac`: ~0.025 — **10× lower than predicted 0.20-0.25**

**Why effective_surf_frac came in 10× low:** CFD meshes are
already heavily BL-refined, so most volume nodes are within d<0.1
of the surface and get high keep_prob anyway. σ=0.05 only really
culls a small far-field tail. The intervention dropped only ~29%
of volume nodes, not the predicted ~85%.

**Three takeaways for the appendix:**
1. **Far-field volume nodes are not just "DropConnect-style
   regularization."** They provide a mean-field anchor, gradient
   noise that prevents over-fitting near-surface oscillations,
   and Re/AoA regime context. Concentrating supervision on BL
   trades all of that for localized refinement.
2. **CFD-mesh structure invalidates "uniform subsample"
   PR-design intuition.** Subsampling fractions can't be reasoned
   about independent of mesh distribution.
3. **`val_geom_camber_rc` (−9.2%) is the single-split signal that
   the BL story holds for ONE regime** (high-Re unseen-camber
   raceCar tandem, where near-surface pressure dominates the
   error budget). Doesn't generalize.

**Lever family VOLUME-MASK-SUBSAMPLING exhausted for round 2.**
Both uniform (#861) and stratified (#888) closed. Future "surface
focus" experiments should use loss-side levers (surf_weight,
ch_weights, surface oversampling at data loader level) where the
merged #754 already established the clean direction.

**fern reassigned PR #920: per-block coordinate skip-connection**
— Re-inject Fourier-encoded coords at each Transolver block.
Mechanistically orthogonal to FiLM (#816, global-scalar LN
modulation) and to one-time Fourier PE (input only). Zero-init for
safety. Predicted −1 to −3%, +10K params, <2% wall-clock cost.

## 2026-04-28 23:50 — PR #880: LinearNO ELU+1 linear attention — **CLOSED (clear regression)**

- Branch: `willowpai2e4-tanjiro/linear-attention-elu1`
- Student: willowpai2e4-tanjiro
- W&B run: [`clwoyuwq`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/clwoyuwq)

**Hypothesis.** Replace softmax slice attention with ELU+1 linear
attention. Predicted −3 to −8% on val_avg/mae_surf_p via richer
slice-pair information flow.

**Results vs prior baseline 99.23 (pre-#820, L1+ch=[1,1,3]):**

| Metric | Prior baseline | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.23 | 106.10 | **+6.92%** ✗ |
| `test_avg/mae_surf_p` | 92.61 | 95.60 | **+3.23%** ✗ |
| Param count | 662 K | 662 K | parity (+0) |
| Wall clock | parity | parity | parity |

**Per-split val:**

| Split | Δ |
|---|---:|
| `val_single_in_dist` | +11.0% ✗ |
| `val_geom_camber_rc` | −0.5% ≈ |
| `val_geom_camber_cruise` | **+15.1% ✗** |
| `val_re_rand` | +4.3% ✗ |

**Mechanism (Qin et al. 2022 attention dilution):** At S=64 with
d_head=32, ELU+1 features become near-uniform, collapsing slice-
pair information flow. Softmax's competitive sharpness was
load-bearing for cruise/single splits where pressure dynamics
span small numeric ranges and need crisp slice differentiation.

**Three takeaways for the appendix:**
1. Attention-kernel substitution is bounded by inductive-bias
   requirements at small S. Cao 2021 Galerkin precedent applies
   to *full-token* attention (large N), not slice-token (small S).
2. Per-split signature confirms the dilution prediction —
   smallest-magnitude splits (cruise, single) regress hardest.
3. `test_single_in_dist` improvement (−1.7%) suggests uniform
   smoothing helps tail outliers, but cost of losing peak focus
   dominates everywhere else. MoE-style "linear-attention at
   high-magnitude regions, softmax elsewhere" is a future direction.

Implementation was correct (param parity, finite grads, healthy
training). Failure was inductive-bias mismatch, not engineering.
**Lever family ATTENTION-KERNEL-SUBSTITUTION exhausted at S=64.**

## 2026-04-28 23:48 — PR #873: EMA Polyak averaging decay=0.99 (val=88.85, test=78.67) — **SENT BACK for rebase onto post-#820, predicted to merge as biggest single-lever win**

- Branch: `willowpai2e4-edward/ema-weights-polyak`
- Student: willowpai2e4-edward
- W&B run: [`6vl5yguo`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/6vl5yguo)

**Hypothesis.** EMA model weights with decay=0.99, eval and save
checkpoint from EMA-swapped weights. Predicted −1 to −3%.

**Results vs PRIOR baseline 99.23 (pre-#820, L1+ch=[1,1,3]):**

| Metric | Prior baseline | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.226 | **88.846** | **−10.46%** |
| `test_avg/mae_surf_p` | 92.610 | **78.669** | **−15.06%** |
| Best epoch | 12 | 14 (timeout) | +2 |
| Wall clock | 30.77 min | 30.74 min | flat |

**Per-split val (epoch 14, EMA weights):**

| Split | Prior | This run | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 116.68 | 109.08 | **−6.51%** |
| `val_geom_camber_rc` | 113.94 | **100.05** | **−12.18%** |
| `val_geom_camber_cruise` | 75.02 | **64.75** | **−13.68%** |
| `val_re_rand` | 91.28 | **81.49** | **−10.72%** |
| **val_avg** | 99.23 | 88.85 | **−10.46%** |

**Per-split test (final eval from saved EMA-best checkpoint):**

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 96.33 |
| test_geom_camber_rc | 88.72 |
| test_geom_camber_cruise | 54.52 |
| test_re_rand | 75.11 |
| **test_avg** | **78.67** |

**vs CURRENT merged baseline (89.71/88.16, post-#820 Fourier PE):**
val=88.85 already sits **−0.96% below**. test=78.67 is **−10.77%
below 88.16**. Even if Fourier PE absorbs half the EMA gain, the
PR merges. PR mergeable=UNKNOWN (likely no real conflict given
EMA touches optimizer/eval, Fourier touches input).

**4-5× larger effect than predicted.** Three observations from
edward's analysis:
1. **val→test tightening (+6.6 → +10.2 delta widening)** — EMA's
   averaged minimum is *more* generalizing, not just better-on-
   selection-target. Implicit-regularization signal is clean.
2. **Per-split prediction was wrong (informatively).** Predicted
   `val_single_in_dist` to gain most (heavy-tail jitter). Actual:
   smallest gain (−6.5%) on that split, biggest gains on
   geometry-extrapolation (cruise/rc) and Re-random (−12 to −14%).
   EMA's gain is dominated by *generalization-direction smoothing*,
   not batch-noise variance reduction.
3. **Strictly monotonic per-epoch curve.** Every single epoch
   improved on the previous. No jitter at all. Best-epoch shift
   +2 (12 → 14) consistent with EMA needing warmup.

**Decision.** **Send back for rebase onto post-#820 advisor branch
HEAD.** Mechanistically orthogonal to Fourier PE (snapshot
averaging vs input feature spectrum). Predicted compounded:
val ≈ 80–83, test ≈ 70–73. **Strongest single-lever result of
round 2.** Sent back at 23:48 with detailed instructions.

## 2026-04-28 23:45 — PR #863: Seed determinism (askeladd, BIT-PERFECT achieved) — **SENT BACK for rebase + canonical seeded baseline run**

- Branch: `willowpai2e4-askeladd/seed-determinism`
- Student: willowpai2e4-askeladd
- W&B runs: [`sk040lf3`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/sk040lf3) (Run A), [`gkqoo0v4`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/gkqoo0v4) (Run B)

**Hypothesis.** Seed `random/numpy/torch/cuda` + pass `Generator`
to `WeightedRandomSampler` and `DataLoader` to eliminate run-to-run
val variance (~6% drift on unseeded runs).

**Results.** **Bit-perfect determinism: 0.0000 absolute drift on
every metric across all 14 epochs.** Run A and Run B produced
identical val_avg, identical per-split val, identical test_avg,
identical per-split test, identical per-epoch train losses. All
match to 4 decimal places.

| Metric | Run A | Run B | |A − B| |
|---|---:|---:|---:|
| `best val_avg/mae_surf_p` | 100.8022 | 100.8022 | **0.0000** |
| `test_avg/mae_surf_p` | 90.1819 | 90.1819 | **0.0000** |
| Mean epoch wall-clock | 132.3s | 132.3s | 0.0s |

**Why bit-exact without `cudnn.deterministic`:** Transolver has no
convolutions (the typical non-determinism source); all linear
projections + softmax attention. Float32 end-to-end (no AMP).
Reduction order in `torch.matmul` is fixed at single-GPU scale.
Determinism comes "for free" without slowdown.

**The `--seed 0` baseline value is val=100.80 / test=90.18 on the
PRE-#820 model.** This sits between the two unseeded runs (99.23
and 105.22), confirming those were lucky/unlucky draws around the
seed-distribution mean.

**Decision.** **Send back for rebase + canonical-baseline run.**
Once rebased onto post-#820 (Fourier PE active), `--seed 0` will
produce a different deterministic value (likely 88-90 range).
That number becomes the new ranking quantity for all future PRs.

**Why one more run, not two:** Bit-exactness was already proven
on the pre-#820 model. Fourier PE is a deterministic prepend; no
new randomness. Save the GPU slot. Sent back at 23:45 with
detailed instructions.

**Once merged**, BASELINE.md will document the seeded canonical
baseline as the new ranking quantity. Borderline ablations (≤1%
predicted effect) will use `--seed {0, 1, 2}` mean for paper-grade
rigor.

## 2026-04-28 23:35 — PR #819: Relative L2 mix α=0.5 (retry on PRE-#820 base, val=98.01) — **SENT BACK for second rebase onto #820 baseline**

- Branch: `willowpai2e4-frieren/relative-l2-loss`
- Student: willowpai2e4-frieren
- W&B run: [`1ho19yku`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/1ho19yku)

**Hypothesis.** Blend α·L_rel + (1-α)·L_abs at α=0.5. The pure
relative L2 (α=1.0, prior run `9enw7nkx`) had landed at val=100.29
with cruise/single −11.6% each but camber_rc +16.9% (geometric
extrapolation regression). α=0.5 should preserve the heterogeneity
gains while restoring gradient signal on the high-magnitude
camber_rc samples.

**Results vs PRIOR baseline 99.23 (pre-#820, L1+ch=[1,1,3]):**

| Metric | Prior baseline | This run (α=0.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.23 | **98.01** | **−1.22%** |
| `test_avg/mae_surf_p` | 92.61 | **88.78** | **−4.13%** |
| Best epoch | 12 | 14 (timeout) | curve still descending |
| Wall time | 30.77 min | 30.94 min | flat |

**Per-split val (best epoch 14):**

| Split | Prior baseline | This run | Δ | (α=1.0 was) |
|---|---:|---:|---:|---:|
| `val_single_in_dist` | 116.68 | 117.71 | +0.9% | (−12.5%) |
| `val_geom_camber_rc` | 113.94 | **111.67** | **−2.0%** | (+12.1% — fixed!) |
| `val_geom_camber_cruise` | 75.02 | 73.72 | −1.7% | (−10.4%) |
| `val_re_rand` | 91.28 | **88.93** | **−2.6%** | (−3.3%) |
| **val_avg** | 99.23 | 98.01 | **−1.22%** | (+1.07%) |

**Per-split test (epoch-14 ckpt, ALL 4 splits finite):**

| Split | Prior baseline | This run | Δ |
|---|---:|---:|---:|
| `test_single_in_dist` | 117.77 | **107.48** | **−8.7%** |
| `test_geom_camber_rc` | 99.49 | 100.05 | +0.6% |
| `test_geom_camber_cruise` | 65.29 | 63.20 | −3.2% |
| `test_re_rand` | 87.89 | 84.39 | **−4.0%** |
| **test_avg** | **92.61** | **88.78** | **−4.13%** |

**vs CURRENT merged baseline (89.71/88.16, post-#820 Fourier PE):**
val_avg=98.01 sits **+9.26% above** baseline; test=88.78 is
+0.70% above. PR is also CONFLICTING. So it is not directly
mergeable.

**Key mechanistic insight (preserved regardless of merge):**

The α=1.0 → α=0.5 sweep pinpoints a real lever. Pure relative
loss is *anti-geometric-extrapolation* — it equalizes per-sample
gradient magnitude, but in CFD the high-magnitude samples are
the geometric outliers, so equalization reduces the model's
incentive to fit them. The α=0.5 blend keeps half the per-sample
equalization (driving cruise/single/re_rand wins) while keeping
half the absolute gradient (preserving geometric extrapolation
on camber_rc). This is a paper-worthy observation about loss
formulation × dataset geometry interaction.

**Decision.** **Send back for second rebase onto post-#820 advisor
branch HEAD.** Rationale:

1. PR is CONFLICTING — needs rebase regardless.
2. Mechanism is orthogonal to Fourier PE (loss formulation vs
   input feature spectrum), so should compound.
3. Test signal (−4.13%, 3 of 4 splits improving, single_in_dist
   −8.7%) typically survives baseline shifts because the
   per-sample magnitude distribution is dataset-fixed.
4. camber_rc reversal (+12.1% → −2.0%) is ch=[1,1,3]-mediated,
   not Fourier-mediated, so should preserve.

**Threshold to merge after rebase:** val_avg < 89.71. Predicted
target: ~88.6 if α=0.5 compounds linearly. Sent back at 23:35
with detailed instructions and per-split predictions.

## 2026-04-28 23:20 — PR #816: FiLM-condition Transolver blocks (rebase #1, val=91.82) — **SENT BACK for second rebase onto #820 baseline**

- Branch: `willowpai2e4-alphonse/film-conditioning`
- Student: willowpai2e4-alphonse
- W&B run: [`kn6so193`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/kn6so193)

**Hypothesis.** FiLM-Zero modulates LayerNorm scale/shift via a
small MLP on the 11 global scalars (log Re, AoA1, NACA1×3, AoA2,
NACA2×3, gap, stagger). Predicted −5 to −12% on val_avg/mae_surf_p.

**Results vs PRIOR baseline 99.23 (pre-#820, L1+ch=[1,1,3]):**

| Metric | Prior baseline | This run (rebase #1) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.23 | **91.823** | **−7.47%** |
| `test_avg/mae_surf_p` | 92.61 | **80.993** | **−12.55%** |
| Param count | 660 K | 829 K | +26% |
| Best epoch | 12 | 13 / 13 (timeout) | curve still descending |
| Wall time | 30.77 min | 31.22 min | +1.5% |

**Per-split val (best epoch 13):**

| Split | Prior baseline | This run | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 116.68 | 115.03 | −1.41% |
| `val_geom_camber_rc` | 113.94 | **98.17** | **−13.84%** |
| `val_geom_camber_cruise` | 75.02 | 69.54 | −7.30% |
| `val_re_rand` | 91.28 | 84.55 | **−7.37%** |
| **val_avg** | 99.23 | 91.82 | **−7.47%** |

**Per-split test:** EVERY split improved 9–16% (single −16.48%,
camber_rc −12.31%, cruise −10.11%, re_rand −9.34%). The cross-split
robustness is far stronger than typical noise.

**vs CURRENT merged baseline (89.71, post-#820 Fourier PE):**
val_avg=91.82 sits **+2.36% above** baseline. PR is also CONFLICTING
on the file paths #820 touched. So it is not directly mergeable.

**Decision.** **Send back for second rebase onto post-#820 advisor
branch HEAD.** Rationale:

1. PR is CONFLICTING — needs rebase regardless.
2. The current val=91.82 was earned against the pre-Fourier-PE
   baseline. We need to re-test against the post-#820 baseline
   (89.71) to see whether FiLM × Fourier PE compound or one
   absorbs the other.
3. Mechanistically they touch orthogonal levers (input-feature
   spectral basis vs in-block LN modulation), so we predict
   compounding to ~85–87 val, but require evidence.
4. The test signal (−12.55% across all 4 splits) is too strong
   to abandon. Camber_rc reversal under ch=[1,1,3] is
   mechanistically clean (the channel up-weight rebalances the
   loss landscape so the conditioning channel finally helps the
   raceCar foil-2 noise dimensions instead of being amplified by
   them).

**Threshold to merge after rebase:** val_avg < 89.71 AND test
4-split mean < 88.16. Sent back at 23:20 with detailed rebase
instructions and per-split predictions.

## 2026-04-28 23:15 — PR #861: Volume subsampling keep_frac=0.15 — **CLOSED (superseded by #820 merge)**

- Branch: `willowpai2e4-fern/volume-subsampling`
- Student: willowpai2e4-fern
- W&B run: [`lgdx6vqn`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/lgdx6vqn)

**Hypothesis.** Uniform random volume subsampling at keep_frac=0.15 to
shift effective surface fraction from ~8% → 30%, predicted to give
−3 to −8% on val_avg/mae_surf_p plus 50–70% wall-clock saving from
fewer tokens.

**Results vs OLD baseline (99.23):**

| Metric | OLD baseline | This run | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.23 | 99.60 | +0.37% (wash) |
| 3-split test mean | 99.34 | **97.63** | **−1.73%** |
| Wall time per epoch | ~131s | 132.86s | flat (no compute saved) |

**Vs NEW baseline 89.71 (#820 merged just before result landed):**
+11.0% on val, +10.7% on test → not mergeable.

**Analysis (fern's mechanism — sharp and useful):**

1. **Implementation was mask-only, not input-subsample.** The PR spec
   masked nodes from the loss but Transolver still processed all N
   tokens. So no wall-clock gain — predicted 50-70% drop didn't
   materialize. Different experiment than the title suggested.

2. **Effective surface fraction was 10.7%, not 30-40%.** Baseline
   surface fraction is ~1.8% (not 8% as I assumed in PR body). Even
   keep_frac=0.15 only triples the surface share. Math: PR-design
   mathematically only testable at keep_frac<=0.05.

3. **Val/test divergence is mechanistically clean.** Random per-sample
   volume mask = DropConnect on input tokens for the loss = a
   regularizer. Helps generalization on test more than fit on val.
   With timeout at ep 14, regularization signal is partially
   unrealized on val.

4. **Per-split inversion on `val_single_in_dist`** (predicted biggest
   gain, got biggest val regression of +4.6%). Mechanism: heavy-tail
   pressure profiles need MORE volume context, not less — the volume
   field around a foil with sharp pressure encodes the BL build-up
   that propagates into surface predictions. Dropping 85% of those
   tokens dilutes the BL signal.

**Decision.** Closed. The mask-only DropConnect lever is sub-1% even
when it works; can't recover the gap to 89.71. Promoted fern's own
follow-up #3 (stratified volume subsampling weighted by
distance-to-surface) to a fresh PR (#888) — tests the BL-density
hypothesis cleanly: keep BL nodes (d<0.05), drop far-field nodes
(d>0.3) at floor probability. Mechanistically distinct from Fourier
PE — should compound.

## 2026-04-28 23:00 — PR #820: Fourier PE K=4 on (x, z) — **MERGED (-9.59%)**

- Branch: `willowpai2e4-thorfinn/fourier-pe` (squash-merged)
- Student: willowpai2e4-thorfinn
- W&B run: [`w9xbc0wl`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/w9xbc0wl) (rebased on L1+ch=[1,1,3] merged baseline)

**Hypothesis.** Prepend multi-scale Fourier positional features
`[sin(π·2^k·x), cos(π·2^k·x), sin(π·2^k·z), cos(π·2^k·z)]` for k=0..K-1
to the input, giving the preprocess MLP a free spectral basis for the
~1000:1 wavelength range from boundary layer to domain background.
Predicted −4 to −10%.

**Results (epoch 14 best, 30.91 min, rebased on merged baseline):**

| Metric | Fourier PE K=4 | Baseline (#754+#797) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **89.71** | 99.23 | **−9.59%** |
| 3-split test mean | **88.16** | 99.34 | **−11.25%** |
| test_avg | NaN (cruise rebase artifact) | 92.61 | — |
| Params | 666,455 | 662,359 | +4,096 (+0.6%) |

Per-split val (epoch 14):

| Split | Baseline | Fourier PE | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 116.68 | 109.16 | −6.4% |
| `val_geom_camber_rc` | 113.94 | 106.62 | −6.4% |
| `val_geom_camber_cruise` | 75.02 | **60.60** | **−19.2%** |
| `val_re_rand` | 91.28 | 82.47 | −9.7% |
| **val_avg** | 99.23 | **89.71** | **−9.59%** |

**Analysis.** Spectral-bias story confirmed. Raw `(x, z)` forces the MLP
to discover high-freq basis via piecewise-affine compositions; Fourier
encoding hands those features for free. +4K params → −9.59% MAE is an
exceptional capacity-vs-inductive-bias signal. val_re_rand flip from −0.2%
(L1-only base) to −9.7% (ch=[1,1,3] base) is mechanistically clean:
ch=[1,1,3] makes pressure dominant, so the Fourier basis earns its keep
on Re-varying splits where near-foil pressure BL matters.

**Cumulative wins:** L1 (101.93) → +ch=[1,1,3] (99.23, −2.65%) → +Fourier
K=4 (89.71, −9.59%) = **−12.0% end-to-end vs L1-only**.

**Decision.** Merged. BASELINE.md updated (new val=89.71, 3-split test=88.16).
Assigned thorfinn → Fourier bands sweep K∈{3,6,8} (#883) — 3-run sweep to
map the bandwidth optimum and confirm K=4 is near-optimal.

## 2026-04-28 22:45 — PR #851: Huber loss δ=1.0 — **CLOSED (negative)**

- Branch: `willowpai2e4-tanjiro/huber-delta-loss`
- Student: willowpai2e4-tanjiro
- W&B run: [`vt737i14`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/vt737i14)

**Hypothesis.** Huber δ=1.0 (smooth quadratic near zero, L1 outlier
asymptote) should improve late-stage convergence near `err=0` over
pure L1, smoothing the optimizer's trajectory.

**Results (epoch 13 best, 30.79 min):**

| Metric | Baseline (L1, m46h5g4s) | Huber δ=1.0 | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 99.23 | 111.69 | **+12.6%** |
| `test_avg/mae_surf_p` (3 finite splits) | 99.34 | 109.61 | +10.3% |

Per-split val damage (predicted opposite!):

| Split | Δ |
|---|---:|
| `val_single_in_dist` (heavy-tail) | +5.3% |
| `val_geom_camber_rc` | +8.3% |
| `val_geom_camber_cruise` | **+23.7%** |
| `val_re_rand` | **+18.0%** |

**Analysis (tanjiro's mechanism — clean and correct).** With
z-normalized targets, residuals concentrate in `|err|<=1` regime where
**Huber's gradient shrinks** (e.g. err=0.3 → grad=0.3 vs L1's 1.0). At
14-epoch budget, Huber gives the optimizer 3-10× weaker per-step
pressure on the very residuals we want it to clean up.

The L1+ch=[1,1,3] baseline succeeds because L1's constant-magnitude
gradient lets the 3× pressure weight bite at every optimization step.
Huber's diminishing-near-zero gradient counteracts that lever — a
direct mechanistic clash.

**Decision.** Closed. Loss-shape lever family is exhausted (Huber is
the canonical smoothing variant). Three negatives in schedule+loss-shape
on tanjiro's account (#758, #818, #851). Reassigned tanjiro → LinearNO
ELU+1 linear attention (#880) — fresh architectural family (kernel
swap inside slice attention).

## 2026-04-28 22:45 — PR #819: Relative L2 loss (per-sample norm) — **SENT BACK (mixed)**

- Branch: `willowpai2e4-frieren/relative-l2-loss`
- Student: willowpai2e4-frieren
- W&B run: [`9enw7nkx`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/9enw7nkx)

**Hypothesis.** Per-sample L1-numerator / L2-denominator normalization
to equalize gradient contribution across samples with heavy-tail
target magnitude. Predicted -5 to -15%.

**Results vs merged baseline (99.23):**

| Split | Baseline (m46h5g4s) | This run | Δ |
|---|---:|---:|---:|
| `val_single_in_dist` | 116.68 | 117.81 | +1.0% |
| `val_geom_camber_rc` | 113.94 | **127.75** | **+12.1%** |
| `val_geom_camber_cruise` | 75.02 | **67.30** | **-10.3%** |
| `val_re_rand` | 91.28 | 88.28 | -3.3% |
| **val_avg** | **99.23** | **100.29** | **+1.07%** |
| **3-split test mean** | 99.34 | 104.03 | +3.2% |

**Analysis (frieren's split-heterogeneity story — very useful):**
Per-sample equalization works as advertised — `val_single_in_dist` and
`val_geom_camber_cruise` improved -11.6% each (vs frieren's #752 L1-only
reference baseline). But `val_geom_camber_rc` regressed +16.9% because
re-weighting reduces incentive to fit high-magnitude samples — exactly
the regime that drives geometric extrapolation to unseen camber. Per-
sample normalization is **anti-geometric-extrapolation** when target
magnitude correlates with geometric extremity.

**Note on baseline confusion:** Frieren reported -1.6% vs #752 (101.93
L1-only), but #754 (L1+ch=[1,1,3], val=99.23) merged after his run
started. Re-anchoring to the new baseline shifts the headline to +1.07%
(noise floor, not a win).

**Decision.** Sent back for one focused retry: mixed loss `α·L_rel +
(1-α)·L_abs` at α=0.5 — frieren's own #1 follow-up. Goal: keep the
cruise/single wins without the camber_rc damage. Single retry; if it
doesn't beat 99.23, close. PR #819 reverted to draft (status:wip).

## 2026-04-28 22:35 — PR #757: 5% warmup + cosine retest on L1+ch=[1,1,3] — **CLOSED (superseded)**

- Branch: `willowpai2e4-nezuko/warmup-cosine`
- Student: willowpai2e4-nezuko
- W&B run: nezuko's retest on the merged baseline

**Hypothesis.** Re-test 5% linear warmup + cosine decay against the
merged L1+ch=[1,1,3] baseline (99.23) — original run predated the channel-
weight merge.

**Results.** `val_avg/mae_surf_p = 103.00` vs baseline 99.23 = **+3.81%
worse**. Damage concentrates on `geom_camber_rc (+7.23%)` and
`re_rand (+7.42%)`. Student's own verdict: *"Don't merge this PR.
Warmup is dominated by the existing L1 + channel-weight baseline at the
30-min budget. Recommendation: close as superseded."*

**Decision.** Closed. **Schedule lever family is now exhausted at this
30-min budget** — three negatives in a row: #758 lr+warmup (+9.7%), this
#757 warmup retest (+3.81%), #818 SGDR T_0=10 (+6%). The 30-minute hard
timeout consistently puts cosine annealing at peak performance and any
schedule that delays / restarts the lr undershoots.

Reassigned nezuko → Domain-ID embedding (#872) as a **structural** lever
(architectural rather than schedule-based) — gives the model an explicit
3-class regime hook (single / tandem-rc / tandem-cruise) derived from
gap and AoA1, addressing the discontinuity baked into the dataset
splits.

## 2026-04-28 22:35 — PR #753: surf_weight sweep 20/30/50 — **CLOSED (superseded)**

- Branch: `willowpai2e4-edward/surf-weight-sweep`
- Student: willowpai2e4-edward
- W&B group: [`willow-pai2e-r4-surf-weight-sweep`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/groups/willow-pai2e-r4-surf-weight-sweep)
- Runs: `0l619f0v` (sw=20), `bbecow9g` (sw=30), `zmu5iilg` (sw=50)

**Hypothesis.** Boost `surf_weight` past the conservative 10 to focus
gradient on surface (the only ranked channel). Predict −3 to −10% with
peaked-not-monotonic shape.

**Results (predates L1+ch=[1,1,3] merge):**

| `surf_weight` | `val_avg/mae_surf_p` | `test_avg/mae_surf_p` |
|---|---:|---:|
| 20 | 136.23 | 126.38 |
| **30** | **125.80** | **119.51** |
| 50 | 133.13 | 128.32 |

Best of sweep: sw=30 with val=125.80. Curve shape was correctly peaked,
not monotonic (sw=50 over-tilts: surface +1.5%, volume +14.9%).
Edward's analysis was clean and the bug-discovery comment that
preceded these runs is what unblocked the cruise `-Inf` issue (now
fixed via merged #797).

**Analysis.** Numbers sit ~+27% above the current merged baseline 99.23
because the runs predate L1+ch=[1,1,3] (#754). The merged channel
weighting `[1,1,3]` already gives pressure 3× per-channel weight, so
sw=10 + ch=[1,1,3] is effectively `sw_eff_p ≈ 30` on pressure. Re-running
this sweep upward on the new baseline would just push past edward's
own identified tipping point.

The `surf_weight × channel_weights` lever family is **multiplicative,
not orthogonal**, and combined with fern's #829 channel-weight
saturation finding (3× is past the inflection at 5×), the per-channel
loss-scaling lever family is now exhausted. Split-aware loss weighting
or learnable surf_weight would be a round-3 redesign, not a round-2
sweep.

**Decision.** Closed as superseded. Edward's in-PR custom NaN workaround
is replaced by canonical guards in merged #797.

Reassigned edward → EMA model weights (Polyak averaging, #873) — a
**snapshot-selection** lever in a fresh family (implicit regularization,
not loss reformulation), orthogonal to all in-flight round-2 work and
predicted to compound with everything that lands.

## 2026-04-28 22:15 — PR #797: NaN/Inf guards in evaluate_split — **MERGED (infra unblock)**

- Branch: `willowpai2e4-askeladd/nan-guard-on-L1` (squashed)
- Student: willowpai2e4-askeladd
- W&B run: [`2hcmefh9`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/2hcmefh9) (rebased run on top of #754)

**Hypothesis.** Add `nan_to_num(pred)` and a per-sample `y_finite` filter
in `evaluate_split` to recover a reportable `test_avg/mae_surf_p` blocked
by two interacting bugs: model-side `Inf` (init-dependent) on cruise
samples, and `-Inf` in the GT p-channel of `test_geom_camber_cruise/000020.pt`
(761 of 225K nodes, all volume).

**Results (rebased on #754, best epoch 14, 30.8 min wall)**

| Metric | This run (`2hcmefh9`) | Prior baseline (`m46h5g4s`) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 105.22 | 99.23 | **+6.0% noise** |
| **`test_avg/mae_surf_p`** | **92.61 ✓** | NaN (blocked) | **NEW finite metric** |
| `test_single_in_dist` | 117.77 | 106.78 | run-to-run noise |
| `test_geom_camber_rc` | 99.49 | 104.87 | run-to-run noise |
| `test_geom_camber_cruise` | **65.29 ✓** | NaN | **first finite cruise test** |
| `test_re_rand` | 87.89 | 86.37 | run-to-run noise |
| nonfinite_pred (all splits) | 0 | — | guard 1 unused on this seed |
| nonfinite_gt_samples (cruise test only) | **1** | — | guard 2 fired exactly as expected |

**Analysis.** This is an **infra unblock**, not a val improvement. With
both diagnostic counts at 0 on every val split, the val pass is bit-
identical to a guard-less run. The val_avg drift (99.23 → 105.22) is
purely run-to-run init/sampler variance: `train.py` does not seed any
RNG. Same code, two seeds, 6% drift — almost entirely on
`val_single_in_dist`.

**The headline win:** `test_avg/mae_surf_p = 92.61` is the **first
reportable test number on this branch**. Before this PR, every PR's
W&B summary showed `None` for the headline test metric because cruise's
`-Inf` GT poisoned the float64 accumulator via IEEE 754 `Inf * 0 = NaN`.
The per-sample `y_finite` filter dropped the bad sample cleanly.

Askeladd also identified a residual cosmetic issue: `cruise/loss = NaN`
(display-only) traced to `nan_to_num(y_norm)` with default args
overflowing through `channel_weights[2]=3` to `+Inf`, then `Inf * 0 = NaN`.
One-line fix (`nan=0.0, posinf=0.0, neginf=0.0`) will ride along with
next `evaluate_split`-touching PR — not worth a 30-min retrain.

**Decision.** Merged. BASELINE.md updated with the new test_avg=92.61
unblock and a note about the run-to-run val variance from missing seeds.
Reassigned askeladd → seed PR (#863) — their own #1 follow-up suggestion.
This is the highest-leverage infra fix on the branch right now: without
seeding, ablation legibility is broken (3% lever effects are within
6% run-to-run noise).

## 2026-04-28 22:05 — PR #820: Fourier PE on (x, z) coordinates — **SENT BACK (rebase)**

- Branch: `willowpai2e4-thorfinn/fourier-pe-coords`
- Student: willowpai2e4-thorfinn
- W&B run: [`rixnmfuk`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/rixnmfuk)

**Hypothesis.** Pre-encode `(x, z)` with multi-scale sin/cos features
(K=4 frequency bands → 16 extra input dims) to overcome MLP spectral bias
on sharp boundary-layer pressure peaks. Predicted −4 to −10%.

**Results (best epoch 13/14, 30.8 min wall, on L1-only baseline)**

| Metric | This run (`rixnmfuk`) | L1 baseline (`8lyryo5g`) | Current baseline (#754) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **91.15** | 101.93 | 99.23 |
| 3-split test mean | **88.63** | 100.83 | 99.34 |
| `val_single_in_dist` | 109.24 | 133.25 | 116.68 |
| `val_geom_camber_rc` | 99.09 | 109.26 | 113.94 |
| `val_geom_camber_cruise` | 67.35 | 76.13 | 75.02 |
| `val_re_rand` | 88.92 | 89.07 | 91.28 |

vs L1-only: **−10.6%** (top of predicted band)
vs current merged baseline (#754): **−8.2%** (after rebase, if it holds — strongest single result on the branch)

**Diagnostics confirmed mechanism.** Coord range `|x_norm|`max=7.32,
p99=4.69 — plenty of dynamic range for high-frequency bands (highest band
≈ 25 rad). Param count delta +4096 weights matches predicted ~+5K. Most
striking: `val_re_rand` improved only −0.2% — clean negative control,
since Re randomization doesn't introduce new spatial frequencies, Fourier
PE shouldn't help there, and it doesn't. Spectral-bias story holds.

Sharp-feature splits gained the most:
- `val_single_in_dist` (raceCar single, ground-effect peaks): −18.0%
- `val_geom_camber_cruise` (sharp tandem-cruise gradients): −11.5%
- `val_geom_camber_rc` (tandem raceCar): −9.3%

Val curve still bending at epoch 13 — the 30-min timeout cut training
short of the asymptotic minimum.

**Decision.** Sent back for **rebase**. Branch was created from L1
baseline #752 before the channel-weight #754 merged. PR is
`CONFLICTING`. Asked thorfinn to rebase, resolve the train.py conflict
(keep both: x_in Fourier-encoded path AND channel_weights multiplication
on abs_err — orthogonal: input encoding vs loss formulation), and
re-run on top of L1+ch=[1,1,3]. Expected after rebase: val_avg comfortably
beats 99.23 by ~−7 to −10%. **Will merge as soon as the rebased run
lands** — this is the strongest single signal so far.

## 2026-04-28 22:00 — PR #829: p-channel weight sweep 5× — **CLOSED**

- Branch: `willowpai2e4-fern/p-channel-weight-sweep` (deleted)
- Student: willowpai2e4-fern
- W&B run: [`ampb9xcb`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/ampb9xcb)

**Hypothesis.** 5× → 10× continuation of the merged 3× channel-weight
to map the curve. Decision rule: skip 10× if 5× regresses past 99.23.

**Results (best epoch 13, 30 min wall)**

| Run | `val_avg/mae_surf_p` | Δ vs 3× baseline |
|---|---:|---:|
| 3× (merged, `m46h5g4s`) | 99.226 | — |
| **5× (this PR, `ampb9xcb`)** | **102.782** | **+3.6% worse** |
| 10× | (skipped per decision rule) | — |

Per-split: only `val_geom_camber_rc` (highest baseline error) improved
(−7.5%). Cruise +16.3% worse, single_in_dist +7.5% worse, re_rand +1.9%
worse. Surface Uy regressed on all 4 splits (+7.1% avg) — velocity
starvation pattern, mild. Surface Ux *improved* on val_avg (−7.9%) —
unexpected nuance not predicted.

**Decision.** Closed as planned by the decision rule. Optimum is in
(3×, 5×); 5× is past the inflection. The split-level heterogeneity
(camber_rc still wants more p-weight; cruise / single / re_rand want
less) suggests **split-aware loss weighting** as a future direction
(student's #3 follow-up suggestion). Reassigned fern to volume
subsampling (PR #861) — a mechanistically different lever.

## 2026-04-28 21:55 — PR #816: FiLM-condition Transolver blocks on global scalars — **SENT BACK (rebase)**

- Branch: `willowpai2e4-alphonse/film-conditioning`
- Student: willowpai2e4-alphonse
- W&B run: [`8pkn0ire`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/8pkn0ire)

**Hypothesis.** Inject the 11 global scalars (log Re, AoAs, NACAs, gap,
stagger) directly into each TransolverBlock via FiLM (AdaLN-Zero)
modulation of LayerNorm. Predicted −5 to −12% on `val_avg/mae_surf_p`.

**Results (best epoch 12/12, 30.27 min wall, on L1-only baseline)**

| Metric | This run (`8pkn0ire`) | L1 baseline (`8lyryo5g`) | Current baseline (#754, `m46h5g4s`) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **96.61** | 101.93 | 99.23 |
| 3-split test mean | **95.12** | 100.83 | 99.34 |
| `val_single_in_dist` | 114.19 | 133.25 | 116.68 |
| `val_geom_camber_rc` | 120.11 | 109.26 | 113.94 |
| `val_geom_camber_cruise` | 64.87 | 76.13 | 75.02 |
| `val_re_rand` | 87.26 | 89.07 | 91.28 |
| Param count | 0.83 M | 0.66 M | 0.66 M |

vs L1-only baseline: **−5.2%** (in predicted band, conservative end)
vs current merged baseline (#754): **−2.6%** (still beats baseline)

**Diagnostics (epoch 12).** FiLM-Zero invariant confirmed (epoch-1 metrics
match baseline within ~2%). Per-block |gamma| grows 0.30 → 0.43, |beta|
grows 0.16 → 0.35 — healthy gradient flow, deeper blocks pull more
conditioning. `cond_mlp_last_w_norm` grew smoothly 0 → 20.6 with no
spikes.

**Per-split surprise.** Predicted biggest gains on OOD camber splits.
Cruise was indeed the largest winner (−14.8%), but **camber_rc regressed
+9.9%** and `val_single_in_dist` (predicted flat) gained −14.3%. Student's
mechanism hypothesis: in raceCar (single-foil) the foil-2 NACA / gap /
stagger conditioning dims carry no real signal but the model learns
spurious correlations on them. Cruise is tandem so all 11 scalars are
meaningful there. Suggests masking inactive conditioning dims is a
strong round-3 follow-up.

**Decision.** Sent back for **rebase**. Branch was created from L1
baseline #752 before the channel-weight #754 merged. PR is currently
`CONFLICTING`. Asked alphonse to rebase, resolve the train-loop conflict
(keep both: channel_weights in train loss AND FiLM-Zero hookpoints —
they are orthogonal regions: loss formulation vs LayerNorm modulation),
and re-run on top of L1+ch=[1,1,3]. Expected after rebase: val_avg
beats 99.23 by a similar ~−2.6 to −5%. Will merge the moment the new
run lands.

## 2026-04-28 21:45 — PR #818: SGDR Cosine Warm Restarts (T_0=10, T_mult=2) — **CLOSED**

- Branch: `willowpai2e4-tanjiro/sgdr-warm-restarts` (deleted)
- Student: willowpai2e4-tanjiro
- W&B run: [`0e5uk8ux`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/0e5uk8ux)

**Hypothesis.** Cosine Warm Restarts let the optimizer escape sharp minima
and find flatter basins. T_0=10, T_mult=2 → cycle 1 ep 0–10, cycle 2 ep
10–30, cycle 3 ep 30–70 (clipped at 50). Predicted −2 to −5%.

**Results (best epoch 10/12, 30 min wall, on L1-only baseline)**

| Metric | L1 baseline (`8lyryo5g`) | SGDR T_0=10 (`0e5uk8ux`) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.93 | 108.07 | **+6.04 worse** |
| `val_single_in_dist` | 133.25 | 138.39 | +5.14 |
| `val_geom_camber_rc` | 109.26 | 115.52 | +6.26 |
| `val_geom_camber_cruise` | 76.13 | 80.95 | +4.82 |
| `val_re_rand` | 89.07 | 97.43 | +8.36 |

LR trace confirmed restart fired at exactly epoch 10 (3759 steps,
LR 5e-6 → 5e-4). Best val landed at the **end of cycle 1** (epoch 10's
eta_min=5e-6), not at any post-restart moment. Restart at epoch 11 wiped
progress: val_avg jumped 108.07 → 133.49 → 147.17 in cycle 2's high-LR
phase before timeout fired at epoch 12.

**Analysis.** Mechanism worked exactly as designed but is **structurally
mismatched to the budget**. With L1 baseline best at epoch 14 (the
natural convergence point), placing the restart at epoch 10 wipes
progress before convergence completes. Cycle-2 needed several more epochs
of cosine decay to potentially escape, but at our 30-min/14-epoch
effective budget cycle 2 cannot complete. Tanjiro's recommended T_0=20
would have the same budget cliff (cycle 2 = ep 20–60, clipped at 50,
natural convergence already in cycle 1).

**Decision.** Closed as dead end — schedule is not where the headroom
lives at this budget. Two consecutive negatives on schedule/LR levers
(#758 lr+warmup, #818 SGDR) confirms this. Reassigned tanjiro to a
different family: **Huber loss δ=1.0** (PR #851). Mechanistically
orthogonal — reshapes per-element error magnitude function, not the
training trajectory.

## 2026-04-28 21:11 — PR #754: Per-channel pressure weight 3× ON L1 — **MERGED**

- Branch: `willowpai2e4-fern/p-channel-3x` (squashed)
- Student: willowpai2e4-fern
- W&B run: [`m46h5g4s`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/m46h5g4s)

**Hypothesis (L1 retest).** Whether `channel_weights = [1, 1, 3]` compounds
on top of the merged L1 baseline (101.93). Original MSE-era result (130.87)
left the question unanswered.

**Results (best epoch 12, 30.77 min wall, on L1)**

| Metric | fern channel-3x ON L1 | L1 baseline (#752) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | **99.226** | 101.93 | **−2.65%** |
| 3-split test mean | 99.34 | 100.83 | −1.48% |
| `val_single_in_dist/mae_surf_p` | 116.68 | 133.25 | −12.4% |
| `val_geom_camber_rc/mae_surf_p` | 113.94 | 109.26 | +4.3% |
| `val_geom_camber_cruise/mae_surf_p` | 75.02 | 76.13 | −1.5% |
| `val_re_rand/mae_surf_p` | 91.28 | 89.07 | +2.5% |
| `val_avg/mae_surf_Ux` | 1.789 | 1.429 | +25.2% |
| `val_avg/mae_surf_Uy` | 0.693 | 0.611 | +13.4% |

**Analysis.** The 3× pressure weight stacks with L1 — net pressure improvement
with acceptable velocity-channel regression (we don't rank on velocity).
Biggest gain on `val_single_in_dist` (heaviest-tail, where extreme pressure
samples dominate). Two splits regressed slightly (rc, re_rand) but the net
across all 4 is favorable. W&B verification: every per-split number matched
fern's report exactly.

**Decision.** Merged as new baseline at val_avg/mae_surf_p = **99.23**.
fern reassigned to **channel-weight sweep continuation (5× and 10×)** —
PR #829.

## 2026-04-28 21:13 — PR #757: 5% linear warmup + cosine — **SENT BACK**

- Branch: `willowpai2e4-nezuko/warmup-cosine`
- Student: willowpai2e4-nezuko
- W&B run: [`2ipmj9ct`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/2ipmj9ct)

**Hypothesis.** Add 5% (2-epoch) linear warmup before cosine decay; expected
benefit from gentler ramp at the start of training.

**Results (MSE-era, best epoch 13/14, 30.7 min wall)**

| Metric | nezuko 5% warmup (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 128.19 |
| 3-split test mean | ~126.86 |

**Analysis.** Better than several MSE round-1 runs (alphonse 256x8 186.40,
frieren slice=128 137.79, thorfinn BS=8 153.68, tanjiro lr=1e-3 151.60) but
not directly comparable to the new baseline (99.23 on L1+ch=[1,1,3]).
Independent confirmation of cruise-test NaN bug (fourth confirmation).

**Decision.** Sent back. Asked nezuko to rebase onto current baseline and
re-run with 5% warmup on top of L1 + ch=[1,1,3]. Tanjiro's lr=1e-3+10%
warmup failed to compound (closed PR #758 at +9.7%) but nezuko's 5% warmup
is mechanistically gentler (no peak-LR change), so the failure pattern
doesn't necessarily transfer.

## 2026-04-28 21:30 — PR #797: Non-finite guards for evaluate_split — **SENT BACK (rebase)**

- Branch: `willowpai2e4-askeladd/nan-guard-on-L1`
- Student: willowpai2e4-askeladd
- W&B run: [`wewusbcj`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/wewusbcj)

**Hypothesis.** Add `nan_to_num(pred)` and a per-sample `y_finite` filter in
`evaluate_split` to (a) guard against model-side `Inf` predictions on certain
cruise-test samples and (b) drop GT-NaN samples (specifically
`test_geom_camber_cruise/000020.pt`, p-channel) before they poison the
`accumulate_batch` accumulator via `NaN * 0 = NaN`.

**Results (best epoch 14, 31.1 min wall, on L1 only — pre-#754 fork)**

| Metric | This run (`wewusbcj`) | L1 baseline (`8lyryo5g`) | L1+ch baseline (`m46h5g4s`) |
|---|---|---|---|
| `val_avg/mae_surf_p` | 94.345 | 101.93 | 99.23 |
| `test_avg/mae_surf_p` | **84.551 (FINITE!)** | NaN | NaN |
| `test_single_in_dist/mae_surf_p` | 102.366 | 106.78 | 106.78 |
| `test_geom_camber_rc/mae_surf_p` | 93.169 | 104.87 | 104.87 |
| `test_geom_camber_cruise/mae_surf_p` | **59.334** | NaN | NaN |
| `test_re_rand/mae_surf_p` | 83.337 | 86.37 | 86.37 |

Diagnostic counts (per-split): `nonfinite_pred_count=0` everywhere (Bug 1
didn't fire on this seed); `nonfinite_gt_samples=1` only on
`test_geom_camber_cruise` (Bug 2 fired exactly once, matching fern's
empirical scan).

**Analysis.** This is the first run in the entire branch with a finite
`test_avg/mae_surf_p`. The fix is the canonical infrastructure change for
the paper-facing metric. The val drop (101.93 → 94.34) is init/sampler
noise — student correctly notes that with all `nonfinite_*` counts zero on
val splits, the val pass is bit-identical to a no-fix run. Without
`torch.manual_seed`, run-to-run variance on this size run easily explains
~5–7 points on `val_avg`.

**Decision.** Sent back for **rebase**. Branch was created from L1 baseline
#752 before the channel-weight #754 merged; the PR is now `CONFLICTING`
because both touch the train loop. Asked askeladd to rebase, resolve the
conflict (keep both: channel_weights in train loss, nan-guards in
evaluate_split — they are orthogonal regions), and re-run on top of
L1+ch=[1,1,3]. Expected after rebase: val_avg ≈ 99.23 ± noise; test_avg
finite for the first time on the new baseline.

This will be merged the moment the rebased run lands. Highest-priority
infra fix on the branch — unblocks the paper-facing test metric.

---

- Branch: `willowpai2e4-askeladd/l1-loss`
- Student: willowpai2e4-askeladd
- W&B run: [`8lyryo5g`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/8lyryo5g)

**Hypothesis.** Headline metric is MAE, training loss was MSE. Switching the
per-element loss to L1 should align gradients with the metric and stop
high-Re samples (per-sample y_std up to 2077) from dominating optimization
through MSE's quadratic outlier penalty.

**Implementation.** One-line changes in train.py: `(pred - y_norm) ** 2` →
`(pred - y_norm).abs()` in both the train loop and `evaluate_split`. No
other knobs touched.

**Results (epoch 14, run hit 30 min timeout)**

| Metric | L1 (#752) | Comparable MSE round-1 runs |
|---|---|---|
| `val_avg/mae_surf_p` | **101.93** | 124–162 |
| `test_avg/mae_surf_p` | NaN (cruise bug) | NaN |
| 3-split test mean | 100.83 | — |

| Split | val mae_surf_p |
|---|---|
| `val_single_in_dist` | 133.25 |
| `val_geom_camber_rc` | 109.26 |
| `val_geom_camber_cruise` | 76.13 |
| `val_re_rand` | 89.07 |

**Analysis.** L1 dropped val_avg/mae_surf_p ~33% relative to the next-best
MSE run in the round. The improvement is consistent across all four val
splits (largest absolute reduction on `val_single_in_dist`, the heaviest-tail
split — exactly where MSE's outlier bias would hurt the most). Train/val
loss gap stayed flush (~0.276 vs 0.281), so no overfitting. L1 was still
improving at epoch 14 / timeout — there is more headroom inside the same
budget if other levers shorten epoch time.

**Decision.** Merged as the new baseline. All round 1 PRs that ran on MSE
must be re-tested on top of L1 to know whether their levers compound.

**Open issue surfaced.** `test_geom_camber_cruise/vol_loss = Inf` and
`test_geom_camber_cruise/mae_surf_p = NaN` in this run and every round-1 run
the student inspected. Cause: the model emits Inf on at least one cruise-test
sample's `p` channel; this propagates through scoring (since `Inf * 0 = NaN`
during masked-sum aggregation). Blocks `test_avg/mae_surf_p` reporting until
guarded.

## 2026-04-28 19:55 — PR #758: Higher peak LR (1e-3) with 10% warmup — **SENT BACK**

- Branch: `willowpai2e4-tanjiro/lr-1e-3-warmup`
- Student: willowpai2e4-tanjiro
- W&B run: [`7wplj1pg`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/7wplj1pg)

**Hypothesis.** Default `lr=5e-4` is conservative for a transformer-style
model on small data; `1e-3` with 10% linear warmup should converge faster
without divergence.

**Implementation.** SequentialLR(LinearLR warmup over 5 epochs → CosineAnnealingLR
over 45 epochs); peak lr=1e-3. No other knobs.

**Results (epoch 11 best, hit 30 min timeout at epoch 14)**

| Metric | tanjiro lr=1e-3 (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 151.60 |
| `test_avg/mae_surf_p` | NaN (cruise bug) |
| 3-split test mean | 147.14 |
| Best epoch | 11 / 50 |

**Analysis.** Stable training, no NaN spikes. Best val arrived at epoch 11
(0.22 of cap) — the higher peak LR did front-load gains as predicted.
Better than every other MSE round-1 run (151.60 vs 161.74 / 154.81 / 130.87
/ 124.41) but well behind the merged L1 baseline (101.93). The lever is not
a dead end — it just needs to be tested on top of L1.

**Decision.** Sent back. Asked tanjiro to rebase onto the L1 baseline and
re-run with `lr=1e-3 + 10% warmup + L1`. The two changes are orthogonal
in the codebase (loss tensor vs scheduler), so the rebase should be clean.

## 2026-04-28 20:15 — PR #754: Per-channel loss weight: pressure 3x — **SENT BACK**

- Branch: `willowpai2e4-fern/p-channel-3x`
- Student: willowpai2e4-fern
- W&B run: [`jr8nfzbg`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/jr8nfzbg)

**Hypothesis.** Pressure is the only ranked channel; weighting per-element MSE
on the `p` channel 3× should focus gradient there at the cost of (acceptable)
slight Ux/Uy degradation.

**Implementation.** `channel_weights = [1.0, 1.0, 3.0]` multiplied into
`(pred - y_norm) ** 2` in train loop and `evaluate_split`. Stock model and
hyperparameters otherwise.

**Results (epoch 14 best, 30.8 min wall, ran on MSE before L1 merge)**

| Metric | fern channel-3x (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 130.87 |
| `test_avg/mae_surf_p` | NaN (cruise bug) |
| 3-split test mean (W&B) | 130.20 |

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 169.98 | 2.47 | 1.08 |
| `val_geom_camber_rc` | 137.29 | 3.34 | 1.30 |
| `val_geom_camber_cruise` | 100.65 | 1.72 | 0.73 |
| `val_re_rand` | 115.55 | 2.63 | 1.00 |

**Analysis.** Worse than the merged L1 baseline (101.93) by ~28%, but the run
was MSE-based (predates the L1 merge). Velocity channels stayed sane: surface
Ux MAE 1.7-3.3, Uy MAE 0.7-1.3 — no qualitative degradation, so the 3× weight
is not destabilizing the velocity field. Per-epoch trajectory still descending
at the cap (146 → 141 → 130.9), so a longer schedule would help, but the
bigger question is whether channel-3x compounds with L1.

**Decision.** Sent back. Asked fern to rebase on L1 and re-run; net
expression `sq_err = (pred - y_norm).abs() * channel_weights[None, None, :]`.
If this compounds with L1, we'll sweep 5×/10× next round.

**Important diagnostic surfaced.** Fern empirically pinned the
`test_geom_camber_cruise` NaN bug to **two** distinct issues:

1. **Model emits non-finite predictions** (`vol_loss = +Inf` in W&B summary)
2. **GT itself has NaN** in `test_geom_camber_cruise/000020.pt` p channel
   (verified via direct file scan — exactly 1 of 200 cruise test samples)

The second finding is critical: even with the model output guarded, scoring
still propagates `NaN * 0 = NaN` from the bad GT sample through the
masked-sum accumulator. PR #797 (askeladd, NaN guard) has been expanded to
handle both bugs — drop samples with non-finite GT before `accumulate_batch`,
in addition to the original `nan_to_num` on `pred`.

## 2026-04-28 20:35 — PR #760: batch_size 4→8 — **SENT BACK**

- Branch: `willowpai2e4-thorfinn/batch-size-8`
- Student: willowpai2e4-thorfinn
- W&B run: [`nvpb4uam`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/nvpb4uam)

**Hypothesis.** Default `batch_size=4` underutilizes 96 GB VRAM; doubling to 8
should reduce gradient variance with marginal wall-clock cost. Predicted
−2 to −7%.

**Implementation.** `--batch_size 8` flag, no other changes. (MSE-era run,
predates the L1 merge.)

**Results (best epoch 10, 30.4 min wall, ran on MSE before L1 merge)**

| Metric | thorfinn BS=8 (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 153.68 |
| `test_avg/mae_surf_p` | NaN (cruise bug) |
| 3-split test mean | 156.4 |
| Peak VRAM | 84.2 GB |
| Step time | ~130 s/epoch |
| Epochs at timeout | 14/50 |

**Analysis.** Worse than L1 baseline (101.93) by ~50%, but the run was MSE-based.
Sits in the middle of the round-1 MSE pack (151–162). BS=8 trains stably,
peak VRAM 84.2 GB (much higher than my predicted ~24 GB — slice attention
buffers dominate). BS=16 not viable on this GPU without mixed precision.
Independent confirmation of the GT-NaN scoring bug (third hit, after fern + alphonse).

**Decision.** Sent back. Asked thorfinn to rebase onto L1 baseline and re-run
with `--batch_size 8`. The two changes are orthogonal — should be a clean
rebase. If BS=8 + L1 beats 101.93, merge; else close.

## 2026-04-28 20:35 — PR #749: Capacity scale-up (256×8) — **CLOSED**

- Branch: `willowpai2e4-alphonse/capacity-256x8` (deleted)
- Student: willowpai2e4-alphonse
- W&B run: [`p4syry7v`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/p4syry7v)

**Hypothesis.** Scale Transolver to `n_hidden=256, n_layers=8, n_head=8`
(~3.94M params, ~7.7× baseline) to exploit unused VRAM. Predicted −5 to −15%.

**Implementation.** Three model_config changes plus three infra additions
(necessary to fit — first attempt OOM'd at 94 GB on cruise sample): bf16
autocast, gradient checkpointing per TransolverBlock, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Results (best epoch 4 of 5, 30.3 min wall, MSE pre-L1)**

| Metric | alphonse 256×8 (MSE+bf16) |
|---|---|
| `val_avg/mae_surf_p` | 186.40 |
| `test_avg/mae_surf_p` | 174.29 (with bug-skipped sample, fp32 reeval) |
| Peak VRAM | 24.7 GB (post-infra fixes) |
| Step time | ~6 min/epoch |
| Epochs at timeout | 5/50 |

| Epoch | val_avg/mae_surf_p |
|---|---|
| 1 | 367.98 |
| 2 | 224.81 |
| 3 | 209.78 |
| 4 | **186.40** (best) |
| 5 | 208.83 |

**Analysis.** 83% worse than L1 baseline (101.93) and not viable in our
budget envelope: ~6 min/epoch means only 5 epochs fit vs ~50 for baseline,
so the model never converges. Trajectory shows decay 367 → 186 over 4 epochs
— even extrapolating linearly we wouldn't catch baseline inside the cap.
The capacity hypothesis at this size needs >30 min wall-clock to be testable.

**Decision.** Closed. Asked alphonse to:
1. Open a separate infra-only PR with bf16 + grad checkpointing +
   expandable_segments on top of L1 baseline (genuinely valuable infra,
   wrong vehicle bundled with the capacity bump)
2. Move to FiLM conditioning of LayerNorm (round-2 idea #2) as the next
   experiment — orthogonal to L1, addresses the structural issue of global
   scalars being diluted across 100K+ nodes.

The capacity question isn't dead — once the infra PR lands, we can revisit
at smaller scales (n_hidden=192, or n_hidden=256 with n_layers=6).

**Important diagnostic surfaced (third independent confirmation).**
`test_geom_camber_cruise/000020.pt` has 761 NaN values in GT `p` channel
(out of 225,077 nodes; ~0.34%). Confirmed by alphonse, fern, and thorfinn
independently. PR #797 (askeladd) already expanded in scope to handle both
the model-output Inf path AND the GT-NaN path.

## 2026-04-28 20:50 — PR #758 (round-2 tag): lr=1e-3 + 10% warmup ON L1 — **CLOSED**

- Branch: `willowpai2e4-tanjiro/lr-1e-3-warmup` (deleted)
- Student: willowpai2e4-tanjiro
- W&B run: rebased L1 retest (best epoch 13)

**Hypothesis (retest).** Whether `lr=1e-3 + 10% warmup` compounds on top of
the merged L1 baseline (101.93). Original MSE-era result (151.60) had been
the best of the round-1 MSE pack.

**Results (L1 retest, best epoch 13/14, 30.7 min wall)**

| Metric | tanjiro lr=1e-3+warmup ON L1 |
|---|---|
| `val_avg/mae_surf_p` | 111.83 |
| Δ vs L1 baseline (101.93) | **+9.7% (worse)** |
| Best epoch | 13 / 50 |

**Analysis.** Clean run, no divergence; warmup ramped from 1e-6 → 1e-3 over
epochs 1-5 and stayed near peak through ep 11 then cosine-decayed. The L1
retest landed at 111.83, ~10% worse than L1-baseline alone. Interpretation:
once L1 fixed gradient quality on outliers, the higher peak LR over-steers
the now-better-aligned gradients into a worse minimum. The MSE-era benefit
of higher LR (151.60 vs ~160 in the MSE pack) was largely compensating for
poor MSE gradient quality — a benefit that disappears with L1.

**Decision.** Closed. Lever is exhausted on this baseline. Tanjiro
reassigned to **SGDR (Cosine Warm Restarts)** — round-2 idea #9, builds on
their schedule expertise but uses a different mechanism (periodic LR resets
to escape sharp minima within the 50-epoch cap).

## 2026-04-28 20:50 — PR #755: slice_num 64→128 — **CLOSED**

- Branch: `willowpai2e4-frieren/slice-num-128` (deleted)
- Student: willowpai2e4-frieren
- W&B run: 11 epochs at slice=128, MSE pre-L1

**Hypothesis.** Doubling slice tokens from 64 → 128 should improve the
slice-token decomposition for large meshes (cruise ~210K nodes), giving
finer physics partitioning. Predicted -3 to -8%.

**Results (best epoch 11/11, MSE pre-L1, 30.7 min wall)**

| Metric | frieren slice=128 (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 137.79 |
| `test_avg/mae_surf_p` (with bug-fix) | 124.18 |
| Δ vs L1 baseline (101.93) | **+35% (worse)** |
| Per-epoch wall | +33% (174 s vs 131 s for slice=64) |
| Epochs at timeout | 11 / 50 |

**Per-split wall-clock-anchored comparison** (frieren's table):
- val_geom_camber_cruise: slice=128 **98.87** vs slice=64 warmup-cosine 99.95 (-1.1%)
- val_geom_camber_rc: slice=128 155.21 vs slice=64 142.40 (+9.0%)
- val_single_in_dist: slice=128 181.62 vs slice=64 179.73 (+1.0%)

**Analysis.** Per-epoch quality gain is real but +33% per-epoch wall-clock
cost (PR's +20% estimate undershot) reduces budget to 11 epochs vs 14 for
the baseline. At fixed wall-clock the baseline wins. Cruise (largest mesh)
is the only split where slice=128 looks competitive — supports the
geometric intuition that finer slicing helps complex/large geometries.

The lever isn't dead in principle — if alphonse's bf16+grad-checkpoint infra
PR lands and recovers ~30% throughput, slice_num=128 + L1 could become
viable. Filed for round 3 reconsideration.

**Independent confirmation of cruise-test NaN bug** — third
confirmation (alphonse, thorfinn, frieren). Frieren added a workaround in
their branch (filter samples with non-finite y in `evaluate_split`); same
mechanism as the canonical fix in PR #797 (askeladd) which has expanded
scope.

**Decision.** Closed. Frieren reassigned to **Relative L2 loss** — round-2
idea #1, highest predicted impact (-5 to -15%). Addresses high-Re scale
variation directly: per-sample loss normalization equalizes gradient
contribution across the 4× spread in y_std within a split.

## Round 1+2 status snapshot (2026-04-28 ~21:15)

**Current baseline:** `val_avg/mae_surf_p = 99.226` (PR #754, run `m46h5g4s`)

| PR | Student | Topic | Status |
|----|---------|-------|--------|
| #749 | alphonse | Capacity scale-up (256×8) | **closed** (no convergence in budget) |
| #752 | askeladd | L1 loss | **merged** (intermediate baseline 101.93) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #754 | fern | Per-channel pressure 3× (L1 retest) | **MERGED** (new baseline 99.23) |
| #755 | frieren | slice_num 64→128 | **closed** (wall-clock cost cancels per-epoch gain) |
| #757 | nezuko | 5% warmup + cosine | sent back (retest on baseline 99.23) |
| #758 | tanjiro | lr=1e-3 + 10% warmup (L1 retest) | **closed** (+9.7% vs L1 baseline) |
| #760 | thorfinn | batch_size 4→8 | closed (during send-back cycle) |
| #797 | askeladd | NaN/Inf guard (scope expanded) | wip |
| #816 | alphonse | **Round-2 #2:** FiLM conditioning | wip |
| #818 | tanjiro | **Round-2 #9:** SGDR warm restarts | wip |
| #819 | frieren | **Round-2 #1:** Relative L2 loss | wip |
| #820 | thorfinn | **Round-2 #3:** Fourier PE on (x,z) | wip |
| #829 | fern | **Round-2:** p-channel weight sweep (5×, 10×) | wip |

## Round 1 status snapshot (2026-04-28 ~20:15)

| PR | Student | Topic | Status |
|----|---------|-------|--------|
| #749 | alphonse | Capacity scale-up | wip |
| #752 | askeladd | L1 loss | merged (baseline 101.93) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #754 | fern | Per-channel pressure 3× | sent back to retest on L1 |
| #755 | frieren | slice_num 64→128 | wip |
| #757 | nezuko | 5% warmup + cosine | wip |
| #758 | tanjiro | lr=1e-3 + 10% warmup | sent back to retest on L1 |
| #760 | thorfinn | batch_size 4→8 | wip |
| #797 | askeladd | NaN/Inf guard (scope expanded) | wip |

## Round 1 status snapshot (2026-04-28 ~20:00)

| PR | Student | Topic | Status |
|----|---------|-------|--------|
| #749 | alphonse | Capacity scale-up | wip |
| #752 | askeladd | L1 loss | **merged** (new baseline) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #754 | fern | Per-channel pressure 3× | wip |
| #755 | frieren | slice_num 64→128 | wip |
| #757 | nezuko | 5% warmup + cosine | wip |
| #758 | tanjiro | lr=1e-3 + 10% warmup | sent back to retest on L1 |
| #760 | thorfinn | batch_size 4→8 | wip |

Next assignment: askeladd → cruise-test NaN/Inf guard fix (unblocks
`test_avg/mae_surf_p` for all future round-1 reviews).
