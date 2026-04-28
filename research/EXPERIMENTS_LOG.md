# SENPAI Research Results — charlie-pai2d-r3

## 2026-04-28 08:27 — PR #626 (CLOSED): BF16 + broader FP32 pred cast (entire pred to FP32)
- Branch: `charliepai2d3-edward/l1ff12-ema-cos14-lr-7p5e-4-bf16-broadguard` (deleted)
- Hypothesis: BF16 autocast everywhere + FP32 cast on entire `pred` tensor for both surf_loss AND aux log-p (vs PR #606's narrow guard on `pred[..., 2]` only). Tests whether broader loss-side precision restoration is sufficient.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | this PR (BF16+broad) | current advisor PR #578 (FP32) | Δ |
|--------|---------------------:|-------------------------------:|--:|
| `val_avg/mae_surf_p` | 78.01 | 75.78 | **+2.94% REGRESSION** |
| `test_avg/mae_surf_p` | 67.66 | 66.27 | +2.09% |
| Wallclock per epoch | ~101 s | ~132 s | **−23.1%** ✓ |
| Peak GPU memory | 33.49 GB | 42.51 GB | **−21.2%** ✓ |

Speedup preserved. Outcome #3 (speedup disappears) firmly rejected.

### Per-split val (the inverse of PR #578's gain pattern)

| split | this PR | PR #578 baseline | Δ% |
|-------|--------:|-----------------:|--:|
| val_single_in_dist | 90.52 | 84.61 | **+6.99%** ↑ |
| val_geom_camber_rc | 91.53 | 85.83 | **+6.64%** ↑ |
| val_geom_camber_cruise | 54.88 | 58.09 | **−5.53%** ↓ |
| val_re_rand | 75.12 | 74.58 | +0.72% |

**Critical pattern**: PR #578 lifted in-dist −7.18% and rc-camber −5.45% via decoupled head LR (2×). This run loses the same splits in roughly the same magnitude (+6.99% / +6.64%) while OOD-cruise improves. **The pattern is the inverse of PR #578's gains.**

### Analysis

Loss-side broad FP32 guard cleanly restores precision (verified by speed/memory matching BF16 fingerprint), but **headline regresses worse than PR #606's narrow guard (+2.61%)**. Within run-to-run noise (±0.5% typical), so loss-side guarding has no marginal value vs upstream — the precision loss is upstream.

**Mechanistic reading**: BF16 in attention (PhysicsAttention slice softmax + matmuls) and the gradients flowing back to mlp2/ln_3 erode the fine-grained signal that the head's 2× decoupled LR was designed to consume. With BF16 attention, the head's 2× boost amplifies *noise* instead of *signal*. The forward-pass `pred` is downstream of attention — casting it to FP32 cannot recover precision already lost upstream.

Cost-benefit: +2.9% val cost wipes out PR #578's full +1.6% gain in exchange for 24% wallclock savings. **Unfavourable until attention precision is solved.**

### Decision: CLOSED

The "BF16 with loss-side FP32 guard" lever is now strongly characterised across PR #606 (narrow) and PR #626 (broad). Both regress similarly (+2.61% / +2.94%). Loss-side guarding is dead.

Reassigning edward to **BF16 + `autocast(enabled=False)` block inside PhysicsAttention.forward** (PR #655). This is the highest-EV next experiment — the inverse-of-#578 per-split signature is the strongest evidence yet that attention is the precision-loss site.

---

## 2026-04-28 08:27 — PR #625 (CLOSED): decoupled head LR 3× (bracket up from 2×)
- Branch: `charliepai2d3-thorfinn/l1ff12-ema-cos14-lr-7p5e-4-decouple-head-3x` (deleted)
- Hypothesis: bracket head LR multiplier upward from PR #578's 2× to 3× to test if optimum is past 2× (PR #578 best-val at ep 14/14 with monotone descent suggested room for more).

### Headline (best-val checkpoint, epoch 14/14)

| Metric | 3× (this PR) | 2× baseline (PR #578) | Δ |
|--------|-------------:|----------------------:|--:|
| `val_avg/mae_surf_p` | 78.10 | 75.78 | **+3.06% REGRESSION** |
| `test_avg/mae_surf_p` | 68.56 | 66.27 | +3.46% |

### Per-split val

| split | 3× | 2× baseline | Δ% |
|-------|---:|------------:|---:|
| val_single_in_dist     | 89.55 | 84.61 | **+5.84% ← worst regressor** |
| val_geom_camber_rc     | 88.41 | 85.83 | +3.00% |
| val_geom_camber_cruise | 57.82 | 58.09 | −0.46% (flat) |
| val_re_rand            | 76.63 | 74.58 | +2.75% |

### Analysis

3× moves past the head-LR optimum. Critically, **`val_single_in_dist` is the worst regressor at +5.84%** — opposite of the prior expectation. Mechanistic refinement of PR #578's story:

PR #578 said "head fits in-dist patterns slowly under conservative backbone LR" — directionally correct, bounded incorrectly. The actual story is **the head-LR sweet spot trades head-convergence speed against gradient noise on high-magnitude splits**:
- 1× (pre-#578): head convergence too slow.
- 2× (PR #578 optimum): balanced.
- 3× (this PR): gradient noise on high-magnitude in-dist (y_std up to 2,077) overwhelms convergence speed.

Head-LR axis is now bracketed:
- 1× → 75.78 → −1.60% from prior baseline
- 2× → 75.78 (current) ← optimum
- 3× → 78.10 → +3.06% above 2×

### Decision: CLOSED

Round-5 stops bracketing the head-LR multiplier upward.

Reassigning thorfinn to **head-only weight decay (5e-4 on head, 1e-4 elsewhere)** (PR #656) — keeps HEAD_LR_MULTIPLIER=2.0 (the proven optimum) and tests whether targeted regularisation absorbs the in-dist over-fit signal observed at 3×. Mechanistic prior: head's higher effective LR makes it sensitive to wd; small targeted dose may recover further headroom without dragging on rc-camber.

Filed for round-5 follow-up: per-block head decoupling (mlp2 + ln_3 of last block + final per-channel output projection at higher multiplier) and head-LR warmup as alternative regularisation routes.

---

## 2026-04-28 08:27 — PR #617 (CLOSED): cosine eta_min=5e-5 (floor LR through cosine tail)
- Branch: `charliepai2d3-fern/l1ff-ema-cos14-eta5e-5-lr-7p5e-4` (deleted)
- Hypothesis: set CosineAnnealingLR `eta_min=5e-5` (vs default 0). Holds floor LR through the cosine tail to keep gradient signal flowing for EMA shadow stabilisation.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | this PR | branched-base PR #596 | current advisor PR #578 |
|--------|--------:|----------------------:|------------------------:|
| `val_avg/mae_surf_p` | 76.47 | 77.01 (**−0.54% won on base**) | 75.78 (**+0.91% REGRESSION**) |
| `test_avg/mae_surf_p` | 66.95 | 67.78 (−1.22%) | 66.27 (+1.03%) |

### Per-split val (Pareto trade)

| split | this PR | PR #596 baseline | Δ% |
|-------|--------:|-----------------:|--:|
| val_single_in_dist | 90.08 | 85.42 | **+5.46% ↑** |
| val_geom_camber_rc | 87.39 | 88.01 | −0.70% |
| val_geom_camber_cruise | 54.85 | 58.13 | **−5.65% ↓** |
| val_re_rand | 73.57 | 76.48 | **−3.81% ↓** |

3 of 4 splits improve, but in-dist regression is large enough to push val_avg above current baseline.

### Analysis

Same merge-order pattern as PR #437/#395/#596 — lever validated on its branched base but doesn't compose with the post-#578 head-LR mechanism merged after assignment. **Mechanistic reading**: floor LR keeps late-epoch gradient flowing → OOD splits benefit from continued EMA-shadow tracking (predicted), BUT in-dist over-fits because PR #578's head LR 2× already amplifies late-epoch updates on the head — adding non-zero floor LR compounds that into in-dist over-training. The lever and head-LR don't compose orthogonally.

### Decision: CLOSED

Within-class brackets (eta_min=1e-5 / 1e-4) deferred — per-split signal already characterised; the in-dist/OOD tradeoff is intrinsic to the lever-mechanism interaction, not a dose issue.

Reassigning fern to **layer scale (CaiT-style residual gating)** (PR #657) — mechanistically distinct architectural axis untouched in round 3. Adds learnable per-channel scalars (γ_init=1e-4) to each residual branch. Strong precedent in modern ViT recipes (CaiT, DeiT III, DINOv2). Single architectural change with deterministic alternative to closed DropPath/stochastic-depth attempts.

---

## 2026-04-28 08:13 — PR #607 (CLOSED): annealed input noise sigma=0.05→0
- Branch: `charliepai2d3-nezuko/l1ff12-ema-cos14-lr-7p5e-4-anneal-noise` (deleted)
- Hypothesis: Linear-anneal input-space Gaussian noise (sigma 0.05 → 0 over 12 epochs, noise-free for ep13-14) front-loads regularisation while letting late-epoch fine-tuning converge cleanly. Addresses PR #569's "best epoch is final epoch" diagnostic.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | annealed (this PR) | branched-base PR #572 | current advisor PR #578 |
|--------|------------------:|----------------------:|------------------------:|
| `val_avg/mae_surf_p` | 78.30 | 77.78 (+0.67%) | 75.78 (**+3.32% REGRESSION**) |
| `test_avg/mae_surf_p` | 68.44 | 67.71 (+1.08%) | 66.27 (+3.27%) |

### Per-split val

| split | branched base (#572) | this PR (annealed) | Δ |
|-------|---------------------:|-------------------:|--:|
| val_single_in_dist     | 92.62 | 92.69 | +0.08% (flat) |
| val_geom_camber_rc     | 91.34 | 90.80 | **−0.59%** (mild win) |
| val_geom_camber_cruise | 52.94 | 54.06 | +2.12% (regressed) |
| val_re_rand            | 74.21 | 75.65 | +1.94% (regressed) |

### Annealed vs PR #569 fixed σ=0.05

| split | fixed σ (#569) | annealed (this PR) | shift |
|-------|---------------:|-------------------:|-------|
| val_single_in_dist     | −1.04% | +0.08% | won → flat (lost the win) |
| val_geom_camber_rc     | +2.50% | −0.59% | **regression → win** |
| val_geom_camber_cruise | +5.49% | +2.12% | **regression halved** |

### Analysis

Anneal partially validated the late-epoch hypothesis: rc-camber flipped from regression to win, cruise regression halved. **But** in-dist lost its win and re_rand regressed — net val_avg worsens. 3 of 4 splits regress or flat.

Both PR #569 (fixed σ) and this PR (annealed σ) provide a **strong null on input-space additive noise** as a lever on this stack. The student's own conclusion: "input-noise lever appears to be at the regularisation knee on round-3's stack" (EMA + L1 + aux log-p + grad clipping is saturating the regularisation budget).

### Decision: CLOSED

Within-class brackets (anneal duration 6 vs 12 epochs, position-channel-only noise) deferred — per-split signal already characterised. Student's suggestion #3 (different regulariser class — SWA-style averaging beyond EMA, target-space transforms, physics-informed constraints) noted for round-5 mechanism-class change.

Reassigning nezuko to **slice_num bracket downward (64 → 32)** — architectural axis untouched in round 3 except PR #292's bracket up (128, +45% regression). Mechanistically distinct from all merged levers (which are loss/encoding/optimisation, not architecture). Tests whether 64 slices over-parametrise the slice-routing softmax for 1500-sample regime.

---

## 2026-04-28 08:02 — PR #616 (CLOSED): max_norm=10.0 (continue bracketing clip up)
- Branch: `charliepai2d3-askeladd/l1ff-ema-cos14-lr-7p5e-4-clip10` (deleted)
- Hypothesis: Continue bracketing clip from 5.0 → 10.0 to test whether clip-tightness optimum is at-or-below 5.0 or further loosening still helps.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | max_norm=10 | branched-base PR #596 (5.0) | current advisor baseline PR #578 |
|--------|-----------:|----------------------------:|---------------------------------:|
| `val_avg/mae_surf_p` | 76.78 | 77.01 (−0.30%) | 75.78 (**+1.32% REGRESSION**) |
| `test_avg/mae_surf_p` | 66.52 | 67.78 (−1.86%) | 66.27 (+0.38%) |

### Per-split val

| split | branched base (5.0) | this PR (10.0) | Δ |
|-------|--------------------:|---------------:|--:|
| val_single_in_dist     | 85.42 | 88.61 | **+3.74%** ↑ |
| val_geom_camber_rc     | 88.01 | 90.32 | **+2.62%** ↑ |
| val_geom_camber_cruise | 58.13 | **54.06** | **−7.00%** ↓ |
| val_re_rand            | 76.48 | 74.15 | −3.05% ↓ |

### Clip dynamics

| Epoch | clip_frac (gn>10) | gn_mean | gn_max |
|-------|------------------:|--------:|-------:|
| 1  | 1.000 | 49.21 | 135.84 |
| 7  | 1.000 | 46.93 | 134.47 |
| 13 | 0.997 | 32.41 | 115.90 |
| 14 | 0.992 | 30.28 | 160.95 |

`gn_mean` settles ~30 — 3× the new clip threshold; clip still binds 99.2% of batches in the last epoch. Natural grad-norm scale of the model is past 30.

### Analysis

Marginal win on the branched base (PR #596) but **regresses against the current advisor stack (post-#578)**. The clip-loosening direction has diminishing returns: −0.99% at 5× → −0.30% at 10×. Per-split signal is informative — cruise wants looser clip (54.06, recovered below pre-clip baseline 52.93), in-dist and rc-camber want tighter clip (both regress meaningfully). This is a Pareto trade between OOD splits, not a Pareto improvement.

Stack-on-stack question (does max_norm=10 still help on the post-#578 stack with decoupled head LR?) is untested. But given:
- diminishing val_avg returns,
- per-split tradeoff is now the dominant signal,
- merged stack already includes max_norm=5.0,

continued vertical bracketing (max_norm=20/50) is lower-priority than orthogonal axes.

### Decision: CLOSED

Per-domain LR / surf_weight (askeladd's suggestion #3) queued as future hypothesis.

Reassigning askeladd to **width-bracket of decoupled head LR (PR #578)**: extending the head set from `mlp2 + ln_3` to the full late-block MLP path (`+ mlp + ln_2`), keeping multiplier at 2×. Complements thorfinn's PR #625 (vertical bracket: 2× → 3×) with horizontal bracket (same multiplier, more parameters).

---

## 2026-04-28 08:10 — PR #597 (CLOSED): aux log-p weight=0.10 (bracket down from 0.25)
- Branch: `charliepai2d3-tanjiro/l1ff-ema-cos14-lr-7p5e-4-logp-aux-0p10` (deleted)
- Hypothesis: Bracket auxiliary log-pressure weight from 0.25 to 0.10 — if the non-monotone cruise behavior continues growing at lower weight, the optimum is below 0.25.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | weight=0.10 (this PR) | weight=0.25 PR #572 | vs current baseline PR #578 (75.78) |
|--------|-----------------------:|--------------------:|------------------------------------:|
| `val_avg/mae_surf_p` | 77.54 | 77.78 | **+2.32% REGRESSION** |
| `test_avg/mae_surf_p` | 68.10 | 67.71 | +2.76% REGRESSION |

### Per-split val

| split | weight=0.25 (PR #572) | weight=0.10 (this PR) | Δ |
|-------|---------------------:|-----------------------:|--:|
| val_single_in_dist | 92.62 | **86.58** | −6.52% (gain!) |
| val_geom_camber_rc | 91.34 | 90.74 | −0.66% |
| val_geom_camber_cruise | 52.94 | **57.29** | **+8.22%** (regression) |
| val_re_rand | 74.21 | 75.54 | +1.79% |

### Analysis

Polarity flip observed. At w=0.25: cruise improved −5.74% but single_in_dist regressed +1.61%. At w=0.10: exactly reversed — cruise regresses +8.22% while single_in_dist improves −6.52%. The aux log-p weight lever is a **capacity-redistribution knob between splits** with a polarity-flip somewhere in (0.10, 0.25). val_avg stays nearly flat (77.54 vs 77.78) because the per-split moves cancel.

Critically, neither 0.10 nor 0.25 beat the current PR #578 baseline (75.78), which already includes decoupled head LR. The aux log-p lever appears orthogonal to or dominated by the head-LR mechanism.

### Decision: CLOSED

Closed as dead end vs current baseline (77.54 vs 75.78, −2.32% regression). The polarity-flip insight is mechanistically useful but the aux log-p family does not improve on the PR #578 baseline at any tested weight. Tanjiro re-assigned to a new direction.

---

## 2026-04-28 07:25 — PR #578 (MERGED): decoupled head LR (2× on `mlp2`+`ln_3`)
- Branch: `charliepai2d3-thorfinn/l1ff12-ema-cos14-lr-7p5e-4-decouple-head-2x`
- Hypothesis: 2× head LR vs backbone — head adapts faster to OOD-camber.
  Predicted −0.5% to −1.5%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #534 (assigned baseline 78.60) | vs current PR #596 (77.01) |
|--------|--------:|------------------------------------:|---------------------------:|
| `val_avg/mae_surf_p` | **75.78** | **−3.59%** ✓ 2.4× upper prediction | **−1.60%** |
| `test_avg/mae_surf_p` | **66.27** | **−2.21%** | **−2.23%** |

### Per-split val — opposite of predicted direction!

| split | this PR | PR #534 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 84.61 | 91.15 | **−7.18%** (largest gain — predicted neutral) |
| val_geom_camber_rc | 85.83 | 90.78 | **−5.45%** |
| val_geom_camber_cruise | 58.09 | 56.16 | +3.44% (mild regression) |
| val_re_rand | 74.58 | 76.33 | −2.29% |

### Mechanistic insight — corrects askeladd PR #489's interpretation

PR #489 found `lr=1e-3` (uniform) gave OOD-camber-cruise gains and
in-dist regression. Predicted from this: OOD-camber wants higher LR
than in-dist.

**Actual mechanism revealed by decoupled-head LR**: the head fits
in-dist patterns slowly under the conservative 7.5e-4 backbone LR.
Giving the head 2× lets it converge in matched-cosine epochs without
dragging the backbone faster. **In-dist gained MOST** (−7.18%) — the
opposite of the prior. The "OOD-camber needs higher LR" reading was
mixing two effects: head-specific LR sensitivity and backbone LR
sensitivity.

### Decision

**Merged.** Twelfth merge of round 3, eleventh proven stacked lever.
Largest single-knob improvement since the schedule × EMA fix.

### Caveat — measurement on pre-#572 advisor

PR #578 was branched off the pre-#572 / pre-#596 advisor (no aux log-p,
max_norm=1.0). Post-merge advisor stacks all three:
- aux log-p (weight=0.25) from PR #572.
- max_norm=5.0 from PR #596.
- decoupled head LR (2×) from this PR.

The actual joint config is untested but expected to land below 75.78
since PR #572 and PR #596 both individually showed val improvements
on their assigned baselines.

### Round-3 baseline lineage updated (12 merges)

| PR | val | test | lever |
|----|----:|-----:|-------|
| 280 | 102.64 | 97.73 | + L1 surface |
| 400 | 91.87 | 81.11 | + 8-freq spatial FF |
| 447 | 82.97 | 73.58 | + EMA(0.999) |
| 461 | 80.28 | 70.92 | + lr=7.5e-4 |
| 462 | 80.06 | 70.04 | + grad clipping (1.0) |
| 506 | 78.80 | 69.13 | + FF=12 |
| 534 | 78.60 | 67.77 | + EMA=0.997 |
| 572 | 77.78 | 67.71 | + aux log-p (weight=0.25) |
| 596 | 77.01 | 67.78 | + max_norm=5.0 |
| **578** | **75.78** | **66.27** | + **decoupled head LR (2×)** |

Cumulative −43.9% on val, −46.2% on test from PR #306 reference.

Re-assigning thorfinn to **3× head LR multiplier** — best-val at ep
14/14 monotone-descending suggests head-LR optimum is past 2×.

---

## 2026-04-28 07:19 — PR #606 (CLOSED, narrow guard insufficient): BF16 + FP32 surf_loss guard
- Branch: `charliepai2d3-edward/l1ff12-ema-cos14-lr-7p5e-4-bf16-fp32guard` (deleted)
- Hypothesis: FP32 cast on `pred[..., 2]` before L1 reduction
  eliminates BF16 precision loss while preserving speedup.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #572 (77.78 / 67.71) | vs current PR #596 (77.01 / 67.78) |
|--------|--------:|---------------------------:|------------------------------------:|
| `val_avg/mae_surf_p` | 79.81 | +2.61% | +3.64% |
| `test_avg/mae_surf_p` | 69.59 | +2.78% | +2.67% |
| Per-epoch wallclock | 100.6 s | (vs 132 s, **−23.8%**) | |
| Peak GPU memory | 33.47 GB | (vs 42.5 GB, **−21.3%**) | |

### Per-split val — distributional regression (broader precision loss)

| split | this PR | PR #572 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 95.56 | 92.62 | +3.17% |
| val_geom_camber_rc | 91.52 | 91.34 | +0.20% |
| val_geom_camber_cruise | 55.97 | 52.94 | +5.72% (worst hit) |
| val_re_rand | 76.17 | 74.21 | +2.64% |

### Decision

**Closed.** Targeted FP32 guard insufficient — precision loss is
broader than the L1 reduction. Speedup preserved (−24% wallclock,
−21% memory) confirms the FP32 cast doesn't introduce conversion
costs.

**Likely culprits** (per student's analysis):
1. PhysicsAttention slice softmax: denominators accumulate over
   `slice_num=64` × 242K nodes; BF16 8-bit mantissa too short.
2. Aux log-p loss path: computed outside autocast but reads
   `pred[..., 2]` (BF16) — gradient flows through BF16 attention.

Cruise/re_rand backslide (the splits PR #572's aux log-p helped
most) is consistent with the second culprit.

Re-assigning edward to **broader FP32 `pred` cast for both surf_loss
and aux log-p loss** (student's follow-up #2) — larger memory cost
but addresses the loss-side precision path comprehensively.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 07:04 — PR #596 (MERGED): max_norm=1.0 → 5.0 (loosened clipping)
- Branch: `charliepai2d3-askeladd/l1ff-ema-cos14-lr-7p5e-4-clip5`
- Hypothesis: loosen clip 5× to unlock real LR sensitivity dampened
  by aggressive clipping. Predicted ±1% (mostly diagnostic).

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #572 baseline (77.78 / 67.71) |
|--------|--------:|-----------------------------------:|
| `val_avg/mae_surf_p` | **77.01** | **−0.99%** |
| `test_avg/mae_surf_p` | 67.78 | +0.10% (essentially flat) |

### Per-split val (best epoch 14/14)

| split | this PR | PR #572 baseline | Δ |
|-------|--------:|-----------------:|--:|
| val_single_in_dist | 85.42 | 92.62 | **−7.78%** (largest gain) |
| val_geom_camber_rc | 88.01 | 91.34 | **−3.65%** |
| val_geom_camber_cruise | 58.13 | 52.94 | +9.81% (regression) |
| val_re_rand | 76.48 | 74.21 | +3.06% (regression) |

### Critical diagnostic — clip still fires on 100% of batches

Pre-clip mean grad-norm 22-47 throughout training, never drops near
5.0. **The clip is still strictly active at max_norm=5.0** — just
allows ~5× larger effective steps than at max_norm=1.0.

This means the previously-measured "narrow LR optimum at 7.5e-4" was
a **clip × LR joint sensitivity**. Loosening clip shifts the joint
operating point: bigger steps via the clip, with the same LR.

### Decision

**Merged.** Eleventh merge of round 3. Val win at noise floor + test
essentially flat. Follows the merge criterion ("lower than current
baseline, even by a small amount").

### Round-3 narrative

Round-3 has now closed two interior optima that both turned out to
be clip-coupled:
- **LR optimum at 7.5e-4** (PR #461, #516, #566) — measured under
  max_norm=1.0; turns out to be the clip-tightness joint optimum.
- **EMA decay optimum at 0.997** (PR #534, #565) — similar story
  may apply; round-5 should consider re-bracketing EMA decay at
  the new clip threshold.

### Best epoch shifted

Best val landed at ep 14/14 (vs 12/14 at PR #534/#572). Looser clip
allows continued progress through the cosine tail rather than
plateauing earlier. May still have headroom at longer schedules.

### Round-3 baseline lineage updated (11 merges)

| PR | val | test | lever |
|----|----:|-----:|-------|
| 280 | 102.64 | 97.73 | + L1 surface |
| 400 | 91.87 | 81.11 | + 8-freq spatial FF |
| 447 | 82.97 | 73.58 | + EMA(0.999) |
| 461 | 80.28 | 70.92 | + lr=7.5e-4 |
| 462 | 80.06 | 70.04 | + grad clipping (max_norm=1.0) |
| 506 | 78.80 | 69.13 | + FF=12 |
| 534 | 78.60 | 67.77 | + EMA=0.997 |
| 572 | 77.78 | 67.71 | + aux log-p (weight=0.25) |
| **596** | **77.01** | **67.78** | + **max_norm=5.0 (loosen clip)** |

Cumulative −43.0% on val, −45.0% on test from PR #306 reference.

Re-assigning askeladd to **max_norm=10.0** — continue bracketing up
toward the natural grad-norm scale.

---

## 2026-04-28 07:00 — PR #588 (CLOSED, SWA mechanistically unsound): SWA last-4-epochs
- Branch: `charliepai2d3-fern/l1ff-ema-swa4-cos14-lr-7p5e-4` (deleted)
- Hypothesis: SWA over last 4 epochs as snapshot-ensemble distinct
  from EMA's continuous averaging. Predicted small effect on val,
  larger on test.

### Headline (best-val checkpoint, epoch 14/14, SWA-averaged for test)

| Metric | This PR | vs PR #534 (78.60 / 67.77) |
|--------|--------:|---------------------------:|
| `val_avg/mae_surf_p` | 78.81 | +0.27% (within noise) |
| `test_avg/mae_surf_p` | 69.64 | **+2.76% (regressed)** |

### Per-split test (SWA-averaged) — OOD-camber-rc worst hit

| split | this PR | PR #534 | Δ |
|-------|--------:|--------:|--:|
| test_single_in_dist | 77.31 | 77.27 | +0.05% (flat) |
| test_geom_camber_rc | 83.25 | 78.98 | **+5.41%** (worst) |
| test_geom_camber_cruise | 49.74 | 48.03 | +3.55% |
| test_re_rand | 68.29 | 66.80 | +2.23% |

### Decision

**Closed.** Above-threshold test regression, mechanistically explained.

### Mechanistic explanation

Per-epoch val trajectory (this run): 82.78 → 80.70 → 79.40 → 78.81
across epochs 11-14. Each step is ~1.5-2.5% improvement — the model
is **still actively converging at the end of training**. SWA
averages a mix of partly-trained and fully-trained weights, moving
test backward.

**SWA's classical regime (Izmailov et al. 2018) requires a flat or
cyclic LR plateau where snapshots come from comparable basins**. Here
we have the opposite — sharply annealed cosine schedule where epochs
11-13 are demonstrably worse than epoch 14. SWA's 4-epoch window is
too wide for this schedule.

### Round-3 narrative

Same mechanism family as PR #476 (matched cosine × EMA(0.999) → +5%
rc-camber regression). The fix that worked there (EMA(0.997), shorter
window) doesn't generalise to SWA — SWA's window is necessarily wider
(epoch-level snapshots) than EMA's (step-level continuous averaging).

EMA(0.997) keeps the prize on trajectory averaging because its window
is short enough to track active convergence. SWA cannot get there
without a constant-LR plateau in the schedule.

Re-assigning fern to **eta_min=5e-5 in CosineAnnealingLR** —
addresses the schedule shape (keeps a small floor LR through cosine
tail) rather than the averaging method.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 06:37 — PR #587 (CLOSED, BF16 precision regression): BF16 autocast
- Branch: `charliepai2d3-edward/l1ff12-ema-cos14-lr-7p5e-4-bf16` (deleted)
- Hypothesis: BF16 autocast for 1.5-2× speedup, ±0.5% headline.
  Round-5 throughput infrastructure.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #534 (78.60) | vs current PR #572 (77.78) |
|--------|--------:|-------------------:|---------------------------:|
| `val_avg/mae_surf_p` | 79.53 | +1.19% | +2.25% |
| `test_avg/mae_surf_p` | 69.38 | +2.37% | +2.46% |
| Per-epoch wallclock | 100 s | (vs 132 s, **−24%**) | |
| Peak GPU memory | 33.4 GB | (vs 42.4 GB, **−21%**) | |

### Diagnostic — partial speedup, precision regression

- Speed unlock real but smaller than predicted (24% vs predicted 50-100%):
  Transolver is **memory-bandwidth-bound**, not compute-bound — pointwise
  ops on irregular meshes dominate.
- Memory drop partial because FP32 master weights, gradients, optimiser
  state, and pad-to-max-mesh batching aren't touched by autocast.
- **`val_single_in_dist` regressed +1.7%** (vs +0.4% on cruise) —
  high-Re extreme pressures (max ±29,000 Pa) lose precision in BF16's
  reduced mantissa during surf_loss reduction.

### Decision

**Closed.** Above-zero regression on headline. Speed unlock alone
doesn't justify merging; round-5 needs precision-guarded variant.

### Round-5 unlock implications

With ~7 min wallclock margin at 14 epochs, BF16 enables previously-
blocked levers:
- DropPath at 0.05 (was 13/14 at FP32): should fit 14/14 with margin.
- slice_num=128 (was 11/14): should fit 14/14.
- Wider+deeper partial unlock (8-10/50 epochs vs 7/50 prior).

### Re-assignment

edward → BF16 + **FP32 surf_loss guard** — keep autocast on forward
pass, but cast `pred[..., 2]` (pressure channel) to FP32 before L1
reduction. Surgical precision restoration without giving up speedup.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 06:37 — PR #569 (CLOSED, over-regularisation): input-space Gaussian noise sigma=0.05
- Branch: `charliepai2d3-nezuko/l1ff12-ema-cos14-lr-7p5e-4-input-noise-0p05` (deleted)
- Hypothesis: input-space additive Gaussian noise for input-perturbation
  robustness. Predicted −0.5% to −2%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #534 (78.60) |
|--------|--------:|-------------------:|
| `val_avg/mae_surf_p` | 79.92 | +1.68% |
| `test_avg/mae_surf_p` | 69.80 | +3.00% |

### Per-split — noise over-regularises OOD

| split | this PR | PR #534 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 90.20 | 91.15 | **−1.04%** (only winner) |
| val_geom_camber_rc | 93.05 | 90.78 | +2.50% |
| val_geom_camber_cruise | 59.24 | 56.16 | **+5.49%** (worst hit) |
| val_re_rand | 77.19 | 76.33 | +1.12% |

### Mechanistic insight

Sigma=0.05 in normalised input space perturbs spatial coordinates by
~5% per step — **on the same order as camber-shift between cruise and
raceCar foils**. Model has to learn the mapping AND denoise inputs
simultaneously with only ~1500 samples. OOD camber extrapolation hurts
because noised training landscape produces less reliable boundaries.

### Decision

**Closed.** Above-zero regression. Round-3 stack is at the
**regularisation knee** — explains why several recent regularisation
levers (wd, beta2, DropPath, channel weighting) have been redundant
or destructive when stacked.

### Re-assignment

nezuko → **annealed input noise** (sigma 0.05 → 0 linearly across
epochs, noise-free for last 2-3 epochs). Front-loads regularisation
while letting late-epoch fine-tuning converge without noise.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 06:28 — PR #572 (MERGED): aux log-pressure loss at weight=0.25
- Branch: `charliepai2d3-tanjiro/l1ff-ema-cos14-lr-7p5e-4-logp-aux-0p25`
- Hypothesis: half PR #551's weight (0.5 → 0.25) for a sweet spot
  below the capacity-competition regime. Predicted −0% to −1.5%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #534 baseline (78.60) |
|--------|--------:|----------------------------:|
| `val_avg/mae_surf_p` | **77.78** | **−1.06%** (just above noise floor) |
| `test_avg/mae_surf_p` | **67.71** | −0.09% (essentially tied) |

### Per-split tradeoff — consistent val/test

| split | val Δ | test Δ |
|-------|------:|------:|
| single_in_dist | +1.61% | +2.91% |
| geom_camber_rc | +0.62% | +1.53% |
| geom_camber_cruise | **−5.74%** | **−4.29%** |
| re_rand | −2.78% | −2.46% |

Cruise/re_rand (low-magnitude pressure) improve cleanly;
single_in_dist/rc-camber (high-magnitude pressure) regress mildly.
**Direction consistent val/test** — confirms this is a real per-split
tradeoff, not val checkpoint overfitting.

### Decision

**Merged.** Tenth merge of round 3. Marginal val win at noise floor
plus tied test, per the merge criterion ("lower than current
baseline, even by a small amount").

### Mechanistic reading — opposite of the original hypothesis

The aux loss was hypothesised as a heavy-tail emphasiser (compress
extreme pressure values). **The data shows it acts as a low-magnitude
emphasiser** — pulls the model toward predicting low-magnitude
pressure cleanly (where log compression has well-defined gradient
signal) at the cost of high-magnitude pressure fidelity. The
non-monotone cruise behaviour vs PR #551 (cruise improvement *grew*
with smaller weight, opposite the predicted dose-response) reinforces
this reading.

### Round-3 narrative — per-split-tradeoff lever family

This is the third round-3 lever showing per-split tradeoffs (after
PR #383 alphonse 3× p-surface-weight on L1-only, in flight, and
PR #551 aux log-p at 0.5). Both flat-or-marginal headline with clear
per-split direction. Qualitatively different from the additive-
distributional levers that dominate the merged stack.

### Round-3 baseline lineage updated (10 merges)

| PR | val | test | lever |
|----|----:|-----:|-------|
| 280 | 102.64 | 97.73 | + L1 surface |
| 400 | 91.87 | 81.11 | + 8-freq spatial FF |
| 447 | 82.97 | 73.58 | + EMA(0.999) |
| 461 | 80.28 | 70.92 | + lr=7.5e-4 |
| 462 | 80.06 | 70.04 | + grad clipping |
| 506 | 78.80 | 69.13 | + FF=12 |
| 534 | 78.60 | 67.77 | + EMA=0.997 |
| **572** | **77.78** | **67.71** | + aux log-p (weight=0.25) |

Cumulative −42.5% on val, −45.0% on test from PR #306 reference.

Re-assigning tanjiro to **weight=0.10** (half current 0.25) — the
non-monotone behaviour suggests further headroom may exist below.

---

## 2026-04-28 06:09 — PR #566 (CLOSED, LR bracket closure): lr=7e-4 on EMA stack
- Branch: `charliepai2d3-askeladd/l1ff-ema-cos14-lr-7e-4` (deleted)
- Hypothesis: bracket LR downward from 7.5e-4 to 7e-4 on EMA-augmented
  stack. Predicted −0% to −1%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current PR #534 (78.60) |
|--------|--------:|---------------------------:|
| `val_avg/mae_surf_p` | 80.66 | +2.62% (under-convergence) |
| `test_avg/mae_surf_p` | 70.13 | +3.48% |

### LR bracket fully characterised

| lr | val | best epoch |
|---:|----:|-----------:|
| 7e-4 (this PR) | 80.66 | **14/14** under-converged |
| **7.5e-4 (current)** | **78.60** | **12/14** converged |
| 8e-4 (#516) | 80.32 | — |
| 1e-3 (#489) | 82.08 | — |

**Narrow optimum at 7.5e-4** — both ±0.5e-4 perturbations regress
~+2-3%. The EMA stack has a **sharp LR well**, not a wide plateau.

### Per-split signal — under-convergence dominates

`val_single_in_dist` regressed +4.28% (worst-hit split), opposite of
the mechanistic prior that EMA wants more conservative LR for the
in-dist regime. Confirms regression is under-convergence (best at
ep 14/14 still descending), not EMA-LR-asymmetry.

### Decision

**Closed.** Above-zero regression. LR bracket closed: 7.5e-4 is the
optimum on the EMA stack at the 14-epoch budget.

Re-assigning askeladd to **max_norm=5.0** — LR × clip joint axis test.
Their grad-clip diagnostic shows pre-clip mean is 20-50× the current
threshold, so clipping is the effective step-size determinant. Loosening
`max_norm` may unlock the real LR sensitivity dampened by clipping.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 06:07 — PR #565 (CLOSED, EMA decay bracket closure): EMA_DECAY=0.998
- Branch: `charliepai2d3-fern/l1ff-ema998-cos14-lr-7p5e-4` (deleted)
- Hypothesis: bracket EMA decay slightly upward from 0.997 to 0.998.
  Predicted −0% to −1%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current PR #534 (78.60) |
|--------|--------:|---------------------------:|
| `val_avg/mae_surf_p` | 79.26 | +0.84% (marginal regression) |
| `test_avg/mae_surf_p` | 69.13 | +2.01% |

### EMA decay bracket fully characterised

| EMA decay | val | val_re_rand | best epoch |
|----------:|----:|------------:|-----------:|
| 0.999 (PR #476) | 83.89 | 81.08 (badly broken) | 14 |
| **0.997 (PR #534, current)** | **78.60** | **76.33** | **12/14** |
| 0.998 (this PR) | 79.26 | 77.47 | 14/14 |

`val_re_rand` drifting from 76.33 → 77.47 confirms the schedule × EMA
interference creeping back in (less severe than 0.999, same direction).

### Mechanistic insight

EMA decay × epoch budget interaction: with longer 0.998 EMA window
(~500 steps ≈ 1.3 epochs), shadow takes longer to warm up. Best at
ep 14/14 (still descending) vs ep 12/14 at 0.997 (already converged).
Cosine-tail benefit real but warm-up cost exceeds gain at fixed
14-epoch budget. Round-5 at longer budgets may shift the optimum.

### Decision

**Closed.** Above-zero regression. EMA decay axis closed for round 3
at 14-epoch budget — 0.997 is optimum.

Re-assigning fern to **SWA-style end-of-training weight averaging**
(snapshot ensemble at convergence, different mechanism from EMA's
continuous trajectory averaging).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:55 — PR #524 (CLOSED, schedule × EMA confirmation): canonical 6-lever stack at FF=8 + EMA=0.999
- Branch: `charliepai2d3-edward/l1ff-ema-cos14-lr-7p5e-4-clip1-canonical` (deleted)
- Hypothesis: pure measurement of the canonical 6-lever stack
  (L1+FF8+EMA(0.999)+matched cosine+lr=7.5e-4+clip). Predicted
  −2% to −5% if all levers compose cleanly.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #462 (80.06) | vs current PR #534 (78.60) |
|--------|--------:|-------------------:|---------------------------:|
| `val_avg/mae_surf_p` | 82.26 | +2.74% | +4.66% |
| `test_avg/mae_surf_p` | 71.70 | +2.37% | +5.80% |

### Decision

**Closed.** Above-threshold regression vs current baseline. Lands in
the "82-84 hidden compose interaction" band predicted by the PR
body's outcome rubric.

### Round-3 ceiling characterisation (the diagnostic value)

This PR + fern PR #534 + PR #461/#462 together fully characterise
the schedule × EMA × FF interaction:

| config | val | EMA decay | FF |
|--------|----:|----------:|----|
| PR #461 (no EMA) | 80.28 | — | 8 |
| PR #462 (no EMA) | 80.06 | — | 8 |
| **this PR** (EMA=0.999) | **82.26** | 0.999 | 8 |
| PR #476 (EMA=0.999) | 83.89 | 0.999 | 8 |
| **PR #534 (current)** | **78.60** | **0.997** | 12 |

The 4.66% gap between this PR (82.26) and current best PR #534
(78.60) decomposes as:
- ~−1.5% from FF=8 → FF=12 (PR #506).
- ~−3.5% from EMA=0.999 → EMA=0.997 (PR #534 fix).

These two effects roughly add, **confirming both lever moves and
validating fern's schedule × EMA decay-tuning derivation**.

### Round-3 ceiling estimate

The "predicted ~76-78 if levers compose cleanly" from the PR body
assumed independent additive contributions. We now know matched
cosine × EMA(0.999) was destructive — the additive estimate
double-counted that overlap. Post-#534 with EMA(0.997) fix, the
canonical 6-lever stack lands at **78.60 (val) / 67.77 (test)**,
which is the round-3 ceiling at single-replicate.

Re-assigning edward to **BF16 autocast** as round-5 throughput
infrastructure (wallclock has been the binding constraint for
DropPath, slice_num=128, and any larger-model attempts).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:56 — PR #558 (CLOSED, wallclock-binding): slice_num=128 retest
- Branch: `charliepai2d3-frieren/l1ff12-ema-cos14-lr-7p5e-4-slice128` (deleted)
- Hypothesis: retest slice_num=128 on the cleaner post-#506 stack
  (PR #292 was inconclusive on MSE baseline due to ~4% seed noise).
  Predicted −0.5% to −2.5%.

### Headline (best-val checkpoint, epoch 11/14)

| Metric | This PR | vs current PR #534 (78.60) |
|--------|--------:|---------------------------:|
| `val_avg/mae_surf_p` | 91.90 | +16.9% |
| `test_avg/mae_surf_p` | 82.51 | +21.7% |
| Per-epoch wallclock | ~167 s | (vs 133 s baseline, +25%) |
| Peak GPU memory | 54.9 GB | (vs 42.5 GB, +29%) |
| Epochs in 30-min cap | **11/14** | (binding) |

### Decision

**Closed.** All four val splits regressed in tight band (+12% to +23%);
no per-split signal favouring slice tokens. Wallclock-binding
overhead — same family as DropPath PR #532 (close).

slice_num axis closed for round 3. Round-5 unblocking requires
either a longer wallclock or fundamentally different attention-axis
levers.

Re-assigning frieren to n_head=8 (different attention compute
structure than slice tokens).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:42 — PR #532 (CLOSED, wallclock-bound at any DropPath rate): DropPath 0.05
- Branch: `charliepai2d3-thorfinn/l1ff-ema-cos14-lr-7p5e-4-droppath-0p05` (deleted)
- Hypothesis: half rate (0.05 vs PR #501's 0.1) reduces overhead and
  fits 14 epochs cleanly. Predicted −0% to −2%.

### Headline (best-val checkpoint, epoch 13/14)

| Metric | This PR | PR #501 (drop_path=0.1) | vs current PR #534 (78.60 / 67.77) |
|--------|--------:|------------------------:|-----------------------------------:|
| `val_avg/mae_surf_p` | 86.42 | 89.54 | +9.95% |
| `test_avg/mae_surf_p` | 76.63 | 80.56 | +13.07% |
| Per-epoch wallclock | ~140 s | ~140 s | (vs 131 s baseline) |
| Epochs in 30-min cap | **13/14** | **13/14** | (binding) |

### Critical diagnostic — per-batch overhead is rate-independent

Both rates ran at ~140 s/epoch — the per-batch Bernoulli sampling +
broadcast multiply dominates over keep-probability fraction. Halving
the rate did not halve overhead. **DropPath as currently implemented
hits the wallclock cliff at any rate** under the 30-min cap.

### FF interference hypothesis refuted

rc-camber is consistently the *least* regressed split at both rates
(PR #501: rc +9.9% vs cruise +15%; this PR: rc +5.9% vs cruise +8.3%).
DropPath does NOT enter the magnitude-regulariser × FF interference
pattern. Regression is dominated by under-training (13/14 cosine
truncation), not mechanism interference.

### Decision

**Closed.** DropPath axis closed for round 3. Round-5 unblocking
requires either (a) per-batch-mask implementation (`drop_prob_mode='batch'`)
to remove per-sample overhead, or (b) longer schedule beyond 30-min cap.

Re-assigning thorfinn to **decoupled-head LR** — different mechanism
(per-parameter-group optimisation, not regularisation/encoding/schedule).
Motivated by askeladd PR #489's finding that OOD-camber tracks tolerate
larger updates than in-dist; final head adapts faster to OOD geometry
while encoder stays stable.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:41 — PR #551 (CLOSED, weight too high): aux log-pressure loss at 0.5 weight
- Branch: `charliepai2d3-tanjiro/l1ff-ema-cos14-lr-7p5e-4-logp-aux-0p5` (deleted)
- Hypothesis: auxiliary `L1(sign(p)·log1p(|p|))` loss on surface nodes
  with weight=0.5 — different mechanism from channel weighting / loss
  shape / EMA. Predicted −1% to −3%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current PR #534 (78.60 / 67.77) |
|--------|--------:|-----------------------------------:|
| `val_avg/mae_surf_p` | 78.94 | +0.43% (within noise) |
| `test_avg/mae_surf_p` | 69.22 | +2.14% |

### Per-split — clean tradeoff signature (NOT uniform pattern)

| split | val Δ | test Δ | direction |
|-------|------:|------:|-----------|
| single_in_dist | −0.41 | +2.37 | inconsistent (val small win, test loss) |
| geom_camber_rc | +1.72 | +1.70 | capacity competition (regression) |
| geom_camber_cruise | **−1.58** | **−2.08** | clean win (low-mag p compression helps) |
| re_rand | +0.83 | −1.64 | inconsistent |

### Decision

**Closed.** Above-zero regression on test, mixed signal on val.

### Round-3 narrative — first lever with clean per-split tradeoffs

This is the **first round-3 lever to show clean per-split tradeoffs**
rather than uniform improvement / regression. The auxiliary log-
pressure loss has a real axis distinct from EMA's mechanism — at
weight 0.5 the heavy-tail benefit on cruise (low-mag pressure
compression) is real but the OOD-camber-rc capacity competition
trades against it.

The auxiliary lever may have a sweet spot at lower weight. Re-
assigning tanjiro to **weight=0.25** (half the dose) to test.

### Round-3 mechanism map updated

The auxiliary log-pressure loss is **not in the EMA-overlap failure
family** (#492, #500, #489, #515): the per-split signature differs,
the auxiliary trace is healthy, and `train/log_p_aux` decreases
smoothly. The compose pattern is **capacity competition with the
main task** at weight 0.5 — different from the saturating-overlap
patterns documented elsewhere.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:24 — PR #543 (CLOSED, FF upper-bracket closure): NUM_FOURIER_FREQS=16
- Branch: `charliepai2d3-nezuko/l1ff16-ema-cos14-lr-7p5e-4` (deleted)
- Hypothesis: bracket FF dose upward from 12 to 16 freqs. Predicted
  −0% to −2%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #506 (78.80) | vs current PR #534 (78.60) |
|--------|--------:|-------------------:|---------------------------:|
| `val_avg/mae_surf_p` | 81.26 | +3.13% | +3.39% |
| `test_avg/mae_surf_p` | 71.92 | +4.04% | +6.13% |

### Per-split — broad-spectrum regression

All 4 val and all 4 test splits regressed (val +1.2% to +5.6%; test
+3.4% to +6.2%). No single-axis failure mode — pure
representation-capacity-spent-on-noise pattern.

### Mechanistic explanation

At 16 freqs, the highest-frequency basis is `2^15 · π ≈ 102,944`
cycles per normalised unit — far above any meaningful spatial scale
in the meshes. Frequencies 12-15 alias on adjacent nodes and inject
high-frequency noise into the input encoding rather than signal.
The +4096 param cost is small (+0.6%) but the representation cost
is real.

### FF dose response curve characterised

| freqs | val | source |
|------:|---:|--------|
| 4 | 81.31 | PR #533 (closed) |
| 12 | **78.80** | PR #506 (merged) |
| 16 | 81.26 | this PR |

Concave with peak at-or-near 12. Round-5 should NOT continue bracketing
FF up — mechanistic argument predicts regression worsens at 24+.

### Decision

**Closed.** Above-threshold regression. FF dose lever closed for
round 3.

Re-assigning nezuko to **input-space additive Gaussian noise**
(sigma=0.05) — different regularisation axis than FF. Round-5 is
moving past FF dose tuning toward mechanistically novel axes.

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:24 — PR #534 (MERGED): EMA_DECAY 0.999 → 0.997 — schedule × EMA interference fix
- Branch: `charliepai2d3-fern/l1ff-ema997-cos14-lr-7p5e-4`
- Hypothesis: tighten EMA decay to end averaging *before* cosine tail.
  Predicted −1% to −3%.

### Headline (best-val checkpoint, epoch 12/14)

| Metric | This PR | vs current PR #506 (78.80 / 69.13) | vs PR #476 EMA(0.999) (83.89 / 74.53) |
|--------|--------:|-----------------------------------:|----------------------------------------:|
| `val_avg/mae_surf_p`  | **78.60** | **−0.25%** | **−6.31%** |
| `test_avg/mae_surf_p` | **67.77** | **−1.97%** | **−9.07%** |

### Per-split — wins on every val and test split

| split | this PR | PR #506 baseline | Δ (val) |
|-------|--------:|-----------------:|--------:|
| val_single_in_dist | 91.15 | 92.73 | −1.7% |
| val_geom_camber_rc | 90.78 | 89.86 | +1.0% |
| val_geom_camber_cruise | 56.16 | 57.32 | −2.0% |
| val_re_rand | 76.33 | 75.30 | +1.4% |

Test side: every split improves. test_single_in_dist −6.24%,
test_cruise −2.96%, test_re_rand −2.57%, test_rc −0.85%.

### Decision

**Merged.** Ninth merge of round 3, eighth proven stacked lever
(refinement of lever #4 EMA decay).

### Round-3 narrative — schedule × averaging interference RESOLVED

PR #476 (matched cosine × EMA(0.999)): destructive interference,
val +1.1% regression. **This PR (matched cosine × EMA(0.997))**:
clean compose, val −0.25% / test −1.97%.

The fix: shorter EMA window. The 0.999 vs 0.997 difference is ~3×
window length (1000 steps → 333 steps), enough to end averaging
before cosine tail weight-collapse. Schedule × averaging now
characterised as a **resolved** interaction, not a closed-out
compose-failure.

### Caveat — measurement on FF=8 advisor

Fern's branch was pre-#506 (FF=8). Post-merge advisor has FF=12 +
EMA=0.997. The actual joint config (FF=12 × EMA=0.997 × matched
cosine × lr=7.5e-4 × clip) is **untested** but expected to land
≤ 78.60 since FF=12 was a +1.57% lever in PR #506.

### Round-3 baseline lineage updated (9 merges)

| PR | val | test | lever |
|----|----:|-----:|-------|
| 280 | 102.64 | 97.73 | + L1 surface |
| 400 | 91.87 | 81.11 | + 8-freq spatial FF |
| 447 | 82.97 | 73.58 | + EMA(0.999) |
| 461 | 80.28 | 70.92 | + lr=7.5e-4 |
| 462 | 80.06 | 70.04 | + grad clipping |
| 506 | 78.80 | 69.13 | + FF=12 |
| **534** | **78.60** | **67.77** | + **EMA_DECAY=0.997** |

Cumulative −41.9% on val, −45.0% on test from PR #306 reference.

---

## 2026-04-28 05:14 — PR #516 (CLOSED, lr past EMA-stack optimum): lr=8e-4
- Branch: `charliepai2d3-askeladd/l1ff-ema-cos14-lr-8e-4` (deleted)
- Hypothesis: bracket LR upward from 7.5e-4 to 8e-4 on EMA-augmented
  stack. Predicted −0.5% to −2%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #462 (80.06) | vs current PR #534 (78.60) |
|--------|--------:|-------------------:|---------------------------:|
| `val_avg/mae_surf_p` | 80.32 | +0.32% | +2.19% |
| `test_avg/mae_surf_p` | 70.52 | +0.69% | +4.06% |

### Decision

**Closed.** Marginal regression. Per-split: val_single_in_dist +0.97%
(dominant negative); val_geom_camber_cruise +2.51%; val_geom_camber_rc
−0.99%; val_re_rand −0.54%.

### LR-curve triangulation (informative)

EMA-stack LR-vs-val data is monotone-increasing in [5e-4, 1e-3]:
| lr | val_avg | EMA? |
|---:|--------:|:----:|
| 5e-4 | 80.06 (PR #462 baseline) | no |
| 7.5e-4 | 80.28 (PR #461) | no |
| 8e-4 | 80.32 (this PR) | yes |
| 1e-3 | 82.08 (PR #489) | yes |

EMA-stack optimum is **at-or-below 7.5e-4** — opposite direction
from no-EMA stack where 7.5e-4 beat 5e-4. Round-5 implication: bracket
DOWN to 6e-4 / 7e-4 on the EMA stack.

Re-assigning askeladd to lr=7e-4 (interior LR bracket down).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 05:11 — PR #533 (CLOSED, bracket-closure): NUM_FOURIER_FREQS=4
- Branch: `charliepai2d3-frieren/l1ff4-ema-cos14-lr-7p5e-4` (deleted)
- Hypothesis: bracket FF dose downward to 4 freqs; tests interior
  optimum below 8. Predicted −0% to +2%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current PR #506 (78.80) | vs PR #462 (80.06) |
|--------|--------:|---------------------------:|-------------------:|
| `val_avg/mae_surf_p` | 81.31 | +3.18% | +1.56% |
| `test_avg/mae_surf_p` | 71.20 | +3.00% | +1.66% |

### Per-split val — under-capacity signature

| split | this PR | PR #462 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 94.97 | 93.59 | +1.47% |
| val_geom_camber_rc | 92.46 | 92.33 | **+0.14%** (flat) |
| val_geom_camber_cruise | 60.08 | 57.74 | **+4.06%** (most bitten) |
| val_re_rand | 77.72 | 76.57 | +1.51% |

Cruise hit hardest — confirms cruise is the most input-encoding-
sensitive split (richest spatial geometry, lowest absolute baseline).

### Decision

**Closed.** Above-zero regression vs current and prior baselines.
Bracket-closure successful: FF dose monotonically improves 4 → 8 → 12.

### FF dose response curve (3 of 4 points characterised)

| freqs | val_avg | source |
|------:|--------:|--------|
| 4 | 81.31 | this PR |
| 8 | 80.06 | PR #462 (no-EMA confound) |
| 12 | **78.80** | PR #506 (current best) |
| 16 | nezuko PR #543 (in flight) |

Once FF=16 lands (nezuko #543), the bracket is closed. If FF=16
also wins, optimum is past 12 and round-5 explores 24+. If FF=16
ties or regresses, optimum sits at 12.

### Re-assignment

frieren re-assigned to `slice_num=128` retest on the post-#506
advisor (their PR #292 was inconclusive on MSE baseline due to seed
noise; cleaner gradient signal on full lever stack should resolve).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 04:52 — PR #515 (CLOSED, EMA-overlap redundancy): 3× p-weight in vol_loss
- Branch: `charliepai2d3-tanjiro/l1ff-ema-cos14-lr-7p5e-4-vol-pwt-3x` (deleted)
- Hypothesis: 3× pressure-channel weight in volume MSE (different axis
  from L1-vol — channel weighting vs loss shape) captures heavy-tail
  benefit without overlapping EMA. Predicted −1% to −3%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current PR #506 (78.80) | vs PR #462 (80.06) |
|--------|--------:|---------------------------:|-------------------:|
| `val_avg/mae_surf_p` | 81.71 | **+3.69%** | +2.06% |
| `test_avg/mae_surf_p` | 71.92 | +4.04% | +2.69% |

### Per-split val — uniform mild regression

| split | this PR | PR #462 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 94.84 | 93.59 | +1.34% |
| val_geom_camber_rc | 94.06 | 92.33 | +1.87% |
| val_geom_camber_cruise | 59.72 | 57.74 | +3.43% |
| val_re_rand | 78.22 | 76.57 | +2.16% |

### Decision

**Closed.** Above-zero regression vs both current and PR #462
baselines. The lever is the **fourth example** of the EMA-overlap
compose-failure family (after #492 L1-vol, #500 wd=5e-4, #489 lr=1e-3).
EMA's trajectory averaging on the volume p-channel absorbs the
heavy-tail benefit that channel weighting would provide — adding an
axis-redundant lever just disrupts surf↔vol balance.

`train/vol_loss` ~2× the no-weight reference confirms the model got
the gradient mass; the metric didn't improve from it. Per-split
signal: `test_single_in_dist` only winning split (−2.30%), suggests
small in-dist effect that's drowned by OOD regression at this dose.

Re-assigning tanjiro to **auxiliary log-pressure loss** —
mechanistically different (target-space rescaling, not channel
weighting / loss shape / schedule manipulation).

Per-epoch metrics not centralised — branch deleted.

---

## 2026-04-28 04:41 — PR #506 (MERGED): NUM_FOURIER_FREQS 8 → 12
- Branch: `charliepai2d3-nezuko/l1ff12-ema-cos14-lr-7p5e-4`
- Hypothesis: bracket the proven spatial FF lever upward from 8 to 12
  octaves. Predicted −1% to −3%.
- Config: post-#462 advisor (L1+FF+EMA+clip baked in), single-line code
  change `NUM_FOURIER_FREQS = 8 → 12`, CLI `--epochs 14 --lr 7.5e-4`.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline (PR #462, 80.06) |
|--------|--------:|-------------------------------------:|
| `val_avg/mae_surf_p` | **78.80** | **−1.57%** (in predicted band) |
| `test_avg/mae_surf_p` | **69.13** | **−1.30%** |
| Param count | 674,647 | +4,096 (= 16 × 256 in `linear_pre`) |

### Per-split val — wins on all 4 splits

| split | this PR | PR #462 baseline | Δ |
|-------|--------:|-----------------:|--:|
| val_single_in_dist | 92.73 | 93.59 | −0.92% |
| val_geom_camber_rc | 89.86 | 92.33 | **−2.67%** (largest gain) |
| val_geom_camber_cruise | 57.32 | 57.74 | −0.73% |
| val_re_rand | 75.30 | 76.57 | −1.66% |

### Per-split test (best-val checkpoint)

| split | this PR | PR #462 | Δ |
|-------|--------:|--------:|--:|
| test_single_in_dist | 78.60 | 82.41 | **−4.62%** |
| test_geom_camber_rc | 80.12 | 79.66 | +0.58% (val/test divergence) |
| test_geom_camber_cruise | 49.29 | 49.50 | −0.42% |
| test_re_rand | 68.50 | 68.56 | −0.09% |

### Decision

**Merged.** Eighth merge of round 3, seventh proven stacked lever
(refines lever #2 spatial FF from 8 to 12 frequencies).

### Caveat — confounded with EMA addition

PR #462 baseline (80.06) was measured pre-EMA at default lr=5e-4.
This PR's measurement uses post-#447 advisor (with EMA) + lr=7.5e-4 +
matched cosine. The −1.57% headline conflates the FF=8→12 marginal
with the post-merge advisor improvements. Edward's PR #524 (canonical
6-lever stack at FF=8 with EMA, in flight) provides the reference
that disambiguates the FF marginal cleanly.

### Round-3 narrative — FF dose response

The FF lever now has multi-point dose-response:
- 8 freqs: PR #400 (val 91.87 vs L1 baseline 102.64 = −10.5%)
- 12 freqs: this PR (val 78.80 on full stack; clean win above 8)
- 4 freqs: frieren PR #533 in flight
- 16 freqs: nezuko's next assignment

Once #533 (FF=4) and edward #524 (FF=8 reference on full stack) land,
we'll have a clean dose-response curve. Round-5 may explore further
upward bracketing if FF=16 wins.

### Round-3 baseline lineage updated (now 8 merges)

| PR | val | test | lever |
|----|----:|-----:|-------|
| 280 | 102.64 | 97.73 | + L1 surface |
| 400 | 91.87 | 81.11 | + 8-freq spatial FF |
| 447 | 82.97 | 73.58 | + EMA |
| 461 | 80.28 | 70.92 | + lr=7.5e-4 |
| 462 | 80.06 | 70.04 | + grad clipping |
| **506** | **78.80** | **69.13** | + NUM_FOURIER_FREQS=12 |

(PR #389 matched cosine omitted from this table since matched cosine
is a CLI flag, not a code change. It's used for all measurements
since merging.)

Cumulative −41.7% on val, −43.9% on test from PR #306 reference.

---

## 2026-04-28 04:19 — PR #476 (CLOSED, schedule × EMA interference): matched cosine + EMA
- Branch: `charliepai2d3-fern/l1ff-ema-cos14` (deleted on close)
- Hypothesis: stack matched cosine `--epochs 14` onto L1+FF+EMA(0.999).
  Predicted −1% to −3%.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs PR #447 baseline (82.97) | vs current (PR #462, 80.06) |
|--------|--------:|----------------------------:|----------------------------:|
| `val_avg/mae_surf_p` | 83.89 | +1.11% | +4.78% |
| `test_avg/mae_surf_p` | 74.53 | +1.29% | +6.41% |

### Per-split val — schedule × averaging interference signature

| split | this PR | PR #447 baseline | Δ |
|-------|--------:|-----------------:|--:|
| val_single_in_dist | 98.16 | 99.44 | **−1.29%** (in-dist liked the averaging) |
| val_geom_camber_rc | 94.04 | 93.14 | +0.97% |
| val_geom_camber_cruise | 62.26 | 61.06 | +1.97% |
| val_re_rand | 81.08 | 78.22 | **+3.66%** (regression on cross-regime axis) |

### Critical structural finding

**Matched cosine × EMA(0.999) actively interfere when combined**.
Mechanism (student's derivation):

- EMA(0.999) averages over ~1000 steps ≈ 2.7 epochs ≈ last 19% of
  training under either schedule.
- Under T_max=50 (PR #447 baseline), the last 19% of training runs at
  `lr ≈ cos(0.56π) × 5e-4 + 0.5 × 5e-4` (plateau of decay) — moderately-
  decayed weights with residual exploration.
- Under T_max=14 (this PR), the last 19% covers the **cosine tail**
  where `lr → 0`. Weights have stopped exploring; they coalesce onto
  a single basin.
- EMA shadow over-specializes to in-dist (helps `val_single_in_dist`
  by −1.29%) while removing OOD regularisation noise (hurts
  `val_re_rand` +3.66%).

This **contradicts the round-3 baseline lineage assumption** that
PR #389 (matched cosine) and PR #447 (EMA) compose additively. Edward
PR #524 (in flight) measures the canonical 6-lever stack and will
clarify how much of the predicted ~76-78 stack number is dragged down
by this interference.

### Round-3 compose pattern map updated

| failure mode | example PRs |
|--|--|
| Same regularisation axis × FF | #437 wd, #446 beta2 |
| Same noise-smoothing axis × EMA | #492 L1-vol, #489 lr=1e-3 |
| Direction-only-update regime cliff | #499 clip=0.5 |
| **Schedule × averaging interference** | **#476 (this PR)** |

### Decision

**Closed.** Re-assigning fern to **EMA_DECAY=0.997** (window ~333
steps ~ 0.9 epochs) — shorter than 0.999, ending the EMA averaging
*before* the cosine tail's weight collapse. Tests whether tightening
the EMA window restores the matched-cosine compose benefit.

(Note: student's writeup recommended 0.99905 as "less aggressive",
but 0.99905 is actually MORE aggressive than 0.999 — longer effective
window. The right direction per their diagnostic is **shorter window**
= lower decay, hence 0.997.)

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 04:19 — PR #500 (CLOSED, no marginal value on full stack): wd=5e-4 + full stack
- Branch: `charliepai2d3-frieren/l1ff-ema-cos14-lr-7p5e-4-wd-5e-4` (deleted on close)
- Hypothesis: wd=5e-4 (validated as sweet spot in PR #469) composes
  on the full 6-lever stack. Predicted −1% to −3%.

### Headline

| Metric | This PR | vs current (PR #462, 80.06) | vs PR #469 (wd=5e-4 standalone, 81.07) |
|--------|--------:|----------------------------:|---------------------------------------:|
| `val_avg/mae_surf_p` | 81.20 | +1.42% (regression) | +0.16% (≈ tied) |
| `test_avg/mae_surf_p` | 71.37 | +1.90% | — |

### Per-split val (mild uniform regression)

| split | this PR | PR #462 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 93.62 | 93.59 | +0.03% |
| val_geom_camber_rc | 94.49 | 92.33 | +2.34% |
| val_geom_camber_cruise | 59.45 | 57.74 | +2.96% |
| val_re_rand | 77.25 | 76.57 | +0.89% |

### Decision

**Closed.** wd=5e-4 contributes **~0 marginal value** once EMA + clipping
are present. The standalone wd=5e-4 win (PR #469, val 81.07) does not
survive the full stack — wd's regularisation overlaps with EMA's
trajectory averaging and clipping's stability work.

`val_geom_camber_rc` regressed +2.34% (vs PR #469's −7.2% on the same
split). The wd × EMA + clip interaction in the rc-camber regime is
the most concrete signal of redundancy.

Re-assigning frieren to NUM_FOURIER_FREQS=4 (FF dose downward bracket,
complementing nezuko's #506 at 12 freqs).

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 04:19 — PR #501 (CLOSED, under-convergence cliff): DropPath 0.1
- Branch: `charliepai2d3-thorfinn/l1ff-ema-cos14-lr-7p5e-4-droppath-0p1` (deleted on close)
- Hypothesis: DropPath 0.1 — mechanistically-different regulariser
  than wd/beta2; tests whether residual-branch stochasticity bypasses
  the FF interference pattern. Predicted −1% to −4%.

### Headline (best-val checkpoint, epoch **13/14** — confounded)

| Metric | This PR | vs current (PR #462, 80.06) |
|--------|--------:|----------------------------:|
| `val_avg/mae_surf_p` | 89.54 | +11.8% |
| `test_avg/mae_surf_p` | 80.56 | +15.0% |

### Confound

DropPath added **+7% per-epoch overhead** (sampling + masked
multiply on residual outputs), pushing 14-epoch total to ~32.7 min,
**over the 30-min cap**. Only 13/14 epochs ran. Trajectory was still
descending steeply (3-5% cuts in last few epochs) — model is
**under-trained relative to its schedule**, not regularised
differently.

### Per-split — uniform regression

| split | Δ vs baseline |
|-------|---:|
| val_single_in_dist | +11.1% |
| val_geom_camber_rc | +9.9% (least-regressed split, *opposite* of FF-interference pattern) |
| val_geom_camber_cruise | +15.0% |
| val_re_rand | +12.6% |

### Decision

**Closed.** Confounded result — under-convergence dominates. The
"DropPath bypasses magnitude-regulariser-FF interference" question
remains unresolved.

Re-assigning thorfinn to **DropPath 0.05** (their suggestion #1).
Half the rate, less per-epoch overhead, 14 epochs should fit cleanly.
Cleanly tests rate-vs-mechanism question.

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 04:14 — PR #499 (CLOSED, under-convergence cliff): max_norm=0.5
- Branch: `charliepai2d3-edward/l1ff-ema-cos14-lr-7p5e-4-clip-0p5` (deleted on close)
- Hypothesis: tighten gradient clipping `max_norm` from 1.0 to 0.5 on the
  full lever stack (L1+FF+EMA + matched cosine + lr=7.5e-4). Predicted
  −1% to −3%.
- Config: post-#462 advisor, single-line code change `1.0 → 0.5`,
  CLI `--epochs 14 --lr 7.5e-4`.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline (PR #462, 80.06) |
|--------|--------:|-------------------------------------:|
| `val_avg/mae_surf_p`  | 80.64 | **+0.72%** (marginal regression, within noise) |
| `test_avg/mae_surf_p` | 70.62 | **+0.83%** |

### Under-convergence cliff diagnostic

Pre-clip grad-norm trajectory shows clip is firing on every batch:

| epoch | val_avg | gn_mean | gn_max |
|------:|--------:|--------:|-------:|
| 1 | 330.53 | 46.20 | 139.30 |
| 5 | 179.58 | 38.11 |  78.85 |
| 14 |  80.64 | 22.85 |  54.07 |

Pre-clip mean is **22-46 throughout training**, never below 0.5.
`max_norm=0.5` is a **direction-only update regime** — every batch's
gradient is rescaled to magnitude 0.5. Qualitatively different from
`max_norm=1.0` (outlier-clip with bulk signal through; ~27× headroom).

### Cliff signature

| epoch | this PR (clip=0.5) | PR #462 baseline (clip=1.0) | gap |
|------:|-------------------:|----------------------------:|----:|
| 1 | 330.53 | 227.23 | +45% slower |
| 5 | 179.58 | (~140) | +28% slower |
| 14 | 80.64 | 80.06 | +0.7% |

Cosine tail does most of the recovery work, but ep1-6 are starved of
gradient magnitude. Net: training runs to completion, doesn't beat
the 1.0 baseline.

### Decision

**Closed** with mechanistically-explained marginal regression.

**Round-3 narrative addition**: tight clipping at 0.5 hits the same
**aggressive-regulariser-overlap** family as wd=1e-3 (#437), beta2=0.95
(#446), L1-volume + EMA (#492), and lr=1e-3 + EMA (#489). Once the
stack already has FF+EMA+matched-cosine providing aggressive
regularisation, additional aggressive regularisation **interferes
with early-epoch convergence** rather than adding signal.

The compose-failure family is now well-characterised:
| failure mode | example PRs |
|--|--|
| Same regularisation axis × FF | #437 wd=1e-3, #446 beta2=0.95 |
| Same noise-smoothing axis × EMA | #492 L1-vol, #489 lr=1e-3 |
| Direction-only-update regime cliff | #499 (this PR, clip=0.5) |

### Re-assignment

The **canonical round-3-best 6-lever stack measurement** has never
been done directly — every merge has been on a slightly different
baseline (PR #389 no FF, #447 no matched cosine, #461 no EMA, #462
no EMA + default lr, #499 clip=0.5 not 1.0). Edward re-assigned to
the canonical clean run.

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 04:00 — PR #492 (CLOSED, L1-vol × EMA destructively overlap): L1 volume on full lever stack
- Branch: `charliepai2d3-tanjiro/l1ff-ema-cos14-lr-7p5e-4-voll1` (deleted on close)
- Hypothesis: stack the validated L1-volume lever (PR #448, −5.18% on
  L1+FF) onto the full proven-lever stack (L1+FF+EMA + matched cosine
  + lr=7.5e-4). Predicted ~76 if additive.
- Config: post-#447 advisor (had FF+EMA), changed vol_loss to L1
  (your PR #448 change), `--epochs 14 --lr 7.5e-4`.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline (PR #462, 80.06) | vs PR #461 (your assigned-against, 80.28) |
|--------|--------:|-------------------------------------:|------------------------------------------:|
| `val_avg/mae_surf_p`  | 84.28 | **+5.27%** | +4.99% |
| `test_avg/mae_surf_p` | 74.07 | **+5.76%** | +4.44% |

### Per-split val — same lever, opposite direction on val_single_in_dist

| split | L1+FF + L1-vol (PR #448) | this PR (L1-vol on EMA stack) | Δ vs current baseline |
|-------|-------------------------:|------------------------------:|----------------------:|
| val_single_in_dist | **−9.44%** (gain) | **+11.56%** (regression) | +11.56% |
| val_geom_camber_rc | −4.18% | +4.49% | +4.49% |
| val_geom_camber_cruise | −6.95% | −1.39% | mild improvement |
| val_re_rand | +1.13% | +3.07% | +3.07% |

The split where L1-volume helped most on the pre-EMA baseline is now
the one it hurts most on the post-EMA baseline. **Strong overlap signal**.

### Decision

**Closed.** Above-threshold regression (+5.27% val).

### Mechanistic read

EMA's trajectory averaging already smooths the per-batch volume noise
that L1-volume's loss-shape change targets. Switching MSE→L1 on volume
also rebalances effective surface↔volume gradient scale (L1
down-weights large per-cell volume errors relative to MSE), and on the
EMA-smoothed trajectory this shifts the optimum toward a different
basin worse on heavy-tail.

### Round-3 narrative — third compose-failure with overlap signature

| PR | lever | overlap with | mechanism |
|----|-------|-------------|-----------|
| #437 | wd=1e-3 + FF | rc-camber regularisation | magnitude-based regulariser overlap |
| #446 | beta2=0.95 + FF | rc-camber regularisation | optimiser-side regulariser overlap |
| #492 (this PR) | L1-vol + EMA | gradient-noise smoothing | sample-noise regulariser overlap |

**Generalisation**: once one "noise/regularisation" lever is in the
stack (FF, EMA), additional same-mechanism levers tend to interfere
on the most-improved split.

Re-assigning tanjiro to per-channel pressure weight in vol_loss
(3× on p) — different axis than loss shape, doesn't overlap with EMA.

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 03:59 — PR #489 (CLOSED, lr=1e-3 past optimum): L1+FF+EMA + matched cosine + lr=1e-3
- Branch: `charliepai2d3-askeladd/l1ff-ema-cos14-lr-1e-3` (deleted on close)
- Hypothesis: bracket peak LR upper end (7.5e-4 → 1e-3) on the
  EMA-augmented stack. Predicted −2% to −5%.
- Config: post-#447 advisor (had FF+EMA), `--epochs 14 --lr 1e-3`.

### Headline

| Metric | This PR | vs current (PR #462, 80.06) | vs PR #461 (80.28) |
|--------|--------:|----------------------------:|-------------------:|
| `val_avg/mae_surf_p` | 82.08 | +2.52% | +2.24% |
| `test_avg/mae_surf_p` | 72.17 | +3.04% | +1.76% |

### Per-split val — interior optimum below 1e-3

| split | this PR | PR #461 | Δ |
|-------|--------:|--------:|--:|
| val_single_in_dist | 95.88 | 89.76 | **+6.82%** |
| val_geom_camber_rc | 91.91 | 90.03 | +2.09% |
| val_geom_camber_cruise | 61.62 | 62.42 | **−1.28%** |
| val_re_rand | 78.89 | 78.92 | flat |

Train losses smooth (`0.757 → 0.186` monotone), no NaN, no
early-epoch bouncing. **Bottleneck is LR overshoot at peak, not
warmup**.

### Decision

**Closed.** Above-zero regression on val and test, concentrated on
the in-dist split. Student's analysis cleanly identifies interior
LR optimum between 7.5e-4 and 1e-3.

### Round-3 narrative addition

LR sensitivity is **stack-dependent**:
- On L1+FF + matched cosine (no EMA): lr=7.5e-4 was a clean win
  (PR #461, val −3.2%).
- On L1+FF+EMA + matched cosine: lr=7.5e-4 likely still wins (TBD via
  fern PR #476's lr=5e-4 reference); lr=1e-3 is past optimum.
- EMA's late-training trajectory averaging benefits from a slightly
  more conservative LR; pushing peak too high moves the late-training
  mean further from the in-dist minimum.

Re-assigning askeladd to lr=8e-4 (interior bracket point).

Per-epoch metrics not centralised — branch deleted on close.

---

## 2026-04-28 03:35 — PR #432 (CLOSED, refuted hypothesis): L1+FF + log(Re) Fourier features
- Branch: `charliepai2d3-nezuko/l1-ff-pos-logre-8freq` (deleted on close)
- Hypothesis: extend the proven 8-freq spatial FF lever to the scalar
  `log(Re)` input. Predicted −2% to −8% on val.
- Config: post-#400 advisor (L1+FF), added 16 log(Re) FF channels
  (`fun_dim=70`, +4,096 weights in input MLP).

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1+FF baseline (91.87) | vs current baseline (PR #462, 80.06) |
|--------|--------:|--------------------------:|-------------------------------------:|
| `val_avg/mae_surf_p`  | 91.79 | −0.09% (≈ tied within noise) | +14.7% |
| `test_avg/mae_surf_p` | 83.30 | **+2.70% (regressed)** | +18.9% |

### Per-split val — refutes the predicted direction

| split | L1+FF baseline | this PR | Δ | predicted? |
|-------|---------------:|--------:|--:|:-----------|
| val_re_rand | 82.64 | 90.44 | **+9.4% (regressed)** | predicted to *improve* disproportionately (cross-regime axis); opposite direction observed |
| val_geom_camber_cruise | 68.61 | 75.12 | +9.5% | unrelated to Re axis, regressed |
| val_geom_camber_rc | 98.99 | 98.84 | flat |
| val_single_in_dist | 117.24 | 102.75 | **−12.4%** | in-dist memorisation, not Re-extrapolation |

### Decision

**Closed** per the >5% rule and the lever's mechanistic premise being
refuted at split level.

### Mechanistic read (round-3 narrative addition)

Student's analysis: the spectral-bias argument that justified spatial
FF (PR #400 winning −10.5%) is **much weaker for log(Re)**:
- Spatial `(x, z)` enters via only 2 input channels + slice-attention
  geometric routing. The model has no other handle on position →
  removing spectral bias matters substantially.
- Log(Re) is already one of 22 input features going through a
  non-linear MLP + 5 attention layers; it's broadcastable to a
  learned non-linear encoding by every layer. Adding 16 high-frequency
  variants is **redundant capacity**, not new signal.

**Round-3 finding for input encoding compose tests**: input-encoding
levers compose with FF only when the targeted input dimension was
previously *poorly exposed* to the model. Spatial FF wins because
position is 2-d and only used for slice routing; log(Re) FF fails
because Re is already rich. Round-5 input-encoding work targeting
gap/stagger/AoA dimensions (1-d each, going through MLP) should expect
similar negative results.

The `val_re_rand` regression (+9.4%) suggests the smooth log(Re)
representation may have been doing implicit cross-regime regularisation;
breaking it with high-frequency components removes that effect.

Re-assigning nezuko to spatial FF frequency-count bracket
(`NUM_FOURIER_FREQS=12` on post-#462 advisor) — their own PR #400
follow-up #1, the input-encoding lever that *did* work.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 03:31 — PR #462 (MERGED): L1+FF + matched cosine + grad clipping (max_norm=1.0)
- Branch: `charliepai2d3-edward/l1ff-cos14-clip1`
- Hypothesis: gradient clipping at `max_norm=1.0` composed with FF
  and matched cosine; predicted −1% to −3%.
- Config: post-#400 advisor (L1+FF) but pre-#447 (no EMA),
  `--epochs 14`. Single-line code change adding `clip_grad_norm_`
  before `optimizer.step()`; added grad-norm logging.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline (PR #461, 80.28) | vs PR #389 (the assigned-against, 90.90) |
|--------|--------:|-------------------------------------:|----------------------------------------:|
| `val_avg/mae_surf_p`  | **80.06** | **−0.27%** (marginal win, within noise) | **−11.9%** ✓ above predicted |
| `test_avg/mae_surf_p` | **70.04** | **−1.24%** (above noise) | **−13.4%** ✓ |

### Per-split val (best epoch 14)

| split | L1+FF baseline | this PR | Δ |
|-------|---------------:|--------:|--:|
| val_single_in_dist     | 117.24 | 93.59 | **−20.2%** |
| val_geom_camber_rc     |  98.99 | 92.33 | −6.7% (additive band, predicted 5-10%) |
| val_geom_camber_cruise |  68.61 | 57.74 | **−15.9%** |
| val_re_rand            |  82.64 | 76.57 | −7.3% |

### Pre-clip grad-norm trajectory (most useful round-3 instrumentation)

| epoch | mean | max | val_avg |
|------:|-----:|----:|--------:|
| 1  | 56.99 | 142.43 | 227.23 |
| 7  | 45.44 |  99.31 | 105.76 |
| 11 | 38.20 | 164.85 |  91.06 |
| 14 | **27.20** | 75.79 | 80.06 |

Pre-clip grad-norm mean drops ~52% over training (57 → 27) but
**stays ~27× above max_norm=1.0 even at the cosine tail**. Clipping
fires on essentially every batch through epoch 14. Tighter values
(0.5, 0.1) are well-motivated for round 5.

### Decision

**Merged.** Marginal val win (within seed noise) but **unambiguous
test win** and uniform per-split improvements.

Most interesting cross-PR finding: clipping on L1-only (PR #423,
closed) had `val_single_in_dist` flat at −0.6%; this run shows it at
**−11.6%**. **Same lever, qualitatively different effect when
composed with matched cosine.** Stability levers compose better with
schedule levers — late-cosine LR decay creates the conditions where
clipping's outlier-suppression matters more.

### Round-3 narrative — clipping is the third "non-overlapping" lever

| compose pattern | with FF | examples |
|----------------|---------|----------|
| Distributional / trajectory-averaging | additive | matched cosine + lr=7.5e-4, EMA, **clipping (this PR)** |
| L1-only-OOD-camber-targeted at high dose | destructive on rc-camber | wd=1e-3, beta2=0.95 |

### Caveat

Branch was pre-#447 (no EMA). Post-merge advisor adds EMA. The
six-lever stack (L1+FF+EMA + matched cosine + lr=7.5e-4 + clip) on
post-merge advisor is untested but expected ≤ 80.06.

### Round-3 proven levers (cumulative, now six stacked)

1. L1 surface loss (PR #280)
2. 8-freq spatial Fourier features (PR #400)
3. Matched cosine `--epochs 14` (PR #389, CLI)
4. EMA-of-weights decay=0.999 (PR #447)
5. Peak LR `lr=7.5e-4` (PR #461, CLI)
6. **Gradient clipping max_norm=1.0** (PR #462) ← this merge

---

## 2026-04-28 03:31 — PR #469 (CLOSED, ties current; validates wd sweet spot): L1+FF + matched cosine + wd=5e-4
- Branch: `charliepai2d3-frieren/l1ff-cos14-wd-5e-4` (deleted on close)
- Hypothesis: interior-point wd test — does wd=5e-4 capture the
  cruise/in-dist compose benefits of wd=1e-3 (PR #437) without the
  rc-camber regression? Predicted −1% to −3% on val.
- Config: post-#400 advisor (L1+FF), `--epochs 14 --weight_decay 5e-4`.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline (PR #461, 80.28) | vs PR #389 (90.90) |
|--------|--------:|-------------------------------------:|-------------------:|
| `val_avg/mae_surf_p`  | 81.07 | +1.0% (≈ tied) | **−10.83%** ✓ |
| `test_avg/mae_surf_p` | 71.75 | +1.2% (≈ tied) | **−11.25%** ✓ |

### The wd ladder (key compose finding)

| split | L1+FF (wd=1e-4) | L1+FF + wd=1e-3 (PR #437) | **L1+FF + cos14 + wd=5e-4 (this PR)** |
|-------|----------------:|--------------------------:|---------------------------------------:|
| val_geom_camber_rc | 98.99 | **110.64 (+11.8% regressed)** | **91.86 (−7.2%, no regression)** |
| val_geom_camber_cruise | 68.61 | 60.70 (−11.5%) | **60.73 (−11.5%, same gain)** |
| val_single_in_dist | 117.24 | 108.46 (−7.5%) | **94.21 (−19.6%, round-3 best on this split)** |
| val_re_rand | 82.64 | 85.61 (+3.6%) | **77.46 (−6.3%, flipped to win)** |

### Decision

**Closed** — ties current baseline within seed noise. But **the wd
sweet spot is now firmly established**: rc-camber regression cliff at
wd=1e-3 does not extend to wd=5e-4, full cruise gain held, in-dist
hits round-3 best.

The "validated-on-L1 OOD-camber lever doesn't compose with FF"
pattern (PR #437, PR #446) is **dose-dependent for wd**: small wd
composes additively, large wd interferes. Important nuance for
round-5 stacking.

Re-assigning frieren to test wd=5e-4 on the post-#462 advisor (which
adds clipping + EMA via merges since #469 was assigned).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 03:31 — PR #446 (CLOSED, second compose-failure confirmation): L1+FF + AdamW(beta2=0.95)
- Branch: `charliepai2d3-thorfinn/l1ff-adamw-beta2-0-95` (deleted on close)
- Hypothesis: stack the validated `beta2=0.95` lever (PR #419) onto
  L1+FF; predicted −1% to −4%.
- Config: post-#400 advisor (L1+FF), `betas=(0.9, 0.95)` in AdamW.

### Headline — clean regression, two-seed confirmed

| Metric | This PR | vs L1+FF (91.87) | vs L1-only `beta2=0.95` (PR #419) |
|--------|--------:|-----------------:|----------------------------------:|
| `val_avg/mae_surf_p` (best epoch 13/14) | 96.37 | **+4.9% (regressed)** | **+5.5%** (vs 91.50) |
| `test_avg/mae_surf_p` | 85.15 | +5.0% | +6.6% |

Two seeds: val 99.92 / 96.37, both regress vs L1+FF baseline. Sign
unambiguous.

### Per-split val — the diagnostic check

| split | L1+FF baseline | L1+FF + beta2=0.95 (this PR) | L1-only Δ (PR #419) |
|-------|---------------:|-----------------------------:|--------------------:|
| val_geom_camber_rc | 98.99 | **109.52 (+10.6% regressed)** | **−13.6%** |
| val_geom_camber_cruise | 68.61 | 73.27 (+6.8%) | +6.3% |
| val_re_rand | 82.64 | 89.01 (+7.7%) | −1.6% |
| val_single_in_dist | 117.24 | 113.67 (−3.0%) | +1.8% |

The OOD-camber gain that motivated the lever (rc-camber **−13.6%**
on L1-only) **inverts to +10.6% regression** when stacked on FF.
Cruise's negative L1-only signal persists.

### Structural finding — second confirmation

| PR | lever | L1-only `val_geom_camber_rc` Δ | L1+FF compose `val_geom_camber_rc` Δ |
|----|-------|-------------------------------:|-------------------------------------:|
| #437 | wd=1e-3 | **−11.9%** | **+11.8%** (regressed) |
| #446 (this PR) | beta2=0.95 | **−13.6%** | **+10.6%** (regressed) |

**Two independent levers, both validated targeting OOD-camber on L1,
both fail to compose with FF, both regress on rc-camber by ~10-12%.**
That's a coherent pattern: the L1-only OOD-camber improvement is *the*
empirical signature of an FF-redundant lever.

Mechanistic read: lower beta2 shortens the second-moment window →
more effective per-step variance → FF inputs add high-frequency
components → combining over-amplifies noisy gradient steps on
geometry-sensitive features.

### Decision

**Closed** with two-seed confirmation of the regression sign.

Re-assigning thorfinn to **DropPath / stochastic depth** — different
regularisation mechanism than weight magnitude (wd) or second-moment
variance (beta2). Should bypass the "OOD-camber-targeted regulariser
doesn't compose with FF" pattern.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 03:15 — PR #448 (CLOSED, validated on L1+FF / loses to current): L1 volume loss
- Branch: `charliepai2d3-tanjiro/l1ff-vol-l1` (deleted on close)
- Hypothesis: replace MSE volume loss with L1 volume loss — does L1
  dominance extend symmetrically to the volume term? Pre-registered
  three decision branches.
- Config: post-#400 advisor (L1+FF, MSE volume), changed vol_loss
  from sq_err to abs_err in train.py. Two-line code diff.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1+FF baseline (PR #400, 91.87) | vs current baseline (PR #461, 80.28) |
|--------|--------:|-----------------------------------:|-------------------------------------:|
| `val_avg/mae_surf_p`  | 87.11 | **−5.18%** ✓ above predicted band | +8.5% (loses) |
| `test_avg/mae_surf_p` | 78.90 | **−2.73%** ✓ in band | +11.2% (loses) |

### Per-split val (best epoch 14) — uniform improvement on 3 of 4 splits

| split | L1+FF baseline | this PR | Δ |
|-------|---------------:|--------:|--:|
| val_single_in_dist     | 117.24 | 106.17 | **−9.44%** (largest gain — exactly the heavy-tail-dominated split) |
| val_geom_camber_rc     |  98.99 |  94.86 | −4.18% |
| val_geom_camber_cruise |  68.61 |  63.84 | −6.95% (refutes "MSE smoothing useful on cruise") |
| val_re_rand            |  82.64 |  83.57 | +1.13% (within seed noise) |

### Decision

**Closed** per the >5% regression rule vs current baseline. **But the
lever is genuinely validated** on the assigned baseline — largest
single-knob lever validation since PR #280 (L1 surface loss).

The student's pre-registered hypothesis branches:
- (1) L1 dominance extends symmetrically → uniform improvement ✓ FIRED
- (2) MSE-volume was doing useful smoothing → cruise regresses ✗ REFUTED
- (3) L1 helps mainly on heavy-tail samples → in-dist disproportionate ✓ FIRED

Reproducibility: second seed at val 86.45 / test 78.27 (slightly more
favourable than the canonical 87.11). The win is robust.

**Cumulative L1 story**: L1-everywhere is strictly better than mixed
L1-surface/MSE-volume. PR #280's finding that "L1's noise robustness
is the dominant effect" generalises symmetrically to volume.

### Re-assignment

Tanjiro re-assigned to test L1 volume as a compose test on the new
post-#461 advisor (L1+FF+EMA + matched cosine + lr=7.5e-4). If L1-
volume composes additively with the other four proven levers, the
result lands around 76 — a meaningful round-3 close.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 03:11 — PR #461 (MERGED): L1+FF + matched cosine + lr=7.5e-4
- Branch: `charliepai2d3-askeladd/l1ff-cos14-lr-7p5e-4`
- Hypothesis: bump peak LR from `5e-4` to `7.5e-4` on the L1+FF +
  matched cosine baseline. Now that the cosine actually anneals,
  `5e-4` should be conservatively low. Predicted −1% to −5%.
- Config: post-#400 advisor (L1+FF), pre-#447 advisor (no EMA),
  `--epochs 14 --lr 7.5e-4`. CLI-only diff.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs current baseline at merge (PR #447, 82.97) | vs PR #389 (the assigned-against config, 90.90) |
|--------|--------:|----------------------------------------------:|-----------------------------------------------:|
| `val_avg/mae_surf_p`  | **80.28** | **−3.2%** | **−11.7%** ✓ above predicted band |
| `test_avg/mae_surf_p` | **70.92** | **−3.6%** | **−12.3%** |
| Per-epoch wallclock | ~131 s | flat | flat |
| Peak GPU memory | 42.38 GB | flat | flat |

### Per-split val — distributional gain (broad across all splits)

| split | PR #389 baseline | this PR | Δ |
|-------|-----------------:|--------:|--:|
| val_single_in_dist     | 105.82 | **89.76** | **−15.18%** |
| val_geom_camber_rc     | 100.82 |  90.03 | −10.70% |
| val_geom_camber_cruise |  71.37 |  62.42 | −12.54% |
| val_re_rand            |   85.60 |  78.92 |  −7.80% |

### Per-split test — broad gain across all 4 splits

| split | PR #389 baseline | this PR | Δ |
|-------|-----------------:|--------:|--:|
| test_single_in_dist     | 94.78 | 78.18 | −17.51% |
| test_geom_camber_rc     | 88.30 | 80.52 |  −8.81% |
| test_geom_camber_cruise | 59.67 | 53.88 |  −9.70% |
| test_re_rand            |  80.62 | 71.12 | −11.78% |

### Validation curve

```
ep  1: 209.87 (best)
ep  2: 173.97 (best)  ep  8: 101.94 (best)
ep  3: 164.04 (best)  ep  9:  97.54 (best)
ep  4: 144.19 (best)  ep 10:  94.40 (best)
ep  5: 120.02 (best)  ep 11:  87.93 (best)
ep  6: 147.32         ep 12:  87.75 (best)
ep  7: 133.41         ep 13:  80.86 (best)
                       ep 14:  80.28 (best) ← final
```

Smooth descent through ep1-3 — no NaN, no early instability — confirms
the "cosine self-warmup from peak" pattern works under matched cosine
without explicit warmup. Train losses decay monotonically from
`surf=0.71/vol=1.42` at ep1 to `surf=0.187/vol=0.241` at ep14.

### Decision

**Merged.** Three findings ride together in this number:

1. **L1+FF + matched cosine compose** is substantively additive (was
   estimated to land below 90.90; landed at 80.28 — much better).
2. **lr=7.5e-4 doesn't destabilise** on matched cosine + no warmup —
   the previous lr=1e-3 failure (PR #288) was warmup-driven, not
   LR-driven.
3. **Per-split gain is distributional, not concentrated**. Unlike
   most round-3 levers (which all hit `val_geom_camber_rc` hardest),
   this run improves *every* split with `val_single_in_dist` (−15.2%)
   leading slightly. Consistent with "removed an LR bottleneck" —
   distributional rather than mechanism-specific.

### Caveat — measurement on pre-#447 advisor

PR #461's branch was based on post-#389 advisor (had FF + matched
cosine via #389) but **before PR #447 merged** (no EMA). So the
measurement is L1+FF + matched cosine + lr=7.5e-4, *no EMA*. The
post-merge advisor includes EMA from #447. Running the post-merge
advisor with `--epochs 14 --lr 7.5e-4` will measure the **L1+FF+EMA
+ matched cosine + lr=7.5e-4 five-lever stack** — should beat 80.28
since EMA was a clean +9% lever.

### Round-3 proven levers (cumulative, now five stacked)

1. L1 surface loss (PR #280)
2. 8-freq spatial Fourier features (PR #400)
3. Matched cosine `--epochs 14` (PR #389, CLI flag)
4. EMA-of-weights, decay=0.999 (PR #447)
5. **Peak LR `lr=7.5e-4`** (PR #461, CLI flag)

Levers 1, 2, 4 baked into `train.py`. Levers 3, 5 are CLI flags.
Recommended reproduce: `python train.py --epochs 14 --lr 7.5e-4`.

### Round-3 narrative (further refined)

We now have three different compose patterns documented:

| compose | pattern | example PR |
|---------|---------|-----------|
| overlap | destructive on shared axis | wd × FF (PR #437, rc-camber) |
| additive | clean orthogonal mechanisms | EMA × FF (PR #447) |
| distributional | broad across all splits | lr=7.5e-4 × matched cosine (PR #461) |

The regularisation/optimisation/encoding landscape is
multi-dimensional. Per-split analysis is the load-bearing diagnostic
for round-5 stacking decisions.

---

## 2026-04-28 02:50 — PR #447 (MERGED): L1+FF + EMA(decay=0.999) — biggest single-lever win since PR #280
- Branch: `charliepai2d3-fern/l1ff-ema-d999`
- Hypothesis: stack EMA-of-weights with budget-aware decay 0.999 onto
  the L1+FF baseline (PR #400). Orthogonal mechanism (weight averaging
  vs input encoding); predicted −1% to −4% on val.
- Config: L1+FF baseline (post-#400, pre-#389-merge), EMA every step
  with `EMA_DECAY=0.999` (derived from `1 − 1/(0.2 × total_steps)` ≈
  0.999 for ~5K steps), swap for val/test eval, save EMA weights to
  best-val checkpoint.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1+FF baseline (PR #400, 91.87) | vs current baseline at merge (PR #389, 90.90) |
|--------|--------:|-----------------------------------:|----------------------------------------------:|
| `val_avg/mae_surf_p`  | **82.97** | **−9.7%** ✓ above predicted band | **−8.7%** |
| `test_avg/mae_surf_p` | **73.58** | **−9.3%** ✓ above predicted band | **−9.0%** |
| Per-epoch wallclock | ~132 s | flat | flat |
| Peak GPU memory | 42.4 GB | flat | flat |
| Param count | 670,551 (EMA shadow ~2.6 MB extra) |

### Per-split val (best epoch 14) — wins on all 4 splits

| split | L1+FF baseline | this PR | Δ |
|-------|---------------:|--------:|--:|
| val_single_in_dist     | 117.24 |  99.44 | **−15.2%** (largest gain) |
| val_geom_camber_rc     |  98.99 |  93.14 | −5.9% |
| val_geom_camber_cruise |  68.61 |  61.06 | −11.0% |
| val_re_rand            |  82.64 |  78.22 | −5.4% |

### Per-split test (best-val checkpoint) — wins on all 4 splits

| split | L1+FF baseline | this PR | Δ |
|-------|---------------:|--------:|--:|
| test_single_in_dist     | 100.17 | 88.79 | −11.4% |
| test_geom_camber_rc     |  85.47 | 81.54 |  −4.6% |
| test_geom_camber_cruise |  61.17 | 52.36 | **−14.4%** |
| test_re_rand            |  77.64 | 71.62 |  −7.8% |

### Decision

**Merged.** Largest single-lever win since L1 surface loss (PR #280's
−24.1%). EMA's weight-averaging mechanism is **fully orthogonal** to
the FF input encoding lever — the EMA-on-L1 gain (−10.4%, PR #396)
and EMA-on-L1+FF gain (−9.7%, this PR) are within 1% of each other,
making this the cleanest "additive compose" signal of round 3.

### Bottleneck status update

`val_single_in_dist` was the persistent worst-performing split through
PR #280, #400, and #389. EMA is the **first round-3 lever to
substantially attack the high-Re raceCar single regime** — gained
−15.2% on val (117.24 → 99.44) and −11.4% on test. This is opposite
the per-split pattern of FF (which gained least on in-dist).

After this merge, the per-split val ranking is:
- val_single_in_dist: 99.44 (still worst, but closing)
- val_geom_camber_rc: 93.14
- val_re_rand: 78.22
- val_geom_camber_cruise: 61.06 (easiest, now firmly under 65)

### Caveat — measurement on pre-#389 advisor

PR #447's branch was based on the post-#400 advisor (had FF) but
**before PR #389 merged** (didn't have matched cosine). So the
measurement is L1+FF + EMA + cosine T_max=50 (default schedule, never
reaches the tail). The post-merge advisor has all four levers in
`train.py`/CLI; running with `--epochs 14` will give the **L1+FF+EMA +
matched cosine** four-lever stack — fern's next assignment tests
exactly that.

### Round-3 proven levers (cumulative, now four stacked)

1. L1 surface loss (PR #280)
2. 8-freq spatial Fourier features (PR #400)
3. Matched cosine `--epochs 14` (PR #389, CLI flag)
4. **EMA-of-weights, decay=0.999** (PR #447) ← this merge

### Round-3 narrative (refined)

The "five convergent OOD-camber levers" narrative was partially
refuted by PR #437 (wd × FF overlap on rc-camber). PR #447's data
adds new structure:

- **EMA × FF compose** is fully additive (this PR).
- **wd × FF compose** has destructive overlap on rc-camber (PR #437).
- **EMA × in-dist** is the strongest single-lever effect on the
  persistent in-dist bottleneck — fundamentally different from
  weight-magnitude regularisation.

Implication: EMA's mechanism (averaging across late-training
trajectory variance) is a different kind of "regularisation" than
weight-magnitude penalty. Round-5 should treat the regularisation
landscape as multi-dimensional, not scalar.

---

## 2026-04-28 02:35 — PR #437 (CLOSED, ties current; reveals compose dynamics): L1+FF + wd=1e-3
- Branch: `charliepai2d3-frieren/l1ff-wd-1e-3` (deleted on close)
- Hypothesis: stack the validated `wd=1e-3` lever (PR #395) onto the
  L1+FF baseline (PR #400). Predicted −1% to −4%.
- Config: L1+FF baseline (post-#400), `--weight_decay 1e-3`. CLI-only.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1+FF baseline (PR #400, 91.87) | vs current baseline (PR #389, 90.90) |
|--------|--------:|-----------------------------------:|-------------------------------------:|
| `val_avg/mae_surf_p`  | 91.35 | −0.6% | **+0.5%** (≈ tied) |
| `test_avg/mae_surf_p` | 81.35 | +0.3% | +0.6% |

### Per-split val (best epoch 14) — three different stacking patterns

| split | L1 + wd (PR #395) | L1+FF (PR #400) | L1+FF+wd (this PR) | what stacks? |
|-------|------------------:|----------------:|-------------------:|--------------|
| val_geom_camber_rc | −11.9% | −20.8% | **+11.8% (worse)** | **destructive** |
| val_geom_camber_cruise | +2.4% | −6.3% | **−11.5%** | additive |
| val_single_in_dist | +6.2% | −3.3% | **−7.5%** (sign-flipped) | additive |
| val_re_rand | −1.0% | −9.3% | +3.6% | flat |

### Decision — close per criterion, but reframe round-3 understanding

**Closed.** Headline ties the current baseline; below merge threshold.
But the per-split signal is **the most informative of round 3** —
contradicts the "five convergent OOD-camber levers all stack
additively" narrative.

### Round-3 narrative shift

The five levers that improved `val_geom_camber_rc` on the L1 baseline
(FF, matched cosine, beta2=0.95, wd=1e-3, grad clipping) were
hypothesised to be independent paths to the same gain → would stack
additively in round 5. PR #437 shows that, at minimum for the wd × FF
pair, **they overlap on rc-camber** (destructive stacking), **compose
on cruise-camber** (additive), and **flip sign on in-dist** (FF gives
the model enough positional richness that higher wd helps in-dist
where it hurt under L1-only).

**Implications for the round-4 compose tests in flight**:
- #446 (thorfinn, beta2=0.95 on L1+FF) — beta2 may have similar overlap
  story as wd (both are optimiser-side regularisers).
- #447 (fern, EMA on L1+FF) — orthogonal mechanism (weight averaging),
  most likely additive.
- #462 (edward, grad clipping on L1+FF + matched cosine) — stability
  mechanism, may overlap with matched-cosine's gradient-decay effect.
- #432 (nezuko, log(Re) FF on L1+FF) — different input dimension,
  most likely additive.

Per-split analysis is now the load-bearing diagnostic, not just the
headline. **Round-5 cannot be a naive "stack everything"** — some
levers will compete on rc-camber even if they each individually
improved it on the L1 baseline.

### Round-5 priorities reordered

1. **wd downward sweep on L1+FF (3e-4, 5e-4, 7.5e-4)** — interpolates
   between baseline (1e-4) and this PR (1e-3). Tests for an interior
   wd that captures cruise/in-dist gain without rc regression. **Most
   informative round-5 single-knob.**
2. **FF frequency-count variation** — tests whether the rc-vs-cruise
   asymmetry is about geometry-interpolation regime
   (boundary-shoulder rc M=6-8 vs centre-band cruise M=2-4).
3. **DropPath / stochastic depth** — different regularisation
   dimension than weight magnitude; may help rc-camber where wd
   doesn't.

Re-assigning frieren to the matched-cosine variant of (1):
L1+FF + matched cosine + wd=5e-4 (intermediate wd on the live
baseline).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

### Harness debt note

Student observed a stale concurrent run dir
(`model-l1ff_wd_1e-3-20260428-021302/`, no agent prefix) created by
the entrypoint launching a parallel `train.py` while their main run
was still in test eval. Empty config-only dir, crashed on epoch 1
from GPU contention. Entrypoint should serialise per-process to
prevent this. Recording for harness cleanup.

---

## 2026-04-28 02:30 — PR #423 (CLOSED, validated on L1 / loses to current): gradient clipping `max_norm=1.0`
- Branch: `charliepai2d3-edward/l1-grad-clip-1` (deleted on close)
- Hypothesis: gradient clipping caps the global gradient norm,
  preventing high-Re samples from dominating; predicted −1% to −3%.
- Config: L1 baseline (pre-FF, pre-matched-cosine), `max_norm=1.0` added
  before `optimizer.step()`. Single-line code change.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) | vs current baseline (PR #389, 90.90) |
|--------|--------:|---------------------------------:|-------------------------------------:|
| `val_avg/mae_surf_p`  | 97.15 | **−5.4%** ✓ above band | +6.9% (loses) |
| `test_avg/mae_surf_p` | 87.61 | **−10.4%** ✓ above band | +8.4% (loses) |

### Per-split val (best epoch 14)

| split | this PR | L1 baseline | Δ |
|-------|--------:|------------:|--:|
| val_single_in_dist     | 120.42 | 121.18 | −0.6% |
| val_geom_camber_rc     | 106.21 | 125.01 | **−15.0%** |
| val_geom_camber_cruise |  73.33 |  73.22 | +0.15% |
| val_re_rand            |  88.63 |  91.14 | −2.8% |

### Pre-clip grad norm diagnostic (most useful round-3 instrumentation)

| epoch | mean | max |
|------:|-----:|----:|
| 1  | 60.25 | 150.23 |
| 5  | 53.88 | 137.94 |
| 10 | 50.77 | 123.23 |
| 14 | 45.73 | 123.07 |

Pre-clip grad norms are **50× max_norm=1.0** — clip is doing heavy
work, not a no-op. Round-5: tighter values (`max_norm ∈ {0.5, 0.1}`)
are well within "still doing work" territory.

### Decision

**Closed.** Same merge-order pattern as PRs #298, #395, #419. Lever
validated on L1 baseline; lost by merge timing to PR #389.

**Notable: 5th convergent OOD-camber improvement signature this round.**
`val_geom_camber_rc` −15.0% lines up with PR #400 (FF, −20.8%), PR #389
(matched cosine, −19.4%), PR #419 (beta2=0.95, −13.6%), PR #395
(wd=1e-3, −11.9%). Five independent mechanisms — input encoding,
schedule, optimiser, regularisation, stability — same direction of
effect.

Re-assigning edward to L1+FF + matched cosine + grad clipping compose
test (three-lever stack on the post-#389 advisor).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close. Pre-clip grad-norm trajectory recorded in
this log entry as a round-5 reference.

---

## 2026-04-28 02:28 — PR #389 (MERGED): L1 + matched cosine schedule (`--epochs 14`)
- Branch: `charliepai2d3-askeladd/l1-cos-matched-14`
- Hypothesis: match cosine T_max to actual wallclock budget (14 epochs)
  so the schedule fully decays inside the 30-min cap; predicted −5% to
  −15%.
- Config: L1 baseline (PR #280, pre-FF), `--epochs 14`, all other knobs
  at defaults. CLI-only diff.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) | vs L1+FF baseline (PR #400, 91.87) |
|--------|--------:|---------------------------------:|-----------------------------------:|
| `val_avg/mae_surf_p`  | **90.90** | **−11.4%** | **−1.06%** ✓ |
| `test_avg/mae_surf_p` | **80.84** | **−17.3%** | −0.33% ✓ |
| Per-epoch wallclock | ~131 s | unchanged | unchanged |
| Peak GPU memory | 42.14 GB | unchanged | unchanged |
| **Reproducibility** | 3 re-runs: 90.90 / 91.47 / 91.94 | ~1% spread |

### Per-split val (best epoch 14)

| split | this PR | L1 baseline | Δ |
|-------|--------:|------------:|--:|
| val_single_in_dist     | 105.82 | 121.18 | −12.7% |
| val_geom_camber_rc     | 100.82 | 125.01 | **−19.4%** |
| val_geom_camber_cruise |  71.37 |  73.22 |  −2.5% |
| val_re_rand            |  85.60 |  91.14 |  −6.1% |

### Per-split test (best-val checkpoint)

| split | mae_surf_p |
|-------|-----------:|
| test_single_in_dist     | 94.78 |
| test_geom_camber_rc     | 88.30 |
| test_geom_camber_cruise | 59.67 |
| test_re_rand            | 80.62 |

### Validation trajectory — the cosine tail is where the gain lives

```
epoch  1: 260.33  best
epoch  2: 195.53  best
…
epoch 11: 103.53  best
epoch 12:  94.07  best
epoch 13:  91.73  best
epoch 14:  90.90  best   ← T_max=14, LR ≈ 0
```

The last 4 epochs (11→14) cut another 12.6 mae_surf_p as the LR anneals
through the cosine tail — exactly the "fine-tune phase" the L1 baseline
was missing.

### Decision

**Merged.** First round-3 PR with effect size large enough (3-seed
spread ~1%, vs −11.4% delta on L1 baseline) to clearly clear the
seed-noise floor that has been blocking attribution on every other
lever this round.

### Convergent OOD-camber signal — now 5 levers

`val_geom_camber_rc` was the dominant winner here too (−19.4%), the
same per-split signature as PR #400 (FF, −20.8%), PR #419 (beta2=0.95,
−13.6%), PR #395 (wd=1e-3, −11.9%), and PR #423 (grad clipping,
−15.0%). **Five different mechanisms hitting the same direction of
effect on OOD-camber generalisation.** The compose tests in flight
(#437 wd, #432 log(Re), #446 beta2, #447 EMA) plus the new ones
(matched cosine + grad clipping compose) will reveal additivity vs
shared dynamic.

### Caveat — measurement on L1-only branch, not L1+FF advisor

PR #389 was branched off the pre-FF advisor, so the measured 90.90 is
L1 + matched cosine *without* FF. The advisor `train.py` now retains
FF (from PR #400) and adds the metrics dir from this PR. The first
round-4 PR running on the post-merge advisor with `--epochs 14` will
give the L1+FF + matched cosine compose number. Expected ≤ 90.90 if
the levers compose; could be as low as ~80 if they fully stack.

---

## 2026-04-28 01:50 — PR #302 (CLOSED): Huber (smooth-L1, δ=1.0) surface loss
- Branch: `charliepai2d3-tanjiro/huber-surf-loss` (deleted on close)
- Hypothesis: Huber on surface loss bounds gradient on heavy-tailed
  high-Re extremes while keeping L2 smoothness near zero; predicted
  −3% to −10% on val_avg/mae_surf_p.
- Config: pre-L1 advisor (MSE volume, MSE surface). PR replaced surface
  MSE with Huber(δ=1.0). Volume MSE unchanged.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) | vs current L1+FF baseline (PR #400, 91.87) |
|--------|--------:|---------------------------------:|-------------------------------------------:|
| `val_avg/mae_surf_p`  | 105.53 | +2.8% | +14.9% |
| `test_avg/mae_surf_p` |  97.41 | −0.3% (near-tie) | +20.0% |

### Per-split val (best epoch 14) — narrow regime-specific effect

| split | this PR | L1 baseline | Δ |
|-------|--------:|------------:|--:|
| val_single_in_dist     | 121.89 | 121.18 | +0.6% |
| val_geom_camber_rc     | 113.93 | 125.01 | **−8.9% (Huber wins)** |
| val_geom_camber_cruise |  88.62 |  73.22 | +21.0% (worse) |
| val_re_rand            |  97.67 |  91.14 | +7.2% (worse) |

### Decision

**Closed.** Net regression on the primary metric. The lever has narrow
regime-specific effect (high-Re raceCar tandem) but doesn't cleanly
improve the headline. Student's analysis: δ=1.0 is too generous —
post-warmup most residuals are already inside ±1 std, so Huber acts
nearly identically to MSE on the bulk of nodes, losing the L1 alignment
with MAE on splits where outliers aren't the bottleneck. δ→0 is L1
(current baseline), so the lever range is bracketed and the maximum
useful effect is bounded.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 01:48 — PR #396 (CLOSED, broken canonical / 0.999 follow-up tied): EMA of weights
- Branch: `charliepai2d3-fern/l1-ema-d9999` (deleted on close)
- Hypothesis: maintain EMA shadow weights, evaluate val/test under EMA;
  predicted −1% to −3% on val_avg/mae_surf_p.
- Config: L1 baseline (PR #280), EMA every step + swap for eval. The
  canonical committed value was `EMA_DECAY=0.9999`; student also ran
  follow-up `EMA_DECAY=0.999`.

### Headline

| Metric | EMA 0.9999 (canonical) | EMA 0.999 (follow-up) | vs L1 (102.64 / 97.73) | vs L1+FF (91.87 / 81.11) |
|--------|----------------------:|---------------------:|----------------------:|-------------------------:|
| `val_avg/mae_surf_p`  | **317.92 (BROKEN)** | 92.00 | 0.999: **−10.4%** ✓ | 0.999: +0.14% (tie) |
| `test_avg/mae_surf_p` | 300.79 (BROKEN) | 82.54 | 0.999: **−15.5%** ✓ | 0.999: +1.76% |

### Why 0.9999 was broken — student diagnosis

`EMA_DECAY=0.9999` averages over ~10K steps. The 30-min wallclock cap
allows ~5K optimizer steps (14 epochs × 375 batches). EMA shadow is
dominated by random init throughout: at step 5,300, ~59% of the
shadow is still init weight. The val curve under 0.9999 EMA descends
monotonically (387.9 → 317.9) but never escapes the random-init basin.

**Budget-aware EMA rule** (student's derivation): `EMA_DECAY = 1 − 1/N`
with `N ≈ 0.2 × total_steps`. For 5,300 steps, `N ≈ 1,000`,
`EMA_DECAY ≈ 0.999` — which is exactly the value that worked. Round-5
should bake this rule into the train.py.

### Per-split val (best epoch 14, EMA 0.999 — uniform improvement)

| split | EMA 0.999 | L1 baseline | Δ |
|-------|----------:|------------:|--:|
| val_single_in_dist     | 108.20 | 121.18 | −10.7% |
| val_geom_camber_rc     | 103.76 | 125.01 | −17.0% |
| val_geom_camber_cruise |  70.22 |  73.22 |  −4.1% |
| val_re_rand            |  85.84 |  91.14 |  −5.8% |

### Decision

**Closed.** Two reasons:
1. The committed canonical value (0.9999) is broken — merging would
   actively harm the baseline. The working value (0.999) was a
   reverted local edit, not in the diff.
2. EMA 0.999 result (val 92.00) is essentially tied with the current
   L1+FF baseline (91.87, ~0.14% gap, well within seed noise). Even if
   the diff were rewritten to canonical 0.999, the win vs the current
   baseline is not clean.

Re-assigning to fern as a compose test on L1+FF with `EMA_DECAY=0.999`
baked in canonically. Predicted small win on val (uniform smoothing
benefit), bigger on test (the per-split late-trajectory smoothing
showed a −25.8% test cruise improvement on L1).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 01:48 — PR #419 (CLOSED, validated on L1 / loses to L1+FF): AdamW(beta2=0.95)
- Branch: `charliepai2d3-thorfinn/l1-adamw-beta2-0-95` (deleted on close)
- Hypothesis: change AdamW `beta2` from 0.999 to 0.95 (transformer
  convention); averages squared gradients over ~20 steps not ~1000;
  predicted −1% to −4%.
- Config: L1 baseline (PR #280), single keyword arg added to AdamW.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 (PR #280, 102.64) | vs L1+FF (PR #400, 91.87) |
|--------|--------:|------------------------:|--------------------------:|
| `val_avg/mae_surf_p`  | 99.70 | **−2.87%** ✓ in band | +8.5% (loses) |
| `test_avg/mae_surf_p` | 91.50 | **−6.37%** ✓ above band | +12.8% (loses) |

### Per-split val (best epoch 14) — OOD-camber dominant

| split | this PR | L1 baseline | Δ |
|-------|--------:|------------:|--:|
| val_single_in_dist     | 123.33 | 121.18 | +1.77% |
| val_geom_camber_rc     | 108.01 | 125.01 | **−13.60%** |
| val_geom_camber_cruise |  77.82 |  73.22 | +6.28% |
| val_re_rand            |  89.65 |  91.14 | −1.63% |

### Per-split test (NaN-safe, best-val checkpoint)

| split | this PR | L1 baseline | Δ |
|-------|--------:|------------:|--:|
| test_single_in_dist     | 110.64 | 109.80 | +0.77% |
| test_geom_camber_rc     |  99.77 | 114.60 | **−12.94%** |
| test_geom_camber_cruise |  66.89 |  79.92 | **−16.30%** |
| test_re_rand            |  88.70 |  86.58 | +2.45% |

### Decision

**Closed.** Same merge-order pattern as PR #298 (FF on MSE), PR #395
(wd on L1) — validated on assigned baseline, loses to current. The
per-split signal is a clean OOD-geometry generalisation story (held-out
camber tracks dominate the win, while in-dist and re_rand are roughly
flat). Mechanistic read: the long-window squared-grad average
(beta2=0.999) over-dampens nominally-larger off-distribution gradients;
beta2=0.95 lets the second-moment respond to those signals within
~20 steps.

**Notable convergent signal**: PR #395 (wd=1e-3), PR #400 (spatial FF),
and PR #419 (beta2=0.95) all hit the same OOD-camber-improvement
pattern (`val_geom_camber_rc` -11.9% / -20.8% / -13.6% respectively).
Three different mechanisms (regularisation / input encoding / optimiser
second-moment), same direction of effect. The compose tests will
reveal whether they each hit independent paths to the same
generalisation gain (additive) or share a common dynamic (diminishing).

Re-assigning thorfinn to L1+FF + AdamW(beta2=0.95) compose test.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 01:35 — PR #395 (CLOSED, validated on L1 / loses to L1+FF): weight_decay 1e-4 → 1e-3
- Branch: `charliepai2d3-frieren/l1-wd-1e-3` (deleted on close)
- Hypothesis: 10× weight_decay bump on the L1 baseline addresses
  under-regularisation on the small training set; predicted −1% to −5%.
- Config: L1 surface loss baseline (PR #280), `weight_decay=1e-3`, all
  other knobs at defaults. CLI-only diff.

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) | vs current L1+FF baseline (PR #400, 91.87) |
|--------|--------:|---------------------------------:|-------------------------------------------:|
| `val_avg/mae_surf_p`  | 100.99 | **−1.6% (validated, in predicted band)** | **+9.9% (loses to current)** |
| `test_avg/mae_surf_p` |  91.68 | **−6.2% (validated, larger than predicted)** | +13.0% (loses to current) |
| Per-epoch wallclock   | ~132 s | unchanged | unchanged |
| Peak GPU memory       | 42.1 GB | unchanged | unchanged |

### Per-split val (best epoch 14) — regularisation hypothesis confirmed

| split | L1 baseline | this PR | Δ |
|-------|------------:|--------:|--:|
| val_single_in_dist     | 121.18 | 128.64 | **+6.2% (worse)** |
| val_geom_camber_rc     | 125.01 | 110.16 | **−11.9% (much better)** |
| val_geom_camber_cruise |  73.22 |  74.98 | +2.4% (slight worse) |
| val_re_rand            |  91.14 |  90.19 | −1.0% |

### Decision

**Closed.** Same merge-order pattern as PR #298 (nezuko's MSE-side
Fourier features): the lever was validated on its assigned baseline
(L1) but loses to the current baseline (L1+FF) which landed mid-round.

**The per-split signal validates the regularisation hypothesis
precisely.** Student predicted that improvement on OOD axes + slight
regression on in-dist would be direct evidence that the L1 regime was
under-regularised on OOD axes specifically — and that pattern is
exactly what the data shows (`val_geom_camber_rc` −11.9% with
`val_single_in_dist` +6.2%).

But that OOD-axis work overlaps with the spatial-FF lever that landed
in PR #400 (which also improved `val_geom_camber_rc` by 20.8%). The
compose question is: is `wd=1e-3` doing additional OOD-camber work
beyond FF, or redundant work? **Re-assigning frieren to the compose
test** to find out.

### Round-4 implications

- **wd sweep (5e-3, 1e-2, 3e-2)** is round-5 priority #1 if the compose
  test wins. The per-split signal — OOD up, in-dist down, but not yet
  in-dist-dominated — suggests there's more headroom.
- **DropPath / stochastic depth** is the orthogonal regularisation
  alternative. Different mechanism (residual paths vs weight magnitude)
  — would compose with wd if both win.
- **Logged-loss accumulator NaN**: same as nezuko's PR #400 finding.
  `evaluate_split`'s squared-error sum doesn't have the per-sample skip
  that `accumulate_batch` got in commit `2eb5c7f`. Round-5 cleanup PR.
- **Schedule truncation**: every round-3 PR is at a 14-of-50-epoch cap.
  PR #389 (matched cosine `--epochs 14`) is the diagnostic for whether
  full convergence changes any of these per-PR rankings.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close. Headline numbers above are from the PR
results comment.

---

## 2026-04-28 01:30 — PR #400 (MERGED): L1 + 8-frequency Fourier positional features (compose)
- Branch: `charliepai2d3-nezuko/l1-fourier-pos-8freq`
- Hypothesis: port the validated FF lever (PR #298, won −13.7% on MSE)
  to the L1 baseline; tests whether L1 and Fourier features compose;
  predicted −2% to −8%.
- Config: L1 surface loss (already in advisor `train.py`), 8-freq
  Fourier features for `(x, z)` (one helper + `fun_dim` update + 3
  concat sites), all other knobs at L1-baseline defaults.
- Diff: 4 files (`train.py` +20 lines, metrics dir).

### Headline (best-val checkpoint, epoch 14/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|---------------------------------:|
| `val_avg/mae_surf_p`  | **91.87** | **−10.5%** |
| `test_avg/mae_surf_p` | **81.11** | **−17.0%** |
| Per-epoch wallclock   | ~131 s   | ≈ same |
| Peak GPU memory       | 42.38 GB | +0.25 GB |
| Param count           | 670,551  | +8,192 (input MLP `linear_pre`) |

### Per-split val (best epoch 14) — gap *widened*, not closed

| split | L1 baseline | this PR | Δ |
|-------|------------:|--------:|--:|
| val_single_in_dist     | 121.18 | 117.24 | **−3.3%** (smallest gain) |
| val_geom_camber_rc     | 125.01 |  98.99 | **−20.8%** (largest gain) |
| val_geom_camber_cruise |  73.22 |  68.61 | −6.3% |
| val_re_rand            |  91.14 |  82.64 | −9.3% |

### Per-split test (best-val checkpoint)

| split | mae_surf_p |
|-------|-----------:|
| test_single_in_dist     | 100.17 |
| test_geom_camber_rc     |  85.47 |
| test_geom_camber_cruise |  61.17 |
| test_re_rand            |  77.64 |

### Decision

**Merged** as the new round-3 baseline. FF and L1 are **independent
levers, not redundant** — most of the FF effect (was ~14% on MSE in
PR #298) survived the loss-shape change to L1 (~10% here). Test gain
(~17%) larger than val gain (~10.5%) is real generalisation evidence,
not val overfit.

### Mechanistic read

Student's analysis: in-distribution samples cluster near training data,
so the model interpolates fine without high-frequency positional
encoding. FF helps most where the model needs to **extrapolate sharp
pressure features it hasn't seen exactly** (camber holdouts). L1 then
sharpens the loss gradient for the high-magnitude tail of the
surface-p distribution, most pronounced in OOD samples. Two levers,
two failure modes — they compose nearly additively, with the bigger
gain showing where both failure modes are active.

### Round-4 implications

- **`val_single_in_dist` is now the dominant bottleneck** at 117.24
  (vs 68.61 cruise, 82.64 re_rand, 98.99 rc camber). Round-5 priorities
  should target the high-Re raceCar single regime specifically.
- **Cross-regime axis (`val_re_rand`)** improved less than camber-OOD
  axes — student's suggested `log(Re)` Fourier features (extending the
  proven FF lever to scalar log-Re) is the natural next step. Assigning
  to nezuko.
- **Frequency search (8 vs 12 vs 16 vs 4)** is round-5 work — hold
  until log(Re) FF lands.
- **`loss=NaN`/`vol_loss=Inf` on `test_geom_camber_cruise`** in
  metrics.yaml is a pre-existing logged-loss accumulator issue (the
  loss accumulator in `evaluate_split` doesn't have the `nan_to_num`
  treatment that `accumulate_batch` got in commit `2eb5c7f`). MAE
  numbers themselves are clean. Round-5 cleanup PR.

---

## 2026-04-28 01:10 — PR #285 (CLOSED): surf_weight 10 → 30 (MSE)
- Branch: `charliepai2d3-edward/surf-weight-30` (deleted on close)
- Hypothesis: tripling the surface loss weight pushes more gradient
  signal to surface nodes; predicted −2% to −8%.
- Config: MSE surface loss (pre-L1 advisor), `surf_weight=30`, all
  other knobs at defaults.

### Headline (best-val checkpoint, epoch 14)

| Metric | This PR (canonical run) | vs L1 baseline (PR #280, 102.64) | vs MSE peer baseline (PR #306, 135.20) |
|--------|------------------------:|---------------------------------:|---------------------------------------:|
| `val_avg/mae_surf_p`  | 125.53 | +22% | −7.2% |
| `test_avg/mae_surf_p` | 112.81 | +15.4% | — |
| Peak GPU memory       | 42.1 GB | — | — |

### Cross-seed variance — the central observation

| seed | val_avg/mae_surf_p (surf_weight=30) | val_avg/mae_surf_p (surf_weight=10) |
|------|------------------------------------:|------------------------------------:|
| 1 | 144.82 (NaN'd test, val still valid) | 127.95 |
| 2 | 125.53 (canonical) | 131.40 |
| **mean ± span** | **135.18 ± 9.6** | **129.67 ± 1.7** |

The surf_weight=30 effect (~3% directional) is **smaller than the
within-condition seed spread (~13%)**. Cannot separate signal from noise
at single replicates.

### Decision

**Closed.** Above-threshold regression vs current L1 baseline. The more
useful round-4 takeaway is that this is the **third PR in a row** where
the predicted effect is comparable to or smaller than measured cross-seed
variance (preceded by frieren #292 slice_num=128 with ~4% noise floor,
fern #288 warmup+lr=1e-3 with std 5.7). Round-3 is operating well below
the seed-noise floor for everything except the L1 surface loss change
(which moved val by 24%, well clear of any seed noise observed).

### Round-4 implications

- **Seed pinning is round-4 infra priority #1.** With current single-run
  comparisons we cannot distinguish ~3% effects from noise. `torch.manual_seed`
  + matching numpy/python seeds at the top of `train.py`, and a documented
  per-PR seed in the experiment metadata.
- **Replicate budget**: at 30 min/run × 8 students/round, doing 3 seeds per
  hypothesis halves the round throughput. This is a real tradeoff — but
  a 3% effect at 1 seed is uninterpretable, and a clean 3% at 3 seeds is
  worth a round-4 win.
- **`test_geom_camber_cruise` sample 020 is a CFD divergence** (761
  non-finite p values). Worth a heads-up to whoever owns the dataset
  preprocessing; not actionable from the advisor branch.

### Bug-fix attribution

Edward independently rediscovered the `0 * NaN = NaN` scoring bug fixed
on advisor branch as commit `2eb5c7f`. Three students (thorfinn, alphonse,
edward) converged on the same fix in the same shape — solid validation
of the merged patch.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 01:05 — PR #390 (CLOSED): L1 baseline + bs=8 + sqrt LR (compose test)
- Branch: `charliepai2d3-thorfinn/l1-bs8-sqrt-lr` (deleted on close)
- Hypothesis: composing the two merged round-3 winners (PR #280 L1 +
  PR #306 bs=8/sqrt-LR) on the L1 baseline; predicted −3% to −8%.
- Config: L1 surface loss (already in advisor `train.py`), `bs=8`,
  `lr=7.07e-4`, all other knobs at defaults.

### Headline (best-val checkpoint, epoch 13/14)

| Metric | This PR | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 119.42 | +16.4% (loses) |
| `test_avg/mae_surf_p` | 105.92 | +8.4% (loses) |
| Peak GPU memory       | 84.25 GB | 2.0× the L1 baseline at bs=4 |
| Per-epoch wallclock   | ~130 s | ≈ same as L1 baseline |

### Per-split val (best epoch 13)

| split | this PR | vs L1 baseline (PR #280) |
|-------|--------:|------------------------:|
| val_single_in_dist     | 172.27 | +42.2% (much worse) |
| val_geom_camber_rc     | 126.38 | +1.1% (≈ tie) |
| val_geom_camber_cruise |  82.57 | +12.8% |
| val_re_rand            |  96.45 | +5.8% |

### Decision

**Closed.** Clean negative — every split regressed, the smoothest split
(`val_single_in_dist`) regressed the most. The student's analysis is the
takeaway: **L1's bounded-derivative property already absorbs the bs=8
noise-reduction effect**. The two round-3 winners are not orthogonal —
bs=8 specifically suppressed the heavy-tailed *squared*-error gradient
noise that L1 already bounds at ±1 per sample. Under a wallclock-iso
budget bs=8 has half the optimizer steps of bs=4, so under cosine
truncation it finishes less converged.

### Round-4 implications

- **bs=12 + sqrt(3) lr is not in reach** without throughput infra:
  linear VRAM extrapolation gives `bs=12 → ~126 GB > 96 GB cap`.
  Any "bigger batch" lever requires activation checkpointing or BF16
  first.
- **AdamW step-size scaling vs batch is not √2 for L1**. The √2-LR
  scaling is the SGD-fixed-noise prescription; AdamW's effective step
  for a bounded loss like L1 surface is closer to flat than to sqrt.
  Round-4 if we revisit batch: try `bs=8, lr=5e-4` (no scaling) and
  `bs=8, lr=6e-4` (geometric mean) as cheap intermediate runs.
- **Closed lever**: bs=8 + L1 doesn't win. Don't compose with anything
  else in round 4.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:35 — PR #298 (CLOSED, positive on MSE / loses to L1): 8-freq Fourier positional features
- Branch: `charliepai2d3-nezuko/fourier-pos-features` (deleted on close)
- Hypothesis: Fourier positional encoding of `(x, z)` at 8 octave-spaced
  frequencies addresses MLP/attention spectral bias against high-frequency
  content of low-d inputs; predicted −2% to −8%.
- Config: MSE surface loss (pre-L1 advisor), all other knobs at defaults.
  +1.2% params (32 extra input channels at the first preprocess MLP).

### Headline (best-val checkpoint, epoch 13/14)

| Metric | This PR | vs PR #306 (MSE peer, 135.20) | vs PR #280 (L1, 102.64) |
|--------|--------:|------------------------------:|------------------------:|
| `val_avg/mae_surf_p`  | 116.62 | **−13.7% (win on MSE)** | +13.6% (loses to L1) |
| `test_avg/mae_surf_p` | 105.85 | **−14.1%** | +8.3% |
| Peak GPU memory       | 42.36 GB | — | — |
| Param count           | 670,551 | +1.2% (+8,192 weights) | — |
| Epochs in 30-min cap  | 14/50 | — | — |

### Per-split val (best epoch 13)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 138.75 |
| val_geom_camber_rc     | 122.13 |
| val_geom_camber_cruise |  95.00 |
| val_re_rand            | 110.60 |

### Decision

**Closed.** The hypothesis was validated on the MSE baseline assigned to
this branch (−13.7% on val), but L1 surface loss landed mid-round and
became the new baseline at val 102.64 — a much larger lever than Fourier
features. Per the merge criterion (must beat current baseline), this PR
does not merge. Per the close criterion (>5% regression vs current
baseline), it is technically closeable — but the regression is an
artefact of the merge-order race, not a failure of the lever.

**The lever is on the round-4 candidate list** as L1 + Fourier features.
The student is the right person to run that compose test (already owns
the code). They've been re-assigned to test exactly that.

**Useful per-split insight**: student observed the worst val split was
`val_single_in_dist` (138.75), inverting their prediction that
single-in-dist would benefit most from Fourier features. The split
ranking is dominated by raceCar high-Re extremes, not by spectral bias
on input position — which points round 5 toward applying Fourier
features to `log(Re)` and other scalar inputs (student's own follow-up #2).

**On bug fix**: student's pred-side workaround in `evaluate_split` is
redundant on the current advisor branch — the GT-side fix landed as
commit `2eb5c7f` per thorfinn/alphonse's earlier identification.
Student validated the merged approach (their option (2) recommendation
matches the merged fix exactly).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:30 — PR #288 (CLOSED): 3-epoch warmup + cosine to 1e-5, peak lr=1e-3
- Branch: `charliepai2d3-fern/lr-warmup-peak1e3` (deleted on close)
- Hypothesis: warmup unlocks higher peak LR; cosine to `eta_min=1e-5`
  preserves late-training fine-tune; predicted −2% to −5%.
- Config: MSE surface loss (pre-L1 advisor), `bs=4`, `lr=1e-3`,
  3-epoch LinearLR warmup with `start_factor=0.1`, then
  `CosineAnnealingLR(T_max=47, eta_min=1e-5)`.
- Diff: ~6 lines of imports + scheduler swap in `train.py`.

### Headline (best-val checkpoint, run 3 of 3 seeded re-runs)

| Metric | This PR | vs PR #306 baseline (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|-----------------------------:|---------------------------------:|
| `val_avg/mae_surf_p` (best, epoch 13/14) | 147.50 | +9.1% | +43.7% |
| `test_avg/mae_surf_p` | 130.55 | +6.0% | +33.5% |
| Peak GPU memory       | 42.11 GB | — | — |
| Per-epoch wallclock   | ~131 s   | — | — |

### Cross-run variance (3 unseeded runs)

| run | best epoch | best val_avg/mae_surf_p |
|-----|---------:|-----------------------:|
| v1  | 12       | 136.88 |
| v2  | 9        | 145.12 |
| v3 (canonical) | 13 | 147.50 |
| **mean ± std** | — | **143.2 ± 5.7** |

Even the **best** run (136.88) does not beat the prior MSE baseline
(135.20). Cross-run std ~5.7 is large enough to swamp ~5% schedule
effects — flagged as round-4 infra debt (seed pinning).

### Decision

**Closed.** >5% regression on the primary ranking metric across 3 seeded
re-runs. The student's analysis nailed the failure mode: this is a
long-horizon optimizer change being evaluated under a short-horizon
wallclock cap. Three structural problems compound:

1. Warmup eats 21% of the actual epoch budget (3/14, vs the 6% the
   schedule was designed for).
2. Higher peak LR amplifies seed noise at bs=4 — bouncy descent shows
   the optimizer can't settle in 11 post-warmup epochs.
3. `eta_min=1e-5` is irrelevant — the LR is still ~9e-4 at the timeout.

The corrective experiment (matched-cosine `T_max=14`) is being run by
askeladd in PR #389 on the L1 baseline. A "1-epoch warmup + matched
cosine" variant would be a reasonable round-5 PR if #389 wins.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:30 — PR #292 (CLOSED): slice_num 64 → 128
- Branch: `charliepai2d3-frieren/slice-num-128` (deleted on close)
- Hypothesis: doubling PhysicsAttention slice tokens halves the
  per-token mesh-node neighborhood and lets the slice basis represent
  finer flow structure; predicted −3% to −8%.
- Config: MSE surface loss (pre-L1 advisor), all other knobs at
  defaults. Single-line diff: `slice_num=128`.

### Headline (best-val checkpoint, epoch 9/11)

| Metric | This PR | vs PR #306 baseline (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|-----------------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 149.08 | +10.3% | +45.3% |
| `test_avg/mae_surf_p` | 136.85 | +11.1% | +40.0% |
| Peak GPU memory       | 54.5 GB | — | — |
| Param count           | 0.67 M | +2% vs slice_num=64 | — |
| Epochs in 30-min cap  | 11/50 | — | — |

### Per-split val (best epoch 9)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 193.75 |
| val_geom_camber_rc     | 160.05 |
| val_geom_camber_cruise | 109.81 |
| val_re_rand            | 132.70 |

### Decision

**Closed.** The student's variance observation is the key takeaway: a
separate identical-config run hit val 142.76 at epoch 11 vs 149.08 here
— ~4% spread from sampler/init noise alone, comparable to the predicted
effect size. With only 11 of 50 epochs and the cosine never decaying,
the signal-to-noise ratio for slice count vs noise was too low to
attribute anything cleanly.

slice_num=128 is **not ruled out** for round 4 — it just needs either
tighter variance control (seeded runs) or a much larger expected effect.
The +2% param bump and 54.5 GB peak memory at slice_num=128 confirm
plenty of headroom for slice_num=256 once throughput is unlocked.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:18 — PR #283 (CLOSED): Wider+deeper Transolver (h=192, l=6, head=6, slices=96)
- Branch: `charliepai2d3-askeladd/wider-deeper-h192-l6-s96` (deleted on close)
- Hypothesis: scale capacity along 4 axes simultaneously; predicted −3% to −8%.
- Config: `bs=4`, `lr=5e-4`, all other knobs at defaults.

### Headline (best-val checkpoint, epoch 7/7)

| Metric | This PR | vs original baseline (PR #306, 135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|---------------------------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 166.64 | +23.3% | +62.4% |
| `test_avg/mae_surf_p` | 155.95 | +26.6% | +59.6% |
| Per-epoch wallclock   | 278 s  | 2.1× slower than baseline shape | — |
| Peak GPU memory       | 83.84 GB | / 96 GB cap → 12 GB headroom | — |
| Epochs in 30-min cap  | 7/50   | half of baseline shape | — |
| Param count           | 1.72 M | +1.0 M vs baseline | — |

### Per-split val (best epoch 7)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 198.42 |
| val_geom_camber_rc     | 183.45 |
| val_geom_camber_cruise | 140.69 |
| val_re_rand            | 144.01 |

### Decision

**Closed.** >5% regression on val_avg/mae_surf_p vs both prior baselines.
Per-epoch the bigger model is **12% better** than the round-3 baseline shape
at matched epoch index (epoch 7: 166.64 vs 188.54), so the architecture has
genuine merit — but the 2.1× per-epoch slowdown halves the cosine-anneal
budget under the 30-min cap, wiping out the per-epoch gain. Compute
starvation is structural; revisits blocked on throughput infra
(mixed-precision / activation checkpointing).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` — branch
was deleted on close before metrics could be cherry-picked into the
advisor branch. Headline numbers above are from the PR results comment.

### Round-4 implications

- Joint scaling on 4 axes is too coarse for clean attribution. Per-axis
  PRs (frieren #292 slice_num, in flight) will give cleaner signals.
- Drop "wider+deeper" from round-4 candidate set until throughput infra
  lands — at that point a single-axis bigger-model PR is justified.
- Student also flagged a pred-side `evaluate_split` y-finite bug fix
  worth keeping in mind if any future PR produces NaN test averages
  from clean GT (current scoring fix only handles GT-side non-finite).

---

## 2026-04-28 00:15 — PR #366 (CLOSED): mlp_ratio 2 → 4
- Branch: `charliepai2d3-thorfinn/mlp-ratio-4` (deleted on close)
- Hypothesis: doubling MLP per-token capacity inside each TransolverBlock;
  predicted −3% to −8% on val_avg/mae_surf_p.
- Config: bs=8 → bs=6 (OOM at bs=8 with the wider MLP), `lr=7.07e-4`
  (kept from baseline PR #306 instructions).

### Headline (best-val checkpoint, epoch 11/13)

| Metric | This PR | vs PR #306 (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|--------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 144.70 | +7.0%  | +41.0% |
| `test_avg/mae_surf_p` | 132.44 | +7.5%  | +35.5% |
| Peak GPU memory       | 78.16 GB | — | — |
| Epochs in 30-min cap  | 13/50  | −1 epoch vs baseline | — |
| Param count           | 0.99 M | +0.33 M  | — |

### Per-split val (best epoch 11) — **revealing pattern**

| split | This PR | vs PR #306 baseline |
|-------|--------:|--------------------:|
| val_single_in_dist     | 176.83 | **−7.0% (improved)** |
| val_geom_camber_rc     | 159.32 | +15.1% |
| val_geom_camber_cruise | 112.82 | +15.2% |
| val_re_rand            | 129.81 | +13.6% |
| **val_avg**            | 144.70 | +7.0% |

### Decision

**Closed.** >5% regression on val_avg/mae_surf_p vs both prior baselines.

The split pattern is the takeaway: in-distribution improved (single-foil
better fit), every OOD axis regressed. Classic generalisation-gap shift —
extra MLP capacity is being spent memorising training-distribution
structure that doesn't transfer to held-out cambers / Re. Validation
peaked at epoch 11 then degraded (144.70 → 159.04 → 171.83 across epochs
11→12→13), confirming overfit before the cosine could anneal.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` — branch
was deleted on close. Headline numbers above are from the PR results
comment.

### Round-4 implications

- mlp_ratio=4 dropped from candidate set per the standalone-loss rule.
- Two follow-up directions remain interesting and would justify their own
  PRs if revisited: (a) `mlp_ratio=4` only in last 1-2 blocks (asymmetric
  capacity), (b) `mlp_ratio=4` paired with stronger regularisation
  (`dropout=0.05` or `weight_decay=2e-4`) to test whether the
  generalisation gap closes.
- The OOM at bs=8 with +0.33 M params is a useful VRAM-headroom signal:
  the bs=8 MSE baseline (PR #306) was running close to the limit.

---

## 2026-04-28 00:03 — PR #280: L1 surface loss to align gradient with reported MAE metric
- Branch: `charliepai2d3-alphonse/l1-surface-loss`
- Hypothesis: switching the surface loss from MSE to L1 (volume MSE
  unchanged) aligns the gradient with the reported `val_avg/mae_surf_p`
  metric and is more robust to the heavy-tailed high-Re samples.
- Config: `bs=4`, `lr=5e-4`, all other knobs at defaults; only loss changed.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best |
|------:|-------------------:|:----:|
| 1  | 244.06 | * |
| 4  | 198.10 | * |
| 8  | 131.46 | * |
| 11 | 113.55 | * |
| 13 | 105.84 | * |
| 14 | **102.64** | * |

Best epoch 14 (the final epoch); curve was still descending at termination.
Stopped at epoch 14 by the 30-min timeout. Full per-epoch metrics committed
at `models/model-charliepai2d3-alphonse-l1-surface-loss-20260427-223604/`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint, epoch 14)

| split | val mae_surf_p | test mae_surf_p (NaN-safe) |
|-------|---------------:|---------------------------:|
| single_in_dist     | 121.18 | 109.80 |
| geom_camber_rc     | 125.01 | 114.60 |
| geom_camber_cruise |  73.22 |  79.92 |
| re_rand            |  91.14 |  86.58 |
| **avg**            | **102.64** | **97.73** |

### Analysis

- **Wins on all four val splits vs the prior baseline (PR #306, val 135.20).**
  Biggest improvement on the hardest split, `val_single_in_dist`: 121.18 vs
  190.14 (−36%). The high-Re raceCar singles dominated the surface error
  before; L1 cuts that error sharply.
- −24.1% on `val_avg/mae_surf_p`, −20.6% on `test_avg/mae_surf_p`. Test < val
  on three of four splits → no overfit.
- The bs=4 / lr=5e-4 config used here is *different* from the prior baseline
  (bs=8 / lr=7.07e-4 / MSE). So the headline 24% win conflates L1 vs MSE
  with bs=4 vs bs=8. Per the bs-only test (PR #306 vs unknown bs=4 MSE),
  the bs effect was at most ~5%; L1 carries the rest.
- Peak memory only 42.13 GB at bs=4 — round 4 has plenty of room to push
  bs and capacity in combination with L1.

### Decision

**Merged** as the new round 3 baseline. Old baseline (PR #306, val 135.20)
becomes round-3 reference 1. New baseline `val_avg/mae_surf_p = 102.64`,
`test_avg/mae_surf_p = 97.73`. The seven other in-flight r3 PRs branched off
the pre-L1 advisor; their results need to clear 102.64 (val) to be winners.
Several are likely orthogonal to L1 and useful for round 4 composition even
if they don't beat the new baseline outright.

---

## 2026-04-27 23:26 — PR #306: Batch size 8 with sqrt LR scaling (lr=7.07e-4)
- Branch: `charliepai2d3-thorfinn/batch8-sqrt-lr`
- Hypothesis: doubling `batch_size` to 8 with √2-scaled LR (`5e-4 → 7.07e-4`)
  reduces gradient noise without changing the data budget; tests whether
  gradient quality alone improves convergence within the 30-min wallclock.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best | sec | peak_mem |
|------:|-------------------:|:----:|----:|---------:|
| 1  | 264.88 | * | 134.5 | 84.2 GB |
| 2  | 215.25 | * | 129.2 | 84.2 GB |
| 5  | 212.78 | * | 130.0 | 84.2 GB |
| 7  | 188.54 | * | 129.9 | 84.2 GB |
| 8  | 155.47 | * | 128.8 | 84.2 GB |
| 11 | 142.97 | * | 129.2 | 84.2 GB |
| 13 | **135.20** | * | 129.7 | 84.2 GB |
| 14 | 142.03 |   | 127.1 | 84.2 GB |

Best epoch 13/14. Stopped at epoch 14 by 30-min timeout (cosine T_max was
50 → never reached the tail). Full per-epoch metrics committed at
`models/model-charliepai2d3-thorfinn-batch8_sqrt_lr-20260427-223454/metrics.jsonl`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint)

| split | val mae_surf_p | test mae_surf_p (corrected) |
|-------|---------------:|----------------------------:|
| single_in_dist     | 190.14 | 173.01 |
| geom_camber_rc     | 138.39 | 120.22 |
| geom_camber_cruise |  97.95 |  82.83 |
| re_rand            | 114.32 | 116.53 |
| **avg**            | **135.20** | **123.15** |

### Analysis

- Run is stable; bs=8 + lr=7.07e-4 fits comfortably (peak 84.2 GB of 96 GB).
- The val curve was still descending at termination (epoch 13 = 135.20 vs
  epoch 11 = 142.97), so this is a **partially-trained model on a truncated
  cosine** — not a converged result.
- Test < val on three of four splits (single, rc, cruise) → no overfit.
- Cruise track (97.95 / 82.83) is by far the easiest; single-in-dist is the
  hardest (190.14 / 173.01) — high-Re raceCar singles dominate the surface
  error.
- **Critical infrastructure bug found and fixed:** `data/scoring.py` had an
  `Inf*0=NaN` reduction bug on the test path (a single sample in
  `test_geom_camber_cruise` has 761 Inf values in its pressure GT). Fix
  applied as advisor commit `2eb5c7f` (attribution to thorfinn).

### Decision

**Merged** as the round 3 measured baseline. No prior r3 baseline existed,
so this becomes the reference for the seven other in-flight r3 PRs. Per-PR
follow-ups for round 4: bs=12 (√3 LR scaling) if "larger batch" wins;
`--epochs ≤ 14` to actually decay cosine inside the wallclock cap;
activation checkpointing for bs=16+. These are recorded in
`research/CURRENT_RESEARCH_STATE.md`.
