# SENPAI Research Results — charlie-pai2d-r5

## 2026-04-28 04:40 — PR #364 (rerun #2): Huber smooth_l1 β=0.5 on full stack — **CLOSE (loss-refinement axis exhausted)**

- Branch: `charliepai2d5-edward/huber-loss` (closed)

### Results

| metric | value | vs current baseline (73.91 / 70.37) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 78.20 | **+5.81%** (worse) |
| `val_single_in_dist/mae_surf_p` | 86.33 | +5.72% |
| `val_geom_camber_rc/mae_surf_p` | 89.64 | +1.92% |
| `val_geom_camber_cruise/mae_surf_p` | 60.51 | **+11.09%** |
| `val_re_rand/mae_surf_p` | 76.33 | +6.69% |
| `test_avg/mae_surf_p` (3 clean) | 74.52 | +5.90% |

**Every clean split regresses** — both val and test, in-dist and OOD.

### Decision

Close. Hits close criterion (val ≥ 76.5). Student's analysis: the gain pattern from β=1.0 didn't survive the rebase. With grad_clip_norm=0.5 already halving per-step update magnitude in the always-clipped regime AND surf_weight=30 tripling the surface gradient signal, the late-training gradient landscape is *already* aggressively smoothed. Huber's quadratic-near-zero behavior just dampens the late-training fine-tuning that L1's constant `sign(err)` updates were getting.

Advisor pre-flagged exactly this branch in the send-back: "If you instead see continued regression on cruise/re_rand at beta=0.5: the issue isn't beta calibration — likely something deeper like per-split residual scale heterogeneity." That's the branch we're in.

**Loss-function refinement on the full stack is now well-explored** — L1 won big, Huber-β=1.0 marginal on partial stack, Huber-β=0.5 lost on full stack. Combined with three saturated regularization-style hypotheses (wd #385, EMA #303, attn-dropout #471), the diminishing-returns regime is firmly characterized.

Reassigned edward to **SiLU activation replacing GELU** in MLP blocks (PR #538) — single-line model_config change, untested architecture axis, well-established modern transformer alternative.



## 2026-04-28 04:30 — PR #498: Larger physical batch size (4 → 8) — **CLOSE (under-trained at fixed budget)**

- Branch: `charliepai2d5-thorfinn/batch-size-8` (closed)

### Results

| metric | value | vs current baseline (73.91 / 70.37) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 84.09 | **+13.8%** (worse) |
| `val_single_in_dist/mae_surf_p` | 109.45 | **+34.0%** (single_in_dist is the dominant regressor) |
| `test_avg/mae_surf_p` (3 clean) | 82.49 | +17.2% (worse) |
| Pre-clip ‖∇‖ trajectory | peak 247, end 59.6 | matches bs=4 baseline 270 / 60–63 — variance reduction marginal |
| Peak GPU memory | 84.7 GB | vs 42 GB at bs=4 — predicted ~80 GB ✓ |

### Decision

Close. Hits close criterion. Student's analysis nails it: bs=8 halves optimizer step count per epoch (~188 vs ~375), and the cosine schedule keyed to `epochs` traverses the same LR trajectory in half the gradient updates. Same val curve shape as bs=4, just lagging by ~2 epochs of effective progress. The +34% `val_single_in_dist` regression is the smoking gun for under-training — that split has the most room to improve in the late-epoch fine-tuning phase that bs=4 reaches.

The pre-clip ‖∇‖ being essentially unchanged confirms the optimizer is variance-insensitive in the clipped, direction-dominated regime — doubling batch barely moves the per-step signal. So the variance-reduction benefit doesn't pay back the halved-step-count cost.

Reassigned thorfinn to **RMSNorm replacing LayerNorm** in TransolverBlock (PR #530) — modern transformer normalization (LLaMA, T5 standard), single-axis architecture change, complementary to all the loss/optimizer/feature axes already explored.

### Note on the bs axis going forward

If anyone wants to revisit this axis, the natural next test is **bs=8 + lr=√2 × 1e-3 ≈ 1.4e-3** (LR-scaling rule for variance compensation). But it's a 2-axis change so not in scope without a separate PR. Skipping for now — 7+ pending PRs on cheaper axes are the priority.



## 2026-04-28 04:15 — PR #471: Attention dropout 0.05 in PhysicsAttention — **CLOSE (3rd regularization saturation)**

- Branch: `charliepai2d5-nezuko/attn-dropout-0p05` (closed)

### Results

| metric | value | vs current baseline (73.91 / 70.37) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 75.44 | **+2.06%** (worse) |
| `test_avg/mae_surf_p` (3 clean) | 71.57 | +1.7% (worse) |
| Train-vs-val L1 gap (surf, vol) | +85% / +182% wider than baseline | artifact: train computed with dropout active is depressed |

### Decision

Close. Hits close criterion (val ≥ 75.0). Third regularization-style hypothesis to saturate on the full stack: **wd=5e-4 (#385), EMA (#303), and now attn-dropout** all worked on simpler stacks but neutral-to-bad once L1+warmup+Fourier+sw=30+grad-clip is in place. Student's analysis is consistent with the pattern: model isn't severely overfitting (~10–13% relative train-val gap), so adding more regularization noise hurts deterministic eval without unlocking generalization.

The "gap widening with dropout" is itself a useful diagnostic: train loss measured *with* dropout active is artificially depressed (signal noise lowers reported train loss); this isn't real overfitting reduction.

Reassigned nezuko to **TTA via random node masking (K=5 passes, drop=10%)** (PR #521) — eval-only intervention that doesn't compete with regularization budget. The model is permutation-invariant by design; averaging predictions across K masked-input passes reduces eval-time variance without affecting training. Different axis from the saturated regularizer family.



## 2026-04-28 03:35 — PR #470: Trainable random Fourier (Tancik 2020) σ=10 — **CLOSE (Fourier axis exhausted)**

- Branch: `charliepai2d5-thorfinn/fourier-trainable-tancik` (closed)

### Results

| metric | value | vs current baseline (73.91 / 70.37) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 84.22 | **+13.9%** (worse) |
| `test_avg/mae_surf_p` (3 clean) | 81.38 | **+15.7%** (worse) |
| Pre-clip ‖∇‖ trajectory | peak 275.75, end 61.84 | matches dyadic baseline 270 / 63 |
| Per-split ratio (this/baseline) | 1.11–1.17 | uniform across splits |

### Decision

Close. Hits the close criterion (val_avg ≥ 75.0). Student's analysis: σ=10 spreads weight uniformly to high frequencies (RMS frequency ~88), but most signal in this CFD problem lives in the low-frequency band where the dyadic grid `{1, 2, 4, 8}π` already concentrates resolution. Same local minimum, just slower path through it — gradient dynamics identical (same start, same end-‖∇‖, uniform per-split ratio). Not pathological dynamics, just under-converged at iso-epoch=14.

**Bigger signal:** the Fourier axis is now thoroughly explored (PR #365 won big at +12% test, PR #414 marginal, PR #470 lost). The dyadic prior was the load-bearing piece; refinements aren't compounding.

Reassigned thorfinn to **`batch_size` 4 → 8** (PR #498) — different SGD-dynamics axis we haven't tested. Larger batches give less noisy gradient direction estimates; with grad-clip in pure direction-only mode, this might compound. Memory predicted ~80 GB (vs 42 at bs=4), fits in 96 GB.



## 2026-04-28 03:30 — PR #464: Tighten gradient clipping `max_norm` 1.0 → 0.5 — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-alphonse/grad-clip-0p5` (CLI-flag-only, empty diff)

### Results

| metric | value | vs PR #387 (74.44 / 72.14) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **73.91** | **−0.71%** |
| `val_single_in_dist/mae_surf_p` | 81.66 | **−5.79%** |
| `val_geom_camber_rc/mae_surf_p` | 87.95 | +2.37% (regressed) |
| `val_geom_camber_cruise/mae_surf_p` | 54.47 | +2.22% (regressed) |
| `val_re_rand/mae_surf_p` | 71.55 | −0.46% (flat) |
| `test_avg/mae_surf_p` (3 clean) | **70.37** | **−2.45%** |
| Pre-clip ‖∇‖ trajectory | peak 269.92 / end 60.61 | matches PR #387's 270.3 / 63.0 |

### Decision

Merge — sixth orthogonal axis (well, refinement of the fifth). val gain is small but lands in the ambiguous band per the decision criteria; test gain (−2.45%) is well past the 1.3% merge threshold and on the more reliable signal (200 vs 100 samples per split). Per-split shows clean redistribution: in-dist gain dominates, camber-OOD regressions small. Config default `grad_clip_norm: 1.0 → 0.5` updated on advisor.

Per-split val regressions on camber-OOD (+2.2-2.4%) are worth tracking on subsequent merges. If they compound, may need to bracket back.

### Diagnostic

Pre-clip norms identical to PR #387 (peak 270, end 60–63), confirming clipping is a per-step magnitude bound, not a gradient computation change. Optimizer sees identical gradients; only applied update magnitudes differ.

Reassigned alphonse to **bf16 mixed precision (autocast) + epochs 14→24** (PR #496) — directly attacks the binding wall-clock constraint by speeding up each step ~1.5–2×, allowing more epochs in the same 30-min budget. Val curve still descending at epoch 14 across every winning PR — more epochs almost certainly help.

---

## 2026-04-28 03:30 — PR #385 (rerun #2 on full stack): wd=5e-4 — **CLOSE (saturation on full stack)**

- Branch: `charliepai2d5-fern/weight-decay-5e-4` (closed)

### Results

| metric | value | vs current baseline (74.44 / 72.14) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 75.94 | **+2.0%** (worse) |
| `test_avg/mae_surf_p` (3 clean) | 73.59 | **+2.0%** (worse) |
| Pre-clip ‖∇‖ at ep 14 | 57.1 | vs alphonse's 63.0 → only −10% reduction |

### Decision

Close. Hypothesis cleanly falsified on the full stack — and **cleanly characterized**:
- WD=5e-4 helped −7.7% on L1+warmup (no Fourier, no clip)
- WD=5e-4 helped −12.0% on L1+warmup+Fourier (no clip)
- WD=5e-4 hurts +2.0% on L1+warmup+Fourier+sw=30+grad-clip (full stack)

Decay-of-returns curve unambiguous: each new orthogonal regularizer absorbs some of the same "training noise" budget WD was after; once 5 axes stack, no budget left for WD to claim. Combined with tanjiro's EMA falsification on the same stack (PR #303 closed earlier), we now have **two independent regularization-style hypotheses** that worked on simpler stacks but saturated on the full stack.

Reassigned fern to **mesh node random subsampling** (PR #497) — different *kind* of regularizer (data augmentation, not weight/optimizer) that may have non-overlapping budget with the current stack. Permutation-invariant model handles random node-loss subsampling cleanly.

Train-vs-val gap was −12.8% relative (small absolute) — the model is *not* severely overfitting, which is itself important context for understanding why heavy regularization saturates on the full stack.



## 2026-04-28 03:00 — PR #303 (rerun): EMA weights (decay=0.999) — **CLOSE (falsified on full stack)**

- Branch: `charliepai2d5-tanjiro/ema-weights` (closed)

### Results

| metric | value | vs current baseline (74.44 / 72.14) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14, EMA model) | 82.44 | **+10.7%** (worse) |
| `test_avg/mae_surf_p` (3 clean, EMA) | 80.28 | +9.4% (worse) |
| Live model val_avg at epoch 14 | 76.38 | +2.6% (essentially current baseline) |
| EMA-vs-live diagnostic | epoch 5: live wins by 38.8%; ep 10: EMA wins by 6.3%; ep 14: live re-overtakes by 7.9% | mid-training crossover real but reverses by ep 14 |

### Decision

Close. Cleanly falsified on the full-stack baseline.

Student's analysis nails the mechanism: with budget-matched cosine (T_max=9 over 14 epochs total), LR decays to ~3% of peak by epoch 14 — so by then the live model has effectively converged with no oscillations to smooth, and EMA's stale-weight component (averaging in higher-LR weights from epochs 7–10) actively drags it backward. The mid-training crossover at epoch 10 (EMA wins by 6.3%, exactly matching round-1's prediction band) is real and reproducible, but the budget-matched schedule lets the live model train *past* the crossover into a regime where EMA hurts.

This is a regime-specific negative result. EMA worked on the round-1 MSE+T_max=50 setup (LR at ~92% of initial when training stopped → live still oscillating → EMA helped). With the current schedule, live converges cleanly → EMA's averaging hurts.

**Systemic finding worth noting:** the more we tighten the schedule and stack regularizers, the less most regularization-style hypotheses help. Diminishing returns are real; the path forward likely involves either bigger architectural changes or attacking specific known failure modes (e.g. `val_geom_camber_rc`'s geometry-extrapolation pattern).

Reassigned tanjiro to **higher peak LR (1e-3 → 2e-3)** with the existing grad-clip-1.0 (PR #473) — clipping protects against instability that would otherwise kill 2e-3, doubles per-step magnitude in clipped regions. Single-axis CLI flag change.

---

## 2026-04-28 02:50 — PR #444: Surface-p extra boost (3×) — **CLOSE (hypothesis falsified)**

- Branch: `charliepai2d5-nezuko/surf-p-extra-3` (closed)

### Results

| metric | value | vs sw=30 baseline (76.68 / 73.40) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 77.92 | **+1.62%** (worse) |
| `test_avg/mae_surf_p` (3 clean) | 76.41 | **+4.10%** (worse) |
| `val_avg/mae_vol_p` | 132.38 | **+26.8%** (worse) |
| `val_*/mae_surf_Ux` | — | **+11–29%** worse (all 4 splits) |
| `val_*/mae_surf_Uy` | — | **+11–18%** worse (all 4 splits) |

### Decision

Close. Cleanly falsified. Student's analysis nails the mechanism: even without explicit per-channel starvation in the loss formulation (Ux/Uy stayed at 1× weighting), raising the *total* surface loss magnitude (effective ratio 30:1 → ~50:1) pulled the shared backbone capacity onto the surface-p subspace at the expense of every other output. So "no per-channel starvation" is necessary but not sufficient — gradient *budget* is finite even when individual channel weights stay at 1×.

Combined with alphonse's earlier `surf_p_weight=5` falsification on L1 baseline (PR #278, closed), we now have two independent data points showing **targeting pressure-channel emphasis at the loss level fails on shared-backbone models**. Future work in this space would need a separate-decoder-head architecture, which is a much bigger swing.

Reassigned nezuko to **attention dropout 0.05** in `PhysicsAttention` (PR #471) — cheap orthogonal regularizer.

---

## 2026-04-28 02:50 — PR #414 (rerun): Fourier on dsdf channels — **CLOSE (marginal benefit, doesn't beat current baseline)**

- Branch: `charliepai2d5-thorfinn/fourier-on-dsdf` (closed)

### Results (rebased onto sw=30 baseline)

| metric | value | vs sw=30 baseline (76.68 / 73.40) | vs current baseline (74.44 / 72.14) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 76.17 | −0.66% | **+2.32%** (worse) |
| `val_avg/mae_surf_p` (iso-epoch 12) | 83.75 | −3.47% | — |
| `test_avg/mae_surf_p` (3 clean) | 72.78 | −0.84% | +0.89% |
| Median per-epoch wall (s) | 133.76 | +1.5% (overhead) | — |

### Decision

Close. Marginal benefit on the rebased sw=30 baseline (−0.66% val) but doesn't beat the current advisor baseline of 74.44 (PR #387 alphonse grad-clip merged during the run). With +1.5% wall overhead and the iso-epoch advantage washing out by epoch 14 (cosine tail lets the no-dsdf baseline catch up), the cost-benefit doesn't justify a merge in the diminishing-returns regime.

Student's iso-epoch analysis (−3.47% at epoch 12 collapsing to −0.66% at epoch 14) was the key diagnostic — protected the research direction from over-claiming on what was partly a schedule artifact.

Reassigned thorfinn to **trainable random Fourier projection (Tancik 2020)** (PR #470) — same spectral-bias-relief mechanism with learned-random Gaussian frequency basis instead of fixed dyadic.

---

## 2026-04-28 02:35 — PR #387 (rerun): Gradient clipping `max_norm=1.0` on full stack — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-alphonse/grad-clip-1` (rebased onto L1+warmup+Fourier+sw=30)

### Results

| metric | value | vs PR #301 (76.68 / 73.40) | vs PR #365 (87.86 / 84.22) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **74.44** | **−2.92%** | −15.3% |
| `val_single_in_dist/mae_surf_p` | 86.68 | −1.04% | — |
| `val_geom_camber_rc/mae_surf_p` | 85.92 | −2.53% | — |
| `val_geom_camber_cruise/mae_surf_p` | 53.29 | −4.34% | — |
| `val_re_rand/mae_surf_p` | 71.88 | −4.49% | — |
| `test_avg/mae_surf_p` (3 clean) | **72.14** | **−1.71%** | −14.3% |
| Median per-epoch wall (s) | 131 | unchanged | — |

All four val splits improved. Strict monotone val_avg descent across all 14 epochs (no oscillations).

### Decision

Merge — fifth orthogonal axis. Stack now: L1 × warmup → cosine × Fourier × sw=30 × grad-clip-1.0.

### Critical diagnostic (alphonse's gradient-norm telemetry)

Pre-clip ‖∇‖ went from peak 105.3 / end 25.2 (pre-Fourier) to peak **270.3** / end **63.0** (post-Fourier). Fourier features ~2.5× the gradient signal — **clipping is doing more work, not less, post-Fourier**. The clipping ratio is 63–270 : 1 throughout, well into pure direction-only mode. This motivates `grad_clip_norm=0.5` as the natural follow-up.

### Why the gain is smaller than the pre-Fourier delta

The pre-Fourier rerun (PR #387 first attempt) gave −13.5% val on the L1+warmup baseline. The rebased stacked version gives only −2.92% on the new sw=30 baseline. Two factors:
1. **Baseline already moved** — sw=30 absorbed some of the headroom that grad-clip would otherwise have captured.
2. **Partial overlap between regularizers** — Fourier improves input conditioning, which reduces some "bad-step" gradients that clipping used to fix. They share the trajectory-smoothing mechanism but Fourier also fixes input-side issues.

The gradient-norm telemetry confirms clipping isn't redundant — it's still doing more work than before — but the ceiling is lower because Fourier already absorbed some noise.

Reassigned alphonse to `grad_clip_norm=0.5` (PR #464) — natural follow-up motivated by the gradient-norm telemetry.



## 2026-04-28 02:25 — PR #364: Huber loss (smooth_l1, beta=1.0) — **REQUEST CHANGES (rebase + refined to beta=0.5)**

- Branch: `charliepai2d5-edward/huber-loss` (on L1+warmup+pos-Fourier, pre-sw=30)

### Results

| metric | value | vs PR #365 baseline (87.86 / 84.22) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **85.58** | **−2.6%** |
| `val_single_in_dist/mae_surf_p` | 97.98 | −6.3% |
| `val_geom_camber_rc/mae_surf_p` | 94.65 | **−9.4%** |
| `val_geom_camber_cruise/mae_surf_p` | 65.49 | **+4.3%** (regressed) |
| `val_re_rand/mae_surf_p` | 84.20 | **+5.7%** (regressed) |
| `test_avg/mae_surf_p` (3 clean) | 83.03 | −1.4% |

Median per-epoch wall: 131.8s (essentially free). Per-split asymmetry: large gains on high-magnitude splits (raceCar single, raceCar tandem); small regressions on lower-magnitude splits (cruise, re_rand).

### Decision

Send back, refined to beta=0.5.

Edward's own follow-up #1 nailed the issue: **beta=1.0 is calibrated for pixel-units bounding boxes, not unit-variance normalized targets**. With targets normalized to ~unit-variance, residuals at convergence are <<1σ, so Huber operates in MSE-mode for the bulk of late training — exactly the opposite of what we want for an MAE-eval metric. The split-level asymmetry confirms this: lower-magnitude splits (cruise camber holdout, Re-rand) spend more training in the quadratic-near-zero regime and regress; higher-magnitude splits where more of the training signal lives in the L1 tail benefit.

Sending back with the refined hypothesis: rerun with **beta=0.5** on the current advisor (post-sw=30 baseline). If beta calibration argument holds, beta=0.5 fixes the cruise/re_rand regression without losing the wins on the high-magnitude splits.

Decision criteria:
- val_avg ≤ 75.0 (≥ 2.2% improvement vs 76.68): **merge**.
- 75.0 < val_avg < 76.5: ambiguous — try beta=0.25 next.
- val_avg ≥ 76.5: **close** — Huber doesn't compose with sw=30.

Edward's analysis was the most useful diagnostic of the round; the path forward is clearer because of it.

---

## 2026-04-28 02:10 — PR #414: Fourier on dsdf channels (4 freqs, dims 2–11) — **REQUEST CHANGES (rebase + iso-epoch concern)**

- Branch: `charliepai2d5-thorfinn/fourier-on-dsdf` (on L1+warmup+pos-Fourier, pre-sw=30)

### Results

| metric | value | vs PR #365 baseline (87.86 / 84.22) | iso-epoch (12 vs 12) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | 76.82 | −12.6% | **−1.8%** at epoch 12 |
| `val_geom_camber_rc/mae_surf_p` | 89.03 | −14.8% | — |
| `val_re_rand/mae_surf_p` | 72.63 | −8.8% | — |
| `test_avg/mae_surf_p` (3 clean) | 74.14 | −12.0% | — |
| Median per-epoch wall (s) | 134.1 | +1.0% | — |

### Decision

Send back. Student's honest at-iso-epoch analysis is the key signal: most of the headline −12.6% is from completing 2 more epochs that the baseline couldn't fit (the baseline run had GPU contention on epochs 7–8). The pure dsdf-Fourier benefit is at most ~2%, with +1% wall overhead. Combined with the rebase mechanic (branch reverts sw=30 if squash-merged), this needs a clean rerun on the post-#301 baseline.

Decision criteria for the rerun communicated to thorfinn:
- val_avg ≤ 75.0 (≥ 2.2% improvement over current 76.68): **merge**.
- 75.0 < val_avg < 76.5: **need seed-stability cross-check**.
- val_avg ≥ 76.5: **close** — dsdf-Fourier doesn't compose meaningfully with sw=30.

Excellent diagnostic discipline from thorfinn — calling out the iso-epoch effect prevented over-claiming on a partly-artifactual headline number.

---

## 2026-04-28 02:00 — PR #301 (rerun): surf_weight 10 → 30 on L1+warmup+Fourier — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-nezuko/surf-weight-30` (rebased onto L1+warmup+Fourier)

### Results

| metric | value | vs PR #365 baseline (87.86 / 84.22) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **76.68** | **−12.7%** ✓ |
| `val_single_in_dist/mae_surf_p` | 87.59 | −16.2% |
| `val_geom_camber_rc/mae_surf_p` | 88.15 | −15.6% |
| `val_geom_camber_cruise/mae_surf_p` | 55.71 | −11.3% |
| `val_re_rand/mae_surf_p` | 75.26 | −5.5% |
| `test_avg/mae_surf_p` (3 clean) | **73.40** | **−12.9%** ✓ |
| `val_avg/mae_vol_p` | 104.43 | **+13.2%** (regressed — tradeoff) |

### Decision

Merge — fourth orthogonal axis. Pure CLI flag → applied as Config default update on advisor (`surf_weight: float = 30.0`). The volume-pressure regression is a real tradeoff that's not ranked but worth tracking.

Reassigned nezuko to **`surf_p_extra=3.0`** (PR #444) — additive boost on the surface-p channel only, leaving surface Ux/Uy gradients untouched. Designed to extract more pressure focus while reducing the volume regression. Per nezuko's own follow-up #2, but with a non-normalized formulation (avoids alphonse's earlier failure mode of starving Ux/Uy).

---

## 2026-04-28 01:55 — PR #385 (rerun #1, on Fourier): wd=5e-4 — **REQUEST CHANGES (sent back; nezuko merged ahead)**

- Branch: `charliepai2d5-fern/weight-decay-5e-4` (on L1+warmup+Fourier)

### Results

| metric | value | vs PR #365 baseline (87.86 / 84.22) | vs new baseline post-#301 (76.68 / 73.40) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **77.29** | **−12.0%** | +0.8% (very slightly worse) |
| `test_avg/mae_surf_p` (3 clean) | **74.60** | **−11.4%** | +1.6% (slightly worse) |

Train-vs-val gap widened from −0.185 (no Fourier) to −0.330 (with Fourier) — Fourier features gave the model more capacity that the same WD has to discipline harder. Suggests room for stronger regularization.

Per-split gain pattern *flipped* with Fourier: original WD=5e-4 run had largest gain on `val_single_in_dist` (in-dist); rebased run had largest gain on `val_geom_camber_rc` (OOD camber holdout, −14.6%). Reading: with Fourier features encoding finer positional structure, the WD penalty disproportionately helps the OOD camber holdouts, which is where the original "WD targets OOD generalization" framing predicted gains.

### Decision

Send back. Result is excellent (−12% / −11.4% vs the Fourier baseline, comparable to nezuko's surf_weight=30 win on a different axis), but nezuko's PR #301 merged ahead. Fern's branch is now on a stale base — needs to rebase onto the new advisor (which has surf_weight=30 default) and rerun with `--weight_decay 5e-4 --epochs 14` to give us the **stacked sw=30 + wd=5e-4** measurement. The two axes are mechanically orthogonal (parameter-magnitude regularization vs surface-volume balance), so stacking should still help.

---

## 2026-04-28 01:25 — PR #387: Gradient clipping `max_norm=1.0` — **REQUEST CHANGES (rebase mechanic, but standout result)**

- Branch: `charliepai2d5-alphonse/grad-clip-1` (on L1+warmup, pre-Fourier)

### Results (on L1+warmup baseline — pre-Fourier)

| metric | value | vs PR #296 (94.54 / 91.85) | vs current baseline PR #365 (87.86 / 84.22) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **81.81** | **−13.5%** | **−6.9%** |
| `val_single_in_dist/mae_surf_p` | 93.27 | −18.4% | — |
| `val_geom_camber_rc/mae_surf_p` | 94.90 | −10.0% | — |
| `val_geom_camber_cruise/mae_surf_p` | 62.92 | −10.7% | — |
| `val_re_rand/mae_surf_p` | 76.18 | −13.4% | — |
| `test_avg/mae_surf_p` (3 clean) | **78.44** | **−14.6%** | **−6.9%** |

### Decision

Send back for rebase mechanic — squash-merging now would revert PR #365's Fourier features. The result is **the largest single-PR delta on this advisor track so far**, and crucially the stacked Fourier × clipping result is expected to give a substantial new best.

### Key diagnostic — generalizes to all future PRs

Pre-clip gradient norms (alphonse's instrumentation): epoch 1 = 69.2, peak at epoch 2 = 105.3 (warmup top), then monotone decay to 25.2 at epoch 14. Mean ≈ 50.8, max_norm = 1.0 — clipping is active **every step**, with scaling factors of 1/25 to 1/100. This explains the magnitude:

> Under L1 loss specifically, gradient magnitudes don't naturally decay with residuals — they stay sign-magnitude bounded — so the cosine-decayed LR alone isn't enough to control step sizes. Clipping is doing fundamental optimization work, not just stability.

This finding generalizes — every PR on this branch is on L1 loss, so clipping should help universally. It's a candidate for inclusion as a defaults-level change in a future merged PR.

---

## 2026-04-28 01:10 — PR #385: weight_decay 1e-4 → 5e-4 — **REQUEST CHANGES (rebase mechanic)**

- Branch: `charliepai2d5-fern/weight-decay-5e-4` (on L1+warmup, pre-Fourier)

### Results (on L1+warmup baseline — pre-Fourier)

| metric | value | vs PR #296 baseline (94.54 / 91.85) | vs current baseline (87.86 / 84.22) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (best ep 14/14) | **87.27** | **−7.7%** | −0.7% (essentially tied) |
| `val_single_in_dist/mae_surf_p` | 103.56 | −9.4% | — |
| `val_geom_camber_rc/mae_surf_p` | 98.71 | −6.4% | — |
| `val_geom_camber_cruise/mae_surf_p` | 65.96 | −6.4% | — |
| `val_re_rand/mae_surf_p` | 80.84 | −8.1% | — |
| `test_avg/mae_surf_p` (3 clean) | **83.68** | **−8.9%** | −0.6% (essentially tied) |

Per-epoch wall unchanged from baseline (~132s). Train-vs-val L1 gap small at final epoch (−0.185). Best epoch landed at the very end of cosine decay (14/14).

### Decision

Send back for rebase mechanic. Squash-merging now would revert PR #365's Fourier features; current branch is comparable to the Fourier baseline through a different mechanism (regularization vs feature augmentation), but stacking them is the obvious next experiment. Pure CLI flag tweak post-rebase.

Notable directional finding from the run: WD gain was broad-based and **largest on `val_single_in_dist`** (the in-distribution split), not the OOD camber holdouts as the hypothesis predicted. Updates us toward "WD=1e-4 was simply too low globally" rather than "WD targets OOD specifically." Generalizes cleanly to the rebased baseline.

---

## 2026-04-28 01:00 — PR #365 (rerun): Fourier features (8 freqs, normalized x,z) — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-thorfinn/fourier-features` (rebased onto L1+warmup post-PR-#296 merge)
- Hypothesis: 8-band Fourier positional encoding relaxes MLP spectral bias on raw `(x, z)` coordinates and improves surface-pressure fidelity.

### Results (on L1+warmup baseline, post-rebase)

| metric | value | vs PR #296 baseline (94.54 / 91.85) |
|---|---:|---|
| `val_avg/mae_surf_p` (best ep 12/14) | **87.86** | **−7.1%** ✓ |
| `val_single_in_dist/mae_surf_p` | 104.53 | −8.5% |
| `val_geom_camber_rc/mae_surf_p` | 104.44 | −1.0% (anomaly — see below) |
| `val_geom_camber_cruise/mae_surf_p` | 62.81 | −10.8% |
| `val_re_rand/mae_surf_p` | 79.64 | −9.5% |
| `test_avg/mae_surf_p` (3 clean) | **84.22** | **−8.3%** ✓ |
| Median per-epoch wall (s) | 132 | unchanged |

### Decision

Merge — third orthogonal axis stacks cleanly: L1 × warmup → cosine × Fourier features. New best on every clean split. `val_geom_camber_rc` improved least (−1.0% vs −8.5% to −10.8% on the others), suggesting that split's residual error is dominated by geometry-extrapolation, not MLP spectral bias — a useful directional finding for future hypotheses.

Reassigned thorfinn to **Fourier on dsdf channels** (PR #414) — natural follow-up that tests the same spectral-bias-relief mechanism on the geometric distance descriptors (`saf`, `dsdf` — dims 2–11).

The honest GPU-contention note for epochs 7–9 (median wall time used for cost comparison) was good rigor.



## 2026-04-28 00:55 — PR #380: Best-val checkpoint averaging (top-3) — **REQUEST CHANGES (rebase + val-on-averaged)**

- Branch: `charliepai2d5-frieren/ckpt-avg-top3` (on L1-only, not L1+warmup)

### Results (on L1 baseline — pre-warmup)

| metric | value | vs L1 baseline (101.87 / 102.61) | vs current baseline (94.54 / 91.85) |
|---|---:|---|---|
| `val_avg/mae_surf_p` (single best) | 104.43 | +2.5% (worse, run-to-run noise) | +10.5% (worse) |
| `val_avg/mae_surf_p` (averaged) | **not measured** | — | — |
| `test_avg/mae_surf_p` (3 clean) | **91.13** | **−11.2%** ✓ | −0.8% (small win) |

Top-3 averaged epochs: 12 (val=104.43), 13 (108.96), 14 (108.42). Per-epoch wall: 131.1s (unchanged from L1 baseline). Averaging adds < 1% overhead.

### Decision

Send back for:
1. **Rebase onto current advisor** (L1 + warmup + budget-matched cosine). Squash-merging now would revert PR #296's warmup scheduler — same mechanic issue as thorfinn's #365.
2. **Add val-on-averaged-model evaluation.** The current implementation only runs the averaged model on test, so we can't rank by `val_avg/mae_surf_p`. Student's own follow-up #3 — easy addition, one extra `evaluate_split` pass.

The technique works. Test improvement is real and large (−11.2% vs L1). Stacked on L1+warmup it should give a clean new test-side best. The val-on-averaged measurement closes the only methodological gap.

---

## 2026-04-28 00:20 — PR #278 (rerun): surf_p_weight=5 on top of L1 — **CLOSE (hypothesis falsified)**

- Branch: `charliepai2d5-alphonse/pressure-surface-weight` (rebased onto L1, not onto current L1+warmup)

### Results

| metric | value | vs L1 baseline (101.87) | vs current baseline (94.54) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (best ep 13/14) | **108.63** | +6.6% (worse) | +14.9% (worse) |
| `val_single_in_dist/mae_surf_p` | 134.66 | +7.5% | — |
| `val_geom_camber_rc/mae_surf_p` | 132.11 | **+22.3%** | — |
| `val_geom_camber_cruise/mae_surf_p` | 70.31 | −6.6% | — |
| `val_re_rand/mae_surf_p` | 97.44 | −1.5% | — |
| `test_avg/mae_surf_p` (3 clean) | 112.49 | +9.6% | — |

### Decision

Close. Hypothesis cleanly falsified: `surf_p_weight=5` on L1 is **+6.6% worse** than L1 baseline (past the 5% close threshold), with the dominant cost on `val_geom_camber_rc` (+22.3%). Student's analysis is excellent — under L1, gradient magnitudes are sign-based and per-element, so 5× channel weighting routes 71% of surface gradient onto `p`, starving Ux/Uy. Since the model is parameter-shared across channels, degraded velocity learning hurts the joint flow representation that pressure prediction relies on.

The same gradient-budget reasoning predicts that any `surf_p_weight > 1` under L1 trades Ux/Uy starvation for pressure emphasis with no good operating point — channel weighting and L1 don't compose well. Reassigned alphonse to **gradient clipping `max_norm=1.0`** (PR #387) — a no-cost stability hypothesis that may also reduce the test-time non-finite-prediction patterns alphonse helped diagnose.

---

## 2026-04-28 00:15 — PR #365: Fourier positional features (8 freqs, normalized x,z) — **REQUEST CHANGES (rebase mechanic only)**

- Branch: `charliepai2d5-thorfinn/fourier-features`
- Hypothesis: 8-band sinusoidal Fourier features on normalized node positions relax MLP spectral bias and improve surface-pressure fidelity.

### Results (on L1 baseline — pre-warmup; not rebased onto current advisor)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 13/14) | **89.30** |
| `val_single_in_dist/mae_surf_p` | 108.97 (-13.0% vs L1) |
| `val_geom_camber_rc/mae_surf_p` | 98.80 (-8.5%) |
| `val_geom_camber_cruise/mae_surf_p` | 67.31 (-10.6%) |
| `val_re_rand/mae_surf_p` | 82.12 (-17.0%) |
| `test_avg/mae_surf_p` (3 clean) | **88.94** (-13.3% vs L1) |
| Per-epoch wall (s) | 131.91 (vs 131.82 baseline — essentially free) |
| Peak GPU memory (GB) | 42.36 (vs 42.11 — +0.6%) |

All four val splits improved monotonically. Result substantially exceeded the predicted 2–5% delta (~12.3% achieved).

### Decision

**Send back for rebase only — the experiment was right, the merge mechanic is wrong.** Thorfinn's branch was created from L1-only (post-PR-#293 but pre-PR-#296), so squash-merging now would revert PR #296's warmup scheduler. Beats current baseline (94.54) by 5.6% even without warmup; rerun on top of L1+warmup is expected to produce a clear new best. No experiment changes — pure git mechanic.

After the rebased rerun lands, this is likely the round-2 winner.

---

## 2026-04-28 00:05 — PR #296 (rerun): Linear warmup → cosine, peak lr 1e-3, --epochs 14 — **MERGE (winner, new baseline)**

- Branch: `charliepai2d5-fern/lr-warmup-1e3` (rebased onto post-L1 advisor)
- Hypothesis: with the schedule matched to the wall-clock budget, warmup → cosine decay should let the model converge into a low-LR refinement regime that L1's plain cosine-over-50 can't reach.

### Results (on top of L1 baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 12/14) | **94.5397** |
| `val_single_in_dist/mae_surf_p` | 114.295 |
| `val_geom_camber_rc/mae_surf_p` | 105.456 |
| `val_geom_camber_cruise/mae_surf_p` | 70.448 |
| `val_re_rand/mae_surf_p` | 87.961 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **91.853** |

Best epoch landed at end-of-epoch-12, LR ≈ 2.5e-4 (mid-cosine-decay) — schedule worked exactly as designed.

### Decision

Merge. Beats the L1-only baseline by **−7.2% val** and **−10.5% test (3-clean-split)**. Two clean orthogonal axes (loss + schedule) now stacked. The `test_geom_camber_cruise/p` NaN is unchanged from the cohort-wide pre-existing data issue.

Reassigned fern to `weight_decay 1e-4 → 5e-4` (PR #385) — single-axis test of whether stronger regularization helps the OOD camber splits.

---

## 2026-04-28 00:05 — PR #303: EMA weights (decay 0.999) — **REQUEST CHANGES (rebase onto L1+warmup)**

- Branch: `charliepai2d5-tanjiro/ema-weights`
- Hypothesis: per-step EMA of model weights with decay 0.999 should improve generalization by 2–5%.

### Results (on pre-L1 MSE baseline — student honestly noted not rebased)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50, EMA model) | **127.65** |
| `test_avg/mae_surf_p` (3 clean) | **125.63** |
| **EMA vs live diagnostic** | epoch 5: live wins by 32; epoch 10: **EMA wins by 4.66%** ✓ |

The EMA-vs-live tracking confirmed the predicted 2–5% delta empirically. The hypothesis works mechanically — the issue is just that this run was on MSE not L1+warmup.

### Decision

Send back. EMA is loss/schedule-agnostic, so the 4–5% relative delta should stack on top of L1+warmup. Action: rebase onto the new advisor branch (which has L1 + warmup + `epochs=14` budget) and rerun with `--ema_decay 0.999 --lr 1e-3 --epochs 14`. Keep the every-5-epoch live-vs-EMA diagnostic — it's a great instrumentation choice we want to retain.

Independent diagnosis of the cruise NaN matches the cohort-wide finding.



## 2026-04-27 23:30 — PR #293: L1 loss in normalized space (alignment with MAE eval metric) — **MERGE (winner)**

- Branch: `charliepai2d5-edward/l1-loss`
- Hypothesis: replace MSE with L1 in normalized space; MAE-aligned with the eval metric, more robust to high-Re outliers.

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **101.868** |
| `val_single_in_dist/mae_surf_p` | 125.264 |
| `val_geom_camber_rc/mae_surf_p`  | 108.034 |
| `val_geom_camber_cruise/mae_surf_p` |  75.262 |
| `val_re_rand/mae_surf_p` | 98.912 |
| `test_avg/mae_surf_p` (4-split, with NaN) | NaN |
| `test_avg/mae_surf_p` (3 clean splits) | **102.606** |
| `test_single_in_dist/mae_surf_p` | 113.966 |
| `test_geom_camber_rc/mae_surf_p` |  99.998 |
| `test_geom_camber_cruise/mae_surf_p` | NaN (data bug) |
| `test_re_rand/mae_surf_p` | 93.854 |

Metric summary: `models/model-l1-loss-20260427-223415/metrics.yaml`

### Analysis

Pure L1 swap, no other changes. Training was numerically clean from epoch 1 (no Huber fallback needed). Validation `val_avg/mae_surf_p` descended monotonically across all 14 reached epochs (266 → 209 → 184 → 171 → 161 → 135 → 142 → 140 → 125 → 124 → 112 → 107 → 106 → 102) and was still trending down at the 30-min timeout. Edward did detective work and identified a pre-existing data + scoring bug that affects the round: `test_geom_camber_cruise` sample 20 has 761 non-finite values in the `p` channel of GT, and `data/scoring.accumulate_batch` computes `err = (pred - y).abs()` *before* masking, which lets NaN propagate into the per-channel sums. Same pattern hit fern (#296) and thorfinn (#305). Read-only constraint on `data/scoring.py` means the fix has to be flagged for the human team or solved via a sanitization pre-step in `train.py`.

### Decision

Merge — clear round-1 winner. New baseline `val_avg/mae_surf_p = 101.87`, 3-clean-split `test_avg/mae_surf_p = 102.61`. The cruise NaN is a pre-existing artifact, not L1's fault, and edward's stability investigation confirmed the model itself produces only finite predictions on that split.

---

## 2026-04-27 23:30 — PR #305: Finer attention: slice_num 64→128, n_head 4→8 — **CLOSE**

- Branch: `charliepai2d5-thorfinn/slices-heads-2x`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 8/50) | **160.676** |
| `val_single_in_dist/mae_surf_p`     | 219.613 |
| `val_geom_camber_rc/mae_surf_p`     | 179.649 |
| `val_geom_camber_cruise/mae_surf_p` | 108.617 |
| `val_re_rand/mae_surf_p` | 134.825 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **162.22** |

Metric summary: `models/model-slices-heads-2x-20260427-223358/metrics.yaml`

### Analysis

Per-epoch wall time was ~252 s vs ~131 s for edward / fern — almost exactly 2× the baseline cost. Inside the 30-min `SENPAI_TIMEOUT_MINUTES` cap this gives only 8 epochs vs 14. Worse, the test split exposed the dim_head=16 instability the PR pre-warned about: model produced non-finite predictions on at least one cruise test sample, `surf_loss=NaN` and `vol_loss=+Inf` on that split. Even granting that the model is far from converged at epoch 8, the per-epoch unit economics make this a poor fit for the current timeout regime.

### Decision

Close. The configuration is fundamentally too slow per epoch to compete with the loss-formulation winners, and the dim_head=16 fragility makes test scoring unreliable. The natural fallback (`n_hidden=192` to restore dim_head=24) overlaps with askeladd's running PR #290, so reassigning thorfinn to a non-overlapping hypothesis is the better use of the slot.

---

## 2026-04-27 23:55 — PR #299: Deeper Transolver: n_layers 5 → 8 — **CLOSE**

- Branch: `charliepai2d5-frieren/deeper-8-layers` (closed)

### Results (on pre-L1 MSE baseline; two replicate runs)

| Run | best `val_avg/mae_surf_p` | best epoch | epochs/30min | per-epoch wall |
|---|---:|---:|---:|---:|
| #1 | 146.31 | 9 | 9 | ~206s |
| #2 (headline) | **139.29** | 9 | 9 | ~206s |

Run #2 per-split val: `val_single_in_dist=169.55`, `val_geom_camber_rc=146.73`, `val_geom_camber_cruise=113.17`, `val_re_rand=127.71`. 3-clean-split test mean: 141.48. test_geom_camber_cruise NaN (same root cause as round-1 cohort).

### Decision

Close. Per-epoch wall time ~206 s (same scale as askeladd's wider-192) → only 9 of 50 epochs reached. Both replicates ~37% worse than the L1 baseline (`val_avg = 101.87`). The val curve was still descending at the cap, so this is again an under-converged snapshot — but as with the wider-192 close (#290) and the slices+heads close (#305), capacity-heavy hypotheses are *structurally* penalized in the 30-min timeout regime: they can't accumulate enough SGD steps to beat the cheaper-per-epoch baselines.

Reassigned frieren to **best-val checkpoint averaging (top-3)** (PR #380) — a no-per-epoch-cost technique that fits the budget regime and addresses the per-epoch noise we saw in their training trajectory.

Worth noting: frieren's run-to-run variance (146.31 → 139.29 from two replicates with the same config) is a useful data point. Single-run round-1 numbers should be treated as having ~5% inherent noise, not as point estimates.

---

## 2026-04-27 23:35 — PR #301: Bump surf_weight 10 to 30 — **REQUEST CHANGES (rebase onto L1)**

- Branch: `charliepai2d5-nezuko/surf-weight-30`
- Hypothesis: push the surface/volume balance harder onto surface fidelity to align with the surface-only eval metric.

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **141.556** |
| `val_single_in_dist/mae_surf_p` | 156.905 |
| `val_geom_camber_rc/mae_surf_p` | 148.448 |
| `val_geom_camber_cruise/mae_surf_p` | 122.728 |
| `val_re_rand/mae_surf_p` | 138.141 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **141.27** |

### Decision

Worse than the L1 baseline of `101.87`, but the change was tested on MSE — we don't know what it does on top of L1. The hypothesis "more surface emphasis improves the surface-only metric" is plausibly orthogonal to the loss type (with L1, gradients are sign-based, so the optimal `surf_weight` may shift). Rebase onto `icml-appendix-charlie-pai2d-r5` (now has L1) and rerun with `--surf_weight 30.0`. Pure CLI flag — trivial rebase.

Excellent independent diagnosis of the cruise NaN scoring path (`err * surf_mask` propagates `NaN * 0 = NaN`); same root cause as edward's PR #293 finding.

---

## 2026-04-27 23:35 — PR #290: Wider Transolver: n_hidden 128→192, slice_num 64→96 — **CLOSE**

- Branch: `charliepai2d5-askeladd/wider-hidden-192`

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 8/9 reached) | **152.238** |
| `val_single_in_dist/mae_surf_p` | 198.823 |
| `val_geom_camber_rc/mae_surf_p` | 155.683 |
| `val_geom_camber_cruise/mae_surf_p` | 120.887 |
| `val_re_rand/mae_surf_p` | 133.559 |
| `test_avg/mae_surf_p` (3 clean) | **151.69** |

### Analysis

Per-epoch wall time was ~205 s vs ~131 s for the loss-formulation winners — the 30-min cap allowed only 9 epochs vs ~14 for the cheaper-per-epoch baselines. Best-val came at epoch 8, still descending, so this is an under-trained snapshot. Even projecting forward, the wider model is structurally penalized by the wall-clock budget: the L1 baseline reached `val_avg = 101.87` in 14 epochs at the same wall time, ~33% better than this wider 8-epoch number.

### Decision

Close. Capacity-heavy hypotheses cannot win in the current 30-min timeout regime — every minute of GPU spent on extra width is a minute not spent annealing through the cosine schedule. Reassigned askeladd to `drop-path 0.1` regularization (PR #369), which has zero per-epoch cost and is well-matched to the small-dataset regime.

Independent NaN observation matches edward / alphonse / nezuko's diagnosis of the `data/scoring.py` bug.

---

## 2026-04-27 23:35 — PR #278: Pressure-channel surface weighting (surf_p_weight=5) — **REQUEST CHANGES (rebase onto L1)**

- Branch: `charliepai2d5-alphonse/pressure-surface-weight`
- Hypothesis: up-weight the pressure channel inside the surface loss by 5× to align gradients with the eval metric.

### Results (on pre-L1 MSE baseline)

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 12/50) | **156.16** |
| `val_single_in_dist/mae_surf_p` | 195.74 |
| `val_geom_camber_rc/mae_surf_p` | 162.81 |
| `val_geom_camber_cruise/mae_surf_p` | 131.15 |
| `val_re_rand/mae_surf_p` | 134.94 |
| `test_avg/mae_surf_p` (3 clean) | **149.65** |

### Decision

Worse than L1 baseline of `101.87`, but the change was on MSE. The pressure-channel-weighting code is a per-element broadcast tensor that composes the same way regardless of whether `abs_err` comes from L1 or MSE — should rebase cleanly. Sent back: rebase onto `icml-appendix-charlie-pai2d-r5` (now has L1) and rerun.

Best independent diagnosis of the cruise NaN bug — found that `test_geom_camber_cruise` sample 20 has `-inf` in 761 volume-cell pressure GT values, scoring path: `inf * 0 = NaN` in IEEE 754. Same root-cause edward identified; alphonse's writeup pinpoints volume-cell vs surface and the exact `data/scoring.py:49–50` lines.

---

## 2026-04-27 23:30 — PR #296: Linear warmup then cosine, peak lr 1e-3 — **REQUEST CHANGES (send back)**

- Branch: `charliepai2d5-fern/lr-warmup-1e3`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **137.319** |
| `val_single_in_dist/mae_surf_p`     | 175.812 |
| `val_geom_camber_rc/mae_surf_p`     | 150.559 |
| `val_geom_camber_cruise/mae_surf_p` |  99.339 |
| `val_re_rand/mae_surf_p` | 123.565 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **136.998** |

Metric summary: `models/model-lr-warmup-1e3-20260427-223514/metrics.yaml`

### Analysis

The hypothesis is reasonable but the schedule isn't matched to the budget: `cosine T_max = MAX_EPOCHS - warmup_epochs = 45`, while only 14 epochs were ever run. So warmup occupied epochs 1–5, and epochs 6–14 ran at near-peak LR (~9.4e-4 → 8.2e-4) — effectively a "warmup + plateau at ~1e-3" run rather than the intended warmup+decay. `val_avg/mae_surf_p` was still descending at the timeout. We can't tell whether the schedule helps until cosine actually decays into the wall budget.

### Decision

Send back — set `--epochs 14` so cosine T_max scales to the actually-reachable budget and we get a clean read on the schedule. Same student branch, same hypothesis, just a one-line config tweak.
