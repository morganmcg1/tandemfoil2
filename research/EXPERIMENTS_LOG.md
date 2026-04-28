# SENPAI Research Results — willow-pai2e-r4

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
