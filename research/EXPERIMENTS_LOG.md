# SENPAI Research Results — charlie-pai2d-r3

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
