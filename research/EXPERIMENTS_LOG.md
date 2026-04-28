# SENPAI Research Results — charlie-pai2d-r3

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
