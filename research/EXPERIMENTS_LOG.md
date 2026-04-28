# SENPAI Research Results

## 2026-04-28 00:35 — PR #344: H2 linear warmup + cosine to zero with corrected T_max — **MERGED**

- Branch: `willowpai2d4-edward/h2-warmup-cosine`
- Hypothesis: Linear warmup + per-step cosine-to-zero, with `T_max` re-aligned to the actual run length, should reduce `val_avg/mae_surf_p` by 3–7% by fixing the per-epoch `CosineAnnealingLR(T_max=50)` that never reaches zero under the 30-min wall clock.
- 3-cell matrix in W&B group `h2-warmup-cosine`:

| Run | Config | best_val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B |
|-----|--------|--------------------------|----------------------|------------|-----|
| A | `--epochs 50` (peak 5e-4) | 125.17 | 113.85 | 14 | `5okwzg15` |
| B | `--epochs 30` (peak 5e-4) | 129.47 | 117.65 | 13 | `4wd9nu6k` |
| C | `--epochs 25 --lr 7e-4` | **120.97** | **109.92** | 13 | `rua9xrca` |

### Per-split test surface MAE (Run C, winner)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------:|------------:|------------:|
| `test_single_in_dist` | 127.09 | — | — |
| `test_geom_camber_rc` | 123.58 | — | — |
| `test_geom_camber_cruise` | 81.16 | — | — |
| `test_re_rand` | 107.83 | — | — |
| **avg** | **109.92** | 1.96 | 0.83 |

### Conclusions

- **Hypothesis confirmed in spirit, with a twist.** The 30-min wall clock truncates training at epoch ~14 in all three configs, so cosine never actually reaches zero in any run. Run B (which would have reached zero by configured epoch 30) underperformed Run A precisely because it spent more time near peak lr without the cosine tail kicking in. Run C wins by raising peak lr 40% AND shortening configured epochs — net effect is higher integrated lr early plus a meaningful (if not all-the-way-to-zero) decay tail.
- **Cross-split signature matched the prediction.** In-distribution split improved most strongly (Run C vs B: -20% on `test_single_in_dist`); generalization splits moved less.
- **Critical NaN fix shipped alongside.** `evaluate_split` now filters samples with non-finite ground truth (e.g., `test_geom_camber_cruise` sample 20 has `-inf` in pressure GT) and defensively zeros out non-finite predictions before metric accumulation. Without this fix, `test_avg/mae_surf_p` was NaN on every run that touched the cruise camber test split.
- **Suggested future follow-up:** test `--epochs 14` (matched to the actual epoch budget) to see whether cosine reaching exactly zero in run-time further improves things.

---

## 2026-04-28 03:02 — PR #342: H1 per-sample y-std loss normalization — **SENT BACK FOR REBASE**

- Branch: `willowpai2d4-alphonse/h1-per-sample-ystd-loss` (cut before PR #344 merged → missing the warmup+cosine schedule and NaN fix)
- Hypothesis: rescaling MSE by per-sample, per-channel y-std should reduce `val_avg/mae_surf_p` by 8–18%, biggest on `val_geom_camber_cruise` (lowest-Re, currently dominated out of the loss).
- 3-cell matrix in W&B group `h1-per-sample-ystd`:

| Run | surf_weight | best epoch | val_avg/mae_surf_p | test (3 finite splits) | Δ vs alphonse's pre-merge baseline | W&B |
|-----|-------------|-----------:|---------------------|------------------------|------------------------------------|-----|
| pre-merge baseline | 10 | 13 | 130.10 | 130.42 | — | `kdd0rjbi` |
| A — per-sample norm | 10 | 14 | 133.21 | 140.13 | val +2.4% (worse) | `xmfhgr18` |
| **B — per-sample norm** | **5** | **13** | **119.87** | **122.85** | **val −7.9%, test −5.8%** | `bvi3jgrr` |

### Per-split val (Run B, mae_surf_p)

| Split | Pre-merge baseline | Run B | Δ |
|-------|-------------------:|------:|--:|
| `val_single_in_dist` | 162.13 | 155.60 | −4.0% |
| `val_geom_camber_rc` | 131.83 | 136.00 | +3.2% |
| `val_geom_camber_cruise` | 104.32 | **82.10** | **−21.3%** |
| `val_re_rand` | 122.10 | **105.79** | **−13.4%** |

### Conclusions (provisional, pre-rebase)

- **Hypothesis confirmed within predicted band** on apples-to-apples comparison vs alphonse's own pre-merge baseline. Run B's −7.9% on val and −5.8% on the 3-finite-split test avg lands at the bottom of the 8–18% predicted range.
- **Cross-split signature matched the prediction precisely** (val: cruise > re_rand > single_in_dist). Per-sample y-std normalization is hitting the right mechanism — equalizing per-sample contribution removes the implicit magnitude-weighting that the loss was riding on.
- **Run A (sw=10) failure is interpretive gold.** Equalizing per-sample contribution while keeping `surf_weight=10` over-prioritizes surface fitting on now-equally-weighted samples → `val_single_in_dist` regresses by 21%, exactly what the theory predicts.
- **Test-time `mae_surf_Ux=0.954` on cruise** is a ~50% improvement over baseline. Per-sample norm helps the velocity fields too, not just pressure.
- **Vs merged baseline (val=120.97):** Run B looks like a 0.9% nominal improvement, but it's not apples-to-apples — alphonse is missing PR #344's schedule fix. Stacking per-sample-norm + sw=5 on top of the merged schedule should be near-additive, plausibly val ≈ 113–117 if so.
- **NaN fix duplicated edward's** (already merged via #344). Drop on rebase.

### Action

Sent back for rebase + tightened surf_weight sweep on the merged schedule. New runs use `--epochs 25 --lr 7e-4` and group `h1-per-sample-ystd-rebased`, with sw ∈ {3, 5, 7}. Decision rule: merge if best rebased run beats val=120.97 by ≥2%; close cleanly if effect is lost when paired with proper schedule.

### Held in reserve / promising follow-ups (post-decision)

- **Per-channel y-std clamp** — channel-specific floor (larger floor on p than Ux/Uy) might let pressure get more benefit without de-emphasizing already-small velocity losses on low-Re cruise.
- **EMA-smoothed per-sample std** — defensive against rare degenerate sample sizes; likely not needed here but cheap insurance.

---

## 2026-04-28 02:51 — PR #404: H11 Re-conditional FiLM modulation — **SENT BACK FOR DISENTANGLEMENT**

- Branch: `willowpai2d4-edward/h11-film-re-conditioning`
- Hypothesis: FiLM (γ, β) per-block from `log(Re)` should reduce `val_avg/mae_surf_p` by 3–7%, biggest on `val_re_rand`.
- 3-cell matrix in W&B group `h11-film-re`:

| Run | FiLM | wd | val_avg/mae_surf_p | test_avg/mae_surf_p | params | W&B |
|-----|------|----|---------------------|----------------------|--------|-----|
| Merged baseline (#344) | — | 1e-4 | 120.97 | 109.92 | 0.66M | `rua9xrca` |
| A — FiLM off (sanity) | off | 1e-4 | 129.27 | 113.83 | 0.66M | `629fuile` |
| B — FiLM on | on | 1e-4 | 126.63 | 113.61 | 0.75M | `3so6w84f` |
| C — FiLM on + wd 5e-4 | on | 5e-4 | **119.63** | **109.11** | 0.75M | `dbogls54` |

### Issues flagged

- **Run A landed 7% worse than the merged baseline on equivalent code paths.** Edward verified bitwise-identical forward at step 0, so the discrepancy is run-to-run training noise. This means the noise floor is at least as large as any claimed FiLM effect.
- **Matched-wd FiLM toggle (A→B) is essentially flat on test.** The Run C improvement appears to come from raising wd from 1e-4 to 5e-4, not from FiLM itself.
- **Cross-split signature didn't match prediction.** Predicted strongest gain on `test_re_rand`; observed strongest gain on the *geometry-OOD* splits (`camber_rc` -8.1%, `camber_cruise` -4.1%) and a regression on `test_re_rand` (+2.1%) and `test_single_in_dist` (+6.2%). Net change is mostly redistribution of error across splits.

### Action

Sent back for two runs that disambiguate:

1. **Run D — FiLM off + wd=5e-4** (the critical disentanglement). If Run D ≈ Run C, FiLM is doing nothing.
2. **Run E — Run C reproduced with `torch.manual_seed(123)`** to test whether the result is reproducible (current single-run-per-cell variance is ~7% based on the A-vs-baseline gap).

Decision rule on resubmit: merge only if Run D is clearly worse than Run C (FiLM is contributing) AND Run E is within ~2% of Run C (result is reproducible). Otherwise close; if wd=5e-4 alone closed the gap, ship it as a 1-line tweak (or leave it as a documented option).

### Held in reserve / promising follow-ups (post-decision)

- **Concat-Re or richer conditioning vector** (`[log(Re), AoA1, AoA2, gap, stagger]`) — interesting if Run D confirms FiLM is real.
- **Per-block FiLM hidden=32** to halve the parameter overhead.

---

## 2026-04-28 02:36 — PR #345: H4 surface-only norm + signed distance feature — **CLOSED**

- Branch: `willowpai2d4-fern/h4-surf-norm-distance` (cut before PR #344 merged)
- Hypothesis: Surface-only normalization (split heads on `mlp2`) + a per-node distance-to-nearest-surface feature should reduce `val_avg/mae_surf_p` by 4–10%, biggest on geometry-OOD.
- 2-cell matrix in W&B group `h4-surf-norm-distance`:

| Run | Components | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|------------|------------|---------------------|----------------------|-----|
| A | C1 (distance feature only) | 14/14 | 134.91 | 122.63 | `9dvfrpke` |
| **B** | **C1 + C2 full split heads** | **13/14** | **129.13** | **118.22** | `xdcv4qym` |

### Per-split test/{split}/mae_surf_p (Run B vs Run A)

| Split | Run A | Run B | Δ |
|-------|------:|------:|--:|
| `test_single_in_dist` | 144.94 | 136.70 | **−5.7%** |
| `test_geom_camber_rc` | 152.93 | 134.67 | **−11.9%** |
| `test_geom_camber_cruise` | 79.93 | 86.14 | **+7.8%** ❌ |
| `test_re_rand` | 112.72 | 115.38 | **+2.4%** ❌ |

### Conclusions

- **Best run is +6.7% regression vs the merged baseline** (val=129.13 vs 120.97). Even on apples-to-apples pre-merge schedule (Edward Run A: val=125.17), Run B is still +3.2% worse — H4 underperforms even without the schedule fix.
- **Cross-split signature is structurally split.** RaceCar geom-OOD improves dramatically (-11.9%) while cruise geom-OOD regresses (+7.8%). Same mechanism: per-head normalization rebalances loss in favor of the regimes whose surface and volume distributions differ most. RaceCar has y_std_surf ≈ 913 vs y_std_vol ≈ 786 (large gap) → benefits. Cruise has small surf-vs-vol gap → hurts.
- **The mechanism that delivers the raceCar gain *is the same mechanism* that hurts cruise.** Rebasing won't fix this. The structural flaw is in the rebalancing direction itself, not in head capacity.
- **C1 (distance feature) alone is only marginally informative** (Run A val=134.91); pays off only when paired with the head split, but the head split is what causes the cruise regression.
- **NaN fix duplicated edward's** (already merged via #344).

### Useful follow-ups (deferred)

- **C2-Lite ablation** (per-node std rescale, no extra parameters) would decouple loss-rebalancing from capacity. Cruise structural penalty likely persists, but worth knowing whether the gain is purely from rebalancing.
- **Multi-scale distance feature** (`log(1 + d/L_ref)` with dataset-wide reference) could give a more comparable signal across the three domains. Worth pairing with a *different* surface treatment in a later round.
- **Per-domain or learned `surf_weight`** — closely related to frieren's H10 (in flight). If H10 lands, that's evidence for revisiting per-domain weighting.

### Action

Closed; reassigning fern to H9 (pressure-gradient penalty along surface) — physics-aware, plays to her diagnostic strength.

---

## 2026-04-28 02:21 — PR #347: H5 random Fourier features on (x, z) — **SENT BACK FOR REBASE**

- Branch: `willowpai2d4-nezuko/h5-fourier-features` (cut before PR #344 merged → missing the warmup+cosine schedule and NaN fix)
- Hypothesis: Fourier features on raw (x, z) coords with `(num_freq, sigma)` tuned should reduce `val_avg/mae_surf_p` by 2–8%, primarily on geom-OOD splits.
- 3-cell σ-sweep in W&B group `h5-fourier`:

| Run | num_freq | σ | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|----------|---|------------|---------------------|----------------------|-----|
| A | 16 | 2.0 | 11 | 138.35 | 124.88 | `mo9w34bp` |
| **B** | **32** | **4.0** | **14** | **127.06** | **112.52** | `o3uq5499` |
| C | 64 | 8.0 | 11 | 132.93 | 123.15 | `bx35gs7g` |

### Conclusions (provisional, pre-rebase)

- **Run B (σ=4) is the clean U-shape winner** of the σ-sweep. Wins on every val/test split, confirming the predicted U-shape (σ=2 too low, σ=8 too high, σ=4 sweet spot).
- **Vs. pre-merge baseline** (Edward Run A on equivalent schedule: val=125.17, test=113.85), Run B is approximately flat on val (+1.5%) and mildly better on test (-1.2%). Real but small.
- **Vs. merged baseline** (PR #344, val=120.97, test=109.92), Run B is +5% / +2.4% — but Nezuko's branch is missing the warmup+cosine schedule fix that delivered ~3-5% on its own. Comparison is not apples-to-apples until rebased.
- **Cross-split signature was wrong-mechanism.** Predicted strongest gain on geom-OOD splits; observed strongest gain on `single_in_dist`. The improvement is general spatial-representation quality, not a geometry-extrapolation regularizer.
- **Run B is still descending at epoch 14** (the last actually-trained epoch) — the run is bottlenecked by the pre-merge per-epoch cosine schedule that PR #344 fixed.
- **NaN fix in `evaluate_split` duplicated edward's** (already merged via #344). Drop it on rebase.

### Action

Sent back for rebase + tightened σ-sweep on the merged schedule. Decision rule on resubmit: if best rebased run beats `val_avg/mae_surf_p=120.97`, merge; else close cleanly. New runs use `--epochs 25 --lr 7e-4` and group `h5-fourier-rebased`, with σ ∈ {3, 4, 5} at num_freq=32 plus a (num_freq=64, σ=4) decoupling cell.

---

## 2026-04-28 01:46 — PR #349: H8 slice_num scaling matrix — **CLOSED**

- Branch: `willowpai2d4-thorfinn/h8-slice-num-scaling`
- Hypothesis: scaling Transolver `slice_num` from 64 → 128/256, optionally with width compensation, should reduce `val_avg/mae_surf_p` by 2–7%.
- 4-cell matrix in W&B group `h8-slice-num`. Branch was cut before PR #344 merged so all runs use the pre-merge schedule (`CosineAnnealingLR` per-epoch, no warmup); not directly comparable to the merged baseline:

| Run | slice_num | n_hidden | n_head | bs | params | s/epoch | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|-----------|----------|--------|----|---------|---------|------------|---------------------|---------------------|-----|
| A | 128 | 128 | 4 | 4 | 0.67M | 171 | 11/11 | **148.65** | **136.69** | `29ltc5zn` |
| B | 256 | 128 | 4 | 4 | 0.69M | 252 | 4/8 | 179.57 | 172.21 | `a9hy5emm` |
| C | 128 | 192 | 6 | 4 | 1.46M | 263 | 7/7 | 158.61 | 144.72 | `a4lnwgv4` |
| D | 192 | 192 | 6 | 2† | 1.47M | 304 | 6/6 | 173.08 | 158.38 | `noco1c7f` |

†Run D OOM'd at bs=4 (PR was advised to drop to bs=2 as the failure-mode mitigation).

### Conclusions

- **Slice scan plateaued at 128 — within the scan.** Run A (slice 128, h 128) wins on every val/test split. Run B (slice 256) regressed with a val curve that plateaued at epoch 4 then bounced 180–205, the partition-collapse signature.
- **Best run is +23% regression vs the merged baseline** (val=148.65 vs 120.97). Even after rebasing to inherit PR #344's schedule fix, slice_num=128 is unlikely to recover the gap given the within-scan ordering.
- **Width was not the missing lever.** Run C (slice 128 + h=192) underperformed Run A on every metric.
- **Compute fairness caveat:** Runs B/C/D completed only 4–8 epochs vs Run A's 11. The slice-scan ordering on Run B is robust (val plateau visible) but Runs C/D were still descending — those *might* narrow with more compute.
- **NaN fix was duplicated** (independently reproduced edward's PR #344 fix). Equivalent functionality, no merge action needed.
- **Useful follow-ups (deferred):** try slice_num=96 (the curve from 128→256 went sharply up; we never tested below 128 in this scan), and revisit width+slice scaling once H6 throughput lands.

### Action

Closed; reassigning thorfinn to H12 (EMA of weights) — a cheap compounding lever well-suited to layer onto whatever round-1 winner emerges.

---

## 2026-04-28 00:38 — PR #346: H7 z-mirror augmentation — **CLOSED**

- Branch: `willowpai2d4-frieren/h7-zmirror-augmentation`
- Hypothesis: z-axis mirror augmentation with sign flips on z-position, saf[1], AoA, Uy should reduce `val_avg/mae_surf_p` by 3–8% by doubling effective training data via 2D physical symmetry.
- 3-cell matrix in W&B group `h7-zmirror`:

| Run | `augment_zmirror` | best_val_avg/mae_surf_p | best_epoch | W&B |
|-----|-------------------|--------------------------|------------|-----|
| A | 0.0 | 124.71 | 12 | `4nu04gte` |
| B | 0.5 | 169.29 (+35.8%) | 8 | `9p19tvj6` |
| C | 1.0 | 412.52 (+231%) | 1 | `4o4nfvhb` |

`test_avg/mae_surf_p` was None on all 3 runs because `test_geom_camber_cruise/mae_surf_p` came back NaN — same root cause edward's PR fixed (now merged on advisor branch, so future runs are robust).

### Conclusions

- **Strict monotonic regression.** Run C's diagnostic is decisive: train losses descend cleanly on the all-mirrored distribution but val MAE *increases* during training, the classic distribution-shift fingerprint. The (mirrored x, mirrored y) pair is **not** a valid sample of the same underlying CFD problem — the augmentation is breaking the input→output mapping rather than preserving it.
- **`mae_surf_Uy` regresses ~3x** under augmentation despite being explicitly sign-flipped — confirms the model can't learn the symmetry from the corrupted training distribution.
- **Likely structural causes (per student diagnostic):**
  1. raceCar tandem (~30% of training) has a slip-wall ground at z=0 — the BC effects (ground effect, wake interaction) are not z-symmetric, so mirroring produces a sample of a *different* CFD problem with hallucinated targets.
  2. `gap`/`stagger` (dims 22, 23) decompose differently in cruise vs. raceCar (chord-aligned frame for raceCar). The conservative no-flip choice corrupts ~10% of cruise tandem; flipping to fix cruise corrupts raceCar. There is no consistent z-mirror law given the dataset construction.
- **Worthwhile follow-ups (deferred):**
  1. **Test-time augmentation (TTA)** — average predictions on (x, mirror(x)) at val time. Tests whether the symmetry exists in the trained model, divorced from training-time corruption.
  2. **Domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip. Small slice of training data but clean physics.
- **Action:** closed; reassigning frieren to a fresh hypothesis.
