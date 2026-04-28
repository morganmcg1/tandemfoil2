# SENPAI Research Results — willow-pai2e-r3

## 2026-04-28 21:28 — PR #814 (MERGED): Huber surface loss (delta=1.0)
- **Branch:** `willowpai2e3-askeladd/huber-surf-loss`
- **Hypothesis:** Replace MSE surface loss with Huber(delta=1.0) to align training objective with MAE metric and gain robustness against heavy-tailed pressure errors. Predicted -5 to -10%.
- **Run:** W&B `at52zeu5`, **14/14 epochs (clean)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 123.94 | 109.10 |
| `*_geom_camber_rc` | 111.30 | 98.88 |
| `*_geom_camber_cruise` | 81.66 | **69.84** |
| `*_re_rand` | **95.62** | **94.17** |
| **avg** | **103.13** | **92.99** |

### Decision: MERGED (2026-04-28) — new leading winner
- **−15.6% val (122.15 → 103.13)** — well outside noise band; beats tanjiro L1 (109.53).
- **−29% test (130.90 → 92.99)** — spectacular test improvement; founding clean test baseline.
- Hypothesis confirmed: L1-like gradient tail robustness on surface pressure with MSE stability near zero.
- Val still falling at epoch 14 (−1.5 from ep13→14) — headroom remains at longer budget.
- Student noted: `reduction='mean'` implicitly weakens surf_weight by ~3× vs old `sum/N_nodes` form; despite this the metric improved — loss shape is doing the work.
- **New beat-threshold: val_avg < 103.13**
- **Follow-up assigned: #847** (huber-delta-sweep: try delta=0.5 to push closer to L1).

## 2026-04-28 19:55 — PR #743: Per-channel surface loss: 3× weight on pressure
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Boost pressure channel by 3× inside surface loss to align training signal with the `mae_surf_p` ranking metric. `y_std_p ≈ 679`, ~30× larger than `y_std_Ux` and ~70× larger than `y_std_Uy`; uniform-weighted MSE under-emphasizes the metric channel.
- **Run:** W&B `zaqz12qi` (entity `wandb-applied-ai-team`, project `senpai-charlie-wilson-willow-e-r3`)
- **Budget consumed:** 14/50 epochs (hit `SENPAI_TIMEOUT_MINUTES=30` cap; ≈131 s/epoch)

### Results

| Split | val (best ckpt @ epoch 14) | test (best ckpt @ epoch 14) |
|---|---|---|
| `*_single_in_dist` | 196.59 | 166.63 |
| `*_geom_camber_rc` | 156.43 | 141.34 |
| `*_geom_camber_cruise` | **107.40** | **null** |
| `*_re_rand` | 124.01 | 122.96 |
| **avg** | **146.10** | **null (cruise NaN propagates)** |

### Analysis & decision: SEND BACK
- Val side is informative: `val_avg/mae_surf_p = 146.10`, with `cruise = 107.40` the best of the four val splits — consistent with the hypothesis that p-channel boost helps where p dominates surface dynamics.
- **Test side blocks merge.** `test_geom_camber_cruise/mae_surf_p = null` (single non-finite prediction polluting the global accumulator in `accumulate_batch` — `data/scoring.py` only skips on non-finite ground truth, not non-finite preds). Three of four test splits finite. Per CLAUDE.md, NaN/missing on the paper-facing metric blocks adoption.
- Budget reality check: at default `SENPAI_TIMEOUT_MINUTES=30` and current model size, only ~14 epochs fit. **All round-1 PRs are timeout-limited to ~14 epochs**, not 50. Future hypothesis design should account for this — recommend setting `--epochs 14` explicitly so cosine annealing reaches end-of-curve LR rather than mid-curve.
- Sent back with feedback to:
  1. Add a NaN-guard / clamp in `evaluate_split` (`pred = torch.nan_to_num(pred, ...).clamp(-20, 20)` before denormalization) — defends MAE numerics for all future students once merged.
  2. Try softer per-channel weights `[1.0, 0.5, 2.0]` instead of `[1.0, 1.0, 3.0]` — 2× boost on p (closer to variance ratio after surface gating absorbs most of it) plus 0.5× on Uy (over-represented and not in ranking metric).
  3. Set `--epochs 14` explicitly to plan for the timeout.
- Once v2 lands with finite `test_avg/mae_surf_p`, this becomes the founding baseline for the branch.

### Cross-cutting findings (apply to ALL round-1 students)

- **Timeout is the binding constraint, not epoch count.** Plan for 14 epochs, not 50.
- **NaN test poisoning is a real `data/scoring.py` bug, not a model issue.** Identified by askeladd in PR #748 (since closed): `accumulate_batch` does `err * mask` where `NaN * 0 = NaN` in IEEE-754. `test_geom_camber_cruise/sample 20` has 761 NaN values in the **ground truth** `p` channel. This poisons `mae_surf_p` for that split on every run regardless of model. Fix: `torch.where(mask, err, 0)`. Same pattern in `evaluate_split` for `vol_loss`/`surf_loss` produces `Infinity`. Assigned to askeladd as PR #807. Once that lands, all future runs will produce clean `test_avg` numbers.
- **Cruise OOD camber (M=2-4)** is otherwise the most extrapolation-prone test split — already the hardest extrapolation track regardless of the NaN bug.

## 2026-04-28 20:02 — PR #750: Linear warmup + cosine LR schedule (lr=1e-3, wd=5e-4)
- **Branch:** `willowpai2e3-edward/lr-warmup-cosine`
- **Hypothesis:** 500-step linear warmup + per-step cosine to 0, lr=1e-3, wd=5e-4 — should buy 3–8% over plain cosine.
- **Run:** W&B `thnnvgaw`, 14/50 epochs (timeout), best ckpt @ epoch 12, peak 83.6 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 162.00 | 144.47 |
| `*_geom_camber_rc` | 148.21 | 136.08 |
| `*_geom_camber_cruise` | 102.53 | null (scoring bug) |
| `*_re_rand` | 130.84 | 127.33 |
| **avg** | **135.89** | **null** |

### Decision: SEND BACK (v1)
- 1.7% better than student's quoted baseline-mean reference (5 runs, range [124.6, 146.1]) — within noise; the hypothesized 3–8% gain isn't demonstrated.
- Student's diagnosis is exactly right: cosine schedule with `T_max = 50 × 375 ≈ 18.7K steps` but only ~5.2K steps fit in 30 min — never reaches the low-LR fine-tuning regime.
- Sent back: set `--epochs 14` explicitly so cosine actually anneals end-to-end; raise peak LR to `2e-3` (warmup makes higher peaks safe — that's where the standard transformer warmup gain lives).
- (Branch hygiene: edward referenced 5 baseline runs t0xgo0zv/6zc9kq6x/6lj642bf/7qi7tbcy/zaqz12qi as a noise band; only zaqz12qi (alphonse v1) is from this advisor branch, the others are out-of-scope. The variance argument stands; the specific run-IDs do not.)

### v2 Results (2026-04-28 20:50) — `lr=2e-3, epochs=14, warmup_steps=500`
- **Run:** W&B `mv16jwsp`, **14/14 epochs (clean finish, no timeout)**, best ckpt @ epoch 14, peak 94.24 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 133.38 | 116.37 |
| `*_geom_camber_rc` | 120.79 | 112.57 |
| `*_geom_camber_cruise` | **85.96** | null (pre-#807 run) |
| `*_re_rand` | 104.37 | 101.46 |
| **avg** | **111.12** | **null (re-eval pending)** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−9.0% vs founding baseline (122.15 → 111.12)** — well outside round-1 noise band.
- All 4 val splits improved uniformly (-17 to -20% vs v1) — gain is from genuine convergence, not split-specific quirk.
- Best epoch landed exactly at 14/14 with cosine asymptote reached; flat last-3 deltas (113.75 → 111.13 → 111.12) suggest near-asymptote for this config. Suggests v3 with lr=3e-3 or longer budget could buy more.
- **Conflict:** branch predates PR #807; both touched `train.py` near loss accumulation. Sent back to rebase + re-run (same config) to verify gain holds + pick up clean `test_avg`. Once verified ≤~115, will merge immediately.

## 2026-04-28 20:04 — PR #748: Transolver 2x capacity scale-up
- **Branch:** `willowpai2e3-askeladd/transolver-2x-capacity`
- **Hypothesis:** n_hidden=192, n_layers=8, n_head=8, slice_num=128, mlp_ratio=4 (3.42M params) — predicted 5–15% gain.
- **Run:** W&B `p486z24b`, **4/50 epochs only** (timeout), best ckpt @ epoch 3, peak 82.5 GB.
- **val_avg/mae_surf_p = 203.16** (raw); test_avg null (scoring bug); **test_avg corrected = 191.71** (offline re-eval with `torch.where`).

### Decision: CLOSE
- val_avg = 203 vs. round-1 baseline-range ~140 — clear regression at 4/50 epochs.
- Approach not broken — it's that 50-epoch cosine schedule with only 4 epochs done means LR is still ~98% of peak. Model never reached convergence regime where 2× capacity is supposed to help.
- **Critical bug discovery embedded in PR comments**: pinpointed the `data/scoring.py` NaN-mask bug, validated the fix offline. Spawned PR #807 to land the fix as their next assignment.
- For round 2 capacity: budget-matched schedule (`--epochs 4` so cosine completes), or smaller capacity boost (n_layers=6 to fit ~8 epochs) — defer until scoring fix merges.

## 2026-04-28 20:15 — PR #756: Fourier features for log(Re) input encoding
- **Branch:** `willowpai2e3-frieren/fourier-re-encoding`
- **Hypothesis:** Replace scalar `log(Re)` (dim 13) with 16 sin/cos features at 8 freqs `[1, 2, 4, 8, 16, 32, 64, 128]` for richer cross-Re generalization. Predicted 3-7% gain.
- **Run:** W&B `t0xgo0zv`, 14/50 epochs (timeout), best ckpt @ epoch 14, peak 42.3 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 202.03 | 169.41 |
| `*_geom_camber_rc` | 139.03 | 122.76 |
| `*_geom_camber_cruise` | 102.50 | null (scoring bug) |
| `*_re_rand` | 121.42 | 123.83 |
| **avg** | **141.25** | **null** |

### Decision: SEND BACK (v1)
- val_avg=141.25 sits inside the round-1 noise band (other v1 runs: alphonse 146.10, edward 135.89). Predicted 3-7% gain not demonstrable at single-seed.
- val_re_rand=121.42 (the strongest per-split) is suggestive — Fourier-of-log(Re) may help cross-Re generalization. Direction worth iterating on, not abandoning.
- val curve still falling steeply at the cutoff (epochs 11-14: 152→160→160→141) — model under-converged.
- Sent back: (a) concatenate Fourier features instead of replacing dim 13 (preserves smooth scalar path); (b) drop top frequencies, use `[1, 2, 4, 8, 16, 32]` (high freqs cycle below the data's Re resolution); (c) `--epochs 14` explicit so cosine completes.

### v2 Results (2026-04-28 21:04) — concat + 6 freqs + `--epochs 14`
- **Run:** W&B `tg59rxt1`, **14/14 epochs (clean finish)**, best ckpt @ epoch 14, peak 42.3 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 147.07 | 126.45 |
| `*_geom_camber_rc` | 125.55 | 119.19 |
| `*_geom_camber_cruise` | 97.37 | null (pre-#807 run) |
| `*_re_rand` | **110.89** | **111.10** |
| **avg** | **120.22** | **null (re-eval pending)** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−1.6% vs founding baseline (122.15 → 120.22)** — small win but positive signal.
- v2 vs v1: −14.9% improvement; uniform across all 4 val splits and all 3 finite test splits.
- **Strongest signal on `val_re_rand=110.89` and `test_re_rand=111.10`** — exactly the cross-Re generalization track the encoding was designed for. This is the genuine encoding gain.
- Frieren correctly noted bundled changes: (a) concat encoding, (b) drop high freqs, (c) `--epochs 14`. Code-level changes are only (a) and (b); (c) is just a runtime flag so the encoding hypothesis is honestly tested by the diff.
- val curve still descending at cutoff (124.65 → 123.74 → 123.26 → 120.22), under-converged.
- **Conflict:** branch predates PR #807; needs rebase. Sent back to rebase + re-run with same config (`--epochs 14`) to verify gain holds + pick up clean `test_avg`.
- Notable: edward's parallel PR #750 v2 (lr-warmup-cosine) hits val_avg=111.12 — a *better* winner via LR schedule changes. The two are mechanism-orthogonal; both can compound. If edward's PR merges first, frieren's encoding contribution is measured on top of edward's optimizer fix.

## 2026-04-28 21:09 — PR #761: L1 (MAE) surface loss aligned with metric
- **Branch:** `willowpai2e3-tanjiro/l1-surface-mae-loss`
- **Hypothesis:** Replace surface MSE with L1 (MAE) loss to align training objective directly with the `mae_surf_p` ranking metric and provide robustness to the heavy-tailed pressure distribution. Predicted -5 to -12% gain.
- **Run:** W&B `ee9p55qd`, 14/50 epochs (timeout), best ckpt @ epoch 13, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 149.36 | 128.91 |
| `*_geom_camber_rc` | 107.87 | 100.02 |
| `*_geom_camber_cruise` | 82.64 | **70.62** |
| `*_re_rand` | **98.25** | **94.20** |
| **avg** | **109.53** | **98.44** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−10.3% vs founding baseline (122.15 → 109.53)** — well outside round-1 noise band.
- **Best winner on the branch**: ahead of edward (111.12) and frieren (120.22).
- **First clean test_avg** beating founding baseline (130.90 → 98.44, −24.6%) — tanjiro bundled their own sample-level NaN-guard which works equivalently to #807.
- Hypothesis directly confirmed: L1 metric-alignment + pressure-tail robustness. `val_re_rand=98.25` and `test_re_rand=94.20` are exceptional per-split numbers.
- val curve still improving at epoch 13 (epoch 14 worsened so checkpoint at 13). Suggests headroom on a longer/better-tuned schedule.
- Bundled NaN-guard in `evaluate_split` is redundant with #807's torch.where (and should be dropped on rebase).
- **Conflict:** branch predates #807; touched same loss-computation lines. Sent back to rebase + re-run with `--epochs 14` (cosine T_max alignment, like edward/frieren).
- Once rebased run lands at val_avg ≤ ~115, merge immediately. This is the single biggest gain on the branch.

### Queued follow-ups (post-merge)
1. surf_weight=3.0 — student's own diagnosis: L1 surface gradient currently dominates 10:1 vs vol_loss. Rebalancing should free volume capacity.
2. L1 on p only, MSE on Ux/Uy — pressure-tail alignment without changing dynamics for the better-behaved velocity channels. Possibly stacks with alphonse's channel-weighted-3xp.

---

## 2026-04-28 20:08 — PR #807 (MERGED): Bug fix — NaN-safe masked accumulation
- **Branch:** `willowpai2e3-askeladd/scoring-nan-mask-fix`
- **Type:** Infrastructure bug fix (not a hypothesis experiment)
- **Scope:** `data/scoring.py::accumulate_batch`, `train.py::evaluate_split`, training-loop loss accumulation.

### Results

Fix: replaced `(err * mask).sum()` with `torch.where(mask, err, zero).sum()` in all three locations. Mathematically equivalent for finite y; zeroes out NaN contributions rather than propagating them.

**Re-evaluated checkpoints (no retraining):**

| Run | Old test_avg/mae_surf_p | New (corrected) test_avg/mae_surf_p | Notes |
|-----|------------------------|--------------------------------------|-------|
| `zaqz12qi` (alphonse channel-weighted v1) | null | **130.897** | test_geom_camber_cruise=92.66 |
| `p486z24b` (askeladd transolver-2x) | null | 192.259 | test_geom_camber_cruise=175.62 (under-trained) |

`--debug` smoke test: identical outputs pre/post fix on clean training splits. New `scripts/reeval_artifact.py` helper included.

### Decision: MERGED (2026-04-28)
- Infrastructure fix unblocking finite `test_avg/mae_surf_p` for all future runs.
- Founding baseline established: `zaqz12qi` val_avg=146.10 / test_avg=130.897.
- Also added BASELINE.md to advisor branch anchored by thorfinn's matched-baseline run (8cvp4x6r, val_avg=122.15 — best unmodified model result in round 1).

---

## 2026-04-28 20:31 — PR #762 (CLOSED): Boundary-layer feature: log(Re·|saf|) as input
- **Branch:** `willowpai2e3-thorfinn/boundary-layer-features`
- **Hypothesis:** Add `log(Re·|saf|)` as a 25th input dimension to give an explicit local Re_x boundary-layer signal. Predicted -10 to -25% gain, strongest on `val_re_rand`.
- **Run:** W&B `7qi7tbcy` (BL feature), `8cvp4x6r` (matched baseline), 14/50 epochs (timeout), same GPU back-to-back.

### Results

| Split | Baseline (8cvp4x6r) | +log(Re·|saf|) (7qi7tbcy) | Δ |
|---|---|---|---|
| `val_single_in_dist` | 143.36 | 180.55 | **+25.9%** |
| `val_geom_camber_rc` | 124.20 | 159.73 | **+28.6%** |
| `val_geom_camber_cruise` | 109.42 | **95.96** | -12.3% |
| `val_re_rand` | 111.63 | 117.47 | +5.2% |
| **val_avg** | **122.15** | **138.43** | **+13.3% (WORSE)** |

Test splits: both have test_avg null (pre-fix runs); 3-split avg: baseline 118.01, BL 139.50 (+18.2% worse).

### Analysis & decision: CLOSE
- Consistent negative across 3/4 val splits and all 3 finite test splits. Cruise is the only win.
- **Information redundancy diagnosis (thorfinn):** dim-13 log(Re) and dims 2:3 saf are already in x; MLP preprocess can construct their product for free. Explicit feature likely competes with rather than augments existing capacity.
- **Volume-node saf mismatch:** saf is arc-length on surface nodes but undefined/different off-surface; broadcasting `log(Re·|saf|)` to all nodes injects physically wrong signal for volume nodes.
- Cruise improvement (-12.3%) is real but isolated (100 samples, test NaN-poisoned, single seed) and insufficient to outweigh the other regressions.
- **Exceeds the >5% close threshold** (13.3% regression). Closed 2026-04-28.
- Matched baseline `8cvp4x6r` (val_avg=122.15) promoted to BASELINE.md as best clean unmodified-model result.

### Cross-cutting: thorfinn's matched-baseline methodology
Thorfinn independently identified the data/scoring.py NaN bug (same root cause as askeladd). Excellent experimental practice: ran matched baseline side-by-side, produced full split breakdown, conducted 3-cause analysis. The matched baseline (122.15) reshapes our noise estimate — round-1 noise band is now 122–146, not 135–146.

## 2026-04-28 22:00 — PR #759 (CLOSED): EMA model weights (decay=0.999)
- **Branch:** `willowpai2e3-nezuko/ema-model-weights`
- **Hypothesis:** Apply Exponential Moving Average (decay=0.999) of model weights to reduce noise in surface pressure predictions; an EMA shadow model averages over the stochastic gradient trajectory, producing smoother predictions. Predicted −5 to −10% on val_avg/mae_surf_p.
- **Run:** W&B `qetkdsku`, 14/14 epochs, askeladd Huber-merged baseline (103.13) used as beat-threshold.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | — | — |
| `*_geom_camber_rc` | — | — |
| `*_geom_camber_cruise` | — | — |
| `*_re_rand` | — | — |
| **avg** | **124.51** | **110.63** |

### Analysis & decision: CLOSE
- val_avg=124.51 is **+20.7% worse** than current best (PR #814 Huber, 103.13) and **+1.9% worse** than founding baseline (122.15) — within round-1 noise but in the wrong direction vs. the current bar.
- **Regime mismatch (fundamental diagnosis):** EMA's benefit is maximal in the *converged-but-noisy* regime — late training where parameters have found a basin but stochastic gradients cause high-frequency jitter. At 14-epoch budget, Transolver is still descending the loss curve (the live weights are improving every epoch). EMA shadow weights (decay=0.999 means ~1000-step effective memory) are staler than the current live weights and drag the ensemble toward earlier, worse states rather than smoothing noise around a convergence point.
- **Supporting evidence:** nezuko's own analysis correctly identified this: "the training loss was still decreasing at epoch 14, which suggests the model hadn't fully converged and EMA might have been averaging over a range of improving but not yet optimal weights."
- **EMA is correctly motivated for a longer budget.** At ~50 epochs or with a lower LR tail, this hypothesis should be revisited. At current 14-epoch ceiling, EMA is a hindrance.
- Notable: nezuko independently discovered a NaN-guard variant during implementation — good diagnostic instinct. That NaN-guard is covered by the already-merged PR #807 (torch.where pattern), so no further action needed on that front.
- **Closed 2026-04-28. Next assignment: #858 focal-surface-loss (gamma=1.0).**
