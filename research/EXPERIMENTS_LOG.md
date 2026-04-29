# SENPAI Research Results — willow-pai2e-r3

## 2026-04-28 22:20 — PR #761 (MERGED): L1 surface MAE loss
- **Branch:** `willowpai2e3-tanjiro/l1-surface-mae-loss`
- **Hypothesis:** Replace surface MSE (then Huber) with pure L1 (MAE) to align training objective directly with `mae_surf_p` metric and exploit pressure's heavy-tailed residual distribution. Predicted −5 to −12% gain.
- **Run (v1-rebased):** W&B `tirux1y1`, **14/14 epochs (clean finish, cosine → 0)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 109.65 | 96.33 |
| `*_geom_camber_rc` | 101.17 | 90.80 |
| `*_geom_camber_cruise` | **72.37** | **61.90** |
| `*_re_rand` | **87.33** | **82.29** |
| **avg** | **92.63** | **82.83** |

### Decision: MERGE PENDING — 2nd rebase required (conflict with #814)
- **−10.2% val vs Huber-merged baseline (103.13 → 92.63)** — new branch best.
- **−10.9% test (92.99 → 82.83)** — new best test metric.
- L1 beats Huber(delta=1.0) by 10%: pure linear gradient on all surface residuals outperforms the smooth-near-zero Huber transition for this dataset's heavy-tailed pressure distribution.
- V1-rebased vs v1-timeout: −15.4% val — bulk of gain is from completing the cosine LR schedule (v1 was at ~95% of peak LR at timeout; v1-rebased reached LR=0 at epoch 14). Loss-shape gain is real but proper schedule alignment amplified it.
- `val_re_rand=87.33`, `test_re_rand=82.29` — exceptional regime-generalization numbers.
- Surface loss dominates volume ~7:1 at convergence (tanjiro's own diagnosis). 2nd rebase resolved cleanly — single block swap, no logic change. Merged 2026-04-28.
- **New beat-threshold: val_avg < 92.63**
- **Round-3 follow-up assigned: #869** (`surf_weight=3.0` rebalancing sweep; tanjiro).

---

## 2026-04-28 22:55 — PR #751 v2 (CLOSED): Dropout 0.05 + stoch-depth 0.05 on L1 base
- **Branch:** `willowpai2e3-fern/dropout-stochastic-depth`
- **Hypothesis:** v2 — halve regularization (`dropout=0.05, drop_path_max=0.05`) on the merged L1 advisor branch. Goal: keep v1's OOD-generalization wins without v1's in-dist regression.
- **Run (v2):** W&B `pbsbkt4g`, **14/14 epochs (clean cosine→0)**, peak ~43 GB.

| Split | Baseline (L1, #761) | v2 | Δ |
|---|---:|---:|---|
| `val_single_in_dist` | 109.65 | **107.85** | −1.6% |
| `val_geom_camber_rc` | 101.17 | 101.77 | +0.6% |
| `val_geom_camber_cruise` | 72.37 | 74.77 | +3.3% |
| `val_re_rand` | 87.33 | 88.26 | +1.1% |
| **val_avg** | **92.63** | **93.16** | **+0.6% (within noise)** |
| **test_avg** | **82.83** | **84.20** | **+1.7%** |

### Decision: CLOSE — hypothesis falsified
- v2 within ±10% noise band but in the wrong direction (+0.6% val).
- v1's OOD wins (-5% on geom_camber) **vanished** at half-strength regularization. Per-split sign pattern flipped: in-dist slightly improved, OOD slightly worse.
- **L1 surface loss already captures the OOD-generalization signal** that motivated regularization. Heavy-tailed pressure residuals are now handled by loss shape; regularization had nothing left to add.
- Student's own analysis correctly identified this and offered close-or-ablate. Choosing close: single-mechanism ablation (drop_path-only or dropout-only) is unlikely to beat L1 baseline given the v1→v2 trajectory.
- **Closed 2026-04-28. Next assignment: #902 volume-l1 (mirror surface L1 success on the volume side).**

### Side observation flagged
Fern noted `test_geom_camber_cruise/vol_loss = Infinity` even with #807's torch.where for sq_err. This is because `evaluate_split`'s vol_loss accumulator doesn't have explicit sample-level non-finite-y skip (only #807's element-wise mask). Doesn't affect MAE metrics (which use scoring.py's accumulator). Worth a future hardening pass but not blocking.

---

## 2026-04-28 22:40 — PR #847 (CLOSED): Huber surface loss delta sweep (delta=0.5, 2.0)
- **Branch:** `willowpai2e3-askeladd/huber-delta-sweep`
- **Hypothesis:** Tighten Huber delta from 1.0 to 0.5 (closer to L1) for tighter L1-like gradient on most residuals. Predicted -2 to -5%. Optional delta=2.0 sensitivity probe.
- **Runs:** W&B `2hi8vsek` (delta=0.5) and `ua6tw32g` (delta=2.0), 14/14 epochs both, full cosine cycles.

| delta | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-------|-------------------:|--------------------:|
| 2.0 (probe) | 106.78 | 95.11 |
| 1.0 (#814 baseline) | 103.13 | 92.99 |
| 0.5 (this PR) | **102.97** | **91.21** |
| **L1 (delta→0, merged #761)** | **92.63** | **82.83** |

### Decision: CLOSE
- delta=0.5 just clears the OLD beat-threshold (103.13 → 102.97 = −0.16% on val). But during the run, **PR #761 L1 merged** with val_avg=92.63 — the new baseline.
- delta=0.5 is **+11.2% worse than current best**. Merging it would code-revert from L1 to Huber(0.5), worsening the model.
- **Loss-shape sensitivity curve confirmed monotonic in 0.5-2.0 range** but flat (3.6% spread). The big lever was MSE→Huber (#814: −15.6%); within the Huber regime, delta tuning hits diminishing returns.
- The Huber→L1 step (delta 0.5 → 0) accounts for −9.9%, much bigger than the entire Huber-delta sensitivity range. **Heavy-tailed pressure residual diagnosis confirmed all the way to the L1 limit.**
- Askeladd's analysis was correct on the trend (smaller delta better) but used the *pre-rebase* tanjiro number (109.53) as the L1 reference; the merged L1 is 92.63.
- **Closed 2026-04-28. Next assignment: #884 RevIN output normalization.**

---

## 2026-04-29 00:00 — PR #869 v1: surf_weight sweep (sw=3, sw=5) on L1 baseline
- **Branch:** `willowpai2e3-tanjiro/surf-weight-sweep`
- **Hypothesis:** Reduce `surf_weight` from 10.0 → 3.0 (and 5.0 bracket) to rebalance surface/volume gradient ratio. Predicted: −1 to −4% val_avg; volume-side gain larger.
- **Runs:** W&B `bffsf626` (sw=5) and `4q4b5kzs` (sw=3), 14/14 epochs both, ran on L1 baseline (pre-FiLM merge).

| metric | sw=10 baseline | **sw=5** | sw=3 |
|---|---|---|---|
| val_avg/mae_surf_p | 92.63 | **91.16** (−1.6%) | 95.55 (+3.2%) |
| test_avg/mae_surf_p | 82.83 | **81.36** (−1.8%) | 84.32 (+1.8%) |
| val_avg/mae_vol_p | 103.16 | **92.49** (−10.4%) | 94.57 (−8.3%) |

### Decision: SEND BACK FOR REBASE (onto FiLM+L1)
- **Mechanism directionally confirmed:** lowering surf_weight frees gradient budget → volume features develop more fully (val_vol_p −10.4%) → surface predictions also benefit (−1.6%) via shared trunk. Volume gain dominates by 7×, exactly as predicted.
- sw=5 wins on L1 baseline; sw=3 over-corrects on surface side (still in surface-dominant regime).
- BUT: PR #815 FiLM+L1 just merged at 82.77. sw=5 result of 91.16 is +10.1% above current best.
- Mechanism may stack with FiLM (different mechanisms: hidden-state Re modulation vs loss balance) OR overlap (FiLM strengthens volume features, reducing rebalancing benefit). Empirical answer matters.
- **Sent back:** rebase onto FiLM+L1 advisor; re-run sw=5 primary + optional sw=7 bracket (in case FiLM shifts the optimum toward sw=10).
- New beat-threshold for v2: **val_avg < 82.77**.
- **Round-3 follow-ups queued:** per-channel surface weighting (Ux/Uy/p within surface loss), longer LR schedule.

---

## 2026-04-29 00:00 — PR #750 v2-rebased (CLOSED): LR warmup + cosine on FiLM+L1
- **Branch:** `willowpai2e3-edward/lr-warmup-cosine`
- **Hypothesis history:** v1 (lr=1e-3, 50ep schedule, timeout-cut at 14ep) — pre-rebase: −19.6% on weak baseline mean. v2 (lr=2e-3, 14ep matched schedule) — pre-rebase: 111.12 (−9.0% vs founding). v2-rebased on FiLM+L1: see below.
- **Run (v2-rebased):** W&B `gqc006f6`, **14/14 epochs (clean finish)**, peak 99.13 GB on 96 GB card.

| Split | FiLM+L1 baseline | v2-rebased | Δ |
|---|---|---|---|
| `val_single_in_dist` | 95.54 | 97.09 | +1.6% |
| `val_geom_camber_rc` | 91.38 | **98.56** | **+7.9%** |
| `val_geom_camber_cruise` | 64.90 | 64.94 | +0.1% |
| `val_re_rand` | 79.26 | 79.69 | +0.5% |
| **val_avg** | **82.77** | **85.07** | **+2.78%** |
| **test_avg** | **72.27** | **74.33** | **+2.85%** |

### Decision: CLOSE — schedule mechanism baked into baseline
- **+2.78% val regression** (within single-seed noise but worse than baseline).
- **`val_geom_camber_rc` regresses +7.9%** — highest LR (lr=2e-3) interacts poorly with FiLM's per-block γ/β modulation in the OOD-camber regime. The schedule and conditioning mechanism are touching the same training dynamics.
- Student's honest analysis: pre-rebase 19.6% gain came from fixing schedule mismatch (cosine over 50ep but timeout at 14ep). Once `--epochs 14` became standard for all assignments (after PR #761), that gain was folded into baseline.
- The schedule-budget alignment principle survives the close: it's now the convention for every assignment in this branch (use `--epochs 14` so cosine actually anneals).
- Iterating on lower LR (lr=1e-3 or 7.5e-4) is low-EV at this point — would land within ~2-5% of baseline at most, not a mechanistic test.
- **Closed 2026-04-29. Next assignment: #924 per-channel output heads** (decouple Ux/Uy/p decoder pathways).

---

## 2026-04-28 23:50 — PR #884 (CLOSED): RevIN — per-sample y normalization for surface loss
- **Branch:** `willowpai2e3-askeladd/revin-output-norm`
- **Hypothesis:** Per-sample, per-channel target-stat normalization of pred and y before computing surface L1 loss. Goal: equalize per-Re-sample gradient contribution; predicted −3 to −10% with biggest gains on `val_re_rand`.
- **Run (v1):** W&B `99ltqjj3`, **14/14 epochs (clean finish)**, val_avg still descending but converging slowly, peak 42.2 GB.

| Split | val surf_p (RevIN) | val surf_p (L1 baseline) | Δ |
|---|---|---|---|
| `single_in_dist` | 221.78 | 109.65 | **+102%** |
| `geom_camber_rc` | 166.30 | 101.17 | +64% |
| `geom_camber_cruise` | 98.42 | 72.37 | +36% |
| `re_rand` | 124.04 | 87.33 | +42% |
| **val_avg** | **152.64** | **92.63** | **+65%** |
| **test_avg** | **141.19** | **82.83** | **+70%** |

### Decision: CLOSE — structurally mismatched mechanism
- **+65% val / +70% test vs L1 baseline** — not noise; clean failure.
- **Mechanism falsified.** Student's own analysis: "RevIN's per-sample normalization breaks the agreement between optimizer and metric. The gradient now treats every sample as unit-variance, so the model under-fits the high-amplitude samples that the metric weights most."
- The metric (`mae_surf_p`) is in physical units and high-Re-dominated by Re² scaling. Per-sample loss normalization decouples training gradient from metric weighting.
- **Splits regress in proportion to amplitude** (single_in_dist +102% > re_rand +42% > cruise +36%) — diagnostic of the failure mode. High-amplitude splits suffer most because RevIN under-fits them most.
- **Target-metric (`val_re_rand`) regressed +42%** — even on its own predicted-best signal, the mechanism failed.
- RevIN paper assumed scale-invariant metric (relative RMSE, normalized error). For absolute physical-units MAE this approach is structurally incorrect.
- **Closed 2026-04-28. Next assignment: #917 Re-input noise augmentation** (smooth FiLM conditioning via training-time log(Re) perturbation).

### Side observation flagged
Askeladd noted `test_geom_camber_cruise/loss = inf` artifact in `evaluate_split` (single test batch with tiny per-sample y_surf_var amplifying residual). Same vol_loss accumulator NaN-safety gap fern flagged earlier. Doesn't affect MAE metrics but worth a future hardening pass.

---

## 2026-04-28 23:35 — PR #815 v2-on-l1 (MERGED): FiLM conditioning per-block on log(Re) + L1
- **Branch:** `willowpai2e3-thorfinn/film-re-conditioning`
- **Hypothesis:** FiLM (γ, β) per-block conditioning on log(Re), rebased onto post-#761 L1 advisor. Predicted stacking: L1 (loss shape) ⊥ FiLM (hidden-state Re modulation) → both gains compound.
- **Run (v2-on-l1):** W&B `mfjoux5g`, **14/14 epochs (clean finish)**, val_avg still falling at epoch 14 (new best at final epoch), peak 44.6 GB. +42.5K params (+6.4%).

| Split | L1 baseline `tirux1y1` | FiLM v2 `mfjoux5g` | Δ |
|---|---|---|---|
| `val_single_in_dist` | 109.65 | **95.54** | **−12.9%** |
| `val_geom_camber_rc` | 101.17 | **91.38** | **−9.7%** |
| `val_geom_camber_cruise` | 72.37 | **64.90** | **−10.3%** |
| `val_re_rand` | 87.33 | **79.26** | **−9.2%** |
| **val_avg** | **92.63** | **82.77** | **−10.6%** |
| **test_avg** | **82.83** | **72.27** | **−12.7%** |

### Decision: MERGED (2026-04-28) — new branch best
- **−10.6% val (92.63 → 82.77)** — beats advisor's predicted target of 89.85 by 7.9%. Every split improved.
- **−12.7% test (82.83 → 72.27)** — new test best.
- **All 4 val splits improved this time** — including `val_geom_camber_rc` (+9.7%), which regressed in v1b. v1b regression was single-seed noise or a Huber/L1 interaction; L1 removed it.
- `val_re_rand` −9.2%, `val_geom_camber_cruise` −10.3% — FiLM-targeted splits still show strongest relative gains (widest Re ranges), confirming the per-block Re-modulation mechanism.
- L1 + FiLM stack constructively as predicted (orthogonal mechanisms: loss shape vs hidden-state Re conditioning).
- `(1+γ)·h+β` zero-init worked perfectly — stable training throughout, FiLM started as identity.
- **New beat-threshold: val_avg < 82.77**
- **Follow-up assigned: #909** (pre-block FiLM: condition attention input rather than output; thorfinn).

---

## 2026-04-28 23:35 — PR #858 (CLOSED): Focal surface loss gamma=1.0 (L1 base)
- **Branch:** `willowpai2e3-nezuko/focal-surface-loss`
- **Hypothesis:** Apply focal-style node weighting `(per-node-err/mean-err)^gamma` on top of L1 surface loss to concentrate gradient on high-error surface nodes (stagnation, suction peak, leading-edge curvature). Predicted −2 to −7%.
- **Runs (v1-l1-gamma05 and v1-l1-gamma10):** W&B `6f8lwss4` (γ=0.5) and `ulobuh9d` (γ=1.0), 14/14 epochs both.

| gamma | val_avg/mae_surf_p | vs L1 baseline | conclusion |
|---|---|---|---|
| 1.0 | 105.06 | **+13.4%** | substantially worse |
| 0.5 | 92.13 | −0.5% | within noise |
| **L1 baseline (#761)** | **92.63** | — | reference |

Per-split val (gamma=0.5): single_in_dist=105.07, geom_camber_rc=100.80, cruise=74.90, re_rand=87.74.

### Decision: CLOSE — mechanism falsified
- **gamma=0.5 (92.13) is within single-seed noise** — not a reliable improvement. The −0.5% delta is well within the ±10% single-seed noise band.
- **gamma=1.0 is substantially worse** (+13.4%): ~10% of nodes get >2× weight, worst single node ~35×. Noisy minibatch gradients on nodes the model cannot yet drive to zero slows convergence.
- **Mechanism is incorrect at this budget**: high-error nodes are not gradient-bottlenecked; they are capacity/convergence-time-bottlenecked. Focal amplification makes convergence worse, not better.
- Student's own diagnosis nailed it: "amplifying their loss only makes that worse." Clean mechanistic falsification.
- **Additional decisive point:** PR #815 just merged at val_avg=82.77 — gamma=0.5's 92.13 is now +11.3% above new baseline.
- **Focal-weight distribution stats** (student measured): gamma=0.5 max=4.7×, p99=2.4×, frac>2×=2.3%; gamma=1.0 max=34.9×, p99=8.2×, frac>2×=10.3%. Stats confirm the tail is moderately concentrated but not gradient-starved.
- **Closed 2026-04-28. Next assignment: #910 Re-stratified batch sampling.**

---

## 2026-04-28 22:30 — PR #815 v1b: FiLM conditioning per-block on log(Re)
- **Branch:** `willowpai2e3-thorfinn/film-re-conditioning`
- **Hypothesis:** Add FiLM (γ, β) conditioning on log(Re) to each Transolver block — explicit per-layer regime adaptation across Reynolds ranges. Predicted −5 to −15% with biggest gains on Re-stratified and OOD-camber splits.
- **Run (v1b):** W&B `ujkwztbk`, **14/14 epochs (clean finish)**, val_avg still falling at epoch 14, peak 44.6 GB. +42.5K params (+6.4% over baseline). Pre-#761 branch (no L1).

| Split | Baseline `8cvp4x6r` | FiLM v1b | Δ |
|---|---|---|---|
| `val_single_in_dist` | 143.36 | 141.12 | −1.6% |
| `val_geom_camber_rc` | 124.20 | 132.08 | +6.3% |
| `val_geom_camber_cruise` | 109.42 | **93.92** | **−14.2%** |
| `val_re_rand` | 111.63 | **106.87** | **−4.3%** |
| **val_avg** | **122.15** | **118.50** | **−3.0%** |
| **test_avg** | NaN (pre-fix) | **107.76** | n/a |

### Decision: SEND BACK FOR REBASE (onto post-#761 advisor)
- **Hypothesis directionally confirmed.** The two splits FiLM should help most are exactly the ones that improved: `val_re_rand` (Re-stratified holdout) and `val_geom_camber_cruise` (widest Re range, 110K-5M). `−14.2%` on cruise is the largest single-split gain seen on this branch from any architecture experiment.
- **Absolute number doesn't beat new baseline (92.63).** Branch predates #761 L1 merge. v1b at 118.50 is +27.9% above current best, but FiLM is orthogonal to L1's loss-shape change — should stack.
- **Single split regressed** (`val_geom_camber_rc` +6.3% / +9.2% test). Single-seed in narrow-Re-band domain — likely noise but flagged for v2 follow-up.
- val curve still falling at epoch 14 — gain has more to give with rebase.
- `(1+γ)·h+β` zero-init worked: training stable through every epoch, FiLM started as identity.
- **Sent back:** rebase onto current advisor (which has L1), no code changes to FiLM itself, re-run with `--epochs 14`. Predicted: 92.63 × 0.97 ≈ 89.85 if mechanisms stack.

### Round-3 follow-ups queued (post-merge)
1. **Pre-block FiLM** — modulate hidden state *before* attention/mlp blocks (currently after).
2. **Layer-targeted FiLM** — only last 2 blocks; tests whether full per-block is necessary.
3. **Investigate `val_geom_camber_rc` regression** — multi-seed re-run of baseline + FiLM; could be real coupling issue or noise.

---

## 2026-04-28 22:25 — PR #743 v2: Per-channel surface loss [1.0, 0.5, 2.0] on Huber base
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Apply per-channel weights `[Ux=1.0, Uy=0.5, p=2.0]` to the surface loss to align training emphasis with the `mae_surf_p` metric. v1 was blocked by NaN poisoning; v2 adds the Huber base (matching PR #814) with normalized channel weighting.
- **Run:** W&B `2tj1e31r`, **14/14 epochs (clean)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 115.33 | 98.01 |
| `*_geom_camber_rc` | 109.49 | 99.81 |
| `*_geom_camber_cruise` | **77.86** | **66.60** |
| `*_re_rand` | **94.18** | **89.40** |
| **avg** | **99.21** | **88.45** |

### Decision: SEND BACK FOR REBASE (onto post-L1-merge advisor)
- **−3.80% val / −4.88% test vs Huber baseline (103.13 / 92.99)** — genuine compound gain over Huber.
- BUT: new baseline is tanjiro L1 (92.63 / 82.83). Alphonse's 99.21 is +7.1% WORSE than L1 alone.
- Channel weighting on top of Huber is confirmed to stack constructively. Natural v3 question: does it also compound on top of L1?
- Prediction: if same -3.8% stacks on L1 → 92.63 × 0.962 ≈ 89.1 val — would be a meaningful improvement.
- **Sent back** with instructions to: rebase onto post-#761 advisor; replace `F.huber_loss(reduction='none')` with per-element `abs_err * surf_chan_w` in both training loop and evaluate_split.
- New beat-threshold for v3: **val_avg < 92.63**.

---

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

## 2026-04-28 22:05 — PR #751: Dropout 0.1 + stochastic depth (regularization)
- **Branch:** `willowpai2e3-fern/dropout-stochastic-depth`
- **Hypothesis:** Add `dropout=0.1` (PhysicsAttention + post-MLP) and linear stochastic depth `0 → 0.1` across the 5 TransolverBlocks. Predicted −3 to −8% with bias toward `geom_camber` OOD splits. Round 1 — pre-#807, pre-#814.
- **Run:** W&B `w5vmv84w` (v1) and `pv21nz5x` (matched baseline), 14/50 epochs (timeout), peak 43.4 GB.

### Results — v1 vs fern's matched baseline (pre-Huber)

| Split | Baseline `pv21nz5x` (val) | v1 `w5vmv84w` (val) | Δ val | Baseline (test) | v1 (test) | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| `single_in_dist` | **163.29** | 182.97 | **+12.0%** | **137.53** | 159.05 | **+15.6%** |
| `geom_camber_rc` | 137.79 | **130.92** | **−5.0%** | 129.92 | **119.45** | **−8.0%** |
| `geom_camber_cruise` | 117.49 | **111.63** | **−5.0%** | 99.89 | **93.99** | **−5.9%** |
| `re_rand` | **128.30** | 129.72 | +1.1% | **127.65** | 128.90 | +1.0% |
| **avg** | **136.72** | **138.81** | **+1.5%** | **123.75** | **125.35** | **+1.3%** |

### Decision: SEND BACK FOR REBASE + WEAKEN REGULARIZATION
- Hypothesis directionally confirmed on the OOD axis: `val_geom_camber_*` improves 5%, `test_geom_camber_*` improves 5.9–8%. That's exactly where regularization should help.
- But the in-distribution `single_in_dist` track regresses 12–15.6% — `dropout=0.1, drop_path_max=0.1` is too strong for a 5-layer network at 1499-sample scale; stochastic depth dropping whole residual paths destabilizes the easy in-dist fit.
- vs current best (PR #814 Huber, 103.13): v1 is +34.6% worse — a regression at the absolute level, BUT branch predates both #807 and #814 so the comparison isn't apples-to-apples.
- **Sent back with feedback to:** (a) rebase onto current advisor (gets Huber baseline); (b) drop the redundant `evaluate_split` y-NaN guard (already covered by #807); (c) halve both regularizers — `dropout=0.05, drop_path_max=0.05`; (d) `--epochs 14` explicit so cosine completes.
- v2 tests whether mild regularization compounds with Huber surface loss or whether the OOD-generalization signal is already saturated by the loss-shape change (Huber alone got `test_geom_camber_cruise` from 99.89 → 69.84).

### Independent NaN-bug discovery (notable)
Fern is the **fourth** student to independently identify the `data/scoring.py` NaN-poisoning bug (askeladd PR #807, tanjiro #761 inline guard, nezuko #759 inline guard, fern #751 inline guard). Fern's `evaluate_split` y-finite-per-sample mask uses a different surface (filter samples vs filter elements), but the underlying diagnosis matches. Cleanly handled at the `data/scoring.py` layer in #807 (now merged), so all student inline guards become redundant on rebase.
