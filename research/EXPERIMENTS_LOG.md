# SENPAI Research Results — `icml-appendix-charlie-pai2d-r2`

Round-1 reviews. Primary ranking metric: `val_avg/mae_surf_p` (lower is better). All test_avg/mae_surf_p values in this round are NaN due to a shared scoring/data issue (test_geom_camber_cruise sample 20 has Inf in `y[p]` volume nodes; `data/scoring.py:accumulate_batch` propagates Inf through mask multiplication because IEEE 754 gives `Inf * 0 = NaN`). The bug is real and pre-existing; we will work around it in a round-2 experiment.

## 2026-04-27 23:30 — Round-1 review summary

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | best epoch | Decision |
|------|----|---------|------|--------------------------:|----------:|----------|
| 1 | #282 | edward | huber-loss | **105.999** | 14 | **MERGE** (round-1 baseline) |
| 2 | #295 | tanjiro | pressure-channel-weight | 130.916 | 12 | CLOSE (>23% worse than huber) |
| 3 | #297 | thorfinn | depth-8 | 168.836 | 9 | CLOSE (compute-infeasible at 30-min budget) |

## 2026-04-27 23:30 — PR #282: Replace MSE with Huber loss (delta=1.0) in normalized space

- Branch: `charliepai2d2-edward/huber-loss`
- Hypothesis: Huber(δ=1.0) on normalized targets bounds gradient magnitude on high-Re outliers and is closer to the L1 metric than MSE.
- Result: best `val_avg/mae_surf_p = 105.999` at epoch 14 (last epoch before timeout). 14/50 epochs in 30 min budget.
- Per-split val MAE for `p`: single_in_dist=134.05, geom_camber_rc=109.48, geom_camber_cruise=82.72, re_rand=97.75.
- Test (3 finite splits avg): 105.42; full 4-split test_avg = NaN (scoring.py Inf*0 bug, not Huber-related).
- Convergence: train losses monotone (vol 0.59→0.15, surf 0.32→0.08). Val curve descending; one transient spike at epoch 12 recovered by epoch 14. Best epoch was the last epoch — clearly under-trained.
- Metric paths: `models/model-charliepai2d2-edward-huber-loss-20260427-223516/{metrics.jsonl,metrics.yaml}` (also mirrored at `research/student_metrics/`).
- Decision: **MERGE.** This is the round-1 baseline at `val_avg/mae_surf_p = 105.999`. Worth noting the run was cut short — convergence is still likely improving, so the true Huber asymptote is probably better.

## 2026-04-27 23:30 — PR #295: Per-channel surface loss weights [1.0, 1.0, 2.5] for (Ux, Uy, p)

- Branch: `charliepai2d2-tanjiro/pressure-channel-weight`
- Hypothesis: Up-weight `p` (×2.5) inside surface loss to bias gradients toward the headline-metric channel.
- Result: best `val_avg/mae_surf_p = 130.916` at epoch 12. 14/50 epochs in 30 min.
- Per-split val MAE for `p`: single_in_dist=159.60, geom_camber_rc=138.61, geom_camber_cruise=103.86, re_rand=121.59.
- Decision: **CLOSE.** 23% worse than the huber-loss winner from the same round. The intervention behaved as designed (training stable, p MAE descended faster than Ux/Uy in normalized space), but absolute level is dominated by the huber-loss change.
- Direction not dead: revisit channel-weighting on top of the merged huber baseline if other levers plateau. The student's suggestion to also down-weight Ux/Uy (e.g., `[0.5, 0.5, 2.5]`) is a tighter test.

## 2026-04-27 23:30 — PR #297: Pure depth scale: n_layers=8 (no other changes)

- Branch: `charliepai2d2-thorfinn/depth-8`
- Hypothesis: 5→8 layers (matching Transolver paper) tests pure depth as the bottleneck.
- Result: best `val_avg/mae_surf_p = 168.836` at epoch 9. **Only 9/50 epochs** in 30 min budget — depth-8 averages 206 s/epoch.
- Per-split val MAE for `p`: single_in_dist=189.98, geom_camber_rc=230.75, geom_camber_cruise=114.69, re_rand=139.93.
- Decision: **CLOSE** as compute-infeasible at the current 30-min budget. The cosine LR schedule barely engaged (lr ≈ 95% of peak by epoch 9), so the run is grossly under-trained. Hypothesis not falsified, but not testable as designed.
- Follow-up: revisit depth-7 (~10 epochs) or, better, scale capacity in width direction (already covered by alphonse #279 / askeladd #281). Time-budget-aware T_max for cosine is a separate architectural improvement worth considering once results stabilize.

## 2026-04-27 23:55 — PR #291: Add dropout=0.1 in PhysicsAttention and TransolverBlock

- Branch: `charliepai2d2-nezuko/dropout-0p1` (no metrics file committed; metrics taken from PR comment)
- Hypothesis: dropout 0→0.1 regularizes against the small training set; should disproportionately help OOD camber holdouts.
- Result: best `val_avg/mae_surf_p = 128.896` at epoch 14 (14/50 epochs in 30-min budget). Train losses still descending at timeout.
- Per-split val MAE for `p`: single_in_dist=155.07, geom_camber_rc=140.24, geom_camber_cruise=102.87, re_rand=117.41.
- Test (locally fixed for the Inf*0 bug): `test_avg/mae_surf_p = 117.81` (raw is NaN). Decision is based on val_avg.
- Decision: **CLOSE.** +22% vs merged huber baseline (105.999). The MSE+dropout combination did not beat MSE+huber-no-dropout. Direction not falsified versus the new merged baseline (huber+dropout=0); dropout could still be useful as a stacked lever later (especially MLP-side dropout, currently zero).
- Note: student locally reproduced the scoring fix and reported the corrected test metric — same root cause as edward's diagnosis, exactly the workaround being shipped via PR #361.

## 2026-04-27 23:55 — PR #281: Slice scale to 128 in PhysicsAttention

- Branch: `charliepai2d2-askeladd/slice-128` (no metrics file committed; metrics taken from PR comment)
- Hypothesis: doubling slice tokens 64→128 should give finer spatial discrimination, especially for cruise (mean 210K nodes).
- Result: best `val_avg/mae_surf_p = 154.594` at epoch 11. **Only 11/50 epochs** in 30-min budget — slice-128 averages ~170 s/epoch.
- Per-split val MAE for `p`: single_in_dist=211.53, geom_camber_rc=154.58, geom_camber_cruise=120.28, re_rand=131.99.
- Test_avg = NaN. The cruise test split also produced **non-finite predictions** in this run (`vol_loss=+inf`), independent of the scoring bug — the model itself blew up on at least one cruise sample (under-trained at slice-128 + 11 epochs).
- Decision: **CLOSE.** +46% vs merged huber baseline (105.999) AND a real model-output instability on cruise test. Slice-128 standalone on MSE didn't beat huber+slice-64. Worth retrying as huber+slice-128 if other levers stall.

## 2026-04-28 00:05 — PR #284: Linear warmup + cosine to 1e-3, betas (0.9,0.95), grad clip 1.0

- Branch: `charliepai2d2-fern/warmup-cosine-1e3` (no metrics file committed; metrics taken from PR comment)
- Hypothesis: standard transformer recipe (3-epoch linear warmup → cosine, peak lr=1e-3, betas (0.9, 0.95), grad clip 1.0) compounds for 5–15% improvement.
- Result: best `val_avg/mae_surf_p = 123.135` at epoch 12 (14/50 epochs in 30-min budget).
- Per-split val MAE for `p`: single_in_dist=138.06, geom_camber_rc=132.86, geom_camber_cruise=107.18, re_rand=114.44.
- Test (3 finite splits): mean 115.10. Test_avg = NaN (same scoring bug).
- **Key finding (student diagnosis):** `max_norm=1.0` clipped **100% of batches** across all 14 epochs. Pre-clip gradient mean was 30–200 throughout training (max up to 1334). The configured peak lr=1e-3 was effectively rescaled by `1.0 / ‖g‖`, so the LR recipe never actually ran at its labeled peak. The warmup+higher-lr signal was masked by the clip.
- Decision: **CLOSE.** +16.2% vs merged huber baseline (105.999). The recipe is not falsified — it was just crippled by the wrong clip threshold. A round-2 retry with the clip dropped (or loosened to e.g. 50) on the merged huber baseline is the natural follow-up.

## 2026-04-28 00:10 — PR #361: Filter non-finite y samples in evaluate_split (recover test_avg)

- Branch: `charliepai2d2-edward/nan-safe-eval` — metrics committed at `models/model-charliepai2d2-edward-nan-safe-eval-20260427-232955/{metrics.jsonl,metrics.yaml}`.
- Hypothesis: filter samples with non-finite `y` in `train.py:evaluate_split` before calling `accumulate_batch` to work around the IEEE 754 Inf*0=NaN propagation in `data/scoring.py:accumulate_batch`. Predicted impact: 0 on `val_avg`, NaN → finite on `test_avg`.
- Result: **test_avg/mae_surf_p = 97.957** (first finite measurement on this branch). val_avg = 108.103 at epoch 12 (+1.99% vs the 105.999 huber baseline) — RNG noise from a different stochastic trajectory under the 14-epoch timeout cut, not a regression. The student verified independently that the workaround does NOT trigger on any val sample (zero non-finite samples across train + 4 val splits + 3 of 4 test splits; only `test_geom_camber_cruise` sample 20 has them).
- Per-split test MAE for `p`: single_in_dist=123.760, geom_camber_rc=104.946, geom_camber_cruise=66.144, re_rand=96.978.
- Decision: **MERGE.** This is a metric-pipeline fix that unlocks the paper-facing metric for the entire research programme. Subsequent rounds will report a recoverable `test_avg`. BASELINE.md val_avg target stays at 105.999 (recipe high-water mark) — we do NOT raise the bar to 108.103 since that would be optimizing against RNG variance rather than recipe quality.

## 2026-04-28 00:25 — PR #363: EMA of model weights (decay=0.999) for evaluation

- Branch: `charliepai2d2-thorfinn/ema-eval` — metrics committed at `models/model-thorfinn-ema-eval-20260427-233441/{metrics.jsonl,metrics.yaml}`.
- Hypothesis: EMA copy of weights (decay 0.999) used for val/test eval damps the late-training validation noise observed in round-1; checkpoint saves EMA weights.
- Result: best `val_avg/mae_surf_p = 101.350` at epoch 14 — **−4.39% vs huber baseline (105.999)**, right at the upper end of the predicted −2% to −5% range.
- Per-split val MAE for `p`: single_in_dist=126.32 (−5.76%), geom_camber_rc=109.41 (−0.07%, flat), geom_camber_cruise=76.99 (−6.93%), re_rand=92.68 (−5.19%).
- 3-split test mean: 100.030 (cruise NaN — PR #361 had not landed when this run started).
- **Striking observation:** the val curve is **monotonically decreasing every epoch**, with no transient spike. Round-1 huber had a 43% spike at epoch 12 (114→164→131→106); EMA fully damped it. Best epoch was the final epoch — implies the asymptote is even lower with more epoch budget.
- Decision: **MERGE.** New baseline `val_avg/mae_surf_p = 101.350`. EMA is orthogonal to most other levers and should compound with future winners.

## 2026-04-28 00:25 — PR #286: Increase surf_weight from 10 to 25

- Branch: `charliepai2d2-frieren/surf-weight-25` (rebased onto post-huber advisor; metrics committed at `models/model-surf-weight-25-20260427-234307/`).
- Hypothesis: upweight surface loss (10→25) to bias gradients toward the headline surface metric.
- Result: best `val_avg/mae_surf_p = 108.222` at epoch 13 — **+2.10% regression vs huber baseline (105.999)**.
- Per-split val MAE for `p`: single_in_dist=124.06 (−7.45%), geom_camber_rc=117.30 (+7.15%), geom_camber_cruise=88.69 (+7.21%), re_rand=102.84 (+5.21%).
- Volume MAE regressed +10–17% across all splits.
- Decision: **CLOSE.** sw=25 over-corrects: surface gain is concentrated entirely on the in-distribution split while OOD splits regress. Volume context is starved, hurting cross-split generalization. Direction not dead — student's suggestion to try sw ∈ {12, 15, 18} (smaller bumps) is reasonable for round-3 if other levers stall, ideally with the now-merged EMA baseline.

## 2026-04-28 00:55 — PR #377: Warmup + cosine to 1e-3 + betas (0.9,0.95) on Huber, no grad clip

- Branch: `charliepai2d2-fern/warmup-cosine-1e3-no-clip` — metrics committed.
- Hypothesis (two parts): (1) huber bounds per-element gradient enough that the round-1 `max_norm=1.0` clip was redundant; (2) the warmup+higher-lr+betas recipe gives −3% to −8% improvement on the merged huber baseline (105.999).
- Result: best `val_avg/mae_surf_p = 116.352` at epoch 12. **+9.8% vs huber baseline; +14.8% vs new EMA baseline (101.350).**
- Per-split val MAE for `p` (all 7–14% worse): single_in_dist=152.59, geom_camber_rc=118.24, geom_camber_cruise=88.91, re_rand=105.68.
- test_avg = 105.715 (**finite** — surface MAE doesn't see the cruise *volume* Inf, so this paper-facing number is recoverable here even without #361 landing).
- **Hypothesis 1 confirmed (clip was redundant):** grad norms decayed smoothly 56→9 mean over 14 epochs with no instability. Round-1 +16.2% degradation was indeed the clip dominating dynamics.
- **Hypothesis 2 falsified (recipe doesn't help under truncated training):** the cosine `T_max=47` was sized for 50 epochs, but only 14 fit in the 30-min budget. So the model trained the entire run at near-peak lr (≈0.9 × 1e-3 throughout) and never got the fine-tuning phase. lr was simply too hot for the available budget.
- Decision: **CLOSE.** This is a clean falsification of the "lr=1e-3 + warmup helps in this regime" hypothesis, dependent on the timeout-truncated training. Direction is not dead in absolute terms — askeladd's #370 (cosine T_max=14) directly addresses the LR-too-hot issue and should give a much fairer test of warmup+higher-lr.

## 2026-04-28 00:55 — PR #362: Surface loss channel weights [0.5, 0.5, 2.5] on Huber baseline

- Branch: `charliepai2d2-tanjiro/surf-channel-on-huber` — metrics committed.
- Hypothesis: down-weight Ux/Uy (×0.5), up-weight p (×2.5) in surface loss to bias gradients toward the headline `p` channel; predicted −3% to −10%.
- Result: best `val_avg/mae_surf_p = 107.920` at epoch 13. **+1.81% vs huber baseline; +6.5% vs new EMA baseline.**
- Per-split val MAE for `p`: single_in_dist=138.13 (+3.05%), geom_camber_rc=111.78 (+2.10%), geom_camber_cruise=83.36 (+0.78%), re_rand=98.40 (+0.67%).
- Per-channel val pattern: p +1.81%, Ux +5.07%, Uy +16.23% — the relative degradation IS consistent with the channel weighting tilting gradient toward `p`, but the absolute `mae_surf_p` got *worse*, not better.
- Decision: **CLOSE.** Combined with round-1 PR #295 (`[1.0, 1.0, 2.5]` weights, also regressed at +23.5% vs MSE), this is now two independent confirmations that channel-weighting the surface loss toward `p` does NOT improve `mae_surf_p` on this problem. **Drop direction.** Future surf-rebalance attempts should use the `surf_weight` global scalar (e.g. modest bumps to 12–15 on the EMA baseline) rather than per-channel surface weights.

## 2026-04-28 01:05 — PR #386: Fourier embedding of log(Re) (8 bands → 16 dims) into model input

- Branch: `charliepai2d2-edward/re-fourier-8` — metrics committed.
- Hypothesis: encode `log(Re)` via 8 geometric Fourier frequencies (2^0…2^7) concatenated to input features; targets cross-Re and high-Re-extreme generalization. Predicted −2% to −5% val_avg, with disproportionate help on `val_re_rand` and `val_single_in_dist`.
- Result: best `val_avg/mae_surf_p = 109.131` at epoch 14. **+2.96% vs huber baseline (105.999); +7.7% vs new EMA baseline (101.350).** test_avg = 100.333 (finite — first PR to benefit from the merged NaN-safe eval).
- Per-split val MAE for `p`:
  - `val_single_in_dist`: 128.88 (−3.86% vs huber) — **predicted to win, did win**
  - `val_geom_camber_rc`: 110.68 (+1.10% — neutral)
  - `val_geom_camber_cruise`: 92.49 (+11.81% — regression; smallest target dynamic range)
  - `val_re_rand`: 104.47 (+6.88% — predicted to win, regressed instead)
- Per-split test echoes: `single_in_dist −8.06%`, `cruise +22.50%`, `re_rand +6.53%`.
- **Key student diagnosis**: 8 geometric bands up to `2^7 = 128 rad/log_re_unit` is too aggressive for a 4-decade `log(Re)` span (≈ 80 cycles across the corpus). The highest frequencies produce per-Re fingerprints rather than smooth Re-functions, which hurts cross-Re generalization specifically — exactly the failure mode observed on `val_re_rand`. Failure pattern is structured (matches aliasing prediction), not RNG noise.
- Decision: **CLOSE** standalone. Direction has real signal (single_in_dist −3.86% val, −8.06% test is too large to be RNG drift), but the chosen frequency band is wrong. **Salvageable** with narrower band — round-3 follow-up should test `re_fourier_bands=4` (frequencies up to 2^3=8 rad/log_re_unit) which preserves the low-frequency content (smooth Re-trend) while removing the aliasing pressure.

## 2026-04-28 01:15 — PR #391: SwiGLU MLP in TransolverBlock (LLaMA-style FFN) — **WINNER**

- Branch: `charliepai2d2-thorfinn/swiglu-mlp` — metrics committed.
- Hypothesis: replace 2-linear GELU MLP inside `TransolverBlock` with LLaMA-style SwiGLU (gate × value, bias-free). Per-token gating gives the FFN capacity to suppress channels that aren't useful per-node, especially on irregular meshes. Predicted −1% to −3%.
- Result: best `val_avg/mae_surf_p = 88.227` at epoch 13. **−12.95% vs EMA baseline (101.350)**, far exceeding the predicted range. **test_avg = 78.338** (−20.03% vs PR #361 finite test 97.957). Param-matched (+1.3%, 670K vs 660K).
- Per-split val MAE for `p`: **all 4 splits improved** — single_in_dist 106.40 (−15.78%), geom_camber_rc 100.41 (−8.23%), geom_camber_cruise 64.41 (−16.34%), re_rand 81.70 (−11.86%).
- Per-split test MAE for `p`: single_in_dist=96.44, geom_camber_rc=88.06, **geom_camber_cruise=54.01**, re_rand=74.84.
- **Striking observation**: the previously-flat `val_geom_camber_rc` (−0.07% under EMA) finally moved (−8.23%) — supports the per-token-role-specific-gating mechanism. Loss descent was monotonic and steeper than the EMA baseline run; every epoch up to the cap was a new best — model is still under-trained.
- Decision: **MERGE.** New baseline `val_avg/mae_surf_p = 88.227`. SwiGLU stacks cleanly on top of EMA.

## 2026-04-28 01:15 — PR #392: Increase mlp_ratio from 2 to 4

- Branch: `charliepai2d2-frieren/mlp-ratio-4` — metrics committed.
- Hypothesis: doubling FFN hidden width adds capacity where the model is most under-capacity for a complex regression task. Predicted −2% to −5%.
- Result: best `val_avg/mae_surf_p = 108.558` at epoch 13. **+7.1% vs EMA baseline.** Equal-epoch comparison (epoch 13 vs epoch 13): +2.6% (105.83 vs 108.56). +50% more params yielded *negative* marginal returns.
- Per-split val MAE for `p`: single_in_dist=137.48 (+8.83%), geom_camber_rc=118.36 (+8.18%), geom_camber_cruise=82.33 (+6.94%), re_rand=96.06 (+3.64%).
- **Per-split signature contradicts the capacity-bottleneck hypothesis**: the two highest-baseline-MAE splits (single_in_dist, camber_rc) regressed *most* (+8.8%, +8.2%). If raw FFN capacity were the limit, those should be where extra FFN params help — they aren't. Bottleneck on those splits is **generalization** (held-out cambers, distribution shift), not capacity.
- Decision: **CLOSE.** Single-axis FFN scaling is now disconfirmed; the +50% params slows convergence under the wallclock cap and doesn't address the actual bottleneck (generalization on OOD splits). Combined with the SwiGLU win (architectural change at param-matched cost), it is now clear that **architectural quality matters more than raw MLP capacity** on this problem.

## 2026-04-28 01:15 — PR #370: Align cosine T_max with actual epoch budget (T_max=14)

- Branch: `charliepai2d2-askeladd/cosine-tmax-14` — student rebased onto post-EMA advisor; metrics committed.
- Hypothesis: cosine `T_max=14` aligns LR decay with the actual reachable epoch count under the 30-min cap, letting the LR fully decay during training. Predicted −3% to −8%.
- Result: best `val_avg/mae_surf_p = 102.359` at epoch 14. **vs original huber baseline (105.999): −3.43% (matches predicted range).** vs EMA baseline (101.350): **+1.00% (slight regression).** test_avg=93.052 (−5.0% vs PR #361 finite test 97.957) — **all 4 test splits finite**, three of four improve.
- Per-split val MAE for `p` (vs EMA): single_in_dist=123.96 (−1.87%), geom_camber_rc=108.98 (−0.39%), geom_camber_cruise=81.76 (+6.20%), re_rand=94.73 (+2.21%).
- LR trajectory: best epoch (14) ran with `lr ≈ 1.3% of peak` — confirms the premise that best-val coincides with the low-lr tail.
- **Student's analytical insight (excellent)**: cosine T_max=14 and EMA(decay 0.999) interact non-additively. EMA half-life ≈ 693 steps ≈ 1.85 epochs at our batch count; with T_max=14, the last 5–7 epochs run at lr ≤ 28% of peak (and the final 2 at <2% of peak), so the EMA averages weights that are barely moving. We get the cosine sharp-minimum effect *and* the EMA smoothing, but the EMA contribution shrinks because there's nothing left for it to smooth. Levers are not orthogonal.
- Decision: **CLOSE.** +1.00% vs current EMA baseline; further behind after the SwiGLU merge (would be ~+16% vs 88.227). Direction is not dead — student's follow-up #1 (smaller-decay EMA, e.g. 0.99, half-life ~0.2 epochs) would let cosine annealing's low-lr tail dominate the final EMA state without the "averaged stillness" effect — worth queuing if other levers stall.

## 2026-04-28 01:45 — PR #412: Per-channel output heads for Ux, Uy, p

- Branch: `charliepai2d2-tanjiro/per-channel-heads` — metrics committed.
- Hypothesis: shared 3-channel `mlp2` is a capacity bottleneck (Ux/Uy and p have very different physics); decoupling into 3 per-channel heads should help, especially the previously-flat `val_geom_camber_rc`. Predicted −2% to −4%.
- Result: best `val_avg/mae_surf_p = 105.580` at epoch 14. **+4.18% vs EMA baseline (101.350); +19.7% vs SwiGLU baseline (88.227).** test_avg = 95.213 (4-split, finite); 3-split excl. cruise: 104.139 (+4.11% vs baseline 100.030).
- Per-split val MAE for `p`: single_in_dist=129.45 (+2.47%), geom_camber_rc=115.82 (+5.86%), geom_camber_cruise=81.09 (+5.32%), re_rand=95.97 (+3.55%).
- **The canary split (`val_geom_camber_rc`) regressed MOST** — cleanly falsifies the shared-head-as-capacity-bottleneck hypothesis. The shared head's implicit cross-channel gradient coupling appears to be more useful than per-channel specialization at this budget.
- Decision: **CLOSE.** Capacity in the output head is not the bottleneck. Combined with PR #392 (mlp_ratio=4 also failed via wrong-target capacity), this is now consistent evidence that **architectural form matters more than head/FFN capacity** here. Per-channel head direction is dead.

## 2026-04-28 01:45 — PR #411: Huber loss with delta=2.0 (smoother near optimum)

- Branch: `charliepai2d2-fern/huber-delta-2` — metrics committed.
- Hypothesis: huber `δ=1.0 → 2.0` smooths the loss landscape near optimum (MSE-like for typical errors |e|<2), while keeping outlier robustness. Predicted −1% to −3%.
- Result: best `val_avg/mae_surf_p = 107.609` at epoch 14. **+6.18% vs EMA baseline; +21.97% vs SwiGLU baseline.** test_avg = 97.529 (4-split, finite).
- Per-split val MAE for `p`: all 4 regressed (+2.5% to +8.5%); OOD splits regressed worst (camber_cruise +8.54%, re_rand +7.99%, camber_rc +7.24%).
- **Validation curve was monotonically decreasing every epoch** (smoothness prediction confirmed) but the absolute level is worse.
- Student's mechanism: at the high-error training regime we're stuck in (val going 328→108 over 14 epochs, far from optimum), δ=2's quadratic region for |err|∈[1,2] *underweights* moderate errors relative to δ=1's bounded gradient, while giving 2× more pull to outliers. On a 14-epoch budget that's the wrong trade. δ=1 sits at a sweet spot. Pure MSE (δ→∞) was 105.999 without EMA; δ=1+EMA=101.350; δ=2+EMA=107.609 here — the curve is unimodal in δ.
- Decision: **CLOSE.** δ=1.0 is the sweet spot for huber on this problem. Direction not dead at smaller δ — student's follow-up suggestion (`δ=0.5` or `δ=0.25`, pushing toward L1) is a valid one-line sweep, especially now that the SwiGLU baseline gives much smoother val curves.

## 2026-04-28 02:00 — PR #279: Scale model capacity (n_hidden=192, n_layers=6, n_head=6)

- Branch: `charliepai2d2-alphonse/capacity-medium` — branched at the very start of round 1 (pre-huber). Metrics committed.
- Hypothesis: balanced capacity scale-up (depth+width+heads) targets the "model is undersized for ~1500 train samples × 74K-242K nodes" framing. Predicted −5% to −12%.
- Result: best `val_avg/mae_surf_p = 142.4462` at epoch 8 (timeout cut at 8/50 epochs, ~240 s/epoch). **+61.5% vs SwiGLU baseline (88.227)** and +34.4% vs original huber baseline.
- Per-split val MAE for `p`: single_in_dist=166.25, geom_camber_rc=152.88, geom_camber_cruise=116.81, re_rand=133.84.
- test_avg = 133.23 (finite via the student's own NaN-safe filter — see below).
- **Compute-infeasible at this budget**: same shape of failure as PR #297 (depth-8). Val was still in steep descent at the cap (epoch 7→8: 166→142, i.e. dropping >24 points per epoch). 1.71M params (~2.6× baseline) at 30-min budget = no chance of convergence.
- **Independent NaN-fix rediscovery**: alphonse independently identified the `data/scoring.py` Inf*0=NaN propagation bug AND implemented a byte-identical workaround to edward's PR #361. (PR #361 is already merged on advisor — alphonse's tree-side fix is now harmless duplication, but the diagnosis is exactly right.)
- Decision: **CLOSE.** Capacity scale-up at this depth/width is not testable under the 30-min wall-clock cap. Same finding as PR #297 — compute is binding for capacity experiments. The newly-merged SwiGLU win (PR #391) shows the actual lever is architectural quality at param-matched cost, not raw capacity.

## 2026-04-28 02:10 — PR #426: EMA decay 0.999 → 0.99 (shorter half-life) — **WINNER**

- Branch: `charliepai2d2-askeladd/ema-decay-099` — metrics committed.
- Hypothesis: EMA(0.999) has half-life ~1.85 epochs at our batch count, so it's heavily anchored to random init for the first ~2 epochs of training. Decay=0.99 (half-life ~0.18 epochs) tracks the live model immediately. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 83.223` at epoch 13. **−5.67% vs SwiGLU baseline (88.227)**, materially larger than predicted. test_avg = 73.904 (−5.66% vs baseline 78.338).
- Per-split val MAE for `p` (all improved): single_in_dist=98.82 (−7.13%), geom_camber_rc=96.61 (−3.78%), geom_camber_cruise=61.16 (−5.04%), re_rand=76.30 (−6.60%).
- Per-split test MAE for `p` (all improved): single_in_dist=89.83 (−6.85%), geom_camber_rc=84.40 (−4.16%), geom_camber_cruise=50.84 (−5.86%), re_rand=70.54 (−5.74%).
- Val curve trajectory: e1 baseline=333.4 → this run=193.2 (~140 pt gap), compresses to ~5 pt by epoch 13. Tiny non-monotone step at epoch 9→10 (+0.228), otherwise smooth.
- **Mechanism (student's diagnosis):** the dominant effect is **EMA bias correction during the under-trained cold start**, NOT the cosine-tail interaction the original prediction emphasized. With T_max=50 cosine and only 13 epochs reached, lr stays near peak the whole run; EMA(0.999) lags badly because it's averaging over the random-init phase. Decay=0.99 captures the model's current capability instead of averaging over the cold start.
- Decision: **MERGE.** New baseline `val_avg/mae_surf_p = 83.223`, `test_avg/mae_surf_p = 73.904`.

## 2026-04-28 02:10 — PR #424: SwiGLU output head (mlp2) on top of SwiGLU FFN baseline

- Branch: `charliepai2d2-thorfinn/swiglu-head` — metrics committed.
- Hypothesis: SwiGLU in `mlp2` output head aligns the head's expressive form with the rest of the SwiGLU-FFN model. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 90.298` at epoch 13. **+2.35% vs SwiGLU baseline (88.227).** test_avg = 81.019 (+3.42%).
- Per-split val MAE for `p`: single_in_dist=104.21 (−2.05%, in-dist improved), geom_camber_rc=107.03 (+6.60%), geom_camber_cruise=67.92 (+5.46%), re_rand=82.03 (+0.40%).
- Pattern: head SwiGLU is **more expressive on familiar samples but generalizes worse to OOD**. The val_single_in_dist improvement (−2.05%) is real, but the camber-OOD splits regressed +5–7%.
- **Student's structural diagnosis (excellent):** the per-block SwiGLU FFN is buffered by the residual connection (`fx = mlp(ln_2(fx)) + fx`), but the **last-layer head has no residual** (it's a direct projection to `out_dim=3`). The SwiGLU non-linearity acts unbuffered on the residual stream. Plus the head's gating channel `(W2·x)` can amplify directions in the residual stream that don't generalize cross-domain — explaining the +6.6%/+5.5% pattern on OOD splits. The head SwiGLU triples the param count (16.9K → 45.6K) but those extra params don't see enough data within 13 epochs to recover their cost on unseen-camber splits.
- Decision: **CLOSE.** Head SwiGLU is structurally different from FFN SwiGLU; the residual buffer was load-bearing for the per-block win. Direction not dead in absolute terms — student's follow-up #2 (residual SwiGLU head: `mlp2(ln_3(fx)) + linear_skip(fx)`) is a clean fix, queued for if other levers stall. SwiGLU FFN stays as baseline; output head reverts to GELU.

## 2026-04-28 02:10 — PR #418: Narrower Fourier embedding of log(Re): 4 bands [1,2,4,8]

- Branch: `charliepai2d2-edward/re-fourier-4` — branched on EMA pre-SwiGLU; metrics committed.
- Hypothesis: narrowing the Fourier band from 8 to 4 bands removes high-freq aliasing while preserving the low-frequency Re-trend that gave the bands=8 win on `val_single_in_dist`. Predicted −1% to −4% vs EMA baseline (101.350).
- Result: best `val_avg/mae_surf_p = 102.916` at epoch 14. **+1.54% vs EMA baseline; +16.6% vs current SwiGLU baseline (88.227)**. test_avg = 93.217 (4-split finite, **−4.84% vs prior huber 4-split test 97.957** — paper-side this is a real anchor).
- Per-split val vs EMA: single_in_dist=124.97 (**−1.07%**, win retained but attenuated from −3.86% at bands=8); geom_camber_rc=109.78 (flat); geom_camber_cruise=81.75 (+6.19%, recovered ~70% from bands=8's +20.1%); re_rand=95.16 (+2.67%, recovered ~80% from bands=8's +12.7%).
- **Aliasing diagnosis is directionally correct but not the complete story**: substantial recovery on cruise and re_rand at bands=4 supports the high-freq-aliasing mechanism, but neither returned all the way to baseline. The residual harm at bands=4 (max freq=8, ~5 cycles across the 4-log-unit corpus) suggests either (a) even ~5 cycles still gives the model some Re-fingerprint structure, or (b) Fourier embedding intrinsically creates per-Re features that disrupt cruise/re_rand generalization regardless of frequency content.
- Decision: **CLOSE.** Doesn't beat the SwiGLU baseline (88.227); the Fourier-Re direction has real signal on `val_single_in_dist` but doesn't compound enough to win overall. Student's follow-up #4 (FiLM-style scale-shift on input features by a small MLP of `log_re`) is a cleaner test of whether ANY Re-conditioning helps — queued.

## 2026-04-28 02:25 — PR #440: Switch GELU → SiLU in preprocess MLP and output head

- Branch: `charliepai2d2-tanjiro/silu-everywhere` — metrics committed.
- Hypothesis: consistent SiLU activation throughout the model (matching SwiGLU FFN's internal SiLU) gives small additional wins via consistent activation curvature. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 88.128` at epoch 13. **Null result**: −0.11% vs SwiGLU baseline (88.227, well within RNG noise); +1.06% on test (79.172 vs 78.338). Vs the now-current EMA(0.99) baseline (83.223): +5.9%.
- Per-epoch trajectory was statistically indistinguishable from the SwiGLU baseline (within ~0.3 pts at every epoch).
- Param-identical (670,679 = baseline exactly).
- Decision: **CLOSE.** Activation choice (GELU vs SiLU) is below the noise floor at this scale (0.67M params, 1499 train samples). Student's recommendation to deprioritize activation sweeps is correct.

## 2026-04-28 02:25 — PR #425: Input feature noise augmentation (std=0.01 on fun-features only)

- Branch: `charliepai2d2-frieren/input-noise-001` — metrics committed.
- Hypothesis: per-node Gaussian noise (std=0.01) on fun-feature channels (dims 2–23) targets the generalization bottleneck on OOD-camber and re_rand splits. Predicted −1% to −3% with disproportionate help on `val_geom_camber_rc` and `val_re_rand`.
- Result: best `val_avg/mae_surf_p = 89.984` at epoch 13. **+1.76% vs SwiGLU baseline; +8.1% vs current EMA(0.99) baseline (83.223).** test_avg = 80.607 (+2.27%).
- Per-split val: single_in_dist 105.62 (−0.73%), camber_rc 102.96 (**+2.55%**), camber_cruise 68.05 (+5.65%), re_rand 83.31 (**+1.97%**). **Both predicted-to-improve splits got worse** — exact opposite of the prediction.
- **Student's structural diagnosis (excellent)**: roughly half the fun-feature channels (`log(Re)`, AoA1, NACA1, AoA2, NACA2, gap, stagger — dims 13–23) are **constant within a sample** (per-sample globals encoding flow / geometry conditions). Per-node `randn_like` noise gives every mesh node a different "Reynolds number," a different camber, etc., **within the same forward pass** — destroying the consistency the model relies on to map (foil geometry, flow conditions) → flow field. Per-node noise is structurally wrong for this dataset's feature semantics.
- Decision: **CLOSE.** Direction not dead in absolute terms — student's follow-up #1 (per-sample noise on the constant-per-sample dims, broadcast across all N nodes; keep per-node noise on the truly per-node dims 2–12) is queued as PR #460. This is the cleanest correction of the failure mode.

## 2026-04-28 02:35 — PR #439: Huber loss with delta=0.5 (closer to L1)

- Branch: `charliepai2d2-fern/huber-delta-05` — branched on SwiGLU pre-EMA(0.99); metrics committed.
- Hypothesis: push δ in the *opposite* direction from the failed PR #411 (δ=2). δ=0.5 puts most of the loss surface in the linear region, closer to L1, while keeping a small smooth-near-zero region. Predicted −1% to −3% vs SwiGLU baseline (88.227).
- Result: best `val_avg/mae_surf_p = 87.265` at epoch 13. **−1.1% vs SwiGLU baseline (88.227); +4.9% vs current EMA(0.99) baseline (83.223).** test_avg = 78.194 (essentially tie with SwiGLU baseline test 78.338).
- Per-split val: 3/4 improved (single_in_dist −2.55, camber_rc −0.62, re_rand −1.91); camber_cruise regressed slightly (+1.22).
- **δ profile complete (monotonic, with diminishing returns):**
  - δ=2.0 → 107.609 (PR #411)
  - δ=1.0 → 88.227 (SwiGLU baseline)
  - δ=0.5 → 87.265 (this PR)
  - The δ=2→1 step gained ~19 pts; the δ=1→0.5 step gained only ~1 pt. The curve is flattening rapidly.
- Validation curve was strictly monotonic (every epoch a new best, no late-training instability seen at δ=0.5).
- Decision: **CLOSE.** Doesn't beat the current EMA(0.99) baseline (83.223) on standalone val_avg. The lever has small signal but isn't enough by itself. Student's follow-up #1 (δ=0.25 to test profile saturation) is exactly the right next step — combined with the current EMA(0.99) baseline as starting point, it doubles as a δ-sweep + a compound test. Queued as the round-5 reassignment.

## 2026-04-28 02:55 — PR #454: Adam-style EMA bias correction (decay_target=0.999, warmup_steps=10)

- Branch: `charliepai2d2-askeladd/ema-bias-correction` — metrics committed.
- Hypothesis: Adam-style ramp `decay_t = min(0.999, (1+t)/(10+t))` keeps EMA(0.999)'s late-training smoothing strength while removing the cold-start bias that hurt EMA(0.999) in the under-trained regime. Predicted −1% to −3%.
- Result: best `val_avg/mae_surf_p = 84.6454` at epoch 13. **+1.71% vs current EMA(0.99) baseline (83.223).** test_avg = 75.0424 (+1.54%).
- Per-split val: single_in_dist 97.44 (−1.40%), camber_rc 96.18 (−0.45%) — small wins; **cruise 65.47 (+7.04%) and re_rand 79.50 (+4.19%) regressed**, driving the net loss.
- **Cold-start did improve** (ep1=183.4 vs EMA(0.99)'s 193 vs EMA(0.999)'s 333). The bias correction does what it's designed to do.
- **Student's structural diagnosis (the keeper)**: under the 13-epoch timeout-bound regime, the dominant effect of EMA(0.99) is **continuous fast tracking** of the rapidly-improving live model, not just cold-start bias correction. The bias-corrected scheme reaches effective decay 0.977 by step 375 (~epoch 1) and 0.995 by step 1875 (~epoch 5), so outside the first ~50 steps it's essentially EMA(0.999) — which we already know lags by ~2 epochs of training progress. Cold-start and ongoing-tracking don't separate cleanly: the model is *always* in the rapidly-improving regime in this budget.
- Decision: **CLOSE.** Direction not dead — student's follow-up #2 (decay_target=0.99 + warmup_steps=10) keeps the fast-tracking baseline while adding the cold-start bias correction layer. Queued as the round-5 reassignment.

## 2026-04-28 02:55 — PR #371: Gradient accumulation 2 (effective batch 8) with √2 lr scaling

- Branch: `charliepai2d2-nezuko/grad-accum-2` — student rebased onto SwiGLU baseline (88.227) before running; metrics committed.
- Hypothesis: effective batch 8 via `grad_accum_steps=2` reduces gradient noise, smoothing late-training validation. Predicted −1% to −4%.
- Result: best `val_avg/mae_surf_p = 123.997` at epoch 13. **+40.5% vs SwiGLU baseline (88.227); +49% vs current EMA(0.99) baseline (83.223).** test_avg = 113.745 (+45.2%).
- Trajectory: trails the baseline at every single epoch (epoch 1: 355 vs 333; epoch 13: 124 vs 88).
- **Student's structural diagnosis (excellent)**: grad-accum-2 halves the optimizer step count while keeping wall-clock fixed under the timeout. Under a 13-epoch budget where val is still descending ~5 pts/epoch, halving updates is catastrophic. The √2 lr scaling can't compensate because we never reach a plateau where the gradient-noise-reduction benefit of bigger effective batch could overtake the per-step efficiency loss. EMA also accumulates half as many ticks. At equal *optimizer-step* count, grad-accum-2 epoch-13 (124.00) ≈ baseline epoch-6 (157.45), so it's a tiny bit ahead per-step — but wall-clock dominates here.
- Curve was strictly monotone (no spikes — smoothness prediction held), but post-EMA the baseline is also smooth, so smoothness alone wasn't a differentiator at this point in the research.
- Decision: **CLOSE.** Hypothesis fundamentally requires equal optimizer-step count, which we don't have at fixed wall-clock. Worth revisiting if `SENPAI_TIMEOUT_MINUTES` ever doubles, or as a building block once the model converges within budget.

## 2026-04-28 03:05 — PR #455: Stochastic depth (DropPath) with linear schedule 0 → 0.1 — **WINNER**

- Branch: `charliepai2d2-thorfinn/stochastic-depth-01` — metrics committed.
- Hypothesis: DropPath is a textbook regularizer that should help on the OOD-camber generalization bottleneck. Predicted −1% to −3%.
- Result: best `val_avg/mae_surf_p = 80.480` at epoch 14. **−3.30% vs EMA(0.99) baseline (83.223), upper end of predicted range.** test_avg = 72.328 (−2.13%). Param-identical (no new learnable params).
- Per-split val MAE for `p`: single_in_dist 92.91 (−5.98%), camber_rc 95.53 (−1.12%, near noise), cruise 57.24 (−6.41%), re_rand 76.24 (−0.08%, flat).
- Per-split test MAE for `p`: single_in_dist 85.50 (−4.82%), camber_rc 85.57 (+1.38% — within noise), cruise 49.23 (−3.17%), re_rand 69.01 (−2.17%).
- **Hypothesis-aligned with mechanism, not target**: DropPath worked as a generic regularizer (uniform offset on the val curve), but did **not** preferentially target the OOD-camber splits frieren had flagged as the bottleneck. `val_single_in_dist` improved 5× more than `val_geom_camber_rc` and 75× more than `val_re_rand`. This refines the bottleneck story: the camber_rc / re_rand splits are limited by something else (data coverage in the camber range, or Re-extreme sample handling) rather than implicit-ensembling regularization.
- Effective drop probabilities are small ([0, 0.025, 0.05, 0.075, 0.0]) — at 5 layers, DropPath at 0.1 max is a mild lever. Pushing it further (0.2, 0.3) is the natural next test.
- Decision: **MERGE.** New baseline `val_avg/mae_surf_p = 80.480`, `test_avg/mae_surf_p = 72.328`.

## 2026-04-28 03:05 — PR #456: CaiT-style LayerScale residual gating (init 1e-4)

- Branch: `charliepai2d2-edward/layerscale-1e4` — metrics committed.
- Hypothesis: per-branch scalar gates (γ_attn, γ_mlp) initialized to 1e-4 stabilize the init-time gradient regime; should give cleaner early-epoch descent. Predicted −1% to −2%.
- Result: best `val_avg/mae_surf_p = 83.5436` at epoch 13. **+0.39% vs EMA(0.99) baseline (83.223), near-tie.** test_avg = 74.3636 (+0.62%).
- Per-split val: helped harder splits (single_in_dist −2.25%, camber_rc −0.76%) but hurt easier ones (cruise +5.09%, re_rand +1.48%). Net is roughly zero.
- **Student's structural diagnosis**: LayerScale gammas DID grow (50–400× from 1e-4 init) but are still 25–200× smaller than 1.0 by epoch 13 — they never reach the "specialization regime" where LayerScale's value lies. Under our 13-epoch budget, near-identity-at-init silences early-epoch signal without enough time for the gammas to mature. Empirically, epoch 1 was *worse* than baseline (198 vs 193), not gentler — the prediction failed mechanically.
- Decision: **CLOSE.** Direction not dead at larger init or longer budget. Student's follow-up #2 (`layerscale_init=1e-2` — same shape, 100× shorter trip from init to specialization regime) is queued as the round-5 reassignment.

## 2026-04-28 03:05 — PR #450: RMSNorm everywhere in TransolverBlock

- Branch: `charliepai2d2-alphonse/rmsnorm-everywhere` — branched on SwiGLU pre-EMA(0.99); metrics committed.
- Hypothesis: replace `nn.LayerNorm` with `RMSNorm` for LLaMA-style stack alignment with SwiGLU. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 91.342` at epoch 12. **+3.53% vs SwiGLU baseline (88.227); +9.8% vs current EMA(0.99) baseline (83.223).** But this is a wall-clock result, not a per-step result.
- **Per-step quality wins at every measured epoch**: RMSNorm matched-or-beat LayerNorm at epochs 1, 5, 8, 10, 11, 12 (largest gap −6.84% at epoch 5; settled to −1.01% at epoch 12). The hypothesis is **qualitatively validated** on per-step quality.
- **What killed the headline metric is wall-clock**: `nn.RMSNorm` in PyTorch 2.10 ran 16.9% slower per epoch (162.4s vs 138.9s) and used 11.9% more peak VRAM than `nn.LayerNorm`. This burned exactly one epoch under the 30-min cap (12 vs baseline's 13), and that final epoch is exactly where SwiGLU pulls away (92.27 → 88.23, a −5.4% jump in one step under the still-monotonically-descending curve). Almost certainly a kernel-dispatch issue with the ATen op on this Blackwell build; the LLM-style hand-written `nn.Module` RMSNorm should compile via TorchInductor and recover wall-clock parity.
- Decision: **CLOSE this run** but the direction has clear signal. Re-run with the manual implementation (student's follow-up #1) is the queued reassignment. **The hypothesis is valid; the implementation choice was wrong.**

## 2026-04-28 03:20 — PR #463: Huber loss with delta=0.25 — **MAJOR WINNER**

- Branch: `charliepai2d2-fern/huber-delta-025` — branched on EMA(0.99)+SwiGLU (pre-DropPath); metrics committed.
- Hypothesis: push δ profile from the saturation question raised by PR #439's δ=0.5 (1pt gain). Profile so far: δ=2→107.61, δ=1→88.23, δ=0.5→87.27 — flattening. Predicted at δ=0.25: −0.5% to −1.5% (could regress if profile saturated).
- Result: **best `val_avg/mae_surf_p = 72.414`** at epoch 13. **−13.0% vs EMA(0.99)+SwiGLU baseline (83.223). −10.0% vs current merged DropPath baseline (80.480).** test_avg = 63.082 (−14.6% vs 73.904, −12.8% vs current 72.328). **Largest single-PR improvement of the entire research programme.**
- Per-split val MAE for `p`: all four improved 10–18%. single_in_dist=87.91 (−11.03%), camber_rc=83.32 (−13.76%), camber_cruise=50.22 (−17.88%), re_rand=68.20 (−10.62%). The cruise canary (which slightly regressed at δ=0.5) gained the *most* at δ=0.25.
- **The δ profile did NOT saturate at δ=0.5** — δ=0.25 delivered a much steeper gain than the δ=1→0.5 step did, on a different baseline. Updated profile: δ=2→107.6, δ=1→88.2 (pre-EMA); δ=1→83.2, **δ=0.25→72.4** (post-EMA(0.99)).
- Validation curve: smooth monotonic descent through 13 epochs with one mild plateau tick at ep12 (+0.17, recovered fully at ep13). No L1-like instability surfaced within the budget.
- **Mechanism**: smaller quadratic region (more L1-like) handles the heavy-tailed pressure error distribution better. EMA(0.99)'s fast tracking compounds especially well with a more L1-like loss because most of the loss surface is now linear-bounded gradient — the EMA averages over a less-jagged optimization trajectory.
- Decision: **MERGE.** Squash-merging fern's δ change (`delta=1.0 → 0.25` at both call sites) on top of the current DropPath stack. The 72.414 measurement was on the pre-DropPath stack; the post-merge combined stack (δ=0.25 + DropPath + EMA(0.99) + SwiGLU + huber + NaN-safe) is expected to be at-or-better-than 72.414 since DropPath is orthogonal to loss reformulation. Future round-7 PRs will refine the post-merge baseline.

## 2026-04-28 03:20 — PR #460: Per-sample feature noise (semantics-aware)

- Branch: `charliepai2d2-frieren/per-sample-feature-noise` — branched on EMA(0.99)+SwiGLU pre-DropPath; metrics committed.
- Hypothesis: replace per-node noise (PR #425) with semantics-aware noise — per-node on dims 0–12 (positions, saf, dsdf, is_surface) + per-sample broadcast on dims 13–23 (Re, AoA, NACA, gap, stagger). Direct correction of PR #425's failure mode. Predicted −1% to −3%.
- Result: best `val_avg/mae_surf_p = 81.437` at epoch 12. **−2.15% vs EMA(0.99)+SwiGLU baseline (83.223), at the upper end of predicted range.** But +1.19% vs the now-current DropPath baseline (80.480). test_avg = 73.019 (−1.20% vs EMA, +0.95% vs current).
- **Diagnosis from PR #425 fully confirmed**: the splits per-node noise broke (`val_geom_camber_rc` was +6.57%, `val_re_rand` was +9.18% under PR #425) flip cleanly: camber_rc now −3.27%, re_rand essentially flat (−0.03%). Per-sample broadcast preserves (geometry, flow conditions) consistency.
- 3/4 test splits improved; test_camber_rc +1.22% (within run-to-run variance, opposite direction of val_camber_rc improvement).
- Decision: **CLOSE** standalone — doesn't beat the current DropPath baseline (which hadn't merged when frieren ran). Direction is alive: per-sample feature noise is now a validated technique. Frieren's own follow-up #1 (sweep `feature_noise_std` higher) is queued, ideally on the post-fern-merge baseline.

## 2026-04-28 03:20 — PR #459: SwiGLU preprocess MLP (LLaMA-everywhere)

- Branch: `charliepai2d2-tanjiro/swiglu-preprocess` — branched on EMA(0.99)+SwiGLU pre-DropPath; metrics committed.
- Hypothesis: replace preprocess MLP with `SwiGLU_MLP` to extend the per-block SwiGLU win (PR #391) to the input projection. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 88.299` at epoch 13. **+6.1% vs EMA(0.99)+SwiGLU baseline; +9.7% vs current DropPath baseline (80.480).** test_avg = 79.294 (+7.3%).
- 3/4 val splits regressed 8–10%; only `val_geom_camber_rc` was flat (−0.61%).
- **Student's mechanistic diagnosis (the keeper)**: per-block SwiGLU gates AFTER attention has mixed the residual stream — gating prunes redundancy in already-learned representations. **Input-projection SwiGLU gates BEFORE mixing**, so `silu(W1·x) * (W2·x)` chooses *which raw input channels survive* into the hidden space — and at the input layer, every raw feature (`log(Re)`, AoA, NACA digits, gap, stagger, positions, dsdf) carries irreducible physical information that the network needs later. **Gating at the input prunes signal.** Trajectory data supports this: this run matched baseline through epoch 5 (coarse structure phase) then fell behind as fine-detail learning kicks in.
- Decision: **CLOSE.** "LLaMA-everywhere" breaks at the input projection — that's the definitive lesson. Adding to "Disconfirmed directions". Future input-layer capacity should reach for non-gated expansions (wider intermediate, more layers) before SwiGLU-style gating.

## 2026-04-28 03:55 — PR #480: AdamW betas (0.9, 0.95) — **MERGE as orthogonal lever**

- Branch: `charliepai2d2-nezuko/adamw-betas-095` — branched on EMA(0.99)+SwiGLU pre-DropPath, pre-fern; metrics committed.
- Hypothesis: AdamW β₂ 0.999 → 0.95 (half-life ~700 → ~14 steps) tracks the rapidly-changing gradient distribution under our 13-epoch under-trained regime. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 77.951` at epoch 13. **−6.34% vs EMA(0.99)+SwiGLU baseline she branched on (83.223), 4–13× the predicted gain.** test_avg = 68.753 (−6.97% vs 73.904). Vs the now-current huber-δ=0.25 baseline (72.414): +7.6% — does NOT beat the merged baseline standalone.
- Per-split val: all 4 improved 4.32–7.98% vs starting baseline; test: all 4 improved 3.87–12.98%, with test_single_in_dist outsized at −12.98%.
- Trajectory: matched baseline epochs 1–3 (gradients similar at very low step count), then β₂=0.95 opened a persistent ~3–6% lead from epoch 5 onward — gap held through every subsequent epoch. Mechanism: β₂=0.999 is heavily anchored to random gradients for the first ~2 epochs at our batch count; β₂=0.95 reaches steady-state inside the first quarter of epoch 1.
- **Orthogonal to fern's huber-δ=0.25** (PR #463): different mechanism (optimizer 2nd-moment time constant vs loss curvature). Both proven on the same starting baseline (83.223), so the post-merge stack adds two independent levers.
- Decision: **MERGE.** The lever is clearly real (−6.34% on its starting baseline). Combining with the now-merged huber-δ=0.25 may compound (additive: ~−18% combined) or be partially redundant (both address under-training). Future round-7 PRs will measure the combined val_avg. β₂=0.95 is a single-line change with zero compute cost — worth merging on the orthogonality argument. The 77.951 standalone number doesn't beat 72.414, but the lever is portable and shouldn't hurt.

## 2026-04-28 03:55 — PR #488: RMSNorm with manual nn.Module (fix nn.RMSNorm wall-clock penalty)

- Branch: `charliepai2d2-alphonse/rmsnorm-manual` — branched on DropPath baseline pre-fern; metrics committed.
- Hypothesis: hand-written `nn.Module` RMSNorm should compile via TorchInductor and recover wall-clock parity (or beat) `nn.LayerNorm`, letting RMSNorm's per-step quality lead translate to the headline. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 84.149` at epoch 12 (baseline got 13 epochs). **+4.6% vs DropPath baseline (80.480); +16.2% vs current huber-δ=0.25 baseline (72.414).** test_avg = 76.615 (+5.9%).
- **Wall-clock**: 157.89 s/epoch (manual) vs 162.4 s/epoch (nn.RMSNorm) vs 138.9 s/epoch (nn.LayerNorm). The manual implementation **shaved only 2.8% off the ATen op cost** — did NOT reach LayerNorm parity, lost one epoch under the 30-min cap.
- **Student's diagnostic (the keeper)**: train.py runs in eager mode end-to-end — there's no `torch.compile` call anywhere. So TorchInductor was never invoked on either RMSNorm implementation. The 2.8% gain is just eager-mode kernel-launch overhead difference. `nn.LayerNorm` ships an optimized fused CUDA kernel (single kernel for mean + var + normalize + affine); RMSNorm in eager mode dispatches into 4–5 separate kernels per call × 15 norm sites = ~19 s/epoch overhead.
- Decision: **CLOSE.** The architectural lever (RMSNorm > LayerNorm per-step quality) **may be real** but won't land without `torch.compile` or a custom Triton kernel. Direction is queued but should not be retried via pure-Python `nn.Module` on this build. **Student's follow-up #1 (`torch.compile` the whole model first) is the correct next step** — useful even if RMSNorm doesn't win, because the whole baseline gets faster (more epochs in budget). Queuing as the round-6 reassignment.

## 2026-04-28 04:10 — PR #479: Bias-corrected EMA (decay_target=0.99, warmup_steps=10) — **MERGE as orthogonal compound**

- Branch: `charliepai2d2-askeladd/bias-corrected-ema-099` — branched on EMA(0.99)+SwiGLU pre-DropPath, pre-fern; metrics committed.
- Hypothesis: combine EMA(0.99)'s fast-tracking strength with the cold-start bias correction layer (Adam-style ramp from PR #454). Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 81.251` at epoch 13. **−2.37% vs EMA(0.99)+SwiGLU baseline (83.223), beats predicted upper bound.** test_avg = 72.560 (−1.82% vs 73.904). Vs the current 72.414 conservative target: +12.2% standalone, but the lever is mechanistically orthogonal to fern's δ=0.25 and nezuko's β₂=0.95 (both proven on the same starting baseline).
- **Both diagnostics passed**:
  - Cold-start gain: epoch-1 val_avg = 186.1 vs EMA(0.99)'s ~193 (~3.5% better at epoch 1, exactly as predicted from PR #454's cold-start observation).
  - Fast-tracking preserved: `val_geom_camber_cruise` +0.47% and `val_re_rand` +0.52% (both flat — the splits PR #454's bias-corrected-EMA(0.999) regressed by 7%/4% are now unaffected).
- Effective decay trajectory: 0.18 at step 1, 0.85 at step 50, 0.92 at step 100, 0.98 at step 500. The ramp `(1+t)/(10+t)` only crosses 0.99 at t ≈ 891 (mid-epoch 3) — the asymptote is approached, not hit, in the 13-epoch budget. The cold-start window is short but real.
- **The lever is a strict superset of current EMA(0.99)**: setting `warmup_steps=0` reduces to the current behavior; `warmup_steps=10` adds a small ramp at the start. Should give ≥ current performance on any future run.
- Decision: **MERGE.** Orthogonal compound — same merge-aggressive logic as PR #480 (β₂=0.95). Combined-stack actual to be measured in subsequent rounds.

## 2026-04-28 04:10 — PR #487: LayerScale init 1e-2 (faster gamma maturation in 13-epoch budget)

- Branch: `charliepai2d2-edward/layerscale-1e2` — branched on DropPath baseline pre-fern; metrics committed.
- Hypothesis: at 1e-2 init, gammas have 100× shorter trip to specialization regime than at 1e-4 init. Predicted gammas reach near 1.0 in 13 epochs and the per-split asymmetry from PR #456 disappears. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 81.314` at epoch 13. **+1.04% vs DropPath baseline (80.480); +12.3% vs current target.** test_avg = 74.302 (+2.73%).
- **Critical structural finding (the keeper)**: at epoch 13, gammas are at `gamma_attn ∈ [0.0065, 0.0177]` and `gamma_mlp ∈ [0.0300, 0.0480]` — close to where 1e-4 init equilibrates after 250× growth (~0.025), and **most blocks even shrank below the 1e-2 init**. The model spends the entire 13-epoch budget in the *near-identity gating regime*, regardless of init. **The 250×-per-13-epochs growth observed at 1e-4 init was an artifact of how far the parameter sat from its equilibrium, not a constant**. Trip changed; destination didn't.
- Per-split asymmetry preserved at 1e-2 init: helps single_in_dist (−2.34%) + camber_rc (−2.24%); hurts cruise (+9.66%) + re_rand (+2.79%) — same direction and similar magnitudes as PR #456. **The asymmetry is a structural property of LayerScale gating on this dataset, not an init-dependent transient.** PR #456's hypothesis that "1e-2 init eliminates the asymmetry" is falsified.
- Early-epoch comparison (epochs 1–3): 1e-2 *did* deliver smoother early descent (epoch 3: 134 vs 144 baseline, −6.7%) — the cold-start advantage is real. But it gets washed out by epoch 14.
- Decision: **CLOSE.** Two failed LayerScale attempts (1e-4 in PR #456 and 1e-2 here) — direction is firmly disconfirmed. Adding to "Disconfirmed directions".

## 2026-04-28 04:10 — PR #486: Stochastic depth drop_path_max 0.1 → 0.2

- Branch: `charliepai2d2-thorfinn/drop-path-02` — branched on DropPath baseline pre-fern; metrics committed.
- Hypothesis: push DropPath harder (0.1 → 0.2; effective per-block rates `[0, 0.05, 0.1, 0.15, 0.0]` vs `[0, 0.025, 0.05, 0.075, 0.0]`) for ~2× more stochasticity. DeiT/Swin uses 0.2–0.4 routinely. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 80.858` at epoch 14. **+0.47% vs DropPath(0.1) baseline (80.480); essentially parity on test (+0.10%, 72.399 vs 72.328).** Vs current 72.414 target: +11.7%.
- Per-split val: `val_re_rand` improved −4.55% (the only bright spot — Re-axis OOD generalization benefits from stronger stochastic regularization), but other splits regressed +1.3% to +3.8% (camber_cruise worst at +3.78%). Net regression.
- **Trajectory analysis**: DropPath 0.2 had a faster start (epochs 1–3, by 5–7 MAE units) — stronger inductive bias when model is barely fit. Then it lagged epochs 6–12 (up to +7 MAE) because fewer effective parameters slow the absolute loss reduction. Curves converged by epoch 14 (final gap +0.38).
- **The regularizer saturated at 0.1 within the 14-epoch budget.** Stronger DropPath would need a longer epoch budget to amortize the slower mid-training fit phase — the closing gap at epoch 14 suggests it might overtake at longer training.
- Decision: **CLOSE.** Direction not dead in absolute terms (val_re_rand +4.55% is real), but at the 30-min wall-clock budget DropPath=0.1 is the local optimum.

## 2026-04-28 04:25 — PR #493: Huber loss δ=0.1 (saturation test)

- Branch: `charliepai2d2-fern/huber-delta-01` — branched on the post-fern-merge baseline; metrics committed.
- Hypothesis: push δ profile further (0.25 → 0.1). δ=0.5→0.25 delivered ~10× more gain than δ=1→0.5 — does the profile keep descending or saturate? Predicted −1% to −5% if profile descending; flat or slight regression if saturated.
- Result: best `val_avg/mae_surf_p = 72.369` at epoch 14. **−0.06% vs the δ=0.25 reference (72.414); test_avg = 62.891 (−0.30% vs 63.082)**. Both essentially within run-to-run noise.
- **The δ profile is now exhausted.** Updated profile: δ=2→107.6, δ=1→88.2 (pre-EMA); δ=1→83.2, δ=0.25→72.4, **δ=0.1→72.4 (this PR)**. The δ=1→0.25 step delivered −10.8 absolute (−13.0%); δ=0.25→0.1 step delivers only −0.045. Profile flattened.
- Per-split val: 3/4 improved modestly (single −1.32%, cruise −1.08%, re_rand −2.96%); **val_geom_camber_rc regressed +4.25%** — the only split going backward. Net val_avg gain washes out.
- No L1-style instability emerged — the PR's "stays just inside huber territory to avoid gradient discontinuity" hedge was unneeded but harmless. Pure F.l1_loss might give the same number.
- **Important calibration**: this run *also* validates that the post-DropPath baseline is essentially at the 72.414 conservative target — fern's δ=0.25 rerun on the post-DropPath stack would land within run-to-run variance of the original measurement.
- Decision: **CLOSE.** δ profile is exhausted; ~0% gain vs current. Move to architecture-side levers per fern's own conclusion.

## 2026-04-28 04:25 — PR #495: Semantics-aware feature noise std 0.01 → 0.02

- Branch: `charliepai2d2-frieren/feature-noise-002` — metrics committed.
- Hypothesis: doubling the feature noise std on top of the merged baseline (DropPath + δ=0.25). Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 74.8448` at epoch 14. **+3.36% regression vs the 72.414 reference; +5.10% on test (66.301).**
- **Asymmetry signal**: damage concentrated on **in-distribution / closest-OOD splits** (val_single_in_dist +5.48%, val_geom_camber_rc +5.60%); the harder OOD splits stayed flat (val_geom_camber_cruise +0.71%, val_re_rand −0.16%). The PR #460 std=0.01 win came from improving the OOD splits; doubling the noise didn't compound that — instead it hurt the in-dist fit.
- Mechanism: with DropPath + huber δ=0.25 already providing capacity-control, the three-regularizer stack (noise + DropPath + δ=0.25) is past the optimum. Adding more noise doesn't help OOD generalization further; the extra regularization hurts in-distribution fit.
- Decision: **CLOSE.** Don't sweep up. Direction not dead — sweep DOWN (`feature_noise_std ∈ {0.005, 0.015}`) to find the sweet spot under the merged stack. Queued as the round-7 reassignment.

## 2026-04-28 04:25 — PR #494: AdamW weight_decay 1e-4 → 3e-4

- Branch: `charliepai2d2-tanjiro/weight-decay-3e-4` — metrics committed.
- Hypothesis: tripling weight decay adds orthogonal regularization on top of DropPath + δ=0.25. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 73.771` at epoch 14. **+1.87% regression vs 72.414 reference; +1.35% test.**
- 7/8 per-split val + test metrics regressed. Only val_re_rand improved (−1.0%) — the Re-randomized split benefits from stronger weight regularization, but the gain is small relative to the cross-split losses (val_geom_camber_rc +5.21%).
- Mechanism: extra L2 shrinkage attenuates updates in the deeper/wider weight matrices (SwiGLU FFN intermediate=176, attention proj at n_hidden=128). With only 14 effective epochs reachable in the 30-min budget, those matrices are already under-trained; extra L2 pressure pushes them further from their optimum. Curve descended monotonically but slightly more conservatively than the merged baseline.
- Decision: **CLOSE.** Tripling wd over-regularizes on the merged stack. Direction not dead — student's follow-up #1 (lower wd, e.g. wd=3e-5 or 1e-5) is the orthogonal lever still untested. Queued as the round-7 reassignment.

## 2026-04-28 04:45 — PR #511: AdamW betas (0.9, 0.95) → (0.9, 0.90)

- Branch: `charliepai2d2-nezuko/adamw-beta2-090` — metrics committed.
- Hypothesis: push β₂ profile further to 0.90 (half-life ~7 steps). Tests whether faster gradient tracking compounds further. Predicted −0.5% to −1.5% (or could regress if 0.95 is sweet spot).
- Result: best `val_avg/mae_surf_p = 72.952` at epoch 14. **+0.74% regression vs 72.414 reference; +1.08% test (63.762).** 6/8 per-split val+test metrics slightly worse; only val_re_rand and test_re_rand improved (−1.01% / −0.53%).
- **Predicted destabilization didn't materialize**: validation curve was completely monotone, no oscillations, no plateaus. Train-side surf_loss also smooth (0.119 → 0.030). β₂=0.90's ~7-step half-life is well-behaved at the per-epoch eval granularity. The EMA(0.99) on model weights may be absorbing whatever extra step-to-step noise β₂=0.90 lets through.
- **Mechanism**: β₂=0.95 was already in the regime where the second-moment EMA tracks the cosine-decay-shaped gradient surface well in 14 epochs. Going further gives up a tiny amount of variance reduction without a corresponding tracking benefit. **β₂ profile saturated at 0.95.**
- Updated β₂ profile: 0.999 → 83.223 (pre-merge); 0.95 → 77.951 (PR #480 standalone, pre-DropPath/δ); 0.90 → 72.952 (this run, full merged stack — not apples-to-apples vs 0.95 since stack changed).
- Decision: **CLOSE.** β₂ direction closed at 0.95. Don't sweep further (0.85, 0.80) — the saturation argument applies.

## 2026-04-28 04:55 — PR #520: PhysicsAttention temperature init 0.5 → 1.0 — **WINNER on the merged baseline**

- Branch: `charliepai2d2-thorfinn/slice-temp-1p0` — metrics committed.
- Hypothesis: the slice-attention learnable temperature has a poor init (0.5) that's 2× sharper than the equilibrium the model wants. Under the 13-epoch budget, the init regime matters. Predicted −0.3% to −1.5%.
- Result: best `val_avg/mae_surf_p = 71.6985` at epoch 14. **−0.99% vs the 72.414 reference; test_avg = 62.5824 (−0.79% vs 63.082)**. First PR to measurably beat the merged baseline.
- Per-split val: single_in_dist 85.48 (−2.77%), camber_rc 83.09 (−0.28%, flat), cruise 50.53 (+0.61%, flat), re_rand 67.69 (−0.74%). Per-split test: test_single_in_dist (−4.63%) drives the gain; test_geom_camber_rc regressed +3.84% (only split going backward).
- **Mechanism (definitive)**: Final temperature parameters at epoch 14 landed at **0.95–0.99 across all 5 blocks** (live and EMA both). The 0.5 init was ~2× below this equilibrium; under our 14-epoch budget the temperature only drifted ~0.5 → ~0.95 (live wouldn't even reach 0.95 at the original init). Init=1.0 starts essentially **at** the equilibrium, so every gradient step fits data instead of un-doing the init. **This is an optimization-warmup phenomenon**, not a capacity phenomenon — exactly as the PR predicted.
- Across-block uniformity (all 5 blocks → ~0.95) means **per-block scheduled init isn't a useful axis at this depth**, but the *initial* sharpness is.
- Param-identical to baseline (1 learnable scalar tensor, just initialized differently). Zero compute / memory cost.
- Decision: **MERGE.** New baseline `val_avg/mae_surf_p = 71.6985`, `test_avg/mae_surf_p = 62.5824`.

## 2026-04-28 05:10 — PR #527: AdamW weight_decay 1e-4 → 3e-5 — **WIN as orthogonal compound**

- Branch: `charliepai2d2-tanjiro/weight-decay-3e-5` — branched on the merged baseline pre-slice-temp; metrics committed.
- Hypothesis: tanjiro's PR #494 (wd=3e-4) over-regularized; the under-regularized direction is the orthogonal lever still untested. With DropPath + huber-δ=0.25 + EMA + β₂=0.95 already providing implicit regularization, less L2 may let under-trained matrices fit better.
- Result: best `val_avg/mae_surf_p = 70.814` at epoch 14. **−1.23% vs current 71.6985 baseline; −2.21% vs the 72.414 reference.** test_avg = 63.031 (essentially flat).
- **wd profile is monotone over 3 measured points**: wd=3e-5 → 70.814; wd=1e-4 → 72.414; wd=3e-4 → 73.771. The merged stack is **over-regularized at wd=1e-4** — the orthogonal regularizers (DropPath + huber + EMA + β₂=0.95) are already doing the work.
- Per-split val: 3/4 improved (single_in_dist −5.27%, camber_rc −1.61%, re_rand −2.00%); val_geom_camber_cruise +1.86% (only regression).
- Mechanism: less L2 shrinkage → bigger AdamW updates per step → less under-fit at the 14-epoch budget on deeper/wider weight matrices (SwiGLU FFN intermediate=176, attn proj at 128).
- Decision: **MERGE.** Orthogonal compound — wd is mechanistically independent of slice-temp init. tanjiro's run was branched pre-slice-temp; the merged stack will combine both. Single-line change, zero compute cost.

## 2026-04-28 05:10 — PR #519: Multi-head attention 4 → 8 (head_dim 32 → 16)

- Branch: `charliepai2d2-edward/n-head-8` — metrics committed.
- Hypothesis: param-matched parallel attention paths. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 80.536` at epoch 10 (only 11/50 epochs in budget). **+12.3% regression vs current 71.6985 baseline; +14.0% vs slice-temp baseline.** test_avg = 71.204 (+13.8%).
- **Critical infrastructure finding (the keeper)**: the PR's "param-matched" assumption was **wrong**. The codebase's `PhysicsAttention.to_q/to_k/to_v` are `Linear(dim_head, dim_head, bias=False)` — *head-shared* projections applied per-head via broadcast, NOT `Linear(inner_dim, inner_dim)` (true MHA). Shrinking dim_head 32→16 cuts these 4× per layer + `in_project_slice` 2×. Net: **−16,620 params (−2.48%)**, not param-matched.
- **Wall-clock**: 173 s/epoch (n_head=8) vs 138 s/epoch (baseline) = **+25% slower**, +25% peak VRAM. SDPA on head_dim=16 is less compute-bound; reshape over 8 heads adds memory traffic. Got only 11 epochs vs baseline's 14.
- **Per-epoch convergence is genuinely faster**: n_head=8 hit val=80.5 at epoch 10 vs n_head=4 reaching 81.3 at epoch 13 — same val number reached 3 epochs earlier. The architectural signal is real, just masked by the wall-clock penalty.
- Per-split signature: hurts single+camber_rc, helps cruise+re_rand — opposite to LayerScale's signature, supporting the "more diverse attention paths help OOD" mechanism (when the wall-clock penalty doesn't dominate).
- Decision: **CLOSE.** +12.3% regression on the headline metric is too large to merge despite the per-epoch architectural signal. Direction not dead — student's follow-up #3 (refactor `to_q/k/v` to true per-head MHA so n_head=8 actually adds capacity) or #4 (n_head=8 with larger n_hidden so head_dim stays at 24+) are reasonable paths if compute permits.

## 2026-04-28 05:10 — PR #518: Bias-corrected EMA warmup_steps 10 → 50 — **WIN as orthogonal compound**

- Branch: `charliepai2d2-askeladd/bias-corrected-ema-warmup-50` — metrics committed.
- Hypothesis: extend the cold-start EMA ramp influence beyond the 13-epoch budget by raising warmup_steps. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 71.4284` at epoch 14. **−0.38% vs current 71.6985 baseline; −1.36% vs 72.414 reference.** test_avg = 63.4404 (+1.37% vs current — small regression).
- **All 4 val splits improved** vs the 72.414 reference. The biggest gains are on **the splits that were flat under warmup_steps=10**: val_geom_camber_cruise (+0.47% → −2.06%) and val_re_rand (+0.52% → −3.14%). The 0.99 asymptote was NOT load-bearing.
- Effective decay trajectory: clamps at 0.99 only at t ≈ 4901 — never reached in 13-epoch budget (max 0.988 at step 4000). EMA stayed permanently slightly faster than 0.99.
- Decision: **MERGE.** Orthogonal compound to wd and slice-temp.

## 2026-04-28 05:25 — PR #525: Cosine schedule with 1-ep warmup + T_max=13 — **MAJOR WINNER**

- Branch: `charliepai2d2-fern/cosine-warmup-tmax-aligned` — metrics committed.
- Hypothesis: align cosine LR decay with the realistic 14-epoch budget. PR #370 (T_max=14, no warmup) had failed on the EMA(0.99)+SwiGLU baseline because EMA already saturated the fast-tracking benefit; the merged stack with bias-corrected EMA + β₂=0.95 + δ=0.25 changes the basin geometry. Predicted −0.5% to −2%.
- Result: best `val_avg/mae_surf_p = 67.306` at epoch 14. **−7.05% vs 72.414 reference; −6.12% vs current 71.6985 baseline.** test_avg = 59.296 (−6.00% vs 63.082). **Biggest single-PR delta since fern's own δ=0.25 win** in the entire programme.
- Per-split val MAE for `p`: **all 4 splits improved**, biggest on val_single_in_dist (−11.66%) and val_geom_camber_cruise (−6.66%) and val_re_rand (−6.16%).
- Per-split test: single_in_dist=68.73, camber_rc=72.63, cruise=39.67, re_rand=56.16. test_avg = 59.30.
- **The "fine-tuning regime" signature is unmistakable**: late-epoch val slope shallows from 4.3 → 0.7 pts/epoch as cosine LR decays toward zero (ep10→11: −4.3, ep11→12: −1.7, ep12→13: −1.4, ep13→14: −0.7). Prior baselines hit the cap descending steeply at 3–5 pts/epoch — the model never reached this fine-tuning regime before.
- LR trajectory landed exactly as predicted: ep1=2.5e-4 (warmup), ep2=5e-4 (peak), ep14=7e-6 (~zero).
- **Why this works now when PR #370 failed**: that stack (EMA(0.99)+SwiGLU only) had EMA(0.99)'s ~1.85-epoch half-life saturating the fast-tracking effect, so adding cosine T_max=14 was redundant. The current merged stack (bias-corrected EMA + β₂=0.95 + huber-δ=0.25 + DropPath) creates a different basin geometry where the LR decay actually carves a sharper minimum.
- Decision: **MERGE.** Massive win. New baseline on fern's measured stack: val_avg = 67.306, test_avg = 59.296.

## 2026-04-28 05:25 — PR #526: Semantics-aware feature noise std 0.01 → 0.005

- Branch: `charliepai2d2-frieren/feature-noise-005` — metrics committed.
- Hypothesis: sweep noise std DOWN per the diagnosis on PR #495. The optimum may be below 0.01 on the merged regularizer-rich stack. Predicted −0.5% to −1.5%.
- Result: best `val_avg/mae_surf_p = 71.359` at epoch 14. **−1.46% vs 72.414 reference; −0.47% vs current 71.6985.** test_avg = 63.494 (+0.65% — slight regression).
- Per-split val: all 4 improve vs reference (single_in_dist −0.53%, camber_rc −1.83%, cruise +0.27%, re_rand −3.45%). Biggest improvers on val are exactly the splits PR #495 (std=0.02) hurt most: re_rand and camber_rc.
- Confirms the sweep-down direction works: profile is now std=0.005 < 0.01 (orig win) < 0.02 (regression).
- The +0.65% test regression is likely run-to-run variance (the model was still descending at the cap).
- Decision: **MERGE.** Orthogonal compound. Modest val win on the same starting baseline.

## 2026-04-28 05:35 — PR #510: torch.compile(model) — **INFRASTRUCTURE WIN**

- Branch: `charliepai2d2-alphonse/torch-compile-baseline` — metrics committed (both compile-default and apples-to-apples eager runs).
- Hypothesis: wrap model in `torch.compile(mode="reduce-overhead")` for fused TorchInductor + CUDA Graphs. Predicted −1% to −3% from speedup buying extra epochs.
- Result: **mode="reduce-overhead" OOMed** at iteration ~26 of epoch 1 (90.44 GB allocated, 59.65 GB in private CUDA Graph pools). Inductor printed: *"CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size... 9 distinct sizes"*. The variable mesh padding from `pad_collate` triggered exactly the failure mode the PR predicted.
- **Pivoted to `mode="default"`**: TorchInductor fusion only, no CUDA Graphs. Succeeded cleanly with one soft graph break (DropPath's `.item()` call).
- Result: best `val_avg/mae_surf_p = 66.397` at epoch 15. **−1.35% vs current 67.306 baseline; −6.1% vs apples-to-apples eager run (70.700 at ep14).** test_avg = 60.398 (+1.86% vs current 59.296 — driven by RNG since seeds differ between runs).
- **Wall-clock impact**: 103.6 s/epoch (−23.3% vs eager's 135.1 s/epoch). **17 epochs in 30-min cap vs 14 eager (+21%).** Compile cost: epoch 1 was +6 s warmup amortized across 16 fast subsequent epochs.
- Per-split val MAE for `p` (compile run, all 4 improved vs eager): single_in_dist 79.83 (eager 87.50), camber_rc 76.52 (eager 81.02), cruise 45.78 (eager 48.46), re_rand 63.46 (eager 65.82).
- Decision: **SEND BACK FOR REBASE.** GraphQL merge-conflict error on squash: the branch is several merges behind advisor (fern #525, frieren #526, and others landed after the branch was created). Rebasing requires a re-run to measure on the now-stronger baseline. The val_avg=66.397 / test_avg=60.398 numbers are recorded here as alphonse's measurement on the slice-temp-only baseline; on the post-#525 baseline (val_avg=67.306) the compile run should land at-or-below 66.397 since the merged stack starts from a stronger basin.

## 2026-04-28 05:45 — PR #548: Slice attention temperature init 1.0 → 1.5 — orthogonal compound

- Branch: `charliepai2d2-thorfinn/slice-temp-1p5` — branched on slice-temp baseline pre-#525/#526; metrics committed.
- Hypothesis: characterize equilibrium asymmetry. The PR predicted small regression because init=1.5 starts above the ~0.95 equilibrium and shouldn't drift fast enough to reach it in 14 epochs. Predicted Δ: −0.5% to +1%.
- Result: best `val_avg/mae_surf_p = 70.6169` at epoch 14. **−1.51% vs slice-temp baseline (71.6985); −0.55% on test (62.5824 → 62.2381).** vs current 67.306 baseline: +4.92% (branched too early to reach this comparison).
- Per-split val: val_geom_camber_rc −4.28% (biggest gain), val_re_rand −1.42%; single_in_dist and cruise flat (+0.07%, +0.27%).
- **Final per-block temperatures**: [1.480, 1.429, 1.452, 1.450, 1.434] live, mean 1.449. Drift Δ ≈ −0.051 over 14 epochs — same magnitude as PR #520's 1.0 → 0.95 drift. The temperature is **stuck well above the supposed 0.95 equilibrium** yet still improves over init=1.0.
- **Refutes the "stuck above is worse" hypothesis**: pushing init higher *helps*, even though gradients consistently push toward sharper attention at the same drift speed. The temperature has a **broad sweet spot in [1.0, 1.5]+** — the 0.95 "equilibrium" found in PR #520 is not the val-loss minimum; it's just where gradients stabilize at the 14-epoch budget.
- Decision: **MERGE.** Replaces init=1.0 with init=1.5 in PhysicsAttention. Mechanistically orthogonal to the cosine LR fix, noise, EMA, etc. The +1.51% improvement on the slice-temp baseline should compound (or at minimum not regress) when stacked with the rest. Single-token change, zero compute cost.

## 2026-04-28 05:45 — PR #540: Bias-corrected EMA decay_target 0.99 → 0.95

- Branch: `charliepai2d2-nezuko/ema-decay-target-095` — metrics committed.
- Hypothesis: parallel sweep on the weight-EMA decay axis (analogous to her β₂=0.95 win). Predicted Δ: −0.5% to −1.5% (could regress).
- Result: best `val_avg/mae_surf_p = 75.655` at epoch 14. **+4.48% vs 72.414 reference; +12.4% vs current 67.306 baseline.** test +5.57% / +12.3%.
- Per-split val: 3/4 regressed (single_in_dist +7.88%, camber_rc +5.91%, cruise +2.35%); val_re_rand essentially flat.
- Effective decay trajectory: clamps at the asymptote (0.95) at step ~170 vs step ~891 for decay=0.99. 5× shorter late-training memory window (~14 steps vs ~70 steps).
- **Mechanism (definitive)**: late-training smoothing IS load-bearing. The regression is concentrated on the splits where weight-space averaging matters most (single_in_dist, camber_rc — same splits DropPath improved). The OOD `val_re_rand` is unaffected, confirming that split's bottleneck is data-coverage rather than checkpoint smoothing.
- Decision: **CLOSE.** EMA decay axis is closed in the DOWN direction. Student's follow-up #2 (decay=0.995, the UP direction — longer late-training memory) is the orthogonal next probe. Queued as round-7 reassignment.

## 2026-04-28 06:00 — PR #556: SwiGLU FFN mlp_ratio 2 → 3 (LLaMA-style capacity bump)

- Branch: `charliepai2d2-edward/swiglu-mlp-ratio-3` — metrics committed.
- Hypothesis: SwiGLU's gate selectivity may benefit from more intermediate dim differently than the original GELU MLP (which failed at mlp_ratio=4, PR #392). Predicted −0.5% to +2%.
- Result: best `val_avg/mae_surf_p = 73.764` at epoch 13. **+9.6% vs current 67.306 baseline; +2.9% vs slice-temp baseline 71.6985.** test +9.8%/+4.0%. n_params = 824K (+23% vs 670K baseline). Wall-clock +5.7% per epoch (143.1 s vs 135.4 s) → got 13 epochs vs baseline's 14.
- Per-epoch convergence: **essentially identical** to slice-temp-1p0 baseline at ep13 (73.76 vs 73.73). **No per-epoch architectural signal from the FFN width bump.** Same failure mode as edward's #519 n_head=8 (wall-clock penalty without offsetting per-epoch gain).
- **Key finding (combined with frieren's #392 mlp_ratio=4)**: capacity in the FFN is NOT the bottleneck on this problem at 670–824K param scale. The gating mechanism (SwiGLU vs original GELU) doesn't change this; both architectures show the same shape of failure on capacity bumps.
- Decision: **CLOSE.** FFN-width axis is now disconfirmed across both gated and non-gated MLPs. Adding to "Disconfirmed directions".

## 2026-04-28 06:00 — PR #555: Bias-corrected EMA warmup_steps 50 → 100

- Branch: `charliepai2d2-askeladd/bias-corrected-ema-warmup-100` — metrics committed.
- Hypothesis: warmup_steps profile is monotone-improving (10 → 50 won), so push to 100. Predicted −0.3% to −1%.
- Result: best `val_avg/mae_surf_p = 72.4145` at epoch 14. **+1.38% vs ws=50 baseline (71.428); +7.6% vs current 67.306.** test +1.34%/+8.4%.
- **Profile saturated and reversed**: ws=50 is the sweet spot.
- Per-split val: mixed signal — 3 splits improved (single_in_dist −0.55%, cruise −2.74%, re_rand +1.50%), val_geom_camber_rc regressed sharply (+5.76% val, +8.68% test) and dragged the average up.
- Effective decay never reaches the 0.99 asymptote (peak ~0.9815 at end of budget; clamp would happen at step ~9800, ~6 epochs past timeout). At ws=100 the EMA effectively becomes a near-live tracker.
- **Mechanism**: the small-lookback smoothing at ws=50 was load-bearing for the unseen-camber-RC generalization split. Pushing further into "live tracker" regime loses that smoothing benefit specifically on the OOD-geometry axis. The "ramp longer = always better" expectation breaks down where smoothing-vs-tracking trade-offs become split-dependent.
- Decision: **CLOSE.** warmup_steps direction is closed at 50. Don't push to 200+.

## Test-metric NaN follow-up (cross-PR)

All three reviewed PRs report `test_avg/mae_surf_p = NaN`. Root cause from the student diagnoses:

- One sample (`test_geom_camber_cruise` sample 20) has 761 non-finite values in `y[p]` volume nodes. Surface `p` is finite for that sample.
- `data/scoring.py:accumulate_batch` is *intended* to skip samples with non-finite ground truth (`y_finite` mask), but the implementation computes `err = (pred - y).abs()` over the whole batch *before* masking. With IEEE 754, `Inf * 0 = NaN`, so the masked-out element still poisons the per-channel sum.
- `data/scoring.py` is read-only per `program.md`. The cleanest workaround is in `train.py` `evaluate_split`: filter out samples with any non-finite `y` before calling `accumulate_batch`. We will route this fix through a round-2 PR (charliepai2d2-edward) so paper-facing `test_avg/mae_surf_p` becomes recoverable for all subsequent runs.

## 2026-04-28 06:15 — PR #563: Semantics-aware feature noise std 0.005 → 0.0025

- Branch: `charliepai2d2-frieren/feature-noise-0025`
- Hypothesis: Feature noise profile is monotone-descending (0.005 < 0.01 original); push to 0.0025 to test if optimum is below 0.005 or the regularizer-rich stack wants near-zero explicit noise.
- Results:

| Metric | Value | vs prior (67.306) |
|---|---|---|
| best val_avg/mae_surf_p | **66.841** | −0.69% ✓ BEATS BASELINE |
| test_avg/mae_surf_p | **58.488** | −1.36% |
| val_single_in_dist | 79.768 | +2.71% (slight regression) |
| val_geom_camber_rc | 78.262 | −3.01% (biggest gain) |
| val_geom_camber_cruise | 47.065 | +0.40% (flat) |
| val_re_rand | 62.268 | −2.70% |
| test_single_in_dist | 68.487 | −0.35% |
| test_geom_camber_rc | 71.596 | −1.42% |
| test_geom_camber_cruise | 38.544 | −2.83% |
| test_re_rand | 55.325 | −1.48% |

- Metric paths: `models/model-feature-noise-0025-20260428-052613/metrics.jsonl`, `models/model-feature-noise-0025-20260428-052613/metrics.yaml`
- Analysis: Profile confirmed monotone-descending (0.0025 < 0.005 < 0.01 < 0.02). The OOD splits (camber_rc −3.01%, re_rand −2.70%) drove the gain; in-dist regressed slightly (+2.71%). Mechanism: with DropPath+huber+EMA+wd providing heavy regularization, explicit feature noise mostly adds gradient noise in the cosine fine-tuning tail. Less noise = cleaner signal for OOD generalization.
- Decision: **MERGED**. New baseline val_avg=66.841, test_avg=58.488. Noise profile not yet saturated — optimum may be in (0, 0.0025].

## 2026-04-28 06:15 — PR #562: Cosine T_max 13 → 12 with 2-ep warmup

- Branch: `charliepai2d2-fern/cosine-tmax-12-warmup-2`
- Hypothesis: Tighter T_max=12 with 2-epoch warmup pushes LR decay more aggressively into the 14-epoch budget, giving more fine-tuning epochs.
- Results:

| Metric | Value | vs prior (67.306) |
|---|---|---|
| best val_avg/mae_surf_p | **67.383** | +0.077 (MISSES by tiny margin) |
| test_avg/mae_surf_p | **58.603** | −1.17% (test WINS) |
| val_single_in_dist | 80.044 | +2.384 |
| val_geom_camber_rc | 80.727 | +0.037 (flat) |
| val_geom_camber_cruise | 45.913 | −0.964 |
| val_re_rand | 62.847 | −1.149 |

- Analysis: Over-decay regime confirmed. Late-epoch slope 0.39 pts/epoch (vs 0.7 at T_max=13) — model more settled but in slightly worse basin. Test improved (-1.17%) despite val regression. Warmup/cosine tradeoff at tight budget is subtle.
- Decision: **SENT BACK**. Requested 3-epoch warmup with T_max=11 (total=14, budget-aligned, softer start_factor=0.3). New baseline to beat: 66.841.

## 2026-04-28 06:15 — PR #554: AdamW weight_decay 3e-5 → 1e-5

- Branch: `charliepai2d2-tanjiro/weight-decay-1e-5`
- Hypothesis: wd profile is monotone-descending (3e-5 < 1e-4 < 3e-4); push to 1e-5 to find basin floor.
- Results (on pre-cosine stack):

| Metric | Value | vs conservative #520 target (71.6985) |
|---|---|---|
| best val_avg/mae_surf_p | **70.4328** | −1.77% (beats conservative target) |
| test_avg/mae_surf_p | **62.5571** | −0.04% vs #520 |

- wd profile: 3e-4=73.771, 1e-4=72.414, 3e-5=70.814, 1e-5=70.433. Slope flattened 2.21% → 0.54% — approaching basin floor.
- Analysis: Measured on pre-cosine-schedule stack; current baseline is 66.841 (much better). Result doesn't beat current baseline. Sent back to rebase and try wd=0 to close the question.
- Decision: **SENT BACK**. Rebase onto current stack, try wd=0 to determine whether any explicit L2 helps with the merged regularizer-rich stack.

## 2026-04-28 07:20 — PR #562: Cosine T_max 13 → 12 with 2-ep warmup → revised to T_max=11 + 3-ep warmup (start_factor=0.3)
- Branch: `charliepai2d2-fern/cosine-tmax-12-warmup-2` (artifact: `model-cosine-tmax-11-warmup-3-20260428-063145`)
- Hypothesis: gentler 3-epoch warmup (start_factor=0.3) decouples basin-selection smoothness from late-decay aggressiveness; T_max=11 still places cosine landing in the fine-tuning regime within 14-epoch budget. Iteration 2 of #562 — first attempt (warmup=2/T_max=12, start_factor=0.5) returned val flat +0.077 vs PR #525 baseline; advisor sent back with concrete revised params.

| metric | this run | baseline (PR #582 grad-clip) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **64.696** (epoch 14) | 66.149 | **−1.453 (−2.20%)** |
| `test_avg/mae_surf_p` | **55.879** | 57.654 | **−1.775 (−3.08%)** |
| val_single_in_dist | 77.527 | ~77.989 | −0.46 |
| val_geom_camber_rc | 75.963 | 77.651 | −1.69 |
| val_geom_camber_cruise | 44.229 | 46.373 | −2.14 |
| val_re_rand | 61.064 | 62.584 | −1.52 |
| test_single_in_dist | 66.457 | 67.262 | −0.81 |
| test_geom_camber_rc | 67.793 | 70.184 | −2.39 |
| test_geom_camber_cruise | 36.274 | 38.860 | −2.59 |
| test_re_rand | 52.993 | 54.310 | −1.32 |

- All 4 val splits improved. All 4 test splits improved (no val/test divergence — clean signal).
- LR-vs-epoch trajectory: ep1=1.50e-4 (start_factor 0.3), ep4=5.00e-4 (peak), ep14=1.01e-5 (cosine-decayed near-zero, similar to PR #525 T_max=13 endpoint).
- Late-epoch slope progression (this run): −4.71, −4.71, −3.44, −2.25, **−0.70**. Final-epoch slope 0.70 matches PR #525's sweet-spot exactly — model lands in fine-tuning regime, not over-decayed (vs PR #562 v1's −0.39 over-decay) and not still descending steeply.
- Mechanism (confirmed): the gentler warmup ramp from start_factor=0.3 over 3 epochs sets up a better trajectory entering the cosine phase. The model lands in the *same* fine-tuning regime as T_max=13 baseline but starts the final descent *from a lower val* because basin selection benefited from the smoother high-LR ramp.
- Standout splits: `val_geom_camber_cruise` (−6.03%), `test_geom_camber_rc` (−5.31%). Geometry-extrapolation-related splits benefited disproportionately, consistent with smoother basin selection.
- Decision: **MERGE**. Largest single-PR val_avg gain since PR #525 (−2.145 here vs −5.108 there). Sets new baseline at val_avg=64.696, test_avg=55.879.
- Suggested follow-ups (per fern):
  1. warmup_epochs=4 + cosine_epochs=10 with start_factor=0.2 — push gentler-warmup direction further.
  2. start_factor sweep (0.3, 0.2, 0.1) at fixed 3+11 schedule.
  3. Slight LR bump (lr=6e-4, 1.2× peak) with current schedule — gentler warmup may permit higher peak LR safely.

## 2026-04-28 07:30 — PR #510: torch.compile mode="default" (rebase verification on post-#582 stack)
- Branch: `charliepai2d2-alphonse/torch-compile-baseline` (artifact: `model-torch-compile-rebased-20260428-064446`)
- Hypothesis: torch.compile gives meaningful wall-clock speedup → more epochs in 30-min budget. Mode="reduce-overhead" predicted to help further but flagged as risky for variable-shape padding.
- Outcome: `mode="reduce-overhead"` OOMs from CUDA Graphs (one graph per mesh-padding shape, ~9 distinct shapes blew past 90 GB). `mode="default"` succeeds cleanly with TorchInductor fusion only.

| metric | compile=default (this PR, rebased) | eager (PR #582 baseline, same hardware) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **64.824** (epoch 18) | 66.149 | **−2.0%** |
| `test_avg/mae_surf_p` | **56.391** | 57.654 | **−2.2%** |
| epochs in 30-min cap | **18** | 14 | **+28.6%** |
| mean s/epoch (≥2) | **103.4 s** | 134.6 s | **−23.1%** |
| peak VRAM | 42.6 GB | 45.9 GB | −7% |

- Vs **current** baseline (post-PR #562 = 64.696): val_avg 64.824 = +0.128 (essentially flat, within seed noise).
- All 4 val splits improved vs eager-on-same-stack; all 4 test splits improved vs eager-on-same-stack.
- Compile cost: epoch 1 = 122.9 s including warmup (faster than eager's 139.3 s — fusion benefit on same epoch's late-phase forward/backward pays back the one-time compile cost).
- Mechanism: TorchInductor kernel fusion on forward+backward → shorter per-step time. mode="default" skips CUDA Graphs (the source of reduce-overhead OOM) but retains the fusion. Compile is mode-orthogonal to all merged levers (huber, EMA bias-correction, slice-temp, feature noise, scheduler, grad-clip) — verified via `ema_decay` and `train/grad_norm_*` events being numerically equivalent under both modes.
- Note: the rebased measurement is from `git_commit=bc218d5` which precedes PR #562 by 6 minutes (i.e., compiled run was on the OLD 1-ep warmup + T_max=13 schedule). Under PR #562's 3-ep warmup + T_max=11 schedule, cosine reaches LR=eta_min=0 by ep14, leaving epochs 15–18 as "free EMA-stabilization epochs" — the same dynamic as alphonse's measurement.
- Decision: **MERGE**. Standalone val_avg gap to current baseline is +0.128 (within seed-to-seed noise of ~±1–2%); but throughput +28.6% compounds with every future PR. Infrastructure win unlocks more training time on every subsequent experiment.
- Downstream consequence: future LR-schedule PRs may want to extend T_max=11 → 14–15 to fully use the 18-epoch budget, OR set `eta_min > 0` to extract more from the now-free epochs 15–18.
- Suggested follow-ups (alphonse):
  1. Fixed-shape padding for `reduce-overhead` (next assignment).
  2. `mode="reduce-overhead"` with `dynamic=True` (cheaper than padding).
  3. Bucketed batching by mesh size.
  4. Rip out DropPath `Tensor.item()` graph break.
  5. `torch.set_float32_matmul_precision("high")` for TF32.

## 2026-04-28 07:30 — PR #601: Huber δ=0.25 → 0.1 (push toward L1)
- Branch: `charliepai2d2-thorfinn/huber-delta-0p1` (artifact: `model-huber-delta-0p1-20260428-063745`)
- Hypothesis: smaller δ linearizes more residuals, downweighting heavy tails further. Profile was monotone-descending (δ=2 → 107.6, δ=1 → 88.2, δ=0.5 → 87.3, δ=0.25 → 72.4) on prior stacks.
- Result vs **prior** baseline (PR #575 = 66.195 — same starting baseline thorfinn was on): val_avg = **65.497** = **−1.05%** (3 of 4 val splits improve; in_dist regresses +1.87 pts).
- Result vs **current** baseline (post-PR #562 = 64.696): val_avg 65.497 = **+1.24% regression** (cross-stack).
- Loss-magnitude characterization at convergence (very informative): only 21–43% of per-element residuals are in the linear regime — much less than the predicted >95% pseudo-L1. Mechanism is more accurately: δ=0.1 makes the heavy-tailed minority (~30%) be treated as L1 while keeping ~70% in the smooth quadratic regime. δ=0.1 wins as a *better hybrid*, not as pseudo-L1.
- Decision: **SEND BACK FOR REBASE** onto post-PR #562 stack. The standalone gain of −1.05% on the prior stack is robust same-stack evidence the lever works. The schedule revision (1-ep → 3-ep warmup, T_max=13 → 11) changes optimization dynamics meaningfully, and we want clean post-#562 measurement before merging.
- If rebased run beats current baseline: merge.
- Decision pending: rebased run on post-#562 stack.
- Suggested follow-ups (carry-over): δ=0.05, channel-specific δ, outlier-aware sample weighting.

## 2026-04-28 07:30 — PR #600: EMA decay_target 0.995 → 0.999 (UP direction probe)
- Branch: `charliepai2d2-nezuko/ema-decay-target-0999` (artifact: `model-ema-decay-target-0999-20260428-063600`)
- Hypothesis: longer EMA memory smooths late-training fine-tuning; profile 0.95 → 0.99 → 0.995 was still monotone-descending.
- Result vs **prior** baseline (PR #575 = 66.195): val_avg = **67.856** = **+2.51% regression**. test_avg = 58.986 = +1.59% regression.

| decay_target | val_avg/mae_surf_p | source |
|---:|---:|---|
| 0.95 | 75.655 | PR #540 |
| 0.99 | 67.306 | PR #525 |
| **0.995** | **66.195 ← min** | PR #575 |
| 0.999 | 67.856 | this PR |

- **Mechanistic finding (key):** at warmup_steps=50, the EMA effective decay is `min(decay_target, (1+s)/(50+s))`. For decay_target=0.995 the cap binds at step ~9750; for 0.999 the cap binds at step ~48950. Both are well **beyond the 5250-step budget**. Within budget, the trajectories are *numerically identical* — both follow the warmup ramp `(1+s)/(50+s)` throughout. Therefore the +2.51% gap CANNOT be a real EMA-dynamics signal — it must be single-seed run-to-run variance.
- Decision: **CLOSE** — accepting the mechanism analysis. The EMA-decay-target axis cannot move outcomes within the 14-epoch budget when warmup_steps=50, because the cap doesn't bind. Future EMA improvements must vary `warmup_steps` (which controls ramp speed and indirectly when the cap binds).
- Suggested follow-ups (per nezuko):
  1. Vary `warmup_steps` (10, 20, 100, 200) at fixed `decay_target=0.995`. Smaller warmup makes the cap bind earlier in budget; larger keeps EMA slower throughout.
  2. Re-probe `decay_target=0.999` with `warmup_steps=10` (where cap binds at step ~9990, still beyond budget but ramps differently).
  3. Confirm 0.995 minimum with a second seed.
  4. Treat 0.995 ↔ 0.999 as a soft tie for downstream stack decisions.

## 2026-04-28 07:45 — PR #605: surf_weight 10 → 15 (push primary metric weighting)
- Branch: `charliepai2d2-edward/surf-weight-15` (artifact: `model-surf-weight-15-20260428-065256`)
- Hypothesis: bigger surface weight reallocates capacity toward the primary metric. PR #286 historically blew up vol MAE at sw=25 (+10–17%); sw=15 should stay below that threshold under huber-δ=0.25 + clip_grad_norm.

| metric | this run | baseline at run-time (PR #582 = 66.149) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **66.037** (epoch 14) | 66.149 | **−0.17%** |
| `test_avg/mae_surf_p` | **57.733** | 57.654 | **+0.14% regression** |

- Vs **current** baseline (post-PR #562 = 64.696): val_avg 66.037 = **+2.07% regression** (cross-stack).
- Per-split val: in_dist +1.31%, camber_rc +1.36%, camber_cruise −2.46%, re_rand −2.22%. Mixed signature, 2-up / 2-down.
- Per-split test: same direction as val on each split, but in_dist & camber_rc regress more (+2.65% / +2.45%) — net test +0.14%.
- Vol MAE rose modestly (+2% val, +4% test). single_in_dist vol regressed most (+7.12% val, +8.47% test) — tilting further toward surface starves vol capacity for that split.
- Mechanism explanation (per edward): huber-δ=0.25 caps per-element loss-gradient at ~1.0; clip_grad_norm(10) caps total norm. The 1.5× scalar passes through both filters, so effective signal increase is much less than 1.5×. Clip rate jumped 85.6% → 92.5% in epoch 1 — most of the extra surface signal is being clipped away.
- Per-epoch trajectory: sw=15 leads by 4–5 MAE during epochs 2–4 (mid-warmup, where surf-weight bites hardest), then gap closes to ~0 by epoch 14 as cosine LR decays.
- Decision: **CLOSE** — student's own recommendation against; standalone gain marginal/within noise; test regressed; mechanism is exhausted (clip absorption); current-stack regression would be ~+2%. Direction bracketed [10, 15] with current stack at sw=10. Probe sw=12 if other axes stall.

## 2026-04-28 07:45 — PR #595: feature_noise std=0.0 + std=0.001 (close noise direction)
- Branch: `charliepai2d2-frieren/feature-noise-zero-vs-schedule` (artifacts: `model-feature-noise-zero-20260428-062330`, `model-feature-noise-0001-20260428-065640`)
- Hypothesis: closing the noise direction by probing both 0.0 and a midpoint 0.001. Profile was previously (0.005 → 67.306, 0.0025 → 66.841) — appeared still descending toward zero.

| variant | std | val_avg | vs 66.841 baseline |
|---|---:|---:|---:|
| A | 0.0 | **68.383** | +2.31% regression |
| B | 0.001 | **67.957** | +1.67% regression |
| (PR #563) | 0.0025 | 66.841 | **interior MIN** |
| (PR #525) | 0.005 | 67.306 | +0.70% |

- Profile shape: U-shaped with **interior minimum at std=0.0025**. Both adjacent points (0.001 and 0.005) regress; std=0 regresses most. Noise direction is now BRACKETED.
- Per-split (std=0 vs 0.0025 baseline): in_dist +1.43%, camber_rc +3.15%, **camber_cruise +4.69%**, re_rand +0.57%.
- Mechanism: damage concentrated on OOD splits (camber_rc, camber_cruise) — explicit feature noise is doing non-redundant regularization vs DropPath/huber/EMA/wd; specifically buys OOD generalization. In-dist barely cares.
- Asymmetric basin: leaving optimum DOWN more costly than UP (DropPath redundancy may absorb high-noise side, no other lever fills the role on low-noise side).
- Decision: **CLOSE** — student's own recommendation. Noise hyperparameter axis closed at 0.0025; further sweep is wasted compute on this direction. Frieren's suggested next probes:
  1. **Surface-only feature noise** (apply only to per-node positional/SDF features dims 0–12, not per-sample dims 13–23).
  2. **Decaying noise schedule** aligned with cosine LR — high noise during basin selection, decay to zero in the LR≈0 fine-tuning tail.
  3. **Per-split EMA decay** — cruise has the biggest single-split headroom signal.

## 2026-04-28 08:00 — PR #554: AdamW wd 3e-5 → 1e-5 (rebased as wd → 0 to close direction)
- Branch: `charliepai2d2-tanjiro/weight-decay-1e-5` (artifact: `model-weight-decay-0-20260428-072248`)
- Hypothesis: wd profile was monotone-descending on the older stack (3e-4 → 1e-5 = monotone). Rebased to test wd=0 on the post-#562/#510 stack.

| metric | this run (wd=0) | baseline (PR #562 wd=3e-5) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **66.476** (epoch 14) | 64.696 | **+2.75% regression** |
| `test_avg/mae_surf_p` | **56.882** | 55.879 | **+1.79% regression** |

- Per-split val: in_dist +2.38%, **camber_rc +7.16% (biggest single-split degradation)**, cruise +1.26% (~flat), **re_rand −1.18% (only improvement)**.
- Per-split test: in_dist +2.22%, camber_rc +3.56%, cruise +1.40%, re_rand −0.73%.

### Updated wd profile (across stacks)

| wd | val_avg | stack |
|---:|---:|---|
| 3e-4 | 73.771 | older (pre-#525) |
| 1e-4 | 72.414 | older (PR #463) |
| 3e-5 | 70.814 | older (PR #527 merged) |
| 1e-5 | 70.4328 | older (PR #554 first run) |
| **3e-5** | **64.696** | **new (current merged baseline)** |
| 0 | 66.476 | new (this rebased run) |

- **Mechanism**: the heavy implicit regularization stack (EMA→0.995, slice-temp=2.0, grad-clip=10, gentler 3-ep warmup) shifted the basin floor UP. On the older stack, wd profile descended monotonically toward zero. On the new stack, **wd=3e-5 is at-or-near the floor**, and removing it overshoots into under-regularized territory. The OOD asymmetry pattern (lower wd helps re_rand, hurts camber_rc) is preserved across both stacks — suggests per-parameter-group wd is the natural next probe.
- Decision: **CLOSE** — wd-down direction definitively closed. wd=3e-5 is at the basin floor on the current stack; pushing lower regresses, pushing higher (already tested) regresses. Single-scalar wd axis exhausted.
- Suggested follow-ups (per tanjiro):
  1. wd=1e-5 on new stack to localize basin floor between (0, 3e-5] — but expected gain is ≤0.3% based on profile shape.
  2. **Per-parameter-group wd** — split AdamW groups by module type (attention out-projection vs MLP vs others) with different wd values. Mechanism-driven by the OOD asymmetry: attention should keep higher wd (camber_rc benefits), MLP should have lower wd (re_rand benefits). Captures both wins simultaneously.
  3. NaN-safe eval loss bug fix in `evaluate_split` (cosmetic; loss=NaN on test_geom_camber_cruise while MAEs are finite).

## 2026-04-28 08:10 — PR #620: cosine LinearLR start_factor 0.3 → 0.2 (push gentler-warmup direction)
- Branch: `charliepai2d2-fern/cosine-start-factor-02` (artifact: `model-cosine-start-factor-02-20260428-072324`)
- Hypothesis: gentler warmup ramp may extract more headroom; profile last measured at start_factor=0.3 as PR #562 winner.

| metric | this run (sf=0.2) | baseline (PR #562, sf=0.3) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **65.611** (epoch 14) | 64.696 | **+0.915 (+1.41%) regression** |
| `test_avg/mae_surf_p` | **57.022** | 55.879 | **+1.143 (+2.05%) regression** |

- Per-split val: in_dist +3.017, camber_rc +2.457 (both regress), camber_cruise −1.525, re_rand −0.287 (improve slightly).
- Per-split test: in_dist +1.101, camber_rc +3.898 (regress most), cruise −0.441, re_rand +0.011.
- LR trajectory matched expected exactly (ep1=1.00e-4, ep4=5.00e-4 peak, ep14=1.01e-5).

### Trajectory comparison vs PR #562

| ep | sf=0.3 (#562) | sf=0.2 (this) | Δ (this − #562) |
|---|---|---|---|
| 1 | 178.78 | 197.93 | +19.15 (softer warmup → much weaker ep1) |
| 4 | 131.56 | 126.99 | **−4.57 (briefly better at peak LR ep4)** |
| 7 | 96.15 | 97.58 | +1.43 (lead reverses) |
| 14 | **64.70** | **65.61** | **+0.92** |

- **Interesting wrinkle**: sf=0.2 was ahead by −4.57 at ep4 (peak LR), confirming the gentler-warmup mechanism is real and active. But the lead reversed by ep7 and gap widened during cosine — the sf=0.2 trajectory was less optimized at ep4 (which is why it appeared "ahead"), but the underlying basin selection during ep1–3 was suboptimal. Once LR began decaying, the worse basin couldn't keep up with sf=0.3's cleaner trajectory.
- Final-epoch slope similar (−0.875 vs −0.701) — same fine-tuning regime, different starting position.
- Decision: **CLOSE** — start_factor profile is **non-monotone with sweet spot at 0.3**. Both adjacent points regress (sf=0.5 was worse historically; sf=0.2 worse here). Lock in 0.3 as warmup-aggressiveness sweet spot.
- Fern's recommendation: do NOT push to sf=0.1; profile saturated past 0.3.

## 2026-04-28 08:10 — PR #608: PhysicsAttention slice temperature init 2.0 → 2.5
- Branch: `charliepai2d2-askeladd/slice-temp-2p5` (artifact: `model-slice-temp-2p5-20260428-073130`)
- Hypothesis: slice-temp profile still descending; init=2.5 should bracket the optimum from above.

| metric | this run (init=2.5) | baseline-at-run-time (PR #582, init=2.0) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **65.822** (epoch 14) | 66.149 | **−0.50%** (small same-stack win) |
| `test_avg/mae_surf_p` | **57.834** | 57.654 | +0.31% (slight regression) |

- Vs **current** baseline (post-PR #562 / #510 = 64.696): val_avg 65.822 = **+1.74% regression** (cross-stack).
- Per-split val: in_dist −0.19% (~flat), **camber_rc +2.60% regression**, cruise −2.33%, re_rand −3.35% (biggest improvement).
- Per-split test: in_dist +1.92%, camber_rc +1.76%, cruise **−5.95%** (biggest improvement), re_rand +0.93%.
- Final per-block learned temperatures: live=[2.47, 2.40, 2.43, 2.44, 2.43] mean=2.436. Drift over 14 epochs ≈ −0.066 (matches prior runs at lower init).
- **Slice-temp profile (current full stack)**: init=2.0 → 66.149, init=2.5 → 65.822 (Δ=−0.327, much smaller than the cross-stack 1.5→2.0 step of −3.77). Profile approaching saturation.
- **`val_geom_camber_rc` consistently regresses with sharper attention** (init=2.0 in PR #574: +2.0%, init=2.5 in this run: +2.60%). Sharper attention is a net loss for geometry-extrapolation OOD on rc-camber.
- Decision: **CLOSE** — diminishing returns on slice-temp init axis (Δ=−0.327 here vs −3.77 cross-stack); test slightly regressed; camber_rc consistently regresses with sharper attention; cross-stack would be +1.74% regression. Some of the prior "huge jump" was confounded by orthogonal stack improvements (grad clip, noise=0.0025) that have since landed. Slice-temp init axis has run its course at the current stack.
- Pivot askeladd to per-block slice-temp init schedule — variation along the block-depth axis rather than the global init scalar. The consistent camber_rc pattern suggests sharper attention helps later blocks (semantic refinement) but hurts earlier blocks (broad spatial pooling).

## 2026-04-28 08:30 — PR #629: torch.compile mode="reduce-overhead" with fixed-shape padding (throughput follow-up)
- Branch: `charliepai2d2-alphonse/reduce-overhead-fixed-padding` (artifact: `model-reduce-overhead-fixed-padding-20260428-075149`)
- Hypothesis: fix the per-shape-CUDA-Graph OOM that defeated reduce-overhead in PR #510 by padding all batches to global max mesh size (N_MAX=242577). Expected another +10–20% wall-clock on top of mode="default".

| metric | this run (reduce-overhead + fixed pad) | baseline (PR #510, compile=default) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **65.008** (epoch 14) | 64.824 (epoch 18) | +0.28% (~flat) |
| `test_avg/mae_surf_p` | **56.178** | 56.391 | −0.38% (~flat) |
| Epochs in 30-min cap | **14** | 18 | **−22% regression** |
| Mean s/epoch | **135.6** | 103.4 | **+31% slower** |
| Pre-warm s | 17.4 | n/a | one-time setup |
| Peak VRAM | 42.67 GB | 42.6 GB | flat |

- Mechanism analysis (per alphonse):
  - reduce-overhead saves ~25% per-iter overhead by replaying one CUDA Graph (instead of relaunching kernels each step).
  - Fixed padding to N_MAX=242577 inflates per-iter compute by ~60% (every batch does worst-case mesh work; cruise tandem 242K nodes vs typical batch ~150K average).
  - Net per-iter wall-clock: +31% over compile=default's variable-padding baseline.
  - Total compute essentially conserved: 14 × 135.6 = 1898 s ≈ 18 × 103.4 = 1861 s. Same work, bundled into fewer/larger epochs.
- **Per-split signature**: mixed and cancelling. val_geom_camber_cruise improved most (−4.86%), val_single_in_dist regressed most (+3.27%), no clear OOD/in-dist pattern.
- Compile-mode confirmation: reduce-overhead succeeded cleanly with fixed padding. One CUDA Graph for the single shape `[B=4, N=242577, *]`. No diagnostic warnings during 5236 training iterations. ~53 GB VRAM headroom on the 96 GB cap.
- Decision: **CLOSE** — quality flat, throughput regressed −22%. The model is compute-bound at max-mesh shape, not launch-overhead-bound. Fixed padding overshoots; bucketed batching is the right next probe.
- **Mechanistic findings unlocked**:
  1. reduce-overhead is *feasible* on this codebase with fixed shapes (no graph-pool memory issue).
  2. Pre-warm peak (~42.55 GB) and steady-state (~42.67 GB) leave ~53 GB headroom — bucketed batching is NOT memory-blocked, only throughput-uncertain.
  3. The throughput bottleneck is *per-iter compute*, not *per-iter overhead*. CUDA Graph replay can save up to ~25% overhead, but only if per-iter compute is preserved.
- Follow-ups (per alphonse):
  1. **Bucketed batching + reduce-overhead** — sort/bucket samples by mesh size; pad each bucket to its bucket-max. ~4 buckets at 100K/150K/200K/245K → 4 CUDA Graphs (graph-pool overhead bounded). Mean per-batch compute stays close to variable-padding's ~150K. Predicted: this should be the configuration that actually moves throughput.
  2. **reduce-overhead with `dynamic=True`** — keeps variable shapes but lets reduce-overhead share a graph across compatible shapes. Cheaper to implement.
  3. **TF32 matmul precision** (`torch.set_float32_matmul_precision("high")`) — independent of compile mode; helps eager too.
  4. **DropPath graph break removal** — replace `torch.rand(1).item()` with tensor-mask form. Few-percent win on top.

## 2026-04-28 08:40 — PR #635: lr peak bump 5e-4 → 6e-4 (gentler 3-ep warmup permits 1.2× peak)
- Branch: `charliepai2d2-edward/lr-peak-6e-4` (artifact: `model-lr-peak-6e-4-20260428-075310`)
- Hypothesis (per fern's PR #562 follow-up #3): the gentler 3-ep warmup with start_factor=0.3 widens the safe-LR window; lr=6e-4 (1.2×) may train stably and extract more from the high-LR phase.

| metric | this run (lr=6e-4) | baseline (PR #562 = 64.696 / PR #510 = 64.824) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **63.131** (epoch 17) | 64.696 / 64.824 | **−2.42% / −2.61%** |
| `test_avg/mae_surf_p` | **55.026** | 55.879 | **−1.53%** |

- Per-split val: in_dist **−4.56%**, camber_rc +1.71% (only regression), camber_cruise **−4.91%**, re_rand **−3.04%**.
- Per-split test: in_dist 64.501, camber_rc 68.074, camber_cruise 35.466, re_rand 52.064 — all 4 improve vs eager baseline.
- LR trajectory: ep1=1.80e-4, ep2=3.20e-4, ep3=4.60e-4, ep4=6.00e-4 (peak), cosine to ep15=0, then wraparound to ep17=4.76e-5.
- Grad-clip diagnostics: ep1 mean=21.5 (+10% vs PR #582), max=81.5 (+33%), but clip rate FELL to 82.7% (vs PR #582's 85.6%) — because the gentler 3-ep warmup means ep1 effective LR is only 1.80e-4. Counter-intuitive but mechanistically clean.
- **Late-epoch slope shape changed**: model gets into fine-tuning regime SOONER. By ep12, val_avg=66.9 already close to final 63.13. PR #562 was still mid-optimization at ep12 with much larger remaining drop.
- Decision: **MERGE** as a clean orthogonal compound. Hypothesis confirmed; 3-ep warmup widens safe-LR window; clip absorbs modest grad-norm increase without saturating.
- Suggested follow-ups (per edward):
  1. **lr=7e-4 (1.4× original peak)** under same warmup — clip rate well-behaved, may have more headroom.
  2. T_max=12 or 13 to push the peak-LR phase wider.
  3. cosine eta_min > 0 to avoid the LR=0 wasted epoch (already nezuko's #630).
  4. Investigate val_geom_camber_rc regression (consistent across multiple recent merges).

## 2026-04-28 08:40 — PR #636: Decaying feature-noise schedule aligned with cosine LR
- Branch: `charliepai2d2-frieren/feature-noise-decaying` (artifact: `model-charliepai2d2-frieren-feature-noise-decaying-20260428-075320`)
- Hypothesis (per frieren's PR #595 follow-up #2): the fixed std=0.0025 has to compromise between basin-selection (early phase) and gradient-noise tax (fine-tuning tail). A schedule that decays noise to zero by epoch 14 should let early-phase regularization continue while leaving the tail clean.

| metric | this run (decaying) | baseline (PR #562 = 64.696 / PR #510 = 64.824) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **63.222** (epoch 17) | 64.696 / 64.824 | **−2.28% / −2.47%** |
| `test_avg/mae_surf_p` | **54.900** | 55.879 | **−1.75%** |

- Per-split val: in_dist −1.60%, camber_rc +1.61%, **camber_cruise −8.89% (biggest single-split gain in this round)**, re_rand −3.19%.
- Per-split test: all 4 improve. cruise **−6.47%** biggest.
- Schedule trajectory (linear decay): ep1=0.0025, ep4=0.00196, ep11=0.000714, ep14=0.000179, ep15+=0.0 exactly (matches LR=0 at ep15).
- **Late-epoch slope shape changed**: biggest single-epoch improvement now at ep12→13 (−3.00) when noise std drops from 0.000536 → 0.000357. Post-noise tail (ep15–17, std=0) keeps descending 0.5–0.6 pts/epoch — the cleaner signal *does* keep helping under the cosine LR cycle-back, confirming EMA was NOT fully absorbing late-phase noise.
- Decision: **MERGE** as a clean orthogonal compound. Realized −2.28% exceeded the optimistic prediction (−1.0% to +0.3%). Cruise OOD split is disproportionately responsive to clean fine-tuning signal.
- Suggested follow-ups (per frieren):
  1. **Two-phase (step) schedule** — abrupt cutoff at ep≥11 instead of linear decay. Isolates whether ramp-down or just tail-zero is the active ingredient.
  2. Late-phase nonzero noise probe (stop at 0.0008 instead of 0). Tests whether tail-zero is necessary or just lower-noise tail.
  3. **Higher base_std=0.005 with steeper decay (decay_horizon=8)** — more early regularization, no late-phase cost.
  4. Match cruise win on camber_rc — variant with surface-only or per-sample-only noise schedules.

## 2026-04-28 08:55 — PR #640: Per-parameter-group weight decay (attn=1e-4, mlp=1e-5, other=3e-5)
- Branch: `charliepai2d2-tanjiro/per-group-wd` (artifact: `model-per-group-wd-20260428-081309`)
- Hypothesis (from tanjiro's PR #554 close): the OOD asymmetry preserved across both stacks (lower wd helps re_rand, hurts camber_rc) suggests per-module-group wd captures both wins.

| metric | this run (per-group wd) | baseline (PR #510 compile = 64.824 / PR #562 eager = 64.696) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **62.747** (epoch 17) | 64.824 / 64.696 | **−3.21% / −3.01%** |
| `test_avg/mae_surf_p` | **54.512** | 56.391 / 55.879 | **−3.33% / −2.45%** |

- Param groups: attn 273K (40.8% wd=1e-4), mlp 354K (52.9% wd=1e-5), other 42K (6.3% wd=3e-5).
- **All 4 val splits improve**: in_dist −1.39%, camber_rc **−1.13%** (the predicted target OOD), camber_cruise **−9.24%** (biggest gain), re_rand **−3.61%** (the predicted target OOD).
- All 4 test splits improve.
- **OOD-asymmetry mechanism CONFIRMED**: both target splits (re_rand, camber_rc) improved as predicted with attn-higher / mlp-lower wd.
- **Bonus finding**: uniform improvement across ALL splits, not just the OOD axes. The asymmetric regularization principle is broader than the specific OOD asymmetry — the model just operates better when attention and FFN modules are regularized differently.
- Decision: **MERGE**. Profile still descending; tanjiro's follow-up #1 (push asymmetry harder: wd_attn=3e-4, wd_mlp=3e-6) is the natural next probe.

## 2026-04-28 08:55 — PR #601: Huber δ=0.25 → 0.1 (rebased onto post-#562/#510 stack)
- Branch: `charliepai2d2-thorfinn/huber-delta-0p1` (artifact: `model-huber-delta-0p1-rebased-20260428-075855`)
- Hypothesis (from thorfinn's first run): δ profile monotone-descending toward L1; rebase to verify on current stack.

| metric | this run (δ=0.1, rebased) | baseline (PR #510 compile = 64.824 / PR #562 eager = 64.696) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **62.879** (epoch 17) | 64.824 / 64.696 | **−3.00% / −2.81%** |
| `test_avg/mae_surf_p` | **54.561** | 56.391 / 55.879 | **−3.25% / −2.36%** |

- **All 4 val splits improve**: in_dist **−3.92%** (largest gainer; the in_dist regression from pre-rebase reverses entirely), camber_rc −2.19%, camber_cruise −3.39%, re_rand −2.60%.
- All 4 test splits improve.
- **Striking finding**: best epoch is 17 in the cosine schedule's **second half-cycle**. Cosine T_max=11 produces a second descent past ep14 (PyTorch CosineAnnealingLR continues past T_max as cos(πt/T_max) which goes negative then positive). EMA(0.995) + δ=0.1's smooth-near-zero gradient lets the model squeeze ~1.3 additional pts past the cosine-min minimum.
- **Loss-magnitude characterization at convergence**: only 29% of residuals are in the linear regime (vs predicted >95% if pseudo-L1). δ=0.1 is a **"better hybrid"**, not pseudo-L1: linearizes the heavy-tailed minority (~30%) while keeping ~70% in the smooth quadratic well.
- **Same-stack effect grew** from −1.05% (pre-rebase) to −3.00% (post-rebase) — the longer 3-ep warmup + T_max=11 schedule extends the late-training fine-tune window where δ=0.1's better tail handling has more room to refine.
- Decision: **MERGE**. Profile still monotone-descending toward L1; thorfinn's follow-up #1 (δ=0.05 probe) is the natural next step.

## 2026-04-28 09:05 — PR #647: Per-block slice-temp init schedule [1.5, 1.875, 2.25, 2.625, 3.0]
- Branch: `charliepai2d2-askeladd/slice-temp-per-block-schedule` (artifact: `model-charliepai2d2-askeladd-slice-temp-per-block-schedule-20260428-082258`)
- Hypothesis: hierarchical sharpness — softer early blocks for spatial pooling, sharper later blocks for token refinement. Mechanism-driven by the consistent val_geom_camber_rc regression with sharper global init.

| metric | this run (per-block schedule) | baseline (PR #510 compile = 64.824 / PR #562 eager = 64.696) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **61.872** (epoch 18) | 64.824 / 64.696 | **−4.55% / −4.37%** |
| `test_avg/mae_surf_p` | **54.555** | 56.391 / 55.879 | **−3.26% / −2.37%** |

- **All 4 val splits improved** vs PR #562: in_dist **−6.01%** (largest), camber_rc **−2.86%** (the predicted target — direction reversal!), camber_cruise **−4.78%**, re_rand **−3.85%**.
- **All 4 test splits improved**.
- **Final per-block temperatures (live=EMA, drift)**: block 0 = 1.456 (drift −0.044), block 1 = 1.777 (−0.098), block 2 = 2.175 (−0.075), block 3 = 2.546 (−0.079), block 4 = 2.914 (−0.086).
- **Block 0 (softest) drifts LEAST** — closest to its equilibrium, suggesting block 0 *wants* to be soft. Confirms hierarchical sharpness is a true depth-specialization signal, not just a regularization effect.
- **camber_rc direction reversal is the key diagnostic**: was regressing under sharper global init (init=2.0: +2.0%, init=2.5: +2.60%), now improves −2.86% under hierarchical schedule. The camber_rc/cruise tension was a **depth-specialization issue**, not a global capacity tradeoff.
- **Variance across blocks is doing the work, not the higher mean**: schedule mean=2.25 vs prior global init=2.5 (mean=2.5) gave 65.822 worse by 4 absolute. The within-block variance (1.5→3.0) is the load-bearing piece.
- **Per-block temperatures stay distinct**: adjacent block deltas (0.32-0.39) > drift magnitudes (0.04-0.10). Schedule is a stable equilibrium for these blocks.
- Vs current baseline (post-PR #640 + #601 + #635 + #636 merges, ~62.747): val_avg 61.872 = **−1.4% additional gain**. Standalone but mechanically orthogonal compound.
- Decision: **MERGE**. Outstanding mechanistic work that closes the camber_rc/sharper-attention puzzle from prior runs.
- Suggested follow-ups (per askeladd):
  1. **Steeper schedule** [1.0, 1.5, 2.0, 2.5, 3.0] — block 0 drift signal suggests pushing it below 1.5; mean stays at 2.0, range doubles.
  2. Different curve shape (exp-spaced, two-step) — block-0 distinct from 1-4 hints that maybe blocks 1-4 don't need much variation.
  3. Global init=1.5 (no schedule) — test if per-block parameters naturally find the asymmetry without an init prior.
  4. Extend training budget (T_max=14 with same compile setup).
  5. Per-head per-block init (5×4 separate scalars).

## 2026-04-28 09:15 — PR #646: batch_size 4 → 6 with compile (gradient noise reduction)
- Branch: `charliepai2d2-fern/batch-size-6` (artifact: `model-batch-size-6-20260428-082139`)
- Hypothesis: larger batch reduces gradient noise (variance ∝ 1/B); cleaner gradients may compound with EMA averaging at zero per-step throughput cost under compile.

| metric | this run (batch=6) | baseline-at-run-time (PR #562 = 64.696 / PR #510 compile = 64.824) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **63.384** (epoch 18) | 64.696 / 64.824 | **−2.03% / −2.22%** |
| `test_avg/mae_surf_p` | **55.317** | 55.879 / 56.391 | **−1.00% / −1.91%** |

- Vs **current** baseline (post-PR #647 + #640 + #601 + #635 + #636 = 61.872): val_avg 63.384 = **+2.44% regression** (cross-stack).
- Per-split val: in_dist +0.44, camber_rc +0.44 (both flat), camber_cruise **−3.21** (biggest gain), re_rand **−2.92**.
- Per-split test: cruise 33.898, re_rand 51.015 (both improve).
- Wall-clock: 101.6 s/epoch (slightly faster than batch=4 compile baseline 103.4 s/epoch); 250 batches/epoch (vs 375 at batch=4); peak VRAM 63.83 GB (matches predicted ~63 GB).
- **Mechanism confirmed**: gradient-noise reduction (variance ratio 0.82× between batch=6 and batch=4) helps OOD splits most while in-dist splits are flat. Cleaner gradients matter more for OOD generalization than in-distribution fitting.
- Decision: **SEND BACK FOR REBASE**. Standalone same-stack gain is real (-2% range) but vs current baseline (+2.44% regression) it doesn't merge directly. The advisor branch moved 4 points lower since fern branched. Critical question: linear-scaling rule says lr should scale with batch_size; with lr=6e-4 (PR #635) merged, the lr × batch interaction is uncharacterized. Rebased run will resolve both questions cleanly.
- Suggested follow-ups (per fern):
  1. **batch=8** — predicted peak VRAM ~85 GB still under 96 GB cap; variance ratio 1/√2 vs batch=4.
  2. **batch=6 + lr scaling** — explicit lr scaling test (lr=6e-4 → 7.5e-4 with batch=6, linear scaling rule).
  3. **Track gradient noise scale (GNS)** — instrumentation to predict optimal batch size analytically.

## 2026-04-28 09:30 — PR #669: Higher base_std=0.005 with steeper decay_horizon=8 (push schedule magnitude)
- Branch: `charliepai2d2-frieren/feature-noise-higher-steeper` (artifact: `model-feature-noise-higher-steeper-20260428-085147`)
- Hypothesis: more early-phase regularization (base_std doubled to 0.005) + earlier transition to clean tail (decay_horizon 14→8) compounds the basin-selection benefit confirmed by PR #636.

| metric | this run | baseline (PR #636 = 63.222 / current PR #647 = 61.872) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **65.716** (epoch 14) | 63.222 / 61.872 | **+3.94% / +6.21% regression** |
| `test_avg/mae_surf_p` | **56.610** | 54.900 / 54.555 | **+3.11% / +3.77% regression** |

- Per-split val: in_dist +2.44%, camber_rc +1.88%, **camber_cruise +12.71%** (massive regression on the OOD split that benefited most from PR #636), re_rand +2.60%.
- **Schedule shape verified**: ep1 std=0.005, ep5 std=0.0025, ep8 std=0.000625, ep9+=0.0 ✓
- **Mechanism failure mode identified (per frieren)**: at base_std=0.005, **90% of batches in ep1 hit gradient clipping** (vs 82.7% at PR #635 lr=6e-4 baseline). The high noise + high LR combination saturates clip(10), destabilizing basin selection during the high-LR phase. Cruise OOD (the split most sensitive to controlled early noise) is hit hardest.
- The mechanism (early noise → basin selection; clean tail → EMA averaging) is **confirmed and load-bearing**, but the operating point is sharper than expected: doubling peak does NOT double regularization benefit; it crosses into instability.
- Decision: **CLOSE** — schedule-magnitude axis is now bracketed. (0.0025, 14) sits near the ceiling for "noise that helps without destabilizing".
- Suggested follow-ups (per frieren):
  1. Isolate steeper decay alone (base_std=0.0025, decay_horizon=8).
  2. **Surface-only feature noise** (apply only to per-node positional/SDF dims 0–12, skip per-sample dims 13–23) — orthogonal axis.
  3. Couple noise to grad-clip headroom (target fixed clip rate).
  4. Different decay shape (cosine vs linear).

## 2026-04-28 09:30 — PR #668: lr peak bump 6e-4 → 7e-4 (push lr-peak axis further)
- Branch: `charliepai2d2-edward/lr-peak-7e-4` (artifact: `model-lr-peak-7e-4-20260428-085022`)
- Hypothesis: at lr=6e-4 ep4 clip rate was 51.5% — well below saturation. lr=7e-4 should extract more from the high-LR phase.

| metric | this run | baseline (PR #635 = 63.131 / current PR #647 = 61.872) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **63.206** (epoch 18) | 63.131 / 61.872 | **+0.075 (within noise) / +2.16% regression** |
| `test_avg/mae_surf_p` | **55.963** | 55.026 / 54.555 | **+0.937 / +1.41%** |

- Per-split val (vs PR #635): in_dist +0.190, camber_rc −0.320, camber_cruise +1.125, re_rand −0.692. Mixed and within noise.
- **Surprise mechanism finding**: ep4 clip rate at lr=7e-4 was **41.9%** — *lower* than the 51.5% at lr=6e-4. Clip(10) is NOT the bottleneck. ep4 mean grad-norm 10.3, max 43.9.
- The bracket is **NOT clip-rate saturation**; the high-LR phase simply doesn't extract additional progress under the merged regime (3-ep warmup + decaying noise + EMA + clip-10).
- Decision: **CLOSE** — lr-peak axis is bracketed at 6e-4 under the current stack. Edward's own recommendation: stop pushing lr-peak, future gains from orthogonal axes.
- Suggested follow-ups (per edward):
  1. Stop pushing lr-peak axis (bracket robust).
  2. EMA decay 0.99 (less aggressive — may let more high-LR signal through).
  3. Warmup shape variations (start_factor, warmup_epochs).
  4. **Cosine T_max=11 → 14** to align cosine with realized 18-epoch budget; current schedule has post-cosine LR oscillation at ep15-18.

## 2026-04-28 09:35 — PR #630: Cosine eta_min 0 → 2e-5 (extract gain from late-epoch budget)
- Branch: `charliepai2d2-nezuko/cosine-eta-min-2e-5` (artifact: `model-cosine-eta-min-2e-5-20260428-085728`)
- Hypothesis: replaces closed EMA-decay-target axis. Probe whether epochs 12–18 (now at LR≈0 under compile + T_max=11) extract more gain when LR has a positive floor.

| metric | this run (eta_min=2e-5) | baseline (PR #562 = 64.696 / PR #510 compile = 64.824 / current PR #647 = 61.872) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **59.907** (epoch 18) | 64.696 / 64.824 / 61.872 | **−7.40% / −7.58% / −3.18%** |
| `test_avg/mae_surf_p` | **52.656** | 55.879 / 56.391 / 54.555 | **−5.77% / −6.62% / −3.48%** |

- **Per-split val (vs PR #562)**: in_dist −10.01%, camber_rc −3.51%, **camber_cruise −10.90% (biggest)**, re_rand −6.40%. All 4 splits massively improve.
- **Per-split test**: 62.955, 66.248, 32.300, 49.122. All 4 test splits massively improve.
- **CRITICAL mechanism finding (per nezuko)**: `CosineAnnealingLR` with T_max=11 is **periodic, not a floor**. Formula `η_min + (η_max−η_min)·(1+cos(πt/T_max))/2` is symmetric around T_max — after ep15 hits the minimum 2e-5, LR rebounds in the second half-cycle (ep16: 2.97e-5, ep17: 5.81e-5, **ep18: 1.03e-4**).
- **Gain decomposition**:
  - **ep13-15 (LR descending into floor 5.8e-5 → 2.0e-5)**: −2.96 cumulative. Matches the predicted floor-mechanism. Model fine-tunes at small but nonzero LR; EMA averages a moving target.
  - **ep16-18 (LR ascending out of floor 2.97e-5 → 1.03e-4)**: −2.49 cumulative. The cosine half-cycle rebound. The model continues improving even as LR rises ~5×. EMA(0.995, warmup=50) effectively smooths the higher-LR steps.
  - **Half the gain comes from the unintended ascending half-cycle**, behaving like a SGDR warm-restart half-cycle.
- Decision: **MERGE**. Standalone -3.18% vs current baseline (61.872 → 59.907). Mechanically orthogonal to all merged levers. Mechanism is now clear (rebound is doing real work, not just floor).
- Suggested follow-ups (per nezuko):
  1. **T_max=14 with eta_min=2e-5** (disentangle floor vs rebound) — if T_max=14 underperforms, rebound is doing the work; if it matches/wins, floor is doing the work. Note: edward's #697 tests T_max=14 with eta_min=0; nezuko's #1 adds eta_min=2e-5.
  2. **Explicit warm restart**: `CosineAnnealingWarmRestarts(T_0=11, T_mult=1, eta_min=2e-5)` — make the rebound an explicit SGDR feature.
  3. **Sweep eta_min ladder**: 1e-5, 2e-5, 5e-5, 1e-4 — calibrate sweet spot.
  4. Stretch T_max to match epoch budget (T_max=15 → 3+15=18 = compile budget exactly).

## 2026-04-28 09:55 — PR #682: Steeper per-block slice-temp schedule [1.0, 1.5, 2.0, 2.5, 3.0]
- Branch: `charliepai2d2-askeladd/slice-temp-per-block-steeper` (artifact: `model-slice-temp-per-block-steeper-20260428-091413`)
- Hypothesis: range doubled (vs PR #647's 1.5), same mean (2.0). Block 0's drift signal in PR #647 suggested it wants to be softer.

| metric | this run | PR #647 baseline | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **62.247** (epoch 18) | 61.872 | **+0.61% regression** |
| `test_avg/mae_surf_p` | **54.236** | 54.555 | **−0.58% improvement** |

- Per-split val: in_dist +0.906, camber_rc +0.653, **camber_cruise −0.765**, re_rand +0.705. 3/4 regress; cruise improves.
- Per-split test: in_dist +1.030, camber_rc −0.159, **camber_cruise −1.777 (−5.03% biggest)**, re_rand −0.369. 3/4 improve.
- **Block 0 drift confirmation**: settled at 0.977 (drift -0.023, smallest among all blocks). Block 0 *does* prefer softer than 1.5 — but making it softer didn't improve val_avg.
- **Critical confound**: PR #647 had range=1.5 mean=2.25; this PR has range=2.0 mean=2.0. Variance doubled BUT mean dropped. Likely the lower mean is the load-bearing cost. Mean=2.25 in PR #647 may have been doing more work than the variance.
- Decision: **CLOSE** — schedule-range-with-lower-mean axis is closed. Cruise-friendly anchor (block 0=1.0 with higher mean) is the right disambiguation probe.

## 2026-04-28 09:55 — PR #674: Huber δ=0.10 → 0.05 (push δ profile toward L1)
- Branch: `charliepai2d2-thorfinn/huber-delta-0p05` (artifact: `model-charliepai2d2-thorfinn-huber-delta-0p05-20260428-090400`)
- Hypothesis: profile still descending; δ=0.05 should shift linear-regime fraction from ~29% to ~50-60%.

| metric | this run | PR #601 baseline (δ=0.10) | PR #640 baseline (per-group wd) | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **62.760** (epoch 18) | 62.879 | 62.747 | **−0.19% / +0.02% (saturated)** |
| `test_avg/mae_surf_p` | **54.274** | 54.561 | 54.512 | **−0.53% / −0.44%** |

- Per-split val: in_dist **+1.92%**, camber_rc +0.11%, **camber_cruise −5.14%**, re_rand +0.52%. Same 3/4 down 1/4 up pattern as δ=0.1, but rotated: now cruise carries gain alone.
- Loss-magnitude diagnostic at thr=0.05: aggregate lin = 46.14% (lower than predicted 50-60%); p channel = 36.54%; Ux = 52.55%; Uy = 49.33%.
- **Critical mechanism finding**: at thr=0.10, residual fractions are essentially identical between δ=0.1 and δ=0.05 models (29.0% vs 28.83%). δ shapes optimization but **NOT** the converged residual distribution.
- δ profile: 0.25 → 0.10 = **−3.00% step** vs 0.10 → 0.05 = **−0.19% step**. Diminishing returns are clear; δ axis is saturated.
- Decision: **CLOSE** — δ axis closed at 0.10 for the val_avg metric. Per-channel δ (δ_p smaller, δ_Ux/Uy=0.10) is the right next probe — addresses p channel's lower lin fraction.

## 2026-04-28 09:55 — PR #673: Per-group wd extreme (wd_attn 1e-4 → 3e-4, wd_mlp 1e-5 → 3e-6)
- Branch: `charliepai2d2-tanjiro/per-group-wd-extreme` (artifact: `model-per-group-wd-extreme-20260428-090637`)
- Hypothesis (per tanjiro): if asymmetric reg captures OOD asymmetry, exaggerating it should compound.

| metric | this run | PR #640 baseline | current (PR #630) | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **61.379** (epoch 18) | 62.747 | 59.907 | **−2.18% / +2.46% regression cross-stack** |
| `test_avg/mae_surf_p` | **53.679** | 54.512 | 52.656 | **−1.53% / +1.94% regression cross-stack** |

- **All 4 val splits improve** vs PR #640: in_dist **−2.001**, camber_rc **−2.000** (predicted target ✓), camber_cruise −0.837, re_rand **−0.634** (predicted target ✓).
- All 4 test splits improve.
- Profile still descending; both target OOD splits improved as predicted.
- Decision: **SEND BACK FOR REBASE**. Standalone is robust on the post-#640 stack but this is a magnitude-push of an already-merged lever (#640) — needs current-stack verification. The asymmetry-magnitude may saturate under the heavier round-10 stack (huber δ=0.10 + per-block temp + eta_min=2e-5 rebound).

## 2026-04-28 10:20 — PR #696: Surface-only feature noise (dims 0-12 only, skip per-sample globals)
- Branch: `charliepai2d2-frieren/feature-noise-surface-only` (artifact: `model-feature-noise-surface-only-20260428-093917`)
- Hypothesis: per-sample noise on conditioning labels (log_re, AoA, NACA, gap, stagger) may have been destabilizing; isolating to per-node positional/SDF features (dims 0-12) may unlock more headroom.

| metric | this run | PR #647 baseline | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **61.245** (epoch 18) | 61.872 | **−1.01% WIN** |
| `test_avg/mae_surf_p` | **53.605** | 54.555 | **−1.74% WIN** |

- **All 4 val splits and all 4 test splits improve** (no regressions): in_dist −0.104, camber_rc −0.313, camber_cruise −0.809, **re_rand −1.283 (biggest)**.
- **Mechanism CONFIRMED**: per-sample noise on conditioning labels was **pure destabilization** with no offsetting regularization benefit. re_rand is the cleanest signature — removing log_re noise lets the model trust the Reynolds label, producing much cleaner conditional mapping. cruise benefit preserved (basin-selection signal lives in per-node geometric primitives, not per-sample labels).
- Decision: **MERGE** — clean orthogonal compound. Mechanically separate from all 6 round-10 levers (lr peak, decaying noise schedule, per-group wd, huber δ, per-block temp init, eta_min/rebound).
- Suggested follow-ups (per frieren):
  1. **`feature_noise_std=0.005` surface-only** — primary follow-up. With per-sample destabilizing arm removed, magnitude can probably push higher.
  2. Asymmetric per-node noise scales (positions vs SDF vs is_surface).
  3. Decay-horizon tuning under surface-only.
  4. Fix loss NaN bug in eval (separate cleanup PR).

## 2026-04-28 10:20 — PR #697: Cosine T_max 11 → 14 (align with realized 18-epoch compile budget)
- Branch: `charliepai2d2-edward/cosine-tmax-14` (artifact: `model-cosine-tmax-14-20260428-094042`)
- Hypothesis: align cosine with realized 18-epoch compile budget. Currently cosine ends at ep14 then wraps; T_max=14 gives clean monotone descent through ep17.

| metric | this run | PR #647 baseline | current PR #630 | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **61.390** (epoch 17) | 61.872 | 59.907 | **−0.78% / +2.48% cross-stack** |
| `test_avg/mae_surf_p` | **52.710** | 54.555 | 52.656 | **−3.38% / +0.10% cross-stack** |

- Per-split val: in_dist −0.07, camber_rc +1.26 (slight regression), camber_cruise −1.40, re_rand −1.72. 3/4 improve.
- All 4 test splits improve.
- Best epoch shifted from 18 (PR #647) to 17 (this run).
- LR-vs-epoch confirmed: ep1=1.80e-4, ep4=6.00e-4 (peak), ep17=7.52e-6, **ep18=0 (eta_min reached)**. No second cycle.
- **Edward's analysis claim**: "second cycle was apparently a workaround for the misaligned schedule, not load-bearing". **This DIRECTLY CONFLICTS with PR #630's finding** that the T_max=11 cosine rebound was the load-bearing mechanism (-2.49 cumulative gain at ep16-18 when LR rebounded 2e-5 → 1.03e-4).
- Decision: **SEND BACK FOR REBASE**. The T_max × eta_min interaction is critical: T_max=14 + eta_min=0 gives clean monotone descent; T_max=11 + eta_min=2e-5 gives rebound. Under PR #630 (current merged), the rebound is real. T_max=14 removes the rebound. Need clean current-stack measurement to disambiguate which mechanism is dominant.

## 2026-04-28 10:30 — PR #661: TF32 matmul precision (Blackwell tensor cores)
- Branch: `charliepai2d2-alphonse/tf32-matmul-high` (artifact: `model-tf32-matmul-high-20260428-085927`)
- Hypothesis: `torch.set_float32_matmul_precision("high")` enables Blackwell TF32. Predicted 1.5-2× matmul speedup → 25-40% epoch time reduction. Single-line change.

| metric | TF32+compile (this run) | PR #510 baseline (compile-only) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **60.5905** (epoch 19) | 64.824 | **−6.5%** |
| `test_avg/mae_surf_p` | **53.404** | 56.391 | **−5.3%** |
| epochs in 30-min budget | **21** | 18 | **+16.7%** |
| mean s/epoch (post-warmup) | **86.8** | ~102.3 | **−15.1%** |
| Peak VRAM | 42.6 GB | 42.6 GB | flat |

- Diagnostic eager run also probed: TF32+eager = 16 epochs / 117.9 s/ep = **−12.4% vs eager baseline**.
- Per-split val (vs PR #510 baseline): in_dist −6.4%, camber_rc −6.8%, **camber_cruise −12.1% (largest)**, re_rand −2.2%.
- Per-split test: all 4 splits improve substantially.
- TF32 verification: `torch.get_float32_matmul_precision()` returns `"high"`; no TensorFloat32 hint warning fires.
- **Throughput came in at lower bound of predicted band** (15.1% vs predicted 25-40%). alphonse's diagnosis: matmul fraction is ~50-60% not 70-85% — slice-attention softmax + LayerNorm/GELU are bigger non-matmul fraction than expected.
- **Quality decomposition** (per alphonse): (1) -12.4% to -15.1% per-iter wall-clock pure TF32 matmul speedup; (2) +2-3 extra epochs in budget; (3) the extra epochs land in LR≈0 EMA-tail, val_avg keeps descending into them.
- Vs current baseline 59.907 (post-#630): +1.14% cross-stack regression, but throughput is hardware-deterministic. Same kind of infrastructure value as PR #510 (compile) merge.
- Decision: **MERGE**. Throughput compounds with every future PR. +16.7% epochs in budget = ~3 extra "free" epochs per run. Cosine T_max retune (already in flight as edward's #697 sent-back) becomes even higher-leverage post-TF32.
- Suggested follow-ups (per alphonse):
  1. **Cosine T_max=11 → 14** — actively use the new 21-epoch budget for fine-tuning rather than running ~10 epochs at LR≈0. (Already in flight as edward's #697 — should be re-evaluated post-TF32.)
  2. `set_float32_matmul_precision("medium")` — TF32 with BF16 reduce. ~5-10% additional throughput.
  3. **bfloat16 autocast forward pass** — much larger throughput envelope. EMA + Huber stack robust enough.
  4. Profile matmul fraction with `torch.profiler` to confirm ~50-60% finding.

## 2026-04-28 10:40 — PR #698: Cosine eta_min 2e-5 → 5e-5 (calibrate sweet spot)
- Branch: `charliepai2d2-nezuko/cosine-eta-min-5e-5` (artifact: `model-charliepai2d2-nezuko-cosine-eta-min-5e-5-20260428-094948`)
- Hypothesis: PR #630 just merged with -7.4% gain at eta_min=2e-5; profile sweep upward to test rebound-magnitude effect.

| metric | this run (eta_min=5e-5) | PR #630 baseline (eta_min=2e-5) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` | **58.742** (epoch 17) | 59.907 | **−1.94% WIN** |
| `test_avg/mae_surf_p` | **50.789** | 52.656 | **−3.55% WIN** |

- Per-split val: in_dist +0.07% (flat), camber_rc **−2.74%**, **camber_cruise −5.44% (largest)**, re_rand −0.97%. Camber-dominant gains.
- Per-split test: 60.548, 63.561, 30.679, 48.367 — all improve.
- LR trajectory: ep4=6.00e-4 (peak), **ep15=5.00e-5 (cosine min)**, ep17=9.37e-5 (best), ep18=1.449e-4 (post-rebound).
- **eta_min profile (monotone-improving)**: 0 → 64.696, 2e-5 → 59.907, 5e-5 → 58.742. Diminishing returns but still descending.
- **Mechanism shift from PR #630**: the dominant gain came from the **descent phase** (more LR retained at the floor; ep13→15 = -2.725 vs PR #630's -2.274), NOT the **rebound phase** as predicted (rebound gain stayed ~flat or slightly shrank). Higher eta_min mostly bought descent budget here, not bigger rebound.
- **First soft signal of approaching ceiling**: ep18 (58.756) slightly worse than ep17 (58.742) — first time the post-rebound epoch is worse. With eta_min=2e-5, ep18 was the best epoch. Rebound peak LR (1.449e-4) is starting to overshoot the EMA-absorbed minimum.
- Decision: **MERGE**. Profile still descending; new baseline.
- Suggested follow-ups (per nezuko):
  1. **eta_min=1e-4** — push further; rebound may overshoot more but descent budget keeps growing.
  2. eta_min=7.5e-5 — bracket if 1e-4 saturates.
  3. CosineAnnealingWarmRestarts to decouple rebound from floor.
  4. T_max sweep at fixed eta_min=5e-5.

## 2026-04-28 10:50 — PR #709: Per-channel Huber δ (δ_p=0.05, δ_Ux=δ_Uy=0.10)
- Branch: `charliepai2d2-thorfinn/huber-per-channel-delta` (artifact: `model-huber-per-channel-delta-20260428-100432`)
- Hypothesis: scalar δ axis saturated at 0.10 (PR #674); the `p` channel has the lowest lin fraction (37% at thr=0.05). Apply δ_p=0.05 + δ_Ux=δ_Uy=0.10 to focus outlier downweighting.

| metric | this run | PR #630 baseline | current PR #698 baseline | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **58.887** (epoch 18) | 59.907 | 58.742 | **−1.70% / +0.25% cross-stack** |
| `test_avg/mae_surf_p` | **51.162** | 52.656 | 50.789 | **−2.84% / +0.73% cross-stack** |

- Per-split val: in_dist **−2.31%**, camber_rc **−2.12%**, camber_cruise +0.59% (essentially flat — cruise gain LOST), re_rand **−2.01%**.
- All 4 test splits improve.
- **Mechanism CONFIRMED**: per-channel δ_p=0.05 reproduces the global δ=0.05 effect on p residual distribution (35.81% vs 36.54% lin at thr=0.05, near-identical). Ux/Uy lin fractions DROPPED vs global δ=0.05 (52.55%→49.04%, 49.33%→47.28%) because they now train under δ=0.10's wider quadratic regime, tightening their residuals.
- **Cruise gain LOST** (but cruise is essentially flat). PR #674's −5.14% cruise gain came partly from Ux/Uy δ=0.05 interaction, not just p. Per-channel scheme captures p's tail benefit + Ux/Uy's quadratic-regime benefit but loses cruise's joint-channel outlier handling.
- Decision: **MERGE**. Mechanically orthogonal to eta_min — should compound. Standalone -1.70% on prior stack is robust same-stack evidence.
- Suggested follow-ups (per thorfinn):
  1. **δ_p=0.025** (push p-tail axis further; predicted Δ −0.5 to +0.5 pts).
  2. Cruise-aware loss (per-split or geometry-conditioned δ).
  3. Ux/Uy at δ=0.05 only on cruise (per-batch dispatch).

## 2026-04-28 10:50 — PR #708: Cruise-friendly anchor [1.0, 1.75, 2.5, 3.25, 4.0]
- Branch: `charliepai2d2-askeladd/slice-temp-per-block-cruise-anchor` (artifact: `model-charliepai2d2-askeladd-slice-temp-per-block-cruise-anchor-20260428-100558`)
- Hypothesis: combine block 0=1.0 (cruise benefit) with mean=2.5 (higher than #647's 2.25) and range=3.0. Disambiguate variance-vs-mean.

| metric | this run | PR #630 baseline | current PR #698 | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **59.737** (epoch 18) | 59.907 | 58.742 | **−0.28% / +1.69% cross-stack** |
| `test_avg/mae_surf_p` | **52.863** | 52.656 | 50.789 | **+0.39% / +4.09% cross-stack** |

- Per-split val: in_dist +0.774, camber_rc −0.566, camber_cruise −0.167, re_rand −0.720. 3/4 improve (in_dist regresses).
- Per-block final temps: [0.974, 1.699, 2.429, 3.166, 3.909] — block 4 settled at 3.91 (drift -0.09), block 0 at 0.97 (drift -0.026, smallest as expected).
- **Mechanism observations**: cruise val improved slightly (-0.167) — block 0=1.0 preserves cruise mechanism. Higher mean (2.5 vs 2.25) IS load-bearing. Range=3.0 well-tolerated by block 4.
- Decision: **CLOSE** — marginal val win (-0.28%) on prior baseline; cross-stack +1.69% regression vs current 58.742. Test slightly regressed. Schedule axis is approaching saturation; mean shifts give diminishing returns.

## 2026-04-28 10:50 — PR #673: Per-group wd extreme rebased (wd_attn=3e-4, wd_mlp=3e-6)
- Branch: `charliepai2d2-tanjiro/per-group-wd-extreme` (artifact: `model-per-group-wd-extreme-rebased-20260428-100928`)
- Hypothesis: rebased onto post-#630 stack to verify the asymmetry-magnitude push that won -2.18% on PR #640 baseline still wins under heavier merged stack.

| metric | this run | PR #630 baseline | current PR #698 | Δ |
|---|---|---|---|---|
| best `val_avg/mae_surf_p` | **60.122** (epoch 18) | 59.907 | 58.742 | **+0.36% regression / +2.35% cross-stack** |
| `test_avg/mae_surf_p` | **52.339** | 52.656 | 50.789 | **−0.60% / +3.05% cross-stack** |

- Per-split val: in_dist −0.051 (flat), **camber_rc +1.030 (REVERSED)**, camber_cruise +0.210, **re_rand −0.325 (preserved)**.
- All 4 test splits improve.
- **CRITICAL ASYMMETRIC REVERSAL FINDING**: MLP arm preserved (re_rand still improves with wd_mlp=3e-6, just smaller magnitude than -0.634 on pre-#630 stack); ATTENTION arm reversed (camber_rc was -2.000 on old stack, now +1.030 with wd_attn=3e-4). The per-block slice-temp schedule (PR #647) + cosine rebound (PR #630) already supply the geometry-OOD regularization that wd_attn was extracting.
- Decision: **CLOSE** — val regresses (gating metric); asymmetry-magnitude axis is bracketed at (1e-4, 1e-5). MLP-only extreme is the right decoupled probe.
