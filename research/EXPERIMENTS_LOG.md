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

## Test-metric NaN follow-up (cross-PR)

All three reviewed PRs report `test_avg/mae_surf_p = NaN`. Root cause from the student diagnoses:

- One sample (`test_geom_camber_cruise` sample 20) has 761 non-finite values in `y[p]` volume nodes. Surface `p` is finite for that sample.
- `data/scoring.py:accumulate_batch` is *intended* to skip samples with non-finite ground truth (`y_finite` mask), but the implementation computes `err = (pred - y).abs()` over the whole batch *before* masking. With IEEE 754, `Inf * 0 = NaN`, so the masked-out element still poisons the per-channel sum.
- `data/scoring.py` is read-only per `program.md`. The cleanest workaround is in `train.py` `evaluate_split`: filter out samples with any non-finite `y` before calling `accumulate_batch`. We will route this fix through a round-2 PR (charliepai2d2-edward) so paper-facing `test_avg/mae_surf_p` becomes recoverable for all subsequent runs.
