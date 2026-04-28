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

## Test-metric NaN follow-up (cross-PR)

All three reviewed PRs report `test_avg/mae_surf_p = NaN`. Root cause from the student diagnoses:

- One sample (`test_geom_camber_cruise` sample 20) has 761 non-finite values in `y[p]` volume nodes. Surface `p` is finite for that sample.
- `data/scoring.py:accumulate_batch` is *intended* to skip samples with non-finite ground truth (`y_finite` mask), but the implementation computes `err = (pred - y).abs()` over the whole batch *before* masking. With IEEE 754, `Inf * 0 = NaN`, so the masked-out element still poisons the per-channel sum.
- `data/scoring.py` is read-only per `program.md`. The cleanest workaround is in `train.py` `evaluate_split`: filter out samples with any non-finite `y` before calling `accumulate_batch`. We will route this fix through a round-2 PR (charliepai2d2-edward) so paper-facing `test_avg/mae_surf_p` becomes recoverable for all subsequent runs.
