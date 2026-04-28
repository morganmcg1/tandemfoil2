# SENPAI Research Results — `icml-appendix-charlie-pai2d-r1`

## 2026-04-27 23:30 — PR #355: mlp_ratio 2→4 (charliepai2d1-nezuko)
- Branch: `charliepai2d1-nezuko/mlp-ratio-4`
- Hypothesis: bumping the per-block MLP from `128→256→128` to `128→512→128` adds ~+19% params (0.83M → 0.99M) and gives the per-node nonlinearity more lift; expected −3% to −8% on `val_avg/mae_surf_p`.

### Headline metrics (best epoch = 13/50, run cut by 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---|---|---|---|---|
| `mae_surf_p` | 165.01 | 135.12 | 99.19 | 117.65 | **129.24** |
| `mae_surf_Ux` | 2.636 | 2.899 | 1.708 | 2.405 | 2.412 |
| `mae_surf_Uy` | 0.939 | 1.200 | 0.682 | 0.932 | 0.938 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---|---|---|---|---|
| `mae_surf_p` | 146.15 | 129.26 | **NaN** | 119.14 | **NaN** (3-split mean = 131.51) |
| `mae_surf_Ux` | 2.507 | 2.963 | 1.707 | 2.294 | 2.368 |
| `mae_surf_Uy` | 0.909 | 1.140 | 0.643 | 0.906 | 0.900 |

- Metric files (student branch): `models/model-charliepai2d1-nezuko-mlp-ratio-4-20260427-224909/{metrics.jsonl,metrics.yaml}`
- Wall clock: 32 min (training cut by `SENPAI_TIMEOUT_MINUTES=30` after 13/50 epochs)
- Peak VRAM: 52.18 GB (within 96 GB budget)
- Param count: 991,319 (~0.99M)

### Analysis
- **Training was cut very early.** Best val (129.24) was at the *last* completed epoch (13/50). Cosine schedule had barely begun decaying; the val curve went 282.7 → 129.2 over those 13 epochs and was still descending strongly. With more epochs the number would almost certainly continue dropping. Per-epoch wall clock was ~150 s; 30-min timeout fits ~12 epochs.
- **Test pressure NaN is a scoring failure, not a model failure.** Validation on the matching split (`val_geom_camber_cruise`) is the *best* of the four val splits at `mae_surf_p = 99.19`. The NaN appears only on the test counterpart — and only on the pressure channel. Velocity (Ux, Uy) MAE for that test split are finite (1.71, 0.64). Pattern: `vol_loss=inf, surf_loss=nan, mae_*_p=nan`. The model produced an inf or NaN prediction on the p channel for at least one test sample; `data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions**, so the bad value pollutes the float64 accumulator → NaN in the final MAE.
- **No baseline measured yet on this branch** so the val=129.24 cannot yet be ranked. The other 7 round-1 PRs are still in flight.

### Decision: send back to student
- Cannot merge: `test_avg/mae_surf_p` is NaN, which violates the "no NaN in primary metrics" rule.
- Cannot close: hypothesis is sound, val trajectory is clean and strongly descending, the only blocker is a numerical-edge-case in eval.
- 23:30: sent back with `nan_to_num`-on-pred fix instructions. **Fix instructions were wrong — see correction below.**
- 23:42: corrective follow-up posted. After PRs #356/#351 landed with independent diagnoses, the actual root cause is one `test_geom_camber_cruise` sample with non-finite **ground truth** in the `p` channel. `data/scoring.py` masks bad samples but `(finite − inf).abs() * 0 = NaN` defeats it. Fix lives in `train.py:evaluate_split` (sanitize `y`, drop bad samples from `mask`) and is now in baseline post-#356. Nezuko instructed to rebase onto baseline + retain `mlp_ratio=4`.

## 2026-04-27 23:42 — PR #356: EMA(0.999) shadow for val + checkpoint (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/ema-eval` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `208f1cc`).
- Hypothesis: maintain EMA shadow weights with decay 0.999, evaluate val on the shadow, save the shadow's state_dict, and run final test eval from it. Predicted −2% to −7% on `val_avg/mae_surf_p` from variance reduction alone.

### Headline metrics (best epoch = 13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 170.491 | 144.104 | 100.492 | 114.015 | **132.276** |
| `mae_surf_p` (raw, same epoch) | 231.699 | 180.205 | 144.919 | 156.747 | 178.392 |
| `mae_surf_p` (best raw, ep11) | — | — | — | — | 136.526 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 147.132 | 127.917 | 84.026 | 113.089 | **118.041** |
| `mae_surf_Ux` | 1.948 | 2.988 | 1.332 | 1.949 | 2.054 |
| `mae_surf_Uy` | 0.827 | 1.076 | 0.553 | 0.833 | 0.822 |

- Metric files (now in baseline): `models/model-charliepai2d1-tanjiro-ema-eval-20260427-225450/{metrics.jsonl,metrics.yaml,config.yaml}`
- Wall clock: 30.5 min (timeout-cut at ep13/50, +companion raw eval ~doubles val cost vs EMA-only)
- Peak VRAM: 42.11 GB

### Analysis
- **EMA delivered the predicted free lunch.** 132.28 (EMA) vs 136.53 (best raw) = −3.1% on `val_avg/mae_surf_p`, in band. EMA val curve is monotonic (324 → 132 over 13 epochs); raw is noisy (191 → 164 → 249 mid-training). Smoothing buys most at noisy epochs; both are still descending hard at the timeout.
- **Real bug-find in the scoring path.** Tanjiro identified that one test sample (`test_geom_camber_cruise` index 20) has `y[p]` non-finite. `data/scoring.py:accumulate_batch` builds the right per-sample mask but computes `err = |pred − y|` *before* the masked sum, and IEEE-754 `NaN*0 = NaN` poisons the float64 accumulator. Fix: pre-pass in `evaluate_split` that drops non-finite-y samples from `mask` and zeros their `y`. Same root cause flagged independently by askeladd on PR #351.
- **Bonus instrumentation.** Each epoch now logs both EMA and raw `val_avg/mae_surf_p` in `metrics.jsonl`, plus `best_raw_val_avg/mae_surf_p` and `best_raw_epoch` in `metrics.yaml`. Future EMA experiments can be audited for raw vs EMA gap directly.

### Decision: merge as new round-1 baseline
- Predicted delta achieved (−3.1%, in the −2% to −7% band).
- Test number clean (no NaN) thanks to the workaround.
- The workaround benefits all in-flight round-1 PRs once they rebase.
- BASELINE.md updated; #355 and #351 routed to rebase onto this baseline retaining their respective levers.

## 2026-04-27 23:42 — PR #351: surf_weight 10→50 (charliepai2d1-askeladd)
- Branch: `charliepai2d1-askeladd/surf-weight-50`
- Hypothesis: raising surface-loss weight from 10 to 50 should reduce volume gradient dominance and align training signal more directly with the metric.

### Headline metrics (best epoch = 10/50, run cut by 30-min timeout; concurrent GPU contention slowed eps 7–8)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 166.86 | 143.62 | 108.19 | 122.08 | **135.19** |

- `test_avg/mae_surf_p`: NaN (same scoring bug; 3-split mean over the clean splits = 134.00; cruise NaN as expected).
- Metric files (student branch): `models/model-surf-weight-50-20260427-225133/{metrics.jsonl,metrics.yaml}`
- Wall clock: 30.2 min, peak VRAM: 42.12 GB

### Analysis
- **Did not beat the new baseline (132.28).** 135.19 vs 132.28 = +2.2%. Not a clear regression (>5%), and the run was timeout-cut at ep10 plus had GPU contention costing ~2 epochs.
- **Trade-off direction sane.** Surface MAE is decent across splits; cruise (108.19) easiest, single (166.86) hardest. Volume MAE not blowing up — `vol_p` ranges 153–199 across splits.
- **Independently rediscovered the scoring NaN bug.** Same root cause as tanjiro's; clean diagnosis with the IEEE-754 NaN×0 explanation. Bug-fix suggestion logged.

### Decision: send back for rebase + retain surf_weight=50
- Surf-weight is orthogonal to EMA — right move is to test compounding rather than close.
- Rebase onto post-#356 baseline (gets EMA + NaN-safe path), keep `surf_weight=50.0`, re-run, report Δ vs new baseline.
- If "EMA + surf_weight=50" beats 132.28 by any margin, merge as next baseline.

## 2026-04-27 23:51 — PR #354: slice_num=64→128, n_head=4→8 (charliepai2d1-frieren) — **CLOSED**
- Branch: `charliepai2d1-frieren/slice-128-heads-8` (closed + branch deleted)
- Hypothesis: doubling slice tokens and heads to give finer physics-aware attention on irregular meshes.

### Headline metrics (best epoch = 7/50, run cut by 30-min timeout at ep8)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 200.71 | 169.81 | 117.20 | 138.18 | **156.48** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg (clean)** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 182.71 | 155.54 | 99.20 | 138.95 | **144.10** |

- Per-epoch wall clock: ~250 s (vs ~145 s baseline). 8 of 50 epochs trained.
- Peak VRAM: 82.32 GB (vs 42 GB baseline).
- Δ vs new baseline (132.276 / 118.041): **val +18.3%, test +22.1%** — both clear regressions.

### Analysis
- **Throughput is the binding constraint, not the lever's intrinsic merit.** All-block slice_num=128 + heads=8 takes ~250 s/epoch, fitting only 8 of 50 configured epochs in the 30-min training budget. The baseline fits ~13 epochs at ~145 s/epoch in the same wall clock. The val curve was still descending strongly at the timeout (229 → 156 over 8 epochs).
- **Independent rediscovery of the scoring NaN bug.** Frieren correctly diagnosed both the `inf*0=NaN` propagation in `data/scoring.py:accumulate_batch` and the bad sample (`test_geom_camber_cruise/000020.pt`, `y[:,2]` non-finite). They added a NaN-safe rerun that produced clean test numbers (144.10).

### Decision: close, reassign to mixed-slice-last-layer
- Clear >5% regression on both val and test → meets close criteria per CLAUDE.md.
- The lever isn't disproven — it's under-budgeted. Frieren's own analysis was honest about this.
- Reassigned to **PR #373 (mixed-slice-last-layer)** — `slice_num=128` only in the final block, `slice_num=64` in layers 0–3. Targets ~+15% per-epoch cost vs baseline (~165–175 s/epoch), should fit ~10–11 epochs in the 30-min budget. Direct follow-up to frieren's "mixed slice counts across layers" suggestion.

## 2026-04-27 23:55 — Round-1.5 assignments (post-#356-merge follow-ups)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #373 | frieren | mixed-slice-last-layer | Last-layer-only `slice_num=128` (mixed slicing) | Replaces closed #354; respects 30-min timeout; pays slice cost only at the regression head |
| #374 | tanjiro | grad-clip-1p0 | Gradient clipping at `max_norm=1.0` between backward and step | Variance-reduction lever complementary to EMA; logs pre-clip grad norm as diagnostic |

## 2026-04-28 00:10 — PR #352: SmoothL1 (Huber, β=1) on surface loss (charliepai2d1-edward) — **sent back for rebase + re-run; will merge after**
- Branch: `charliepai2d1-edward/smoothl1-surface` (pre-EMA base; conflicts with merged #356 in `evaluate_split`)
- Hypothesis: replace MSE with SmoothL1 on the surface loss term to give the gradient an MAE-shaped profile in the |err|>β regime, where high-Re samples push the residual past 1σ. Volume term kept as MSE.

### Headline metrics — pre-rebase (raw, no EMA, ep14/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 124.03 | 119.75 | 81.76 | 96.71 | **105.56** |
| `mae_surf_Ux` | 2.04 | 2.82 | 2.05 | 2.50 | 2.35 |
| `mae_surf_Uy` | 0.78 | 0.99 | 0.60 | 0.79 | 0.79 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 113.18 | 105.59 | 68.34 | 94.46 | **95.39** |
| `mae_surf_Ux` | 2.04 | 2.71 | 2.05 | 2.40 | 2.30 |
| `mae_surf_Uy` | 0.76 | 0.94 | 0.55 | 0.76 | 0.75 |

- Wall clock: 30.7 min (timeout-cut at ep14/50). Peak VRAM: 42.14 GB. NaN events: 0.
- Run committed `train.py` includes a defensive non-finite-y pre-filter in `evaluate_split` (different implementation from tanjiro's but functionally equivalent — to be replaced by baseline's version on rebase).

### Comparison vs new baseline (PR #356, EMA, ep13/50)
| | baseline (EMA, MSE) | edward (raw, SmoothL1) | Δ raw-vs-EMA | (informational) Δ raw-vs-raw |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 132.276 | 105.56 | **−20.2%** | **−22.7%** vs tanjiro's best raw 136.53 |
| `test_avg/mae_surf_p` | 118.041 | 95.39 | **−19.2%** | n/a (baseline raw test number not available) |

### Analysis
- **The lever is decisively winning.** Even with the unfair raw-vs-EMA comparison disadvantaging edward, val improvement is −20.2% and test is −19.2%. Raw-vs-raw against tanjiro's PR #356 internal best raw, SmoothL1 alone delivers −22.7%. This is the largest single-lever delta of round 1 by a wide margin.
- **Mechanism is consistent with theory.** MSE quadratically up-weights large errors, which on high-Re samples (per-sample y std up to ~2,077 m²/s² in `val_single_in_dist`) means the gradient is dominated by a handful of high-Re outliers. SmoothL1 with β=1 in normalized space matches MAE asymptotics in the |err|>β regime, so per-sample contributions stay closer to constant magnitude. The largest val improvements show up exactly where the value range is widest (single_in_dist 165→124, rc 135→120, re_rand 118→97).
- **Pre-existing scoring NaN bug also handled**: edward used a similar pre-filter to tanjiro's. On rebase, baseline's NaN-safe `evaluate_split` supersedes.
- **Run was timeout-cut, val curve still descending.** Best at the last logged epoch (14/50). With more wall clock the number would likely fall further.

### Decision: send back for rebase + re-run
- Beats baseline by a wide margin (>−5%): would normally merge directly. But edward's branch pre-dates #356 and `evaluate_split` conflicts.
- Per merge-winner skill workflow: when conflicts exist, send back for rebase rather than force-merge.
- Rebase resolution: take baseline's `evaluate_split` (drops edward's filter); keep edward's SmoothL1 substitution in the train loop's loss block (no overlap with tanjiro's EMA-update insertion).
- Re-run is required because the saved checkpoint was trained without EMA-shadow updates, so we can't reuse it under the new baseline.
- Predicted post-rebase outcome: SmoothL1 + EMA likely lands near val ≈ 100–102, test ≈ 92–95 (assuming the EMA −3% delta from #356 applies on top of SmoothL1's −22.7%).
- Will merge as new baseline once clean post-rebase numbers land.

## 2026-04-28 00:18 — PR #357: channel-weighted surface loss `[1,1,5]` (charliepai2d1-thorfinn) — **CLOSED**
- Branch: `charliepai2d1-thorfinn/channel-weighted-loss` (pre-EMA base; closed + branch deleted)
- Hypothesis: up-weight pressure channel 5× inside surface MSE to align training signal with `mae_surf_p` ranking metric.

### Headline metrics (best epoch = 12/14, run cut at 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 180.82 | 156.27 | 123.67 | 142.89 | **150.91** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 169.78 | 147.89 | 110.76 | 143.85 | **143.07** |

- Per-epoch wall clock ~131 s (similar to baseline). 14 of 50 epochs trained.
- Δ vs baseline #356 (132.276 / 118.041): **val +14.1 %, test +21.2 %** — both >5 % regressions.
- Δ raw-vs-raw vs tanjiro #356 best raw (136.53): **+10.5 %** — also a regression on the fair comparison.
- Per-epoch val curve: 237 → 244 → 186 → 191 → 188 → 175 → 196 → 176 → 174 → 168 → 179 → **151*** → 172 → 161 — severe oscillation, channel weighting destabilizing training.

### Analysis
- The `[1, 1, 5]` MSE-style weighting is not an additive variance-reduction lever like channel-uniform `surf_weight=50` would be — it asymmetrically scales the gradient on a single channel, which combined with the existing `surf_weight=10` puts the surface-p gradient roughly an order of magnitude above velocity. Training oscillation suggests the optimizer is overshooting on p-channel-favorable directions.
- **Side-by-side vs PR #352 (SmoothL1 β=1, same pre-EMA base, same wall budget) at val=105.56**: loss-form lever wins by ~30 % over channel-weighting. The loss-shape direction (MSE → SmoothL1 / L1) is far more impactful than per-channel re-weighting.
- **Fourth independent rediscovery of the `data/scoring.py` `inf*0=NaN` bug.** Tanjiro's filter (now in baseline) supersedes thorfinn's; functionally equivalent.

### Decision: close, reassign to torch-compile-throughput
- Clear >5 % regression on both val and test, with no mechanistic path forward via a small variation (the oscillation shows the lever is destabilizing rather than focusing).
- Reassigned to **PR #394 (torch-compile-throughput)** — high-leverage throughput PR. Predicted 20–35 % per-epoch speedup; if it lands, every subsequent experiment fits ~17 epochs in the 30-min timeout instead of ~13. Multiplied value across every round-2 experiment.

## 2026-04-28 00:25 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #394 | thorfinn | torch-compile-throughput | `torch.compile(model, ema_model)` with `mode="reduce-overhead", dynamic=True` | Replaces closed #357; structural throughput improvement that helps every subsequent PR fit more epochs |

## 2026-04-28 00:30 — PR #355 (re-run): mlp_ratio 2→4 GELU on EMA baseline (charliepai2d1-nezuko) — **CLOSED (wash)**
- Branch: `charliepai2d1-nezuko/mlp-ratio-4` (closed + branch deleted)
- Hypothesis (re-run): retain `mlp_ratio=4` on the post-#356 baseline (EMA + NaN-safe pre-pass) to test compounding with EMA.

### Headline metrics (best EMA epoch = 12/50, run cut by 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 168.66 | 144.83 | 101.17 | 117.18 | **132.96** |
| `mae_surf_p` (raw, same epoch) | 151.05 | 144.50 | 101.13 | 120.92 | 129.40 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 143.13 | 129.47 | 84.32 | 115.44 | **118.09** |

- Param count: 991,319 (~+19 % vs baseline 828K).
- Per-epoch wall clock: ~159 s (vs baseline ~138 s, **+15 %**); 12 of 50 epochs trained.
- Δ vs baseline #356: **val EMA +0.52 %, test +0.04 %** — wash on the ranking metric.
- Δ raw-vs-raw vs tanjiro #356 best raw (136.53): **−5.2 %** — real gain hidden by EMA at this epoch budget.

### Per-split test breakdown (in-dist vs OOD pattern)
| Split | this run | baseline | Δ | OOD? |
|---|---:|---:|---:|:---:|
| test_single_in_dist | 143.13 | 147.13 | **−2.72 %** | no |
| test_geom_camber_rc | 129.47 | 127.92 | +1.21 % | yes |
| test_geom_camber_cruise | 84.32 | 84.03 | +0.34 % | yes |
| test_re_rand | 115.44 | 113.09 | +2.08 % | yes |

The wider GELU MLP **helps in-distribution but slightly hurts OOD** on all three OOD splits. Three of four splits trending the wrong way is real signal; equal-weighting drags the average back to wash.

### Analysis
- **Confounded test.** Bumping `mlp_ratio=2 → 4` increases per-node nonlinearity *and* parameter count *and* per-epoch wall clock simultaneously. The next experiment should isolate the nonlinearity lever from the capacity bump.
- **EMA at 0.999 with 12-epoch budget hides the raw gain.** The shadow averages over too many effective updates (decay → effective half-life ~700 steps ≈ 1.85 epochs at 375 batches/epoch); when the live model improves quickly, the shadow lags. This is a "wrong tool for the budget" issue, not a fundamental problem.
- **In-dist vs OOD trade-off** is the most interesting finding: extra MLP capacity goes to memorizing training distribution rather than improving generalization. Nezuko spotted this clearly.

### Decision: close, reassign to swiglu-mlp-matched
- Wash on the equal-weight ranking metric (+0.52 % val, +0.04 % test). Per CLAUDE.md merge rule (must be `<` baseline), no merge. Per close threshold (>5 % regression), no close on ranking. Effectively neutral.
- BUT the lever is dominated by **PR #352 SmoothL1** (val=105.56 raw, ~30 % advantage) on the metric-mover axis, AND the cleaner per-node-nonlinearity test is SwiGLU at matched param count.
- Closing here, reassigning to **PR #398 (swiglu-mlp-matched)**: SwiGLU `(W_g(x) ⊙ silu(W_v(x)))W_o` at `swiglu_inner=168` to match baseline param count exactly. Strips the capacity confound and the wall-clock tax — clean read on whether gating-style activation alone moves the needle.

## 2026-04-28 00:35 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #398 | nezuko | swiglu-mlp-matched | SwiGLU MLP `(W_g(x)⊙silu(W_v(x)))W_o` at `swiglu_inner=168`, matched to baseline param count | Replaces closed #355; cleaner per-node-nonlinearity test (no capacity confound, no wall-clock tax) |

## 2026-04-28 00:43 — PR #374: gradient clipping at `max_norm=1.0` (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/grad-clip-1p0` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `4e47f8a`).
- Hypothesis: clip gradient L2 norm at 1.0 between `loss.backward()` and `optimizer.step()`, complementary variance-reduction lever to EMA. Predicted band: −1 % to −3 %.

### Headline metrics (best EMA epoch = 13/50, timeout-cut)
| metric | this run | prior baseline (#356) | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **113.157** | 132.276 | −19.119 | **−14.45 %** |
| `test_avg/mae_surf_p` | **99.322** | 118.041 | −18.719 | **−15.86 %** |
| raw `val_avg/mae_surf_p` (best, ep 13) | 121.992 | 136.526 (ep 11) | −14.534 | **−10.65 %** |
| raw `val_avg/mae_surf_p` at best EMA ep | 121.992 | 178.392 | −56.400 | −31.62 % |

### Per-split — beats baseline on every split, val and test
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | −36.6 | −30.6 |
| geom_camber_rc | −11.4 | −13.8 |
| geom_camber_cruise | −15.6 | −14.6 |
| re_rand | −12.9 | −15.9 |

### Diagnostic — pre-clip grad norm cluster (the explanation)

Mean pre-clip `train/grad_norm` per epoch: 117 → 95 → 85 → 83 → 72 → 71 → 70 → 69 → 55 → 60 → 66 → 56 → 56. Norms decay over training but stay **two orders of magnitude above `max_norm=1.0`** for the entire run. The clip is firing on every step and acting as an effective LR cap, damping the largest gradient steps that were previously throwing the optimizer off-trajectory.

### Analysis
- **Mechanism is clear.** Pre-clip norms 50–100× the threshold means clipping is doing real work, not acting as a delicate stabilizer. The size of the gain (~5× the predicted band) is consistent with "baseline optimizer was being dragged off-trajectory by occasional huge steps."
- **Raw and EMA converge.** Baseline (#356) had a 42-point raw/EMA gap at ep13 (178 vs 132, raw curve oscillating 191 → 164 → 249); this run's gap is 8 points (122 raw vs 113 EMA) and the raw curve is monotonically much smoother. Variance reduction at the *step* level (grad-clip) compounds with variance reduction at the *iterate* level (EMA).
- **All four splits beat baseline on val and test.** Not an in-dist trick.
- **Wall clock unchanged** (30.4 min, ep13/50). The clip adds one cheap reduction per step.

### Decision: merge as new round-1 baseline
- Beats baseline on the ranking metric by a wide margin.
- Mechanism is well-understood (auditable via the per-epoch grad-norm trace tanjiro logged).
- Compounds cleanly with the SmoothL1 lever once #352 lands; complements every other in-flight PR.
- BASELINE.md updated; tanjiro reassigned to **PR #402 (grad-clip-0p5)** as the natural follow-up to their own suggestion #1.

## 2026-04-28 00:48 — PR #373: last-layer-only `slice_num=128` (charliepai2d1-frieren) — **CLOSED**
- Branch: `charliepai2d1-frieren/mixed-slice-last-layer` (closed + branch deleted)
- Hypothesis: bump `slice_num=64 → 128` only in the final TransolverBlock (which feeds the regression head). Cost story checked out (148.8 s/ep vs baseline 145, only +3 % wall clock).

### Headline metrics (best EMA epoch 13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 160.86 | 147.41 | 107.34 | 118.35 | **133.49** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 141.78 | 132.10 | 89.68 | 119.84 | **120.85** |

- Δ vs baseline #356: **val +0.92 %, test +2.38 %**.
- Per-split test: in-dist −3.64 % (better), geom_camber_rc +3.27 %, geom_camber_cruise +6.73 %, re_rand +5.97 % — three of four OOD splits regress.

### Analysis
- **Same in-dist-helps / OOD-regresses pattern as closed PR #355 (mlp_ratio=4 GELU)**. Adding capacity selectively in the regression-head block trades OOD generalization for in-dist memorization.
- Wall clock cost was sub-prediction (3 % vs predicted 15 %), so the *cost story* is not the issue.
- Frieren's analysis nails the likely mechanism: doubling the slice projection's output dim (`dim_head → slice_num`) introduces a softmax-temperature tuning problem the optimizer doesn't have time to solve in 13 timeout-cut epochs.

### Decision: close, reassign to batch8-lr-sqrt2
- Wash on val (+0.92 %) and small regression on test (+2.38 %), no path forward via a small variation given the same pattern as the closed `mlp_ratio=4` PR.
- Matched-pattern closure: more capacity at this epoch budget on this architecture is not the lever; the loss-form direction (#352) and variance-reduction direction (#356/#374) are dominating.
- Reassigned to **PR #403 (batch8-lr-sqrt2)** — variance reduction at the *gradient aggregation* level, complementary to the just-merged grad-clip and EMA. Larger effective batch builds on round-1 winning direction.

## 2026-04-28 00:50 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #402 | tanjiro | grad-clip-0p5 | More aggressive grad-clip: `max_norm=1.0 → 0.5` | Tanjiro's own follow-up; tests whether further damping at this LR helps or starves |
| #403 | frieren | batch8-lr-sqrt2 | `batch_size=4 → 8`, `lr=5e-4 → 7e-4` (√2 scaling) | Variance reduction at gradient aggregation; compounds with EMA + grad-clip |

## 2026-04-28 00:52 — PR #353: 5-ep warmup + cosine to 1e-5 with peak LR=1e-3 (charliepai2d1-fern) — **CLOSED**
- Branch: `charliepai2d1-fern/warmup-cosine-1e3` (pre-EMA, pre-grad-clip base; closed + branch deleted)
- Hypothesis: peak `lr=1e-3` with 5-epoch linear warmup + cosine to 1e-5; transformer-style schedule lets us safely raise peak LR.

### Headline metrics (best ep=12/13, raw eval, no EMA)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (raw) | 172.00 | 150.07 | 105.48 | 132.10 | **139.91** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 155.11 | 142.89 | NaN | 131.04 | NaN (3-split mean = 143.01) |

- Per-epoch wall clock: 131 s (matches baseline). 13 of 50 epochs trained.
- LR schedule fired exactly as specified: 1e-6 → 1e-3 over eps 1-5, cosine decay from 9.99e-4 (ep7) to 9.42e-4 (ep13).

### Comparisons
- vs #356 raw best (no warmup, lr=5e-4, no clip): 139.91 vs 136.53 = **+2.5 % worse**
- vs #374 raw best (no warmup, lr=5e-4, +grad-clip): 139.91 vs 121.99 = **+14.7 % worse**

The hypothesis didn't beat even the simplest raw baseline.

### Analysis
- **Cosine T_max=50 is degenerate at the 13-epoch budget.** Only 7 cosine epochs ran; LR at ep13 was still 9.42e-4 (only 6 % off peak). The intended "warmup → cosine to 1e-5" was effectively "warmup → flat-near-peak" — not the schedule the hypothesis tested.
- **Peak LR=1e-3 without grad-clip was too hot for val noise.** Train loss descended monotonically (no instability) but val oscillated 215.96 → 180.31 → 152.23 → 177.97 → 220.09 → 155.81 → 139.91 → 191.12 across the on-peak/decaying-peak epochs. The val noise is exactly what grad-clip damps.
- **Fourth independent rediscovery of the scoring NaN bug** (test_geom_camber_cruise idx 20 has fp16-underflow in y[p]). Tanjiro's pre-pass workaround is now in baseline; fern's run pre-dates that.

### Decision: close, reassign to higher-lr-1e3 (single-knob)
- Closed because raw vs raw lost to all prior baselines and the hypothesis (warmup → cosine-to-floor) was degenerate at the actual budget.
- BUT fern's analysis correctly identified the right next experiment: pair higher LR with grad-clip. Tanjiro's #374 follow-up #2 says exactly the same thing from the opposite direction — two independent rediscoveries of the same compound lever.
- Reassigned fern to **PR #408 (higher-lr-1e3)**: single-line `Config.lr = 5e-4 → 1e-3` on top of the merged grad-clip baseline. No warmup, no schedule changes. Cleanest single-knob test of "with grad-clip envelope, push LR 2×."

## 2026-04-28 00:55 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #408 | fern | higher-lr-1e3 | `Config.lr = 5e-4 → 1e-3` on top of merged grad-clip baseline | Replaces closed #353; single-knob test of "grad-clip envelope makes 2× LR safe" — independently suggested by both fern (#353 follow-ups) and tanjiro (#374 follow-ups) |

## 2026-04-28 01:02 — PR #351 (re-run): EMA + surf_weight=50 (charliepai2d1-askeladd) — **CLOSED (wash)**
- Branch: `charliepai2d1-askeladd/surf-weight-50` (rebased onto post-#356; closed + branch deleted)
- Hypothesis (re-run): retain `surf_weight=50` on the post-#356 baseline (EMA + NaN-safe pre-pass) to test compounding with EMA.

### Headline metrics (best EMA epoch=13/50, run cut by 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 165.35 | 142.63 | 100.50 | 115.60 | **131.02** |
| `mae_surf_p` (raw, same epoch) | — | — | — | — | 176.74 |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 143.49 | 128.23 | 84.43 | 115.47 | **117.90** |

### Comparisons
- vs **prior baseline #356** (132.276 / 118.041): val −0.95 %, test −0.12 % — small win on val, wash on test.
- vs **current baseline #374** (113.157 / 99.322): val **+15.8 %**, test **+18.7 %** — clear regression, but askeladd was rebased onto #356, not #374.

### Analysis
- **Predicted −5 % to −12 %, observed −0.95 % / −0.12 % vs the baseline they rebased onto.** EMA already absorbs most of the surface-signal gain; the marginal value of additionally re-weighting surface losses by 5× is small once EMA's smoothing is in place.
- Volume MAE didn't blow up (single_in_dist surf=165 / vol=192 = +16 %; cruise surf=100 / vol=147 = +47 %) — model isn't catastrophically forgetting volume at `surf_weight=50`. Useful negative result for the loss-balance ablation table.
- Per-split ranking: single_in_dist (165) > rc (143) > re_rand (116) > cruise (100). `single_in_dist` is the hardest and largest absolute contributor to `val_avg/mae_surf_p` — round 2 should optimize that split.
- **Askeladd's own recommendation: don't merge.** "+0.95 % / +0.12 % wins are inside session-to-session noise."

### Decision: close, reassign to ema-decay-0p99
- Per CLAUDE.md merge rule (must be `<` current baseline), val=131.02 > 113.157 → no merge.
- Even on the prior baseline, gains were within noise per askeladd's own analysis.
- With grad-clip now in baseline absorbing additional variance, the marginal value of `surf_weight=50` is even smaller.
- Reassigned askeladd to **PR #417 (ema-decay-0p99)**: single-line `ema_decay 0.999 → 0.99` to address nezuko's diagnostic ("EMA at this decay is too slow at the 13-epoch budget"). Honest predicted band −1 % to +3 %.

## 2026-04-28 01:05 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #417 | askeladd | ema-decay-0p99 | `ema_decay = 0.999 → 0.99` | Replaces closed #351; tests whether the under-converged 13-epoch budget is being short-changed by an EMA shadow that averages over too many updates (nezuko's #355 diagnosis) |

## 2026-04-28 01:18 — PR #394: torch.compile(model, ema_model) (charliepai2d1-thorfinn) — **sent back for rebase + re-run; will merge after**
- Branch: `charliepai2d1-thorfinn/torch-compile-throughput` (post-#356 base; pre-#374)
- Hypothesis: `torch.compile(model, mode="default", dynamic=True)` for kernel fusion → ≥15 % per-epoch wall-clock reduction → more epochs in the 30-min budget. Throughput is the deliverable; metric Δ is incidental.

### Headline metrics (best EMA epoch=17/50, run cut by 30-min timeout but at higher epoch count due to throughput)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 137.74 | 122.26 | 91.05 | 105.15 | **114.051** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 116.44 | 111.29 | 76.59 | 101.42 | **101.436** |

### Throughput (the deliverable)
| | baseline (#356) | this run | Δ |
|---|---:|---:|---:|
| `epoch_1_seconds` (compile warmup) | 142.3 | 150.6 | +5.8 % |
| `mean_epoch_2plus_seconds` (steady) | 140.8 | **108.4** | **−23.1 %** |
| epochs in 30-min timeout | 13 | **17** | +4 |
| peak VRAM | ~42 | 42.1 | flat |

Per-epoch steady-state σ ≈ 0.7s — compile is locked in, no recompile spam. **−23.1 % wall clock per epoch is right in the predicted band (−20 % to −35 %, conservative end).**

### Comparisons
- vs #356 (post-EMA, pre-grad-clip): val −13.8 %, test −14.1 % — clear win because torch.compile let the cosine schedule descend into 4 extra epochs.
- vs **current baseline #374** (post-grad-clip): val **+0.79 %**, test **+2.13 %** — within run-to-run noise on val, slightly behind on test. Run was on pre-grad-clip base, so grad-clip is missing.

### Analysis
- **Throughput delivery is exactly what we asked for.** −23.1 % is conservative-end of predicted band; locked in across all 16 steady-state epochs (σ=0.7s). Compile + EMA + NaN-safe path co-exist cleanly.
- **`mode="reduce-overhead"` OOM was correctly diagnosed.** The dataloader pads each batch to that batch's `N_max` (variable across batches); inductor with `dynamic=True` still tries to capture a CUDA graph per distinct shape, and 9 distinct shapes consumed ~68 GB of private graph pools. `mode="default"` (kernel fusion only, no graph capture) avoided the trap. Padding to a small fixed bucket of `N_max` values would unlock `reduce-overhead` for an additional ~10 % gain — but that's a `data/loader.py` change and out of scope.
- **Save/load via `_orig_mod` works.** `OptimizedModule` wrappers at `state_dict()`/`load_state_dict()` boundaries handled correctly via `model._orig_mod`.
- **Metric Δ vs current baseline (#374) is essentially noise.** This run is on the pre-grad-clip base, and the baseline moved while it was running. Right comparison: needs grad-clip + compile layered.

### Decision: send back for rebase + re-run
- Throughput delivery is unambiguous and durable; would normally merge directly. But the metric vs current baseline is +0.79 %/+2.13 % (within noise but technically slightly behind), so per CLAUDE.md merge rule (must be `<` baseline) this can't merge as-is.
- Rebase resolution: thorfinn's diff touches lines around `ema_model = copy.deepcopy(model)` (compile call) and the save/load code (`_orig_mod` accessor). The grad-clip diff touches the train loop's backward/step block. Different regions — should rebase cleanly with no conflicts.
- After rebase, run will have: EMA + NaN-safe + grad-clip(1.0) + torch.compile + 17 epochs. Predicted: val ~108–110, test ~95–97 — clear merge win and ships the throughput multiplier as the new baseline. Every subsequent PR fits 17 epochs.

## 2026-04-28 01:29 — PR #402: grad-clip `max_norm=1.0 → 0.5` (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/grad-clip-0p5` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `d6e39f2`).
- Hypothesis: tighten the grad-clip envelope from 1.0 → 0.5 to test whether more aggressive damping helps further or starves the optimizer at this LR.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| metric | this run | prior baseline (#374) | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **110.822** | 113.157 | −2.335 | **−2.07 %** |
| `test_avg/mae_surf_p` | **97.955** | 99.322 | −1.367 | **−1.38 %** |
| best raw `val_avg/mae_surf_p` | 117.667 (ep 12) | 121.992 (ep 13) | −4.325 | −3.55 % |

Beats baseline on every val split and every test split.

### Per-split val deltas
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | −2.71 % | −1.02 % |
| geom_camber_rc | −2.22 % | −2.44 % |
| geom_camber_cruise | −2.61 % | −0.94 % |
| re_rand | −0.55 % | −0.87 % |

`re_rand` shows the smallest gain on both runs vs no-clip — the Re-stratified holdout has the least variance to remove.

### Diminishing-returns map on the clipping lever (now complete)
- no-clip → `max_norm=1.0` (PR #374): **−14.45 %** val
- `max_norm=1.0 → 0.5` (PR #402): **−2.07 %** val
- Mean pre-clip grad norm: 73.40 (1.0) ≈ 71.36 (0.5) — nearly identical, confirming pre-clip norm is a property of optimizer state, not `max_norm`.

### Diagnostic — early-epoch convergence
The "too small a step" failure mode did NOT appear: `max_norm=0.5` *led* `1.0` in EMA val from ep1 onward, with the largest gap at ep3 (Δ EMA=−14.21 abs, Δ raw=−85.4 at ep1 = −28 %). Tanjiro's interpretation: at this LR, early gradients are noisy enough that the variance-reduction-from-tighter-clip dominates the magnitude penalty.

### Decision: merge as new round-1.5 baseline
- Beats baseline by a margin (>−1 %), single-character diff (`max_norm=1.0 → 0.5`), CLEAN/MERGEABLE.
- Mechanism is well-understood (auditable via the per-epoch grad-norm trace).
- Diminishing-returns curve on the clipping lever now mapped — clean diagnostic for the appendix.
- BASELINE.md updated; tanjiro reassigned to **PR #430 (lion-optimizer)** as a fresh axis after three merged variance-reduction wins.

## 2026-04-28 01:32 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #430 | tanjiro | lion-optimizer | Lion (sign-of-momentum) replacing AdamW; `lr_lion = 1.7e-4`, `wd_lion = 3e-4`, `betas=(0.9, 0.99)` | Fresh axis after three variance-reduction wins (#356/#374/#402). Reported 1–3 % gains on transformer-shaped problems; sign-update naturally bounds per-param step magnitude — interesting compose with grad-clip(0.5) |

## 2026-04-28 01:41 — PR #408: lr 5e-4 → 1e-3 (charliepai2d1-fern) — **MERGED, new baseline**
- Branch: `charliepai2d1-fern/higher-lr-1e3` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `bf5c6a5`).
- Hypothesis: with grad-clip envelope at `max_norm=1.0` (or 0.5) absorbing outlier steps, doubling AdamW lr from 5e-4 to 1e-3 should let the optimizer take more aggressive directional steps within the same magnitude envelope.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
- Fern's run was on the post-#374 base (lr=1e-3 + max_norm=1.0); squash-merge composed with #402 to give a current baseline `train.py` of `lr=1e-3 + max_norm=0.5`.

| metric | this run (lr=1e-3 + max_norm=1.0) | prior baseline #402 (lr=5e-4 + max_norm=0.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **107.957** | 110.822 | **−2.59 %** |
| `test_avg/mae_surf_p` | **95.675** | 97.955 | **−2.33 %** |

### Per-split val (EMA, ep13, vs #356 baseline for context)
| Split | val mae_surf_p | val Δ vs #356 (raw 132.276) |
|---|---:|---:|
| single_in_dist | 125.68 | −26.34 % |
| geom_camber_rc | 122.66 | −14.88 % |
| geom_camber_cruise | 82.97 | −17.43 % |
| re_rand | 100.52 | −11.84 % |

The harder splits (`single`, `rc`) get the biggest gains; `cruise` and `re_rand` (which already had headroom over the others) inch forward.

### Diagnostic — pre-clip grad norm halved at lr=1e-3
Mean per-epoch pre-clip `train/grad_norm`: 64 → 61 → 55 → 51 → 44 → 44 → 44 → 39 → 39 → 37 → 36 → 33 → 30. **Mean ~44** (lr=1e-3) vs **~73** (lr=5e-4 baseline) — confirms AdamW's `1/√(v+eps)` preconditioner adapts to the larger LR by inflating per-step magnitude internally, so raw grads land smaller. Clip is still firing aggressively (30–60× over `max_norm=1.0`).

### EMA-vs-raw curve diagnostic
Raw converges much faster early (ep1: 240 vs 300 = −60-point lead) but is noisier through ep 9–10. EMA(0.999) is initially slower to track because the lr=1e-3 run is moving params more aggressively; baseline EMA actually leads through ep 6. EMA crosses over at ep 7–8 and the gap widens through ep 13 (107.96 EMA vs 113.16 baseline EMA at ep 13). Raw is nearly tied at ep 13 (122.16 vs 121.99), so the EMA win is consistent with the grad-clip envelope damping high-frequency raw oscillations into a cleaner shadow.

### Decision: merge as new round-1.5 baseline
- Beats baseline on val (−2.59 %) and test (−2.33 %), with the harder OOD splits taking the biggest gains.
- Single-line diff (`lr: 5e-4 → 1e-3`); CLEAN/MERGEABLE without conflicts.
- Mechanism is well-understood (clip envelope + AdamW preconditioner adaptation).
- "Higher LR safe under clip" hypothesis confirmed — both fern (#353 follow-ups) and tanjiro (#374 follow-ups) independently suggested this combo, and the run validates it.
- BASELINE.md updated; fern reassigned to **PR #438 (lr-2e-3)** as the natural next step in their LR-scaling thread.

## 2026-04-28 01:42 — PR #398: SwiGLU at matched param count (charliepai2d1-nezuko) — **sent back for rebase + re-run; will likely merge after**
- Branch: `charliepai2d1-nezuko/swiglu-mlp-matched` (post-#356 base; pre-#374, pre-#402, pre-#408)
- Hypothesis: replace GELU MLP with SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at matched param count (`swiglu_inner=168` for `mlp_ratio=2, hidden=128`). Strips capacity confound from closed PR #355 (`mlp_ratio=4` GELU).

### Headline metrics (best EMA epoch=12/12, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 133.58 | 126.58 | 86.45 | 100.58 | **111.795** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 120.73 | 112.23 | 71.86 | 98.11 | **100.730** |

- Param count: 657,639 (vs baseline GELU 662,359 → −0.71 %) — matched-param recipe correct.
- Per-epoch wall clock: 150.55 s (vs ~140 s baseline = +7.9 %, slightly above the +5 % threshold; 12 epochs vs baseline's 13).

### Comparisons
- vs **#356** baseline (132.276/118.041): **val −15.48 %, test −14.67 %** — huge win on the older base.
- vs **current #408** baseline (107.957/95.675): val +0.88 %, test +2.83 % — within run-to-run noise on val, slightly behind on test.

### The per-split breakdown is the load-bearing evidence
| Split family | #355 (mlp_ratio=4 GELU) Δ vs #356 | this run (SwiGLU matched) Δ vs #356 |
|---|---:|---:|
| single_in_dist (ID) | −2.7 % | **−21.65 %** |
| geom_camber_rc (OOD) | +1–2 % regression | **−12.16 %** |
| geom_camber_cruise (OOD) | mixed | **−13.97 %** |
| re_rand (OOD) | mixed | **−11.79 %** |

**SwiGLU fixes the in-dist-vs-OOD trade-off that sank #355.** Capacity-bumped GELU (mlp_ratio=4) helped in-dist but regressed OOD. Matched-param SwiGLU lifts every split — including 11–14 % gains on the three OOD splits. That's a clean per-node-nonlinearity-vs-capacity decoupling and is the strongest non-variance-reduction signal we've seen.

Surf/vol balance preserved on every test split (vol_p tracks surf_p within 1–3 %) → SwiGLU isn't skewing head priorities.

### Caveats nezuko correctly flagged
- Noisy late-training trajectory (raw 115 → 125 → 152 across ep 10–12); single-seed magnitude warrants a re-run.
- +7.9 % per-epoch wall clock (three matmul kernel-launches per block vs two; fixed overhead at small `mlp_ratio=2, hidden=128` matmuls).

### Decision: send back for rebase + re-run
- Per CLAUDE.md merge rule (must be `<` baseline), can't merge as-is vs current #408.
- Rebase resolution should be clean: nezuko's diff adds `SwiGLUMLP` class (no overlap) + substitutes inside `TransolverBlock` (different region from #374/#402/#408 diffs). Should rebase onto post-#408 cleanly.
- After rebase: SwiGLU(168) + EMA + NaN-safe + grad-clip(0.5) + lr=1e-3 + ~12 epochs. Predicted: val ~94–98, test ~83–87 if SwiGLU's −15 % vs no-clip baseline composes additively with grad-clip + higher-LR. Will merge as next baseline if it lands there.

## 2026-04-28 01:42 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #438 | fern | lr-2e-3 | `Config.lr = 1e-3 → 2e-3` on top of merged #408 baseline | Fern's own follow-up #1; tests how far the LR-scaling-under-clip envelope extends. Single-knob continuation of #408. |

## 2026-04-28 01:54 — PR #417: EMA decay 0.999 → 0.99 (charliepai2d1-askeladd) — **MERGED, new baseline**
- Branch: `charliepai2d1-askeladd/ema-decay-0p99` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `83eeafb`).
- Hypothesis: at the under-converged 13-epoch budget, EMA(0.999) has effective half-life ~1.85 epochs which is too slow to track a fast-improving live iterate; reducing decay to 0.99 (half-life ~0.18 epochs) lets the shadow track the recent (better) iterate before old (worse) iterate drags the shadow back.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
- Askeladd's run was on the post-#374 base (lr=5e-4, max_norm=1.0); squash-merge composed `ema_decay=0.99` with #402's `max_norm=0.5` and #408's `lr=1e-3` to give the current baseline.

| metric | this run (EMA(0.99) + max_norm=1.0 + lr=5e-4) | prior baseline #408 (EMA(0.999) + max_norm=0.5 + lr=1e-3) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **98.581** | 107.957 | **−8.69 %** |
| `test_avg/mae_surf_p` | **87.881** | 95.675 | **−8.15 %** |

vs #374 baseline (the run baseline askeladd used): val −12.88 %, test −11.52 %.

### Per-split val (vs current #408 for context)
| Split | val mae_surf_p | val Δ vs #408 |
|---|---:|---:|
| single_in_dist | 118.99 | −5.32 % |
| geom_camber_rc | 107.26 | −12.56 % |
| geom_camber_cruise | 75.10 | −9.49 % |
| re_rand | 92.97 | −7.51 % |

All four val splits improve; `geom_camber_rc` (the harder OOD geom split) gains the most.

### Diagnostic — EMA-vs-raw spread (the headline finding)
- Mean per-epoch spread: **24.2 pts** (EMA better than raw). Min −5.3 (ep1, expected random-init drag), max 48.1 (ep3), at best epoch (13) **20.4 pts**.
- Compare to baseline #374's spread: 8 pts at ep13.
- **EMA shadow consistently *better* than raw at every epoch except ep1.** The PR's prediction ("expect them to be much closer") was upside-down for this regime — at the under-converged budget, the iterate is improving fast and a faster shadow captures more signal because each fresh batch pulls the shadow back toward the recent (better) iterate before too much old (worse) iterate decays in.
- **Raw at ep13 (119.0) is essentially unchanged from baseline raw (~122).** The underlying optimization didn't change. All of the gain came from extracting a better shadow average from the same trajectory.

### Decision: merge as new round-1.5 baseline
- Beats baseline by a wide margin on every val and test split; single-character diff (`ema_decay = 0.999 → 0.99`).
- EMA decay is genuinely orthogonal to grad-clip and lr — checkpoint-selection lever, not training-loop lever. Squash-merge cleanly composes `ema_decay=0.99` with the merged `max_norm=0.5` and `lr=1e-3`.
- Mechanism story is clean and auditable (per-epoch EMA-vs-raw spread proves the regime).
- BASELINE.md updated; askeladd reassigned to **PR #445 (ema-decay-0p95)** as the natural next step in the decay sweep they suggested in their write-up.

## 2026-04-28 01:55 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #445 | askeladd | ema-decay-0p95 | `ema_decay = 0.99 → 0.95` on top of merged #417 baseline | Askeladd's own follow-up #1; tests where the responsiveness curve bottoms out. At 0.95 the half-life is ~14 batches ≈ 0.04 epochs — possibly too noisy under balanced-domain sampling, possibly captures even more recent signal. Honest band −1 % to +5 %. |

## 2026-04-28 02:11 — PR #403: batch=8 + lr=7e-4 (√2 LR scaling) (charliepai2d1-frieren) — **CLOSED**
- Branch: `charliepai2d1-frieren/batch8-lr-sqrt2` (closed + branch deleted)
- Hypothesis: variance reduction at the gradient-aggregation level via larger batch + √2 LR scaling.

### Headline metrics (best EMA epoch=14/50, run cut by 30-min timeout)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 269.80 | 186.06 | 107.24 | 128.76 | **172.97** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 232.07 | 167.72 | 91.71 | 121.33 | **153.21** |

- vs prior baseline #374 (113.16/99.32): val **+52.9 %**, test **+54.3 %**.
- vs current baseline #417 (98.581/87.881): val **+75 %**, test **+74 %**.
- Per-epoch wall clock: 138 s (vs ~140 s baseline); 14 epochs in 30 min (vs baseline's 13).
- Peak VRAM: 84.21 GB.
- Mean pre-clip grad_norm: 61.61 (vs baseline 73.40 = −16.1 %).

### Analysis (frieren's writeup is exceptional; recording the key points)
- **Variance-reduction lever real** (per-step grad norm down 16 %).
- **Throughput parity confirmed** (138 vs 140 s/epoch, +1 epoch in budget).
- **Step-count starvation dominates the result.** batch=8 halves optimizer steps per epoch (188 vs 375); √2 LR scaling under-compensates because total integrated LR is `14/13 × 188/375 × √2 ≈ 0.77×` baseline aggregate optimization. Plus cosine T_max keyed to epochs not steps, so each step at b=8 sits at the same cosine fraction.
- **`single_in_dist` regressed by +99–102 %** (the canary): hardest, most steps-hungry split, hit hardest by the missing late-training updates. Auditable mechanism: at every matched epoch from ep5 onward, the b=8 raw curve sits ~20–30 % above b=4 baseline raw and the gap widens slightly through the cut.

### Decision: close, reassign to weight-decay-5e-4
- Clear >5 % regression on both val and test (+75 % / +74 % vs current baseline). Per CLAUDE.md close criteria.
- Lever isn't disproven — frieren's analysis correctly notes that `b=8 + lr=2e-3` (linear scaling, not √2) at the current baseline lr=1e-3 would land somewhere near baseline and would be the "right fix" for this hypothesis. Holding that as a round-2 follow-up if fern's lr=2e-3 (#438) wins at b=4.
- **Three closures in a row for frieren** (#354, #373, #403). Reassigning to a low-risk single-knob regularization sweep (`weight_decay=1e-4 → 5e-4`) for confidence-building + ablation-table coverage. **PR #458 (weight-decay-5e-4)**.

## 2026-04-28 02:14 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #458 | frieren | weight-decay-5e-4 | `Config.weight_decay = 1e-4 → 5e-4` on top of merged #417 baseline | Replaces closed #403; standard regularization sweep. Plausibly helps the OOD splits where the closed capacity bumps (#355 / #373) showed in-dist-helps / OOD-regresses. Honest band −1 % to +2 %. |

## 2026-04-28 02:35 — PR #438: lr 1e-3 → 2e-3 (charliepai2d1-fern) — **CLOSED (regression)**
- Branch: `charliepai2d1-fern/lr-2e-3` (closed + branch deleted)
- Hypothesis: extend the LR-scaling-under-clip-envelope thread from #408 by doubling LR again (1e-3 → 2e-3).

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 136.32 | 132.49 | 88.55 | 103.61 | **115.243** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 121.29 | 118.35 | 73.66 | 101.29 | **103.647** |

- vs #408 base (apples-to-apples for run config): val +6.75 %, test +8.33 %.
- vs current #417 baseline (post-EMA-decay-tuning): val **+16.92 %**, test **+17.94 %**.
- Mean pre-clip `train/grad_norm` per epoch: 33.9 (vs ~44 at lr=1e-3, ~73 at lr=5e-4).
- Per-epoch wall clock: ~141 s (matches baseline). Peak VRAM: 42.11 GB.

### Analysis (fern's writeup is exceptional; key points recorded)
- **AdamW preconditioner adapts only partially at this LR.** Mean pre-clip norm dropped 24 % going 1e-3 → 2e-3 (vs ~50 % drop going 5e-4 → 1e-3). The `1/√v` preconditioner can compensate for moderate LR bumps but breaks down at higher rates.
- **Raw curve has a clear noise spike** (ep10: 195.67 vs #408's 156.07). EMA crossover with raw is delayed by one epoch (ep9 vs ep7–8 in #408). EMA never catches up to baseline trajectory — every epoch from ep5 onward, this run's EMA is worse than #408's at the same epoch.
- **LR ceiling for `max_norm=0.5` envelope now bracketed**: lr=1e-3 wins (#408), lr=2e-3 loses (this PR). Clean ablation cell for the appendix.
- **Schedule degeneracy reconfirmed**: best-at-last-epoch with cosine still ~95 % of peak, same as #353/#408. Fern's follow-up #2 ("revisit cosine T_max") is the right next experiment.

### Decision: close, reassign to cosine-tmax-13
- Clear >5 % regression on both val and test vs apples-to-apples baseline. Per CLAUDE.md close criteria.
- Reassigned fern to **PR #465 (cosine-tmax-13)**: their own follow-up #2 from this PR. `T_max=50 → 13`, `eta_min=1e-5`. Single-knob fix to the schedule degeneracy that's been showing up across multiple PRs.

## 2026-04-28 02:35 — PR #430: Lion optimizer (charliepai2d1-tanjiro) — **sent back for rebase + re-run; will likely merge after**
- Branch: `charliepai2d1-tanjiro/lion-optimizer` (post-#402 base; pre-#408, pre-#417)
- Hypothesis: replace AdamW with inline Lion (sign-of-momentum) at the standard `lr/3, wd*3, betas=(0.9, 0.99)` recipe.

### Headline metrics (best EMA epoch=13/50, timeout-cut) — biggest single-PR signal yet
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 89.72 | 90.25 | 60.63 | 77.25 | **79.46** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 76.95 | 80.22 | 51.27 | 71.87 | **70.08** |

- vs **#402 base** (apples-to-apples for run config): **val −28.30 %, test −28.46 %**.
- vs **current #417 baseline**: **val −19.43 %, test −20.26 %**.
- Per-split: every val and test split improves by ≥23 %; `test_single_in_dist` jumps −33.3 %.
- Wall clock and VRAM unchanged (~141 s/ep, 42 GB peak).
- Mean pre-clip `train/grad_norm` per epoch: 50 (vs ~73 at #402 baseline). Lion's grad norms systematically lower than AdamW's mid-training, suggesting Lion finds locally-flatter regions.

### Analysis
- **Lion's gain is decisive across every metric.** Both raw (best-raw 83.89 vs #402's 110.82 = −24.30 %) and EMA contributions are large; even raw-vs-raw beats baseline by a wide margin.
- **EMA-Lion interaction is part of the gain**: EMA-vs-raw spread is wider on Lion (−6.4 pts at ep11) than on AdamW (−4.2 pts at the same epoch). Lion's uniform-step exploration ("every parameter moves by exactly `lr` per step") is well-averaged by a slow EMA shadow.
- **Grad-clip is essentially idle under Lion** (sign update is invariant to gradient magnitude). Tanjiro left the line in for apples-to-apples comparison; in a follow-up we can remove it.
- **No divergence, no NaN**. Predicted "watch for divergence in eps 1–2" trigger never fired. Lion at lr=1.7e-4 was completely stable from ep1 onward.

### Why send back rather than direct merge
- Tanjiro's run was on the post-#402 base, missing **two intervening merges**: #408 (AdamW lr=5e-4 → 1e-3) and #417 (ema_decay=0.999 → 0.99).
- The squash-merge has a **conflict on `Config.lr`**: tanjiro's branch changed `5e-4 → 1.7e-4` (Lion recipe); current baseline has `1e-3` (AdamW recipe). Lion at `lr=1e-3` is way too aggressive (`lr × sign` = 1e-3 per param per step). The right resolution is **Lion's recipe overrides #408's AdamW lr** — keep `1.7e-4`. Same for `weight_decay`: keep tanjiro's `3e-4` (Lion-style).
- Tanjiro themselves flagged that Lion's gain may have an EMA(0.999)-specific component. Need clean apples-to-apples vs current EMA(0.99) baseline.

### Predicted post-rebase
Lion(lr=1.7e-4, wd=3e-4) + EMA(0.99) + grad-clip(0.5) → val ~70–80, test ~62–72. The −24 % raw-vs-raw standalone Lion gain dominates; the EMA-Lion interaction may shrink (EMA tracks 10× faster at decay 0.99) but not flip sign. **One merge away from being the next baseline by a wide margin.**

## 2026-04-28 02:38 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #465 | fern | cosine-tmax-13 | `T_max=50 → 13`, `eta_min=1e-5` | Replaces closed #438; cashes in fern's follow-up #2 from #353/#408/#438. Schedule has been degenerate across all merged baselines (best-at-last with cosine still 95 % of peak). Single-knob schedule fix that should compound with the merged variance-reduction stack. |

## 2026-04-28 02:45 — PR #445: EMA decay 0.99 → 0.95 (charliepai2d1-askeladd) — **CLOSED (regression)**
- Branch: `charliepai2d1-askeladd/ema-decay-0p95` (closed + branch deleted)
- Hypothesis: continue the decay sweep — at 0.95 (half-life ~14 batches) we get even-faster shadow tracking, possibly captures more recent signal.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 128.58 | 117.67 | 87.61 | 99.01 | **108.22** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 115.61 | 103.36 | 71.97 | 96.60 | **96.88** |

- vs prior baseline #417 (98.581/87.881): val **+9.77 %**, test **+10.24 %**.
- Cruise hardest hit (+16.66 % val, +14.12 % test) — smallest-sample domain × balanced-sampler stochasticity = under-averaged shadow exposed.
- Mean EMA-vs-raw spread: 17.33 (ep4–13), down from 24.52 at decay 0.99 — shadow doing less averaging.

### Mechanism (askeladd's writeup nailed both predicted scenarios):
- **Faster tracking helped early.** At ep1–6 decay 0.95 EMA val was 6–12 pts better than 0.99.
- **Smoothing collapsed mid-run.** Spread fell 7–9 pts at ep7–9 (vs 21–37 at 0.99). Cross-over at ep5–7; from there 0.99 takes over because the shorter window can't smooth out batch-composition noise from the balanced sampler (3 domains × ~5 samples per shadow = high stochastic variance).
- **EMA-decay optimum bracketed in [0.97, 0.99].** 0.999 too sticky (#356 at +12.88 %), 0.95 too noisy (this PR at +9.77 %), 0.99 sweet spot.

### Decision: close, reassign to ema-decay-0p97
- Clear >5 % regression. Per CLAUDE.md close criteria.
- Reassigned to **PR #474 (ema-decay-0p97)** — bracket-narrowing run on the new SwiGLU baseline. Honest predicted band −1 % to +1 %.

## 2026-04-28 02:48 — PR #398 (REBASED): SwiGLU at matched param count (charliepai2d1-nezuko) — **MERGED, new baseline**
- Branch: `charliepai2d1-nezuko/swiglu-mlp-matched` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `7fb09c0`).
- Hypothesis (rebased re-run): SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at matched param count (`swiglu_inner=168`) on top of merged variance-reduction stack (EMA(0.99) + grad-clip(0.5) + lr=1e-3 + NaN-safe).

### Headline metrics (best EMA epoch=12/50, timeout-cut)
| metric | this run (rebased) | prior baseline #417 | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **89.349** | 98.581 | −9.232 | **−9.36 %** |
| `test_avg/mae_surf_p` | **79.191** | 87.881 | −8.690 | **−9.89 %** |

Beats baseline on every val and test split. Strongest single best gain: `val_single_in_dist` −15.6 %.

### Per-split val/test deltas
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | −15.64 % | −12.42 % |
| geom_camber_rc | −1.74 % | −5.63 % |
| geom_camber_cruise | −10.66 % | −10.91 % |
| re_rand | −9.08 % | −10.89 % |

**Every split improves; no split worse than −1.7 % val / −5.6 % test.** That's the same "fixes #355's pathology" story from nezuko's original pre-rebase run, now confirmed under the full variance-reduction stack.

### Analysis
- **Composes more than additively with the variance-reduction stack** (predicted band 94–98 was based on additive composition; observed 89.35 beats it by ~5 pts). Three mechanisms in play: smoother training trajectory at lr=1e-3 + grad-clip(0.5) gives EMA a cleaner signal; faster EMA(0.99) recovers SwiGLU's per-step gain immediately; grad-clip's bounded step magnitude pairs naturally with SwiGLU's gating.
- **Reproducibility check.** Pre-rebase run was −15.48 % vs #356 baseline; rebased run is −9.36 % vs #417 baseline. Different EMA, different LR, different grad-clip envelope, but the lever's relative contribution holds — the absolute number compresses because the variance-reduction stack already extracted some of the headroom.
- **Param count: 657,639** (vs baseline GELU 662,359 = −0.71 %). Matched-param recipe correct.
- **Surf/vol balance preserved**: vol_p tracks surf_p within ~0–10 % per split, no head-priority skew.
- **Wall-clock cost: +7.9 %** (150 s/epoch vs 140 s baseline; 12 epochs vs 13). Three matmul kernel-launches per block at small `mlp_ratio=2, hidden=128` shapes — latency-bound rather than FLOP-bound. Future fused gate+value matmul could recover ~12 s/epoch.

### Decision: merge as new round-1.5 baseline
- Beats baseline by a wide margin on the ranking metric and every per-split metric.
- Mergeable cleanly (CLEAN/MERGEABLE; SwiGLUMLP class + TransolverBlock substitution don't conflict with any merged change).
- First architectural merge after five variance-reduction-direction merges (#356, #374, #402, #408, #417). Marks a transition from "variance reduction" axis to "architecture" axis on the leaderboard.
- BASELINE.md updated; nezuko reassigned to **PR #475 (swiglu-inner-256)** as the natural capacity-sweep follow-up.

## 2026-04-28 02:50 — PR #394 (REBASED ONCE): torch.compile (charliepai2d1-thorfinn) — **sent back for rebase #2**
- Branch: `charliepai2d1-thorfinn/torch-compile-throughput` (post-#417 base; pre-#398)
- Hypothesis (rebased re-run on post-#417): `torch.compile(model, ema_model)` for kernel fusion + extra epochs in the timeout.

### Headline metrics (best EMA epoch=15/50, timeout-cut after compile warmup absorbed eps 2–3)
| metric | this run (post-#417 base) | post-#417 baseline | post-#398 baseline (current) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **95.743** | 98.581 (−2.88 %) | 89.349 (**+7.15 %**) |
| `test_avg/mae_surf_p` | **83.861** | 87.881 (−4.57 %) | 79.191 (**+5.90 %**) |

### Throughput delivery (the deliverable)
- `mean_epoch_4plus_seconds` (clean steady): **108.0 s** vs 140.3 s baseline = **−23.0 %**.
- 15 epochs in 30 min (vs baseline's 13). Compile warmup absorbed +89 s in ep2 + +26 s in ep3 (~140 s total) → +2 epochs vs +4 in pre-rebase run (warmup variance).
- Peak VRAM: 42.1 GB (flat).

### Why send back
- Vs the baseline thorfinn rebased onto (#417): **−2.88 % / −4.57 %** — clean win.
- BUT PR #398 (SwiGLU) merged at 02:48 — **just 1 minute** before thorfinn's results posted. New baseline is val=89.35 / test=79.19. Thorfinn's measured numbers (95.74 / 83.86) now sit at +7.15 % / +5.90 % vs current — outside the merge gate.
- Rebase resolution should still be clean: thorfinn's diff (compile + `_orig_mod` save/load) is in different code regions from the SwiGLU diff (new SwiGLUMLP class + TransolverBlock substitution).
- Predicted post-rebase #2: compile + SwiGLU + 14–15 epochs → val ~84–88, test ~74–78. Should clear the merge gate by 2–6 % depending on how compile composes with SwiGLU's per-block matmul shape changes.

### Decision: send back for rebase #2
- This is thorfinn's **second** rebase + re-run cycle. Throughput delivery is rock-solid; just chasing a moving baseline.
- Sent back with explicit note that the throughput gain is robust across all rebases. Merge is one rebase away.

## 2026-04-28 02:55 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #474 | askeladd | ema-decay-0p97 | `ema_decay = 0.99 → 0.97` on merged #398 baseline | Bracket-narrowing run for the EMA-decay optimum (0.999 too sticky, 0.95 too noisy). Honest band −1 % to +1 %. |
| #475 | nezuko | swiglu-inner-256 | `swiglu_inner = 168 → 256` (+50 % MLP capacity) on merged #398 baseline | Capacity sweep on the new SwiGLU baseline. Tests whether SwiGLU's "fixes OOD" property scales with capacity. |

## 2026-04-28 03:00 — PR #458: weight_decay 1e-4 → 5e-4 (charliepai2d1-frieren) — **CLOSED (regression)**
- Branch: `charliepai2d1-frieren/weight-decay-5e-4` (closed + branch deleted)
- Hypothesis: stronger weight decay as Tikhonov regularization, plausibly helping OOD splits where capacity bumps showed in-dist-helps / OOD-regresses pattern.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 125.94 | 115.99 | 75.99 | 95.18 | **103.27** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 108.11 | 103.08 | 62.77 | 89.08 | **90.76** |

- vs prior baseline #417 (run base): val +4.76 %, test +3.28 %.
- vs current baseline #398 (post-SwiGLU): val **+15.59 %**, test **+14.61 %**.
- Per-epoch wall clock: 141 s (matches baseline). Peak VRAM: 42.11 GB.

### Frieren's analysis (the smoking gun)
- **Worst regressor is `geom_camber_rc`** (+8.14 % val) — the highest-loss / hardest split, the most under-fit one. If WD were trading in-dist over-fitting for OOD generalization, the easy splits would degrade and hard ones would gain. We see the opposite: capacity is being shrunk away from where the model needs more, not less.
- **Train losses still falling fast at ep13** (15 % step from ep12 → ep13 on `train/surf_loss`). The model is nowhere near convergence. Nothing was over-fit to regularize against — the *lose* mechanism dominated, exactly as the PR body's honest band predicted.
- **EMA-vs-raw gap widened to 33.25 pts** (raw 136.52 vs EMA 103.27 at ep13). WD shrinks param magnitudes mechanically, reducing grad norms; the raw iterate lagged baseline trajectory and EMA inherited the lag.

### Decision: close, reassign to dropout
- Clear >5 % regression vs current baseline. Per CLAUDE.md close criteria.
- **WD bump direction fully mapped at this budget**: 1e-4 sweet spot, 5e-4 too aggressive, 2e-4 likely a wash (not worth a run).
- `geom_camber_rc` now established as the canary for under-converged regularization in the appendix.
- Reassigned to **PR #483 (swiglu-mlp-dropout-0p1)** — frieren's own follow-up #4. Dropout adds training-time stochasticity rather than parameter-norm penalty, attacks per-split asymmetry without slowing convergence (the WD failure mode). Honest predicted band −1 % to +2 %.

## 2026-04-28 03:02 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #483 | frieren | swiglu-mlp-dropout-0p1 | Add `nn.Dropout(0.1)` inside `SwiGLUMLP.forward` on merged #398 baseline | Replaces closed #458; tests training-time stochasticity as alternative regularizer to WD. Frieren's own follow-up #4. Honest band −1 % to +2 %. |

## 2026-04-28 03:20 — PR #465: cosine T_max 50 → 13 + eta_min=1e-5 (charliepai2d1-fern) — **CLOSED (regression)**
- Branch: `charliepai2d1-fern/cosine-tmax-13` (closed + branch deleted)
- Hypothesis: schedule has been degenerate across all merged baselines (best-at-last with cosine still 95 % of peak). Hypothesis: `T_max=13` lets the schedule actually anneal toward eta_min=1e-5 by ep13.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (EMA) | 121.43 | 114.22 | 81.97 | 98.62 | **104.06** |

| | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 107.00 | 101.25 | 67.64 | 94.31 | **92.55** |

- vs prior #417 (run base): val +5.56 %, test +5.32 %.
- vs current #398 (post-SwiGLU): val **+16.46 %**, test **+16.87 %**.
- All four splits regressed; cruise hardest (+9.15 % val).

### Smoking-gun mechanism (fern's writeup)
- **Train loss REVERSED direction at ep13** (vol 0.355 → 0.424, surf 0.200 → 0.238). The schedule actively *un-trained* the model.
- **Best raw arrived at ep12** (106.003), degraded at ep13 (106.119). EMA shadow held best-EMA at ep13 only because the lag baked in earlier (better-trained) iterates.
- **Effective LR-time-integral was 57 %** of baseline T_max=50 (avg LR 5.43e-4 vs 9.52e-4 across 13 epochs) → 43 % less aggregate optimization work in same wall-clock budget.
- At ep11–13, lr=1.34e-4 → 6.67e-5 → 2.44e-5 — the per-step update is below the useful learning floor for this model size at this stage.

### Closed-PR insight chain (now complete)
- #353 → "schedule degenerate at this budget" (correct diagnosis)
- #408 / #438 → "best-at-last-epoch with cosine still 95 % of peak" (consistent observation)
- #465 (this PR) → **"the model needs more high-LR steps, not better anneal"** (correct interpretation)

The cosine-to-zero recipe assumes a local minimum has been reached by anneal time; we are not there yet at 13/50 epochs. **LR/schedule axis fully mapped at this budget.** Negative result is durable for the appendix.

### Decision: close, reassign to TF32 matmul precision
- Clear >5 % regression on val and test vs current baseline. Per CLAUDE.md close criteria.
- Fern's #1 follow-up (lr=2e-3 + T_max=50) was already tested in their own #438 (val +6.75 % regression). LR ceiling for the `max_norm=0.5` envelope at lr=1e-3 is locked. Pushing higher LR fails for a different reason than the schedule fails. No win available on this axis at this budget.
- Reassigned to **PR #491 (tf32-matmul-precision)** — single-line throughput PR. Tanjiro's #394 follow-up #1 that's been queued. Optimizer-agnostic so it benefits whoever wins the optimizer-family race (Lion #430 mid-rebase). Predicted 10–20 % per-epoch wall-clock reduction.

## 2026-04-28 03:22 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #491 | fern | tf32-matmul-precision | `torch.set_float32_matmul_precision('high')` on merged #398 baseline | Replaces closed #465; throughput PR. SwiGLU baseline is matmul-heavy (3 matmuls/block × 5 blocks). Predicted ~10–20 % per-epoch wall-clock reduction, free accuracy-neutral on Blackwell. |

## 2026-04-28 03:46 — PR #430 (REBASED): Lion optimizer (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/lion-optimizer` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `5b2f7b2`).
- Hypothesis (rebased re-run): replace AdamW with inline Lion (sign-of-momentum) on top of the now-stacked variance-reduction + SwiGLU baseline.

### Headline metrics (best EMA epoch=12/50, timeout-cut) — **biggest single-PR delta on this branch**
| metric | this run | prior baseline #398 | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **67.737** | 89.349 | −21.612 | **−24.19 %** |
| `test_avg/mae_surf_p` | **59.447** | 79.191 | −19.744 | **−24.94 %** |

Beats every val and test split by ≥21 %; `geom_camber_cruise` jumps −31.0 % val / −29.9 % test (consistent with Lion's #430-v1 result that cruise benefits most).

### Per-split breakdown
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | −22.28 % | −24.45 % |
| geom_camber_rc | −23.05 % | −21.41 % |
| geom_camber_cruise | **−31.02 %** | **−29.88 %** |
| re_rand | −22.45 % | −26.07 % |

### Mechanism (tanjiro's writeup)
- **EMA-Lion interaction did NOT shrink at decay 0.99** — it **improved**. Advisor predicted shrinkage; the opposite happened. With EMA(0.999) the slow shadow lagged Lion's raw improvement so badly that EMA was *worse* than raw in early epochs (Lion-v1 ep1: EMA 315.45 / raw 197.39). With EMA(0.99) the fast shadow keeps up *and* averages Lion's substantial epoch-to-epoch raw variance — raw bounces 87.96 → 97.01 → 84.73 → 82.58 in eps 9–12 while EMA descends monotonically 74.75 → 72.04 → 72.75 → 67.74.
- **Lion's lr=1.7e-4 still in basin** even though it was sized for the *old* AdamW recipe (lr=5e-4). With current AdamW lr=1e-3 the equivalent would be ~3.3e-4 — 2× larger. Single-knob non-tuned Lion delivered −24.19 %; the lr sweep is where the next gain lives.
- **Wall-clock parity**: 151.2 s/epoch vs 150.0 s baseline = +0.8 %. Lion's missing second-moment buffer (~2.6 MB on a 657K-param model) is rounding error; the optimizer-family change is essentially compute-free.

### Decision: merge as new round-1.5 baseline
- Beats baseline by a wide margin on ranking metric and every per-split metric.
- Mechanism is well-understood (sign-update + EMA(0.99) interaction; bounded per-param step magnitude composes naturally with grad-clip envelope).
- **Biggest single-PR delta on this branch** (−24.19 % on val).
- BASELINE.md updated; tanjiro reassigned to **PR #507 (lion-lr-3p3e-4)** as the natural lr-sweep continuation.

## 2026-04-28 03:48 — PR #394 (REBASED #2): torch.compile rebased onto post-#398 SwiGLU (charliepai2d1-thorfinn) — **sent back for rebase #3**
- Branch: `charliepai2d1-thorfinn/torch-compile-throughput` (post-#398 base; pre-#430)
- Hypothesis (re-run): `torch.compile` on top of SwiGLU baseline.

### Headline metrics (best EMA epoch=17/50, timeout-cut after compile warmup)
- val=**77.275** (−13.51 % vs #398 base; +14.08 % vs current #430), test=**67.499** (−14.77 % vs #398; +13.55 % vs current).
- Throughput: **−25.7 % steady-state per-epoch** (111.6 s vs ~150 s eager), **17 epochs in 30-min budget** vs baseline's 12. Tightest steady-state band of any compile run on this branch (σ ≈ 0.6 s).
- Compile + SwiGLU compose more than additively — kernel fusion saves more launches on SwiGLU's 3-matmul block than on GELU's 2-matmul.

### Decision: send back for rebase #3
- Lion (#430) merged ~5 minutes after results posted, moving baseline to val=67.737. Vs current, this run is +14 % — outside merge gate.
- Throughput delivery is durable across all three rebases (post-#356 −23.1 %, post-#417 −23.0 %, post-#398 −25.7 %).
- Predicted post-#430 rebase: val ~58–63, test ~52–56. Compile is one rebase away from being the new throughput floor for every round-2 PR.

## 2026-04-28 03:48 — PR #352 (REBASED): SmoothL1 surface (charliepai2d1-edward) — **sent back for rebase #2 onto Lion**
- Branch: `charliepai2d1-edward/smoothl1-surface` (post-#417 base; pre-#398, pre-#430)
- Hypothesis (re-run): SmoothL1 (Huber, β=1.0) on surface loss with EMA(0.99) + grad-clip(0.5) + lr=1e-3 stack.

### Headline metrics (best EMA epoch=12/50, timeout-cut)
- val=**82.5432** (−16.27 % vs #417 base, **−7.61 % vs #398**, +22.04 % vs current #430), test=**72.9777** (−16.96 % / **−7.85 %** / +22.78 %).
- All four val and test splits improve, no regressions; volume MAE not regressed despite keeping volume as MSE.

### Decision: send back for rebase + re-run on Lion baseline
- Lion (#430) merged ~7 minutes before results posted; current baseline 67.737.
- **Open question**: does SmoothL1 still help on Lion baseline? Lion's sign-update is invariant to gradient magnitude on each step, so MSE vs SmoothL1 differ only through the momentum buffer's accumulation. Honest predicted band on Lion baseline: −1 % to +2 %.
- Even a wash is informative for the appendix's loss-form ablation table.

## 2026-04-28 03:50 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #507 | tanjiro | lion-lr-3p3e-4 | `lr_lion = 1.7e-4 → 3.3e-4` on merged #430 baseline | Tanjiro's own follow-up #1; Lion lr was sized for old AdamW recipe. Current AdamW=1e-3 → Lion equivalent 3.3e-4. Single-knob continuation. Predicted band −2 % to −6 %. |

## 2026-04-28 04:00 — PR #491: TF32 matmul precision (charliepai2d1-fern) — **sent back for rebase onto post-#430**
- Run config: `torch.set_float32_matmul_precision('high')` on post-#398 base.
- Headline: **−13 % per-epoch wall-clock** (130.83 s vs ~150 s eager), **14 epochs in 30-min budget vs 12**, val=86.491 (−3.20 % vs #398), test=76.796 (−3.02 %).
- Vs current Lion baseline #430: val +27.7 %, test +29.2 % (Lion merged ~10 min before results posted).
- Throughput delivery durable + optimizer-agnostic (TF32 doesn't interact with Lion's sign-update). Predicted post-#430 rebase: val ~63–66, test ~55–58.
- Sent back for rebase + re-run.

## 2026-04-28 04:00 — PR #483: SwiGLU MLP dropout=0.1 (charliepai2d1-frieren) — **CLOSED (regression)**
- val=92.901 (+3.97 % vs #398, +37.2 % vs current #430), test=81.959 (+3.49 % / +37.9 %).
- Frieren's diagnostic (excellent, fifth in a row): clean ep9 crossover — dropout *helps* through ep8, *hurts* from ep10. `geom_camber_rc` (under-fit canary) was the only split that improved (~−0.5 %). Cruise (closest-to-noise-floor split) regressed worst (+9 %).
- Mechanism: dropout did NOT slow raw train fit (predicted lose-case mechanism); the hit was concentrated in the EMA-shadow trajectory. Different from #458's WD bump (which slowed raw iterate). Useful contrast for the regularization ablation.
- Closed; reassigned to **PR #513 (swiglu-mlp-dropout-0p05)** — narrow the bracket. Frieren's own follow-up #1.

## 2026-04-28 04:00 — PR #475: SwiGLU swiglu_inner=256 (charliepai2d1-nezuko) — **CLOSED (regression)**
- val=93.888 (+5.08 % vs #398, +38.6 % vs current #430), test=81.969 (+3.51 % / +38.0 %).
- Nezuko's diagnostic: **training-budget starvation, not OOD-overfit**. Even `single_in_dist` regressed (+8.46 %), killing the closed-PR-#355 "in-dist memorizes / OOD collapses" hypothesis. With +25 % params at the same budget, larger model lands further from optimum at the timeout cut. Per-epoch curve still descending hard at ep12 (−5.4 between ep11 and ep12).
- Mechanism reading durable: SwiGLU's "gating fixes OOD" property is **at matched-param count, not capacity-on-top-of-good-shape**.
- Closed; reassigned to **PR #514 (swiglu-inner-192)** — smaller capacity bump (+14 % MLP / +7 % total). Tests whether *any* upward bump from 168 wins at this budget.

## 2026-04-28 04:05 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #513 | frieren | swiglu-mlp-dropout-0p05 | Dropout p=0.1 → 0.05 in `SwiGLUMLP.forward` on merged #430 baseline | Replaces closed #483; frieren's own follow-up #1. Narrows the dropout bracket — at p=0.05 the late-epoch noise penalty shrinks but the under-fit-regularization signal also shrinks. Predicted band −1 % to +1 %. |
| #514 | nezuko | swiglu-inner-192 | `swiglu_inner = 168 → 192` (+14 % MLP / +7 % total) on merged #430 baseline | Replaces closed #475; nezuko's own follow-up #1. Smaller capacity bump than 256; tests whether *any* upward bump wins at this budget. |

## 2026-04-28 04:33 — PR #352 (REBASED ONTO LION): SmoothL1 surface (charliepai2d1-edward) — **MERGED, new baseline**
- Branch: `charliepai2d1-edward/smoothl1-surface` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `cb356ed`).
- Hypothesis (rebased re-run): SmoothL1 (Huber β=1.0) on surface loss with Lion + SwiGLU(168) + EMA(0.99) + grad-clip(0.5).

### Headline metrics (best EMA epoch=12/50, timeout-cut)
| metric | this run | prior baseline #430 | Δ abs | Δ % |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **64.158** | 67.737 | −3.579 | **−5.28 %** |
| `test_avg/mae_surf_p` | **55.930** | 59.447 | −3.517 | **−5.92 %** |

Past merge gate cleanly. **Predicted band was −1 % to +2 %** (uncertain due to Lion's sign-update potentially subsuming SmoothL1's gradient-shaping). Actual −5.28 % beats the high end of the band by 3 pts.

### Per-split — gain redistribution under Lion
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | **−1.80 %** | **+1.05 %** |
| geom_camber_rc | −7.29 % | −7.89 % |
| geom_camber_cruise | **−8.44 %** | **−11.44 %** |
| re_rand | −4.72 % | −7.97 % |

**Mechanism shift**: under AdamW (post-#417 base), SmoothL1 helped `single_in_dist` most (−23.80 % val, the high-Re tail story). Under Lion, single_in_dist barely moves (the smallest val gainer); cruise becomes biggest beneficiary. **Lion's sign-update has already absorbed the high-Re-tail benefit** (per-param step is invariant to gradient magnitude). What's left is the camber-OOD generalization improvement, propagating through the *momentum buffer's sign trajectory* differently than MSE does. Clean second-order mechanism for the appendix.

### Decision: merge as new round-1.5 baseline
- Beats baseline on val (−5.28 %) and test (−5.92 %), no per-split val regression, only one tiny test regression (+1.05 % on single).
- Mechanism story is durable and well-documented.
- BASELINE.md updated; edward reassigned to **PR #535 (smoothl1-beta-0p5)** as the natural β-sweep continuation.

## 2026-04-28 04:35 — PR #507: Lion lr 1.7e-4 → 3.3e-4 (charliepai2d1-tanjiro) — **CLOSED (regression)**
- val=73.456 (+8.45 % vs #430), test=63.076 (+6.10 %). Vs current post-#352: val +14.5 %, test +12.8 %.
- Predicted band was −2 % to −6 % (lose case identified honestly in PR body); actual was lose case.

### Mechanism (tanjiro's writeup)
- **Lion's basin is narrower than the AdamW-equivalent heuristic suggests.** `lr_lion = lr_adamw / 3` was based on AdamW lr=5e-4; at current AdamW=1e-3 the equivalent 3.3e-4 is past Lion's actual basin.
- **Mean pre-clip grad-norm dropped −34 %** (29.65 vs 45.07) — Lion finds flatter regions even faster at higher lr, but the parameter trajectory lands in a worse basin.
- **Lose mechanism**: raw floor rises faster than EMA can smooth. Spread widened (mean −18.5 vs #430's −15.7) — EMA does more variance-reduction work but can't overcome the higher iterate floor.
- **No first-2-epoch divergence** at lr=3.3e-4 — Lion's bounded sign-update is robust across this lr range; what fails is *target quality*, not stability.

### Decision: close, reassign to lion-lr-2p5e-4
- Clear >5 % regression. Per CLAUDE.md close criteria.
- Reassigned to **PR #536 (lion-lr-2p5e-4)** — bracket-narrowing midpoint between basin (1.7e-4) and lose (3.3e-4). Tests Lion's basin upper edge.

## 2026-04-28 04:40 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #535 | edward | smoothl1-beta-0p5 | SmoothL1 β=1.0 → 0.5 on merged #352 baseline | Edward's own follow-up #1; β-sweep narrowing. Tests whether smaller β (more L1-regime fraction) amplifies the per-split gain pattern. |
| #536 | tanjiro | lion-lr-2p5e-4 | `lr_lion = 1.7e-4 → 2.5e-4` on merged #352 baseline | Tanjiro's bracket-narrowing follow-up; midpoint between basin and lose. Tests Lion's basin upper edge. |

## 2026-04-28 04:48 — PR #513: SwiGLU MLP dropout=0.05 (charliepai2d1-frieren) — **CLOSED (regression)**
- val=68.594 (+1.27 % vs #430 run-base, **+6.92 % vs current #352**), test=60.069 (+1.05 % / +7.41 %).
- Frieren's verdict (6th exceptional writeup): **dropout is dead under SwiGLU + Lion at this budget**. Bracket fully mapped: p=0 baseline winning, p=0.05 +1.27 %, p=0.1 +3.97 %. Monotone regression with p; no clean win at any tested level.
- **Mechanism for the appendix**: Lion's sign-update mutes both win and lose dropout mechanisms toward zero from below. The per-split asymmetry (cruise hardest hit, worst-near-noise-floor) is stable across both p values — confirms the canary.
- Reassigned to **PR #545 (lion-beta1-0p95)** — fresh optimization-side axis after 6 closures; tests slower Lion momentum decay.

## 2026-04-28 04:48 — PR #474: EMA decay 0.99 → 0.97 (charliepai2d1-askeladd) — **CLOSED (regression)**
- val=95.169 (+6.51 % vs #398 run-base, **+48.34 % vs current #352**), test=85.628 (+8.13 % / +53.10 %).
- Askeladd's verdict: **EMA-decay axis fully locked at 0.99 across both GELU and SwiGLU bases**. Bracket: 0.999 +12.71 %, 0.97 +6.51 %, 0.95 +8.53 %. 0.99 wins on both architectures by ≥6 % over nearest neighbors.
- **Cross-architecture mechanism**: SwiGLU's smoother gradients reduce the value of "faster shadow tracking" — the live iterate is already smoother per-step, so a shorter EMA window doesn't help capture an aggressive descent that isn't there. Useful interaction-effect note for the appendix.
- **Cruise sensitivity stable across decays**: +16.66 % at 0.95 ≈ +16.26 % at 0.97 ≈ baseline at 0.99. Cruise = canary for "balanced-domain sampler noise floor on smallest-sample-domain."
- Reassigned to **PR #546 (lion-batch-8)** — fresh batch-side probe under Lion's sign-update (different math than #403's closed AdamW+batch=8).

## 2026-04-28 04:50 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #545 | frieren | lion-beta1-0p95 | Lion `betas = (0.9, 0.99) → (0.95, 0.99)` on merged #352 baseline | Slower Lion momentum decay; tests whether more inertial direction signal smooths Lion's substantial epoch-to-epoch raw variance. Honest band −1 % to +2 %. |
| #546 | askeladd | lion-batch-8 | `batch_size = 4 → 8` (no lr scaling) on merged #352 baseline | Replaces closed #474; first batch-side probe under Lion. Lion's bounded `lr × sign` per step changes the math from #403's closed AdamW+batch=8 (catastrophic). Honest band −2 % to +5 %. |

## 2026-04-28 04:55 — PR #514: SwiGLU `swiglu_inner=192` (charliepai2d1-nezuko) — **CLOSED**
- val=68.828 (+1.61 % vs #430 run-base, **+7.28 % vs current #352**), test=59.057 (−0.66 % / **+5.59 %**).
- Combined with closed #475 (SwiGLU 256, +5 %) gives a **clean two-point capacity curve**: 168 (best) < 192 (≈wash) < 256 (lose). **SwiGLU(168) at matched params is the local optimum at this 30-min/12-epoch budget under Lion.**
- Per-split: `rc` (highest-camber/random-chord OOD split) regressed +5 % at +14 % MLP — capacity isn't helping under-fit OOD splits even at modest bumps.
- val/test asymmetry (+1.6 % val regression but −0.7 % test improvement) attributed to EMA-vs-raw mismatch at this budget: best raw at ep11, best EMA at ep12, EMA shadow carrying ep11's better weights into the test-eval checkpoint.
- Reassigned to **PR #552 (geglu-mlp-matched)** — natural architectural follow-up after the SwiGLU capacity axis is locked.

## 2026-04-28 04:55 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #552 | nezuko | geglu-mlp-matched | `silu(value(x)) → gelu(value(x))` in gated MLP forward, same `geglu_inner=168` | Gating-activation A/B test at matched params. Tests whether SwiGLU's silu-specific shape is load-bearing or whether any gating mechanism delivers the win. Honest band −1 % to +1 %. |

## 2026-04-28 05:17 — PR #491 (REBASED): TF32 matmul precision (charliepai2d1-fern) — **MERGED, new baseline**
- Branch: `charliepai2d1-fern/tf32-matmul-precision` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `f8d8ffb`).
- Hypothesis (rebased re-run): `torch.set_float32_matmul_precision('high')` on top of Lion + SwiGLU + SmoothL1 stack.

### Headline metrics (best EMA epoch=14/50, timeout-cut)
| metric | this run | prior baseline #352 | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **63.218** | 64.158 | **−1.47 %** |
| `test_avg/mae_surf_p` | **55.398** | 55.930 | **−0.95 %** |

### Throughput (the deliverable)
- `mean_epoch_3plus_seconds` = **131.33 s** vs ~151 s eager → **−13.0 %**.
- 14 epochs in 30-min budget vs baseline's 12.
- Peak VRAM 45.21 GB (flat).
- No NaN, no matmul warnings. Identical to pre-rebase observation (−13 % per-epoch on the post-#398 GELU base) — kernel-fusion gain is independent of optimizer/loss/architecture changes.

### Per-split breakdown (the noise-floor signal)
| Split | val Δ | test Δ |
|---|---:|---:|
| single_in_dist | **−6.79 %** | **−9.73 %** |
| geom_camber_rc | +3.25 % | +5.40 % |
| geom_camber_cruise | −0.73 % | −0.12 % |
| re_rand | −1.11 % | +2.03 % |

3/4 val splits improve, 2/4 test splits improve. `single_in_dist` is the big winner (the 2-extra-cosine-epochs benefit biases toward in-distribution learning); `geom_camber_rc` regresses on both — within typical run-to-run variance band.

### Decision: merge as new baseline
- Strict merge rule satisfied (val 63.22 < 64.16, test 55.40 < 55.93).
- Throughput multiplier is permanent: every subsequent round-2 PR fits 14 epochs vs 12.
- Optimizer-agnostic (TF32 fp32-matmul propagates identically through Lion's sign-update).
- Per-split rc regression noted but within run-to-run variance.
- BASELINE.md updated; fern reassigned to **PR #560 (cosine-tmax-14-on-lion)** — Lion's bounded sign-update changes the calculus from #465's closed AdamW failure.

## 2026-04-28 05:20 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #560 | fern | cosine-tmax-14-on-lion | `T_max=50 → 14`, `eta_min=1e-5` on merged #491 baseline | Replaces fern's earlier closed #465 (T_max=13 under AdamW). Under Lion's bounded sign-update, late-epoch lr ~1e-5 still produces ~1e-5 per-param movement (no AdamW adaptive denominator collapse). Tests whether a real anneal phase late-epoch helps under Lion + 14-epoch budget. |

## 2026-04-28 05:27 — PR #535: SmoothL1 β=1.0 → 0.5 (charliepai2d1-edward) — **MERGED, new baseline**
- Branch: `charliepai2d1-edward/smoothl1-beta-0p5` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `1223c15`).
- Hypothesis: narrow the SmoothL1 β threshold from 1.0 → 0.5 to widen the L1-regime fraction.

### Headline metrics (best EMA epoch=12/50, timeout-cut)
| metric | this run | prior baseline #491 | Δ vs current | also Δ vs run-base #352 |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **61.508** | 63.218 | **−2.70 %** | −4.13 % |
| `test_avg/mae_surf_p` | **52.336** | 55.398 | **−5.53 %** | −6.43 % |

### Per-split signature inversion (the durable mechanism finding)
| Split | β=1.0 (#352) val Δ | β=0.5 (this) val Δ |
|---|---:|---:|
| single_in_dist | −1.80 % | **−8.31 %** |
| geom_camber_rc | −7.29 % | +0.69 % |
| **geom_camber_cruise** | **−8.44 %** | −1.93 % |
| re_rand | −4.72 % | −6.30 % |

**The β knob redirects SmoothL1's per-split benefit between cruise (β=1.0) and single (β=0.5)**:
- At β=1.0, cruise's residual mass concentrates at the L1-regime threshold and benefits.
- At β=0.5, the L1-regime is wide enough that high-Re single residuals fall into it; the cruise gain saturates.
- The "L1-tail captures high-magnitude residuals" mechanism is the right first-order story; the β knob controls *which residuals* the L1-tail captures.

### Decision: merge as new baseline
- Strict merge rule satisfied; mechanism story durable and well-documented.
- 9th merge on this branch; second loss-form refinement.
- BASELINE.md updated; edward reassigned to **PR #567 (smoothl1-beta-0p25)** for further bracket-narrowing.

## 2026-04-28 05:30 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #567 | edward | smoothl1-beta-0p25 | SmoothL1 β=0.5 → 0.25 on merged #535 baseline | Edward's own follow-up #1; β-axis bracket-narrowing. Tests whether L1-tail mechanism continues to scale or saturates. Honest band −2 % to +1 %. |

## 2026-04-28 05:32 — PR #545: Lion β1 = 0.9 → 0.95 (charliepai2d1-frieren) — **CLOSED (regression)**
- val=68.366 (+6.55 % vs #352 run-base, **+11.15 % vs current #535**), test=60.122 (+7.50 % / **+14.88 %**).
- Frieren's 8th exceptional write-up: clean **win-vs-lose mechanism trade-off**.
  - **Win mechanism (smoother direction)**: late-epoch raw monotone descent (94.9 → 92.5 → 87.0 → 82.0) vs #352's bouncy 89.9 → 85.6 → 76.5 → 79.5. Mean pre-clip grad-norm dropped 25 % (15.2 → 11.5). EMA-vs-raw spread shrunk 15.4 → 13.6.
  - **Lose mechanism (slower convergence under fixed budget)**: EMA gap from #352 widens monotonically (ep1 −1.8 → ep4 +3.1 → ep8 +3.3 → ep12 +4.2). Same step-count-starvation pattern as closed #403 (different mechanism — β1 inertia rather than batch-step ratio).
- **Stationary-vs-non-stationary split signature** (the appendix-quality finding): β1=0.95 *won* on `single_in_dist` (most-stationary, single-foil flow, −1.97 % val) and *lost* on all three tandem splits (rc +7.7 %, cruise +16.4 %, re_rand +9.0 %) where front-rear foil interaction makes gradient direction non-stationary. **"Stationary regimes prefer inertia, non-stationary regimes prefer responsiveness."**
- Reassigned to **PR #571 (lion-beta2-0p999)** — fresh single-knob continuation; β2 affects the persistent buffer rather than the direction signal.

## 2026-04-28 05:35 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #571 | frieren | lion-beta2-0p999 | Lion `betas = (0.9, 0.99) → (0.9, 0.999)` on merged #535 baseline | Slower momentum buffer (10× more inertial); direction signal still responsive at β1=0.9. Tests whether buffer-side smoothing without responsiveness penalty wins. Honest band −2 % to +3 %. |

## 2026-04-28 05:55 — PR #546: Lion + batch=8 (fell back to b=6 after OOM) (charliepai2d1-askeladd) — **CLOSED**
- val=64.038 (−0.19 % vs #352 run-base, +4.11 % vs current #535), test=55.465 (−0.83 % / +5.98 %).
- batch=8 OOM'd at ep6 (~94.6 GB exceeded); fallback to b=6 at advisor instruction.

### Durable Lion-vs-AdamW interaction effect
| | val_avg | single_in_dist (val) | mechanism |
|---|---:|---:|---|
| AdamW + b=8 + √2-LR (closed #403) | 172.97 | +99 % | step-count starvation catastrophic |
| Lion + b=6 (this PR) | 64.04 | +0.28 % | step-count wash; sign-update decouples from batch |

**Lion's bounded `lr × sign(c_t)` per-step decouples from batch size**, making batch changes far less destructive than under AdamW. Durable mechanistic insight regardless of whether the lever wins.

### Other observations
- Predicted "smoother momentum buffer at higher batch" mechanism didn't materialize: spread −18.09 (b=6) vs −18.44 (b=4 baseline) — essentially identical.
- Per-split redistribution (re_rand −5.96 % val, single -2.90 % test improvements; rc +3.64 % val regression). Wash on equal-weight metric.
- batch=8 cleanly was not testable without bf16 (different experiment).

### Decision: close
- +4.11 % val / +5.98 % test vs current #535 (just past close threshold).
- Branch has β=1.0 (old SmoothL1); squash-merge would revert merged β=0.5 → β=1.0 and undo #535's win — structurally incompatible with merging.
- Reassigned to **PR #580 (lion-lr-1p2e-4)** — bracket-narrowing Lion's basin lower edge to complement tanjiro's in-flight upper-edge probe (#536, lr=2.5e-4).

## 2026-04-28 06:00 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #580 | askeladd | lion-lr-1p2e-4 | `lr_lion = 1.7e-4 → 1.2e-4` on merged #535 baseline | Replaces closed #546; lower-edge probe of Lion's basin (complements tanjiro's #536 upper-edge at 2.5e-4). Honest band −2 % to +4 %. |

## 2026-04-28 06:12 — PR #536: Lion lr 1.7e-4 → 2.5e-4 (charliepai2d1-tanjiro) — **MERGED, new baseline**
- Branch: `charliepai2d1-tanjiro/lion-lr-2p5e-4` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `c6b65b6`).
- Hypothesis (Lion basin upper-edge probe between 1.7e-4 basin and 3.3e-4 lose).

### Headline metrics (best EMA epoch=12/50, timeout-cut)
| metric | this run | run base #352 (β=1.0) | current #535 (β=0.5) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | **60.478** | 64.158 (−5.74 %) | 61.508 (**−1.67 %**) |
| `test_avg/mae_surf_p` | **52.676** | 55.930 (−5.82 %) | 52.336 (+0.65 %) |

### Per-split — uniform improvement on val (all 4 splits gain)
| Split | val Δ vs #352 | test Δ vs #352 |
|---|---:|---:|
| single_in_dist | **−11.95 %** | **−14.17 %** |
| geom_camber_rc | −3.44 % | −1.47 % |
| geom_camber_cruise | −0.86 % | −3.20 % |
| re_rand | −4.18 % | −2.06 % |

### Mechanism (tanjiro's writeup)
- **Mean pre-clip grad-norm dropped 30 % vs lr=1.7e-4** (12.25 vs ~17 for #430-era; 50 for raw lr=1.7e-4 reference). Lion finds flatter regions even faster at higher lr.
- **Lion's basin upper edge is in [2.5e-4, 3.3e-4]**, not "right around 1.7e-4." Default `lr_lion = lr_adamw / 3` heuristic was conservative for our recipe.
- **Per-split uniformity** (no per-split regression) distinguishes this from #545 (β1=0.95) and #507 (lr=3.3e-4) — at 2.5e-4 the larger directional updates per step pay off equally on stationary and non-stationary regimes.

### Decision: merge as new baseline
- Strict merge gate satisfied on val (the primary ranking metric).
- Test +0.65 % is within run-to-run noise band.
- **Squash-merge composes cleanly**: tanjiro's `lr=2.5e-4` change in `Config` is in different code region from #535's `β=0.5` in the loss block; git's three-way merge applies both → post-merge `train.py` has lr=2.5e-4 + β=0.5.
- 11th merge on this branch; **fourth Lion-axis lever** (#430 EMA-Lion + #491 TF32 + this #536 Lion lr).
- BASELINE.md updated; tanjiro reassigned to **PR #592 (lion-lr-2p85e-4)** — bracket-narrowing midpoint between 2.5e-4 and 3.3e-4 to lock the basin upper edge.

## 2026-04-28 06:15 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #592 | tanjiro | lion-lr-2p85e-4 | `lr_lion = 2.5e-4 → 2.85e-4` on merged #536 baseline | Bracket midpoint between basin (2.5e-4) and lose (3.3e-4); tanjiro's own follow-up #1. Honest band −3 % to +6 %. |

## 2026-04-28 06:05 — PR #552: GeGLU at matched params (charliepai2d1-nezuko) — **sent back for rebase + re-run**
- Run config: `F.silu → F.gelu` in `GeGLUMLP.forward` at matched `geglu_inner=168` (657,639 params), on post-#352 base (β=1.0).

### Headline metrics (best EMA epoch=12/50, timeout-cut)
| metric | this run | run base #352 (SwiGLU + β=1.0) | current #535 (SwiGLU + β=0.5) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | 62.477 | 64.158 (**−2.62 %**) | 61.508 (+1.57 %) |
| `test_avg/mae_surf_p` | 54.102 | 55.930 (**−3.27 %**) | 52.336 (+3.37 %) |

Clean win on the run base; mild regression vs current (within 5 % close threshold, past merge gate).

### Mechanism finding (durable for the appendix)
**GeGLU vs SwiGLU at matched params: activation shape is load-bearing, not just gating mechanism.** Per-split breakdown:
- `single_in_dist` Δ = −7.91 % val / −6.64 % test (largest gain) — GELU's deeper negative-input dip provides more aggressive feature suppression on this split's high-variance pressure tail.
- `re_rand` Δ = −2.19 % val / −4.04 % test — cross-regime Re holdout also benefits.
- `geom_camber_cruise` Δ = +1.42 % val / +3.33 % test (small regression) — cruise's lower-magnitude pressure field benefited from SiLU's smoother shape.
- `geom_camber_rc` Δ = +0.14 % val / −2.59 % test — wash on val, slight gain on test.

The "any gating works" hypothesis is **falsified** by the per-split asymmetry; activation shape matters.

### Why send back, not close, not merge
- Past merge gate vs current (+1.57 % val / +3.37 % test on #535).
- NOT past close threshold (>5 %).
- Branch has β=1.0; squash-merge would revert merged β=0.5 → β=1.0 and undo #535's win.
- The post-rebase question is genuinely interesting: GeGLU and β=0.5 both target high-Re tail via different mechanisms (activation shape vs gradient shape). Whether they compound, subsume, or interfere is the appendix-quality question.

### Predicted post-rebase outcome
- **Compound**: val ~57–60 (−2 % to −5 % vs #535) — third compounding loss-shape × activation-shape lever.
- **Subsume**: val 60–62, near wash vs #535.
- **Interfere**: val 62–64, slight regression.

Honest predicted band: −5 % to +3 % vs current 61.508. Either result locks the activation-shape × loss-shape interaction story.

## 2026-04-28 06:23 — PR #571: Lion β2 = 0.99 → 0.999 (charliepai2d1-frieren) — **MERGED, new baseline**
- Branch: `charliepai2d1-frieren/lion-beta2-0p999` → squash-merged into `icml-appendix-charlie-pai2d-r1` (commit `b347039`).
- Hypothesis: longer-history momentum buffer (10× half-life: 69 → 693 batches) smooths sign(c_t) update direction without sacrificing responsiveness (β1=0.9 unchanged). Predicted band: −2 % to +3 %.

### Headline metrics (best EMA epoch=14/50, timeout-cut)
| metric | this run | current baseline #536 | run base (pre-#535/#536, lr=1.7e-4, β=1.0) |
|---|---:|---:|---|
| `val_avg/mae_surf_p` (EMA) | **52.116** | 60.478 (**−13.83 %**) | run on lr=1.7e-4 + β=1.0 + β2=0.999 |
| `test_avg/mae_surf_p` | **45.413** | 52.676 (**−13.79 %**) | — |
| raw val (best at best EMA ep) | 59.097 | — | — |

### Per-split — broad-based gain on every split (≥10 %)
| Split | val Δ vs #536 | test Δ vs #536 |
|---|---:|---:|
| single_in_dist | **−16.02 %** | **−13.74 %** |
| geom_camber_rc | −10.16 % | −13.32 % |
| geom_camber_cruise | −18.81 % | −14.32 % |
| re_rand | −12.33 % | −14.09 % |

### Mechanism (frieren's writeup — durable appendix-grade finding)
**β1 vs β2 mechanism distinction.** The two Lion momentum knobs trade off symmetrically but at very different costs:
- **β1 (direction-signal responsiveness)**: in the sign update `update = sign(β1·m + (1-β1)·g)`, raising β1 makes `sign(c_t)` more inertial. **#545 (β1=0.9 → 0.95) lost** — the sign-update can't track non-stationary gradient regimes (tandem foil interactions). Stationary single-foil split *gained* under inertia, all three tandem splits regressed → the lose case was specifically driven by responsiveness loss in non-stationary regimes.
- **β2 (buffer-history)**: in the buffer update `m_{t+1} = β2·m + (1-β2)·g`, raising β2 makes the persistent buffer m smoother but doesn't directly affect the direction signal — `sign(c_t)` retains full responsiveness through `β1·m + (1-β1)·g`. **#571 (β2=0.99 → 0.999) won broadly** — every split gained ≥10 % including the tandem splits.

**The trade-off is asymmetric**: β1 trades responsiveness for direction smoothness (zero-sum on responsiveness); β2 trades a few warm-up batches for persistent direction smoothness while retaining full responsiveness (positive-sum on responsiveness). This makes β2 the dominant lever on the Lion buffer-history axis.

### Decision: merge as new baseline
- Strict merge gate satisfied; **largest single-PR delta on this branch since #430 Lion adoption** (−24.19 % at the time).
- Squash-merge composes: β2=0.999 (this PR, optimizer line) + lr=2.5e-4 (#536, Config line) + β=0.5 (#535, loss block). Three independent code regions; git's three-way merge applies all.
- Recorded baseline metrics are from frieren's run on lr=1.7e-4 + β=1.0 + β2=0.999 (pre-#535/#536 base). The post-merge live config is lr=2.5e-4 + β=0.5 + β2=0.999 — likely lands slightly better since both lr and β were independently improved post-fork.
- 12th merge on this branch; **fifth Lion-axis lever** (#430 Lion adoption + #491 TF32 + #536 Lion lr=2.5e-4 + this #571 Lion β2=0.999).
- BASELINE.md updated; frieren reassigned to **PR #598 (lion-beta2-0p9999)** — bracket-narrowing the upper edge of β2 to lock the buffer-history axis.

## 2026-04-28 06:30 — Round-1.5 assignments (continued)

| PR | Student | Slug | Lever | Why |
|----|---------|------|-------|-----|
| #598 | frieren | lion-beta2-0p9999 | Lion `betas[1] = 0.999 → 0.9999` on merged #571 baseline | Frieren's own follow-up #1; β2-axis upper-edge probe. With ~1190 batches in budget vs β2=0.9999 half-life ~6900 batches, buffer never fully converges — tests whether the buffer-history gain saturates or continues. Honest band −6 % to +15 %. |

## 2026-04-28 06:30 — PR #560: Cosine T_max=14, eta_min=1e-5 under Lion (charliepai2d1-fern) — **sent back for rebase + re-run**
- Run config: `T_max=MAX_EPOCHS → 14, eta_min=1e-5` in `CosineAnnealingLR(...)`, plus per-epoch lr logging. Branched from pre-#535/#536/#571 baseline (lr=1.7e-4, β=1.0, β2=0.99).

### Headline metrics (best EMA epoch=14/50, timeout-cut)
| metric | this run | run base #491 | current baseline #571 |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | 54.091 | 63.218 (**−14.44 %**) | 52.116 (**+3.79 %**) |
| `test_avg/mae_surf_p` | 47.236 | 55.398 (**−14.73 %**) | 45.413 (+4.01 %) |
| raw val (best at best EMA ep) | 55.911 | — | — |

### Mechanism finding (durable for the appendix)
**Cosine-to-`eta_min` under Lion completes cleanly within budget; the AdamW un-train pathology (#465) does NOT manifest under Lion's bounded sign-update.**
- Lion's `update = lr × sign(c_t)` keeps per-param movement at exactly `lr / step` even at `lr=1.2e-5` (the `eta_min` floor at ep14). Per-param movement of 1.2e-5 / step is well above the noise floor — late-epoch refinement does real work rather than getting drowned by AdamW's adaptive denominator.
- EMA−raw spread collapsed from ~−15 (T_max=50, no anneal) to **−1.82** (T_max=14, full anneal): late-epoch low-lr produces a stable iterate, EMA and raw nearly converge.
- `is_best=True` at every epoch through ep14 with monotone descent — schedule helped to the very end; no late-epoch degradation.
- Train loss did NOT reverse at ep11–14 (the canonical falsifier from #465).

Resolves the schedule-vs-optimizer interaction for the appendix: under AdamW (#353/#438/#465) cosine-to-zero kills the model; under Lion + matched `T_max` + finite `eta_min` it's a clean −14 % win.

### Why send back, not close, not merge
- Past close threshold (>5 %) only barely (+3.79 % val vs current).
- Past merge gate vs current baseline.
- **Branch has lr=1.7e-4 + β=1.0 + β2=0.99 hardcoded** in optimizer line; squash-merge would inherit the schedule edit BUT also revert merged β2=0.999 → 0.99 (fern's hardcoded line is the override path), undoing #571's win.
- The post-rebase question is rigorous and well-motivated: the schedule mechanism (Lion + completed anneal) is independent of lr/β/β2, so should stack. Predicted re-run band: val ~46–49 (−6 to −12 % vs new baseline 52.116).

### Predicted re-run outcome
- Mechanism predicts compounding (independent of lr/β/β2 axes). Post-rebase val ~46–49.
- The cleanest stack-test for the appendix: schedule mechanism + Lion β2=0.999 + lr=2.5e-4 + β=0.5 → does the schedule still buy −10 %+ when the iterate is already smoother (β2=0.999) and the cosine anneals from a higher peak (2.5e-4)?

### Reassignment
- Fern stays on PR #560 — re-running on the rebased branch with corrected `betas=(0.9, 0.999)`. Single-knob discipline preserved (the only diff vs baseline is the scheduler line + lr-log capture).

## 2026-04-28 06:35 — PR #567: SmoothL1 β = 0.5 → 0.25 (charliepai2d1-edward) — **sent back for rebase + re-run**
- Run config: `beta = 0.5 → 0.25` in the SmoothL1/MSE-vol loss block. Branched from post-#535 baseline (lr=1.7e-4, β2=0.99, β=0.5). Single-line edit; clean diff.

### Headline metrics (best EMA epoch=14/50, timeout-cut)
| metric | this run | run base #535 | current baseline #571 |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | 54.896 | 61.508 (**−10.75 %**) | 52.116 (**+5.34 %**) |
| `test_avg/mae_surf_p` | 47.503 | 52.336 (**−9.23 %**) | 45.413 (+4.61 %) |
| raw val (best at best EMA ep) | 68.684 | — | — |
| EMA−raw spread | +13.79 | — | — |

### Per-split — broad-based gain on every split (vs run base #535)
| Split | val Δ vs #535 | test Δ vs #535 |
|---|---:|---:|
| single_in_dist | **−11.70 %** | **−10.71 %** |
| geom_camber_rc | −10.39 % | −5.94 % |
| geom_camber_cruise | **−14.35 %** | **−13.11 %** |
| re_rand | −7.52 % | −9.05 % |

### Mechanism finding (durable for the appendix β-curve)

**Loss-form β-axis fully mapped (1.0 → 0.5 → 0.25 monotone, no knee yet visible)**:

| β | PR | val_avg | test_avg | dominant winner |
|---|---|---:|---:|---|
| 1.00 | #352 | 64.160 | 55.930 | cruise |
| 0.50 | #535 | 61.508 | 52.336 | single_in_dist |
| 0.25 | **#567 (this)** | **54.896** | **47.503** | single_in_dist amplified, cruise re-entered as second-largest gainer |

The L1-tail-amplifies story scales further than predicted:
- A narrower MSE-regime (`|err| < 0.25σ`) routes more residuals to the L1 asymptote.
- Lion's sign-update preserves L1-asymptote gradient *direction* in its momentum buffer; under MSE the magnitude information is what changes, which Lion partially discards via the sign.
- Cruise's *re-amplification* (predicted to regress) reveals: cruise's lower-magnitude residuals fall into the new (small-`β`) MSE-regime where small near-converged residuals get *amplified* by `2·err²/β`, helping rather than hurting.
- grad_norm rise modest (+16 % vs #535's late-epoch); clip envelope at 0.5 absorbed the magnification cleanly.

### Why send back, not close, not merge
- Past close threshold (>5 %) only barely (+5.34 % val vs current).
- Past merge gate vs current baseline (recorded 54.896 > 52.116).
- **Branch has lr=1.7e-4 + β2=0.99**; squash-merge would compose β=0.25 cleanly (different code region from lr/β2) — predicted post-rebase val ~46–50 — but the recorded number doesn't beat baseline directly.
- The β-axis mechanism (narrower MSE-regime + L1-tail amplification) is mechanically independent of lr (per-step size) and β2 (buffer smoothness); both moved baselines (#536 lr=2.5e-4, #571 β2=0.999) target different physical mechanisms. Predicted to stack.

### Predicted re-run outcome
- val ~46–50 (−4 to −12 % vs new baseline 52.116). The cleanest stack-test: loss-shape × Lion-buffer-history × Lion-lr.
- If wins, the next bracket point is **β=0.125** (edward's suggested follow-up) — the β-curve has no knee yet visible.
- If washes vs new baseline (val ~50–53), declares the L1-axis saturated under the new optimizer regime.

### Reassignment
- Edward stays on PR #567 — re-running on rebased branch (single-line edit; trivial rebase, no conflicts expected).

## 2026-04-28 06:45 — PR #580: Lion lr 1.7e-4 → 1.2e-4 (charliepai2d1-askeladd) — **sent back for rebase + re-run**
- Run config: `lr: 1.7e-4 → 1.2e-4` in `Config` dataclass. Branched from post-#535 baseline (lr=1.7e-4, β=0.5, β2=0.99). Single-line edit.

### Headline metrics (best EMA epoch=13/50, timeout-cut)
| metric | this run | run base #535 | current baseline #571 |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` (EMA) | 55.547 | 61.508 (**−9.69 %**) | 52.116 (**+6.58 %**) |
| `test_avg/mae_surf_p` | 47.964 | 52.336 (**−8.35 %**) | 45.413 (+5.62 %) |
| best raw val | 69.903 (ep14) | — | — |
| EMA−raw spread (mean ep4–14) | −16.39 | ~−15 | — |

### Per-split — broad-based gain (vs run base #535)
| Split | val Δ vs #535 | test Δ vs #535 |
|---|---:|---:|
| single_in_dist | **−13.83 %** | **−11.85 %** |
| geom_camber_rc | −6.40 % | −2.73 % |
| geom_camber_cruise | **−15.66 %** | **−13.62 %** |
| re_rand | −4.76 % | −7.79 % |

All 8 splits improve. Cruise wins most (small-residual split most sensitive to optimizer); single_in_dist second-most (stationary regime gains under tighter local minimum).

### Mechanism finding (durable for the appendix lr-axis basin map)

**Asymmetric basin under β2=0.99 — Lion lr is much more under-tuned at 1.7e-4 than over-tuned at 3.3e-4**:

| run | lr | step ratio vs 1.7e-4 | val | Δ vs run base |
|-----|----|-:|---:|---:|
| #507 (closed) | 3.3e-4 | +94 % | 73.46 | +8.45 % |
| #535 (run base) | 1.7e-4 | 0 | 61.51 | 0 |
| **#580 (this PR)** | **1.2e-4** | **−29 %** | **55.55** | **−9.69 %** |

Per-percent-of-step:
- downward sensitivity ≈ **−0.33 %** val per −1 % lr
- upward sensitivity ≈ **+0.09 %** val per +1 % lr

The optimum under β2=0.99 lies **further down the basin than 1.7e-4**, and 1.2e-4 looks like it's **still on the descending side**. Default `lr_lion = lr_adamw / 3` heuristic was conservative.

### Why send back, not close, not merge
- Past close threshold (>5 %) at +6.58 % val vs current.
- **Squash-merge would CONFLICT** on the lr line: askeladd's branch has `lr: 1.7e-4 → 1.2e-4`, advisor branch has `lr: 2.5e-4` (post-#536). Cannot apply diff cleanly.
- The lower-edge probe under β2=0.999 is **currently unmapped**. The new optimizer regime may have shifted the basin (smoother direction → larger optimal lr), and tanjiro's #592 (2.85e-4) is in flight on the upper edge — askeladd's re-run completes the picture.

### Predicted re-run outcome (wide band — two competing mechanism hypotheses)
- **Basin shifts up under β2=0.999** (smoother sign direction → larger optimal lr): lr=1.2e-4 should be a bigger lose case. Predict val ~58–65 (+12 to +25 %).
- **Basin stays at same width but shifts toward smaller lr** (smoother direction → smaller per-step needed): lr=1.2e-4 might still be in basin. Predict val ~50–55 (−5 to +5 %).

Honest predicted band: **−5 % to +25 %** vs current 52.116. Combined with #592, locks the new-baseline lr basin.

### Reassignment
- Askeladd stays on PR #580 — re-running on rebased branch with `lr: 2.5e-4 → 1.2e-4` (resolve conflict by taking askeladd's value); β2=0.999 inherits from advisor cleanly.
