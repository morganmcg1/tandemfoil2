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
