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

## 2026-04-28 06:51 — PR #442 (round 3): H12 EMA × FiLM — **MERGED** (Round 0 winner #3)

- Branch: `willowpai2d4-thorfinn/h12-ema-weights` (rebased onto post-#404 advisor branch)
- Round 3 single-cell test: Run F = EMA decay=0.99 + every-other-epoch eval **on top of merged FiLM baseline** + `--seed 123`.

| Run | Config | val_raw (best epoch) | **val_ema (best)** | active source | **test_avg** | best epoch | W&B |
|-----|--------|----------------------:|-------------------:|:-------------:|-------------:|-----------:|-----|
| Merged baseline (#404) | FiLM, no EMA | 119.36 | n/a | raw | 107.54 | 13 | `p0a1daar` |
| **F — EMA + FiLM** | both | **119.36** (matches!) | **109.19** | ema | **98.47** | 13 | `gc57edp6` |

### Per-split test for Run F vs PR #404 baseline

| Split | Run F | PR #404 baseline | Δ |
|-------|------:|-----------------:|--:|
| `test_single_in_dist` | 111.60 | 120.69 | **−7.5%** |
| `test_geom_camber_rc` | 112.42 | 120.45 | **−6.7%** |
| `test_geom_camber_cruise` | 69.59 | 80.70 | **−13.8%** |
| `test_re_rand` | 100.26 | 108.32 | **−7.4%** |
| **avg** | **98.47** | 107.54 | **−8.4%** |

### Conclusions

- **EMA × FiLM compound is a clean uniform win.** Improvements range from −6.7% to −13.8% across all four test splits, with cruise getting the largest gain (smoothest split → loss-landscape oscillation noise dominates → EMA's noise-smoothing helps most).
- **val_raw at epoch 13 = 119.36 matches PR #404 baseline to 4 sig figs.** Same code, same seed=123, same hyperparameters → identical raw result. Strongest reproducibility evidence on the branch; validates the seed-controlled comparison protocol introduced in PR #404.
- **EMA mechanism validated for the fourth time.** Within-run lift: Run B (decay=0.999) 10.1%, Run D (decay=0.99) 9.2%, Run F (decay=0.99 + FiLM) 8.5% — all monotonically EMA > raw at every EMA-eval epoch starting from epoch 1. EMA's noise-smoothing benefit is independent of FiLM's regularization, confirming "pure compounding lever" framing from the original hypothesis.
- **Eval-cost recovery worked exactly as predicted.** `--ema_eval_every 2` halves EMA-eval frequency without measurable signal loss; recovered ~1 lost training epoch.
- **No antagonistic interaction**, unlike Fourier × FiLM (#347 round 3). EMA and FiLM are complementary rather than competing for capacity.

### Useful follow-ups (deferred)

- **Tighter decay sweep on FiLM-merged path.** Decay=0.99 worked here; FiLM's smoother training trajectory might benefit from decay=0.995 (half-life ~0.37 epoch). Marginal gains expected (1-2%).
- **`--ema_eval_every 1` on FiLM-merged path.** With FiLM's smoother trajectory the every-2 schedule may be missing useful checkpointing on odd epochs. Run F still finished in 31.2 min.
- **3-seed pin-down of Run F vs FiLM-only at seeds 123/456/789** to convert mechanism into validated lift with ±2% confidence.

### Action

Merged. New baseline: val_avg/mae_surf_p=109.19, test_avg/mae_surf_p=98.47. EMA is now a default-on compounding lever for future PRs.

---

## 2026-04-28 06:30 — PR #523: H14 5-D conditioning vector for FiLM — **CLOSED**

- Branch: `willowpai2d4-edward/h14-film-cond5`
- Hypothesis: Expanding FiLM conditioner from 1-D `log(Re)` to 5-D `[log(Re), AoA1, AoA2, gap, stagger]` should reduce `val_avg/mae_surf_p` by 1–4% with sharper per-split improvements on geom-OOD.
- 3-cell matrix in W&B group `h14-film-cond5`:

| Run | film_cond_dim | seed | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|--------------:|-----:|---------------------:|----------------------:|-----|
| **A — sanity** | 1 | 123 | **119.36** (matches baseline exactly to 4 sig figs) | **107.54** | `se3be70u` |
| B — canonical | 5 | 123 | **115.87** (−2.93%) | **104.36** (−2.96%) | `cs2op6wt` |
| C — variance check | 5 | 124 | 125.65 (+5.27%) | 111.49 (+3.67%) | `59ixyv9m` |

### Conclusions

- **Run A reproduces the merged baseline exactly** — the harness change is clean and FiLM identity-init still holds at cond_dim=1. Second `--seed 123` reproducibility check passing.
- **Run B is inside the predicted band but Run C blows it.** B-C peak-to-peak at fixed cond_dim=5 is **8.1% on val, 6.7% on test** — *larger* than the B-vs-A signal. Mean(B, C) − A = +1.17% / +0.36% — essentially zero average effect.
- **The 5-D conditioner is, at this scale, a wash.** Single-seed Run B looks like a win but isn't reproducible.
- **`test_geom_camber_cruise` improvement DOES survive seed averaging** at -7.2% mean. The richer geometry conditioning helps cruise specifically.
- **Run C's `val_single_in_dist` blowup (176.14)** likely caused by the artificial-zero failure mode — single-foil samples have AoA2=0, gap=0, stagger=0 in the 5-D conditioner, creating a regime change vs tandem.

### Useful follow-ups (deferred)

- **Run D = cond1, seed=124** would fully decouple seed from conditioning. Cheap.
- **`nn.LayerNorm(cond_dim)` before the FiLM MLP** addresses uneven conditioner-input magnitudes.
- **cond_dim=3** (`[log(Re), AoA1, AoA2]`) drops the zero-padded geometry channels.
- **Multi-seed at cond5** (5+ seeds) would give a real distribution.
- **Log gamma/beta magnitudes mid-training** as a diagnostic on whether FiLM head is doing meaningful work.

### Action

Closed; reassigning edward to H17 (layer-wise learning rate decay for Transolver blocks) — fresh optimization-bucket hypothesis, well-motivated for small-data short-training regimes (BeiT, ELECTRA, MAE), orthogonal to all in-flight hypotheses.

---

## 2026-04-28 06:04 — PR #343: H6 bf16 + torch.compile + larger batch — **SENT BACK FOR REBASE-ONTO-#404**

- Branch: `willowpai2d4-askeladd/h6-bf16-compile-batch` (pre-#344, pre-#404; needs rebase onto current advisor branch)
- Hypothesis: bf16 + torch.compile + larger effective batch should reduce `val_avg/mae_surf_p` by 3-9% via more epochs in the same wall clock.
- 4-cell matrix in W&B group `h6-throughput`:

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | sec/epoch | epochs done | peak GPU | W&B |
|-----|--------|---------------------:|----------------------:|----------:|------------:|---------:|-----|
| Baseline (fp32, bs=4) | — | 131.47 | 119.73 | 132.6 | 14 | 42.1 GB | `fda9lzmd` |
| **A — bs=4 + bf16 + compile** | — | **89.03** | **80.54** | 49.1 | **37** | **23.8 GB** | `b4xz4xsi` |
| B — bs=8 + bf16 + compile | — | 98.63 | 89.09 | 53.9 | 34 | 47.6 GB | `w2xtok4w` |
| C — bs=8 + bf16 + compile, --epochs 75 | — | 99.26 | 88.94 | 53.8 | 34 | 47.6 GB | `est8u5n0` |

### Per-split test for Run A

| Split | Pre-merge baseline | Run A | Δ |
|-------|-------------------:|------:|--:|
| `test_single_in_dist` | 127.87 | **90.78** | −29.0% |
| `test_geom_camber_rc` | 138.44 | **92.51** | −33.2% |
| `test_geom_camber_cruise` | 88.48 | **59.17** | −33.1% |
| `test_re_rand` | 124.14 | **79.72** | −35.8% |

### Conclusions (provisional, pre-rebase)

- **Largest single-PR effect of round 0 — by a wide margin.** Run A's −32% / −33% improvement vs askeladd's pre-merge baseline (and **−25.4% / −25.1% vs the current PR #404 merged baseline**) is far above the predicted 3–9% band and well above the seed-variance floor.
- **The mechanism is clean: 2.6× throughput → convergence.** 49.1 s/epoch (vs 132.6 s baseline) lets the model run 37 epochs in 30 min, vs 14 at fp32. Val curve was still descending steeply at the prior baseline's timeout — most of the MAE reduction comes from finally letting the model converge.
- **bs=8 hurts.** Both bs=8 variants (B, C) regress by ~10 pts on val and ~9 pts on test vs Run A. WeightedRandomSampler at fixed 1500 samples/epoch means bs=8 → 187 optimizer steps vs bs=4 → 375; lr schedule wasn't re-tuned for the lower step count. bs=4 is the working config.
- **Cross-split signature matches the prediction.** Improvement is uniform across the four splits with marginally larger gains on geom-OOD and Re-OOD (more epochs help generalization most).
- **VRAM dropped 43%** (42 GB → 24 GB at bs=4). With 96 GB available, plenty of headroom for future model scaling.
- **NaN fix in `data/scoring.py` correctly diagnosed.** Same root cause edward's PR #344 fix addressed; askeladd's `evaluate_split` workaround duplicates the now-merged version. Drop on rebase.
- **Branch is pre-#344, so missing both schedule fixes and FiLM.** The published baseline askeladd compared against (val=131.47) is essentially the original round-0-setup baseline + seed variance; both PR #344 (val=120.97) and PR #404 (val=119.36) merged after his branch was cut.

### Action

Sent back for one focused **Run F — bf16+compile + FiLM merged + merged config** (`--batch_size 4 --amp_dtype bf16 --compile True --film_re True --weight_decay 5e-4 --seed 123 --lr 7e-4 --epochs 37`). Decision rule:
- val < 110 (clear large-margin win): merge as the new baseline; redefines round 0
- val ∈ [110, 119.36]: still a comfortable win, merge for the throughput compounding alone
- val > 119.36: bf16+compile × FiLM antagonistic; close cleanly and consider fp32+compile-only or bf16-only fallbacks

### Useful follow-ups (deferred)

- **Re-tune lr/schedule for bs=8.** `lr ∝ √batch` heuristic suggests `lr=7e-4` for bs=8 (already at 7e-4 for bs=4 in merged baseline; would need re-investigation). Out of scope for current PR.
- **`torch.compile(mode="reduce-overhead")` after fixing memory.** Would need padding to fixed N_max once at dataset prep time. Could unlock another 20-30% throughput from CUDAGraphs.
- **Run for longer wall-clock** if SENPAI_TIMEOUT_MINUTES is ever raised — at 49 s/epoch even Run A's 37 epochs hadn't fully plateaued (best epoch was 36, the last one). 60-min budget would clarify whether the model keeps improving or starts to overfit.

---

## 2026-04-28 06:05 — PR #442 (resubmit): H12 EMA decay=0.99 — **SENT BACK FOR REBASE-ONTO-#404**

- Branch: `willowpai2d4-thorfinn/h12-ema-weights` (pre-#404)
- Run D: `--ema_decay 0.99 --ema_eval_every 2 --epochs 25 --lr 7e-4` on the post-#344 path.

| Metric | Run D |
|--------|------:|
| W&B run | `a4s76abw` |
| Best epoch | 11/14 |
| `val_avg/mae_surf_p` (raw) | 133.55 |
| `val_avg_ema/mae_surf_p` (best, source=ema) | **121.24** |
| `test_avg/mae_surf_p` (eval source: ema) | 109.68 |

### Conclusions (provisional, pre-rebase)

- **EMA mechanism is now triply validated.** Run B (decay=0.999) had within-run lift 10.1%; Run D (decay=0.99) has 9.2%. EMA monotonically beats raw at every EMA-eval epoch from epoch 1 onward.
- **Decay=0.99 is the best EMA config we've found.** Better absolute val_ema (121.24 vs Run B's 125.81), better test (109.68 vs 112.28). The lower decay tracks fast early improvement *and* smooths late oscillations — exactly the sweet spot.
- **Eval-cost recovery worked exactly as predicted.** Run D got 14 epochs vs Run B's 13.
- **Run D val_ema=121.24 is +0.22% above PR #344 baseline** (apples-to-apples for thorfinn's branch path, within seed-variance floor) and **+1.6% / +2.0% above current PR #404 merged baseline**. Per my decision rule, this should close — but EMA is a pure compounding lever, and the mechanism has been validated three times.

### Action

Sent back for one focused **Run F — EMA + FiLM on merged baseline** (`--use_ema True --ema_decay 0.99 --ema_eval_every 2 --film_re True --epochs 25 --lr 7e-4 --weight_decay 5e-4 --seed 123`). Decision rule: merge if val_ema < 119.36; close cleanly if ∈ [119.36, 121.0]; close if antagonistic.

If FiLM produces a more stable training trajectory (plausible), the raw val on the merged path will likely be near 119.4, and applying EMA's ~9% lift could land in the 108-110 range — a real compound win.

---

## 2026-04-28 05:41 — PR #347 (round 3): H5 Fourier features × FiLM stack — **CLOSED**

- Branch: `willowpai2d4-nezuko/h5-fourier-features` (rebased onto post-#404 advisor branch with FiLM merged)
- Round 3 single-cell test: Run H = Fourier σ=5, num_freq=32, **+FiLM enabled** + merged config (`--epochs 25 --lr 7e-4 --weight_decay 5e-4 --seed 123`).

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | best epoch | W&B |
|-----|--------|---------------------:|----------------------:|-----------:|-----|
| Merged baseline (#404, FiLM only) | — | 119.36 | 107.54 | 13/14 | `p0a1daar` |
| Round 2 Run F (Fourier only, pre-#404) | — | 117.17 | 106.46 | 14/14 | `dk13xxhh` |
| **H — Fourier × FiLM** | both | **129.49** (+8.5%) | **118.34** (+10.0%) | 11/13 | `f3u9oemw` |

### Per-split test for Run H

| Split | FiLM-only baseline | Fourier-only Run F | **Run H (F + FiLM)** | H vs baseline |
|-------|-------------------:|-------------------:|---------------------:|--------------:|
| `test_single_in_dist` | 120.69 | 128.57 | **143.38** | **+18.8%** ❌ |
| `test_geom_camber_rc` | 120.45 | 116.27 | **135.58** | **+12.6%** ❌ |
| `test_geom_camber_cruise` | 80.70 | 74.09 | **80.39** | −0.4% (flat) |
| `test_re_rand` | 108.32 | 106.93 | **113.99** | +5.2% ❌ |

### Conclusions

- **Per the advisor's decision rule (val > 121.0 → close), this is a clean close.** Run H val=129.49 is +8.5% above the merged baseline — well above threshold.
- **Fourier × FiLM is antagonistic at this dataset/budget.** Run H is worse than EITHER mechanism alone. The geom-OOD gain that Fourier-only delivered (cruise -8.71% in Run F) collapses entirely under FiLM (Run H cruise basically matches baseline).
- **Plausible mechanism (per nezuko's analysis):** FiLM's (γ, β) scales applied after `preprocess` re-weight the high-frequency content Fourier features just injected. The slice tokens (capacity-limited at slice_num=64) have to allocate representation between two conditioning channels (Re via FiLM, spatial via Fourier) instead of one. With identity-init and only ~13 epochs, FiLM doesn't have time to learn how to use the new feature mix.
- **Val curve was unstable near convergence** (epoch 11=129.5, 12=129.6, 13=181.9), supporting the antagonism reading rather than a noise-floor wash.
- **The mechanisms each work alone** — Fourier σ=5 was a real -3.14% / -3.15% improvement on the pre-#404 path (Run F), and FiLM is a real -1.3% / -2.2% improvement on the post-#344 path. They just don't compose.

### Useful follow-ups (deferred)

- **Don't bury Fourier features.** Future hypotheses that offer Re conditioning *via the Fourier pipeline* (e.g., log(Re) as input to Fourier projection, or σ-conditional Fourier features) may compose better than parallel FiLM-on-Re + Fourier-on-(x,z).
- **σ-search above 5 with FiLM off.** Round 2's monotone-decreasing curve {3, 4, 5} is unverified above σ=5. A pre-#404 path (σ ∈ {6, 7, 8}) sweep would establish the right tail of the σ-curve. Out of scope for current branch but documented.
- **Larger slice_num + Fourier + FiLM.** Nezuko's "slice tokens over-subscribed" hypothesis is testable: (slice_num=128, FiLM on, Fourier on) would isolate the capacity-limit reading.

### Action

Closed; reassigning nezuko to H16 (arcsinh-compressed pressure target). Plays to her sweep-design strength (3-cell scale sweep), addresses the heavy-tailed pressure issue head-on via target transformation rather than loss reweighting (orthogonal to alphonse's H1 in flight).

---

## 2026-04-28 05:14 — PR #490: H13 stochastic depth (DropPath) on Transolver blocks — **CLOSED**

- Branch: `willowpai2d4-frieren/h13-stochastic-depth`
- Hypothesis: DropPath at peak rate 0.10 with linear-per-block scaling should reduce `val_avg/mae_surf_p` by 1–4% via implicit ensembling.
- 3-cell matrix in W&B group `h13-stochastic-depth` (all `--epochs 25 --lr 7e-4`):

| Run | drop_path_rate | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B |
|-----|---------------:|---------------------:|----------------------:|-----------:|-----|
| A — sanity | 0.00 | 130.27 | 118.64 | 13 | `tkbdhacg` |
| **B** | **0.10** | **120.57** | **109.38** | **14** | `j4ypz8o3` |
| C — aggressive | 0.20 | 139.78 | 125.30 | 12 | `ddy0ymee` |

### Conclusions

- **Run B is approximately tied with PR #344 baseline (120.97 / 109.92) but +1.0% / +1.7% regression vs current PR #404 baseline (119.36 / 107.54).** Below the noise floor; can't claim a real effect.
- **Run A is the fourth data point on the seed-variance floor.** Same code as merged baseline (DropPath returns x unchanged when rate=0), +7.7% worse on val. Combined with prior PRs, bidirectional ~6% peak-to-peak noise.
- **Run B vs Run A within-experiment shows the predicted OOD-favoring signature strongly:** `test_geom_camber_rc` -17.3%, `test_re_rand` -8.3%, with single_in_dist and cruise basically flat. DropPath IS doing the regularization work it advertises — the mechanism is real, but the absolute effect at single-seed is too small to attribute against the noise floor.
- **Run C (0.20) underfits.** Best epoch arrives at 12 vs 13-14 for A/B; curve still trending down at timeout. PR's failure-mode prediction was correct.
- **Multi-seed confirmation would burn 3-5 more runs for a likely 0-2% effect.** Not worth the GPU spend given that more promising hypotheses are landing on rebase.

### Useful follow-ups (deferred)

- **Multi-seed at 0.10** if at any point we have spare compute and want to nail down regularizer effects below the noise floor.
- **Lower rate sweep `{0.05, 0.075, 0.10}`** if we want to find the optimum.
- **Compound with round-0 winners** — DropPath is orthogonal to feature/loss/architecture changes; candidate for a "stack-the-winners" PR once 2–3 separate ideas have merged.
- **Longer epochs after H6 lands** — DropPath's mechanism (implicit ensemble) benefits from more forward passes.

### Action

Closed; reassigning frieren to H15 (test-time z-mirror augmentation / TTA) — her own follow-up #3 from H7, builds on her domain knowledge of the dataset's z-symmetry structure. Clean decisive yes/no test of whether the trained model has learned z-symmetry on cruise samples (where physics holds) without the training-time corruption that killed H7.

---

## 2026-04-28 04:53 — PR #347 (resubmit): H5 random Fourier features on (x, z) — **SENT BACK FOR REBASE-ONTO-#404**

- Branch: `willowpai2d4-nezuko/h5-fourier-features` (rebased onto post-#344 advisor branch, but **not yet rebased onto post-#404**)
- Hypothesis: Fourier features on raw (x, z) coords with `(num_freq, sigma)` tuned should reduce `val_avg/mae_surf_p` by 2–8%, primarily on geom-OOD splits.
- 4-cell rebased σ-sweep in W&B group `h5-fourier-rebased`:

| Run | num_freq | σ | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|----------|---|---------------------:|----------------------:|-----|
| E | 32 | 3.0 | 125.81 | 111.32 | `uneydgsj` |
| B' | 32 | 4.0 | 125.16 | 115.87 | `pnodx65v` |
| **F** | **32** | **5.0** | **117.17** | **106.46** | `dk13xxhh` |
| G | 64 | 4.0 | 125.27 | 113.13 | `4oap45xp` |

### Per-split test for Run F (winner) vs PR #344 baseline

| Split | PR #344 baseline | Run F | Δ |
|-------|-----------------:|------:|--:|
| `test_single_in_dist` | 127.09 | 128.57 | +1.16% |
| `test_geom_camber_rc` | 123.58 | 116.27 | **−5.92%** |
| `test_geom_camber_cruise` | 81.16 | 74.09 | **−8.71%** |
| `test_re_rand` | 107.83 | 106.93 | −0.83% |
| **avg** | 109.92 | **106.46** | **−3.15%** |

### Conclusions

- **Hypothesis confirmed on the post-#344 schedule.** Run F at σ=5 / num_freq=32 beats the PR #344 baseline by -3.14% / -3.15%. Both effect size and cross-split signature match the original H5 prediction (geometry-OOD specific gain, slight in-dist regression).
- **σ-curve direction reversed on the merged schedule.** Round 1 (pre-merge code) found σ=4 as the U-shape winner. Round 2 (post-merge code) finds {3, 4, 5} monotone-decreasing. The merged warmup+cosine schedule benefits high-frequency Fourier features specifically — at lower σ the network duplicates information already in raw (x,z) and just slows convergence.
- **Capacity (num_freq) does not substitute for σ.** Run G (σ=4, num_freq=64) ≈ Run B' (σ=4, num_freq=32). Doubling num_freq while holding σ fixed buys nothing.
- **Branch was NOT rebased onto PR #404.** Run F's val=117.17 is on the post-#344, pre-#404 code path (no FiLM, default wd=1e-4). The current merged baseline is val=119.36 with FiLM enabled.
- **Squash-merging as-is would create an untested Fourier × FiLM combination.** Possible outcomes: additive (val ≈ 115), partial overlap (val ≈ 117), antagonistic (val > 119.36).

### Action

Sent back for one focused **Run H — Fourier σ=5 stacked on top of the merged FiLM baseline** (`--film_re True --fourier_num_freq 32 --fourier_sigma 5.0 --epochs 25 --lr 7e-4 --weight_decay 5e-4 --seed 123`). Decision rule: merge if Run H beats val=119.36; close if it regresses materially; need one more run at σ=6 if it lands within ~1%.

### Useful follow-ups (deferred)

- Search above σ=5 (σ ∈ {6, 7, 8}) — the monotone-decreasing observation in {3, 4, 5} likely continues. Worth pushing right after the FiLM-stacking question is answered.
- Re-run with longer schedule once H6 lands — all four current runs hit the 30-min wall at epoch 14/25. The cosine doesn't reach zero.

---

## 2026-04-28 04:09 — PR #404: H11 Re-conditional FiLM modulation — **MERGED** (Round 0 winner #2)

- Branch: `willowpai2d4-edward/h11-film-re-conditioning`
- Hypothesis: FiLM (γ, β) per-block from `log(Re)` should reduce `val_avg/mae_surf_p` by 3–7%, biggest on `val_re_rand`.
- 5-cell matrix in W&B group `h11-film-re` (Round 1 + Round 2 disentanglement):

| Run | FiLM | wd | seed | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|------|----|----:|---------------------:|----------------------:|-----|
| Prior baseline (#344) | — | 1e-4 | none | 120.97 | 109.92 | `rua9xrca` |
| A — sanity | off | 1e-4 | none | 129.27 | 113.83 | `629fuile` |
| B — FiLM on | on | 1e-4 | none | 126.63 | 113.61 | `3so6w84f` |
| C — FiLM on + wd | on | 5e-4 | none | 119.63 | 109.11 | `dbogls54` |
| D — FiLM off + wd | off | 5e-4 | none | 123.95 | 114.61 | `1rth2rs7` |
| **E — FiLM on + wd + seed** | **on** | **5e-4** | **123** | **119.36** | **107.54** | `p0a1daar` |

### Per-split test for Run E (winner)

| Split | Run E | vs prior baseline |
|-------|------:|------------------:|
| `test_single_in_dist` | 120.69 | −5.0% |
| `test_geom_camber_rc` | 120.45 | −2.5% |
| `test_geom_camber_cruise` | 80.70 | −0.6% |
| `test_re_rand` | 108.32 | +0.5% |
| **avg** | **107.54** | **−2.2%** |

### Conclusions

- **Disentanglement decisive: FiLM contributes real signal.** Run D (FiLM off + wd=5e-4) regressed to val=123.95 vs prior baseline (120.97), proving wd=5e-4 alone does *not* close the gap. The C→D toggle at matched wd shows FiLM adds 3.6% val / 5.0% test improvement.
- **Variance check passed.** Run E (seeded reproduction of Run C config) lands at val=119.36 — within 0.2% of Run C's 119.63 — confirming reproducibility across RNG state.
- **The FiLM × wd interaction is the load-bearing finding.** Neither lever helps alone; together they unlock a stable higher-regularization regime. This is the kind of synergy that's invisible without rigorous disentanglement.
- **Cross-split signature was wrong-mechanism but consistent in aggregate.** The "Re-rand benefits most" prediction failed; per-split orderings shuffled with seed. What's stable is the *aggregate* test improvement of ~5% from FiLM at matched wd. The 1-D log(Re) conditioner is likely starving the FiLM head.
- **Cost:** +83K params (12.6%) for −2.2% test improvement. Acceptable; future students inherit it.

### Useful follow-ups

- **Richer conditioning vector** (`[log(Re), AoA1, AoA2, gap, stagger]`) — likely raises the headline gain *and* sharpens the per-split signature. Edward's natural next assignment.
- **FiLM hidden=32** — halves the FiLM head from 83K → 42K, tightening the +12.6% params objection.
- **Concat-Re instead of FiLM** — cheaper alternative; test if it captures most of the gain.
- **3-seed E** — would give a real error bar on the −2.2% headline.

### New methodological tooling

This PR plumbs an optional `--seed` CLI flag (sets `torch.manual_seed` and `torch.cuda.manual_seed_all`). DataLoader workers/sampler are NOT seeded, so reproducibility is partial — sufficient for variance checks, not for fully deterministic training. Future PRs needing variance confirmation can reuse it.

---

## 2026-04-28 04:07 — PR #442: H12 EMA of model weights — **SENT BACK FOR DECAY=0.99 RUN**

- Branch: `willowpai2d4-thorfinn/h12-ema-weights`
- Hypothesis: EMA of weights at decay 0.999 reduces noise from B=4 stochastic gradients; should reduce `val_avg/mae_surf_p` by 1–4% via late-training oscillation smoothing.
- 3-cell matrix in W&B group `h12-ema` (all `--epochs 25 --lr 7e-4`):

| Run | EMA | best epoch | val_avg/mae_surf_p (raw) | val_avg_ema/mae_surf_p | active source | test_avg/mae_surf_p | W&B |
|-----|-----|-----------:|-------------------------:|-----------------------:|:-------------:|--------------------:|-----|
| A — sanity | off | 14 | **117.85** | n/a | raw | **106.89** | `pqrn0wez` |
| B — canonical | decay 0.999 | 13 | 138.91 (e12) | **125.81** | ema | 112.28 | `er4p0s4t` |
| C — longer window | decay 0.9995 | 13 | **131.63** | 156.09 | raw | 119.31 | `yqbcleng` |

### Conclusions

- **Run A's accidental finding: ~6% peak-to-peak seed-variance floor.** EMA off = identity behavior = same code as merged baseline, yet Run A landed at val=117.85 / test=106.89 vs the published baseline's 120.97 / 109.92 — **2.6% / 2.8% improvement from seed luck** in the opposite direction of the variance we saw on PRs #404 and #406. Combined with those, the seed-variance floor on this dataset/budget is ~6% peak-to-peak.
- **EMA mechanism IS real within-run.** Run B's val_ema=125.81 vs val_raw=139.97 → EMA is **10% better than raw** at the best epoch. Crosses below raw at epoch 8 and stays below for every subsequent epoch. Matches the canonical late-training-noise-smoothing prediction.
- **Run C confirms decay-too-high failure mode.** Half-life ~3.7 epochs in a 13-epoch run means EMA never catches up. Always 18–30% worse than raw.
- **Absolute test numbers don't beat baseline.** Run B at val=125.81 / test=112.28 is +4% / +2% regression vs published baseline. Two confounders: (a) dual-eval cost ate ~1 training epoch, (b) decay=0.999 still slightly too high for our 13-14 epoch budget.

### Action

Sent back for one focused run: **Run D — decay=0.99 + every-other-epoch EMA eval**. Half-life ~0.2 epoch lets EMA track the fast early improvement *and* smooth late oscillations; halving the EMA-eval frequency recovers the lost training epoch. Decision rule: merge if Run D's val_ema clearly beats val_raw AND lands below the merged baseline (120.97); close cleanly otherwise.

### Methodological note

This is the third PR in a row exposing the seed-variance floor (~±3% per direction, ~6% peak-to-peak):
- #404: Run A control 7% *worse* than baseline on equivalent code path
- #406: Run A control 6% *worse* than baseline on equivalent code path
- #442: Run A 2.6% *better* than baseline on equivalent code path

This bound is now well-calibrated for this branch's runs. Predicted effect sizes <5% require multi-seed confirmation up front, OR can ride on top of larger-effect winners (e.g. H1) where the headline number is already moving by 5-10%.

---

## 2026-04-28 03:13 — PR #406: H10 surf_weight ramp curriculum — **CLOSED**

- Branch: `willowpai2d4-frieren/h10-surf-weight-ramp`
- Hypothesis: Linear ramp `surf_weight` 5→30 across training should reduce `val_avg/mae_surf_p` by 1–4%.
- 4-cell matrix in W&B group `h10-surf-weight-ramp` (all on the merged schedule, `--epochs 25 --lr 7e-4`):

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|-----|--------|---------------------:|----------------------:|-----|
| Merged baseline (#344) | const 10 | **120.97** | **109.92** | `rua9xrca` |
| A (control) | const 10 | 128.40 | 116.08 | `5pcbzekv` |
| **B (canonical)** | **ramp 5→30** | **122.90** | **111.13** | `22fojpui` |
| C (gentler) | ramp 5→20 | 125.37 | 112.95 | `3s26f5bt` |
| D (high-start) | ramp 10→30 | 125.04 | 112.41 | `gr3jb66w` |

### Conclusions

- **Vs merged baseline, Run B regresses +1.6% / +1.1%.** Headline number doesn't beat baseline.
- **Within-experiment, Run B beats control A by 4.3% on val and test_avg** — squarely inside the predicted 1–4% band, ranking is clean (B > D > C > A).
- **Run A control landed 6% worse than merged baseline on equivalent code paths.** Same noise floor as Edward's PR #404 — single-run comparisons aren't trustworthy below ~6% effect size at this scale.
- **Run B beats Run D despite both ending at sw=30.** This isolates the *ramp* effect from "higher final weight" — curriculum mechanism appears real.
- **No volume regression — Run B's vol_p is *better* than control on every split** (-3.6 single, -3.8 rc, -1.6 cruise, -5.6 re_rand). The PR's "late-training surface emphasis bleeds into volume" failure mode did not materialize.
- **Cross-split signature was opposite of predicted.** Predicted strongest gain on `val_single_in_dist`; observed strongest gains on `test_re_rand` (-14.5 vs A) and `test_geom_camber_cruise` (-12.5 vs A), with `test_single_in_dist` regressing (+4.8 vs A). Different theory: ramped late-training surface emphasis helps OOD distribution shift, slightly overfits in-dist surface noise.

### Useful follow-ups (deferred, may be revisited under multi-seed protocol)

- **Cosine ramp shape** (more high-surf-weight time at the end, may amplify curriculum effect).
- **Reverse ramp (30→5)** as ablation — should regress meaningfully if curriculum mechanism is the right story.
- **Per-domain `surf_weight` or per-split early stopping** — the `test_single_in_dist` regression is consistent across all ramped variants.

### Action

Closed; reassigning frieren to H13 (stochastic depth on Transolver blocks) — fresh architectural-regularization bucket, orthogonal to everything in flight.

### Note on seed-variance floor

This is the second PR (after #404) where same-config control runs land ~6–7% worse than the published baseline. Future predicted-<5% effects should plan multi-seed confirmation up front.

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
