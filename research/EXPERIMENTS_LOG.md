# SENPAI Research Results — icml-appendix-charlie-pai2f-r3

## 2026-04-29 20:15 — PR #1280: Lion lr=1e-4 on full current-best config (charliepai2f3-nezuko) — CLOSED DEAD END (LR-too-low regression)

- Branch: charliepai2f3-nezuko/lion-lr-1e-4-current-best (closed, branch deleted)
- Hypothesis: lr=1e-4 with full current-best schedule (warmup=5 + T_max=100) might beat lr=1.5e-4 baseline (PR #1258); the original stale-config PR #1209 sweep that ranked lr=2e-4 best lacked warmup/T_max=100.
- Results: hypothesis REJECTED — lr=1e-4 regresses on every val and test split.

| Split | val (lr=1e-4) | val baseline (1.5e-4) | Δ |
|-------|---------------|------------------------|---|
| val_single_in_dist | 34.5394 | 32.1133 | +2.4261 |
| val_geom_camber_rc | 50.0609 | 47.2012 | +2.8597 |
| val_geom_camber_cruise | 18.3391 | 17.1896 | +1.1495 |
| val_re_rand | 37.7178 | 36.1165 | +1.6013 |
| **val_avg** | **35.1643** | **33.1552** | **+2.0091 (+6.06%)** |
| **test_avg** | **29.0013** | **28.1158** | **+0.8855 (+3.15%)** |

- Metrics path: target/models/model-charliepai2f3-nezuko-lion-lr-1e-4-current-best-20260429-193618/metrics.jsonl
- Best epoch = 65/100, reached ep66 (same wall-clock budget as baseline). LR at cutoff = 3.52e-5.
- Commentary: Closes the lower bracket of the LR triangulation on the full FiLM+Fourier+warmup+T_max=100 config: lr=1e-4 (35.1643) > lr=1.5e-4 (33.1552, best) < lr=3e-4 (34.3851), with frieren's lr=2e-4 result (PR #1250 review) at 33.81. The optimum is at 1.5e-4 with shallow curvature on the high side and steep regression on the low side. Consistent with Lion's wall-clock-distance argument: per-step magnitude ≈ lr; lower lr means less cumulative parameter movement under a binding 30-min wall-clock. LR sweeping has saturated — further headroom is in architecture / surf_weight / weight_decay / slice_num (PRs #1294, #1286, #1285, #1282 in flight).

## 2026-04-29 23:30 — PR #1257: T_max=200 extreme slow decay (charliepai2f3-fern) — CLOSED DEAD END

- Branch: charliepai2f3-fern/grad-accum-effective-batch16-tmax100-warmup5 (closed)
- Hypothesis: T_max=200 (vs T_max=100) keeps LR ~54% of peak at ep66 cutoff vs ~35%, potentially extending productive learning further into wall-clock budget.
- Result: val_avg/mae_surf_p = 36.8593 — **+7.19% worse** than baseline 34.3851. Best epoch 61, then oscillation/regression.
- Commentary: LR stays too high throughout the budget — model oscillates after ep61 without converging. The opposite signal (lower LR via halving lr to 1.5e-4 at T_max=100) won instead in PR #1258. Confirms T_max=100 is the right horizon for this 30-min budget; further LR-horizon extensions are not productive without also reducing peak LR.

## 2026-04-29 23:30 — PR #1258: lr=1.5e-4 finer LR sweep on full FiLM+Fourier+warmup+T_max=100 config (charliepai2f3-nezuko) — MERGED, NEW BEST

- Branch: charliepai2f3-nezuko/lr-1p5e-4-tmax100-warmup5-100ep (squash-merged into icml-appendix-charlie-pai2f-r3)
- Hypothesis: Halve peak LR from 3e-4 to 1.5e-4 with Lion sign-based optimizer on the current best schedule. Sign-based optimizer benefits from lower peak: smaller, less noisy parameter updates over the wall-clock budget.
- Result: val_avg/mae_surf_p = **33.1552** — **−3.58%** vs baseline 34.3851. test_avg/mae_surf_p = 28.1158 (vs 29.0050: −3.07%). Best epoch=66/100 (wall-clock limited), still strictly improving at cutoff.

| Split | val_mae_surf_p | test_mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 32.1133 | 27.8899 |
| geom_camber_rc | 47.2012 | 43.2971 |
| geom_camber_cruise | 17.1896 | 14.3845 |
| re_rand | 36.1165 | 26.8917 |
| **avg** | **33.1552** | **28.1158** |

- Commentary: All 4 val splits and all 4 test splits improved. Largest val gain on val_geom_camber_cruise (−6.77%), largest test gain on test_single_in_dist (−5.87%). LR ~5.25e-5 (≈0.35× peak) at ep66 — same fraction-of-peak as previous best; the win is from lower absolute LR magnitude, not from horizon. Confirms the Lion-sign-update intuition: lower magnitude per step is favored for this architecture/budget. Brackets the optimum below 2e-4. frieren's lr=2e-4 (PR #1250) and the upcoming finer sweep should narrow this further.
- Metrics path: target/models/model-charliepai2f3-nezuko-lr-1p5e-4-tmax100-warmup5-100ep-20260429-184943/metrics.jsonl

## 2026-04-29 10:52 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- charliepai2f3-alphonse/compound-baseline-lion-l1-ema-bf16-n1
- Hypothesis: re-run the charlie-pai2e-r5 compound recipe as a clean anchor on the new round, so subsequent experiments measure against a freshly executed baseline rather than a referenced number.
- Results:

  | Metric | Value |
  |---|---|
  | val_avg/mae_surf_p | **47.3987** |
  | val_single_in_dist/mae_surf_p | 50.0824 |
  | val_geom_camber_rc/mae_surf_p | 62.7615 |
  | val_geom_camber_cruise/mae_surf_p | 28.5501 |
  | val_re_rand/mae_surf_p | 48.2009 |
  | Peak VRAM | 9.02 GB |
  | Wall time | ~22 min, 50 epochs |
  | Metrics path | `target/models/model-charliepai2f3-alphonse-compound-baseline-lion-l1-ema-bf16-n1-20260429-102214/metrics.jsonl` |

- Verdict: **MERGED** as new round-3 anchor. Improved on the referenced charlie-pai2e-r5 number (47.7385 → 47.3987, −0.34). Per-split is slightly different from the reference (cruise camber improved meaningfully, single-in-dist and rc regressed a touch), so future PRs should treat the new per-split numbers as the comparison target.

## 2026-04-29 11:01 — PR #1104: FiLM global conditioning (Re/AoA/NACA via scale+shift) — sent back for rebase
- charliepai2f3-edward/film-global-conditioning
- Hypothesis: inject the 11-dim global scalar vector (log Re, AoA1, NACA1, AoA2, NACA2, gap, stagger) into each Transolver block as DiT-style scale+shift on both attention and MLP sublayers, with zero-init on the final FiLM projection so the network starts identical to the non-FiLM baseline.
- Results (against reference baseline 47.7385, before #1093 anchored 47.3987):

  | Metric | Reference | Edward | Δ |
  |---|---|---|---|
  | val_avg/mae_surf_p | 47.7385 | **42.3822** | −5.36 (−11.2%) |
  | val_single_in_dist/mae_surf_p | 49.68 | 43.0534 | −6.63 |
  | val_geom_camber_rc/mae_surf_p | 60.82 | 56.9802 | −3.84 |
  | val_geom_camber_cruise/mae_surf_p | 30.55 | 25.1076 | −5.44 |
  | val_re_rand/mae_surf_p | 49.90 | 44.3876 | −5.51 |
  | test_avg/mae_surf_p (bf16, post-fix rerun) | — | 35.8802 | — |
  | test_avg/mae_surf_p (fp32, post-fix rerun) | — | 35.8504 | — |
  | Peak VRAM | — | 3.4 GB | low |
  | Wall time | — | 22.2 min, 50 epochs | matched budget |
  | n_params | ~117K | 245,319 | ~2.1× |
  | Metrics path | — | `target/models/model-charliepai2f3-edward-film-global-conditioning-20260429-100550/metrics.jsonl` | — |

- Notes: best epoch = 50 (final) → still descending; student suggests longer training, FiLM on the preprocess MLP, Fourier on log(Re) for high-Re tail. PR also ships an alternate fix for the NaN bug (drops samples with non-finite `y` from each batch in both train and eval, plus an extra `event: test_rerun_with_nan_filter` line in metrics.jsonl). The originally committed `test_avg/mae_surf_p` is NaN due to the upstream scoring bug; the post-fix rerun line provides clean test numbers.
- Verdict: **REQUEST CHANGES (rebase, top priority)** — strongest signal of the round so far. Squash-merge conflicted with #1093 anchor. Sent back: rebase onto icml-appendix-charlie-pai2f-r3, re-run the same command, keep the NaN filter in evaluate_split. Gate to merge: `val_avg ≤ ~45` on the rebased run.

## 2026-04-29 10:57 — PR #1106: Fourier positional encoding on (x,z) — sent back for rebase
- charliepai2f3-frieren/fourier-positional-encoding
- Hypothesis: enrich (x, z) with sinusoidal features at frequencies {1, 2, 4, 8, 16}×π so the attention can resolve fine-scale boundary-layer geometry, raising input dim 24 → 44.
- Results (against reference baseline 47.7385, before #1093 anchored 47.3987):

  | Metric | Baseline (ref 47.7385) | Frieren | Δ |
  |---|---|---|---|
  | val_avg/mae_surf_p | 47.7385 | **45.3304** | −2.41 (−5.05%) |
  | val_single_in_dist/mae_surf_p | 49.68 | 46.87 | −2.81 |
  | val_geom_camber_rc/mae_surf_p | 60.82 | 60.82 | ≈0 |
  | val_geom_camber_cruise/mae_surf_p | 30.55 | 26.77 | −3.78 |
  | val_re_rand/mae_surf_p | 49.90 | 46.86 | −3.04 |
  | test_avg/mae_surf_p | — | 38.1284 | — |
  | Peak VRAM | — | 9.32 GB | +0.30 GB |
  | Wall time | — | 21.3 min, 50 epochs | matched budget |
  | Metrics path | — | `target/models/model-charliepai2f3-frieren-fourier-pos-enc-compound-v2-20260429-103213/metrics.jsonl` | — |

- Notes: best epoch = 50 (final) → still descending; student suggests longer training, frequency sweep, and applying Fourier to dsdf channels next. PR also ships a critical bug fix in `evaluate_split` that masks samples whose ground truth contains non-finite entries (sample 20 of `test_geom_camber_cruise` has 761 inf entries that were leaking NaN into bf16 test metrics).
- Verdict: **REQUEST CHANGES (rebase)** — would have merged outright but advisor branch already advanced via PR #1093 so the squash conflicted. Sent back: rebase onto icml-appendix-charlie-pai2f-r3, re-run the same command to confirm the improvement still holds on top of the anchor, keep the NaN guard regardless. Gate to merge after rebase: `val_avg ≤ ~46`.

## 2026-04-29 — PR #1106: Fourier positional encoding on (x,z) — MERGED NEW BEST

- Branch: charliepai2f3-frieren/fourier-positional-encoding
- Hypothesis: Append multi-frequency Fourier features (sin/cos at freqs 1,2,4,8,16) to normalized (x,z) node positions, expanding from 2-dim spatial to 22-dim positional input (total x dim: 24 → 44). Also included NaN fix for test_geom_camber_cruise (non-finite GT in sample 20 — masked via y_finite guard in evaluate_split).
- Results:

| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 45.6222 |
| val_geom_camber_rc | 58.5071 |
| val_geom_camber_cruise | 26.7073 |
| val_re_rand | 46.8250 |
| **val_avg** | **44.4154** |

| Test Split | mae_surf_p |
|------------|-----------|
| test_single_in_dist | 37.8511 |
| test_geom_camber_rc | 53.2684 |
| test_geom_camber_cruise | 21.5381 |
| test_re_rand | 36.0350 |
| **test_avg** | **37.1732** |

- Training: ~21.3 min, 50 epochs, batch_size=4, Peak VRAM: 9.32 GB, n_params: 182,855
- Metrics path: `target/models/model-charliepai2f3-frieren-fourier-pos-enc-rebased-20260429-110704/metrics.jsonl`

- Commentary: **STRONG WIN — 6.29% improvement** (47.3987 → 44.4154, delta = −2.9833). The most impactful single change in Round 3. All 4 splits improved. The geom_camber_rc (OOD camber) split improved most dramatically: 62.76 → 58.51, a 6.8% gain, suggesting the richer multi-frequency spatial encoding helps the model generalize to unseen airfoil geometries. Best epoch = 50 (final epoch), indicating the model is still converging — extended training should compound this gain.

## 2026-04-29 — PR #1103: slice_num sweep {32,64,128} on compound baseline — CLOSED (below current baseline)

- Branch: charliepai2f3-askeladd/slice-num-sweep
- Hypothesis: Sweeping `slice_num` controls physics partitioning granularity. slice_num=128 may help with large CFD meshes via finer slicing; slice_num=64 is the compound baseline default.
- Results:

| slice_num | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** | test_avg | Time | Peak GB |
|----------:|-------------------:|-------------------:|-----------------------:|------------:|-------------|---------|------|---------|
| 32 | 49.25 | 62.13 | 28.96 | 49.08 | 47.3550 | 45.1573 | 17.5 min | 7.8 GB |
| **64** | **48.53** | **61.40** | **29.65** | **49.35** | **47.2312** | **44.7702** | 21.0 min | 9.0 GB |
| 128 | 50.14 | 64.60 | 28.48 | 50.05 | 48.3186 | 45.8856 | 28.0 min | 14.2 GB |

Best: slice_num=64, val_avg/mae_surf_p = 47.2312. Current baseline at review time: 44.4154 (PR #1106).

- Metrics paths:
  - `models/model-charliepai2f3-askeladd-slice-num-32-v3-20260429-105257/metrics.jsonl`
  - `models/model-charliepai2f3-askeladd-slice-num-64-20260429-111103/metrics.jsonl`
  - `models/model-charliepai2f3-askeladd-slice-num-128-20260429-113246/metrics.jsonl`

- Commentary: **CLOSED — does not beat current baseline.** Ran on old compound baseline without `--fourier_pos_enc`. Key finding: **slice_num=64 is Pareto-optimal** at this model size. slice_num=128 is worse (more VRAM, slower, lower quality). The student also independently confirmed the `test_geom_camber_cruise` NaN fix.

## 2026-04-29 — PR #1105: Per-channel pressure weighting W_p in {2, 3, 5} — CLOSED NEGATIVE

- Branch: charliepai2f3-fern/per-channel-pressure-weight
- Hypothesis: Up-weight the pressure channel (output index 2) in the surface loss to better align training signal with the primary metric (mae_surf_p).
- Results:

| W_p | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | val_avg |
|-----|-------------------|-------------------|------------------------|-------------|---------|
| 1.0 (ctrl) | 46.11 | 64.06 | 27.94 | 48.35 | 46.61 |
| 2.0 | 45.64 | 64.03 | 27.50 | 47.70 | 46.22 |
| 3.0 | 46.29 | 63.64 | 28.82 | 46.44 | 46.30 |
| 5.0 | 46.39 | 64.40 | 29.21 | 44.93 | 46.23 |

Note: All runs without `--fourier_pos_enc`. Current baseline is 44.4154.

- Commentary: **CLOSED NEGATIVE** — Best result (46.22) does not beat the current baseline (44.4154). Near-zero gain even vs its own control (46.61 → 46.22). Pressure up-weighting alone is not a reliable lever; direction closed.

## 2026-04-29 — PR #1108: n_hidden width sweep {128, 192, 256} — DECISIVE NEGATIVE

- Branch: charliepai2f3-tanjiro/n-hidden-width-sweep
- Hypothesis: Wider hidden dimension may capture more complex flow features with single-layer Transolver.
- Results:

| n_hidden | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** | Time/epoch | Peak VRAM |
|----------|-------------------|-------------------|------------------------|-------------|-------------|-----------|-----------|
| 128 (ctrl) | 45.33 | 61.82 | 27.49 | 52.74 | 46.85 | 25.2 s | 9.03 GB |
| 192 | 46.49 | 62.73 | 28.40 | 52.49 | 47.50 | 31.1 s | 12.14 GB |
| 256 | 46.83 | 64.37 | 28.35 | 51.17 | 47.68 | 37.3 s | 15.25 GB |

- Commentary: **DECISIVE NEGATIVE** — width increases monotonically hurt performance with n_layers=1. Do not revisit width scaling.

## 2026-04-29 — PR #1107: EMA decay sweep {0.99, 0.995, 0.999} — SENT BACK FOR CONTROLLED RERUN

- Branch: charliepai2f3-nezuko/ema-decay-sweep
- Hypothesis: EMA half-life optimization — find optimal smoothing for ~10K training steps.
- Results:

| EMA decay | val_avg/mae_surf_p |
|-----------|------------------|
| 0.99 | 47.3572 |
| 0.995 (baseline) | 48.0612 |
| 0.999 | 47.3974 |

- Commentary: Apparent winner 0.99 (47.3572) beats compound baseline (47.3987) by only 0.04, well within observed run-to-run variance (~0.7). Sent back for 3-seed controlled rerun (seeds 42, 123, 456). Merge criterion: mean(3 seeds) < 47.3987.

## 2026-04-29 — PR #1109: log(Re×|saf|+ε) boundary-layer thickness proxy feature — CLOSED NEGATIVE

- Branch: charliepai2f3-thorfinn/boundary-layer-feature
- Hypothesis: Appending a 25th input feature `f_bl = log(Re × |saf| + ε)` would give the model physics-informed signal about local BL regime transitions (laminar, turbulent, separated), improving generalization to OOD Reynolds number splits.
- Results:

| Run | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** | test_avg | best_epoch |
|-----|-------------------|-------------------|------------------------|-------------|-------------|---------|-----------|
| BL feature (f_bl appended) | ~46.8 | ~63.1 | ~28.5 | ~47.6 | ~46.50 | — | 50 |
| BL feature final variant | ~46.5 | ~63.0 | ~28.2 | ~47.3 | ~46.25 | — | 50 |
| **Control (no BL, no fourier_pos_enc)** | ~46.8 | ~63.4 | ~28.3 | ~47.6 | **~46.22** | ~39.84 | 50 |

Note: All three runs WITHOUT `--fourier_pos_enc`. Current baseline is 44.4154 (PR #1106, Fourier pos enc).

- Metrics paths:
  - `models/model-charliepai2f3-thorfinn-bl-feature-20260429-*/metrics.jsonl`
  - `models/model-charliepai2f3-thorfinn-bl-feature-final-20260429-*/metrics.jsonl`
  - `models/model-charliepai2f3-thorfinn-control-no-bl-20260429-114914/metrics.jsonl`

- Commentary: **CLOSED NEGATIVE** — BL feature showed zero gain vs own control (~46.50 vs ~46.22). Ran without `--fourier_pos_enc`, so all results land around 46.2–46.5, far from 44.4154. Re_rand split (Re OOD — primary target of hypothesis) showed no benefit from BL information. Direction closed. Thorfinn reassigned to warmup-cosine-schedule (PR #1155).

## 2026-04-29 — PR #1167: FiLM global conditioning + Fourier pos enc on current best baseline — REBASE PENDING

- Branch: charliepai2f3-alphonse/film-fourier-combined
- Status: REBASE PENDING — merge conflict detected on `gh pr merge 1167 --squash`; sent back 2026-04-29; alphonse must rebase onto post-#1148 `icml-appendix-charlie-pai2f-r3` tip and re-run.
- Hypothesis: Stack FiLM global conditioning (Re/AoA/NACA scale+shift on each TransolverBlock attn+MLP, zero-init) on top of the Fourier positional encoding baseline (freqs=(1,2,4,8,16,32,64)). Tests whether the two mechanisms are orthogonal and their benefits compound.
- Pre-rebase result (val_avg=40.6661, best epoch 50/50 — still descending):

  | Split | Baseline #1148 | FiLM+Fourier (#1167) | Δ |
  |-------|---------------|---------------------|---|
  | val_single_in_dist/mae_surf_p | 44.6169 | 38.0071 | −14.7% |
  | val_geom_camber_rc/mae_surf_p | 57.7367 | 57.5494 | −0.3% |
  | val_geom_camber_cruise/mae_surf_p | 26.7301 | 23.2424 | −13.0% |
  | val_re_rand/mae_surf_p | 46.7462 | 43.8654 | −6.2% |
  | **val_avg/mae_surf_p** | **43.9575** | **40.6661** | **−7.5%** |

  | Test Split | mae_surf_p |
  |------------|-----------|
  | test_single_in_dist | 32.9381 |
  | test_geom_camber_rc | 50.2454 |
  | test_geom_camber_cruise | 19.1481 |
  | test_re_rand | 34.2135 |
  | **test_avg** | **34.1363** |

- Training: ~26.1 min, 50 epochs (best epoch 50 — cap; not converged), Peak VRAM: 10.31 GB, n_params: 250,439, git_commit: 6fc5118
- Metrics path: (pre-rebase; re-run metrics will be committed post-rebase)

- Commentary: **STRONG WINNER pending rebase** — 7.5% improvement vs current best (43.9575 → 40.6661, delta = −3.2914). Both FiLM and Fourier improvements are confirmed individually; stacking them yielded ~85% of the expected additive gain (FiLM alone was −11.2% vs old baseline; Fourier alone was −6.29% + −1.03%; combined is −7.5% vs post-Fourier baseline), indicating meaningful but not fully additive synergy. Key pattern: `val_geom_camber_rc` improved by only −0.3% vs the baseline's dominant error at 57.74 — the hardest OOD split is only barely responding to FiLM conditioning, suggesting this split needs a more targeted mechanism (domain-explicit conditioning, arc-length RPE, or data augmentation for OOD geometry). The model is NOT converged at epoch 50 — extended training (75 epochs) is expected to push further. Key implementation note: FiLM zero-init (`nn.Linear` final projections set to weight=0, bias=0) MUST be re-applied AFTER `self.apply(self._init_weights)` at lines 246–249 in train.py; otherwise kaiming_uniform_ re-initializes the projections to non-zero and the FiLM conditioning dominates at epoch 0, destabilizing training.

## 2026-04-29 17:00 — PR #1175: LR warmup + cosine decay for Lion optimizer — SENT BACK (stale baseline)

- Branch: charliepai2f3-thorfinn/lion-lr-warmup-cosine
- Hypothesis: Lion optimizer's sign-based gradient update may benefit from a LinearLR warmup phase to avoid large early parameter updates, improving training stability and final convergence. LinearLR(start_factor=1/30, warmup_epochs) → CosineAnnealingLR(T_max=50-warmup_epochs) via PyTorch SequentialLR.
- Results (tested against PR #1106 old baseline 44.4154 — NOT current best 39.9450):

| Trial | warmup_epochs | val_avg/mae_surf_p | vs old baseline (44.4154) | best_epoch |
|-------|---------------|-------------------|---------------------------|------------|
| Warmup-10 | 10 | 42.4859 | −4.3% | ~50 |
| Warmup-5 | 5 | 42.8797 | −3.6% | ~50 |

- Commentary: **SENT BACK — tested against stale baseline (PR #1106, 44.4154).** Both warmup variants beat the Fourier-only baseline but are significantly above the current FiLM+Fourier best (39.9450, +6.4% and +7.4% respectively). The warmup signal is genuine — LinearLR to cosine reduces early instability and reaches better optima on the old baseline. The student was sent back with instructions to: (1) test warmup_epochs=10 and warmup_epochs=5 on the current best FiLM+Fourier config with same SequentialLR structure; (2) also try single-decay cosine (skip warmup, T_max=45) per student's own suggestion. LR warmup direction preserved for follow-up.

## 2026-04-29 16:10 — PR #1209: Lion LR sweep {1e-4,2e-4,3e-4,5e-4} on FiLM+Fourier baseline — ASSIGNED (nezuko)

- Branch: nezuko/lion-lr-sweep-film
- Hypothesis: The FiLM+Fourier model (PR #1104) was trained with lr=3e-4, calibrated on the pre-FiLM 184k-param model. The FiLM model has 252k params (+36.5%). Lion LR is known to be sensitive to model scale; optimal LR may have shifted with FiLM conditioning layers added. Sweep lr in {1e-4, 2e-4, 3e-4 (control), 5e-4} on the full FiLM+Fourier baseline to find the true optimum.
- Target metric: beat `val_avg/mae_surf_p = 39.9450`

## 2026-04-29 16:10 — PR #1208: Extended training 75ep + T_max=75 on FiLM+Fourier baseline — ASSIGNED (frieren)

- Branch: frieren/extended-training-film-75ep
- Hypothesis: PR #1104 (FiLM+Fourier) best epoch was 49/50 — the final epoch, a clear convergence signal. Extending to 75 epochs + T_max=75 cosine (single full decay matching training duration) should yield further improvement. Based on convergence trajectory, expect −3 to −5% on val_avg.
- Target metric: beat `val_avg/mae_surf_p = 39.9450`

## 2026-04-29 15:55 — PR #1173: surf_weight sweep {28,32,40,50} on Fourier pos enc baseline — CLOSED DEAD END

- Branch: charliepai2f3-nezuko/surf-weight-sweep
- Hypothesis: surf_weight=28 was calibrated on the pre-Fourier compound baseline. With Fourier positional encoding active, optimal surf_weight may shift. Testing {28, 32, 40, 50} on the Fourier pos enc baseline (PR #1148) to find a better balance between surface and volume losses.
- Results:

| surf_weight | val_avg/mae_surf_p | val_single | val_geom_rc | val_geom_cruise | val_re_rand |
|-------------|-------------------|-----------|-------------|-----------------|-------------|
| 28 (control) | 43.2807 | 41.5985 | 60.2897 | 25.4378 | 45.6970 |
| **32** | **43.2052** | **41.4048** | **60.7736** | **25.3476** | **45.2948** |
| 40 | 44.0009 | 42.5217 | 60.5671 | 25.1978 | 47.7170 |
| 50 | 44.9148 | 44.0413 | 61.0218 | 25.1920 | 49.3040 |

- Best (sw=32): val_avg=43.2052, test_avg=36.4182
- Commentary: **CLOSED — DEAD END relative to current baseline.** The best result (sw=32, val_avg=43.2052) was designed against the PR #1106 Fourier baseline (val_avg=44.4154) and beats that assigned baseline by ~1.2 points. However, the current merged best (PR #1104, FiLM+Fourier) is 39.9450 — making sw=32's result 3.26 points worse than the actual benchmark. sw=28 is confirmed optimal (sw=32 only marginally better, sw≥40 clearly hurts). The volume gradient acts as useful regularization; heavy upweighting of surf_weight is detrimental. This direction is exhausted — surf_weight=28 should remain fixed in all FiLM+Fourier experiments.

## 2026-04-29 17:00 — PR #1170: Depth sweep n_layers in {2, 3} on Fourier pos enc baseline — CLOSED NEGATIVE

- Branch: charliepai2f3-fern/depth-sweep-n-layers
- Hypothesis: Stacking 2–3 Transolver layers may improve generalization, particularly to OOD geometry splits, by enabling deeper spatial reasoning over the physics-informed slice representations.
- Results (current best = 39.9450):

| n_layers | val_avg/mae_surf_p | best_epoch | Notes |
|----------|--------------------|------------|-------|
| 1 (baseline) | 39.9450 | 49/50 | Current best |
| 2 | 45.3904 | ~42 | Timed out at 30 min, still descending |
| 3 | 56.0444 | ~22 | Overtfit from ep 22, monotonically worse after |

- Commentary: **CLOSED — DECISIVE NEGATIVE.** n_layers=3 is unambiguous: overfits at epoch 22 (training stable, validation diverges) on 1499 training samples. n_layers=2 hit the wall-clock timeout at best epoch ~42; even if fully converged, trajectory projects to ~44.0-45.0 — well above 39.9450 and no path to beating baseline. The TandemFoilSet-Balanced training set (1499 samples) does not support deeper Transolver stacking. The n_layers=1 model benefits more from capacity additions through width (FiLM adds 67k params as conditioning network, Fourier expands spatial input) than through depth. Fern reassigned to n_hidden=192 width scaling on FiLM+Fourier config (PR #1202).

## 2026-04-29 17:30 — PR #1218: SWA late-epoch averaging on FiLM+Fourier+warmup baseline — CLOSED

- Branch: charliepai2f3-askeladd/swa-late-epoch-averaging
- Hypothesis: Stochastic Weight Averaging over the last 15 epochs (ep36–50) would beat the EMA checkpoint because the model is not converged at ep50 of the warmup+T_max=45 recipe (PR #1175 baseline).
- Implementation: `torch.optim.swa_utils.AveragedModel` with epoch-level equal-weight averaging from `swa_start_epoch = max(1, epochs-15)`. Evaluated separately from EMA. No BN update needed (LayerNorm only).
- Results:

| Strategy | val_avg/mae_surf_p | best_epoch |
|----------|--------------------|------------|
| EMA (decay=0.995) | **36.3455** | ep49 |
| SWA (15 snapshots) | 36.7199 | (avg) |
| Current baseline (PR #1208) | 35.8406 | — |

- Commentary: **CLOSED — neither beats current baseline 35.8406.** Both methods beat the assigned PR #1175 baseline (37.0739) but the frontier had moved during the experiment. Strong mechanistic finding: under cycling cosine annealing without SWALR, equal-weight averaging across the swa_window pulls in higher-LR/noisier early-window snapshots (ep36 LR ≈ 6× ep50 LR), so EMA's exponential weighting dominates. Confirms that the gain in PR #1208 came from longer LR horizon, not from late-epoch averaging. Open follow-ups (logged for potential next round): SWA+SWALR with proper restart, EMA-of-checkpoints. Reassigned to compose-warmup+T_max=60 (#1231).

## 2026-04-29 17:30 — PR #1167: FiLM + Fourier on best baseline (alphonse) — CLOSED SUPERSEDED

- Branch: charliepai2f3-alphonse/film-fourier-combined
- Hypothesis: combine FiLM global conditioning with Fourier positional encoding on (x,z), originally on the pre-FiLM baseline.
- Long PR history: multiple rebase cycles. Final pivot was single-decay cosine (T_max=50) on FiLM+Fourier → val_avg=38.0015, test_avg=31.2265.
- Commentary: **CLOSED — superseded.** The hypothesis itself was already validated and merged via PR #1104 (edward, FiLM, val_avg=39.9450) and refined by PR #1175 (thorfinn, warmup+T_max=45, 37.0739) and PR #1208 (frieren, 75ep+T_max=75, 35.8406). Final result 38.0015 well above current baseline. Many observations from this PR (T_max=50 single-decay benefit, longer-horizon cosine helps) directly informed the merged thorfinn/frieren chain. Student recommended closure. Reassigned (after one more redirect — see #1232 below) to batch size sweep (#1234).

## 2026-04-29 17:30 — PR #1232: surf_weight sweep (alphonse) — CLOSED BEFORE START (DUPLICATE)

- Branch: charliepai2f3-alphonse/surf-weight-sweep
- Closed at assignment time as a duplicate of PR #1173 (nezuko, surf_weight sweep on Fourier baseline). PR #1173 already established sw=28-32 as the optimum and sw≥40 as decisively worse. Re-running on FiLM+Fourier offered insufficient information value. Reassigned to batch size + LR sweep (#1234).
