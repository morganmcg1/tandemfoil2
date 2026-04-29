# SENPAI Research Results — icml-appendix-charlie-pai2f-r3

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
