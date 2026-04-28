# Round Baseline — `icml-appendix-charlie-pai2d-r2`

Lower is better. Primary ranking metric is `val_avg/mae_surf_p` (mean surface pressure MAE across the four val splits). Paper-facing metric is `test_avg/mae_surf_p` from the best-val checkpoint.

## 2026-04-28 08:58 — PR #601: Huber δ=0.25 → 0.1 (rebased on post-#562/#510 stack)

- **Best `val_avg/mae_surf_p`** (this PR's standalone measurement, on PR #510 same-stack baseline): **62.879** (epoch 17, −3.00% vs 64.824)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **54.561** (vs 56.391 PR #510, −3.25%)
- **Per-split val MAE for `p`** (best epoch 17, EMA, compile=True):
  - val_single_in_dist: **72.871 (−3.92% — largest gainer)**
  - val_geom_camber_rc: 76.167 (−2.19%)
  - val_geom_camber_cruise: 43.897 (−3.39%)
  - val_re_rand: 58.580 (−2.60%)
- **Per-split test MAE for `p`** (best ckpt):
  - test_single_in_dist: 63.120 (−4.02%)
  - test_geom_camber_rc: 67.894 (−3.42%)
  - test_geom_camber_cruise: 36.184 (−1.27%)
  - test_re_rand: 51.046 (−3.42%)
- **Recipe**: **huber(δ=0.10)** (was 0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + per-parameter-group wd: attn=1e-4, mlp=1e-5, other=3e-5 + PhysicsAttention temperature init=2.0 + cosine 3-ep warmup (start_factor=0.3) + T_max=11 + decaying noise std (linear: ep1=0.0025, ep14=0.000179, ep15+=0) + NaN-safe + clip_grad_norm_(max_norm=10.0) + compile=True + lr=6e-4.
- **Loss-magnitude diagnostic at convergence**: only 29% of residuals are linear-regime (vs predicted >95% pseudo-L1). δ=0.1 is a "**better hybrid**", not pseudo-L1: linearizes the heavy-tailed minority (~30%) while keeping ~70% in the smooth quadratic well. Per-channel: p=78% quad (smallest linear fraction), Ux=66%, Uy=68%.
- **Striking finding (second-cycle behavior)**: best epoch 17 in cosine **second half-cycle** (cosine continues past T_max=11 producing a second descent). EMA(0.995) + δ=0.1's smooth-near-zero gradient lets the model squeeze ~1.3 additional pts past the cosine-min minimum.
- **Same-stack effect grew** from −1.05% (pre-rebase) to −3.00% (post-rebase) — the longer 3-ep warmup + T_max=11 schedule extends the late-training fine-tune window where δ=0.1's better tail handling has more room.
- **Compound expectation**: this run was on pre-#635/#636/#640 baseline. δ=0.1 is mechanistically orthogonal to lr peak, noise schedule, and per-group wd — should compound below 62.879 on the combined stack.
- **Metric summary**: `models/model-huber-delta-0p1-rebased-20260428-075855/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-delta-0p1-rebased --agent <name> --compile=True
  ```

## 2026-04-28 08:57 — PR #640: Per-parameter-group weight decay (attn=1e-4, mlp=1e-5, other=3e-5)

- **Best `val_avg/mae_surf_p`** (this PR's standalone measurement, on PR #510 same-stack baseline): **62.747** (epoch 17, −3.21% vs 64.824)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **54.512** (vs 56.391 PR #510, −3.33%)
- **Per-split val MAE for `p`** (best epoch 17, compile=True):
  - val_single_in_dist: 74.787 (−1.39%)
  - val_geom_camber_rc: 76.990 (−1.13% — predicted OOD target ✓)
  - val_geom_camber_cruise: **41.238 (−9.24%, biggest single-split gain)**
  - val_re_rand: 57.972 (−3.61% — predicted OOD target ✓)
- **Per-split test MAE for `p`** (best ckpt):
  - test_single_in_dist: 64.971
  - test_geom_camber_rc: 67.720
  - test_geom_camber_cruise: 34.209
  - test_re_rand: 51.148
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + **per-parameter-group wd: attn=1e-4, mlp=1e-5, other=3e-5** (was wd=3e-5 single scalar) + PhysicsAttention temperature init=2.0 + cosine 3-ep warmup (start_factor=0.3) + T_max=11 + decaying noise std (linear: ep1=0.0025, ep14=0.000179, ep15+=0) + NaN-safe + clip_grad_norm_(max_norm=10.0) + compile=True + lr=6e-4.
- **Param groups**: attn 273K (40.8%, wd=1e-4), mlp 354K (52.9%, wd=1e-5), other 42K (6.3%, wd=3e-5).
- **Mechanism CONFIRMED**: OOD asymmetry mapping correct (attn-higher penalizes geometry overfitting → camber_rc improves; mlp-lower preserves Re-extrap capacity → re_rand improves).
- **Bonus finding**: uniform improvement across ALL splits, not just the OOD axes — the asymmetric regularization principle is broader than the specific OOD asymmetry.
- **Compound expectation**: this run was on pre-#635 (lr=6e-4) and pre-#636 (decaying noise) baseline. Per-group wd is mechanistically orthogonal — should compound below 62.747 on the combined stack.
- **Metric summary**: `models/model-per-group-wd-20260428-081309/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name per-group-wd --agent <name> --compile=True
  ```

## 2026-04-28 08:45 — PR #636: Decaying feature-noise schedule (linear decay 0.0025→0 over 14 ep)

- **Best `val_avg/mae_surf_p`** (this PR's standalone measurement, on PR #562 baseline): **63.222** (epoch 17, −2.28% vs 64.696)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **54.900** (vs 55.879 prior, −1.75%)
- **Per-split val MAE for `p`** (best epoch 17, compile=True):
  - val_single_in_dist: 76.288 (−1.60%)
  - val_geom_camber_rc: 77.187 (+1.61%)
  - val_geom_camber_cruise: **40.296 (−8.89%, biggest gain)**
  - val_re_rand: 59.118 (−3.19%)
- **Per-split test MAE for `p`** (best ckpt):
  - test_single_in_dist: 65.550
  - test_geom_camber_rc: 69.199
  - test_geom_camber_cruise: **33.927 (−6.47%)**
  - test_re_rand: 50.924
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + cosine 3-ep warmup (start_factor=0.3) + T_max=11 + **feature noise decaying schedule** (linear: ep1=0.0025, ep14=0.000179, ep15+=0) + NaN-safe + clip_grad_norm_(max_norm=10.0) + compile=True + lr=6e-4 (from PR #635 just merged on top).
- **Schedule trajectory**: linear from 0.0025 at ep1 to 0 at ep15+; matches LR=0 at ep15. Confirms "noise vanishes when LR vanishes" property.
- **Mechanism**: schedule preserves AND amplifies the early-phase noise benefit on OOD splits while removing the late-phase noise tax. Post-noise tail (ep15-17, std=0) keeps descending 0.5-0.6 pts/epoch under cosine cycle-back — confirms EMA was NOT fully absorbing late-phase noise; the schedule does real work.
- **Compound expectation**: PR #635 (lr=6e-4) just merged with val=63.131. Frieren's PR #636 was measured against the pre-#635 baseline 64.696. Combined-stack measurement (lr=6e-4 + decaying noise) will be measured by next round PRs; expected to compound below 63.131 if levers are orthogonal.
- **Metric summary**: `models/model-charliepai2d2-frieren-feature-noise-decaying-20260428-075320/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name feature-noise-decaying --agent <name> --compile=True
  ```

## 2026-04-28 08:44 — PR #635: lr peak bump 5e-4 → 6e-4 (gentler 3-ep warmup permits 1.2× peak)

- **Best `val_avg/mae_surf_p`**: **63.131** (epoch 17, −2.42% vs 64.696 prior, −2.61% vs 64.824 compile prior)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **55.026** (vs 55.879 prior, −1.53%)
- **Per-split val MAE for `p`** (best epoch 17, compile=True):
  - val_single_in_dist: 73.993 (−4.56%)
  - val_geom_camber_rc: 77.263 (+1.71% — only regression)
  - val_geom_camber_cruise: 42.057 (−4.91%)
  - val_re_rand: 59.209 (−3.04%)
- **Per-split test MAE for `p`** (best ckpt):
  - test_single_in_dist: 64.501
  - test_geom_camber_rc: 68.074
  - test_geom_camber_cruise: 35.466
  - test_re_rand: 52.064
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + cosine 3-ep warmup (start_factor=0.3) + T_max=11 + feature noise std=0.0025 + NaN-safe + clip_grad_norm_(max_norm=10.0) + compile=True option + **lr=6e-4** (was 5e-4).
- **LR-vs-epoch trajectory**: ep1=1.80e-4 (sf 0.3 × 6e-4), ep4=6.00e-4 (peak), ep14=1.21e-5 (cosine end), ep15=0 (eta_min), ep17=4.76e-5 (cosine wraparound).
- **Grad-clip diagnostics**: ep1 mean=21.5 (+10% vs PR #582), max=81.5 (+33%), but **clip rate FELL to 82.7% (vs PR #582's 85.6%)** — the gentler 3-ep warmup means ep1 effective LR is only 1.80e-4. Clip rate at peak ep4=51.5% (well below saturation; more headroom for higher peak LR).
- **Late-epoch slope shape changed**: model gets into fine-tuning regime SOONER. By ep12, val_avg=66.9 already close to final 63.13. Higher peak LR extracts more from the high-LR optimization phase (ep4–ep10).
- **Metric summary**: `models/model-lr-peak-6e-4-20260428-075310/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name lr-peak-6e-4 --agent <name> --compile=True
  ```

## 2026-04-28 07:30 — PR #510: torch.compile mode="default" (infrastructure: +28.6% epochs in budget)

- **Best `val_avg/mae_surf_p`**: **64.824** (epoch 18, on rebased post-#582 stack; vs eager-same-stack 66.149 = −2.0%)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **56.391** (vs 57.654 eager, −2.2%)
- **Per-split val MAE for `p`** (best epoch 18, compile=default):
  - val_single_in_dist: 75.841
  - val_geom_camber_rc: 77.872
  - val_geom_camber_cruise: 45.438
  - val_re_rand: 60.143
- **Per-split test MAE for `p`** (best ckpt):
  - test_single_in_dist: 65.765
  - test_geom_camber_rc: 70.295
  - test_geom_camber_cruise: 36.650
  - test_re_rand: 52.855
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + cosine 3-ep linear warmup (start_factor=0.3) + T_max=11 + feature noise std=0.0025 + NaN-safe + clip_grad_norm_(max_norm=10.0) + **`--compile=True` (torch.compile mode="default")**.
- **Throughput**: 18 epochs in 30-min budget (vs 14 eager), 103.4 s/epoch mean (vs 134.6 s eager). +28.6% epochs / −23.1% wall-clock. Peak VRAM 42.6 GB (−7%).
- **Compile mechanism**: TorchInductor kernel fusion on forward+backward only. CUDA Graphs disabled (mode="reduce-overhead" OOMs from one-graph-per-mesh-shape with variable padding). Compile is mode-orthogonal to all merged levers (verified via `ema_decay` and `train/grad_norm_*` numerical equivalence vs eager). One soft graph break in DropPath at `torch.rand(1).item()` (no error, just split). Compile cost: epoch 1 = 122.9 s including warmup (faster than eager's 139.3 s — fusion benefit on rest of epoch 1 pays back).
- **Schedule downstream consequence**: cosine T_max=11 + 18-epoch budget leaves epochs 12–18 at LR≈eta_min=0 (7 "free" EMA-stabilization epochs at the floor). Future schedule PRs may want to extend T_max=11→14 or set `eta_min>0` to fully utilize the new budget.
- **Vs current baseline (PR #562)**: val_avg 64.824 vs 64.696 = +0.128 (essentially flat — within seed noise). The merge is justified by the throughput compounding for every future PR, not by standalone val_avg improvement.
- **Metric summary**: `models/model-torch-compile-rebased-20260428-064446/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name torch-compile-rebased --agent <name> --compile=True
  ```

## 2026-04-28 07:15 — PR #562: Cosine schedule revision (3-ep warmup, T_max=11, start_factor=0.3)

- **Best `val_avg/mae_surf_p`**: **64.696** (epoch 14, −2.20% vs 66.149 prior / −3.91% vs 67.306)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **55.879** (vs 57.654 prior, −3.08%)
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 77.527
  - val_geom_camber_rc: 75.963
  - val_geom_camber_cruise: 44.229
  - val_re_rand: 61.064
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 66.457
  - test_geom_camber_rc: 67.793
  - test_geom_camber_cruise: 36.274
  - test_re_rand: 52.993
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + **cosine 3-ep linear warmup (start_factor=0.3) + T_max=11** (was 1-ep + T_max=13) + feature noise std=0.0025 + NaN-safe + clip_grad_norm_(max_norm=10.0).
- **LR-vs-epoch trajectory**: ep1=1.50e-4 (start_factor 0.3 × lr 5e-4), ep4=5.00e-4 (peak — cosine begins), ep14=1.01e-5 (cosine-decayed near-zero).
- **Late-epoch slope progression**: ep10→11=−4.71, ep11→12=−3.44, ep12→13=−2.25, ep13→14=−0.70 (matches PR #525 sweet spot of 0.7 — fine-tuning regime cleanly reached).
- **Mechanism**: gentler 3-ep warmup decouples basin-selection smoothness from late-decay aggressiveness. The model lands in the *same* fine-tuning regime as T_max=13 baseline but starts the final descent *from a lower val* because basin selection benefited from the smoother high-LR ramp. T_max=11 still lets cosine reach near-zero LR by ep14 without over-decaying. All 4 val splits and all 4 test splits improved.
- **Standout splits**: val_geom_camber_cruise (−6.03% vs prior baseline reference), test_geom_camber_rc (−5.31%) — geometry-extrapolation splits benefited disproportionately from smoother basin selection.
- **Metric summary**: `models/model-cosine-tmax-11-warmup-3-20260428-063145/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name cosine-tmax-11-warmup-3 --agent <name>
  ```

## 2026-04-28 06:42 — PR #582: Gradient clipping max_norm=10 (under huber-δ=0.25)

- **Best `val_avg/mae_surf_p`**: **66.149** (epoch 14, −0.07% vs 66.195 prior / −1.72% vs 67.306)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **57.654** (vs 58.063 prior, −0.70%)
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 77.989
  - val_geom_camber_rc: 77.651
  - val_geom_camber_cruise: 46.373
  - val_re_rand: 62.584
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 67.262
  - test_geom_camber_rc: 70.184
  - test_geom_camber_cruise: 38.860
  - test_re_rand: 54.310
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.995, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + cosine 1-ep warmup + T_max=13 + feature noise std=0.005 + NaN-safe + **clip_grad_norm_(max_norm=10.0)**.
- **Gradient norm diagnostics**: mean norm starts at ~19.5 epoch 1, decays to ~5 by late training; max norms 17–61 throughout; clip fired 85.6% of batches epoch 1 → ~8% epoch 14. huber-δ=0.25 does NOT bound total parameter gradient norms to <10 — the surf_weight=10 amplifier and model-size scaling keep norms in the 5–20 range throughout training.
- **Mechanism**: aggressive early-epoch clipping (cap 85% of batches at norm 10) acts as a soft early-LR cap, stabilizing the high-curvature warmup phase. The mid-training clip rate of 6–14% removes occasional outlier batches that would otherwise destabilize fine-tuning. Net effect: smoother loss curve and better convergence within the 14-epoch budget.
- **Metric summary**: `models/model-gradient-clip-norm-10-20260428-060150/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name gradient-clip-norm-10 --agent <name>
  ```

## 2026-04-28 06:45 — PR #575: Bias-corrected EMA decay_target 0.99 → 0.995 (UP direction)

- **Best `val_avg/mae_surf_p`**: **66.195** (epoch 14, −1.65% vs 67.306 reference / −0.98% vs #574's 66.847)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **58.063** (vs 58.112 prior, −0.08%)
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 77.239 (−9.71% vs PR #548 reference)
  - val_geom_camber_rc: 78.683 (−1.07% vs PR #548)
  - val_geom_camber_cruise: 46.888 (−7.45% vs PR #548)
  - val_re_rand: 61.972 (−7.13% vs PR #548)
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 67.362
  - test_geom_camber_rc: 71.301
  - test_geom_camber_cruise: 38.809
  - test_re_rand: 54.781
- **Recipe**: huber(δ=0.25) + **bias-corrected EMA(0.995, warmup=50)** (was 0.99) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=2.0 + cosine 1-ep warmup + T_max=13 + feature noise std=0.005 + NaN-safe.
- **EMA decay profile** (monotone-improving, cap never reached within budget): 0.95 → 75.655, 0.99 → ~67.306, **0.995 → 66.195**. Profile still descending — 0.999 not yet probed.
- **EMA decay trajectory** (warmup_steps=50, 14-epoch / ~5250-step budget): step 1→0.039, step 100→0.673, step 1000→0.953, step 4000→0.988. Cap of 0.995 not reached until step ~9750 (well beyond budget) — EMA stays in warmup-ramp regime throughout.
- **Val curve**: smooth monotone descent epochs 1–14 (194.30→66.20), no per-epoch wobble or lag. EMA and live model converge by run end (cosine LR near zero).
- **Mechanism**: Longer EMA memory (0.99→0.995) under cosine LR decay (T_max=13) compounds well: by epoch 14, LR is near zero so live model weights are stable — the slower-tracking EMA doesn't lag, it just averages over a longer window of the fine-tuning tail. Uniform improvement across all 4 splits consistent with generic smoothing knob, not split-specific effect.
- **Confound caveat**: baseline reference (67.306) is on slice-temp=1.0 stack; this run uses slice-temp=2.0 (+feature-noise=0.005). The residual decay-only effect after subtracting the ~1.5% slice-temp contribution is small but likely real given the monotone profile shape.
- **Metric summary**: `models/model-ema-decay-target-0995-20260428-055129/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-decay-target-0995 --agent <name>
  ```

## 2026-04-28 06:35 — PR #574: Slice attention temperature init 1.5 → 2.0

- **Best `val_avg/mae_surf_p`**: **66.847** (epoch 14, on post-#548+#526 stack; combined with feature-noise=0.0025 expected to beat 66.841)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **58.112** (vs 58.488 prior, −0.64%)
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 77.823 (−9.0% vs PR #548 reference)
  - val_geom_camber_rc: 81.137 (+2.0% vs PR #548 — slight regression, within noise)
  - val_geom_camber_cruise: 47.064 (−7.1% vs PR #548)
  - val_re_rand: 61.366 (−8.0% vs PR #548)
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 66.220
  - test_geom_camber_rc: 71.850
  - test_geom_camber_cruise: 38.727
  - test_re_rand: 55.653
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.99, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + **PhysicsAttention temperature init=2.0** (was 1.5) + cosine 1-ep warmup + T_max=13 + feature noise std=0.005 + NaN-safe.
- **Slice-temp profile** (3 measured points, accelerating descent): init=1.0 → 71.699, init=1.5 → 70.617 (Δ=−1.08), **init=2.0 → 66.847** (Δ=−3.77, ~3.5× larger drop). Profile still descending — optimum not yet bracketed from above. Next probes: init=2.5, 3.0.
- **Final temperatures** (mean abs per block): live=[1.9620, 1.9150, 1.9181, 1.9323, 1.9439] mean=1.934, EMA≈identical. Drift rate ~−0.066 over 14 epochs matches predicted ~0.05/14ep from prior runs.
- **Mechanism**: Higher init T → sharper initial attention → model operates in T≈1.93–1.96 regime throughout training (vs T≈1.45 at init=1.5). Sharper attention regime benefits 3 of 4 val splits cleanly; camber_rc regression is within noise.
- **Metric summary**: `models/model-slice-temp-2p0-20260428-054930/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name slice-temp-2p0 --agent <name>
  ```

## 2026-04-28 06:15 — PR #563: Semantics-aware feature noise std 0.005 → 0.0025

- **Best `val_avg/mae_surf_p`**: **66.841** (epoch 14, −0.69% vs 67.306 prior baseline)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **58.488** (vs 59.296 prior, −1.36%)
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 79.768 (+2.71% vs prior — slight in-dist regression)
  - val_geom_camber_rc: 78.262 (−3.01% — biggest val gain)
  - val_geom_camber_cruise: 47.065 (+0.40%, flat)
  - val_re_rand: 62.268 (−2.70%)
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 68.487 (−0.35%)
  - test_geom_camber_rc: 71.596 (−1.42%)
  - test_geom_camber_cruise: 38.544 (−2.83%)
  - test_re_rand: 55.325 (−1.48%)
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.99, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + PhysicsAttention temperature init=1.5 + cosine 1-ep warmup + T_max=13 + **feature noise std=0.0025** (was 0.005) + NaN-safe.
- **Noise profile** (monotone-descending, no saturation yet): 0.0025 < 0.005 < 0.01 (orig) < 0.02 (regression). Profile still heading downward; optimum may be in (0, 0.0025].
- **Mechanism**: With DropPath(0→0.1) + huber(δ=0.25) + bias-corrected EMA(0.99) + wd=3e-5 + warmup already providing strong regularization, explicit feature noise mostly adds gradient noise in the cosine fine-tuning tail. Less noise = cleaner signal, especially benefiting the harder OOD splits (camber_rc, re_rand).
- **Metric summary**: `models/model-feature-noise-0025-20260428-052613/metrics.jsonl`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name feature-noise-0025 --agent <name>
  ```

## 2026-04-28 05:50 — PR #548 merged on top of #525 (slice-temp init 1.5 replaces 1.0)

- **Best `val_avg/mae_surf_p`** (last directly measured): **67.306** (PR #525 with slice-temp=1.0). Post-#548 measurement pending; thorfinn's standalone showed slice-temp=1.5 beats slice-temp=1.0 by −1.51% on the same starting baseline (71.6985 → 70.617), so combined-stack expected to be at-or-below 67.306.
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(0.99, warmup=50) + SwiGLU + DropPath(0.1) + AdamW betas (0.9, 0.95) + wd=3e-5 + **PhysicsAttention temperature init=1.5** (was 1.0) + cosine 1-ep warmup + T_max=13 + feature noise std=0.005 + NaN-safe.
- **Slice-temp profile** (now 3 measured points): init=0.5 (original) underperforms; init=1.0 → 71.6985; init=1.5 → 70.617 (this run). Drift Δ ≈ −0.05 over 14 epochs in both directions, suggesting the temperature has a broad sweet spot in [1.0, 1.5]+ and the 0.95 "equilibrium" found in the live parameters is an artifact of the 14-epoch drift speed, not the val-loss minimum.

## 2026-04-28 05:30 — Previous baseline (PR #525, cosine warmup + T_max=13)

- **Best `val_avg/mae_surf_p`** (directly measured): 67.306 (epoch 14, with slice-temp=1.0)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **59.296**
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 77.660 (−11.66% vs 87.914 reference)
  - val_geom_camber_rc: 80.690 (−3.16% vs 83.323)
  - val_geom_camber_cruise: 46.877 (−6.66% vs 50.222)
  - val_re_rand: 63.996 (−6.16% vs 68.199)
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 68.728
  - test_geom_camber_rc: 72.629
  - test_geom_camber_cruise: 39.668
  - test_re_rand: 56.158
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(decay_target=0.99, warmup_steps=50) + SwiGLU FFN + DropPath(0→0.1) + AdamW betas (0.9, 0.95), wd=3e-5 + PhysicsAttention temperature init=1.0 + **cosine schedule with 1-epoch linear warmup + cosine T_max=13** + **feature noise std=0.005** + NaN-safe.
- **Mechanism**: late-epoch slope shallows from 4.3 → 0.7 pts/epoch as cosine LR decays toward zero — the model finally lands in a "fine-tuning regime" within the 14-epoch budget. Prior baselines were cut off mid-descent at 3–5 pts/epoch. PR #370's earlier T_max=14 attempt failed on EMA(0.99)+SwiGLU only because that stack saturated the fast-tracking benefit; the merged stack creates different basin geometry.
- **Caveat**: 67.306 was measured on slice-temp baseline + the LR schedule fix only (fern branched before wd=3e-5, warmup_steps=50, and feature-noise=0.005 merged). The current advisor stack adds those compounds on top. Combined-stack actual val_avg is expected at-or-better than 67.306; future runs will refine.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name cosine-warmup-tmax-aligned --agent <name>
  ```

## 2026-04-28 05:15 — Previous baselines (PRs #527 + #518 on top of #520)

- val_avg target stays at conservative 71.6985 (PR #520 directly measured) for cross-PR ranking.
- PR #527 (wd=3e-5): val_avg = 70.814 (−1.23%)
- PR #518 (warmup_steps=50): val_avg = 71.4284 (−0.38%)

## 2026-04-28 04:55 — Previous baseline (PR #520, slice-temp init 1.0)

- **Best `val_avg/mae_surf_p`**: 71.6985 (epoch 14)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **62.5824**
- **Per-split val MAE for `p`**:
  - val_single_in_dist: 85.482 (−2.77% vs 72.414 reference)
  - val_geom_camber_rc: 83.090 (−0.28%, flat)
  - val_geom_camber_cruise: 50.527 (+0.61%, flat)
  - val_re_rand: 67.694 (−0.74%)
- **Per-split test MAE for `p`**:
  - test_single_in_dist: 73.353 (−4.63% vs 63.082)
  - test_geom_camber_rc: 75.546 (+3.84% — only split going backward)
  - test_geom_camber_cruise: 41.805 (−0.83%)
  - test_re_rand: 59.625 (−1.46%)
- **Recipe**: huber(δ=0.25) + bias-corrected EMA(decay_target=0.99, warmup_steps=10) + SwiGLU FFN + DropPath(0→0.1) + AdamW betas (0.9, 0.95) + **PhysicsAttention temperature init=1.0** + NaN-safe. Single-token change vs prior baseline.
- **Mechanism**: final per-block temperatures converged to [0.95, 0.99] — the 0.5 init was ~2× below equilibrium. Init=1.0 starts at optimum, so gradients fit data instead of un-doing init. This is an optimization-warmup phenomenon, not a capacity one.
- **First directly measured combined-stack baseline.** Prior post-merge measurements (fern's 72.414 pre-DropPath, askeladd's 81.251, etc.) were on intermediate stacks; this run measures the full compound.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name slice-temp-1p0 --agent <name>
  ```

## 2026-04-28 04:15 — Previous baseline (PR #479, bias-corrected EMA)

- val_avg/mae_surf_p target: 72.414 (fern's δ=0.25 measurement, pre-DropPath)
- Bias-corrected EMA standalone: 81.251 (EMA(0.99)+SwiGLU pre-DropPath, −2.37% vs 83.223)

## 2026-04-28 03:55 — PR #480: AdamW betas (0.9, 0.95) — earlier orthogonal compound

- **β₂=0.95 standalone measurement**: 77.951 on EMA(0.99)+SwiGLU baseline (−6.34% vs 83.223).
- **Caveat**: post-merge baseline number not directly measured — both fern's δ=0.25 (72.414) and nezuko's β₂=0.95 (77.951) were measured on the same pre-DropPath, single-axis stack. The combined stack (δ=0.25 + DropPath + β₂=0.95 + ...) will be measured by round-6/7 PRs. Predicted combined val_avg: ~67–72 if levers compound additively, ~72 if partially redundant.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name post-betas095 --agent <name>
  ```

## 2026-04-28 03:25 — Previous baseline (PR #463, huber δ=0.25)

- **Best `val_avg/mae_surf_p`**: 72.414 (epoch 13, measured pre-DropPath; post-merge stack adds DropPath, expected at-or-better)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **63.082** (same caveat — pre-DropPath measurement; post-merge stack expected at-or-better)
- **Per-split val MAE for `p` (pre-DropPath measurement)**:
  - `val_single_in_dist`: 87.914 (−11.03% vs EMA(0.99)+SwiGLU baseline)
  - `val_geom_camber_rc`: 83.323 (−13.76%)
  - `val_geom_camber_cruise`: 50.222 (−17.88% — biggest split improvement)
  - `val_re_rand`: 68.199 (−10.62%)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 76.918
  - `test_geom_camber_rc`: 72.750
  - `test_geom_camber_cruise`: 42.154
  - `test_re_rand`: 60.506
- **Recipe**: huber(**δ=0.25**, was 1.0) + EMA(decay=0.99) + SwiGLU FFN + DropPath(0→0.1, last block kept) + NaN-safe `evaluate_split`. δ change is the only diff vs the post-DropPath state.
- **δ profile** (closes the question): δ=2 → 107.6, δ=1 → 88.2 (pre-EMA), δ=0.5 → 87.3 (pre-EMA), δ=1 → 83.2 (post-EMA), **δ=0.25 → 72.4 (post-EMA, this PR)**. Profile is monotone toward L1 with **non-diminishing returns** in this regime — δ=0.5→0.25 delivered ~10× more improvement than δ=1→0.5 did. The smaller quadratic region handles the heavy-tailed pressure error distribution; EMA(0.99)'s fast tracking compounds especially well with a more L1-like loss.
- **Note on measurement vs merged stack**: this val_avg was measured on EMA(0.99)+SwiGLU+huber(δ=0.25)+NaN-safe (no DropPath, since fern branched before DropPath merged). The current advisor branch now has DropPath added on top of fern's δ change. Since DropPath is a generic regularizer and orthogonal to loss reformulation, the merged compound is expected to be at-or-better-than 72.414. Round-7 PRs will measure the actual post-merge baseline.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-delta-025 --agent <name>
  ```

## 2026-04-28 03:10 — Previous baseline (PR #455, DropPath)

- **Best `val_avg/mae_surf_p`**: 80.480 (epoch 14, measured directly)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **72.328**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 92.907 (−5.98% vs EMA(0.99) baseline)
  - `val_geom_camber_rc`: 95.534 (−1.12%, near noise)
  - `val_geom_camber_cruise`: 57.237 (−6.41%)
  - `val_re_rand`: 76.241 (−0.08%, flat)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 85.502 (−4.82%)
  - `test_geom_camber_rc`: 85.565 (+1.38% — within noise)
  - `test_geom_camber_cruise`: 49.233 (−3.17%)
  - `test_re_rand`: 69.012 (−2.17%)
- **Recipe**: huber(δ=1.0) + EMA(decay=0.99) + SwiGLU FFN + **DropPath (per-block linear schedule 0→0.1, last block always kept)** + NaN-safe `evaluate_split` filter. Effective per-block drop = `[0.0, 0.025, 0.05, 0.075, 0.0]`. Param-identical to baseline (no new learnable parameters).
- **Mechanism (refined)**: DropPath acts as a generic regularizer here (val curve is parallel to baseline with a uniform offset, not a late-training kick). The OOD-camber splits (`val_geom_camber_rc`, `val_re_rand`) showed near-flat improvements — the gain is concentrated on `val_single_in_dist` (in-dist) and `val_geom_camber_cruise` (easier OOD). Implies the camber_rc / re_rand bottleneck is NOT implicit-ensembling-shaped; it's likely data-coverage / extrapolation-shaped.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name stochastic-depth-01 --agent <name>
  ```

## 2026-04-28 02:15 — Previous baseline (PR #426, EMA(0.99))

- **Best `val_avg/mae_surf_p`**: 83.223 (epoch 13)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **73.904**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 98.815 (−7.13% vs SwiGLU baseline)
  - `val_geom_camber_rc`: 96.612 (−3.78% vs SwiGLU baseline)
  - `val_geom_camber_cruise`: 61.160 (−5.04% vs SwiGLU baseline)
  - `val_re_rand`: 76.304 (−6.60% vs SwiGLU baseline)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 89.833
  - `test_geom_camber_rc`: 84.398
  - `test_geom_camber_cruise`: 50.843
  - `test_re_rand`: 70.541
- **Recipe**: huber(δ=1.0) + EMA(decay=0.99 — was 0.999) + SwiGLU FFN inside `TransolverBlock` + NaN-safe `evaluate_split` filter. All other defaults unchanged.
- **Mechanism**: under the default cosine `T_max=50` schedule with only 13 reachable epochs in the 30-min budget, EMA(0.999)'s ~1.85-epoch half-life means the EMA is heavily anchored to random-init weights for the first ~2 epochs. EMA(0.99) (half-life ~0.18 epochs) tracks the live model immediately, removing the bias-from-cold-start. Gain is uniform across all 4 val + test splits.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-decay-099 --agent <name>
  ```

## 2026-04-28 01:20 — Previous baseline (PR #391, SwiGLU FFN)

- **Best `val_avg/mae_surf_p`**: 88.227 (epoch 13)
- **`test_avg/mae_surf_p`** (paper-facing, all 4 splits finite): **78.338**
- **Per-split val MAE for `p`**:
  - `val_single_in_dist`: 106.398 (−15.78% vs EMA)
  - `val_geom_camber_rc`: 100.406 (−8.23% vs EMA — finally moved after being flat under EMA alone)
  - `val_geom_camber_cruise`: 64.409 (−16.34% vs EMA)
  - `val_re_rand`: 81.696 (−11.86% vs EMA)
- **Per-split test MAE for `p`**:
  - `test_single_in_dist`: 96.439
  - `test_geom_camber_rc`: 88.064
  - `test_geom_camber_cruise`: 54.011
  - `test_re_rand`: 74.837
- **Recipe**: huber(δ=1.0) + EMA(decay=0.999, eval+ckpt) + LLaMA-style SwiGLU FFN (gate × value, bias=False, intermediate=176) inside `TransolverBlock`. NaN-safe `evaluate_split` workaround active. All other defaults unchanged (lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, slice_num=64, n_layers=5, n_hidden=128, n_head=4, mlp_ratio=2). Param-matched (+1.3% n_params: 670K vs prior 660K).
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name swiglu-mlp --agent <name>
  ```

Loss curve was still descending monotonically at the 30-min cap — model is under-trained, more epochs likely give further gains.

## 2026-04-28 00:25 — Previous baseline (PR #363, EMA-eval)

- **Best `val_avg/mae_surf_p`**: 101.350 (epoch 14)
- **`test_avg/mae_surf_p`** (paper-facing): pending finite re-measurement on the EMA-merged baseline (cruise NaN here because PR #361 had not landed when this run started); **3-split test mean = 100.030** — `single_in_dist=113.32, geom_camber_rc=97.44, re_rand=89.33`.
- **Per-split val MAE for `p` (EMA, epoch 14)**:
  - `val_single_in_dist`: 126.323 (−5.76% vs huber)
  - `val_geom_camber_rc`: 109.406 (−0.07%, flat)
  - `val_geom_camber_cruise`: 76.988 (−6.93% vs huber)
  - `val_re_rand`: 92.682 (−5.19% vs huber)
- **Recipe**: huber(δ=1.0) loss in normalized space + EMA copy of weights (decay 0.999), checkpoint = EMA weights. All other defaults unchanged from the merged baseline.
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name ema-eval --agent <name>
  ```

## 2026-04-27 23:30 — Previous baseline (PR #282 + #361)

- **Best `val_avg/mae_surf_p`**: 105.999 (PR #282 huber-loss)
- **`test_avg/mae_surf_p`**: 97.957 (first finite measurement, PR #361 NaN-safe eval rerun)
- **Per-split val surface MAE for `p`**:
  - `val_single_in_dist`: 134.048
  - `val_geom_camber_rc`: 109.479
  - `val_geom_camber_cruise`: 82.718
  - `val_re_rand`: 97.751
- **Per-split val Ux / Uy / p (surface)**: see `research/EXPERIMENTS_LOG.md`
- **Model**: Transolver, 0.66M params, default config (`n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`).
- **Optimizer**: AdamW, lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10, epochs=50 (timeout-truncated at 14/50 epochs).
- **Loss**: Huber(δ=1.0) on normalized targets, applied identically in train and val/test eval.
- **Metrics path**: `models/model-charliepai2d2-edward-huber-loss-20260427-223516/{metrics.jsonl,metrics.yaml}`
- **Reproduce**:
  ```bash
  cd target
  python train.py --epochs 50 --experiment_name huber-loss --agent <name>
  ```

## 2026-04-28 00:10 — PR #361 follow-up: per-split test surface MAE for `p` (first finite test_avg)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| `test_single_in_dist`     | 123.760 | 1.737 | 0.746 |
| `test_geom_camber_rc`     | 104.946 | 2.090 | 0.877 |
| `test_geom_camber_cruise` |  66.144 | 0.959 | 0.480 |
| `test_re_rand`            |  96.978 | 1.532 | 0.706 |
| **avg**                   | **97.957** | **1.579** | **0.702** |

PR #361 added a 3-line filter in `train.py:evaluate_split` that drops samples with any non-finite `y` from the batch before calling `accumulate_batch`. The `data/scoring.py:accumulate_batch` Inf-times-0 propagation bug remains (file is read-only); the workaround triggers exactly once per test pass — on `test_geom_camber_cruise` sample 20 (761 non-finite `y[p]` volume nodes; surface `p` and Ux/Uy unaffected) — and is a no-op everywhere else.

## Ranking note

Future PRs are scored against `val_avg/mae_surf_p < 105.999` (recipe high-water mark from PR #282), **not** against the 108.103 RNG draw from PR #361. The val computation path on PR #361 is byte-identical to the merged recipe (the workaround does not trigger on any val sample); the +1.99% delta is purely run-to-run variance under a 14-epoch timeout-truncated training.
