# Research Ideas — Round 1 — 2026-04-27 (initial)

**Branch:** `icml-appendix-willow-r1` (fresh)
**Vanilla anchor on this branch:** val_avg/mae_surf_p ≈ 88 (baseline `train.py` defaults, MSE+sw=10).
**Proven recipe (kagent_v_students):** val ≈ 49.4 / test ≈ 42.5 with L1+sw=1 + AMP+grad_accum=4 + Fourier PE σ=0.7 m=160 + FiLM(log Re) + SwiGLU FFN + nl=3 + slice_num=8.

The branch is FRESH — none of those merged improvements have been ported yet. This means R1 has a dual mandate: (a) ship the proven recipe in stages so the GPUs are not idle re-discovering known wins, and (b) genuinely *new* directions that compound on top of it. Below I prioritize ideas in the second category — fresh hypotheses that have not been tried in prior research, ranked by expected impact and ease of stacking with the winning recipe.

The dominant theme last round was **compute reduction → more epochs in the 30-min budget → lower val_mae_surf_p**. Every winning config was budget-bound (best val at terminal epoch). Anything that frees per-step compute or accelerates convergence is high-value here. Anything that *adds* per-step compute needs to be radically more sample-efficient or it loses the epoch race.

---

## Top-priority FRESH ideas (build directly on the winning recipe)

### 1. torch.compile + FlashAttention-2 explicit + (optional) bf16 weight storage
- **Predicted Δ on val_avg/mae_surf_p:** −2 to −6 % beyond winning recipe (49.4 → ~46–48) via 1.3–1.8× more epochs.
- **Mechanism:** torch.compile fuses the einsum + softmax + attention + MLP graph; PyTorch 2.x's SDPA already picks an SDPA backend but explicit `attn_backend="flash"` or `xformers.memory_efficient_attention` on the slice-token attention removes one Python branch and unlocks Flash kernels. bf16 weight storage halves model memory and lets us push batch size or n_hidden later without OOM. The slice attention runs over `[B, H, slice_num=8, D]` — tiny — so the *real* speedup is in `in_project_x`, `in_project_fx`, MLP, and the two einsums (`bhnc,bhng->bhgc` and inverse), all of which `torch.compile` should fuse aggressively.
- **Implementation outline:**
  1. After model construction: `model = torch.compile(model, mode="max-autotune", fullgraph=False)`. Add `--torch_compile` CLI flag (default off so we can A/B).
  2. Wrap the slice-token attention with explicit `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=False)` inside an `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context. Currently it already uses SDPA, but pinning Flash and disabling math/mem-efficient fallback removes branchy dispatch.
  3. On a debug run, log epoch_time_s and verify ≥ 1.4× speedup on the per-epoch wall clock.
- **Builds on:** winning recipe (must include AMP). Risk if torch.compile introduces graph-breaks on variable mesh sizes; use `dynamic=True` and benchmark.
- **Risk/effort:** med risk (graph breaks on dynamic shapes), low effort (∼20 LOC).

### 2. Lion / AdEMAMix / Sophia optimizer + scheduled-free training
- **Predicted Δ:** −1 to −5 % (49.4 → 47–48.5) via faster early descent and better terminal val under wall-clock.
- **Mechanism:** AdamW is the default but suboptimal in budget-bound regimes. Lion (Chen et al. 2023) uses sign-momentum and converges faster on small models per FLOP. AdEMAMix (Pagliardini 2024) maintains two EMAs and doesn't need warmup. Sophia (Liu 2024) approximates a diagonal Hessian and is reported to converge in fewer steps. Most relevant for us: a *Schedule-Free* AdamW (Defazio 2024) avoids the cosine sweet-spot brittleness — every winner hit best val at terminal epoch, meaning cosine annealing under-decays at the end. SF-AdamW dispenses with the schedule entirely.
- **Implementation outline:**
  1. Add `--optimizer {adamw, lion, ademamix, sophia, sf_adamw}` CLI flag.
  2. For Lion, depend on `lion-pytorch` (or copy 30-line implementation; no new pkg). For SF-AdamW, depend on `schedulefree` (PyPI). For AdEMAMix and Sophia, copy reference implementations into `train.py`.
  3. Tune LR per-optimizer: Lion typically wants 3–10× lower LR than AdamW; SF-AdamW uses `lr=1e-3` with momentum-warmup. Sweep small grid.
  4. CRITICAL: SF-AdamW removes the cosine sched — disable it under that flag.
- **Builds on:** winning recipe (AdamW lr=5e-4 → swap optimizer, sweep LR).
- **Risk/effort:** low risk (drop-in optimizers), low effort (~50 LOC + sweep).

### 3. Gradient-magnitude (Sobolev) loss on surface pressure
- **Predicted Δ:** −2 to −7 % (49.4 → 46–48), largest lift on `val_geom_camber_*`.
- **Mechanism:** Surface pressure is *highly correlated along the chord*. The current L1 loss is point-wise; a model can lower L1 while having wiggly per-node errors that don't cancel — the *integrated* surface MAE (the ranking metric) penalizes that. A Sobolev / gradient-matching term `||∇_arclength p_pred − ∇_arclength p_true||_1` on surface nodes pushes the model to predict the *shape* of the pressure curve, not just per-node values. This is exactly what airfoil-pressure work in Mishra et al. (2021) and Kashefi et al. (2021) report helps. The arclength gradient can be approximated using `saf` (signed arc-length, dim 2) — sort surface nodes by saf within each foil, compute finite differences. We don't need a perfect mesh derivative; the difference operator over the saf-sorted surface nodes is enough.
- **Implementation outline:**
  1. At the loss step, identify surface nodes per sample, sort by `x[:, :, 2]` (saf), compute `∇p = pred_p[i+1] - pred_p[i]` and same for ground truth.
  2. Add aux loss `λ_grad * L1(∇p_pred, ∇p_true)` (default `λ_grad=0.1`).
  3. Sweep `λ_grad ∈ {0.01, 0.1, 0.3, 1.0}`. Also test on Ux, Uy as bonus.
  4. Skip foils with < 4 surface nodes (degenerate).
- **Builds on:** winning recipe.
- **Risk/effort:** low risk (loss aux), med effort (~80 LOC for arclength sort and gradient).

### 4. Equivariance via chord-line frame (SE(2) symmetry)
- **Predicted Δ:** −2 to −8 % on `val_geom_camber_*` and `val_re_rand`.
- **Mechanism:** The physics is (approximately) invariant to a rotation that aligns the chord line with the x-axis. The current model sees AoA as a scalar feature and has to learn the rotation transformation implicitly. Hard-coding SE(2) equivariance via a *change of frame*: rotate every node coordinate `(x, z) → R(-AoA) · (x, z)` so the chord is always aligned, AND rotate the velocity output (Ux, Uy) → R(AoA) · (Ux, Uy) at the end. Pressure is rotation-invariant. This collapses an entire continuous symmetry, immediately reducing the effective input variation by an O(10°) range. For tandem cases with two AoAs, use the front foil's AoA as the canonical frame (or test averaging the two).
- **Implementation outline:**
  1. Pre-loss: rotate `(x, z)` and `(saf_x, saf_y)` (if present in dsdf) by `-AoA1`. Don't rotate dsdf magnitudes (rotation-invariant).
  2. Post-prediction: rotate predicted `(Ux, Uy)` back by `+AoA1`.
  3. At normalization: `x_mean`, `x_std` should be re-computed in the rotated frame, OR re-use existing stats and accept a small calibration loss (the rotation is ≤10° in raceCar so x and z are not strongly mixed).
  4. Test as a strict augmentation OR as a hard transform (compare both).
- **Builds on:** winning recipe. Note this *replaces* the implicit AoA learning — drop AoA dim from x or keep it as a helper signal.
- **Risk/effort:** med risk (frame rotation must be consistent for tandem), med effort (~150 LOC, plus careful test on geom_camber splits).

### 5. SDF + surface normal + curvature features (geometric injection)
- **Predicted Δ:** −2 to −5 %, largest lift on OOD camber and `val_re_rand`.
- **Mechanism:** dsdf is a distance-based descriptor but doesn't encode the *normal direction* or *local curvature* of the foil. Adding signed distance to the nearest surface plus the unit normal `(n_x, n_z)` and unsigned curvature `κ` gives the model direct boundary-layer geometry. For a non-surface node these features can be computed at preprocessing once per sample (the foil shape is fixed per sample). Actually, dsdf is 8-dim but the minimum coordinate of |dsdf| as in PR #21 was used. Here we want SIGNED distance + the gradient of the SDF (which IS the normal). We can compute both at training time: `sdf = x[:, :, 4:12].abs().min(-1).values` (signed by `is_surface` proximity) AND `normal = autograd-style finite-diff` (skip — too expensive at runtime; prefer pre-computed at __getitem__ time). Cleanest path: compute one new tensor per sample on the fly from `x[:, :, 4:12]` — namely, the *sign* of each dsdf component and a few softmin-weighted directions.
- **Implementation outline:**
  1. Add `--use_geom_features` flag.
  2. Compute `sdf_min = x[:, :, 4:12].abs().min(-1)`, `sdf_argmin`, `sdf_sign = sgn(min)`. These together approximate signed distance + nearest-surface direction.
  3. Optionally add `softmin(|dsdf|, β=10)` to get a smoothed nearest-surface vote.
  4. Concatenate as 4 extra features on `x` (recompute `fun_dim`).
- **Builds on:** winning recipe.
- **Risk/effort:** low risk (additive feature), low effort (~30 LOC).

### 6. Spectral / FFT loss along chord (paper-quality pressure-curve fidelity)
- **Predicted Δ:** −1 to −4 %, with disproportionate effect on test (which weighs surface integrals more uniformly).
- **Mechanism:** Surface pressure on a 2D airfoil has a characteristic Fourier spectrum along the arclength. A per-sample FFT magnitude + phase loss on `(p_pred, p_true)` along arclength forces the model to match the spectral signature, which discourages high-frequency wiggles and missing low-frequency stagnation/peaks. Cheap (FFT on ~300 surface nodes per foil per sample). Different mechanism from #3 (gradient-matching is local; FFT loss is global spectral).
- **Implementation outline:**
  1. Sort surface nodes by saf per sample per foil.
  2. Pad to `N_max` and compute `fft(p_pred)` and `fft(p_true)`.
  3. Loss: `λ_fft * L1(|fft|_pred, |fft|_true)` on the lowest K frequency bins (K=16).
- **Builds on:** winning recipe. Stack with #3 (orthogonal mechanism).
- **Risk/effort:** med risk (sort by saf needs care for split foils), med effort (~100 LOC).

### 7. RMSNorm + ReZero residuals (architecture norm/residual change)
- **Predicted Δ:** −1 to −3 %.
- **Mechanism:** LayerNorm has a learned bias term that interacts poorly with batch-1-style attention; RMSNorm (Zhang & Sennrich 2019) drops it, halving norm cost and slightly improving training stability on small models. ReZero (Bachlechner 2020) replaces residual `x + f(x)` with `x + α · f(x)` where α is a learned scalar initialized to 0. This is effectively pre-norm starting from identity, often improves fast convergence and removes the need for warmup. Together: RMSNorm + ReZero is what modern smaller transformers use.
- **Implementation outline:**
  1. Replace `nn.LayerNorm` with a 30-line RMSNorm implementation (`x / sqrt(mean(x**2) + eps) * g`).
  2. In `TransolverBlock.forward`, change `fx = self.attn(self.ln_1(fx)) + fx` to `fx = fx + self.alpha_attn * self.attn(self.ln_1(fx))` where `alpha_attn` is `nn.Parameter(torch.zeros(1))`. Same for the MLP residual.
  3. Add `--norm_type {layernorm, rmsnorm}` and `--rezero` flags.
- **Builds on:** winning recipe (replace at architecture level — minimal disturbance).
- **Risk/effort:** low risk, low effort (~50 LOC). 2-seed verify.

### 8. Sinusoidal activations (SIREN-style) in the preprocess + MLP head
- **Predicted Δ:** −1 to −3 %, largest lift on OOD camber.
- **Mechanism:** The current preprocess MLP uses GELU. For coordinate inputs, sinusoidal activations (Sitzmann 2020 SIREN) represent high-frequency structure with fewer parameters and converge faster. Use `sin(ω₀ · Wx + b)` only in the *first two* preprocess layers (deep SIREN is hard to train stably). The Fourier PE already gives high-frequency input features, but a sin activation in the preprocess layer keeps them coherent through the first nonlinearity. Initialization: SIREN's specific scheme (uniform `(-√6/n, √6/n)` for hidden, `(-1/n, 1/n)` for first).
- **Implementation outline:**
  1. New activation `Sine(ω₀=30)` class in `train.py`.
  2. Add `--preprocess_act {gelu, sine}` flag. When `sine`, use SIREN init for the preprocess MLP only.
  3. Keep main blocks as-is (don't touch the FFN inside transformer blocks).
- **Builds on:** winning recipe. Compounds with #3 and #6 cleanly.
- **Risk/effort:** med risk (init must be right), low effort (~40 LOC).

---

## Second-tier ideas (still fresh, slightly higher effort or smaller predicted lift)

### 9. Test-time augmentation (TTA) via mesh perturbation + horizontal flip ensemble
- **Predicted Δ:** −0.5 to −2 % at zero training cost, just inference cost.
- **Mechanism:** At test/val time, evaluate the model on N=4 perturbations: original, hflip-AoA-mirrored (with output Uy sign-flip — note PR #15 showed this *hurts training* but it can still help *inference* via averaging), and small node-position jitter (σ=0.001). Average the predictions in physical space. This is the cheapest free lift available and is orthogonal to all training changes.
- **Implementation outline:** Wrap the eval loop. Add `--tta_n 4` flag. Average pred_phys across views before MAE accumulation.
- **Builds on:** winning recipe. Pure inference.
- **Risk/effort:** low risk, low effort (~50 LOC). Could be the FIRST PR after winning recipe is shipped.

### 10. Distillation from larger teacher (or ensemble) into compact student
- **Predicted Δ:** −2 to −6 %.
- **Mechanism:** Train a teacher with bigger n_hidden (256) for fewer epochs OR ensemble 2 winning-recipe runs into a teacher; distill into a student with the winning compact recipe (nl=3, sn=8). Distillation loss: `L1(student_pred, teacher_pred)` + small L1 to ground truth. Distillation typically gives the student the convergence advantage of a bigger model in a smaller-budget run. The dataset is small (1500 train samples) — perfect for distillation.
- **Implementation outline:**
  1. Phase 1: load a saved teacher checkpoint OR train a fresh teacher.
  2. Phase 2: student loop computes both teacher and student forward; loss = `α · L1(student, gt) + (1-α) · L1(student, teacher)`.
  3. Sweep α ∈ {0.3, 0.5, 0.7}.
- **Builds on:** winning recipe (student) + a teacher trained off-budget (e.g. on a separate GPU at 60 min, or a cached artifact).
- **Risk/effort:** med risk (teacher artifact must be loaded), med effort (~120 LOC).

### 11. Multi-resolution / multi-scale features via adaptive node pooling
- **Predicted Δ:** −1 to −4 %.
- **Mechanism:** The mesh has up to 3 zones of varying density. The single-scale slice attention treats them all the same. A multi-scale variant: at each block, also pool nodes into a coarse representation (k-means in feature space, or top-K by mesh density), apply attention there, scatter back. This is structurally similar to a U-Net, which has been used in CFD work (Pfaff 2020 MeshGraphNets). Using slice attention itself as the pooling avoids the need for full graph operators.
- **Implementation outline:** Add a coarse slice-attention pathway with slice_num=2 in parallel to the main slice_num=8 pathway. Sum outputs.
- **Builds on:** winning recipe.
- **Risk/effort:** med risk (param count grows), med effort (~150 LOC).

### 12. Per-block FiLM(log Re) plus AoA + gap + stagger conditioning
- **Predicted Δ:** −1 to −3 % on cross-regime splits.
- **Mechanism:** PR #7's FiLM is at the input only. Per-block FiLM (used in MetaFormer) injects the conditioning at each block, similar to per-block Fourier (which failed). The *failure* mode of per-block Fourier was that depth-distributed signal dilutes the input signal. Per-block FiLM is different: the conditioning is a *scalar→vector* gating, not new spatial information. We also add the *full conditioning vector* (log Re, AoA1, AoA2, gap, stagger) instead of just log Re, exploiting the fact that AoA + geometry are continuous parameters varied across splits.
- **Implementation outline:**
  1. Add 5-dim conditioning vector `c = [log Re, AoA1, AoA2, gap, stagger]` (already exists per-sample).
  2. Per-block FiLM: `(γ_i, β_i) = MLP_i(c)`, applied after each TransolverBlock's LN.
  3. Sweep block-share: shared MLP across blocks vs. per-block.
- **Builds on:** winning recipe. **Important novelty vs PR #7:** uses ALL conditioning, not just log Re; per-block injection is gating not new info.
- **Risk/effort:** med risk (parameter growth), med effort (~80 LOC).

### 13. Bernoulli-coherent regularizer (physics prior on surface pressure)
- **Predicted Δ:** −1 to −4 %, largest on `val_re_rand`.
- **Mechanism:** Bernoulli's equation says along an inviscid streamline `p + ½ρ |U|² = const`. On the foil surface (no-slip, U=0 in viscous, but tangential U exists in inviscid limit), we can use the *integrated pressure coefficient* identity: `Cp = 1 - (U_tangential / U_∞)²`. Build a soft regularizer that penalizes when predicted pressure violates Bernoulli relative to predicted velocity in the *near-surface* layer (a few cells out from the surface). The penalty doesn't have to be exact — it just biases predictions toward physically coherent fields. Different from PR #21 which weighted near-surface volumes; this *constrains* the relationship between p and U via Bernoulli.
- **Implementation outline:**
  1. For each surface node, find nearest 3 volume nodes (use dsdf-min as proxy).
  2. Compute `Cp_pred = 1 - (U_pred_tangent / U_∞)²` at near-surface volume nodes, vs `Cp_at_surface_pred`.
  3. `λ_bern * L1(Cp_pred_volume, Cp_pred_surface)`.
- **Builds on:** winning recipe.
- **Risk/effort:** high risk (defining "near-surface" cleanly), high effort (~200 LOC).

### 14. Schedule-Free training + extended-budget compounding
- **Predicted Δ:** −1 to −3 %.
- **Mechanism:** Already mentioned in #2 but worth its own slot. Defazio's Schedule-Free SGD/AdamW *removes* the LR schedule and instead uses iterate-averaging. Reported to match cosine without any schedule tuning, AND it's well-suited for budget-bound training where you don't know the terminal step ahead of time.
- **Implementation outline:** As in #2 but isolate the schedule-free comparison from optimizer choice. Just SF vs cosine on AdamW.
- **Builds on:** winning recipe.
- **Risk/effort:** low risk, low effort.

### 15. Gradient checkpointing + 2× n_hidden (fit a bigger model in same VRAM, same epoch count)
- **Predicted Δ:** −0 to +5 % (could go either way).
- **Mechanism:** With gradient checkpointing on each TransolverBlock, the activation memory per block is recomputed on backward. This frees enough VRAM that we can double n_hidden from 128→256 without OOM. The capacity expansion failed previously (PRs #4, #16) — but those didn't pair with checkpointing or the modern compact recipe. The hypothesis is that with the winning recipe stable, capacity expansion *might* now be useful if the cost is paid via wall-clock (extra recompute), not memory. Risk: more wall clock per epoch → fewer epochs → worse val. Test as a one-shot probe.
- **Implementation outline:** `torch.utils.checkpoint.checkpoint(block, fx)` in the model forward. Add `--grad_ckpt` and `--n_hidden 256` flags.
- **Builds on:** winning recipe. **Caution:** could regress; run as a single seed probe first.
- **Risk/effort:** med risk, low effort (~20 LOC).

### 16. Ensembling via stochastic weight averaging (SWA)
- **Predicted Δ:** −1 to −3 % (almost free).
- **Mechanism:** Average model weights from the last K epochs of training. Equivalent to a cheap ensemble. PR #8 (EMA) is similar but EMA decays old weights; SWA uses a uniform window. SWA was *not* tried previously. Cheap to add; eval uses SWA weights and we save a separate SWA checkpoint.
- **Implementation outline:**
  1. After epoch `> swa_start`, accumulate `swa_model = mean(swa_model, current_model)` with running update.
  2. Final eval uses SWA weights. (Note: requires re-computing batch-norm stats; we don't have BN, so this is straightforward.)
  3. `--swa_start_epoch 25` typical.
- **Builds on:** winning recipe.
- **Risk/effort:** low, low.

---

## Quick map of which ideas COMPOUND vs CONFLICT with the winning recipe

| Idea | Stacks on winning recipe? | Conflicts with? |
|------|---------------------------|-----------------|
| 1. torch.compile + Flash | YES (purely throughput) | None |
| 2. Lion / SF-AdamW | YES (replaces optimizer) | Cosine sched if SF |
| 3. Sobolev/grad loss | YES (loss aux) | None |
| 4. SE(2) frame | YES (input transform) | Hflip TTA |
| 5. SDF features | YES (input add) | None |
| 6. Spectral loss | YES (loss aux) | None |
| 7. RMSNorm + ReZero | YES (norm/residual swap) | LayerNorm |
| 8. SIREN preprocess | YES (preprocess swap) | None |
| 9. TTA (hflip ens.) | YES (eval-only) | None |
| 10. Distillation | YES (extra loss) | None |
| 11. Multi-res pool | YES (extra branch) | None |
| 12. Per-block FiLM | YES (block-level) | None |
| 13. Bernoulli reg | YES (loss aux) | None |
| 14. SF training | YES | Cosine sched |
| 15. Grad ckpt + 2× n_hidden | RISKY (slows wall-clock) | Compute reduction theme |
| 16. SWA | YES (eval) | None |

---

## Recommended round-1 assignment matrix

We have **8 students** and need to keep all GPUs busy. R1 should *both* re-establish the winning recipe AND start novel experiments. The proven recipe is so dominant (val 88 → val 49) that the first 2–3 PRs should ship it as a single bundle, but in parallel we can launch novel ideas that *don't* depend on the recipe being merged (because they're orthogonal).

**Recommended R1 PR slate:**

1. **`port-winning-recipe`** (PR 1): single PR shipping L1 + sw=1 + AMP + grad_accum=4 + Fourier PE σ=0.7 m=160 + FiLM(log Re) + SwiGLU + nl=3 + slice_num=8 → expected val ~49.4. This is a recipe-bundle PR, not a single-hypothesis PR — exception justified because the prior research is documented.
2. **`torch-compile`** (Idea #1): one student tries torch.compile on top of recipe.
3. **`lion-ademamix-sf`** (Idea #2): another student sweeps optimizers.
4. **`sobolev-grad-loss`** (Idea #3): chord-arclength gradient loss aux.
5. **`se2-frame`** (Idea #4): rotate to chord-line frame.
6. **`sdf-normal-features`** (Idea #5): geometric extra inputs.
7. **`rmsnorm-rezero`** (Idea #7): norm/residual swap.
8. **`sf-training`** (Idea #14): Schedule-Free vs cosine.

R2 (after winners merge): SIREN, distillation, multi-res, per-block FiLM, TTA, SWA, Bernoulli, FFT loss.

---

## Notes on priors and previously failed directions

We DID NOT re-suggest:
- capacity scaling up (n_hidden, n_layers) — **failed**, except in the controlled grad-ckpt probe.
- target reparam (asinh) — **failed**.
- channel-weighted aggressive surf_p — **failed**.
- horizontal flip with Uy sign-flip in *training* — **failed** (but allowed in TTA per #9).
- input feature jitter — **failed**.
- per-block Fourier reinjection (zero-init or otherwise) — **failed**.
- attention temperature annealing — **failed**.
- sample-wise renorm with Re-scale — **failed**.
- near-surface volume-band weighting (3-tier) — **failed**.
- alpha-gated PBF — **failed**.
- cross-attention surface decoder, slice-bottleneck decoder, zero-init residual decoder — **failed**.

This list represents ~10 failed mechanisms — useful to know what *not* to retry on the same axis.
