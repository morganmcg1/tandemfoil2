# Research Ideas — TandemFoilSet Round 2

- Date: 2026-04-27
- Branch: `icml-appendix-willow-pai2c-r2`
- Primary metric: `val_avg/mae_surf_p` (equal-weight surface-pressure MAE across the four val splits, lower is better).
- Constraints: 1 GPU/student (96 GB), `epochs=50` cap, ~30 min wall clock per run, batch=4, no changes to `data/`.
- Round 1 already covers: vanilla seed anchor; capacity scale (n_hidden 192, n_layers 7); `surf_weight` ∈ {5, 20, 50}; Huber/L1 loss; lr=1e-3 + warmup + cosine; n_head=8/slice_num=128; per-channel reweighting; cruise y-flip augmentation. Ideas below are **disjoint** from those.

The ranking metric is dominated by surface-pressure MAE on three OOD axes (camber-rc, camber-cruise, Re-rand) plus an ID sanity check. Per-sample `y_std` varies by an order of magnitude (Re-driven), and the cruise hold-out has small magnitudes (~164 std) while raceCar hold-outs are ~377-458 — so a uniform MSE objective under-weights cruise pressure error in surface MAE. Ideas optimize for either (a) sharper surface-pressure recovery, (b) better OOD generalization (especially camber/Re extrapolation), or (c) more samples seen per wall-clock minute.

---

## 1. Pressure-aware target reformulation: signed log target for `p`

- **Hypothesis.** Predicting `sign(p) * log1p(|p|/p_scale)` for the pressure channel (with `p_scale ≈ y_std[2]`) compresses the heavy tail driven by high-Re samples and turns multiplicative pressure errors into additive ones. Predicted ∂(MAE_surf_p): −5 to −12% if cruise/low-Re samples currently sit in the floor of the loss.
- **Mechanism.** With raw `p` ranges spanning 4 orders of magnitude across Re and across cruise vs raceCar, the MSE gradient is dominated by the few highest-|p| samples; the cruise camber holdout has `y_std ~164` and is pressure-error-starved during training. log compresses the dynamic range, equalizing gradient contribution. Prior art: Karniadakis-style "log loss" in PINNs; Magnetic Fusion (Brandstetter et al., 2022) used asinh on stiff fields.
- **Implementation.** `train.py` loss block (~L487–500): apply `signed_log` only to channel 2 of `(pred − y_norm)` after model output, and convert back before `accumulate_batch`. Stats already include `y_mean/y_std` so denormalize first, then transform. Keep `Ux,Uy` in the linear MSE space.
- **Risk.** Introduces a non-linear forward/inverse pair that can numerically degrade for `p ≈ 0`. Choose `p_scale` ≥ 1.0 so `log1p` is well-conditioned. Could improve cruise but hurt raceCar high-|p| extremes — watch `mae_surf_p` per split.
- **Stretch.** Try `asinh(p / p_scale)` instead (smooth, antisymmetric, no clipping), and a learnable `p_scale` parameter trained jointly.

## 2. Sobolev / gradient-matching surface loss

- **Hypothesis.** Adding a surface-only finite-difference loss on the directional gradient `dp/ds` (along arc-length `saf`) produces a smoother predicted pressure profile and reduces `mae_surf_p` by 3-8%, especially on cruise where suction-side pressure peaks are sharper.
- **Mechanism.** Surface pressure is a 1D function of arc-length around each foil. Gradient supervision penalizes ringing/oscillation at the leading edge that pure pointwise MSE allows. Refs: Czarnecki et al., "Sobolev Training for Neural Networks" (NeurIPS 2017); Li et al., FNO papers using H^1 norms.
- **Implementation.** In `train.py` loss block, compute pairwise differences between adjacent surface nodes ordered by `saf` (feature dim 2-3). The mesh already has `is_surface` mask; `saf` provides ordering within each foil surface. Add a small-weight (e.g. 0.5) `(Δp_pred − Δp_true)^2` term over consecutive surface pairs.
- **Risk.** Requires sorting/grouping by foil within a batch — may add 5-15% per-step overhead. If misordered (saf signs flip across surfaces), the term can hurt. Validate on a single batch first by reading a saved `.pt` file.
- **Stretch.** Add a Laplacian-style 2nd-order term `Δ²p` and let the weight be tuned jointly via uncertainty weighting (idea #11).

## 3. Reynolds-conditioned modulation (FiLM)

- **Hypothesis.** Inject `log(Re)` (and optionally AoA) as a global FiLM (γ, β) modulation on every TransolverBlock's hidden state. Predicted ∂(MAE_surf_p): −4 to −10%, biggest win on `val_re_rand`.
- **Mechanism.** Currently `log(Re)` is just a per-node feature (dim 13) that gets squashed in the preprocess MLP. FiLM (Perez et al., AAAI 2018) gives the model explicit control to scale every feature by a Re-dependent factor — analogous to how the Reynolds number rescales viscous vs inertial terms in Navier-Stokes. PDE-Refiner and Geo-FNO use similar conditioning.
- **Implementation.** `train.py` Transolver class: build a small `cond_mlp(global_features) -> 2 * n_hidden * n_layers` from the constant-per-sample features (`log Re`, `AoA1`, `AoA2`, `gap`, `stagger`, `NACA1`, `NACA2`). Apply `fx = γ * fx + β` in `TransolverBlock.forward` after each LN. Global features extracted from `x[:, 0, 13:24]` (they're constant per sample by construction).
- **Risk.** Adds ~2-3% params; if conditioning network is too large it can overfit. Initialize γ to 1, β to 0 (warm start = identity).
- **Stretch.** Replace the addition `placeholder` with a per-sample learned token derived from the same conditioning MLP. Or use AdaLN-Zero (DiT-style) for stable scaling.

## 4. Ground-effect / signed-distance geometry feature

- **Hypothesis.** RaceCar samples have a ground (slip wall) below the foils, modeled implicitly through node positions. Adding an explicit signed distance `min_y - z_ground` (or `z_position` already in `x[:, :, 1]`) — and a foil-surface signed distance computed via the existing `dsdf` (dims 4-11) reduced to a scalar — strengthens the inductive bias for ground-effect physics. Predicted ∂(MAE_surf_p): −3 to −7% on raceCar splits.
- **Mechanism.** The model sees a distance-shape descriptor `dsdf` but no scalar "distance to nearest surface" or "distance to ground plane". Both are first-class geometric quantities for boundary-layer effects (cf. wall functions in turbulence modeling). Simple feature engineering can free the network from re-deriving these.
- **Implementation.** `train.py` data path inside training loop — augment `x_norm` with two extra channels: (a) `z` clamped at `[0, z_max]` for raceCar (use `gap > 0 OR feature 22 != 0` as raceCar indicator? actually inverted check needed — raceCar is single+P1+P2+P3 vs cruise via stagger/gap distribution; safer: derive from training stats). (b) `min(dsdf)` over the 8 dims as scalar wall distance. Modify model `space_dim=2 → 4`.
- **Risk.** Increases input dim → first MLP grows. Stats pipeline assumes `X_DIM=24` constant — augment in `train.py` after normalization, not in `data/`. Risk of doubling features that are redundant with `dsdf`.
- **Stretch.** Compute on-the-fly `(x, z)` Eulerian distance to the foil surfaces by gathering surface-node positions per sample (one-time per batch on GPU), then concat as feature.

## 5. Importance sampling for high-|p| / surface-rich samples

- **Hypothesis.** A weighted sampler over the training set, where weight ∝ per-sample max-surface-|p| or per-sample y_std, focuses training on the high-stakes simulations (high Re, sharp pressure peaks). Predicted ∂(MAE_surf_p): −3 to −6%, primarily on `val_re_rand`.
- **Mechanism.** Current sampler equalizes by domain (raceCar single / raceCar tandem / cruise) but not by difficulty. The MAE score is dominated by samples with the largest pressure peaks — so concentrating gradient steps on those gives more bang per step. Cf. "hard example mining" in detection; Loshchilov & Hutter 2015 "Online Batch Selection".
- **Implementation.** `train.py` ~L398: extend `sample_weights` to multiply the existing domain weight by a difficulty score precomputed once at startup (loop through training set, compute `y[is_surface, 2].abs().max()` per sample, normalize to mean=1.0). Pass new weights to `WeightedRandomSampler`.
- **Risk.** Could starve easy samples and worsen ID metrics. Cap weight ratio at, say, 4×. Pre-pass takes ~1-2 min — fine.
- **Stretch.** Train two models — uniform and importance — and ensemble (free lunch from idea #10).

## 6. Stochastic depth / DropPath in TransolverBlock

- **Hypothesis.** Add per-block stochastic depth (linearly increasing rate from 0 to 0.1 across layers) to combat overfitting on the small (~1500 sample) training set, especially helping the camber hold-outs. Predicted ∂(MAE_surf_p): −2 to −5%, biggest on `val_geom_camber_*`.
- **Mechanism.** Regularization for transformers (Huang et al., 2016; ConvNeXt 2022) is consistently a small but additive win on small-data regimes. Camber hold-out is the cleanest test of regularization since the network has memorized which cambers it has seen.
- **Implementation.** `train.py` `TransolverBlock.forward`: wrap each residual branch with `if self.training and random.random() < drop_path: skip` or use `timm.layers.DropPath`. Add `drop_path_rate: float = 0.1` to `Config`.
- **Risk.** Might slow convergence in the 30-min budget; combine with longer effective training via gradient accumulation? Keep rate small (≤0.1).
- **Stretch.** Combine with weight EMA (idea #9) — stochastic depth + EMA is a Kaggle-grade combo.

## 7. Mixup at the sample level (geometry-aware blend)

- **Hypothesis.** Two random training samples within the same domain blended `α x_i + (1-α) x_j` with the same `α` on `y` — giving the model intermediate shapes — should improve the camber hold-out where it must interpolate. Predicted ∂(MAE_surf_p): −2 to −5% on `val_geom_camber_*`.
- **Mechanism.** Mixup (Zhang et al., ICLR 2018) regularizes by enforcing local linearity in input. For CFD this is questionable physically (NS is non-linear) but at the **feature level** — between two geometries with similar (Re, AoA), the predicted velocity/pressure fields are roughly affine for small geometric perturbations near linearization. Tractable for camber interpolation specifically.
- **Implementation.** `train.py` training loop: with prob 0.3, sample two batches, pad to common length, blend `x` (only the geometric features 4-11 + 15-23, leaving positional features 0-3 untouched would be safest) and `y`. Use `mask_i & mask_j` as effective mask.
- **Risk.** Blending positions doesn't make physical sense; restrict to global-conditioning channels (NACA, gap, stagger). Cruise hold-out cambers M=2-4 are blends of M=0-2 and M=4-6 which **are in train** — promising. RaceCar hold-out M=6-8 sits between M=2-5 and M=9 which is also in train.
- **Stretch.** "CutMix" variant: keep raceCar P1 mesh but swap NACA conditioning to P3's — see if the conditioning carries.

## 8. Schedule-Free or Lion optimizer

- **Hypothesis.** Replace AdamW + cosine schedule with Schedule-Free AdamW (Defazio et al., 2024) — it tracks the exponential moving average internally and removes the need for a learning-rate schedule, often delivering a free 1-3% improvement and handling early-stopping more gracefully. Predicted ∂(MAE_surf_p): −2 to −5%.
- **Mechanism.** Schedule-Free implicitly averages parameters along the trajectory, similar to SWA but online. With our 30-min wall clock cap that often cuts the schedule mid-cosine, removing the schedule sensitivity is a win.
- **Implementation.** `train.py` ~L434: replace `AdamW + CosineAnnealingLR` with `schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)` from the `schedulefree` package (add to `pyproject.toml`). Call `optimizer.train()` / `optimizer.eval()` at training/eval boundaries.
- **Risk.** Requires correct train/eval mode toggling; adds 1 dependency. Tune lr — Schedule-Free often wants slightly lower lr than AdamW.
- **Stretch.** Try Lion (Chen et al., NeurIPS 2023) which uses sign(grad·momentum), often 30% lower memory and competitive on mid-size models.

## 9. Weight EMA + best-EMA checkpoint selection

- **Hypothesis.** Maintain an EMA of model parameters with decay 0.999 (updated every step) and select the best checkpoint by `val_avg/mae_surf_p` evaluated on the **EMA weights** rather than the raw model. Predicted ∂(MAE_surf_p): −1 to −4%.
- **Mechanism.** Polyak averaging / SWA reduces noise in the loss landscape; EMA models routinely outperform raw weights at convergence (cf. Diffusion model practice, BYOL, MAE). With a 30-min cutoff, EMA gives a smoother trajectory and can compensate for noisy late-epoch validation steps.
- **Implementation.** `train.py` after `optimizer.step()`: maintain `ema_model = clone(model)` updated as `ema = 0.999 * ema + 0.001 * model`. At validation, evaluate both, pick the better. Save the better as the checkpoint.
- **Risk.** Doubles model VRAM footprint (~2× param memory; activations dominate, so manageable). EMA with very short training can lag behind — start EMA after epoch 5.
- **Stretch.** Combine with stochastic depth (#6) and DropPath — both are well-known to compound.

## 10. Checkpoint averaging / model souping (post-hoc, no extra training)

- **Hypothesis.** Average the weights of the top-K checkpoints (by val score) saved during a single training run. Predicted ∂(MAE_surf_p): −1 to −3%, free.
- **Mechanism.** Wortsman et al. "Model Soups" (ICML 2022) showed that averaging fine-tuned models in weight space yields better generalization than any individual model. Within a single run, the last few epochs around the best checkpoint sit in the same loss basin — averaging smooths them.
- **Implementation.** `train.py` save the top-3 checkpoints by val score (rolling top-K). After training, average them and re-evaluate. If better, save the soup as the artifact.
- **Risk.** Almost none — it's evaluation-time only. Need extra disk for K checkpoints.
- **Stretch.** Across-run averaging — soup multiple independent seeds (would require coordination; see #12).

## 11. Uncertainty-weighted multi-task loss (Kendall et al.)

- **Hypothesis.** Replace the hand-tuned `surf_weight=10` with learnable log-variance parameters for each of {volume, surface} × {Ux, Uy, p}, computing the loss as `Σ exp(-σ_k) L_k + σ_k`. Predicted ∂(MAE_surf_p): −2 to −4%.
- **Mechanism.** Kendall, Gal & Cipolla, CVPR 2018 — automatic balancing of multi-task losses. Removes the surf_weight hyperparameter entirely. With six effective sub-losses (3 channels × 2 regions), hand tuning is brittle; learning the weights gives the model the right trade-off automatically.
- **Implementation.** `train.py`: add `log_var = nn.Parameter(torch.zeros(6))` to a small wrapper, compute `loss = sum(exp(-log_var[k]) * L_k + log_var[k] for k in 0..5)`. Initialize so initial weights match current `surf_weight=10`.
- **Risk.** Could collapse one channel to zero if not initialized carefully. Use `softplus(σ)` for non-negativity. May interact poorly with surf_weight sweep results from round 1 — wait for those before merging.
- **Stretch.** Per-split learnable weights — add 4 more parameters keyed by domain group at training time.

## 12. Multi-seed ensembling at test time (with mesh-conditional TTA)

- **Hypothesis.** Train 3-5 independent seeds with diverse hyperparameters (different lr / loss / random init), then ensemble predictions at test time. Predicted ∂(test_avg/mae_surf_p): −5 to −12%, biggest single empirical win available, but expensive.
- **Mechanism.** Standard Kaggle ensemble. Each model lives in a different basin; averaging predictions in physical space reduces variance more than bias. The 30-min cap limits a single run, so ensembles also serve as a soft compute-budget extension.
- **Implementation.** Multi-PR coordination (or a single PR with `--seed` and `--n_seeds`). Save predictions to W&B artifacts; build a small offline script (in `train.py` end-of-run) that loads peer artifacts via wandb API and averages. Note: ensembling is a meta-experiment that builds on round-1 winners — frame the PR as "select N best checkpoints across recent runs and ensemble them post-hoc".
- **Risk.** Increases student wall-clock by N×; may not fit in our advisor budget. Best framed as a single PR that loads K artifacts from wandb and reports ensembled metrics — no additional training.
- **Stretch.** Test-time augmentation: predict on (x, +y_flip(x)) and average back (only valid for cruise where AoA is symmetric around 0; sign-flip Uy and AoA accordingly). Could be combined with seed-ensemble for compounding gains.

---

## Cross-cutting notes

- **Order of attack.** Ideas #1, #2, #3 attack the loss/architecture inductive bias directly and are likely the highest-impact; #6, #9 are cheap regularization wins; #10 is a free post-hoc trick. #12 is the bigger swing but requires multiple seeds in flight.
- **Compounds well.** {#1 or #2} + #3 + #6 + #9 + #10 is a plausible "kitchen-sink" Kaggle stack — once each is independently validated, a final consolidation PR could merge them.
- **Drop these if a seed of doubt arises.** #7 (mixup) is most physics-questionable; gate it behind cruise camber hold-out improvement on a small pilot. #4 (extra geometry features) overlaps with `dsdf` — check redundancy first.
- **What's NOT here.** Graph attention (heavy compute, doesn't slot into Transolver cleanly at our mesh sizes), full FNO (changes the data interface), distillation (no large teacher available yet — would need a 2-stage program). Defer to round 3+ once the architectural ceiling here is mapped.
