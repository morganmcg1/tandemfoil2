# Opening-Round Research Ideas — TandemFoilSet (icml-appendix-charlie-pai2c-r2)

Date: 2026-04-27
Branch: icml-appendix-charlie-pai2c-r2
Baseline: NONE — first-round experiments establish empirical floor.

## Reasoning anchors

- **Primary metric**: `val_avg/mae_surf_p` (surface pressure MAE in physical space, equal-weighted across 4 splits). Lower is better. Surface-only, pressure-only — every design choice should be evaluated through that lens.
- **Loss vs. metric mismatch**: training uses MSE in **normalized** space (`vol_loss + 10 * surf_loss`), but ranking is L1 in **physical** space. MSE inflates the leverage of high-Re samples (y std ~2000) over low-Re (~50) — small Re samples are essentially ignored by normalized MSE because they are themselves close to zero in normalized space. This is a leverage point that touches every other knob.
- **Surface fraction is small**: surface nodes are a tiny minority of each mesh (a few hundred out of 85K-242K). The volume loss gets summed over 100x more nodes; the `surf_weight=10` is what keeps surface pressure relevant at all. Pushing this harder is plausibly the single highest-leverage knob.
- **Generalization story**: two of four val splits test interpolation to unseen NACA M (camber). Models that overfit raw `(x, z)` coordinates without abstracting through the foil-shape descriptors will fail those splits. Anything that improves geometry abstraction (relative positional features, Re-conditioned normalization, camber-aware loss balancing) should pay disproportionately on `val_geom_camber_*`.
- **Training budget**: 50 epochs, ~30 min wallclock. Batch size 4. With 96 GB VRAM and meshes max ~242K nodes, there is headroom on batch and capacity — but cosine LR over 50 epochs is short enough that warmup + better schedule choices matter.
- **Architecture context**: Transolver (n_hidden=128, 5 blocks, 4 heads, 64 slices, mlp_ratio=2). ~2-3M params. Small by modern standards — capacity is plausibly under-utilized given the variability in target ranges.

I deliberately spread across at least 6 strategy tiers below to maximize information from the first round. Each idea is single-variable, attributable, and includes a concrete predicted delta on `val_avg/mae_surf_p` (assuming a baseline in the 200-500 range; will calibrate after Round 1).

---

## 1. surf-weight-aggressive

- **Title**: `surf-weight-aggressive`
- **Strategy tier**: loss
- **Hypothesis**: The primary metric is surface pressure MAE, but the default training loss has `surf_weight=10`. Volume nodes outnumber surface nodes by ~100x; even with a 10x reweighting, the gradient on surface error is roughly equal to that on volume error per scalar contribution. Pushing `surf_weight` to 30 should redirect optimizer focus toward the surface, where the metric lives, with only modest cost to volume MAE (which is not in the ranking metric). I predict a 5–15% reduction in `val_avg/mae_surf_p`.
- **Concrete change**: In `train.py`, change `surf_weight: float = 10.0` to `surf_weight: float = 30.0` on the `Config` dataclass (one line). Everything else identical.
- **Risk**: At very high surf_weight, the model collapses toward minimizing surface error and lets volume drift, which can paradoxically hurt surface generalization because the model loses physical regularization from the volume field. Early-epoch signal: if `train/vol_loss` blows up (>2x) and `surf_loss` is not improving correspondingly, we are over-weighting. Run a parallel `surf_weight=20` if a single sweep slot allows.

## 2. surface-pressure-l1-loss

- **Title**: `surface-pressure-l1-loss`
- **Strategy tier**: loss
- **Hypothesis**: The training loss is MSE in normalized space; the ranking metric is L1 in physical space. MSE penalizes outliers quadratically — high-Re samples with y std ~2000 dominate the gradient and the model overfits to them at the expense of low-Re samples. Switching the **surface** term to a **physical-space L1** (denormalize prediction first, then mean-absolute pressure error, optionally / `y_std[2]` so it is comparable) should better align gradient direction with the metric. Volume term stays MSE (volume is not ranked, and MSE provides good geometric smoothness). I predict 8–15% improvement on `mae_surf_p`.
- **Concrete change**: In the train loop, after `pred = model({"x": x_norm})["preds"]`, compute:
  ```
  pred_phys = pred * stats["y_std"] + stats["y_mean"]
  surf_p_l1 = (((pred_phys[..., 2] - y[..., 2]).abs()) * surf_mask).sum() / surf_mask.sum().clamp(min=1) / stats["y_std"][2]
  surf_uxuy_mse = (((pred[..., :2] - y_norm[..., :2]) ** 2) * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
  surf_loss_terms = surf_p_l1 + surf_uxuy_mse  # all in roughly normalized magnitude
  loss = vol_loss + cfg.surf_weight * surf_loss_terms
  ```
- **Risk**: L1 has noisier gradients than MSE near optimum and can stall in a sub-grid-resolution band. Early signal: if surface MAE plateaus by epoch 10 well above the volume term's surface implication, fall back. Also watch validation noise — L1 sometimes gives lumpier per-epoch curves.

## 3. ema-evaluation

- **Title**: `ema-evaluation`
- **Strategy tier**: optimizer/schedule
- **Hypothesis**: With batch_size=4 and 50 epochs, the optimizer takes a few hundred steps per epoch and validation noise per epoch is high. Maintaining an exponential moving average of weights and using **EMA weights for validation and best-checkpoint selection** typically smooths out high-frequency optimizer noise and improves test/val numbers by 3-8% on small-batch transformers. The cost is one extra parameter set in memory (~2-3 MB, negligible). EMA is also robust to the sharp minima that small-batch training tends to land in.
- **Concrete change**: After `model = Transolver(...).to(device)`, instantiate `ema_model = copy.deepcopy(model)` and freeze its grads. After each `optimizer.step()`, do `for ep, p in zip(ema_model.parameters(), model.parameters()): ep.data.mul_(0.999).add_(p.data, alpha=0.001)`. In the val loop, evaluate `ema_model` instead of `model`. Save `ema_model.state_dict()` as the best checkpoint.
- **Risk**: With only 50 epochs and decay 0.999, the EMA may not fully integrate by the end. If the early epochs dominate, EMA can lag the live model. Use 0.995 instead if epoch-level metrics show EMA tracking too slowly. Also, dropout/BN do not interact with EMA without `model.eval()`; we have LayerNorm so this is fine.

## 4. lr-warmup-cosine-floor

- **Title**: `lr-warmup-cosine-floor`
- **Strategy tier**: optimizer/schedule
- **Hypothesis**: Pure cosine annealing from 5e-4 to 0 over 50 epochs starts at full LR (no warmup) on a fresh transformer with random init — this is known to be unstable in the first few hundred steps. A 2-epoch linear warmup followed by cosine decay to a non-zero floor (1e-5, ~2% of peak) should both stabilize the early loss landscape and prevent the last epochs from being effectively zero-LR (wasted compute). I expect a small but reliable 3–6% improvement.
- **Concrete change**: Replace the scheduler with a `SequentialLR` of `LinearLR(start_factor=0.1, end_factor=1.0, total_iters=2)` followed by `CosineAnnealingLR(T_max=MAX_EPOCHS-2, eta_min=1e-5)`. Also try peak `lr=1e-3` (aggressive variant) since warmup permits it.
- **Risk**: Aggressive peak with only 2 epochs of warmup can still spike. If `train/surf_loss` shows a NaN or blow-up in epoch 1-2, drop peak to 7e-4. Floor of 1e-5 might be too high — adjust to 1e-6 if the last few epochs show no movement.

## 5. relative-position-features

- **Title**: `relative-position-features`
- **Strategy tier**: feature-engineering
- **Hypothesis**: The model receives raw `(x, z)` node positions normalized only by global mean/std. But for an airfoil flow, the meaningful geometry is the position **relative to the foil**. The dataset gives us `dsdf` (distance descriptor, dims 4-11) and signed arc-length (`saf`, dims 2-3), but these are pre-computed once and are not Re- or AoA-aware. Adding **Fourier-encoded relative-to-mid-chord position** (i.e., `[sin(2πf·x_rel), cos(2πf·x_rel)]` for several frequencies) gives the network high-frequency geometric resolution without it having to invent that itself. This should pay especially on `val_geom_camber_*` where unseen geometries demand interpolation in geometry space. Predicted delta: 5–10% on the geom splits, smaller on others.
- **Concrete change**: In `train.py`, before the Transolver forward, compute Fourier features for the position dims and concatenate them into `x`. New `fun_dim` increases. Specifically, for `f ∈ [1, 2, 4, 8, 16]`, add `sin(2πf·x[...,0:2])` and `cos(...)` → +20 dims. Update `model_config["fun_dim"]` accordingly. Apply the projection inside `train.py` before normalization (use raw position, then divide by a chord-length constant of 1.0 or by `y_std`-style scale).
- **Risk**: More input dims → larger preprocessing MLP → mild VRAM and compute increase. The Fourier features could also introduce high-frequency noise that hurts the smoother volume field. Watch for `val_avg/mae_vol_*` regressing >10%; if so, halve the frequency bank.

## 6. slice-num-doubled

- **Title**: `slice-num-doubled`
- **Strategy tier**: architecture-tweak
- **Hypothesis**: Transolver's slice tokens are the bottleneck through which all node information passes. With `slice_num=64` and meshes of 80-240K nodes, each slice represents ~1500-4000 nodes on average — a coarse abstraction. Doubling to `slice_num=128` gives the attention block roughly 2x more capacity to represent distinct flow regions (boundary layer, wake, freestream, between-foil). The marginal cost is small (slice attention is GxG, so 128 vs 64 is 4x but G is small). Predicted delta: 3-8% improvement, larger on cruise tandem (deepest meshes).
- **Concrete change**: In `model_config`, set `slice_num=128`. One-line change.
- **Risk**: More slices may underfit (each slice gets fewer assigned nodes, learning noisier representations). With 50 epochs this can be a real concern. Watch for sluggish train loss in first 10 epochs versus baseline runs. Also: VRAM scales with slice attention as O(slices^2 * heads). 128^2 = 16384 entries per head — still tiny, no VRAM concern.

## 7. layerscale-stochastic-depth

- **Title**: `layerscale-stochastic-depth`
- **Strategy tier**: regularization
- **Hypothesis**: Modern transformer recipes (CaiT, ConvNeXt) consistently show LayerScale (a learned per-channel scalar at each residual branch initialized to ~1e-4) and DropPath (stochastic residual drop, p~0.1) jointly improve stability and final performance, especially in low-data regimes. Our 1499 training samples is firmly in low-data territory. The combination should give 4-8% improvement on val avg and is particularly likely to help the held-out geometry splits where overfitting is the concern.
- **Concrete change**: In `TransolverBlock.__init__`, add `self.gamma_attn = nn.Parameter(torch.ones(hidden_dim) * 1e-4)`, `self.gamma_mlp = nn.Parameter(torch.ones(hidden_dim) * 1e-4)`, and modify forward to `fx = drop_path(self.gamma_attn * self.attn(self.ln_1(fx))) + fx`. Implement `drop_path(x)` as a stochastic depth wrapper with `p=0.1`. Use `timm.layers.DropPath`.
- **Risk**: LayerScale init at 1e-4 means the residual branches are ~zero at init and the model effectively has identity-only depth for the first few epochs — slow start. Watch epoch-1 train loss versus baseline; if it is >1.5x baseline epoch-1, drop init to 1e-2 instead. Also check that DropPath does not interact badly with the small batch size (variance is higher).

## 8. wider-shallower-arch

- **Title**: `wider-shallower-arch`
- **Strategy tier**: architecture-tweak
- **Hypothesis**: The default arch is 5 blocks of n_hidden=128. For irregular-mesh problems, **wider** (more channels per node) typically beats **deeper** (more layers) because the bottleneck is per-node feature representation, not global reasoning. A 4-block, n_hidden=192 model has similar params (~1.4x) but more representational headroom for the per-node feature transform. Predicted delta: 5-10% on surface metrics, especially `mae_surf_p` which depends on local per-node features.
- **Concrete change**: In `model_config`, set `n_hidden=192, n_layers=4, n_head=6` (so dim_head stays at 32). Param count goes from ~2.5M to ~3.5M, comfortably within VRAM for batch 4.
- **Risk**: Sometimes deeper is needed for global structure (wake propagation can require 4+ rounds of attention to span). If `mae_vol_p` regresses while `mae_surf_p` improves only marginally, the wider arch is hurting global flow. Watch volume metrics in epochs 5-15 for evidence.

## 9. residual-head-baseline

- **Title**: `residual-head-baseline`
- **Strategy tier**: data-representation
- **Hypothesis**: A common trick in surrogates is to predict **residuals** from a cheap analytical baseline rather than the full target. For a foil flow, a reasonable baseline is the freestream velocity (Ux=cos(AoA), Uy=sin(AoA), p=0). Subtracting this from the target before normalization concentrates the learning signal on the **deviations** from freestream — which is where the physics interest lives, and where surface pressure MAE is dominated. The output target distribution becomes more centered and lower-variance, which should give 5-12% improvement on surface metrics where the deviations are largest.
- **Concrete change**: At dataset load (in `train.py`), construct `y_baseline = freestream(x)` per sample where Ux_fs=Re·cos(AoA), Uy_fs=Re·sin(AoA), p_fs=0 (use AoA from `x[..., 14]` and a Re proxy from `x[..., 13]`). Recompute targets as `y_residual = y - y_baseline`. Recompute `y_mean`/`y_std` over residuals — or use a fresh sample-based estimate. At eval, add baseline back before MAE. Subtract should be done in physical space, not normalized.
- **Risk**: The freestream subtraction can be miscalibrated — the actual inflow speed isn't `Re` (Re also includes chord and viscosity). If the baseline is wrong, we are just adding noise. Validate on a single-foil sample by hand: compute freestream-subtracted y_std and confirm it is meaningfully smaller. If `y_std_residual / y_std_original > 0.9`, the baseline is too weak; close the experiment early.

## 10. domain-id-conditioning

- **Title**: `domain-id-conditioning`
- **Strategy tier**: feature-engineering
- **Hypothesis**: The three training domains (raceCar single, raceCar tandem, cruise tandem) have very different flow regimes (ground-effect vs. freestream BC, inverted vs. positive AoA, shared vs. distinct rear-foil). Right now the model has to **infer** the domain from gap=0/non-zero, AoA sign, etc. Adding an explicit 3-dim one-hot domain indicator gives it the conditioning information directly. This is a tiny change that reduces a confounder. Predicted delta: 3-7% improvement, with larger gains on `val_re_rand` and `val_geom_camber_cruise` where the model has to cleanly separate the two tandem regimes.
- **Concrete change**: In `train.py`, before normalization, compute `is_single = (x[..., 22] == 0) & (x[..., 23] == 0)` (gap and stagger both 0), `is_cruise = x[..., 14] > 0` (cruise has positive-friendly AoA range), `is_rc_tandem = ~is_single & ~is_cruise`. Concatenate as 3 extra dims. Update `fun_dim`.
- **Risk**: The detection rules are heuristic and could mis-classify edge cases (e.g., a cruise sample at AoA=0 would be flagged as single). A cleaner approach uses the dataset metadata, but per the constraints, we can't change `data/loader.py`. If results degrade, the heuristic is wrong; an alternative is to learn the embedding from the existing cluster of features (gap, stagger, AoA-sign, Re-range).

## 11. gradnorm-clip-1.0

- **Title**: `gradnorm-clip-1.0`
- **Strategy tier**: optimizer/schedule
- **Hypothesis**: With batch_size=4 and a heavy-tailed target distribution, individual gradient norms can spike enormously when a high-Re sample lands in the batch. Without gradient clipping, these spikes destabilize the weight updates and effectively waste several optimizer steps. Adding `clip_grad_norm_(model.parameters(), max_norm=1.0)` is a standard low-cost intervention that reliably gives 2-5% on training-noise-limited problems. Combined with the bullet that LR=5e-4 with no warmup is already stress-testing the optimizer, this should be near-strictly beneficial.
- **Concrete change**: After `loss.backward()` and before `optimizer.step()`, add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Single line. Optionally also log the clipped grad norm to JSONL for diagnostic.
- **Risk**: Setting max_norm too low chokes off learning. 1.0 is conservative and standard for transformers; if `train/surf_loss` is consistently flat even when LR is non-trivial, raise to 5.0. Also: gradient clipping interacts with weight_decay differently in AdamW — but the effect is small.

## 12. asinh-pressure-target

- **Title**: `asinh-pressure-target`
- **Strategy tier**: data-representation
- **Hypothesis**: Pressure values in the dataset span (-29K, +2.7K) — heavily heavy-tailed and wildly Re-dependent. Standard mean/std normalization compresses this into a Gaussian-like blob but does not remove the long tails. An **asinh transform** (`y' = asinh(y / s)` for some scale s ≈ surface-typical magnitude) is a smooth, monotone transform that compresses the tails while staying linear near zero — preserving both the small-Re structure and the high-Re extrema. The model then predicts in asinh-space and the loss naturally weights both ends comparably. This is a highly leveraged change for heavy-tailed targets. Predicted delta: 8-15% on `mae_surf_p`.
- **Concrete change**: After loading stats, set a scale `s = stats["y_std"]` (per channel). Replace target normalization `y_norm = (y - y_mean) / y_std` with `y_norm = torch.asinh(y / s)`. At eval, denormalize via `y_pred = s * torch.sinh(pred)`. This needs the eval helper in `evaluate_split` to also apply this transform consistently. The MAE in `data/scoring.py` is on physical-space y; pass through `sinh` first.
- **Risk**: asinh inverts cleanly only if the model output stays bounded — a runaway prediction in asinh-space becomes astronomically large after sinh. Add a clamp on the predicted asinh-space value (e.g., `pred.clamp(-20, 20)`), corresponding to physical-space ~5e8. Watch epoch-1 for any non-finite gradients or NaN losses.

---

## Diversity check

Strategy tier coverage in the 12 ideas:

| Tier | Ideas |
|---|---|
| loss | #1 surf-weight, #2 L1-loss |
| optimizer/schedule | #3 EMA, #4 warmup-cosine, #11 grad-clip |
| regularization | #7 layerscale-droppath |
| architecture-tweak | #6 slice-num, #8 wider-shallower |
| feature-engineering | #5 fourier-pos, #10 domain-id |
| data-representation | #9 residual-head, #12 asinh |

7 distinct tiers covered, 5+ as required. No architecture-replacement bets in Round 1 — those are higher-risk with no baseline to compare against; reserve for Round 2 once we have empirical floors.

## Suggested Round-1 priority order (top 5 + reasoning)

1. **#1 surf-weight-aggressive** — single-line change, directly targets the metric, almost no downside risk. Highest likely value-per-PR.
2. **#12 asinh-pressure-target** — heavy-tailed targets are the obvious antagonist. Big upside if it works.
3. **#3 ema-evaluation** — small-batch training nearly always benefits from EMA. Strong base hit.
4. **#4 lr-warmup-cosine-floor** — fundamental schedule fix that compounds with everything else.
5. **#7 layerscale-stochastic-depth** — strongest pure regularization play; should help both held-out splits.

Spread the rest (5-11) to keep students busy across diverse tiers so we get a good empirical map of the design space in Round 1.
