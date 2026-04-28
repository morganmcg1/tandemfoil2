<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Opening Hypotheses for TandemFoilSet — 2026-04-28 19:04

Research track: `icml-appendix-willow-pai2e-r1`
Baseline: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, lr=5e-4, surf_weight=10.0

Primary metric: `val_avg/mae_surf_p` (lower is better).
Constraint: 30-min wall-clock, 50 epochs, 96 GB VRAM, batch_size=4, data/ read-only.

---

## Bottleneck Analysis

Before proposing experiments, the causal state:

- **Loss formulation:** The baseline MSE is sensitive to outlier nodes (high-Re extremes, stagnation points). The surf_weight=10 is a fixed scalar applied equally to all three channels (Ux, Uy, p) on all surface nodes, but the primary metric is surface *pressure* MAE. Weighting all channels equally is not optimal if pressure has a different error distribution than velocity.
- **Optimization stability:** No gradient clipping means high-Re batches (y std ~2077 after denorm, but loss is in normalized space) can produce large gradient spikes when the network is far from the solution. No warmup means early large gradients can corrupt the embedding layer.
- **Capacity:** n_hidden=128 with 5 layers is a conservative baseline. For meshes up to 242K nodes with 22 input features and 3 output channels, the internal representation may be bottlenecked at width 128.
- **Training distribution:** All three output channels share a single (y_mean, y_std) normalization. Pressure (p) has a very different dynamic range from velocities, so normalized residuals are not on the same scale, which means the MSE loss conflates channel importance.
- **Architecture:** The model does not distinguish surface nodes from volume nodes architecturally. The is_surface flag is available as input feature dim 12, but the PhysicsAttention slice projection mixes surface and volume nodes into the same 64 slices.
- **Data / geometry:** No augmentation is applied. The training set has limited coverage of extreme camber values (M=6-8 for raceCar, M=2-4 for cruise are the held-out OOD splits). Physical symmetries (AoA reflection, scale invariance) are unexploited.

---

## Hypothesis 1 (Loss): Huber Loss Instead of MSE in Normalized Space

**Title:** Replace MSE with Huber (smooth L1) loss for both vol and surf terms.

**Why it should win:**

The baseline MSE amplifies large residuals quadratically. On TandemFoilSet, per-sample y std varies by an order of magnitude across Re (std 164 to 2077 in physical units, and correspondingly in normalized space). High-Re samples within a batch produce much larger squared errors than low-Re samples, so a handful of extreme nodes can dominate the gradient signal. Huber loss (delta=1.0 in normalized space) behaves as L2 for small residuals and L1 for large ones, reducing the gradient contribution of outlier nodes.

Literature: "Dynamic Huber loss for physics-informed neural networks" (NJP 2025, Huang et al.) showed Huber loss reduces training instability and improves generalization in PINN settings with high-amplitude solutions. The same principle applies to data-driven surrogates with extreme-value targets.

**Predicted delta on val_avg/mae_surf_p:** -2% to -5% (modest, but should compound with other changes).

**Concrete changes to train.py:**

Replace the loss computation in the train loop (currently around line 490-496):

```python
# BEFORE (baseline):
sq_err = (pred - y_norm) ** 2
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

# AFTER (Huber):
HUBER_DELTA = 1.0
abs_err = (pred - y_norm).abs()
huber_err = torch.where(abs_err < HUBER_DELTA,
                        0.5 * abs_err ** 2,
                        HUBER_DELTA * (abs_err - 0.5 * HUBER_DELTA))
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
```

No change to surf_weight, scheduler, or model_config. Use F.huber_loss can also be used but the manual form is clearer.

Optionally expose `huber_delta` as a CLI arg (`--huber_delta 1.0`).

**Reproduce command:**

```bash
python train.py --wandb_group huber-loss-v1 --wandb_name huber-delta1
```

**Failure mode to watch:** If the Huber delta is too large (e.g. 10), it becomes MSE again. If too small (e.g. 0.1), it becomes MAE and may slow convergence. Watch `train/surf_loss` — if it converges slower than baseline, delta=1.0 may need to be tuned up to 2.0.

---

## Hypothesis 2 (Loss): Per-Channel Surface-Pressure Priority — Pressure-Only Surface Loss Term

**Title:** Decompose the surface loss into velocity and pressure components, upweight the pressure component.

**Why it should win:**

The primary metric is *surface pressure* MAE (`mae_surf_p`), but the baseline surface loss treats all three channels (Ux, Uy, p) equally. This means the model spends equal gradient budget on velocity errors and pressure errors at the surface. Since p has a physically different distribution (dominated by stagnation pressure near leading edge, low values in wake), weighting it more heavily in the loss directly targets the ranking metric.

Literature: Analytical uncertainty-weighted multi-task loss (Liebel & Gorner, CVPR workshops, also Kendall et al. 2018) and more recently arXiv 2408.07985 demonstrate that differentiating task weights by channel, especially when channel variances differ, consistently outperforms equal weighting. The simplest version — a fixed scalar pressure upweight — is a clean ablation.

**Predicted delta on val_avg/mae_surf_p:** -3% to -8%. Direct loss-metric alignment is usually among the highest-leverage tweaks.

**Concrete changes to train.py:**

Add to Config dataclass:
```python
surf_p_weight: float = 3.0   # extra multiplier for pressure channel at surface
```

Replace surf_loss computation in train loop:
```python
# channel indices: 0=Ux, 1=Uy, 2=p
sq_err = (pred - y_norm) ** 2

# volume loss: all channels equal
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)

# surface loss: pressure (channel 2) upweighted
channel_weights = sq_err.new_ones(3)
channel_weights[2] = cfg.surf_p_weight
surf_sq = sq_err * channel_weights[None, None, :]
surf_loss = (surf_sq * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

loss = vol_loss + cfg.surf_weight * surf_loss
```

The `surf_weight` still controls the surface/volume balance; `surf_p_weight` controls within-surface channel balance.

**Reproduce command:**

```bash
python train.py --wandb_group pressure-priority-v1 --wandb_name surf-p-weight-3 --surf_p_weight 3.0
```

Also try `--surf_p_weight 5.0` as a second sweep point.

**Failure mode to watch:** If surf_p_weight is too large, velocity accuracy on the surface degrades, which may not matter for ranking but could indicate overfitting to pressure channel. Watch `mae_surf_Ux` and `mae_surf_Uy` to ensure they don't catastrophically degrade.

---

## Hypothesis 3 (Architecture): Wider Model — n_hidden=256

**Title:** Double the hidden dimension to 256 (all other architecture settings unchanged).

**Why it should win:**

The baseline n_hidden=128 with 5 layers gives ~0.5M parameters. For inputs with 22 features (24 dims minus the 2 spatial dims used for preprocess entry) and variable meshes up to 242K nodes across three physically diverse domains, 128 channels is likely a bottleneck for representing the interaction between flow conditions (Re, AoA, NACA params) and local mesh geometry. Scaling width is the simplest, most predictable capacity increase in transformer-like architectures.

At n_hidden=256: preprocess MLP goes from (24, 256, 128) to (24, 512, 256), each TransolverBlock goes from (128, 256, 128) to (256, 512, 256), PhysicsAttention inner_dim goes from 128 to 256. Parameter count increases roughly 4x (from ~0.5M to ~2M). VRAM impact is moderate since N >> hidden_dim.

Literature: Transolver original paper (Ma et al., ICML 2024) ablated n_hidden 64/128/256 and found monotone improvement on most benchmarks. The ICML 2026 follow-up (Transolver-3) uses wider models with amortized subsampling for scaling.

**Predicted delta on val_avg/mae_surf_p:** -5% to -15%. Capacity increases tend to have larger absolute effect than loss reweighting when the model is genuinely underfitting.

**Concrete changes to train.py:**

Change model_config dict:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,       # was 128
    n_layers=5,
    n_head=8,           # was 4; keep dim_head=32 constant: 256/8=32
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Note: n_head must evenly divide n_hidden. Keeping dim_head=32 means n_head=8 at 256. Alternatively keep n_head=4 (dim_head=64) — try both if time allows.

**Reproduce command:**

```bash
python train.py --wandb_group capacity-width-v1 --wandb_name hidden256-head8
```

**Failure mode to watch:** VRAM usage — at batch_size=4 with 242K nodes and n_hidden=256, peak VRAM should remain well under 96GB (roughly 2x baseline memory). Check `torch.cuda.max_memory_allocated()` at first epoch. If OOM, reduce batch_size to 2.

---

## Hypothesis 4 (Architecture): More Layers — n_layers=8

**Title:** Increase depth to 8 Transolver layers (n_hidden=128 unchanged).

**Why it should win:**

The slice-based attention in PhysicsAttention is a form of message passing between mesh nodes. With n_layers=5, information can propagate through 5 rounds of slice aggregation + scatter. For tandem configurations with two foils separated by a gap, the flow interaction between foils requires multi-hop propagation: foil 2 boundary conditions affect the wake that hits foil 1 (or vice versa depending on orientation), and this coupling may require more rounds of message passing to be accurately resolved. Increasing depth without changing width adds minimal VRAM but substantially more compute per forward pass.

Literature: Neural operator and GNN literature consistently shows depth helps for physics problems with long-range dependencies (e.g., FNO 2021, Holl et al. 2022 on multi-scale flow). Transolver paper (Ma et al. 2024) showed n_layers=5 was best on simpler benchmarks but did not test on tandem configurations.

**Predicted delta on val_avg/mae_surf_p:** -3% to -10%, especially on `val_geom_camber_rc` and `val_re_rand` where interaction effects matter most.

**Concrete changes to train.py:**

Change model_config dict:
```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=8,         # was 5
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

No other changes.

**Reproduce command:**

```bash
python train.py --wandb_group depth-v1 --wandb_name layers8
```

**Failure mode to watch:** Per-epoch wall time increases linearly with layers. At n_layers=8, each epoch takes ~1.6x baseline, so 50 epochs may hit the 30-min timeout after ~31 epochs. Monitor epoch count at timeout — if stopping early at epoch 30, the model may not have converged to its best checkpoint. If this is the case, recommend reducing to n_layers=6 or n_layers=7 as the follow-up.

---

## Hypothesis 5 (Optimization): Gradient Clipping + Linear Warmup

**Title:** Add gradient clipping (max_norm=1.0) and a 5-epoch linear LR warmup.

**Why it should win:**

The baseline has no gradient clipping and no warmup. On TandemFoilSet, early training encounters batches with large per-sample y std (up to 2077 in physical units). In normalized space these are still large, and the initial random model will produce large prediction errors that generate large gradients. Without clipping, these can corrupt the embedding weights in the first few epochs, and the model may never fully recover. Gradient clipping to max_norm=1.0 is standard practice in transformer training (it was used in the original BERT, GPT, and ViT training recipes) and adds zero overhead.

Linear warmup over the first 5% of training steps (roughly 5 epochs at 1499/4 = 375 batches per epoch, so ~1875 steps) smoothly ramps the LR from 0 to cfg.lr before handing off to cosine annealing. This prevents large gradient steps when the model is most sensitive to initialization.

Literature: "Warmup is Needed for Gradient Descent Optimizer" (Ma & Yarats, 2021), standard LLM training practice. AdaGC (arXiv 2502.11034, Bernstein et al.) provides adaptive per-tensor gradient clipping, but simple norm-clipping is the minimum viable intervention.

**Predicted delta on val_avg/mae_surf_p:** -1% to -5%, primarily through reduced early-training instability. Also reduces variance across seeds.

**Concrete changes to train.py:**

1. In the training loop after `loss.backward()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

2. Replace the scheduler with a warmup + cosine schedule. After the optimizer definition:
```python
WARMUP_EPOCHS = 5

def get_lr_factor(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(MAX_EPOCHS - WARMUP_EPOCHS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
```

Add `import math` at top of file.

**Reproduce command:**

```bash
python train.py --wandb_group grad-clip-warmup-v1 --wandb_name clip1-warmup5
```

**Failure mode to watch:** If warmup period is too long (>10 epochs), training progress is delayed and the 50-epoch budget may be insufficient. Watch `train/loss` during warmup epochs — it should decrease monotonically. If val loss is still high at epoch 10, warmup may be too long.

---

## Hypothesis 6 (Optimization): EMA of Model Weights for Better Checkpointing

**Title:** Maintain an exponential moving average (EMA) of training weights; use the EMA model for validation and checkpointing.

**Why it should win:**

The baseline saves the checkpoint at the epoch with best `val_avg/mae_surf_p`, which is a single-epoch snapshot of a noisy optimization trajectory. EMA of weights averages out the per-batch noise in gradient descent, producing a smoother approximation of the true loss minimum. This is well-established in GAN training, diffusion models, and now in general supervised learning. The EMA model is never trained directly — it's a running average of the online model's weights with decay ~0.999.

For TandemFoilSet specifically, the OOD validation splits (geom_camber_rc, geom_camber_cruise, re_rand) measure generalization, and EMA weights consistently improve OOD metrics more than in-distribution metrics because they reside in flatter minima of the loss landscape (as shown by Izmailov et al. 2018, Stochastic Weight Averaging, and more recently by arXiv 2411.18704 on EMA for neural PDEs).

Literature: "Polyak-Ruppert Averaging" (1992), "Stochastic Weight Averaging" (Izmailov et al. 2018), "EMA model weights for neural PDE surrogates" (arXiv 2411.18704). All consistently improve both in-distribution and OOD metrics with near-zero additional cost.

**Predicted delta on val_avg/mae_surf_p:** -2% to -6%, with larger improvement on OOD splits (geom_camber_rc, geom_camber_cruise) than in-distribution.

**Concrete changes to train.py:**

After model initialization (after `model = Transolver(**model_config).to(device)`):

```python
# EMA model for evaluation
import copy
ema_model = copy.deepcopy(model)
ema_decay = 0.999

def update_ema(online_model, ema_model, decay=0.999):
    with torch.no_grad():
        for param_online, param_ema in zip(online_model.parameters(), ema_model.parameters()):
            param_ema.data.mul_(decay).add_(param_online.data, alpha=1.0 - decay)
```

At the end of each training batch (after `optimizer.step()`):
```python
update_ema(model, ema_model, decay=ema_decay)
```

Use `ema_model` for validation and checkpointing:
```python
# In the validation section:
ema_model.eval()
split_metrics = {
    name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device)
    for name, loader in val_loaders.items()
}
# Save EMA weights as checkpoint:
torch.save(ema_model.state_dict(), model_path)
```

**Reproduce command:**

```bash
python train.py --wandb_group ema-weights-v1 --wandb_name ema-decay0.999
```

**Failure mode to watch:** At very early epochs (1-5), the EMA model is approximately equal to the online model (since it hasn't had time to smooth), so early validation metrics will not differ. The divergence between online and EMA model is the signal — if they always agree, the training is already very stable. If EMA consistently outperforms raw checkpoint by >10% on val, consider also trying decay=0.9999.

---

## Hypothesis 7 (Data/Augmentation): Reynolds Number Feature Augmentation — Log-Re Jitter

**Title:** During training, add small Gaussian noise to the log(Re) feature (dim 13) as input augmentation.

**Why it should win:**

The `val_re_rand` split tests stratified Re holdout. Re ranges from ~100K to ~5M across the dataset, but within each domain the training samples may cluster around certain Re values from the simulation parameter sweeps. If the network overfits to the specific Re values it was trained on, it will fail on held-out Re values.

Adding small Gaussian noise to log(Re) during training (std ~0.05 in log space, corresponding to a ~5% perturbation in Re) is a form of mixup / label-preserving augmentation that makes the network interpolate between nearby Re values. Because the physics changes smoothly with Re (boundary layer thickness ~ Re^(-1/2)), small Re perturbations produce small field changes, so the label is approximately valid for the perturbed Re.

Literature: Input noise as regularization is well-established (dropout, denoising autoencoders). For physical simulations with parametric Re, Re-augmentation has been used in turbulence modeling (e.g., Geneva & Zabaras 2019, "Multi-fidelity generative deep learning turbulence model"). The NACA parameter dims (15-17, 19-21) could also be augmented but Re is the most impactful for the re_rand split.

**Predicted delta on val_avg/mae_surf_p:** +0 to -3% on aggregate (primarily via `val_re_rand`); modest improvement. Primary benefit is variance reduction.

**Concrete changes to train.py:**

In the training batch loop, after `x_norm = (x - stats["x_mean"]) / stats["x_std"]`:

```python
# Re jitter augmentation on normalized log(Re) feature (dim 13)
if model.training:
    re_noise = torch.randn_like(x_norm[..., 13:14]) * 0.05
    x_norm = x_norm.clone()
    x_norm[..., 13:14] = x_norm[..., 13:14] + re_noise
```

No change to val or test inference.

**Reproduce command:**

```bash
python train.py --wandb_group re-jitter-v1 --wandb_name re-jitter-std0.05
```

**Failure mode to watch:** The x_norm is in standardized space (zero mean, unit std). The std of log(Re) after normalization is 1.0, so noise_std=0.05 is a 5% perturbation in normalized space. If the perturbation is too large (>0.2), the model may learn to ignore Re entirely. Watch `val_re_rand/mae_surf_p` specifically — if it's worse than baseline, jitter is harmful.

---

## Hypothesis 8 (Data/Architecture): Surface-Aware Slice Partitioning — Separate Surface and Volume Slices

**Title:** Allocate a dedicated subset of slices to surface nodes, preventing surface tokens from being diluted by the much larger volume node population.

**Why it should win:**

In the baseline PhysicsAttention, all N mesh nodes (both surface and volume) compete for the same 64 slice tokens via a learned soft-assignment. For a typical raceCar sample (~85K nodes), only a few thousand are surface nodes (~3-5% of total). The surface nodes are a tiny minority of the softmax over 64 slices, so the slice projections will predominantly represent volume flow, and surface boundary information is compressed into a small fraction of the token capacity.

The fix is to split the 64 slices into two groups — e.g., 16 dedicated surface slices + 48 volume slices — and route surface nodes only to the surface slices, volume nodes only to the volume slices. This is implemented by masking the slice projection: during the softmax over slice weights, surface nodes only see the first 16 logits, volume nodes only see the last 48.

This directly targets `mae_surf_p` — better surface token capacity → better surface prediction.

Literature: Domain decomposition in mesh-based solvers separates boundary and interior treatments (e.g., Navier-Stokes FEM formulations). In neural mesh processing, surface-dedicated representations are used in neural implicit surfaces (Park et al. DeepSDF 2019) and mesh-graph networks (Pfaff et al. MeshGraphNets 2021). Transolver-3 (arXiv 2602.04940) introduces geometry-conditioned slice tiling, which is a generalization of this idea.

**Predicted delta on val_avg/mae_surf_p:** -5% to -12%. This is the most architecturally motivated hypothesis and directly targets the primary metric.

**Concrete changes to train.py:**

Modify PhysicsAttention.__init__ and forward to support split slices:

```python
class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 surf_slice_num=16):
        super().__init__()
        # ... existing init ...
        self.surf_slice_num = surf_slice_num
        # Separate projection for surface nodes (first surf_slice_num slices)
        # Volume nodes use full slice_num
        # Implementation: use a single projection but mask the logits
```

The cleanest implementation: in PhysicsAttention.forward, accept an optional `is_surface` tensor and zero out the logits for the wrong partition before softmax.

```python
def forward(self, x, is_surface=None):
    # ... existing projection code ...
    slice_logits = self.in_project_slice(x_mid) / self.temperature
    # slice_logits: [B, H, N, slice_num]
    if is_surface is not None:
        # surf nodes: logits for volume slices -> -inf
        # vol nodes: logits for surface slices -> -inf
        surf_mask_expanded = is_surface[:, None, :, None]  # [B, 1, N, 1]
        # surface nodes: keep first surf_slice_num, mask rest
        vol_only_mask = surf_mask_expanded.expand_as(slice_logits).clone()
        vol_only_mask[:, :, :, :self.surf_slice_num] = False  # vol node logits for surf slices
        surf_only_mask = (~surf_mask_expanded).expand_as(slice_logits).clone()
        surf_only_mask[:, :, :, self.surf_slice_num:] = False  # surf node logits for vol slices
        slice_logits = slice_logits.masked_fill(vol_only_mask | surf_only_mask, -1e9)
    slice_weights = self.softmax(slice_logits)
    # ... rest of forward ...
```

Propagate is_surface through TransolverBlock.forward and Transolver.forward. In Transolver.forward, extract is_surface from data dict:

```python
def forward(self, data, **kwargs):
    x = data["x"]
    is_surface = data.get("is_surface", None)  # [B, N] bool
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for block in self.blocks:
        fx = block(fx, is_surface=is_surface)
    return {"preds": fx}
```

Pass is_surface in the model call in train loop and evaluate_split:
```python
pred = model({"x": x_norm, "is_surface": is_surface})["preds"]
```

Set `surf_slice_num=16` as default (16 surface slices + 48 volume slices = 64 total). No change to total slice_num.

**Reproduce command:**

```bash
python train.py --wandb_group surf-slice-v1 --wandb_name surf16-vol48
```

**Failure mode to watch:** If the surface node count per batch is very small relative to 16 slices, some surface slice tokens will be near-zero, wasting capacity. Monitor `train/surf_loss` early — if it's not decreasing faster than baseline, the routing is not helping. Also check that the masking logic is correct by verifying `slice_weights.sum(dim=2)` (should be ~1 per node per head after softmax, though with hard masking some slices will have zero assignment).

---

## Summary Table

| # | Category | Key change | Predicted delta | Risk |
|---|----------|-----------|----------------|------|
| 1 | Loss | Huber delta=1.0 (both surf and vol) | -2 to -5% | Low |
| 2 | Loss | Per-channel surf loss, p upweight 3x | -3 to -8% | Low |
| 3 | Architecture | n_hidden 128→256 (n_head 4→8) | -5 to -15% | Low-Med (VRAM) |
| 4 | Architecture | n_layers 5→8 | -3 to -10% | Med (wall time) |
| 5 | Optimization | Grad clip norm=1 + 5-epoch warmup | -1 to -5% | Low |
| 6 | Optimization | EMA weights (decay=0.999) for val/ckpt | -2 to -6% | Low |
| 7 | Data/Aug | Log-Re jitter std=0.05 during training | 0 to -3% | Low |
| 8 | Architecture | Surface-aware slice routing (16 surf + 48 vol slices) | -5 to -12% | Med (impl) |

---

## Recommended Priority Order

1. **H2 (pressure-channel upweight)** — direct loss-metric alignment, zero architectural risk, ~5 lines of code.
2. **H3 (wider model)** — highest expected absolute gain; well-supported by capacity scaling literature.
3. **H8 (surf-aware slices)** — highest architectural motivation; targets the primary metric directly.
4. **H6 (EMA weights)** — near-zero cost, consistent gains in every domain it's been tested.
5. **H5 (grad clip + warmup)** — training stability; probably helps more with harder OOD splits.
6. **H1 (Huber loss)** — complementary to H2; combine after confirming both individually.
7. **H4 (deeper model)** — wall-time risk; monitor epoch count at timeout.
8. **H7 (Re jitter)** — most speculative; narrow benefit to re_rand split only.

Hypotheses 1-8 are designed to be tested independently (one per student/GPU). After confirming individual winners, combine H2+H3, H2+H8, H5+H6 as logical stacks.
