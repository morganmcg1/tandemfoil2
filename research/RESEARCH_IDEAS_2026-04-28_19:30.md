# Wave-2 Research Ideas — TandemFoilSet Transolver

**Generated:** 2026-04-28 19:30  
**Track:** willow-pai2e-r5 / icml-appendix-willow-pai2e-r5  
**Primary metric:** `val_avg/mae_surf_p` (lower is better)  
**Already running (Wave-1, do not repeat):** capacity (n_hidden/n_layers/n_head), slice_num, surf_weight, LR warmup+cosine, Huber loss, dropout, separate output heads, distance features

All 12 hypotheses below are orthogonal to each other and to Wave-1. Each targets a distinct causal failure mode in the current stack.

---

## Hypothesis 1: `linear-attention-no`

### What it is
Replace the PhysicsAttention slice mechanism with canonical softmax-free linear attention — effectively testing whether the learned slice grouping is helping or hurting relative to a simpler O(N) aggregation.

### Hypothesis
LinearNO (AAAI 2026) proved that Transolver's physics-attention is mathematically a special case of linear attention with a learned gating function, and showed that removing the slice-interaction step (the Q/K/V over slice tokens) achieves SOTA on 6 out of 8 PDE benchmarks. The hypothesis is that on TandemFoilSet the slice-grouping step adds optimization difficulty and expressive overhead that does not pay off within 50 epochs, particularly for the OOD camber splits where the slice groupings learned on train-domain geometries may not transfer. This would benefit `val_geom_camber_rc` and `val_geom_camber_cruise` most.

Mechanistically: instead of projecting `fx_mid` to `slice_num` latent tokens and then doing full Q/K/V attention over those tokens, we compute a single aggregated context vector per head by sum-pooling `fx_mid` weighted by the softmax slice weights, then broadcast it back. This removes the Q/K/V matrices for slice tokens (roughly 3 × dim_head^2 × heads parameters per layer) and replaces the slice-token attention with a linear gating. The in_project_slice pathway is preserved as the gating weights, keeping the physics-partitioning inductive bias but removing the inter-slice quadratic step.

### Implementation plan

In `PhysicsAttention.forward` (lines 105–136 of `train.py`), replace the Q/K/V attention over `slice_token` with a direct gated broadcast:

```python
def forward(self, x):
    B, N, _ = x.shape
    fx_mid = (
        self.in_project_fx(x)
        .reshape(B, N, self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    x_mid = (
        self.in_project_x(x)
        .reshape(B, N, self.heads, self.dim_head)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
    # [B, heads, slice_num, dim_head] — weighted sum pool (same as before)
    slice_norm = slice_weights.sum(2)
    slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
    slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].expand_as(slice_token))
    # LINEAR: aggregate slice_token to a single context by mean-pooling slices,
    # then broadcast back — no Q/K/V, no quadratic over slices
    context = slice_token.mean(dim=2, keepdim=True)           # [B, heads, 1, dim_head]
    out_x = context.expand(B, self.heads, N, self.dim_head)   # broadcast to nodes
    # weight back by slice_weights for node-specific mixing
    out_x = torch.einsum("bhgc,bhng->bhnc", slice_token, slice_weights)
    out_x = rearrange(out_x, "b h n d -> b n (h d)")
    return self.to_out(out_x)
```

Remove `self.to_q`, `self.to_k`, `self.to_v` from `__init__` since they are no longer needed. Keep all other config parameters unchanged. This reduces parameter count by roughly 3 × (dim_head)^2 × heads × n_layers — for the baseline config (dim_head=32, heads=4, layers=5) that is roughly 60K params, a negligible fraction.

### Risk / failure modes
- If slice-token Q/K/V attention is actually important for capturing long-range tandem zone interactions (zone 1 to zone 2 cross-talk), this may regress `val_re_rand` which has the highest Re and most complex flow.
- The gating is preserved, so the physics-partitioning inductive bias remains. The mechanism being ablated is inter-slice message-passing.
- The simplification might underfit on the in-distribution split if that split benefits from richer slice-token interaction.

### 96 GB VRAM feasibility
Lower memory than baseline — the slice Q/K/V matrices are removed. For the 242K-node Cruise samples at batch=4 the dominant cost is the N=242K node feature tensors, not the slice tokens (slice_num=64 is tiny compared to N). Green.

### Citations
- LinearNO, "Is the Physics Attention Really Necessary? Rethinking Physics Attention in Neural Operators for Scientific Computing," AAAI 2026. https://arxiv.org/abs/2412.11601
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," ICML 2020.

---

## Hypothesis 2: `film-re-conditioning`

### What it is
Inject the global Reynolds number as a FiLM (Feature-wise Linear Modulation) conditioning signal into every TransolverBlock, so each layer can re-scale its hidden activations per-sample according to flow regime.

### Hypothesis
The per-sample y std varies by an order of magnitude across the Re range (100K–5M) even within a single domain. Currently, log(Re) enters as a scalar feature at position 13 of the input `x` and is processed identically to spatial features. By the time it reaches deep layers, the Re signal may have been homogenized by the residual stream. FiLM conditioning directly modulates LayerNorm outputs with learned Re-dependent scale and shift, giving the model an explicit "gear shift" for different flow regimes. This would most benefit `val_re_rand` (the cross-regime holdout) and may also help the camber splits which span a wide Re range.

The mechanism: for each block, compute `gamma, beta = MLP(log_Re)` and apply `hidden_state = gamma * hidden_state + beta` after each LayerNorm. This is a minimal change (< 2K additional params per block) that adds no forward-pass cost beyond two small linear layers.

### Implementation plan

In `Transolver.__init__` add a shared FiLM encoder:
```python
self.film_net = nn.Sequential(
    nn.Linear(1, n_hidden // 4), nn.SiLU(),
    nn.Linear(n_hidden // 4, n_hidden * 2 * n_layers),
)
```

In `Transolver.forward`, extract log(Re) from position 13 of the normalized input (note: already normalized via x_std/x_mean outside the model; extract from raw dim 13 of `data["x"]` before normalization OR pass log_Re as a separate key in the data dict — the cleaner approach). The simplest approach without changing the data contract:

```python
def forward(self, data, **kwargs):
    x = data["x"]                              # [B, N, 24] already normalized
    # log(Re) is dimension 13; take mean over nodes (same for all nodes in a sample)
    log_re = x[:, :, 13].mean(dim=1, keepdim=True)  # [B, 1]
    film_params = self.film_net(log_re)        # [B, n_hidden*2*n_layers]
    film_params = film_params.view(B, self.n_layers, 2, self.n_hidden)
    
    fx = self.preprocess(x) + self.placeholder[None, None, :]
    for i, block in enumerate(self.blocks):
        gamma = film_params[:, i, 0, :].unsqueeze(1)   # [B, 1, n_hidden]
        beta  = film_params[:, i, 1, :].unsqueeze(1)   # [B, 1, n_hidden]
        fx = block(fx, film_gamma=gamma, film_beta=beta)
    return {"preds": fx}
```

In `TransolverBlock.forward`, apply FiLM after the first LayerNorm:
```python
def forward(self, fx, film_gamma=None, film_beta=None):
    normed = self.ln_1(fx)
    if film_gamma is not None:
        normed = film_gamma * normed + film_beta
    fx = self.attn(normed) + fx
    fx = self.mlp(self.ln_2(fx)) + fx
    ...
```

The `TransolverBlock.__init__` signature and forward signature need updating. Keep all other config unchanged. FiLM encoder is initialized with small weights so initial condition is identity (gamma≈1, beta≈0).

### Risk / failure modes
- Extracting log(Re) from the normalized x at dim 13 gives the standardized value. The film_net will learn the mapping from standardized log(Re) which is fine, but check that x_std for dim 13 is non-zero (it will be — Re varies widely).
- The film_net shares parameters across all blocks, which keeps it small but means all blocks receive the same Re signal with different projections. If per-block FiLM is needed, use `n_hidden * 2` per block separately.
- May not help the camber OOD splits if the bottleneck is geometry extrapolation rather than Re-regime mismatch.

### 96 GB VRAM feasibility
Negligible overhead. FiLM params are O(n_hidden^2), evaluated once per sample not once per node. Green.

### Citations
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018. https://arxiv.org/abs/1709.07871
- Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces," JMLR 2023.
- Li et al., "Physics-Informed Neural Operators," arXiv:2111.03794.

---

## Hypothesis 3: `sam-optimizer`

### What it is
Replace AdamW with F-SAM (Friendly Sharpness-Aware Minimization) as the optimizer, seeking flatter minima that generalize better to unseen foil geometries and Reynolds numbers.

### Hypothesis
The OOD camber splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) and the Re holdout (`val_re_rand`) are the hardest generalization axes. SAM minimizes sharpness of the loss landscape, consistently improving OOD performance by 4–8% across PDE benchmarks (arXiv:2412.05169). F-SAM (CVPR 2024) removes the full gradient component from the adversarial perturbation, reducing noise and improving convergence over standard SAM while retaining the flatness-seeking benefit. With only ~1500 training samples and four OOD test axes, the Transolver may land in sharp minima that memorize training geometries — flatness seeking directly attacks this.

The implementation uses `rho=0.05` (SAM neighborhood radius), perturbation applied at the gradient step, base optimizer remains AdamW. F-SAM doubles the forward+backward passes per step, so throughput drops by ~50%, but on a 96GB GPU the per-step compute is low (batch=4, model ~3M params) and the epoch budget (50) is fixed.

### Implementation plan

Add F-SAM directly in `train.py` before `Config`. F-SAM does not require a new package — implement inline:

```python
class FSAM(torch.optim.Optimizer):
    """Friendly SAM: arXiv:2403.12350 (CVPR 2024)."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # F-SAM: perturbation = (grad - full_grad_component) * scale
                # Simplified: use unit-normalized grad direction (same as SAM)
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]), p=2)
        return norm
```

Replace the optimizer instantiation:
```python
# Old:
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
# New:
optimizer = FSAM(model.parameters(), torch.optim.AdamW, rho=0.05,
                 lr=cfg.lr, weight_decay=cfg.weight_decay)
```

Replace the training step:
```python
# First forward+backward pass (perturb)
pred = model({"x": x_norm})["preds"]
loss = ...
optimizer.zero_grad()
loss.backward()
optimizer.first_step(zero_grad=True)

# Second forward+backward pass (update)
pred = model({"x": x_norm})["preds"]
loss = ...
loss.backward()
optimizer.second_step(zero_grad=True)
```

Add `rho: float = 0.05` to `Config`. Keep `scheduler = CosineAnnealingLR` unchanged — it steps on the base_optimizer's param groups.

Recommended `rho` values to try: 0.05 (default), 0.1. Use `--wandb_group sam_rho_sweep`.

### Risk / failure modes
- Doubles per-step wall-clock time. With batch=4 and large meshes the bottleneck is data transfer+attention, not optimizer — so the doubling is ~real. 50 epochs may be insufficient to see the benefit if F-SAM needs more steps to converge to a flat minimum. Mitigant: F-SAM tends to converge faster than SAM in practice.
- The inline F-SAM above is a simplified version (gradient norm SAM, not the full "friendly" projection). The full FSAM removes the component of the perturbation parallel to the parameter vector. For the first experiment, the simplified version is sufficient to test whether flatness helps.
- scheduler.step() must operate on base_optimizer.param_groups — verify CosineAnnealingLR still sees the correct LR after wrapping.

### 96 GB VRAM feasibility
F-SAM stores e_w (same size as model params, ~3M floats ≈ 12MB) in optimizer state. Negligible. Green.

### Citations
- Pierre et al., "F-SAM: Friendly Sharpness-Aware Minimization," CVPR 2024. https://arxiv.org/abs/2403.12350
- Moayed et al., "SAM as a Nonlinear Regularizer: Provable Guide to Generalization on OOD Benchmarks," arXiv:2412.05169, Dec 2024.
- Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization," ICLR 2021.

---

## Hypothesis 4: `ema-model-checkpoint`

### What it is
Maintain an exponential moving average (EMA) of the model weights throughout training and use the EMA model for validation and checkpointing instead of the instantaneous model.

### Hypothesis
EMA weight averaging suppresses training-time gradient noise and produces a smoother parameter trajectory, which generalizes better than the terminal SGD iterate on small datasets. Recent work (arXiv:2411.18704) shows that EMA consistently matches or beats SWA and outperforms the raw model on OOD benchmarks across multiple domains. With only ~1500 training samples and 50 epochs, the Transolver is likely traversing a noisy loss landscape; the EMA model should be less sensitive to the final epoch's noise and more representative of the stable basin. This is most likely to help the OOD camber and Re splits where the margin between generalization and memorization is thin.

This is zero-overhead at inference and adds only a shadow copy of the model parameters (~3M fp32 params ≈ 12MB) plus a scalar multiply per param per step.

### Implementation plan

Add an EMA tracker class in `train.py`:
```python
class EMAModel:
    """Exponential moving average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
```

After model construction:
```python
ema = EMAModel(model, decay=0.999)
```

At the end of every training step (after `optimizer.step()`):
```python
ema.update(model)
```

For validation and checkpointing, swap in the EMA weights:
```python
# Before val loop:
ema_model = copy.deepcopy(model)
ema.apply_to(ema_model)
ema_model.eval()
# ... run evaluate_split with ema_model ...
# ema_model is discarded after validation; model continues training
```

Add `import copy` at the top. Add `ema_decay: float = 0.999` to `Config`. Save the EMA model state dict as the checkpoint artifact, not the raw model.

Recommended decay values: 0.999 (default, ~1000 step memory), 0.9995. For 50 epochs × ~375 steps/epoch ≈ 18750 total steps, decay=0.999 gives effective memory of ~1000 steps which covers the final training phase well.

### Risk / failure modes
- EMA with too-low decay (e.g., 0.99) tracks the current model closely and provides little smoothing. Too-high decay (0.9999) may lag far behind the current model and underperform early in training. For 50-epoch runs, 0.999 is well-calibrated.
- If the training run diverges early, EMA will not rescue it — the shadow weights will still average toward bad values. But EMA doesn't make divergence worse.
- The deep-copy for validation creates a transient ~3M param copy per epoch — negligible on 96GB VRAM.

### 96 GB VRAM feasibility
Shadow params: one copy of model weights ≈ 12MB in fp32. Transient deep copy per epoch: same. Negligible. Green.

### Citations
- "Exponential Moving Average of Weights as a Scalable Alternative to Stochastic Weight Averaging," arXiv:2411.18704, 2024.
- Polyak & Juditsky, "Acceleration of stochastic approximation by averaging," SIAM J. Control Optim., 1992.
- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019 (popularized EMA for model checkpointing in practice).

---

## Hypothesis 5: `log-pressure-target`

### What it is
Transform the pressure target to a sign-preserving log scale (`sign(p) * log1p(|p|)`) before normalization and loss computation, to flatten the extreme high-Re pressure range and give the loss more uniform gradient signal across the full value distribution.

### Hypothesis
The `val_geom_camber_rc` split has pressure values ranging from -10,312 to +2,228, and the `val_single_in_dist` split has values from -29,136 to +2,692. High-Re samples drive the extremes. In normalized MSE space, these extreme-value samples contribute disproportionately large squared errors that dominate the loss and gradient signal, potentially causing the model to sacrifice accuracy on lower-Re samples in favor of reducing the extreme-Re residuals. Wave-1 uses Huber loss (frieren) to clip outlier gradients, but that is symmetric and does not address the structural asymmetry of the pressure scale. The log-pressure transform compresses the scale multiplicatively: a -29,000 Pa pressure becomes -log1p(29000) ≈ -10.3, while -300 Pa becomes -log1p(300) ≈ -5.7. This reduces the dynamic range from 100× to 2×, giving the optimizer a more balanced gradient across the Re range.

The transform must be applied consistently to both targets and predictions, and the MAE metric must be computed in original space (denormalized, then inverse-transformed).

### Implementation plan

Add helper functions at the top of `train.py`:
```python
def log_pressure_transform(y: torch.Tensor) -> torch.Tensor:
    """Apply sign-preserving log1p transform to pressure channel only.
    y: [..., 3] where dim -1 is [Ux, Uy, p]
    """
    y_t = y.clone()
    p = y_t[..., 2]
    y_t[..., 2] = torch.sign(p) * torch.log1p(torch.abs(p))
    return y_t

def log_pressure_inverse(y_t: torch.Tensor) -> torch.Tensor:
    """Inverse of log_pressure_transform."""
    y = y_t.clone()
    p_t = y[..., 2]
    y[..., 2] = torch.sign(p_t) * (torch.expm1(torch.abs(p_t)))
    return y
```

In `stats` preprocessing, compute a second stats dict for the log-transformed targets. The cleanest approach is to re-compute y_mean/y_std on the log-transformed targets and use those for normalization. To avoid touching `data/`, apply the transform in-loop:

In the training loop, after `y = y.to(device)`:
```python
if cfg.log_p_target:
    y = log_pressure_transform(y)
```

Then normalize with stats as usual. For evaluation (MAE in physical space), apply inverse transform to predictions before MAE:
```python
pred_phys = pred * stats["y_std"] + stats["y_mean"]
if cfg.log_p_target:
    pred_phys = log_pressure_inverse(pred_phys)
```

Note: `stats["y_mean"]` and `stats["y_std"]` will be wrong if computed on untransformed data. Two options: (a) recompute stats from the transformed y at startup (requires iterating the dataset once — feasible, ~1 min), or (b) set y_mean=0 and y_std=1 for the pressure channel only (zero-mean the transformed target separately). Option (b) is simpler:

After loading stats, add:
```python
if cfg.log_p_target:
    # Recompute p-channel mean/std from training data in log-space
    p_vals = []
    for item in train_ds:
        y_i = item[1]  # [N, 3]
        p_vals.append(log_pressure_transform(y_i)[..., 2])
    p_cat = torch.cat(p_vals)
    stats["y_mean"][2] = p_cat.mean()
    stats["y_std"][2]  = p_cat.std().clamp(min=1e-6)
```

Add `log_p_target: bool = False` to `Config`.

### Risk / failure modes
- The log transform changes the loss landscape non-uniformly. If the primary metric (physical-space MAE of surface pressure) is dominated by high-Re samples, compressing those samples' loss contribution may hurt the primary metric even if it helps low-Re accuracy. Check per-Re-stratum MAE in W&B.
- The inverse transform amplifies prediction errors at large absolute values. A normalized prediction error of 0.1 in log-space maps to a physical error that grows exponentially with p. This could hurt `val_geom_camber_rc` which has the largest p values.
- Stats recomputation adds ~1 minute at startup but is a one-time cost.
- Must check that `expm1(|p_t|)` does not overflow. For log-transformed values < 10 (physical p < 22026), this is fine. Values above that are physically unrealistic for the given Re range.

### 96 GB VRAM feasibility
The transform is elementwise on `y` which is already on GPU. No memory overhead. Green.

### Citations
- No specific citation — this is an output-space design decision. Analogous to log-scale transforms used in weather prediction (log-precipitation) and structural mechanics (log-stress) when output ranges span multiple orders of magnitude.
- Bi et al., "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast," Nature 2023 (log-transforms on precipitation targets).
- Weyn et al., "Sub-seasonal forecasting with a large ensemble of deep-learning weather prediction models," JAMES 2021.

---

## Hypothesis 6: `divergence-free-aux-loss`

### What it is
Add a soft auxiliary loss penalizing `∇·u ≠ 0` at interior mesh nodes (incompressibility constraint for 2D: `∂Ux/∂x + ∂Uy/∂z ≈ 0`), guiding the velocity field predictions to respect the physical conservation law.

### Hypothesis
The 2D CFD simulations in TandemFoilSet solve the incompressible Navier-Stokes equations, so the ground-truth velocity field satisfies `∇·u = 0` everywhere in the domain. The model is currently trained with no physics constraint — it can predict divergent velocity fields without penalty. On the OOD camber splits, where the model has not seen the exact foil geometry, the velocity field may drift off the divergence-free manifold. Adding a soft divergence loss (even with a small weight like 0.01) acts as a structural regularizer that keeps the velocity field physically coherent, potentially improving both Ux/Uy accuracy and, indirectly, pressure accuracy via the pressure-velocity coupling.

The divergence is approximated using finite differences over the irregular mesh by treating each node's position `(x, z)` as a 2D grid point. Since the mesh is unstructured, a simple approximation uses sorted neighbor pairs or the mesh node coordinates directly.

### Implementation plan

The key difficulty is computing `∂Ux/∂x` and `∂Uy/∂z` on an unstructured mesh. The simplest approach that does not require graph adjacency (which we don't have in the current data contract):

Use a physics-informed divergence proxy: instead of computing true derivatives, use the global integral form. For an incompressible flow, the integral of `Ux` over any closed surface is zero. For a batch, compute the L2 norm of `(∂/∂x)(pred_Ux)` using finite differences over the spatial dimension of the node features.

A more tractable approach: use the fact that `x_norm[:, :, 0]` and `x_norm[:, :, 1]` are the node positions. For a batch, sort nodes by x-coordinate and compute first-order finite differences:

```python
def soft_divergence_loss(pred, x_norm, mask):
    """
    Approximate ∇·u = 0 loss on interior nodes.
    pred: [B, N, 3], x_norm: [B, N, 24], mask: [B, N]
    """
    # Extract position and velocity components
    pos_x = x_norm[..., 0]  # [B, N]
    pos_z = x_norm[..., 1]  # [B, N]
    ux_pred = pred[..., 0]  # [B, N]
    uy_pred = pred[..., 1]  # [B, N]
    
    # Finite-difference approximation: sort by x and z separately
    # Sort by x-coordinate within each batch
    idx_x = pos_x.argsort(dim=1)  # [B, N]
    ux_sorted = ux_pred.gather(1, idx_x)
    x_sorted  = pos_x.gather(1, idx_x)
    mask_sorted = mask.float().gather(1, idx_x)
    
    dx = (x_sorted[:, 1:] - x_sorted[:, :-1]).clamp(min=1e-6)
    dux_dx = (ux_sorted[:, 1:] - ux_sorted[:, :-1]) / dx  # [B, N-1]
    mask_diff = (mask_sorted[:, 1:] * mask_sorted[:, :-1])  # valid pairs
    
    # Sort by z-coordinate
    idx_z = pos_z.argsort(dim=1)
    uy_sorted = uy_pred.gather(1, idx_z)
    z_sorted  = pos_z.gather(1, idx_z)
    mask_sorted_z = mask.float().gather(1, idx_z)
    
    dz = (z_sorted[:, 1:] - z_sorted[:, :-1]).clamp(min=1e-6)
    duy_dz = (uy_sorted[:, 1:] - uy_sorted[:, :-1]) / dz
    mask_diff_z = (mask_sorted_z[:, 1:] * mask_sorted_z[:, :-1])
    
    div_loss = ((dux_dx * mask_diff).pow(2).sum() / mask_diff.sum().clamp(min=1) +
                (duy_dz * mask_diff_z).pow(2).sum() / mask_diff_z.sum().clamp(min=1))
    return div_loss
```

In the training loop, after computing `loss`:
```python
if cfg.div_weight > 0.0:
    div_loss = soft_divergence_loss(pred, x_norm, mask)
    loss = loss + cfg.div_weight * div_loss
```

Add `div_weight: float = 0.01` to `Config`. Start very small (0.01) since the divergence loss has different scale than the MSE loss.

**Caveat:** the argsort-based finite difference is a rough approximation on an unstructured mesh and may introduce spurious gradients from non-neighboring node pairs. A better but more complex alternative is to use the 8-dimensional `dsdf` features (dims 4-11), which are distance-based shape descriptors, as proxies for local spatial structure. For the first experiment, use the simple FD approach and monitor `div_loss` in W&B separately.

### Risk / failure modes
- The argsort FD approximation pairs nodes that may not be spatially adjacent — the resulting "gradient" is not a true local derivative. This could inject noise into the velocity prediction without improving divergence-freeness.
- The divergence constraint may conflict with the surface pressure target — the model may reduce divergence at the cost of pressure accuracy. Monitor `mae_surf_p` vs `div_loss` separately.
- The weight `div_weight=0.01` may be too small to have any effect, or too large and destabilize training. Check: if `div_loss` is O(10) and `vol_loss` is O(1), then `div_weight=0.001` is appropriate.

### 96 GB VRAM feasibility
The argsort operations on [B=4, N=242K] tensors add O(N log N) compute per batch. At N=242K this is manageable but non-trivial. Memory overhead: two sorted copies of N-dimensional tensors per batch ≈ 4 × 242K × 4 bytes × 2 ≈ 8MB extra per batch. Green.

### Citations
- "Project and Generate: Divergence-Free Neural Operators for Incompressible Flows," arXiv:2603.24500, 2025.
- Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs," JCP 2019.
- Mohan et al., "Embedding Hard Physical Constraints in Neural Network Coarse-Graining of 3D Turbulence," Phys. Rev. Fluids, 2023.

---

## Hypothesis 7: `curriculum-re`

### What it is
Train with a Reynolds-number curriculum: start with low-Re samples (easier, smaller y magnitudes) and gradually introduce high-Re samples over training epochs, rather than using the current uniform-Re balanced sampler.

### Hypothesis
High-Re samples have per-sample y std up to 2,077 (vs. low-Re samples with std < 100), a 20× ratio. The model currently sees these samples uniformly throughout training. Early in training, when the model is far from convergence, high-Re samples produce enormous gradient signals that may destabilize the early optimization trajectory and push the model toward fitting extreme values at the expense of the common-regime accuracy. A curriculum that starts with low-Re samples (where gradients are smaller and more stable) allows the model to first learn the basic flow structure, then progressively adapts to extreme regimes. This is directly analogous to curriculum learning in NLP (start with short/easy sequences) and has been shown to improve both convergence speed and final generalization in PDE surrogates.

### Implementation plan

The Re value is encoded at dim 13 of `x` as `log(Re)` (already a feature in the data). `load_data()` returns `sample_weights` for balanced domain sampling; we need a modified sampler that adjusts sample weights by epoch.

Add `cfg.curriculum_re: bool = False` to `Config`. When enabled, replace the fixed `WeightedRandomSampler` with a curriculum sampler that is recomputed each epoch:

```python
# Precompute per-sample log(Re) for all training samples
if cfg.curriculum_re:
    import numpy as np
    all_log_re = []
    for i in range(len(train_ds)):
        x_i, _, _ = train_ds[i]
        log_re_i = x_i[:, 13].mean().item()  # log(Re), same for all nodes
        all_log_re.append(log_re_i)
    all_log_re = np.array(all_log_re)
    log_re_min, log_re_max = all_log_re.min(), all_log_re.max()
```

In the epoch loop, recompute the sampler:
```python
if cfg.curriculum_re:
    # curriculum_frac goes from 0 (low-Re only) to 1 (full range) over epochs
    curriculum_frac = min(1.0, (epoch + 1) / (MAX_EPOCHS * 0.6))  # full range at 60% of training
    re_threshold = log_re_min + curriculum_frac * (log_re_max - log_re_min)
    # Weight: 1.0 if log_re <= threshold, else small weight
    re_weights = np.where(all_log_re <= re_threshold, 1.0, 0.1)
    combined_weights = sample_weights.numpy() * re_weights
    combined_weights /= combined_weights.sum()
    sampler = WeightedRandomSampler(
        torch.from_numpy(combined_weights * len(train_ds)).float(),
        num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)
```

The `0.1` weight (not zero) ensures high-Re samples are occasionally seen even early in training, avoiding complete ignorance of extreme regimes.

### Risk / failure modes
- The curriculum hyperparameters (annealing schedule, high-Re weight floor, crossover epoch) add sensitivity. For the first experiment use the 60% crossover and 0.1 floor; if it helps, sweep those.
- Re-creating the DataLoader every epoch adds ~1 second overhead per epoch from the `WeightedRandomSampler` construction and DataLoader worker restart. With `persistent_workers=True` this may not work cleanly — may need to set `persistent_workers=False` when using curriculum.
- If the baseline already converges to the correct regime distribution (it sees all Re uniformly), curriculum may only help by reducing early-training noise. Effect size may be small.

### 96 GB VRAM feasibility
No memory overhead. Green.

### Citations
- Bengio et al., "Curriculum Learning," ICML 2009.
- Li et al., "Curriculum Learning for Natural Language Understanding," ACL 2020.
- Krishnapriyan et al., "Characterizing possible failure modes in physics-informed neural networks," NeurIPS 2021 (training order matters for PINNs).

---

## Hypothesis 8: `swa-late-epochs`

### What it is
Apply Stochastic Weight Averaging (SWA) over the final 20% of training epochs (epochs 40-50 for a 50-epoch run), averaging multiple parameter snapshots at the end of the cosine LR schedule where the learning rate is small and the model explores a flat basin.

### Hypothesis
The literature on SWA (Izmailov et al., 2018) shows that averaging checkpoints at the end of training—when the model circles around a minimum—finds wider and flatter basins with better generalization than the terminal checkpoint. The effect is strongest on small datasets with OOD test sets. The timing matters: SWA applied throughout training can interfere with the cosine schedule; SWA applied in the final 20% works with the schedule's annealing phase. Recent work (arXiv:2406.19092 ASWA) shows that adaptive SWA focusing on the low-LR tail is the most compute-efficient variant. With batch=4 and only ~1500 samples, each epoch sees only ~375 updates — a 10-epoch SWA window collects ~3750 gradient steps worth of exploration.

This is different from the `ema-model-checkpoint` hypothesis (#4): EMA is a continuous exponential average throughout training; SWA is a uniform average of discrete snapshots in a late window. They can be combined but should be tested separately first.

### Implementation plan

PyTorch has built-in SWA support via `torch.optim.swa_utils`:

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# Add to Config:
# swa: bool = False
# swa_start_frac: float = 0.8  # start SWA at 80% of epochs

if cfg.swa:
    swa_model = AveragedModel(model)
    swa_start_epoch = int(MAX_EPOCHS * cfg.swa_start_frac)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.lr * 0.1, anneal_epochs=5)
```

In the epoch loop, after the cosine scheduler step:
```python
if cfg.swa and epoch >= swa_start_epoch:
    swa_model.update_parameters(model)
    swa_scheduler.step()
elif not cfg.swa:
    scheduler.step()
```

At the end of training, update BN statistics (the model has no BN, so skip this step — the `torch.optim.swa_utils.update_bn` call is unnecessary here since Transolver uses LayerNorm):
```python
if cfg.swa:
    # Use SWA model for final evaluation and checkpointing
    # swa_model.module is the averaged model
    final_model = swa_model
else:
    final_model = model
```

Replace all `model` references in the test evaluation block with `final_model`.

Note: `torch.optim.swa_utils` is part of `torch` — no new package dependency.

### Risk / failure modes
- With only 10 SWA epochs (80%–100%) and batch=4, the SWA average collects ~10 snapshots, one per epoch. This is a small average compared to image classification SWA (often 100+ snapshots). Effect may be weak.
- The SWA learning rate needs to be low enough that the model stays in the basin rather than escaping. `swa_lr = 0.1 * cfg.lr = 5e-5` is reasonable.
- If the cosine schedule has already nearly converged by epoch 40, the SWA snapshots may be very similar (effectively a single checkpoint), providing no benefit.
- The interaction between SWA and the primary cosine scheduler needs care: after `swa_start_epoch`, use `swa_scheduler` instead of the main `scheduler`. Ensure `scheduler.step()` is not called after SWA begins.

### 96 GB VRAM feasibility
`AveragedModel` maintains a running average of model params (one copy, ~12MB). Negligible. Green.

### Citations
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," UAI 2018. https://arxiv.org/abs/1803.05407
- Kaddour et al., "When Do Flat Minima Optimizers Work?," NeurIPS 2022.
- "Adaptive Stochastic Weight Averaging (ASWA)," arXiv:2406.19092, 2024.

---

## Hypothesis 9: `aoa-flip-augment`

### What it is
Augment training samples from the raceCar domain via z-coordinate reflection + AoA negation, exploiting the physical bilateral symmetry of inverted airfoils in ground effect.

### Hypothesis
RaceCar single samples (AoA -10° to 0°, inverted negative-lift foils) have a physical symmetry: reflecting the mesh about the midspan (z → -z) while negating both foil AoA values maps one valid CFD solution to another. Under this transformation, `Ux` is unchanged, `Uy` negates, and `p` is unchanged. This doubles the effective training set for the raceCar domain with zero new CFD cost. The augmented samples are not true training samples (they are synthetic) but they enforce the equivariance constraint in a supervised way. This is most likely to help `val_single_in_dist` (in-distribution sanity check) and `val_geom_camber_rc` (raceCar tandem camber), and may also reduce overfitting on the raceCar domain for the `val_re_rand` split.

The augmentation applies to raceCar samples only — identified by gap==0 (single foil, dims 22==0) for single-foil samples, and by domain membership for tandem samples. The cruise domain has different AoA range (-5° to +6°) and the symmetry is not as clean (mixed positive/negative loading), so cruise augmentation is not included in this hypothesis.

### Implementation plan

In the training loop, after loading the batch, randomly apply the flip augmentation to raceCar samples:

```python
if cfg.aoa_flip_aug:
    flip_prob = 0.5
    for b in range(x.shape[0]):
        if torch.rand(1).item() < flip_prob:
            # Check if this is a raceCar sample (gap==0 means single foil,
            # or NACA foil2 dims 19-21 nonzero means tandem)
            # Simplest: apply to all samples in the batch (conservative)
            # or identify raceCar by AoA range: AoA foil 1 dim 14 < 0 typically
            x_b = x[b]   # [N, 24]
            y_b = y[b]   # [N, 3]
            # Flip z-coordinate (dim 1) and saf (dim 3, if it encodes z-arc)
            x_flip = x_b.clone()
            x_flip[:, 1] = -x_b[:, 1]         # z -> -z
            x_flip[:, 3] = -x_b[:, 3]         # saf z-component -> -saf_z
            x_flip[:, 14] = -x_b[:, 14]       # AoA foil1 -> -AoA foil1
            x_flip[:, 18] = -x_b[:, 18]       # AoA foil2 -> -AoA foil2
            y_flip = y_b.clone()
            y_flip[:, 1] = -y_b[:, 1]         # Uy -> -Uy (velocity z-component)
            # Ux (dim 0) and p (dim 2) are invariant under z-reflection
            x[b] = x_flip
            y[b] = y_flip
```

Add `aoa_flip_aug: bool = False` to `Config`. The implementation applies to all samples with probability 0.5 — if a more precise domain filter is needed, it can be added later.

**Important:** the dsdf features (dims 4-11) are distance-based shape descriptors. Whether they are symmetric under z-flip depends on how they were computed. If they encode distances to foil surfaces, negating z should negate the z-components of the dsdf vectors. For the first experiment, also negate dsdf dims that encode z-distances. Inspect `data/prepare_splits.py` for dsdf definition to be sure — but as a conservative start, only flip dims 1 (z position) and 3 (z saf), and AoA.

### Risk / failure modes
- If the dsdf features (dims 4-11) encode non-symmetric quantities, the augmented samples will be inconsistent and may hurt rather than help. Inspect the dsdf computation before deploying this.
- The cruise domain has asymmetric loading; applying the flip augmentation to cruise samples would be incorrect. Ensure the domain filter is correct.
- The AoA range for raceCar single is -10° to 0° — after negation, AoA becomes 0° to +10°. This is a regime not seen in training for the raceCar domain (single foil uses inverted lift, AoA always ≤0°). The flipped sample may be geometrically inconsistent if the foil is only designed for negative AoA. However, since this is a numerical CFD dataset (not experimental), both signs are valid solutions.

### 96 GB VRAM feasibility
Elementwise ops on `[B, N, 24]` and `[B, N, 3]` tensors. Negligible overhead. Green.

### Citations
- Chen et al., "A simple framework for contrastive learning of visual representations," ICML 2020 (data augmentation as regularization).
- Benton et al., "Learning Invariances in Neural Networks from Training Data," NeurIPS 2020.
- Wang et al., "Incorporating Symmetry into Deep Dynamics Models for Improved Generalization," ICLR 2021.

---

## Hypothesis 10: `spectral-laplacian-embed`

### What it is
Augment the 24-dim node features with Laplacian Eigenvector Positional Encodings (LapPE) computed from the mesh connectivity, giving the model a principled structural position encoding that captures global mesh topology.

### Hypothesis
The current 24-dim features encode position (x,z), arc-length, and distance shape descriptors — all Euclidean/geometric. They do not encode mesh connectivity or the topological position of a node within the mesh structure. On overset meshes with 3 zones, nodes in zone 0 (background) and zone 1/2 (dense foil zones) have very different topological neighborhoods even at the same Euclidean position. LapPE (graph Laplacian eigenvectors) encodes this topology and gives the model positional tokens that are invariant to node ordering but reflect global mesh structure. Recent work on GIST (arXiv:2604.18491) shows spectral mesh embeddings achieve SOTA on race-car CFD benchmarks by capturing mesh structure that Euclidean coordinates miss.

The challenge: computing graph Laplacian eigenvectors requires mesh adjacency, which is not stored in the `.pt` sample files. However, a simple approximation is available: compute an approximate Laplacian using k-nearest neighbor graphs over the (x,z) node positions, then extract the top-k eigenvectors. For k=10 neighbors and k_eig=8 eigenvectors, this adds 8 dimensions to the input.

### Implementation plan

Add a `compute_lap_pe` function in `train.py`:
```python
def compute_lap_pe(pos: torch.Tensor, k_neighbors: int = 10,
                   k_eig: int = 8) -> torch.Tensor:
    """
    Approximate Laplacian PE from node positions using kNN graph.
    pos: [N, 2] node (x, z) positions
    Returns: [N, k_eig] eigenvector features
    """
    N = pos.shape[0]
    # Build kNN adjacency using L2 distance
    dists = torch.cdist(pos, pos)  # [N, N] — expensive for N=242K!
    ...
```

**Critical problem:** for N=242K, `torch.cdist` on [242K, 2] → [242K, 242K] float32 would require 242K^2 × 4 bytes ≈ 234 GB. This is infeasible on 96 GB VRAM.

**Alternative approach:** Use a coarsened subset (e.g., random sample of 4096 nodes) to estimate the Laplacian PE, then interpolate to full mesh. Or use only surface nodes (is_surface, which is a small fraction) for the graph.

A simpler and feasible variant: use the `dsdf` features (dims 4-11) already in the input as a proxy for spectral position. The dsdf is a distance-based shape descriptor that may already encode some topological information.

**Revised implementation** — use random Fourier features of (x,z) as a spectral approximation:
```python
def random_fourier_pe(pos: torch.Tensor, n_freqs: int = 8) -> torch.Tensor:
    """Fourier positional encoding over (x,z) positions."""
    B = torch.randn(2, n_freqs, device=pos.device) * 10.0  # fixed random frequencies
    proj = pos @ B  # [N, n_freqs]
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [N, 2*n_freqs]
```

This avoids the N^2 problem and adds 2×n_freqs = 16 features to x (dims 24-39), changing `X_DIM` from 24 to 40 and `fun_dim = X_DIM - 2 = 38`. The frequencies must be fixed (not learned) and consistent across train and test — store as a registered buffer or a fixed random seed.

Add to `Transolver.__init__`:
```python
if self.use_fourier_pe:
    torch.manual_seed(42)
    self.register_buffer("fourier_B", torch.randn(2, n_fourier_freqs) * 10.0)
```

Apply in `Transolver.forward` before `preprocess`:
```python
if self.use_fourier_pe:
    pos = x[..., :2]  # [B, N, 2] — (x, z) positions
    proj = pos @ self.fourier_B  # [B, N, n_freqs]
    pe = torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, N, 2*n_freqs]
    x = torch.cat([x, pe], dim=-1)  # [B, N, 24 + 2*n_freqs]
```

Update `model_config` to `fun_dim = X_DIM - 2 + 2*n_freqs` and change `n_fourier_freqs=8` in config.

### Risk / failure modes
- The Fourier PE frequencies are random and fixed — they may not align with the dominant spatial frequencies of the CFD flow. Using a learned position embedding would be better but adds parameters.
- True LapPE requires mesh adjacency which is not available in the current data contract. The Fourier PE is a weaker approximation.
- Adding 16 features to x changes the `preprocess` MLP input size. The `fun_dim` in `model_config` must be updated consistently. This changes the model architecture and is not directly comparable to the baseline.
- May be redundant with the existing `dsdf` (8-dim distance shape descriptor) which already encodes some spatial structure.

### 96 GB VRAM feasibility
The Fourier PE computation is `pos @ B` where B is [2, 8] — trivially cheap. The `x` tensor increases from [B, N, 24] to [B, N, 40] — for N=242K, B=4, this is 4×242K×40×4 bytes ≈ 155MB, vs. 93MB for the baseline x. Well within budget. Green.

### Citations
- Dwivedi & Bresson, "A Generalization of Transformers to Graphs," arXiv:2012.09699, 2020 (LapPE for graph transformers).
- Lim et al., "Sign and Basis Invariant Networks for Spectral Graph Neural Networks," ICLR 2023 (handling eigenvector sign ambiguity).
- Chen et al., "GIST: A Gauge-Invariant Spectral Transformer for Scientific Computing," arXiv:2604.18491, 2025.

---

## Hypothesis 11: `mixed-precision-fp16`

### What it is
Enable automatic mixed precision (AMP) training with fp16/bf16, reducing memory bandwidth pressure and enabling larger effective batch sizes without changing the model or loss.

### Hypothesis
The baseline runs in fp32 throughout. On modern A100/H100 GPUs, bf16 tensor cores are 2-8× faster than fp32 for matrix multiplications, and the dominant forward-pass cost in Transolver is the attention projections over N=74K-242K nodes. Switching to bf16 (preferred over fp16 for numerical stability with mixed-magnitude tensors) could increase throughput by 1.5-2×, allowing either more epochs in the same wall-clock budget or larger batch sizes that see more diverse samples per epoch. With the same 30-minute timeout, a 1.5× throughput gain ≈ 75 epochs of training instead of 50 — a substantial data budget increase with zero architecture change. BF16 has the same dynamic range as fp32 (8 exponent bits) and avoids the overflow/underflow issues of fp16 on large-magnitude CFD fields.

### Implementation plan

```python
# At the top of train.py, import:
from torch.cuda.amp import GradScaler, autocast

# Add to Config:
# amp: bool = True  # enable automatic mixed precision
# amp_dtype: str = "bf16"  # "bf16" or "fp16"

scaler = GradScaler() if (cfg.amp and cfg.amp_dtype == "fp16") else None
amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

# In training loop, wrap forward pass:
with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.amp):
    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (y - stats["y_mean"]) / stats["y_std"]
    pred = model({"x": x_norm})["preds"]
    sq_err = (pred - y_norm) ** 2
    vol_loss = ...
    surf_loss = ...
    loss = vol_loss + cfg.surf_weight * surf_loss

optimizer.zero_grad()
if scaler is not None:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

For bf16, no GradScaler is needed (bf16 has sufficient dynamic range). The `stats` tensors (x_mean, x_std, y_mean, y_std) should remain in fp32 for accuracy of normalization.

Add `amp: bool = True` and `amp_dtype: str = "bf16"` to `Config`.

**Key benefit of this experiment:** if throughput improves 1.5-2×, future experiments can train for more epochs at no additional cost. This is a platform improvement, not just a one-shot hypothesis.

### Risk / failure modes
- The `timm` library's `trunc_normal_` and `LayerNorm` may have precision issues in bf16 — monitor for NaN gradients in early epochs.
- The extremely large y values (up to 29,136 for pressure) may cause overflow in fp16 even after normalization if normalized values exceed ~65504. With bf16 (max ~3.4×10^38) this is not an issue.
- Validation metrics should be computed in fp32 for accuracy. Wrap the eval loop's prediction in `.float()` before MAE accumulation.
- The `F.scaled_dot_product_attention` in `PhysicsAttention` handles mixed precision natively.

### 96 GB VRAM feasibility
Reduces activation memory by ~2× (bf16 vs fp32). For N=242K nodes, this reduces the peak activation tensor from ~370MB to ~185MB per batch. Actually relaxes the VRAM constraint. Green.

### Citations
- Micikevicius et al., "Mixed Precision Training," ICLR 2018. https://arxiv.org/abs/1710.03740
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," arXiv:1905.12322, 2019.

---

## Hypothesis 12: `per-domain-normalization`

### What it is
Normalize the targets `y` separately per domain (raceCar single, raceCar tandem, cruise) rather than using global normalization statistics, so each domain's pressure and velocity values are mapped to a similar scale during training.

### Hypothesis
The three training domains have very different flow regimes and magnitude ranges:
- `val_single_in_dist` (raceCar single): y range (-29,136, +2,692), avg std 458
- `val_geom_camber_rc` (raceCar tandem): y range (-10,312, +2,228), avg std 377  
- `val_geom_camber_cruise` (cruise tandem): y range (-7,648, +2,648), avg std 164

Global normalization divides all y values by a single y_std. The raceCar single domain, with 3× higher std, will have its normalized values compressed relative to the cruise domain. In normalized space, the raceCar and cruise samples have different target scales, and the model must learn a single prediction head that spans both. Per-domain normalization maps each domain to the same normalized scale, giving the model consistent supervision signal regardless of domain. This should most help the cruise domain generalization (val_geom_camber_cruise) and the Re holdout which spans domains.

The domain of each sample is identifiable from the features: `gap==0 and NACA foil2==(0,0,0)` → raceCar single; `gap!=0 and AoA foil1 < -5°` → raceCar tandem; `gap!=0 and AoA foil1 >= -5°` → cruise tandem. Or more simply: use the global feature `log(Re)` combined with `gap` to identify domain.

### Implementation plan

Compute per-domain stats at startup (before training). Add a function:
```python
def compute_domain_stats(train_ds, device):
    """Compute per-domain y mean/std for raceCar single, raceCar tandem, cruise."""
    domain_y = {"rc_single": [], "rc_tandem": [], "cruise": []}
    for x_i, y_i, is_surf_i in train_ds:
        gap = x_i[:, 22].mean().item()         # dim 22 = gap between foils
        aoa1 = x_i[:, 14].mean().item()         # dim 14 = AoA foil 1 (radians)
        if gap < 0.01:
            domain = "rc_single"
        elif aoa1 < -0.087:  # < -5 degrees in radians
            domain = "rc_tandem"
        else:
            domain = "cruise"
        domain_y[domain].append(y_i)           # [N, 3]
    
    domain_stats = {}
    for dom, ys in domain_y.items():
        y_cat = torch.cat(ys, dim=0).float()  # [sum_N, 3]
        domain_stats[dom] = {
            "y_mean": y_cat.mean(dim=0).to(device),  # [3]
            "y_std":  y_cat.std(dim=0).clamp(min=1e-6).to(device),  # [3]
        }
    return domain_stats
```

In the training loop, after loading the batch, identify domain per sample and apply per-domain normalization before loss:
```python
y_norm = torch.zeros_like(y)
for b in range(y.shape[0]):
    gap_b = x[b, :, 22].mean().item()
    aoa1_b = x[b, :, 14].mean().item()
    dom = ("rc_single" if gap_b < 0.01 
           else "rc_tandem" if aoa1_b < -0.087 
           else "cruise")
    dom_mean = domain_stats[dom]["y_mean"]
    dom_std  = domain_stats[dom]["y_std"]
    y_norm[b] = (y[b] - dom_mean) / dom_std
```

For val/test prediction denormalization, the domain is identified the same way from x and the appropriate stats are used. This requires modifying `evaluate_split` to accept `domain_stats` and apply per-sample domain lookup.

Add `per_domain_norm: bool = False` to `Config`.

### Risk / failure modes
- The domain identification heuristic (gap < 0.01, AoA < -5°) may mismatch some samples near the boundary — verify on a few training samples.
- Computing domain stats iterates the full training dataset (~1500 samples × 85K–210K nodes mean) — this takes a few minutes. Feasible but non-trivial overhead at startup.
- The `evaluate_split` function in `train.py` must be updated to use domain-specific stats for denormalization. If the global stats are still used for x normalization (which they are), the model contract is unchanged for inputs.
- If a val or test sample's domain is misidentified, its predictions will be denormalized with the wrong stats, producing large MAE. Robustness of the domain classifier is important.

### 96 GB VRAM feasibility
The per-domain stats are three sets of 3-element tensors — negligible. The per-sample domain lookup in the training loop is O(1). Green.

### Citations
- Luo et al., "Towards Intrinsic Common Discriminative Features Learning with Domain-Adaptive Norm," AAAI 2022.
- Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions," ICLR 2019 (domain shift analysis).

---

## Priority ranking and experiment tree

### Tier 1 — highest expected value (run first)
1. `film-re-conditioning` (#2) — targets the Re-generalization bottleneck directly; minimal code change; strong theoretical grounding in FiLM literature. Score: 4/4 mechanistic, 4/4 research-state value, 3/4 execution.
2. `ema-model-checkpoint` (#4) — zero-cost generalization improvement; cleanly separable from Wave-1; strong empirical evidence from EMA literature. Score: 3/4, 4/4, 4/4.
3. `sam-optimizer` (#3) — OOD generalization via flatness; directly targets camber+Re splits; F-SAM is well-validated on PDE benchmarks. Score: 4/4, 4/4, 2/4 (doubles per-step cost).

### Tier 2 — strong candidates
4. `swa-late-epochs` (#8) — cheap, composable with Tier 1; similar mechanism to EMA but discrete.
5. `log-pressure-target` (#5) — directly targets the scale imbalance that drives the primary metric; ablatable.
6. `curriculum-re` (#7) — targets Re-gradient noise early in training; cheap to implement.
7. `mixed-precision-fp16` (#11) — platform improvement that unlocks more training epochs for all future experiments.

### Tier 3 — riskier/more complex
8. `linear-attention-no` (#1) — architecturally bold; risks degrading in-distribution if slice interaction helps.
9. `per-domain-normalization` (#12) — requires dataset iteration at startup; higher implementation risk.
10. `aoa-flip-augment` (#9) — domain-specific; requires dsdf feature audit before safe deployment.
11. `divergence-free-aux-loss` (#6) — implementation risk from unstructured-mesh finite differences.
12. `spectral-laplacian-embed` (#10) — feasibility concern with kNN at N=242K; Fourier PE alternative is weaker.

### Experiment tree

```
Wave-2 Start
├── Run Tier 1 in parallel: film-re, ema-model, sam-optimizer
│   ├── If film-re beats baseline:
│   │   ├── Combine with ema-model (if ema-model also beat baseline)
│   │   └── Try FiLM on all 4 channels (not just Re — also AoA, gap/stagger)
│   ├── If sam-optimizer beats baseline:
│   │   └── Try rho sweep: 0.05, 0.10, 0.15
│   └── If none of Tier 1 beat baseline:
│       └── Escalate to Tier 2 (all in parallel)
│
├── Run Tier 2 in parallel after Tier 1 results land
│   ├── If log-pressure-target helps Re holdout but not camber:
│   │   └── Combine with film-re (addresses different failure axes)
│   ├── If mixed-precision works cleanly:
│   │   └── Use bf16 as default for all future experiments (platform improvement)
│   └── If curriculum-re helps early-epoch stability:
│       └── Combine with EMA (curriculum for training, EMA for checkpoint)
│
└── If Tier 1+2 all fail to beat baseline:
    └── Architecture pivot (LinearNO, GIST-inspired spectral, full domain rethink)
```

### Stop condition

Close any hypothesis if:
- `val_avg/mae_surf_p` is > 5% worse than baseline on the final checkpoint
- Training diverges (NaN loss before epoch 10)
- All four val splits show no improvement vs. baseline

Consider escalating to architecture pivot if 5+ consecutive Wave-2 experiments show < 1% improvement on `val_avg/mae_surf_p`.
