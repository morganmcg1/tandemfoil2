<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Round 2 Research Ideas — TandemFoilSet CFD Surrogate
Date: 2026-04-28

## Context and Constraints

Round 1 levers already running (do NOT duplicate any of these):
- n_hidden 128 → 256
- n_layers 5 → 8
- MSE → L1 loss
- surf_weight 10 → 30
- per-channel p upweighting 3x
- slice_num 64 → 128
- LR warmup 5%
- peak LR 5e-4 → 1e-3
- batch_size 4 → 8

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

All implementation targets `train.py` (the primary editable entrypoint). All other files in `data/` are read-only.

---

## Ranked Hypotheses

### Hypothesis 1 — Relative L2 Loss (Scale-Normalized Loss)

**One-liner:** Divide MSE by per-sample target norm to equalize gradient scale across the 10x Re-driven magnitude variation.

**Expected delta:** -5% to -15% on `val_avg/mae_surf_p`. This is the single change most likely to improve OOD (high-Re) generalization without any architectural modification.

**Mechanism:** The global normalization by `y_std` from `stats.json` does not account for within-batch variation — a single high-Re sample at Re=5M can have `||y||` 10-15x larger than a low-Re sample at Re=100K. The current MSE loss therefore receives gradients dominated by extreme samples. Relative L2 (also called "relative H1" in the FNO literature) normalizes each sample's contribution: `loss_i = ||pred_i - y_i||^2 / (||y_i||^2 + eps)`. This is essentially learning in fractional error space, which aligns with how engineers care about CFD accuracy.

**Implementation (lines 490-496 of train.py):**

Replace:
```python
sq_err = (pred - y_norm) ** 2

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

With:
```python
sq_err = (pred - y_norm) ** 2

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface

# Per-sample denominator in normalized space (shape [B, 1, 3])
# Clamp to avoid dividing by near-zero for trivially small samples
y_norm_sq = (y_norm ** 2 * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
denom = y_norm_sq.clamp(min=1e-4)

vol_loss = (sq_err * vol_mask.unsqueeze(-1) / denom).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1) / denom).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add a CLI flag `--relative_loss` (bool, default False) to the Config dataclass so the baseline MSE path remains testable.

**Risk:** If some samples have near-zero `y_norm` (e.g. very low-Re or peculiar normalization artifacts) the denominator clamp kicks in and reverts to standard MSE for those samples — safe fallback. Could slightly hurt in-distribution accuracy if the extreme-Re samples are actually the hardest and you want high gradient from them. Validate that `val_single_in_dist` does not regress.

**Citations:** Fourier Neural Operator (arXiv:2010.08895) — relative L2 metric and training objective; GINO (arXiv:2207.05209) uses the same formulation; consistent with how the ML4CFD 2024 competition top-3 handled Re-driven scale variation.

---

### Hypothesis 2 — FiLM Conditioning of LayerNorm (Global Scalar Injection)

**One-liner:** Use Feature-wise Linear Modulation (FiLM) to inject global physical parameters (Re, AoA, gap, stagger) directly into each transformer block's LayerNorm, replacing the current implicit injection through the preprocess MLP.

**Expected delta:** -5% to -12% on `val_avg/mae_surf_p`, with the biggest gain on the camber-generalization splits (`val_geom_camber_rc`, `val_geom_camber_cruise`) where the geometry differs from training.

**Mechanism:** Dims 13-23 of x (log(Re), AoA foil 1, NACA foil 1 (3D), AoA foil 2, NACA foil 2 (3D), gap, stagger) are global scalars repeated at every node. Currently these are mixed into the 24-dim preprocess MLP alongside spatial features — the attention mechanism then has to separate physical conditioning from spatial computation. FiLM/AdaLN-Zero projects these 11 scalars to per-block (gamma, beta) pairs that multiplicatively modulate LayerNorm scale and shift, a pattern validated in Stable Diffusion, DiT, and SAR (physics surrogate on Transolver backbone, OpenReview 2025). This is architecturally cleaner: the spatial stream processes geometry, while the physical conditioning stream controls how the layers transform those geometric features.

**Implementation:**

1. Add `ConditioningMLP` class after the MLP class definition:
```python
class ConditioningMLP(nn.Module):
    """Projects global scalars to (gamma, beta) for each TransolverBlock."""
    def __init__(self, cond_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_layers * 2 * hidden_dim),
        )
    def forward(self, cond):  # cond: [B, cond_dim]
        out = self.net(cond)  # [B, n_layers * 2 * hidden_dim]
        return out.reshape(cond.shape[0], self.n_layers, 2, -1)  # [B, L, 2, H]
```

2. Modify `TransolverBlock.__init__` to accept `use_film=False` flag and `TransolverBlock.forward` to accept optional `(gamma, beta)` tensors [B, H]:
```python
def forward(self, fx, gamma=None, beta=None):
    h = self.ln_1(fx)
    if gamma is not None:
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
    fx = self.attn(h) + fx
    h = self.ln_2(fx)
    if gamma is not None:
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
    fx = self.mlp(h) + fx
    ...
```

3. Modify `Transolver.__init__` to add `ConditioningMLP(cond_dim=11, n_layers=n_layers, hidden_dim=n_hidden)`.

4. Modify `Transolver.forward` to extract dims 13-23 from the un-normalized x (or from `data["cond"]` if passed separately), pass through `ConditioningMLP`, then pass per-layer gammas/betas to each block.

5. Modify the train loop to extract dims 13-23 from the raw (un-normalized) x before passing to model: `cond = x[..., 13:24].mean(dim=1)` (mean over nodes since they are constant per sample).

Add a CLI flag `--film_conditioning` (bool, default False). Keep cond_dim=11 to cover all global scalars.

**Risk:** Small parameter increase (~11 * hidden_dim * 2 * n_layers extra weights). Main risk is that the conditioning MLP overfits to training domains if cond_dim features are not well normalized — use the x_norm stats for dims 13-23 when extracting cond features. Verify that grad flow through conditioning path is stable (monitor via grad norms in W&B).

**Citations:** Feature-wise Linear Modulation (FILM, arXiv:1709.07871); AdaLN-Zero in DiT (arXiv:2212.09748); SAR physics surrogate (OpenReview 2025, builds FiLM on Transolver for aerodynamic fields); Hyper-DiffusionNet (ECCV 2024) uses similar conditioning for shape processing.

---

### Hypothesis 3 — Huber Loss with Tuned Delta

**One-liner:** Replace MSE with Huber loss (smooth L1) to reduce sensitivity to the most extreme pressure outliers (stall/separation events at high AoA or high Re) without sacrificing gradient scale near zero.

**Expected delta:** -3% to -8% on `val_avg/mae_surf_p`. Most impactful on the raceCar splits where extreme suction peaks at AoA -10° create large per-node outliers.

**Mechanism:** In normalized prediction space, most errors are O(1) but stall-related peaks can generate errors of O(10-100). MSE squares these, causing the optimizer to dedicate disproportionate capacity to a small number of extreme nodes at the expense of the majority. Huber loss is quadratic for |error| < delta and linear beyond, capping the influence of extreme nodes. Setting delta in normalized space is the key — delta=1.0 means errors larger than 1 normalized unit are treated linearly, which empirically captures the transition between typical and outlier errors for physics surrogates. NeuralFoil (arXiv:2503.16323) demonstrates benefit for aerodynamic surrogates; the ML4CFD 2024 OB-GNN used Huber for robustness.

**Implementation (lines 490-496 of train.py):**

Replace:
```python
sq_err = (pred - y_norm) ** 2
...
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
```

With:
```python
# Huber loss in normalized space, element-wise
err = pred - y_norm
huber_err = torch.where(
    err.abs() < cfg.huber_delta,
    0.5 * err ** 2,
    cfg.huber_delta * (err.abs() - 0.5 * cfg.huber_delta)
)
vol_loss = (huber_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Add to Config dataclass: `huber_delta: float = 1.0`. Add CLI flag `--huber_delta`.

Note: this is distinct from the round 1 "MSE → L1" lever. Pure L1 loses the quadratic basin near zero; Huber gives MSE behavior near zero (important for typical nodes) and L1 behavior for outliers. Try delta values 0.5, 1.0, 2.0.

**Risk:** If delta is set too small it degrades to L1 (already tested in round 1, unclear result). If set too large it degrades to MSE. The sweet spot depends on the typical normalized error magnitude, which should be checked from a round 1 run. Recommend starting with delta=1.0.

**Citations:** Huber (1964, original); F.huber_loss in PyTorch; NeuralFoil arXiv:2503.16323 Sec 4.3 (aerodynamic surrogate loss ablation); Fast.ai course (delta selection heuristics for regression).

---

### Hypothesis 4 — Fourier Positional Encoding for (x, z) Coordinates

**One-liner:** Replace the raw (x, z) node coordinates in dims 0-1 with multi-scale sinusoidal Fourier features before the preprocess MLP to help the model resolve sharp pressure gradients near foil surfaces.

**Expected delta:** -4% to -10% on `val_avg/mae_surf_p`. The pressure field has boundary-layer gradients that are spatially narrow — a linear embedding of (x, z) cannot represent these; Fourier features create high-frequency basis functions that span them.

**Mechanism:** Transformer models are known to struggle with high-frequency spatial variation when coordinates are given as raw reals (Rahaman et al., spectral bias / NTK analysis, arXiv:1806.08734). Fourier features [sin(2^k * pi * x), cos(2^k * pi * x)] for k=0..K-1 create an explicit high-frequency positional basis. The preprocess MLP then maps these to the hidden dim. This is exactly how NeRF handles 3D coordinate learning, and how PhysicsInformedNN (PINN) papers handle coordinate inputs for PDE surrogate training. With K=4 frequency bands for each of x and z, we replace 2 coordinate dims with 4*2*2=16 Fourier features, changing `space_dim=2` to `space_dim=16` (or extending fun_dim).

**Implementation:**

1. Add a `fourier_encode_coords` function before the model class:
```python
def fourier_encode_coords(coords: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    """coords: [..., 2] -> [..., 2 + 2*num_bands*2]"""
    freqs = 2.0 ** torch.arange(num_bands, device=coords.device, dtype=coords.dtype)  # [K]
    # coords: [..., 2], freqs: [K] -> [..., 2, K]
    angles = coords.unsqueeze(-1) * freqs * math.pi  # [..., 2, K]
    encoded = torch.cat([coords, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
    return encoded  # [..., 2 + 4*K]
```

2. In the training loop, after computing `x_norm`, extract and re-encode the coordinate dims:
```python
# x_norm: [B, N, 24], dims 0-1 are normalized (x, z)
coords_encoded = fourier_encode_coords(x_norm[..., :2], num_bands=4)  # [B, N, 2+16]
x_augmented = torch.cat([coords_encoded, x_norm[..., 2:]], dim=-1)  # [B, N, 2+16+22=40]
pred = model({"x": x_augmented})["preds"]
```

3. Update `model_config` in train.py: `space_dim = 2 + 4*2 = 10` (since space_dim is the coordinate portion — but note the Transolver uses `fun_dim + space_dim` as total input dim to preprocess MLP). Specifically, change `space_dim=2` to `space_dim=2 + 4*2` and `fun_dim=X_DIM - 2` stays the same (but the total preprocess input dim changes). Actually simpler: keep `space_dim=2` but increase `fun_dim` to `X_DIM - 2 + 4*2 = 30`, and update x_augmented construction accordingly.

Actually the cleanest hookpoint: add CLI flag `--fourier_bands` (int, default 0 = disabled). When non-zero, augment x_norm with Fourier features of dims 0-1 before passing to model, and update preprocess MLP input dim accordingly. Requires `import math` at top of file (already present via standard imports).

**Risk:** Increases the total preprocess MLP input from 24 to 24+4*2*2=40, adding ~16*n_hidden*2 = ~4K params to the preprocess MLP for n_hidden=128 — negligible. Main risk: if coordinate normalization in x_norm compresses the coordinates to a small range, the high-frequency Fourier features will be near-zero and provide no benefit. Check that normalized x,z span a reasonable range (should be ~[-3, +3] for most nodes). Use num_bands=4 as starting point; try 3 and 6.

**Citations:** NeRF (arXiv:2003.08934) Sec 5.1 — positional encoding motivations; Fourier features for neural fields (arXiv:2006.10739) — Tancik et al.; PINN spectral bias analysis (Rahaman et al., arXiv:1806.08734); TransFlowNet (ICLR 2024 workshop) applies Fourier features to mesh-based aerodynamic surrogates.

---

### Hypothesis 5 — LinearNO: Replace Slice Attention with Linear Attention

**One-liner:** Replace the `scaled_dot_product_attention` call on slice tokens (lines 128-132) with ELU-kernel linear attention, eliminating the O(K^2) softmax step and enabling larger slice counts within the same VRAM budget.

**Expected delta:** -3% to -8% on `val_avg/mae_surf_p` from the combination of better scaling and elimination of the softmax bottleneck. LinearNO achieves SOTA on AirFRANS and ShapeNet Car with 36% less compute vs. standard Transolver.

**Mechanism:** The PhysicsAttention module computes standard QKV attention on K=64 slice tokens (lines 125-132). The softmax forces competitive attention allocation across all K slices — but there is no physical reason slices should compete. LinearNO (arXiv:2511.06294) shows that replacing softmax attention on the slice tokens with linear (ELU+1 kernel) attention maintains or improves accuracy while reducing compute and allowing K to be increased cheaply. The key insight from their ablation: the slice/deslice projection (which compresses N nodes to K slices) is what does the heavy lifting; the attention over slices is secondary. Linear attention also avoids the temperature parameter and is numerically more stable.

**Implementation (PhysicsAttention class, lines 105-136):**

Replace lines 128-132:
```python
out_slice = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=False,
)
```

With:
```python
# ELU linear attention: phi(Q) * (phi(K)^T * V) / (phi(Q) * phi(K)^T.sum())
def elu_feature(x):
    return F.elu(x) + 1.0

q_f = elu_feature(q)   # [B, H, K, d]
k_f = elu_feature(k)   # [B, H, K, d]
# Compute KV context: [B, H, d, d]
kv = torch.einsum("bhkd,bhke->bhde", k_f, v)
# Normalization: [B, H, K, 1]
z = 1.0 / (torch.einsum("bhkd,bhd->bhk", q_f, k_f.sum(dim=2)).unsqueeze(-1) + 1e-6)
# Output: [B, H, K, d]
out_slice = torch.einsum("bhkd,bhde->bhke", q_f, kv) * z
```

Remove the `self.temperature` parameter from `__init__` (it was only used in the softmax path). Add CLI flag `--linear_attn` (bool, default False). Also consider increasing `slice_num` from 64 to 128 or 256 when linear attention is enabled (test whether that further helps — log under same `--wandb_group`).

**Risk:** Linear attention is known to be slightly less expressive than softmax for associative recall tasks. On this physics problem the slice tokens represent spatial averages over mesh regions, not sequences — so the competitive softmax is less motivated, and linear attention's associative memory property may actually be better suited. The main risk is numerical instability if ELU+1 features become very small — the `1e-6` stabilizer in z handles this.

**Citations:** LinearNO (arXiv:2511.06294) — full ablation vs. Transolver on AirFRANS/ShapeNet Car; Linear Transformer (arXiv:2006.16236) — ELU kernel; Performer (arXiv:2009.14794) — random feature approximation of softmax (alternative to ELU, more accurate but more complex).

---

### Hypothesis 6 — POD/PCA Output Reparameterization for Surface Pressure

**One-liner:** Precompute a K=32 PCA basis from training-set surface pressure fields, add a small learned decoder head that maps model output → PCA coefficients → reconstructed surface pressure, and train with a PCA coefficient loss on surface nodes only.

**Expected delta:** -5% to -15% on `val_avg/mae_surf_p`. This was the winning approach in the NeurIPS 2024 ML4CFD competition (MMGP, 1st place). PCA encoding forces the model to predict physically coherent pressure modes rather than independent per-node values, acting as a strong regularizer for OOD geometry generalization.

**Mechanism:** Surface pressure fields lie on a low-dimensional manifold for a family of foil shapes — the first 32 POD modes typically capture >99% of variance in airfoil pressure datasets. By learning to predict in this modal basis, the model: (a) cannot predict physically incoherent high-wavenumber noise, (b) gets stronger gradient signal from modes that matter for lift/drag, (c) generalizes better to unseen camber values because modal shapes vary smoothly with NACA parameters. The MMGP approach (ML4CFD winner) concatenates Re/AoA to PCA coefficients as conditioning, demonstrating that physics conditioning + output reparameterization compound.

**Implementation:**

1. Add a one-time PCA fitting step before the training loop (using sklearn, which is available):
```python
from sklearn.decomposition import PCA
import numpy as np

# Collect surface pressure from all training samples
surf_p_list = []
for x, y, is_surface, mask in DataLoader(train_ds, batch_size=1, shuffle=False):
    surf_p_list.append(y[0, is_surface[0], 2].numpy())  # surface p for each sample

# Fit PCA on surface pressure (samples may have different lengths — pad to max)
max_surf = max(p.shape[0] for p in surf_p_list)
surf_p_padded = np.stack([
    np.pad(p, (0, max_surf - len(p))) for p in surf_p_list
])  # [N_train, max_surf]
pca = PCA(n_components=cfg.pca_k)
pca.fit(surf_p_padded)  # fits on raw (un-normalized) surface pressure
```

2. Add a `PCADecoder` module and a new `--pca_k` flag (int, default 0 = disabled). When enabled: the Transolver head outputs K PCA coefficients for surface nodes, the decoder reconstructs surface pressure, and the loss is MSE on coefficients + reconstruction.

The exact hookpoint is after `pred = model({"x": x_norm})["preds"]`, for surface nodes only. This requires model_config change to add a 4th output or a separate head for coefficients.

**Risk:** PCA basis is computed once on training set surface nodes, but surface node count varies per sample (different meshes). This requires careful padding/masking when projecting. The variable mesh topology is the primary engineering challenge. Consider using PCA on the normalized pressure values and projecting each sample independently. If per-sample surface node counts vary too much, a simpler version is to use PCA on the latent representation of the Transolver (output of last block before the final linear head), which has fixed dimension K=n_hidden — that is effectively what latent-space regularization methods do. Try the simpler latent-space version first.

**Citations:** MMGP (NeurIPS 2024 ML4CFD competition, 1st place, arXiv:2506.08516 Sec 3.2); POD for CFD (Lumley 1967, classic); MeshGraphNets + POD (Sanchez-Gonzalez et al. 2020, DeepMind); ReducedOrderModel survey (Quarteroni et al., Springer 2016).

---

### Hypothesis 7 — Domain-ID Embedding

**One-liner:** Add a 3-way learnable domain embedding (raceCar-single, raceCar-tandem, cruise) derived deterministically from input features and concatenate it to the preprocess MLP input.

**Expected delta:** -2% to -6% on `val_avg/mae_surf_p`, with most gain on the camber-OOD splits where the domain boundary informs the prior on plausible pressure distributions.

**Mechanism:** The three domains have fundamentally different physics: raceCar-single has ground effect (features near z=0), raceCar-tandem has inverted dual foils with negative loading, cruise has positive-lift tandem foils. The model currently must infer domain from the raw features (mostly from dims 18-23 being zero vs. non-zero, and from the NACA geometry). An explicit domain embedding provides a direct shortcut, reduces the burden on the first few layers, and allows domain-specific attention patterns to emerge without the model having to discover the domain boundary implicitly. This is analogous to speaker embeddings in speech synthesis or task embeddings in multi-task learning.

**Implementation:**

1. Add a deterministic `get_domain_id` function:
```python
def get_domain_id(x: torch.Tensor) -> torch.Tensor:
    """x: [B, N, 24] -> domain_id: [B] (0=racecar_single, 1=racecar_tandem, 2=cruise)
    Uses dim 22 (gap) as discriminator: gap==0 -> single; gap!=0 + AoA<0 -> racecar_tandem; else -> cruise.
    dim 18 (AoA foil 2) and dim 22 (gap) are zero for single-foil samples.
    """
    gap = x[:, 0, 22]  # scalar per sample (constant across nodes)
    aoa2 = x[:, 0, 14]  # AoA foil 1, negative for racecar, positive for cruise
    is_single = (gap.abs() < 1e-6)
    is_racecar = (~is_single) & (aoa2 < 0)
    domain_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    domain_id[is_racecar] = 1
    domain_id[~is_single & ~is_racecar] = 2
    return domain_id
```

2. Add a `nn.Embedding(3, domain_embed_dim)` to the Transolver, where `domain_embed_dim=16`. Concatenate the embedding to the preprocess MLP input: `preprocess` takes `fun_dim + space_dim + domain_embed_dim` instead of `fun_dim + space_dim`.

3. In `Transolver.forward`:
```python
domain_id = get_domain_id(data["x_raw"])  # pass raw x, not normalized
domain_emb = self.domain_embedding(domain_id)  # [B, 16]
domain_emb = domain_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, N, 16]
fx = self.preprocess(torch.cat([x, domain_emb], dim=-1)) + self.placeholder[None, None, :]
```

4. In the train loop, pass `data["x_raw"] = x` (un-normalized) alongside `data["x"] = x_norm`.

Add `--domain_embed_dim 16` as CLI flag (0 = disabled). Increase model's preprocess input accordingly.

**Risk:** The deterministic domain ID derivation from gap and AoA is a heuristic that may misclassify edge cases. Test on val splits to verify domain ID assignment is correct for all 4 splits. The main risk is that the domain embedding prevents cross-domain transfer learning. Monitor whether `val_geom_camber_rc` (OOD geometry but same domain as training) improves more than other splits.

**Citations:** Domain embeddings in multi-task learning (Caruana 1997); Conditional ResNets (arXiv:1707.01217); Multi-domain neural operators (Herde et al., arXiv:2407.04121 — "Poseidon").

---

### Hypothesis 8 — Surface-Node Geometric Oversampling in Loss

**One-liner:** During loss computation, independently subsample interior (volume) nodes to 15-20% of their count while retaining all surface nodes, effectively amplifying the surface gradient fraction by 5-6x without changing surf_weight.

**Expected delta:** -3% to -8% on `val_avg/mae_surf_p`. The OB-GNN entry in ML4CFD 2024 used 8x surface node oversampling and won 2nd place. The current `surf_weight=10` achieves a similar effect through loss scaling, but node subsampling also reduces VRAM and allows effectively larger batch size.

**Mechanism:** Surface nodes are ~0.5-1% of total mesh nodes (foil boundary layer). With `surf_weight=10`, the surface loss is 10x upweighted, but the gradient of the volume loss still flows through all ~99% of volume nodes. By randomly dropping 80-85% of volume nodes in each training step, the effective gradient ratio becomes: 15% of 99% volume nodes vs. 100% of 1% surface nodes, approximately 15:1 → 1:1 (surface nodes contribute ~7x as many gradient units as volume). This change targets the physical bottleneck directly rather than through scalar weighting. Additionally, masking 80% of volume nodes reduces forward pass VRAM by ~15-20%, enabling batch_size increase.

**Implementation (train loop, lines 481-506):**

After the batch is loaded and moved to device, add:
```python
# Node subsampling: keep all surface nodes, randomly drop vol_keep_frac of volume nodes
if cfg.vol_keep_frac < 1.0:
    vol_nodes = mask & ~is_surface  # [B, N]
    keep_prob = torch.ones_like(vol_nodes, dtype=torch.float32)
    keep_prob[vol_nodes] = cfg.vol_keep_frac
    keep_mask = torch.bernoulli(keep_prob).bool()
    active_mask = keep_mask | is_surface  # always keep surface
    mask = mask & active_mask  # update mask to exclude dropped vol nodes
    # is_surface unchanged
```

Then the existing loss computation (lines 492-496) automatically uses the reduced mask.

Add to Config: `vol_keep_frac: float = 1.0`. Add CLI flag `--vol_keep_frac 0.15`. Recommend starting with 0.15 (keep 15% of volume nodes).

**Risk:** The model never sees gradients from the dropped volume nodes in a given step, so it must learn volume fields from the subset. Over an epoch all nodes should be seen (in expectation), but with high variance. If batch_size is small (4), some batches may have very few total nodes per sample. Use `vol_keep_frac=0.15` with the larger batch_size configs from round 1 for best results. Note that evaluation always uses the full mesh (mask is unchanged at eval time).

**Citations:** OB-GNN (ML4CFD 2024, 2nd place, arXiv:2506.08516 Sec 3.3) — surface node oversampling; Point-MAE (arXiv:2203.06604) — random node masking for point cloud transformers; DropEdge (arXiv:1907.10903) — random masking for graph neural networks.

---

### Hypothesis 9 — Test-Time Augmentation (Vertical Mirror Symmetry)

**One-liner:** At inference, evaluate each sample twice: once normally and once with z-coordinates flipped and Uy sign-flipped, then average the two predictions to reduce prediction variance.

**Expected delta:** -1% to -4% on `val_avg/mae_surf_p`. Pure inference-time improvement, zero training cost, zero code change to the model.

**Mechanism:** The CFD setup is not strictly mirror-symmetric (ground effect for raceCar breaks z-symmetry, and AoA introduces top/bottom asymmetry), but for many interior volume nodes and some surface nodes, the flow field has approximate vertical symmetry. TTA averaging two plausible predictions reduces random errors from underparameterization. This pattern is standard in medical imaging (flip/rotate TTA) and has been applied to aerodynamic fields in NeuralFoil and similar surrogates. The key: when flipping z, you must also negate Uy (dim 1 of y), negate dim 1 of x (z coordinate), negate dim 2-3 of x (signed arc-length z-component), and negate AoA features (dims 14, 18) since AoA sign reversal corresponds to z-flip for symmetric foils. For inverted raceCar foils, this approximation is less exact, but TTA averaging should still reduce variance.

**Implementation (evaluate_split function or inference wrapper, lines 220+):**

Add a helper:
```python
def tta_flip(x: torch.Tensor, y_pred: torch.Tensor, model, stats, device) -> torch.Tensor:
    """Flip z-coord, get second prediction, average. x: [B, N, 24], y_pred: [B, N, 3]."""
    x_flip = x.clone()
    x_flip[..., 1] = -x_flip[..., 1]    # flip z coordinate (dim 1)
    x_flip[..., 3] = -x_flip[..., 3]    # flip saf z-component (dim 3)
    x_flip[..., 14] = -x_flip[..., 14]  # flip AoA foil 1
    x_flip[..., 18] = -x_flip[..., 18]  # flip AoA foil 2
    x_flip_norm = (x_flip - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        pred_flip = model({"x": x_flip_norm})["preds"]  # [B, N, 3] in normalized space
    # Uy (dim 1 of pred) must be negated back
    pred_flip[..., 1] = -pred_flip[..., 1]
    return 0.5 * (y_pred + pred_flip)
```

Add `--tta` bool flag (default False). In `evaluate_split` (and test evaluation), when `cfg.tta=True`, call this helper after the primary forward pass. Note: TTA is inference-only — do NOT apply during training.

**Risk:** For strongly asymmetric flows (raceCar ground effect at negative AoA) the flipped sample may be physically inconsistent, potentially hurting prediction quality. Monitor per-split improvement: if `val_geom_camber_rc` degrades, the asymmetry is too large for TTA to help on that split. Also verify that the AoA sign flip correctly identifies which input dims to negate — confirm with `program.md` dim table.

**Citations:** TTA for medical imaging (arXiv:2002.01259); NeuralFoil (arXiv:2503.16323) — symmetry augmentation for aerodynamic surrogates; Deep ensemble averaging (arXiv:1612.01474) — TTA as the low-cost analog to model ensembles.

---

### Hypothesis 10 — Cosine LR with Warm Restarts (SGDR)

**One-liner:** Replace `CosineAnnealingLR` with `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` to cycle through multiple high-LR phases within the 50-epoch budget, escaping shallow local minima in the transformer loss landscape.

**Expected delta:** -2% to -5% on `val_avg/mae_surf_p`. This is a low-risk, low-effort change that tends to help when models are undertrained relative to the available budget, which is likely given the 50-epoch cap and large mesh sizes.

**Mechanism:** With the 50-epoch cap and the default CosineAnnealingLR, the LR decays monotonically and the optimizer reaches a low-LR regime by epoch ~35-40. Warm restarts periodically reset the LR to the initial value, allowing the optimizer to jump out of sharp local minima and explore flatter, more generalizing minima (Hochreiter & Schmidhuber 1997 showed flat minima generalize better). With T_0=10 and T_mult=2, the restart schedule is: epoch 0-10 (first cycle), 10-30 (second), 30-70 (would be third, but clipped at 50). In practice this gives two full cosine cycles and part of a third, keeping the LR elevated longer than the monotonic cosine decay.

**Implementation (lines 434-435 of train.py):**

Replace:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

With:
```python
if cfg.lr_restarts:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.restart_t0, T_mult=2, eta_min=cfg.lr * 0.01
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

Add to Config: `lr_restarts: bool = False`, `restart_t0: int = 10`. Add CLI flags `--lr_restarts`, `--restart_t0`.

Note: the `scheduler.step()` call must move inside the training loop (called every batch) when using `CosineAnnealingWarmRestarts`, not once per epoch. Change line 508:
```python
if cfg.lr_restarts:
    scheduler.step(epoch + batch_idx / len(train_loader))  # per-step
# else: scheduler.step() at end of epoch as now
```

**Risk:** With T_0=10, the first restart happens at epoch 10 — if the model has not converged to a reasonable basin yet, the LR reset could destabilize early training. Use `--lr_warmup_epochs 2` alongside this to ensure initial convergence. The main risk is that restarts cause metrics to oscillate, making checkpoint selection noisy — the `best_avg_surf_p` tracking already handles this correctly.

**Citations:** SGDR (arXiv:1608.03983, Loshchilov & Hutter) — original warm restarts paper; Snapshot Ensembles (arXiv:1704.00109) — using SGDR for ensemble diversity; transformer training schedules survey (arXiv:2307.02972).

---

## Research Notes: What the Literature Says That Is Most Surprising or Under-Explored

### 1. Slice attention may be hurting, not helping

LinearNO (arXiv:2511.06294) performed a careful ablation showing that the softmax attention step in Transolver's PhysicsAttention is the *weakest* component. The slice/deslice projection matrices (in_project_slice, in_project_x, in_project_fx) do the heavy lifting by compressing N nodes into K slice tokens; what happens among those K tokens matters much less. Replacing softmax attention with linear attention OR even with a simple averaging MLP gave competitive or better results. This is counter-intuitive given how much attention mechanism design dominates the ML literature. **Implication:** if round 1 architectural experiments (deeper, wider, more slices) do not help, it may be because the attention bottleneck is in the slice mechanism, not in the model's overall capacity.

### 2. Surface pressure is a modal phenomenon

GLOBE (arXiv:2511.15856) achieves 600x lower surface pressure MAE than standard Transolver on AirFRANS Scarce split using a boundary-element Green's function formulation with only 117K parameters — smaller than the baseline by a factor of 10. The key insight: pressure is fully determined by the surface integral equation (Green's third identity), so a model that directly embeds this structure will learn in a far more constrained and physically correct space. This suggests that the current approach of treating all 3 output channels uniformly is suboptimal — surface pressure deserves a dedicated physical head with boundary constraints, not just a scalar surf_weight multiplier.

### 3. Re-driven scale variation is the primary OOD challenge

The y_std table in program.md shows `val_single_in_dist` max per-sample std of 2077 vs. mean of 458 — a 4.5x spread within a single split. For `val_geom_camber_rc`, the ratio is also large (1237 / 377 = 3.3x). Any loss that treats all samples equally (standard MSE, even with global y_std normalization) will devote most gradient to the 5-10% of samples at the extreme high-Re tail, starving the low-Re (low-amplitude) samples of training signal. Relative L2 (Hypothesis 1) and Huber (Hypothesis 3) directly target this, but a deeper fix might be per-sample adaptive normalization (z-score each sample's targets independently before the loss) — though this requires careful rescaling of the metric computation.

### 4. The NACA-camber OOD splits are a geometry interpolation problem, not pure extrapolation

Val splits `val_geom_camber_rc` (M=6-8 held out, M=2-5 and M=9 in training) and `val_geom_camber_cruise` (M=2-4 held out, M=0-2 and M=4-6 in training) are *interpolation* problems in NACA parameter space, not extrapolation. The model has seen nearby camber values; it just needs to interpolate smoothly. This means any conditioning mechanism that makes the NACA features more explicit (FiLM conditioning, domain embedding) should help substantially on these splits, while changes that improve pure predictive capacity (larger model, better optimizer) may help less than expected. The geometry interpolation framing also suggests that a model with an explicit parameterization of the NACA → pressure response curve (e.g., via a hypernetwork that maps NACA parameters to attention weights) could be highly effective.

### 5. Kaggle-style ensembling remains unexplored

In Kaggle competitions, the single most reliable gain is model ensembling — averaging predictions from 5-10 models trained with different seeds, architectures, or data splits. On TandemFoilSet, training 5 models each with a different random seed and averaging their predictions should yield a 2-5% gain on `val_avg/mae_surf_p` for free (no architecture change). The standard Kaggle technique is to use the same architecture but vary the seed, augmentation, and learning rate slightly. This is orthogonal to all other hypotheses and can be stacked on top of any winning single-model configuration. Consider a dedicated "ensemble" PR once the best single-model architecture is identified.
