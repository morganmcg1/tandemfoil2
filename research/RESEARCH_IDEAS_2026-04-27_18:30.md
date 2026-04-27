# SENPAI Research Ideas — Willow r4 Launch

- Date: 2026-04-27 18:30
- Track: `icml-appendix-willow-pai2c-r4`
- Author: researcher-agent (advisor pre-launch)
- Scope: 12 launch hypotheses, deliberately diverse strategy mix.
- Constraint reminders: 96 GB GPU; meshes up to 242K nodes at batch_size=4; SENPAI_MAX_EPOCHS / SENPAI_TIMEOUT_MINUTES bound the run; primary = `val_avg/mae_surf_p`, paper metric = `test_avg/mae_surf_p`. **No ideas pulled from sibling launches** — sourced only from `target/program.md`, `train.py`, the dataset description in `data/SPLITS.md`, and the literature.

A note on the primary metric. Surface pressure MAE is dominated by (a) the few high-Re samples whose y std is ~10× the median, (b) the sharp suction peaks on cambered foils where MAE can be hundreds of Pa·s/kg, and (c) the held-out cambers (M=6-8 raceCar, M=2-4 cruise) which the model has never seen. So good ideas should help one of: tail behavior in y, sharp local features near the surface, geometry interpolation. Hypotheses below are tagged with which axis they primarily target.

---

### H1: re-conditional-loss-reweighting
- **Strategy tier:** loss
- **Predicted delta on val_avg/mae_surf_p:** −5 to −12% vs baseline
- **Rationale:** Per-sample y std spans an order of magnitude inside every split (`val_single_in_dist` mean std 458, max 2,077). MSE in normalized space therefore implicitly under-weights tail samples whose absolute pressure swings dominate the eventual physical-MAE metric. Re-conditional reweighting (or, equivalently, normalizing the loss by per-sample y_std rather than the global y_std) realigns gradient magnitude with what the metric is actually measuring. This is essentially the "scale-invariant" trick from Eigen et al. (2014) for depth, applied per-sample. Closes the train-loss / eval-metric mismatch.
- **Specific change:** In `train.py` training loop, replace surf_loss with `((pred-y_norm)**2 / (per_sample_norm_std**2 + eps)) * surf_mask`, where `per_sample_norm_std = y[mask].std()` per sample over surface nodes. Equivalent: weight each sample's loss by `(log10(Re) - log10(Re_min))`. Try both; weight-by-y-std is the cleaner formulation.
- **Risk / failure mode:** Could over-weight extreme outliers and degrade low-Re predictions. Mitigation: clamp the per-sample weight to [0.25, 4.0] and validate on `val_re_rand` first.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h1-re-cond-loss --wandb_name h1-yvar`
- **GPU budget:** Identical to baseline (≤16 GB peak at 242K nodes). Fine.

---

### H2: huber-surface-loss
- **Strategy tier:** loss
- **Predicted delta on val_avg/mae_surf_p:** −2 to −6% vs baseline
- **Rationale:** Huber loss (Huber, 1964) is L2 near zero and L1 in the tail; it directly minimizes MAE-like penalties for the high-error nodes that dominate the surface pressure metric while preserving smoothness near zero where MSE is well-conditioned. Particularly attractive because the eval metric is L1 — using L1/Huber for training reduces the train/eval gap (Wang et al., 2024 "Match the loss to the metric").
- **Specific change:** Replace `(pred - y_norm)**2` with `F.huber_loss(pred, y_norm, reduction='none', delta=1.0)` for the surface term only; keep MSE for volume to maintain smooth field reconstruction. Sweep delta ∈ {0.5, 1.0, 2.0}.
- **Risk / failure mode:** Huber can slow convergence in normalized space; if 50 epochs is not enough, the run looks worse on the time axis. Mitigation: warm up first 5 epochs with MSE, then switch.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h2-huber --wandb_name h2-delta1.0`; sweep delta via group.
- **GPU budget:** Same as baseline.

---

### H3: surf-weight-grid
- **Strategy tier:** loss
- **Predicted delta on val_avg/mae_surf_p:** −1 to −5% vs baseline
- **Rationale:** Default `surf_weight=10` is unjustified. The surface comprises a tiny fraction of nodes (typically <2%), so the surface-loss-per-node was only "balanced" with the volume term up to a constant. Sweeping is the cheapest way to find headroom — particularly because the primary metric is surface, not volume.
- **Specific change:** Sweep `surf_weight ∈ {3, 5, 10, 20, 50}` keeping all other defaults. 4 GPU-runs of one student covers most of the curve.
- **Risk / failure mode:** Very high surf_weight harms volume MAE — but program.md only ranks surface pressure MAE so this is permitted. Track volume MAE as a guardrail: stop tuning if vol_p MAE exceeds 2× baseline.
- **Reproduce flag:** `python train.py --surf_weight 20 --wandb_group h3-surfw-sweep --wandb_name h3-w20`
- **GPU budget:** Same as baseline.

---

### H4: per-channel-loss-balancing
- **Strategy tier:** loss
- **Predicted delta on val_avg/mae_surf_p:** −3 to −7% vs baseline
- **Rationale:** Ux, Uy, p have grossly different scales. After normalization they are unit-variance globally, but the *gradient interaction* between them in a shared backbone still varies — pressure shows the steepest gradients at the surface (suction peak). Multi-task loss balancing (Kendall et al., 2017 — uncertainty weighting; Liu et al., 2019 GradNorm) lets the optimizer auto-weight the three channel losses. Cheapest version: learnable per-channel log-σ.
- **Specific change:** Add `self.log_sigma = nn.Parameter(torch.zeros(3))` to model. Loss becomes `Σ_c (sq_err[..., c] / (2*exp(2*log_sigma[c])) + log_sigma[c])`, applied separately for surface and volume terms. Keep surf_weight on top.
- **Risk / failure mode:** Bad initialization can starve one channel. Mitigation: initialize log_sigma=0 and clip to [-3, +3].
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h4-uncertainty-weight --wandb_name h4-multitask`
- **GPU budget:** Negligible.

---

### H5: re-mach-feature-augmentation
- **Strategy tier:** features (physical priors)
- **Predicted delta on val_avg/mae_surf_p:** −4 to −9% vs baseline
- **Rationale:** Currently `log(Re)` is the only flow-condition feature. Pressure is bounded above by stagnation-pressure-like dimensionless quantities, and the baseline gets no pressure-coefficient prior. Adding `1/Re`, `Re^{-1/2}` (Blasius / boundary-layer scaling), and `sin(AoA)`, `cos(AoA)` (continuous AoA representation that respects rotational equivariance) gives the model nondimensional features that explicitly hint at the asymptotic scaling laws of the pressure field. This is the simplest "physics-informed feature" idea and historically strong (PIESRGAN, Wang et al., 2020).
- **Specific change:** In `train.py`, before normalization, augment `x` with extra dims `[1/sqrt(Re), sin(AoA1), cos(AoA1), sin(AoA2), cos(AoA2)]`. Update `fun_dim` accordingly and recompute (online) means/stds for these new dims, OR pre-normalize them analytically (sin/cos are already in [-1,1]). Preserve original 24 dims so the model still sees the raw values.
- **Risk / failure mode:** New stats need to match what the model expects. Mitigation: compute in-script and concatenate after the existing normalization.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h5-physfeats --wandb_name h5-remach`
- **GPU budget:** Adds ~5 dims to a 24-dim input — negligible.

---

### H6: fourier-position-features
- **Strategy tier:** features
- **Predicted delta on val_avg/mae_surf_p:** −3 to −8% vs baseline
- **Rationale:** Transolver's PhysicsAttention slices the mesh adaptively but the input still contains raw `(x, z)` positions. Sinusoidal positional encodings (Tancik et al., 2020 — Fourier features; Mildenhall et al., 2020 — NeRF) drastically improve high-frequency spatial reconstruction in MLP/coordinate networks. Surface pressure is exactly the high-frequency component the model struggles with (suction peaks). Adds capacity to represent fine-scale spatial variation without adding parameters in the attention.
- **Specific change:** Construct Fourier features `[sin(2^k π x), cos(2^k π x), sin(2^k π z), cos(2^k π z)]` for k ∈ {0..5} (12 freqs × 2 axes × 2 trig = 48 dims) and concatenate to x. Keep position dims; the existing space_dim=2 layer absorbs raw coords. `fun_dim = X_DIM - 2 + 48`.
- **Risk / failure mode:** Higher-frequency features (k ≥ 6) can overfit to mesh discretization. Mitigation: cap at k=5; verify val_geom_camber MAE doesn't degrade vs baseline.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h6-fourier --wandb_name h6-k5`
- **GPU budget:** +48 input dims through a single Linear → 48 × 128 ≈ 6K params. Trivial.

---

### H7: deeper-narrower-transolver
- **Strategy tier:** architecture
- **Predicted delta on val_avg/mae_surf_p:** −4 to −10% vs baseline
- **Rationale:** Baseline is 5×128, ~1-2M params. Surface pressure prediction needs more depth to compose distant geometry features (front foil affects rear foil's surface). Recipe from Tay et al., 2022 (deep narrow Transformers): doubling depth and slightly trimming hidden dim usually beats doubling hidden dim at constant param count, especially on small data. 96 GB headroom is huge — a ~10M-param Transolver still fits well within budget.
- **Specific change:** `n_layers=10, n_hidden=128, n_head=4, slice_num=64, mlp_ratio=2`. Optionally try `n_layers=8, n_hidden=192`. Sweep in a wandb group.
- **Risk / failure mode:** Deeper = slower per epoch; 50 epochs may not be enough. Mitigation: track time-per-epoch; if >1.6× baseline, fall back to 8 layers.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h7-depth --wandb_name h7-l10-h128`
- **GPU budget:** ~3M params, peak VRAM ~30 GB at 242K. Fine.

---

### H8: slice-num-and-heads-sweep
- **Strategy tier:** architecture
- **Predicted delta on val_avg/mae_surf_p:** −2 to −6% vs baseline
- **Rationale:** PhysicsAttention slices nodes into `slice_num=64` learned tokens (Wu et al., 2024 Transolver). 64 tokens may be too few for 242K-node cruise meshes — average cluster size of 3.7K nodes per token is a coarse grouping. Higher slice_num and more heads improve effective receptive resolution. Cheap structural change.
- **Specific change:** Sweep `slice_num ∈ {64, 128, 256}` × `n_head ∈ {4, 8}` (4-cell grid). Keep all else fixed.
- **Risk / failure mode:** Memory in PhysicsAttention is O(B × H × G × N) for slice_weights — at 256 slices × 8 heads × 242K nodes × batch 4 × fp32 ≈ 2 GB just for that tensor (manageable). Could OOM at the high end; fall back to slice_num=128 if so.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h8-slice-heads --wandb_name h8-s128-h8`
- **GPU budget:** Up to ~50 GB at 256 slices × 8 heads. Will fit, but worth profiling.

---

### H9: warmup-onecycle-lr
- **Strategy tier:** optimizer
- **Predicted delta on val_avg/mae_surf_p:** −3 to −7% vs baseline
- **Rationale:** Baseline starts at lr=5e-4 with cosine annealing. Cosine starts already at peak — no warmup. Smith & Topin (2017) show OneCycle (linear warmup + cosine annealing + a higher peak) typically beats cosine-from-zero at the same epoch budget. With ~1000 train steps per epoch and 50 epochs, a 5-epoch linear warmup to peak lr=1e-3 then cosine to 1e-5 is a strong default for transformer-style models on small data (DeiT, Touvron et al., 2021).
- **Specific change:** Replace `CosineAnnealingLR` with `OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs*len(train_loader), pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100)`. Keep AdamW.
- **Risk / failure mode:** Higher peak lr can destabilize early training. Mitigation: gradient clip to 1.0 (`torch.nn.utils.clip_grad_norm_`).
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h9-onecycle --wandb_name h9-peak1e3`
- **GPU budget:** None.

---

### H10: ema-weights
- **Strategy tier:** optimizer
- **Predicted delta on val_avg/mae_surf_p:** −2 to −5% vs baseline
- **Rationale:** EMA of model weights consistently improves test-time accuracy on small training sets in Transformer/transformer-derived architectures (Polyak averaging; in modern form, ModelEMA from timm; widely used in ConvNeXt, Stable Diffusion). For our 1499-train, 100-val regime where validation noise is non-trivial, EMA acts as an implicit ensemble across late training and reduces selection variance. Negligible compute cost.
- **Specific change:** Add `from timm.utils import ModelEmaV3` (timm is already in pyproject.toml). Maintain `ema_model = ModelEmaV3(model, decay=0.999)`; call `ema_model.update(model)` after each `optimizer.step()`. At validation, evaluate with `ema_model.module` instead of `model`. Save EMA weights as the checkpoint.
- **Risk / failure mode:** Decay too aggressive at small step counts. Mitigation: warmup the EMA decay schedule (timm has built-in warmup); start at 0.99 and ramp to 0.9995.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h10-ema --wandb_name h10-ema-d999`
- **GPU budget:** Doubles model params in memory only (still <100 MB). Negligible.

---

### H11: geometry-reflection-augmentation
- **Strategy tier:** data
- **Predicted delta on val_avg/mae_surf_p:** −3 to −9% vs baseline (esp. on `val_geom_camber_*`)
- **Rationale:** The flow around an airfoil is reflection-equivariant in z under sign-flip of `(z, Uy, AoA)`. Cruise samples cover AoA ∈ [-5°, +6°] — small offset but a real reflection group. Symmetry augmentation increases effective camber coverage which directly attacks the `val_geom_camber_*` holdouts. Standard CFD-surrogate trick (Fan et al., 2024 ML4PhysicalSciences).
- **Specific change:** In the train loop, with p=0.5 reflect each sample: `x[:, 1] *= -1` (z), `x[:, 14] *= -1` (AoA1), `x[:, 18] *= -1` (AoA2), and `y[:, 1] *= -1` (Uy). Also flip the z component of saf if it has a sign — verify by reading `data/prepare_splits.py` to check whether `saf` (signed arc-length, dim 2-3) is z-signed; if so flip dim 3. For raceCar this is invalid (ground effect breaks symmetry) — gate the augmentation on `is_cruise = (gap != 0 and stagger != 0 and AoA1 > -10°)`. Simpler safe gate: condition on the file-id but we don't have it; instead gate on `dim 1 (z-mean of mesh) > -threshold` to detect cruise. Cleanest: skip when the sample has any node with z below ground-effect floor. **Worth one careful student PR — mention this in instructions.**
- **Risk / failure mode:** Wrongly applying to raceCar destroys the result. Mitigation: detect cruise via `dim 22 != 0 AND ground-effect signature`. Validate on `val_single_in_dist` first.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h11-flip-aug --wandb_name h11-cruise-only`
- **GPU budget:** No additional VRAM.

---

### H12: surface-aware-attention-bias
- **Strategy tier:** domain (architecture-adjacent)
- **Predicted delta on val_avg/mae_surf_p:** −5 to −10% vs baseline
- **Rationale:** Surface nodes are <2% of mesh nodes, but they own 100% of the primary metric. The current PhysicsAttention slice mechanism distributes nodes into 64 learned slices via softmax — there is no architectural pressure to keep surface nodes on dedicated slices. A surface-aware bias term in `in_project_slice` reserves a portion of the slice budget for surface nodes, ensuring fine-grained surface representation regardless of mesh size. Inspired by mixture-of-experts router gating (Shazeer et al., 2017) and surface-pooling tricks in MeshGraphNet (Pfaff et al., 2020).
- **Specific change:** In `PhysicsAttention.forward`, before the softmax, add a learned bias `bias = self.surf_bias(x)` (a simple Linear(1, slice_num)) computed from the `is_surface` channel (dim 12 of the input). Add this bias to `slice_weights` pre-softmax. Surface nodes will be pushed onto a small set of slices the model learns to dedicate to surface flow.
- **Risk / failure mode:** Adds an `is_surface` plumbing path through the model — currently the model only sees `x` not the explicit surface flag separately. Workaround: extract dim 12 of the `x` tensor as the surface signal inside the model.
- **Reproduce flag:** `python train.py --epochs 50 --wandb_group h12-surf-attn --wandb_name h12-surf-bias`
- **GPU budget:** A linear Layer(1, 64) per attention layer × 5 layers = 5 × 65 params. Trivial.

---

## Confidence ordering (best first)

This ordering reflects expected EV under the constraint that we are launching cold without prior r4 evidence. Hypotheses near the top combine high expected delta with low complexity/risk and orthogonal-to-baseline-tuning mechanisms.

1. H1 (re-conditional-loss-reweighting) — largest expected delta, addresses train/metric mismatch directly.
2. H7 (deeper-narrower-transolver) — straightforward capacity scale with strong literature support and headroom in 96 GB.
3. H10 (ema-weights) — near-free, consistently positive across small-data Transformer regimes.
4. H5 (re-mach-feature-augmentation) — physics priors targeting tail behavior.
5. H9 (warmup-onecycle-lr) — well-established schedule improvement, lifts most subsequent ideas.
6. H12 (surface-aware-attention-bias) — high-leverage, directly attacks surface metric.
7. H11 (geometry-reflection-augmentation) — best lever for `val_geom_camber_*` if implemented carefully.
8. H4 (per-channel-loss-balancing) — cheap, principled, often a small win.
9. H6 (fourier-position-features) — likely modest but free; risk of mesh overfit needs monitoring.
10. H2 (huber-surface-loss) — solid, modest.
11. H3 (surf-weight-grid) — very cheap; weak prior the optimum is far from 10.
12. H8 (slice-num-and-heads-sweep) — likely smaller delta and higher VRAM cost than H7 for similar capacity gain.

## Cross-cutting notes for the advisor

- **Combine winners aggressively in r4+.** H1+H7+H10 are orthogonal; if any two land, the next round should test their stack.
- **Time budget:** at 50 epochs and current throughput, every student should fit one full run with margin. H7-l10 and H8-s256-h8 are the only ideas that will materially eat into timeout — assign those to students whose pods have the freshest GPUs.
- **Test-from-best-checkpoint** is already wired in `train.py`. Good.
- **Watch out for full-test eval** — `evaluate_split` runs on test at the end. H7's deeper model adds ~2 minutes there. Acceptable.
