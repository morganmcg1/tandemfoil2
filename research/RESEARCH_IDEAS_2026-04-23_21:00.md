# Research Ideas — TandemFoilSet Round 1+

**Compiled:** 2026-04-23 21:00 by researcher-agent
**Primary metric:** `val_avg/mae_surf_p` (lower is better)
**Baseline:** Transolver, n_hidden=128, n_layers=5, slice_num=64, bs=4, lr=5e-4, wd=1e-4, surf_weight=10, 50 epochs, 30 min wall-clock, 96 GB VRAM/GPU, 8 GPUs per student.

This document brainstorms fresh hypotheses organized by impact vs. complexity. Each idea is tied to a dataset property from `DATASET_ANALYSIS.md` (heavy-tailed pressure, 10x per-sample y_std variation, ~1% surface fraction, OOD camber/Re splits, 3 physical domains). The first-round obvious tunings (Huber/L1, capacity scaling, EMA, grad clip, Fourier features, asinh-p, FiLM on log(Re), per-domain normalization, surface-specific decoder head) are explicitly out-of-scope here.

---

## Section 1 — High-Impact / Low-Complexity (recipe & loss tweaks)

These can be delivered in <50 LOC diffs touching only `train.py`. Good for early rounds because they give fast iteration and high compound value.

### 1.1 Relative-error (log-cosh / scaled-MAE) loss for pressure
- **Hypothesis:** Replace the p-channel loss with `mean(|err| / (|p_true| + eps))` or `log-cosh(err / scale(Re))` so per-sample scale of 10x doesn't tilt the gradient toward high-Re samples.
- **Why:** Per-sample y_std varies 10x. Under standard MSE in normalized space, high-Re samples still dominate gradient magnitude after normalization because `y_std` is global. Relative loss de-biases this.
- **Complexity:** trivial (~5 LOC)
- **Expected impact:** medium; most likely improves `val_re_rand` and `val_geom_camber_*`.

### 1.2 Sample-wise z-score normalization (normalize by per-sample y_std instead of global)
- **Hypothesis:** Compute per-batch, per-sample `y_std`, normalize targets with that instead of the global stats; training loss then targets a dimensionless residual.
- **Why:** 10x per-sample y_std variance is the root cause of gradient imbalance. Per-sample normalization makes the learning signal scale-invariant. At inference, the model's output is rescaled by a predicted or feature-derived per-sample scale (use log(Re) or a simple MLP on sample-level features to predict `log(y_std)`).
- **Complexity:** moderate (~30 LOC: scale prediction head + rescaling at eval).
- **Expected impact:** large on `val_re_rand` and high-Re test samples.

### 1.3 Loss weighting by expected magnitude (inverse sqrt importance)
- **Hypothesis:** Weight each sample's loss by `1 / sqrt(|p_true|.abs().mean() + eps)` so small-magnitude samples contribute proportionally to their fraction of the metric.
- **Why:** The ranking metric is a global node-level mean over all surface nodes in a split, not a per-sample mean. Low-Re (small-p) samples see little gradient under MSE but still contribute to the MAE average at eval. Inverse-magnitude weighting recovers this balance.
- **Complexity:** trivial (~5 LOC)
- **Expected impact:** small-medium; most likely improves `val_geom_camber_cruise` (lower magnitudes) and low-Re samples.

### 1.4 Huber loss with per-channel delta from empirical stats
- **Hypothesis:** Use Huber (smooth-L1) but pick `delta` per channel from the 75th percentile of training errors, updated each epoch.
- **Why:** The obvious Huber idea was listed — but picking a smart adaptive delta per channel (p is heavy-tailed, Ux/Uy are not) is the extension that usually wins. Heavy-tailed p wants a smaller delta; Gaussian-ish Ux/Uy want a larger one.
- **Complexity:** trivial (~10 LOC)
- **Expected impact:** medium on surf p specifically.

### 1.5 Surface-only batch balancing (oversample by surface fraction)
- **Hypothesis:** Samples with more surface nodes (e.g., tandem samples with 2 foils) have more surface-loss signal. Balance batches on total surface node count, not sample count.
- **Why:** Single-foil has ~1x surface fraction, tandem has ~2x. Under bs=4 and balanced domain sampler, surface gradient per batch fluctuates. Balance surface node count across batches → stabler training.
- **Complexity:** moderate (custom sampler, ~30 LOC)
- **Expected impact:** small-medium on training stability / final metric.

### 1.6 Warmup-stable-decay (WSD) schedule instead of cosine
- **Hypothesis:** Replace cosine annealing with linear warmup (5%) → flat peak LR (70%) → linear decay to 0 (25%).
- **Why:** Cosine spends a lot of the schedule in fine-tune territory. On a small dataset like this (~1500 train samples, 50 epochs), WSD gives more time at peak LR to explore, then a sharper decay to converge. Proven on LLMs and vision alike.
- **Complexity:** trivial (~5 LOC, manual scheduler)
- **Expected impact:** small-medium, broadly applicable.

### 1.7 Gradient accumulation to effective batch 16 or 32
- **Hypothesis:** Accumulate gradients across 4–8 micro-batches to smooth gradient noise from the tiny bs=4.
- **Why:** Variable-size meshes (74K–242K) mean per-batch loss noise is high. Larger effective batch smooths this; VRAM headroom is there (96GB, current peak probably far below).
- **Complexity:** trivial (~10 LOC)
- **Expected impact:** small-medium; most likely improves final-epoch stability.

### 1.8 Test-time ensemble via horizontal flip (with Uy sign flip)
- **Hypothesis:** Predict for both the original geometry and its x-axis flipped counterpart (flip x, z, saf, Uy); average predictions in physical space.
- **Why:** The physics is not x-symmetric for inverted raceCar (ground effect breaks symmetry) but is for cruise. Zero-cost TTA that usually gives 1–3% MAE reduction.
- **Complexity:** trivial (~15 LOC at eval time only)
- **Expected impact:** small, broad.

### 1.9 Multi-crop inference — evaluate on subsampled meshes and average
- **Hypothesis:** At eval, run inference with 3 different random 80% node subsets; average per-node predictions where they overlap.
- **Why:** Transolver's slice-based attention is mildly permutation-sensitive; ensembling over subsampled inputs regularizes.
- **Complexity:** trivial (~20 LOC)
- **Expected impact:** small.

### 1.10 Output-clamping / target-clipping hack for extreme outliers
- **Hypothesis:** Clip p targets to 99.5th percentile during training (and rescale accordingly); evaluate in physical space unchanged.
- **Why:** 0.5% of training nodes dominate the MSE loss but contribute <<1% to the MAE metric. Clipping avoids gradient distortion.
- **Complexity:** trivial (~5 LOC)
- **Expected impact:** small-medium; most important for `val_re_rand`.

---

## Section 2 — Mid-Complexity Architectural

Changes that extend or replace a Transolver block, add a head, or swap an attention variant. Typical diffs 50–300 LOC.

### 2.1 Learnable slice centroids initialized by K-means over dsdf features
- **Hypothesis:** Instead of letting `in_project_slice` learn slice assignments from scratch, initialize the first-layer slice weights from K-means clusters over the training set's dsdf (dims 4–11) features.
- **Why:** Slice attention groups nodes into 64 "physical regions". Data-driven initialization from the multi-scale shape descriptor should give the model a head start on meaningful regions (near-surface, wake, freestream zones) before gradients kick in.
- **Complexity:** moderate (~80 LOC: one-time K-means computation + custom init)
- **Expected impact:** medium; likely helps OOD geometry splits by making the partition structure more interpretable.

### 2.2 Cross-attention decoder with separate query tokens for surface vs. volume
- **Hypothesis:** After the Transolver backbone, run a cross-attention where surface-node tokens query the slice tokens through a different head than volume-node tokens (via a learned gate or two separate query projections).
- **Why:** Surface physics (boundary-layer, pressure) is different from volume physics (wake, freestream). Letting the final decoder specialize without duplicating the whole trunk recovers dual-decoder benefits at ~10% the parameter cost.
- **Complexity:** moderate (~120 LOC: new decoder module, gated query)
- **Expected impact:** medium-large; directly targets the primary metric.

### 2.3 Geometric positional encoding via Laplacian eigenmaps of mesh
- **Hypothesis:** Precompute the first K=16 Laplacian eigenvectors of the KNN graph (k=8) over mesh nodes per sample; concatenate as extra input features.
- **Why:** Transolver has no mesh-topology signal; positions are euclidean only. Laplacian eigenmaps encode global mesh connectivity in a low-dim basis, which is known to help graph-based surrogates. This complements dsdf (which is local).
- **Complexity:** moderate (~150 LOC: precompute on the fly with `scipy.sparse.linalg.eigsh`, cache to disk if too slow; 16 extra input dims).
- **Expected impact:** medium; broadly helpful, most on OOD geometry.

### 2.4 Perceiver IO-style architecture: fixed-size latent bottleneck
- **Hypothesis:** Replace the N-node slice-attention Transolver with a Perceiver IO — encode all N nodes into M=256 latents via cross-attention, run L=8 self-attention layers in latent space, decode back via cross-attention.
- **Why:** Transolver slice-attention is closely related but constrained to a partition. Perceiver's learned latent queries are more flexible and benefit from more depth at the same compute. Also decouples compute from N.
- **Complexity:** substantial (~250 LOC, but modular)
- **Expected impact:** large if it works; risky because the inductive bias changes. Best assigned to a student after simpler wins.

### 2.5 GNO / FNO-style global message-passing on mesh
- **Hypothesis:** Add a single graph neural operator layer (message passing over KNN=8 graph with learnable message MLP) as a residual branch alongside each Transolver block.
- **Why:** Transolver's slice attention is global but through a partition bottleneck; GNO message passing is local but without bottleneck. Fusing both gives multi-scale spatial context.
- **Complexity:** substantial (~200 LOC: KNN, message passing, memory-aware impl)
- **Expected impact:** medium-large; OOD geometry likely benefits.

### 2.6 Swap PhysicsAttention for a variant with Gated Linear Unit (GLU) feedforward
- **Hypothesis:** Replace the MLP in each `TransolverBlock` with a SwiGLU (`x -> (xW1) * silu(xW2) W3`) keeping total params roughly constant.
- **Why:** SwiGLU outperforms standard MLP in LLMs and vision transformers. Low-risk, well-known win. Likely to compound with other changes.
- **Complexity:** trivial-moderate (~30 LOC)
- **Expected impact:** small-medium, broadly applicable.

### 2.7 Multi-scale slice resolution (hierarchical slicing)
- **Hypothesis:** First layer uses slice_num=256, middle layers 64, last layer 256. This gives coarse-to-fine-to-coarse structure.
- **Why:** The first block needs to see fine local detail to build node features; middle blocks want global context; final block needs fine again for node-level decoding. Hierarchical scale = U-Net-like.
- **Complexity:** moderate (~60 LOC)
- **Expected impact:** medium; most on mesh-size-heterogeneous domains (cruise has 210K nodes).

### 2.8 Replace last-layer MLP with a per-channel small MLP + physics residual
- **Hypothesis:** The final `mlp2` head mixes Ux, Uy, p together via shared weights until the last linear. Use three separate MLPs instead, one per channel, with an extra input branch that takes the model's own prediction of the upstream field (e.g., p head receives predicted Ux, Uy).
- **Why:** Pressure is strongly correlated with velocity (Bernoulli: p ~ -0.5 * |U|^2 in potential flow); exposing the prediction of U to the p head gives it a physics-aware residual target.
- **Complexity:** moderate (~50 LOC)
- **Expected impact:** medium on mae_surf_p specifically.

### 2.9 Attention temperature warmup
- **Hypothesis:** The `self.temperature = 0.5` in `PhysicsAttention` is hardcoded. Add a schedule that starts higher (uniform slice weights) and anneals down over the first 20% of training.
- **Why:** At init, slice projection is orthogonal but attention is sharply peaked. Annealing the temperature lets the model first explore soft assignments then sharpen them — similar to soft-to-hard clustering curricula.
- **Complexity:** trivial (~10 LOC)
- **Expected impact:** small-medium.

### 2.10 Replace `in_project_slice` with a coordinate-based assignment (MLP on (x, z, is_surface))
- **Hypothesis:** Current slice assignment learns from all dim_head features. Instead, let the slice assignment depend only on physical position features so the partition is a learned spatial clustering that's consistent across samples.
- **Why:** Makes slices correspond to spatial regions (pressure field, suction side, wake zone) rather than arbitrary feature clusters. This gives a cleaner inductive bias and may generalize better to OOD geometries.
- **Complexity:** moderate (~40 LOC)
- **Expected impact:** medium, especially on OOD camber splits.

### 2.11 Conditional layer-norm driven by log(Re) and domain indicators
- **Hypothesis:** Replace `LayerNorm` in each block with a conditional variant where the scale/shift parameters are functions of (log(Re), is_tandem, is_cruise).
- **Why:** Strong conditioning signal throughout the backbone, not just at input. Similar to AdaGN in diffusion models, extended to Re-dependent normalization.
- **Complexity:** moderate (~60 LOC)
- **Expected impact:** medium on `val_re_rand` and domain transfer.

### 2.12 Add a near-surface volume-penalty — explicit supervision on within-ε band around foils
- **Hypothesis:** Identify volume nodes within distance ε=0.05 chord-length of any foil surface (approximated from dsdf[0]); apply `surf_weight * 0.5` to those nodes. Three-tier weighting instead of binary surf/vol.
- **Why:** Surface pressure accuracy requires accurate boundary-layer region, but the boundary layer is in the volume. Boosting the near-surface volume gives the surface decoder cleaner context.
- **Complexity:** moderate (~30 LOC)
- **Expected impact:** medium-large on surf_p.

---

## Section 3 — High-Impact / High-Complexity

Physics-informed regularization, data augmentation pipelines, new model families. Typical diffs 200–600 LOC.

### 3.1 Panel-method residual learning (model predicts the correction to an analytical prior)
- **Hypothesis:** For each sample, compute the inviscid panel-method (thin-airfoil) pressure distribution as an analytical prior; train the model to predict the viscous/interaction correction only. At eval, sum prior + learned correction.
- **Why:** Classical theory gives surprisingly good first-order p predictions for attached flow. The model then only needs to learn the viscous / downwash / ground-effect / tandem-interaction deviations, which is a much lower-variance target. Direct targeted approach to the primary metric.
- **Complexity:** substantial (~400 LOC: panel-method implementation, vectorized; precompute + cache)
- **Expected impact:** large, especially on single-foil and mild-AoA cruise samples. May help OOD because the prior carries the geometry info.

### 3.2 Bernoulli / incompressibility regularizer
- **Hypothesis:** Add an auxiliary loss term: `|div(U)|_1` at interior nodes (approximated via local finite-difference on the KNN graph) + `|p + 0.5*|U|^2 - p_total|_1` (Bernoulli head with learned `p_total` per sample for far-field regions).
- **Why:** CFD solutions are incompressible and follow Bernoulli outside the viscous wake. Hard-coding these constraints as soft regularizers gives physical coherence and should reduce p error indirectly.
- **Complexity:** substantial (~300 LOC: KNN finite diff, aux heads)
- **Expected impact:** medium-large; broad effect on all splits.

### 3.3 Synthetic panel-method single-foil pre-training
- **Hypothesis:** Generate 5K synthetic single-foil samples using a high-fidelity panel method (with boundary-layer correction). Pre-train for 20 epochs on synthetic + 30 epochs on real.
- **Why:** Training set is small (1499 samples). Panel-method pretraining gives the model a strong geometric/physics prior for free. Similar in spirit to simulation-to-real transfer in robotics.
- **Complexity:** substantial (~500 LOC: panel gen script + pretrain loop + fine-tune loop)
- **Expected impact:** large on OOD camber splits. Moderate on `val_re_rand`.

### 3.4 Horizontal-flip augmentation with sign-flip on Uy
- **Hypothesis:** At each training step, with probability 0.5, flip x → -x, z unchanged, saf negated, foil AoA negated, Uy predictions negated. Doubles effective dataset size.
- **Why:** Cruise foil physics is x-symmetric modulo sign flip on Uy. For raceCar (ground effect), the ground is z=0, so x-flip also preserves physics. This is a known trick from CFD-ML literature.
- **Complexity:** moderate (~80 LOC: careful handling of input features 15-17 (NACA), 18-21 (foil 2), 22-23 (gap/stagger — stagger sign flips))
- **Expected impact:** medium-large (2x effective data).

### 3.5 Geometry curriculum: train low-camber → high-camber progressively
- **Hypothesis:** For the first 10 epochs, weight training samples with |M-5| > 2 lightly (inverse of distance from mid-range); gradually uniform-weight.
- **Why:** Camber extremes (M=0 or M=9) are hardest to learn first. Starting in the geometric middle and expanding outward builds stable representations.
- **Complexity:** moderate (~50 LOC)
- **Expected impact:** small-medium on OOD camber splits (where camber is in the middle of the held-out range).

### 3.6 Re curriculum: low Re → high Re progressively
- **Hypothesis:** First 15 epochs: sample inversely proportional to Re; gradually switch to uniform Re.
- **Why:** Low-Re flows are smoother, higher-signal-to-noise. Building the representation on smooth physics first, then extending to turbulent regimes, mirrors human physics education. Low-Re provides cleaner gradient signal.
- **Complexity:** moderate (~40 LOC)
- **Expected impact:** medium on `val_re_rand`.

### 3.7 Spectral-loss on pressure (match power spectrum along surface)
- **Hypothesis:** Additional loss: `|FFT(pred_p_surf)[low-freq] - FFT(true_p_surf)[low-freq]|`, applied along the arc-length parameter for each foil.
- **Why:** Surface pressure along a foil has a known low-frequency structure (smooth except at stagnation point, trailing edge). Explicitly penalizing spectral mismatch at low frequencies forces the model to get the overall envelope right before fitting details.
- **Complexity:** substantial (~200 LOC: identify each foil's surface nodes, sort by saf, FFT)
- **Expected impact:** medium on surf_p.

### 3.8 Diffusion-model surrogate (score-matching on residual)
- **Hypothesis:** Train a conditional diffusion model that predicts the denoising direction for (Ux, Uy, p) given the 24-dim features. Sample via 4–8 denoising steps.
- **Why:** Diffusion models have set new SOTA in PDE surrogate literature recently (e.g., DiffusionPDE). Their probabilistic nature handles the heavy tails in pressure naturally.
- **Complexity:** substantial (~600 LOC: noise scheduler, score-head, sampler)
- **Expected impact:** large if it works; training stability risk. Best as a longer-term track.

### 3.9 Neural Operator (FNO) on structured mesh grid as parallel backbone
- **Hypothesis:** Interpolate the irregular mesh onto a 256×512 Cartesian grid; run 8-layer FNO over grid; interpolate back to mesh. Concatenate with Transolver output.
- **Why:** FNO captures global PDE operators in spectral space very efficiently. For 2D CFD, this is natural. Combining with mesh-native Transolver gives both regular and irregular receptive fields.
- **Complexity:** substantial (~400 LOC: mesh-to-grid interp, FNO impl, fusion layer)
- **Expected impact:** medium-large.

### 3.10 Per-node mesh coarsening (Voronoi pooling + unpooling)
- **Hypothesis:** Coarsen mesh to 1/4 size via farthest-point sampling + Voronoi assignment; run deep model on coarse; unpool predictions via inverse-distance weighting.
- **Why:** Reduces compute and memory 4x, enables deeper models. A la PointNet++ but with mesh topology.
- **Complexity:** substantial (~350 LOC)
- **Expected impact:** medium (unlocks capacity scaling without OOM).

### 3.11 Kutta condition soft enforcement at trailing edge
- **Hypothesis:** Identify trailing-edge node on each foil (largest x-coordinate among surface nodes on that foil); add loss `|p_upper - p_lower|` at the TE vertex (approximated via nearest neighbors).
- **Why:** The Kutta condition is a fundamental constraint in aerodynamics — pressure must be continuous at the sharp trailing edge. Hard-coding this physically constrains the model.
- **Complexity:** substantial (~150 LOC: foil-node decomposition, TE identification, aux loss)
- **Expected impact:** small-medium on surf_p.

### 3.12 Hard-example mining with online re-weighting
- **Hypothesis:** After epoch 10, track per-sample val surf_p MAE; oversample the worst 20% by 2x in subsequent epochs.
- **Why:** The model's generalization failure is concentrated on a few hard samples (likely very high-Re or extreme geometry). Focus compute there.
- **Complexity:** moderate (~80 LOC)
- **Expected impact:** medium on tail performance, which is what drives the aggregate metric.

---

## Section 4 — Speculative / Long-Shot

Cross-pollination ideas that may not pay off but are under-explored in CFD surrogates.

### 4.1 HyperNetwork for Re-dependent weights (Schmidhuber-inspired)
- **Hypothesis:** A small MLP takes log(Re) and produces (or perturbs) the Transolver block weights per sample. Re-conditioned fast weights.
- **Why:** log(Re) fundamentally changes the physics regime (Stokes → inertial → turbulent). A single fixed-weight model has to average across regimes. HyperNetworks let weights specialize continuously in Re.
- **Complexity:** substantial (~300 LOC)
- **Expected impact:** medium-large on `val_re_rand` if tuned well; high risk.

### 4.2 State-Space Model (Mamba) over sorted-by-saf surface sequence
- **Hypothesis:** Treat the surface of each foil as a 1D sequence (sort by signed arc-length), apply a Mamba block to refine surface predictions after the main Transolver backbone.
- **Why:** Surface pressure has a natural 1D topology along the foil. S4/Mamba are excellent at long 1D sequences with sharp features. Also: selective-scan is cheap and interpretable.
- **Complexity:** substantial (~250 LOC: Mamba impl + surface-sequence extraction)
- **Expected impact:** medium on surf_p.

### 4.3 Mixture-of-Experts across (domain, Re-bin) cells
- **Hypothesis:** 6 experts: {raceCar single, raceCar tandem, cruise tandem} × {low-Re, high-Re}. Router is deterministic (not learned) based on features 18-21 and dim 13.
- **Why:** The three domains have very different physics (ground effect, tandem interaction, freestream). Hard-coded routing avoids router training instability and captures known domain structure.
- **Complexity:** substantial (~250 LOC)
- **Expected impact:** medium; risk of overfitting single-expert slices.

### 4.4 Echo-State / Reservoir for wake modeling
- **Hypothesis:** A frozen random-weight recurrent "reservoir" operating on wake-region nodes (downstream of foils); trainable readout combines reservoir with main model.
- **Why:** Echo-state networks are known to model chaotic dynamics well with very little data. Wake flows are chaotic. Zero-cost exploration given the infrastructure.
- **Complexity:** substantial, but much of it is simple code.
- **Expected impact:** speculative; could be a surprise.

### 4.5 Self-supervised pre-training on unlabeled CFD via masked mesh modeling
- **Hypothesis:** Randomly mask 15% of input nodes' features (set to a learnable mask token); pre-train the model to predict them from surrounding context. Then fine-tune on supervised targets.
- **Why:** Standard MAE/BERT-style SSL. Should build better geometric representations even without more labeled data.
- **Complexity:** substantial (~200 LOC: masked collator, reconstruction head, two-phase training)
- **Expected impact:** medium on OOD geometry; long-pole effect.

### 4.6 Target smoothing / label perturbation for regression robustness
- **Hypothesis:** Add Gaussian noise N(0, 0.05 * y_std) to targets each batch.
- **Why:** Analogue to label smoothing for classification. Regularizes against overfit to exact training values; particularly useful on small datasets.
- **Complexity:** trivial (~5 LOC)
- **Expected impact:** small; cheap to try.

### 4.7 Lookahead optimizer wrapping AdamW
- **Hypothesis:** Wrap AdamW in Lookahead (k=5 fast steps, α=0.5 slow-weight update).
- **Why:** Lookahead stabilizes training especially for noisy gradients (which we have due to mesh-size variability). Proven small-but-consistent gains on many tasks.
- **Complexity:** trivial (~50 LOC, or import)
- **Expected impact:** small-medium.

### 4.8 Shampoo / SOAP second-order optimizer
- **Hypothesis:** Replace AdamW with Shampoo (or SOAP — the more tractable variant). Second-order preconditioning.
- **Why:** Second-order methods are theoretically better for small-batch regimes with curvature variation (our setup). Recent Shampoo variants are tractable.
- **Complexity:** substantial (~200 LOC or external package)
- **Expected impact:** medium; active area of research.

### 4.9 Target tokenization with VQ-VAE prior
- **Hypothesis:** Train a VQ-VAE on (Ux, Uy, p) tuples per node; then the surrogate predicts discrete tokens → decoded back. Discretization adds a learned manifold prior.
- **Why:** VQ-VAE is used for PDE surrogates in recent work; its discrete bottleneck regularizes and prevents the model from predicting unphysical values.
- **Complexity:** substantial (~500 LOC)
- **Expected impact:** speculative; likely helps tail performance but risks median.

### 4.10 Test-Time Adaptation (TTA) via a few steps of gradient descent with aux loss
- **Hypothesis:** At inference, run 5 gradient-descent steps on the model's weights using a divergence-free auxiliary loss; then predict.
- **Why:** The held-out splits have OOD geometries where the base model may be uncalibrated. A few steps of physics-based TTA can close the gap.
- **Complexity:** substantial (~150 LOC)
- **Expected impact:** medium on OOD; expensive at eval.

### 4.11 Learnable signed distance function head (auxiliary geometric supervision)
- **Hypothesis:** Add an auxiliary head that predicts, for each node, the signed distance to the nearest foil surface (we have this in dsdf[0] as ground truth). Supervised aux loss forces the trunk to encode geometry cleanly.
- **Why:** dsdf features are inputs, but the model may not fully internalize them. Adding them as a prediction target ensures the trunk builds geometry-aware representations.
- **Complexity:** moderate (~40 LOC)
- **Expected impact:** small-medium; cheap insurance.

### 4.12 Cross-sample Siamese contrastive pretraining
- **Hypothesis:** Use contrastive learning to pull same-geometry-different-Re representations together and push apart different-geometry.
- **Why:** Re should be a smooth axis of variation; geometry is discrete. Contrastive objective enforces this structure.
- **Complexity:** substantial (~200 LOC)
- **Expected impact:** speculative, likely helps OOD Re.

### 4.13 Wavelet positional encoding instead of Fourier
- **Hypothesis:** Replace/augment positional encoding with wavelets (Daubechies) — multi-resolution spatial features.
- **Why:** Wavelets have localization in both position and frequency, which Fourier lacks. For a field problem with sharp local gradients (stagnation point, TE), wavelets can be more expressive.
- **Complexity:** moderate (~80 LOC)
- **Expected impact:** small-medium; under-explored.

### 4.14 Reversible residual layers for memory savings + deeper model
- **Hypothesis:** Make the TransolverBlock reversible (RevNet-style); free up activations → run 10 layers instead of 5 at same VRAM.
- **Why:** The 96GB VRAM is not necessarily bottleneck, but activations at 242K nodes and n_hidden=128 across 10 layers may be significant. Reversibility unlocks depth cheaply.
- **Complexity:** substantial (~250 LOC)
- **Expected impact:** medium; enables one of the simpler wins (deeper model).

### 4.15 Grokking schedule — long training beyond the timeout
- **Hypothesis:** Train for 200 epochs (would need SENPAI_MAX_EPOCHS change or throughput gains); look for late grokking jumps.
- **Why:** Grokking is real for small datasets with rich structure. The 50-epoch cap may be pre-grokking.
- **Complexity:** NA — this is a policy question, not a code change. Included for completeness; **out of scope given SENPAI_MAX_EPOCHS constraint** — but mentioning that throughput gains (AMP, fused ops, better data loading) could free up epoch budget.
- **Expected impact:** speculative.

### 4.16 Bayesian deep ensembles (5 models with different init seeds, average at eval)
- **Hypothesis:** Train 5 models with different seeds in parallel (across 8 GPUs, use 5); average predictions.
- **Why:** Ensemble averaging is the most reliable way to reduce variance-driven MAE. Trivially parallel.
- **Complexity:** moderate (~50 LOC if training orchestration handles 5 seeds)
- **Expected impact:** medium, but adds orchestration overhead. Worth a one-time run for a "best-we-can" number.

---

## Final Notes

**Dataset-property recap drive these ideas:**
- 10x per-sample y_std variance ⇒ relative loss, per-sample normalization, Re-conditioned weights.
- Heavy-tailed p ⇒ Huber, relative error, clipping, spectral loss, VQ-VAE.
- ~1% surface fraction ⇒ surface-aware decoders, near-surface volume bands, TE Kutta.
- 3 physical domains with very different mesh sizes and physics ⇒ MoE, per-domain LayerNorm, hierarchical slicing.
- OOD camber / OOD Re ⇒ HyperNetworks, curriculum, SSL, panel-method pretraining.
- Small dataset (~1500 samples) ⇒ augmentation (h-flip), pretraining (synthetic), label smoothing, ensembles.
