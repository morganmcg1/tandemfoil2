# SENPAI Research State

- **Date**: 2026-04-27 21:30 UTC
- **Most recent research direction from human researcher team**: None — no new directives for this track (last check: operational issue #257 about GitHub label-indexing regression, fully resolved)
- **Current research focus**: Round 2 experiments on TandemFoilSet CFD surrogate. Two baselines merged (EMA + vanilla). Current best val_avg/mae_surf_p=131.71. All 7 students WIP with diverse hypotheses. No idle students.

## Current Baseline

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| val_avg/mae_surf_p | **131.71** | #193 | Vanilla baseline anchor (EMA always-on), best epoch 11/14, timeout-limited |
| test_avg/mae_surf_p | NaN | #193 | test_geom_camber_cruise NaN (pre-fix run; fix already in train.py) |
| test_avg/mae_surf_p | **119.58** | #209 | EMA run (NaN-corrected), best available test metric |

**Key structural insight**: EMA is always-on in train.py (baked into code, ema_decay=0.999). All future experiments train model + maintain EMA; val/test always evaluates from EMA model. The "vanilla" baseline IS an EMA run.

**Timeout constraint**: ~30 min wall clock → only ~14 epochs of 50 configured at baseline speed. Per-epoch improvements (larger batch = fewer steps, better loss = faster convergence) are especially valuable.

**VRAM headroom**: 42 GB used of 96 GB available — significant room for larger models or batch sizes.

## Merged Experiments

| PR | Student | Hypothesis | Result |
|----|---------|------------|--------|
| #193 | charliepai2c3-alphonse | Vanilla baseline anchor (default train.py, EMA always-on) | **131.71** — CURRENT BEST |
| #209 | charliepai2c3-nezuko | "EMA experiment" (same as vanilla, confirms always-on EMA) | 133.66 — MERGED first |

## Active Round 2 Experiments (WIP)

| PR | Student | Hypothesis | Key Change |
|----|---------|------------|------------|
| #198 | charliepai2c3-askeladd | L1 loss with surf_weight=1 | Loss: MSE→L1, aligns training with MAE metric |
| #200 | charliepai2c3-edward | surf_weight sweep: 20 and 50 | Stronger surface pressure signal vs vol |
| #203 | charliepai2c3-fern | Wider Transolver n_hidden=256 | More model capacity for complex flow fields |
| #207 | charliepai2c3-frieren | LR warmup + cosine to 1e-3 | Higher peak LR with stability via warmup |
| #214 | charliepai2c3-tanjiro | Per-channel pressure up-weighting (3× on p) | Target pressure channel in MSE loss |
| #219 | charliepai2c3-thorfinn | Per-channel decoder heads (Ux, Uy, p) | Field-specific output MLPs |

## Round 2 — Additional Active Experiments

| PR | Student | Hypothesis | Key Change |
|----|---------|------------|------------|
| #261 | charliepai2c3-nezuko | Fourier PE on (x,z) — spatial frequency encoding | Add sin/cos Fourier features for mesh node positions before preprocess MLP (num_freqs=16, sigma=1.0) |

## Current Research Themes

1. **Loss formulation**: L1 vs MSE (#198), per-channel pressure weighting (#214)
2. **Hyperparameter tuning**: surf_weight sweep 10→20,50 (#200), LR warmup + cosine to 1e-3 (#207)
3. **Architecture exploration**: Wider model n_hidden=256 (#203), per-channel decoder heads (#219), Fourier PE (nezuko TBD)
4. **Regularization**: EMA always-on (both merged); next: dropout sweep, stochastic depth

## Key Observations

- **val_geom_camber_cruise** consistently best split (~100-107), **val_single_in_dist** consistently worst (~162-172). Pattern tracks with domain complexity and pressure magnitude ranges in program.md.
- **VRAM**: Only 42 GB used of 96 GB — room for batch_size=8 (more stable gradients) or n_hidden=256+ (fern is testing this)
- **Speed bottleneck**: ~132 s/epoch with batch_size=4 → only 14 epochs in 30 min. Speed improvements would be highly valuable.
- **EMA always-on**: Both merged PRs had EMA enabled — it's structural, not experimental. New experiments cannot compare "with vs without" EMA at this point.
- **Convergence**: Both runs improved through epoch 11-14 with no sign of plateau. Runs are timeout-limited, not convergence-limited.

## Potential Next Research Directions (After Round 2 Results)

**High priority (likely to beat baseline given timeout constraint):**
- **Fourier PE on (x,z)**: Spatial frequency features for mesh node positions (nezuko, assigned)
- **Huber loss (delta=0.5-1.0)**: Smooth transition between L1 and L2, robustness to high-Re pressure extremes
- **SwiGLU activations**: Gated MLP replacement in TransolverBlock (known win in transformers)
- **Deeper model**: n_layers=7 or 8, wider hidden (128→192) — depth vs width tradeoff
- **Batch size=8 or 16**: 42 GB VRAM → room to double/quadruple batch; fewer optimizer steps may need higher LR

**Architecture innovations:**
- **FiLM conditioning**: Re and NACA params (dims 13-23) injected as global FiLM modulation per TransolverBlock
- **n_head sweep**: Currently nh=4; try nh=1, 2, 8 — attention head width affects slice quality
- **slice_num sweep**: Currently 64; try 32 or 128 — controls number of physics tokens

**OOD generalization:**
- **Domain-tag embeddings**: Append domain ID (raceCar/cruise) as learned embedding
- **Re log-normalization**: Global log(Re) normalization as additional input feature
- **Geometry augmentation**: Random x-flip of mesh (symmetric geometry) during training

## Key Constraints

- VRAM: 96 GB per GPU, meshes up to 242K nodes
- Timeout: ~30 min wall clock → ~14 epochs at baseline speed (SENPAI_TIMEOUT_MINUTES)
- Epochs cap: controlled by SENPAI_MAX_EPOCHS env var
- Data loaders are read-only (only train.py is editable)
- Primary metric: val_avg/mae_surf_p (lower is better, current best = 131.71)
- Baseline model: 0.66M params, 42 GB VRAM at batch_size=4
