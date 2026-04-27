# Hypothesis Briefs — Round 1 (2026-04-27)

The `icml-appendix-willow-r1` branch starts from vanilla Transolver baseline (see `target/BASELINE.md`). Prior research on `kagent_v_students` established a proven winning recipe that reached val_avg/mae_surf_p ~49.4 (vs vanilla ~88).

**Round 1 strategy:**
- **alphonse** ports the proven winning recipe in a single bundled PR (with ablations) — sets the new track baseline.
- The other 7 students each test ONE novel direction not previously tried. They build on vanilla; if alphonse's bundle merges first they can rebase and re-verify.

All recipe ingredients (L1+sw=1, AMP+grad_accum, Fourier PE σ=0.7 m=160, FiLM(log Re), SwiGLU, nl=3, slice_num=8) are covered in alphonse's bundle ablation matrix — no need for separate single-ingredient PRs.

---

## alphonse — `proven-recipe-bundled-verify`

**Hypothesis.** Apply ALL proven prior wins as a bundled recipe and verify val_avg/mae_surf_p ≈ 49 / test_avg/mae_surf_p ≈ 42 reproduces on this fresh branch. This sets the new track baseline in one merge and gives subsequent experiments a clean target.

**Predicted delta.** vs vanilla baseline (~88 val): ~ −38 val (−43%). Strong winner.

**Recipe components to apply (all in one PR):**
1. **Loss**: switch MSE → L1 (`F.l1_loss` over masked elements). Set `surf_weight=1.0` (was 10.0). Add `--loss_type {mse, l1}` and `--surf_weight` flags.
2. **AMP + grad_accum**: enable `torch.cuda.amp.autocast(dtype=torch.bfloat16)` around forward + loss (eval in fp32). Add `--amp` flag and `--grad_accum N` (default 4). Loss is divided by `grad_accum`; `optimizer.step()` only every `grad_accum` mini-batches. No GradScaler needed for bf16.
3. **Fourier PE + FiLM(log Re)**: add a `FourierPositional2D` module mapping `(x_pos, z_pos)` (dims 0-1 of x) through fixed Gaussian Fourier features (m=160 frequencies, σ=0.7) → 320-dim. Concatenate with `x[..., 2:24]`; update `preprocess` input dim to 320+22=342. Add a `FiLM` block conditioned on `log_Re = x[..., 13:14].mean(dim=1)` producing (γ, β) modulating the preprocess output: `fx = fx * (1 + γ) + β`. Add `--fourier_features {none, fixed}`, `--fourier_m`, `--fourier_sigma`, `--film` flags.
4. **SwiGLU MLP**: replace the GELU MLP in `TransolverBlock.mlp` with SwiGLU: `gate=Linear(d, d·r)`, `value=Linear(d, d·r)`, `out=Linear(d·r, d)`, with `silu(gate) * value`. Keep `mlp2` head as plain MLP. Add `--swiglu` flag (default True).
5. **Architecture floor**: `n_layers=3, slice_num=8` (replace defaults `n_layers=5, slice_num=64`). Add `--n_layers`, `--slice_num` flags.
6. **Seed control**: add `--seed` flag, set torch.manual_seed(seed) at start.

**Sweep matrix (8 GPUs).** 3-seed verification of bundled recipe + 5 single-axis ablations:

| GPU | Config | seed | wandb_name |
|-----|--------|------|------------|
| 0 | full recipe | 0 | `alphonse/recipe-s0` |
| 1 | full recipe | 1 | `alphonse/recipe-s1` |
| 2 | full recipe | 2 | `alphonse/recipe-s2` |
| 3 | recipe but `n_layers=5, slice_num=64` | 0 | `alphonse/abl-vanilla-arch` |
| 4 | recipe but MSE+sw=10 | 0 | `alphonse/abl-mse` |
| 5 | recipe but no SwiGLU (gelu) | 0 | `alphonse/abl-gelu` |
| 6 | recipe but no Fourier+FiLM | 0 | `alphonse/abl-nofourier` |
| 7 | recipe but no AMP+ga (fp32, ga=1) | 0 | `alphonse/abl-noamp` |

**Reporting.** Per-run val_avg, test_avg, best_epoch, peak VRAM, time/epoch. 3-seed mean/std on full recipe. Per-split breakdown at the best seed. Confirm budget-bound (best_epoch == terminal_epoch).

**Merge criterion.** 3-seed mean val < 70 (clear win over vanilla). Predicted: ~49.

**Reproduce vanilla anchor (for comparison)**:
```bash
cd target && python train.py --agent alphonse --epochs 50 \
    --wandb_group alphonse/proven-recipe-bundled-verify \
    --wandb_name alphonse/abl-vanilla-arch
```

**Reproduce winner**:
```bash
cd target && python train.py --agent alphonse --epochs 50 \
    --loss_type l1 --surf_weight 1 --amp --grad_accum 4 \
    --fourier_features fixed --fourier_m 160 --fourier_sigma 0.7 --film \
    --swiglu --n_layers 3 --slice_num 8 --seed 0 \
    --wandb_group alphonse/proven-recipe-bundled-verify \
    --wandb_name alphonse/recipe-s0
```

**W&B group**: `alphonse/proven-recipe-bundled-verify`.

---

## askeladd — `optimizer-sweep-lion-ademamix-sf`

**Hypothesis.** AdamW with cosine annealing is the default but every prior winner hit best val at the **terminal epoch**, meaning cosine under-decays at the end. Alternative optimizers may give better convergence in the budget-bound regime: Lion (sign-momentum, faster early descent), AdEMAMix (dual-EMA, no warmup needed), and Schedule-Free AdamW (Defazio 2024, no schedule at all).

**Predicted delta.** vs vanilla (~88 val): ~ −3 to −8 val. Best optimizer beats AdamW+cosine by 1-3 val on top of any baseline.

**Instructions.** Modify only `target/train.py`.
1. Add `--optimizer {adamw, lion, ademamix, sf_adamw}` flag (default `adamw`).
2. **Lion**: copy 30-line implementation from the Lion-pytorch reference. Lion typically uses 3-10× lower LR than AdamW.
3. **AdEMAMix**: copy reference implementation. Has 3 EMA scales; uses lr=1e-3, betas=(0.9, 0.999, 0.9999).
4. **SF-AdamW**: install `schedulefree` package (add to pyproject.toml in same PR). When `--optimizer sf_adamw`, DISABLE cosine scheduler (use constant LR via the schedule-free internal averaging). Default lr=1e-3.
5. For non-SF optimizers, keep cosine. For SF, no cosine, just pass-through.
6. Add `--seed` flag.

**Sweep matrix.**

| GPU | optimizer | lr | seed | wandb_name |
|-----|-----------|-----|------|------------|
| 0 | adamw | 5e-4 | 0 | `askeladd/adamw-s0` (anchor) |
| 1 | adamw | 5e-4 | 1 | `askeladd/adamw-s1` |
| 2 | lion  | 5e-5 | 0 | `askeladd/lion-lr5e5-s0` |
| 3 | lion  | 1e-4 | 0 | `askeladd/lion-lr1e4-s0` |
| 4 | ademamix | 1e-3 | 0 | `askeladd/ademamix-s0` |
| 5 | ademamix | 5e-4 | 0 | `askeladd/ademamix-lr5e4-s0` |
| 6 | sf_adamw | 1e-3 | 0 | `askeladd/sfadamw-s0` |
| 7 | sf_adamw | 1e-3 | 1 | `askeladd/sfadamw-s1` |

**Reporting.** Per-run val_avg, test_avg, best_epoch, time/epoch. Loss curves at every 10% of training to see convergence speed differences. Note whether SF-AdamW reaches best val at the same epoch as AdamW or later.

**Merge criterion.** Best alternative optimizer 2-seed mean < AdamW 2-seed mean by ≥1.5 val.

**Notes.** Run on **vanilla baseline** (no other recipe changes). If alphonse's recipe merges first, this PR can be re-verified on top.

**W&B group**: `askeladd/optimizer-sweep`.

---

## edward — `torch-compile-flash-attn`

**Hypothesis.** Pure throughput win. `torch.compile(mode="max-autotune")` fuses einsums + softmax + linears in PhysicsAttention and SwiGLU/MLP. Pinning `SDPBackend.FLASH_ATTENTION` removes Python branching in attention dispatch. Predicted 1.3-1.8× per-epoch speedup → 30-60% more epochs in the 30-min cap → val gain.

**Predicted delta.** vs vanilla (~88 val): ~ −5 to −10 val from extra epochs alone (vanilla currently runs ~25 epochs in 30min).

**Instructions.** Modify only `target/train.py`.
1. Add `--torch_compile` flag (default True). After model construction:
```python
if cfg.torch_compile:
    model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
```
2. Add `--amp` flag (bf16 autocast around train forward + loss; eval in fp32). torch.compile pairs naturally with AMP.
3. Wrap the slice-token attention in `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`:
```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out_slice = F.scaled_dot_product_attention(q, k, v, ...)
```
4. Add `--seed` flag.
5. **Critical sanity check**: log `epoch_time_s` and verify ≥1.4× speedup vs no-compile. If torch.compile causes graph breaks or recompilations on variable mesh sizes, set `dynamic=True` and benchmark again.

**Sweep matrix.**

| GPU | torch_compile | amp | flash_attn | seed | wandb_name |
|-----|---------------|-----|------------|------|------------|
| 0 | off | off | off | 0 | `edward/baseline-s0` (anchor) |
| 1 | off | off | off | 1 | `edward/baseline-s1` |
| 2 | off | bf16 | off | 0 | `edward/amp-only-s0` |
| 3 | off | bf16 | off | 1 | `edward/amp-only-s1` |
| 4 | on | bf16 | off | 0 | `edward/compile-amp-s0` |
| 5 | on | bf16 | off | 1 | `edward/compile-amp-s1` |
| 6 | on | bf16 | on  | 0 | `edward/compile-amp-flash-s0` |
| 7 | on | bf16 | on  | 1 | `edward/compile-amp-flash-s1` |

**Reporting.** Per-run val_avg, test_avg, best_epoch, time/epoch (CRITICAL — biggest diagnostic), epochs reached. Speedup table comparing each to baseline anchor.

**Merge criterion.** compile+amp+flash 2-seed mean val < baseline 2-seed mean val by ≥3 val OR ≥40% time/epoch reduction.

**Notes.** Run on **vanilla baseline** (no other recipe changes). torch.compile first-iteration overhead is large; report best_epoch + epochs_reached carefully.

**W&B group**: `edward/torch-compile-flash`.

---

## fern — `sobolev-arclength-grad-loss`

**Hypothesis.** L1 surface pressure loss is point-wise — a model can lower mean L1 while having wiggly per-node errors. The integrated MAE ranking metric penalizes wiggly predictions. Adding a **Sobolev** (gradient-matching) loss on the *arclength derivative* of pressure forces the model to predict the *shape* of the pressure curve along the foil surface, not just per-node values. The prior research showed loss reformulations have outsized effect; Sobolev loss is a clean, novel addition.

**Predicted delta.** vs vanilla (~88 val): ~ −5 to −12 val, largest lift on `val_geom_camber_*`.

**Instructions.** Modify only `target/train.py`.
1. Add a function `surface_arclength_grad_loss(pred, y, x, is_surface, mask)`:
   - For each sample in batch, get surface nodes (`is_surface=True & mask=True`).
   - Sort surface nodes by `x[..., 2]` (saf, signed arc length).
   - Compute finite difference: `∇p_pred = pred_p[i+1] - pred_p[i]` along sorted order. Same for `∇p_true` (in normalized space, then de-normalize for fair scale).
   - Return `L1(∇p_pred, ∇p_true)` averaged over surface-node-pairs.
   - Skip foils with < 4 surface nodes.
2. Add `--lambda_grad` flag (default 0.0, sweep over {0.01, 0.1, 0.3, 1.0}).
3. Total loss = `vol_loss + surf_weight * surf_loss + lambda_grad * grad_loss`.
4. Use L1 base loss (more aligned with the metric).
5. Add `--seed` flag.

**Sweep matrix.**

| GPU | base_loss | lambda_grad | seed | wandb_name |
|-----|-----------|-------------|------|------------|
| 0 | mse (vanilla) | 0.0 | 0 | `fern/vanilla-s0` (anchor) |
| 1 | l1 | 0.0 | 0 | `fern/l1-only-s0` (L1 control) |
| 2 | l1 | 0.01 | 0 | `fern/l1-grad0.01-s0` |
| 3 | l1 | 0.1  | 0 | `fern/l1-grad0.1-s0` |
| 4 | l1 | 0.1  | 1 | `fern/l1-grad0.1-s1` |
| 5 | l1 | 0.3  | 0 | `fern/l1-grad0.3-s0` |
| 6 | l1 | 1.0  | 0 | `fern/l1-grad1.0-s0` |
| 7 | l1 | 0.1  | 2 | `fern/l1-grad0.1-s2` (3rd seed at best lambda) |

**Reporting.** Per-run val_avg, test_avg, best_epoch, per-split breakdown. **Critical**: compare per-split MAE (especially OOD camber splits where shape-fidelity should matter most). Report whether grad loss helps/hurts surface vs volume.

**Merge criterion.** Best lambda_grad 2-seed mean val < `l1-only` 2-seed mean val by ≥1.5 val.

**Notes.** Run on **vanilla baseline** + L1 (so we isolate the grad-loss effect). If alphonse's recipe merges first, re-verify on top.

**W&B group**: `fern/sobolev-arclength-grad-loss`.

---

## frieren — `se2-chord-line-frame`

**Hypothesis.** The CFD physics is approximately invariant under SE(2) rotations of the airfoil + flow field. AoA is currently a scalar input feature the model must learn to translate into a coordinate transformation. Hard-coding equivariance via a *change of frame* — rotate `(x, z)` by `−AoA1` so the chord-line is always horizontal, then rotate predicted `(Ux, Uy)` back by `+AoA1` — collapses the AoA continuous symmetry, removing 5-10° (raceCar) to 11° (cruise) of input variation the model would otherwise have to learn.

**Predicted delta.** vs vanilla (~88 val): ~ −5 to −15 val on OOD camber splits, smaller on val_single_in_dist.

**Instructions.** Modify only `target/train.py`.
1. Pre-loss frame rotation:
   - Extract `aoa1 = x[..., 14:15]` (AoA foil 1 in radians).
   - Build rotation matrix per sample: `R = [[cos(-aoa1), -sin(-aoa1)], [sin(-aoa1), cos(-aoa1)]]`.
   - Rotate `(x[..., 0], x[..., 1])` (positions): `xz_rot = R @ xz`.
   - Rotate `(y[..., 0], y[..., 1])` (Ux, Uy targets): `Uxy_rot = R @ Uxy` for loss.
   - Pressure (`y[..., 2]`) is rotation-invariant: leave alone.
2. After model prediction:
   - Rotate predicted `(Ux, Uy)` back by `+aoa1`: `R_inv @ pred_Uxy`.
   - Pressure pred: leave alone.
3. CRITICAL: re-compute or re-use `x_mean`, `x_std`. Since rotation preserves magnitudes, re-using existing stats incurs only ~5% calibration loss. For round 1, RE-USE existing stats (simpler, smaller code change).
4. For tandem cases (foil2 AoA exists in `x[..., 18:19]`), use `aoa1` as the canonical frame — adequate since we're testing the principle.
5. Add `--use_chord_frame` flag (default True). 
6. Add `--seed` flag.

**Sweep matrix.**

| GPU | use_chord_frame | seed | wandb_name |
|-----|-----------------|------|------------|
| 0 | off | 0 | `frieren/baseline-s0` (anchor) |
| 1 | off | 1 | `frieren/baseline-s1` |
| 2 | on  | 0 | `frieren/chord-s0` |
| 3 | on  | 1 | `frieren/chord-s1` |
| 4 | on  | 2 | `frieren/chord-s2` |
| 5 | on (drop AoA from x dim 14) | 0 | `frieren/chord-noaoa-s0` |
| 6 | on (drop AoA from x dim 14) | 1 | `frieren/chord-noaoa-s1` |
| 7 | on (drop AoA + drop AoA2) | 0 | `frieren/chord-noaoa12-s0` |

**Reporting.** Per-split breakdown is critical. Expected: largest improvement on `val_geom_camber_*` and `val_re_rand` (OOD splits), modest on `val_single_in_dist`.

**Merge criterion.** chord_frame 2-seed mean val < baseline 2-seed mean val by ≥3 val on OOD splits.

**Notes.** Run on **vanilla baseline**. The frame-rotation can be implemented in ~50 lines.

**W&B group**: `frieren/se2-chord-line-frame`.

---

## nezuko — `sdf-geometric-features`

**Hypothesis.** The current `dsdf` (dims 4-11) is a distance-based descriptor but doesn't directly encode the **signed distance** to the nearest surface or a **nearest-surface direction**. Adding 4-6 explicit geometric features per node (sdf_min, sdf_argmin, sdf_sign, softmin-direction) gives the model direct access to boundary-layer geometry — particularly useful for OOD camber splits where the boundary layer changes shape but distance metrics generalize.

**Predicted delta.** vs vanilla (~88 val): ~ −3 to −8 val, largest lift on OOD camber and val_re_rand.

**Instructions.** Modify only `target/train.py`.
1. Add a `compute_geom_features(x)` function called on the batch:
   - `dsdf_abs = x[..., 4:12].abs()` # [B, N, 8]
   - `sdf_min, sdf_argmin = dsdf_abs.min(dim=-1)` # [B, N], [B, N]
   - `sdf_sign = torch.where(x[..., 12:13] == 1, torch.zeros_like(sdf_min), torch.sign(x[..., 4:12].gather(-1, sdf_argmin.unsqueeze(-1)).squeeze(-1)))` # signed distance
   - `softmin_weights = F.softmin(dsdf_abs * 10.0, dim=-1)` # smoothed nearest-vote
   - `softmin_direction = (softmin_weights * one_hot_dirs).sum(...)` (use a fixed 8-direction one-hot encoding, or just keep softmin_weights as-is).
2. Concatenate 4-6 new features to `x` before normalization. Update `fun_dim` accordingly.
3. Re-use existing x_mean / x_std stats; the new features are already in [-1, 1] range approximately, so add them after normalization (skip normalization for them) OR normalize them to mean=0, std=1 with batch stats on first batch.
4. Add `--use_geom_features` flag (default True).
5. Add `--seed` flag.

**Sweep matrix.**

| GPU | use_geom_features | seed | wandb_name |
|-----|-------------------|------|------------|
| 0 | off | 0 | `nezuko/baseline-s0` (anchor) |
| 1 | off | 1 | `nezuko/baseline-s1` |
| 2 | on (sdf_min only) | 0 | `nezuko/sdf-only-s0` |
| 3 | on (sdf_min + sdf_sign) | 0 | `nezuko/sdf-sign-s0` |
| 4 | on (full: sdf_min, sign, softmin) | 0 | `nezuko/sdf-full-s0` |
| 5 | on (full) | 1 | `nezuko/sdf-full-s1` |
| 6 | on (full) | 2 | `nezuko/sdf-full-s2` |
| 7 | on (sdf_min + sdf_sign + raw dsdf duplicated) | 0 | `nezuko/sdf-redundant-s0` |

**Reporting.** Per-split breakdown. The hypothesis specifically predicts OOD camber improvement.

**Merge criterion.** sdf-full 3-seed mean val < baseline 2-seed mean val by ≥1.5 val.

**Notes.** Run on **vanilla baseline**. Total ~30 LOC change.

**W&B group**: `nezuko/sdf-geometric-features`.

---

## tanjiro — `rmsnorm-rezero-arch`

**Hypothesis.** Modern transformer architecture: replace LayerNorm with RMSNorm (drops bias, halves norm cost) and replace `x + f(x)` with `x + α · f(x)` (ReZero, α initialized to 0). RMSNorm is faster (small but compounding 5-15% throughput gain). ReZero starts each block at identity, often improves convergence speed in budget-bound regimes and can remove the need for warmup. Combined: faster + more stable.

**Predicted delta.** vs vanilla (~88 val): ~ −2 to −5 val. Modest but compounds with everything.

**Instructions.** Modify only `target/train.py`.
1. Add `RMSNorm` class:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
```
2. Add `--norm_type {layernorm, rmsnorm}` flag (default `rmsnorm`).
3. Add `--rezero` flag (default True). In `TransolverBlock`:
```python
self.alpha_attn = nn.Parameter(torch.zeros(1))
self.alpha_mlp = nn.Parameter(torch.zeros(1))
# In forward:
fx = fx + self.alpha_attn * self.attn(self.ln_1(fx))
fx = fx + self.alpha_mlp * self.mlp(self.ln_2(fx))
```
When `--rezero=False`, use plain residuals.
4. Add `--seed` flag.

**Sweep matrix.**

| GPU | norm_type | rezero | seed | wandb_name |
|-----|-----------|--------|------|------------|
| 0 | layernorm | off | 0 | `tanjiro/ln-baseline-s0` (anchor) |
| 1 | layernorm | off | 1 | `tanjiro/ln-baseline-s1` |
| 2 | rmsnorm   | off | 0 | `tanjiro/rms-s0` |
| 3 | rmsnorm   | off | 1 | `tanjiro/rms-s1` |
| 4 | layernorm | on  | 0 | `tanjiro/ln-rezero-s0` |
| 5 | rmsnorm   | on  | 0 | `tanjiro/rms-rezero-s0` |
| 6 | rmsnorm   | on  | 1 | `tanjiro/rms-rezero-s1` |
| 7 | rmsnorm   | on  | 2 | `tanjiro/rms-rezero-s2` |

**Reporting.** Time/epoch is a key diagnostic — RMSNorm should be measurably faster. Then val_avg comparison. ReZero should accelerate early convergence.

**Merge criterion.** Best (rms+rezero) 3-seed mean val < layernorm-baseline 2-seed mean val by ≥1.0 val.

**Notes.** Run on **vanilla baseline**. ~50 LOC.

**W&B group**: `tanjiro/rmsnorm-rezero`.

---

## thorfinn — `siren-preprocess-activation`

**Hypothesis.** Coordinate inputs `(x, z)` benefit from sinusoidal activations (Sitzmann 2020 SIREN). The current `preprocess` MLP uses GELU, which has limited high-frequency expressivity. Replacing the first 1-2 layers with SIREN-style `sin(ω₀ · Wx + b)` activations gives the model direct sinusoidal basis functions, which compose well with positional encoding and match the periodic-like flow features seen in CFD around airfoils. SIREN was never tried in prior research.

**Predicted delta.** vs vanilla (~88 val): ~ −2 to −5 val, largest lift on OOD camber.

**Instructions.** Modify only `target/train.py`.
1. Add a `Sine(omega_0=30.0)` activation class:
```python
class Sine(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x):
        return torch.sin(self.omega_0 * x)
```
2. Add SIREN initialization scheme — for the FIRST sine layer:
```python
def siren_init_first(m, omega_0=30.0):
    if isinstance(m, nn.Linear):
        n_in = m.weight.shape[1]
        nn.init.uniform_(m.weight, -1.0/n_in, 1.0/n_in)
```
For subsequent sine layers:
```python
def siren_init_hidden(m, omega_0=30.0):
    if isinstance(m, nn.Linear):
        n_in = m.weight.shape[1]
        bound = math.sqrt(6.0 / n_in) / omega_0
        nn.init.uniform_(m.weight, -bound, bound)
```
3. Replace ONLY the `preprocess` MLP with a 2-layer SIREN MLP when `--preprocess_act sine`. Do NOT touch the FFNs inside `TransolverBlock` (deep SIREN is unstable).
4. Add `--preprocess_act {gelu, sine}` flag (default `sine`).
5. Add `--omega_0` flag (default 30.0; sweep 10, 30, 60).
6. Add `--seed` flag.

**Sweep matrix.**

| GPU | preprocess_act | omega_0 | seed | wandb_name |
|-----|----------------|---------|------|------------|
| 0 | gelu | — | 0 | `thorfinn/gelu-s0` (anchor) |
| 1 | gelu | — | 1 | `thorfinn/gelu-s1` |
| 2 | sine | 10  | 0 | `thorfinn/sine-w10-s0` |
| 3 | sine | 30  | 0 | `thorfinn/sine-w30-s0` |
| 4 | sine | 30  | 1 | `thorfinn/sine-w30-s1` |
| 5 | sine | 30  | 2 | `thorfinn/sine-w30-s2` |
| 6 | sine | 60  | 0 | `thorfinn/sine-w60-s0` |
| 7 | sine | 30 (3-layer SIREN preprocess) | 0 | `thorfinn/sine-3layer-s0` |

**Reporting.** Per-split breakdown. SIREN may be unstable — report any divergent runs explicitly. Compare convergence speed (loss curves at every 5 epochs) — SIREN often converges faster on coordinate inputs.

**Merge criterion.** Best SIREN config 3-seed mean val < gelu 2-seed mean val by ≥1.0 val.

**Notes.** Run on **vanilla baseline**. INIT IS CRITICAL. ~40 LOC. If divergence occurs at omega_0=30, try omega_0=10.

**W&B group**: `thorfinn/siren-preprocess`.

---

## Notes for the assign-experiment skill

- All hypotheses build on the **vanilla baseline** (current `train.py` HEAD, val ~88).
- Training cmd template: `cd target && python train.py --agent <student> --wandb_group <group> --wandb_name <name> --epochs 50 [hypothesis flags]`.
- Use `python train.py --help` to confirm CLI flag spellings.
- Per-PR W&B group should match the slug prefix.
- Pin seeds explicitly; `--seed` should be added by the student if missing.
- After alphonse's recipe merges, the other 7 PRs will be evaluated with their delta vs the recipe baseline (re-verification on top, in a follow-up if the original PR was vs vanilla).
