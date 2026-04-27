# SENPAI Research State

- **Date:** 2026-04-27
- **Advisor branch:** `icml-appendix-charlie-r4`
- **Target:** TandemFoilSet (CFD surrogate over tandem-airfoil meshes)
- **Primary metric:** `val_avg/mae_surf_p` (lower is better); reported alongside `test_avg/mae_surf_p`

## Most recent human-team direction
- No GitHub Issues from the human research team at the moment.
- Charlie tracks (`icml-appendix-charlie-r1..r5`) are appendix ablation rounds for the ICML appendix, focused on small, clean, paper-defensible single-axis ablations against the vanilla Transolver baseline shipped in `target/train.py`.

## Current research focus
- We are running on a **fresh `target/`** copy: a stripped-down `train.py` with a Transolver baseline (5L / 128d / 4h / 64 slices / mlp_ratio=2 / GELU, AdamW lr=5e-4 wd=1e-4, surf_weight=10, batch=4, 50 epochs, cosine decay).
- The track is paper-appendix oriented: each PR should be a single-axis ablation that produces a clean monotonic curve (or shows clearly which value of the axis is best). We want results that read as a Table-1 row in the appendix.
- Round 4 is parallel to Round 5 (`icml-appendix-charlie-r5`). To avoid duplicate work, charlie-r4 covers axes complementary to charlie-r5:
  - charlie-r5 round-1 axes already in flight: L1-vs-MSE loss, AMP+grad-accum throughput, SwiGLU FFN, fixed-Fourier features sigma sweep, surf_weight sweep, slice_num sweep, n_layers sweep, n_head shape-preserving sweep.
  - charlie-r4 round-1 axes (this round): learning rate, weight decay, activation function, mlp_ratio, n_hidden width, batch size, optimizer family, cosine T_max / scheduler choice.

## Round 1 assignments (charlie-r4)
| Student | Axis | Sweep |
|---|---|---|
| charlie4-alphonse | learning rate | `{1e-4, 3e-4, 5e-4 (anchor), 1e-3}` |
| charlie4-askeladd | weight decay | `{0, 1e-5, 1e-4 (anchor), 1e-3}` |
| charlie4-edward | activation | `{gelu (anchor), silu, relu, tanh}` |
| charlie4-fern | mlp_ratio | `{1, 2 (anchor), 4, 8}` |
| charlie4-frieren | n_hidden width | `{64, 128 (anchor), 192, 256}` |
| charlie4-nezuko | batch size | `{2, 4 (anchor), 8, 16}` (no LR scaling) |
| charlie4-tanjiro | optimizer | `{AdamW (anchor), Adam, SGD-mom, Lion}` |
| charlie4-thorfinn | scheduler / cosine T_max | `{T_max=epochs (anchor), T_max=10, T_max=2*epochs, no-scheduler}` |

## Potential next research directions (round 2+)
- **Initialization / normalization ablations:** trunc_normal std sweep, pre-LN vs post-LN, RMSNorm vs LayerNorm.
- **Output head / loss-space ablations:** per-channel surf_weight (Ux, Uy, p separately), Huber loss, robust scale via asinh on pressure.
- **Architecture-orthogonal ablations:** unified_pos toggle (`ref` grid 4/8/16), placeholder removal, dropout sweep `{0, 0.05, 0.1, 0.2}`.
- **Slice-attention internals:** temperature initialization sweep, orthogonal init ablation on `in_project_slice`, slice-token gating.
- **Sampler ablations:** balanced-domain WRS vs uniform vs Re-stratified.
- **Training-stability ablations:** EMA decay sweep, gradient clipping value sweep, grad-norm logging only.
- **Capacity-vs-depth Pareto:** combine the winning width with the winning depth, then test efficiency-frontier configs.
- **Compounding round:** stack the per-axis winners from charlie-r4 + charlie-r5 to produce a "best appendix recipe" against vanilla.
