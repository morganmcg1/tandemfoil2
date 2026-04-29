# Baseline (icml-appendix-charlie-pai2f-r1)

Seven winners merged into `train.py`:
- **PR #1101 (thorfinn)** — regime-matched schedule (warmup=1, T_max=13, eta_min=lr/100)
- **PR #1138 (frieren)** — Random Fourier Features on (x, z), n_freq=32, sigma=1.0
- **PR #1160 (alphonse)** — SwiGLU FFN replacing GELU MLP in TransolverBlocks (param-matched, ~0.689M)
- **PR #1158 (thorfinn)** — FiLM domain conditioning: per-sample global features → (γ,β) scale/shift on all LayerNorms
- **PR #1197 (alphonse)** — AMP (bfloat16) + n_hidden=160 capacity scaling: same VRAM, +53% params, +2 epochs/30-min
- **PR #1198 (askeladd)** — Online loss-weighted curriculum: EMA per-sample importance weighting in loss (not sampler), ema_alpha=0.3, temperature=0.3 pow scaling, 3-epoch warmup
- **PR #1221 (thorfinn)** — Wider FiLMNet: Linear(11→512)→GELU→Linear(512→3200), 2.70M total params

All subsequent experiments compare against this stacked baseline.

## Current best (round-7 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **61.114** (epoch 13/13, still descending) | #1221 | Wider FiLMNet MLP 512 on online curriculum baseline |
| `test_avg/mae_surf_p` | **52.989** (4 splits, all finite MAE) | #1221 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 63.878 | 0.783 | 0.408 |
| `val_geom_camber_rc` | 73.107 | 1.453 | 0.631 |
| `val_geom_camber_cruise` | 43.974 | 0.487 | 0.320 |
| `val_re_rand` | 63.496 | 0.978 | 0.472 |
| **avg** | **61.114** | 0.925 | 0.458 |

Per-split test (best epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 57.262 | 0.785 | 0.402 |
| `test_geom_camber_rc` | 65.395 | 1.397 | 0.591 |
| `test_geom_camber_cruise` | 36.187 | 0.451 | 0.278 |
| `test_re_rand` | 53.112 | 0.801 | 0.424 |
| **avg** | **52.989** | 0.858 | 0.424 |

Notes:
- `test_geom_camber_cruise` vol_loss=inf is a pre-existing dataset issue (extreme residuals in 1 sample); MAE is valid.
- Best checkpoint is epoch 13 (model still descending under 30-min wall-clock cap).
- FiLMNet widened from Linear(11→64)→GELU→Linear(64→3200) to Linear(11→512)→GELU→Linear(512→3200); total params: 2.70M (+57% vs 1.714M baseline).
- Improvement vs round-6 baseline: -8.3% on val (61.114 vs 66.636), -7.7% on test (52.989 vs 57.355). Every split improved.
- Metric summary: `models/model-charliepai2f1-thorfinn-wider-film-mlp-512-20260429-161722/metrics.yaml`

## Previous best (round-6 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **66.636** (epoch 13/13, still descending) | #1198 | Online loss-weighted curriculum on AMP+n_hidden=160+FiLM+SwiGLU+RFF baseline |
| `test_avg/mae_surf_p` | **57.355** (4 splits, all finite MAE) | #1198 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 67.542 | 0.733 | 0.419 |
| `val_geom_camber_rc` | 74.544 | 1.357 | 0.624 |
| `val_geom_camber_cruise` | 56.484 | 0.589 | 0.408 |
| `val_re_rand` | 67.974 | 0.985 | 0.519 |
| **avg** | **66.636** | 0.916 | 0.493 |

Per-split test (best epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 57.813 | — | — |
| `test_geom_camber_rc` | 66.678 | — | — |
| `test_geom_camber_cruise` | 45.332 | — | — |
| `test_re_rand` | 59.598 | — | — |
| **avg** | **57.355** | — | — |

Notes:
- `test_geom_camber_cruise` vol_loss=inf is a pre-existing dataset issue (extreme residuals in 1 sample); MAE is valid.
- Best checkpoint is epoch 13 (model still descending under 30-min wall-clock cap).
- Online loss-weighted curriculum (v2): EMA per-sample importance weighting applied in the loss function (not the DataLoader sampler), ema_alpha=0.3, temperature=0.3 pow scaling to cap weight spread to ~5-7×, 3-epoch uniform warmup.
- WeightedRandomSampler (3-domain balance) preserved; curriculum weights only affect per-sample loss scaling.
- Improvement vs round-5 baseline: -12.1% on val (66.636 vs 75.750), -11.7% on test (57.355 vs 64.983). Every split improved.

## Previous best (round-3 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **97.981** (epoch 13/13, still descending) | #1160 | SwiGLU FFN on RFF baseline |
| `test_avg/mae_surf_p` | **86.303** (4 splits, all finite MAE) | #1160 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 112.728 | 1.386 | 0.676 |
| `val_geom_camber_rc` | 108.895 | 2.079 | 0.868 |
| `val_geom_camber_cruise` | 76.103 | 0.905 | 0.528 |
| `val_re_rand` | 94.199 | 1.495 | 0.706 |
| **avg** | **97.981** | 1.466 | 0.695 |

Per-split test (best epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 95.408 | 1.328 | 0.628 |
| `test_geom_camber_rc` | 95.916 | 1.993 | 0.811 |
| `test_geom_camber_cruise` | 64.418 | 0.869 | 0.478 |
| `test_re_rand` | 89.468 | 1.326 | 0.688 |
| **avg** | **86.303** | 1.379 | 0.651 |

Notes:
- `test_geom_camber_cruise` loss=NaN/vol_loss=inf is a pre-existing dataset issue (extreme residuals in 1 sample); MAE is valid.
- Best checkpoint is the **final** epoch (epoch 13) — model still descending under the 30-min cap.
- SwiGLU gates: silu(W_gate·x) × W_up·x, replaces GELU MLP in all 5 TransolverBlocks.

## Previous best (round-2 winner — merged 2026-04-29 12:42)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **108.543** (epoch 14/14, still descending) | #1138 | RFF on (x,z) at n_freq=32, sigma=1.0 |
| `test_avg/mae_surf_p` | **96.942** (4 splits, all finite) | #1138 | All-finite paper-facing test metric |

Per-split val (epoch 14):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 125.815 | 1.686 | 0.750 |
| `val_geom_camber_rc` | 114.589 | 2.385 | 0.956 |
| `val_geom_camber_cruise` | 86.371 | 1.289 | 0.585 |
| `val_re_rand` | 107.397 | 1.797 | 0.775 |
| **avg** | **108.543** | 1.789 | 0.766 |

Per-split test:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 104.402 | 1.612 | 0.698 |
| `test_geom_camber_rc` | 106.273 | 2.367 | 0.908 |
| `test_geom_camber_cruise` | 74.043 | 1.196 | 0.541 |
| `test_re_rand` | 103.048 | 1.654 | 0.753 |
| **avg** | **96.942** | 1.707 | 0.725 |

Notes:
- frieren's RFF run was launched on the pre-#1101-merge train.py. After the
  squash merge, the merged train.py has **RFF + schedule together** for the
  first time — next runs may show further compounding gains below 108.5.
- Best checkpoint is the **final** epoch — model still descending under the
  30-min cap.

## Improvement chain

| Stage | val_avg | test_avg | PR |
|---|---|---|---|
| Provisional round-1 best (confounded) | 133.892 | 132.106 (3 finite) | #1095 (sent back) |
| Round-1 winner: regime-matched schedule | 125.438 | 112.988 | #1101 ← merged |
| Round-2 winner: RFF (on top of schedule) | 108.543 | 96.942 | #1138 ← merged |
| Round-3 winner: SwiGLU FFN (on top of RFF) | 97.981 | 86.303 | #1160 ← merged |
| Round-4 winner: FiLM domain conditioning | 84.371 | 75.076 | #1158 ← merged |
| Round-5 winner: AMP + n_hidden=160 capacity scaling | 75.750 | 64.983 | #1197 ← merged |
| Round-6 winner: Online loss-weighted curriculum | 66.636 | 57.355 | #1198 ← merged |
| Round-7 winner: Wider FiLMNet MLP 512 | **61.114** | **52.989** | #1221 ← merged |

Round-1→Round-7 cumulative improvement: **-54.2% on val, -59.9% on test**.

## Default config (`train.py` at HEAD, post-merge of #1221)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | LinearLR warmup (1 ep, 5e-7 → 5e-4) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Batch size | 4 |
| Surf weight (loss) | 10.0 |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` ≈ 15 effective epochs with AMP) |
| Sampler | WeightedRandomSampler (balanced across 3 domains) |
| Loss | MSE on normalized targets, vol + surf_weight·surf; **online EMA per-sample curriculum weighting** (ema_alpha=0.3, temp=0.3, 3-ep warmup) |
| AMP | bfloat16 autocast + GradScaler + clip_grad_norm(max_norm=1.0) |
| Model | Transolver, **n_hidden=160**, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **RFF on (x,z) n_freq=32 sigma=1.0**, **SwiGLU FFN**, **FiLM domain conditioning (wider: 512 hidden)** |
| Params | ~2.70M (1.054M base + 1.646M FiLMNet with 512 hidden) |

Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
Test-time metric for paper: `test_avg/mae_surf_p`.

## Reproduce

```
cd target/ && python train.py --agent <student> --experiment_name "<student>/baseline-default"
```

(All defaults; do NOT pass `--lr`, `--batch_size`, `--surf_weight`, etc.)
