# Baseline (icml-appendix-charlie-pai2f-r1)

Ten winners merged into `train.py`:
- **PR #1101 (thorfinn)** — regime-matched schedule (warmup=1, T_max=13, eta_min=lr/100)
- **PR #1138 (frieren)** — Random Fourier Features on (x, z), n_freq=32, sigma=1.0
- **PR #1160 (alphonse)** — SwiGLU FFN replacing GELU MLP in TransolverBlocks (param-matched, ~0.689M)
- **PR #1158 (thorfinn)** — FiLM domain conditioning: per-sample global features → (γ,β) scale/shift on all LayerNorms
- **PR #1197 (alphonse)** — AMP (bfloat16) + n_hidden=160 capacity scaling: same VRAM, +53% params, +2 epochs/30-min
- **PR #1198 (askeladd)** — Online loss-weighted curriculum: EMA per-sample importance weighting in loss (not sampler), ema_alpha=0.3, temperature=0.3 pow scaling, 3-epoch warmup
- **PR #1221 (thorfinn)** — Wider FiLMNet: Linear(11→512)→GELU→Linear(512→3200), 2.70M total params
- **PR #1183 (edward)** — Cautious AdamW: sign-agreement mask on momentum updates, rescaled by 1/mask.mean() to preserve update norm
- **PR #1244 (edward)** — n_hidden=160→192 capacity probe: +20% hidden width, 3.47M params (+28.5%), VRAM 57 GB (vs 42 GB), 12 epochs/30-min
- **PR #1236 (askeladd)** — Sobolev-style surface gradient auxiliary loss: penalizes arc-length finite-difference pressure gradient errors along foil surface nodes; surf_grad_weight=10.0

All subsequent experiments compare against this stacked baseline.

## Current best (round-10 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **59.121** (epoch 12/12, still descending) | #1236 | Sobolev surface gradient loss, surf_grad_weight=10.0 |
| `test_avg/mae_surf_p` | **51.170** (4 splits, all finite MAE) | #1236 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (best epoch 12):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 59.445 | 0.616 | 0.384 |
| `val_geom_camber_rc` | 71.611 | 1.319 | 0.596 |
| `val_geom_camber_cruise` | 45.249 | 0.484 | 0.318 |
| `val_re_rand` | 60.178 | 0.885 | 0.447 |
| **avg** | **59.121** | 0.826 | 0.436 |

Per-split test (best epoch 12):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 53.986 | 0.627 | 0.372 |
| `test_geom_camber_rc` | 62.288 | 1.247 | 0.544 |
| `test_geom_camber_cruise` | 37.042 | 0.441 | 0.282 |
| `test_re_rand` | 51.364 | 0.752 | 0.405 |
| **avg** | **51.170** | 0.767 | 0.401 |

Notes:
- Sobolev-style auxiliary loss penalizes arc-length finite-difference pressure gradient errors along foil surface nodes.
- `surf_grad_loss = mean_b[MSE(pred_grad_b, gt_grad_b)]` where grad = first-difference along sorted surface points.
- At surf_grad_weight=10.0 the gradient term contributes ~10% of total loss (~0.07 vs ~0.84 total), sufficient to influence the optimizer.
- First trial at weight=1.0 had gradient term <1% of loss and regressed; weight=10.0 correctly scales to match surf_loss magnitude.
- 3/4 val splits improved; all 4 test splits improved. `geom_camber_rc` was the only val regressor (+0.38).
- Improvement vs round-9 baseline: -0.34% on val (59.121 vs 59.321), -1.43% on test (51.170 vs 51.915).
- Metric summary: `models/model-charliepai2f1-askeladd-surf-grad-w10-r9-20260429-190044/metrics.yaml`
- Reproduce: `cd target/ && python train.py --agent charliepai2f1-askeladd --experiment_name "charliepai2f1-askeladd/surf-grad-w10-r9"`

## Previous best (round-9 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **59.321** (epoch 12/12, still descending) | #1244 | n_hidden=192 capacity probe; 3.47M params; VRAM 57 GB |
| `test_avg/mae_surf_p` | **51.915** (4 splits, all finite MAE) | #1244 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (best epoch 12):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 60.116 | 0.660 | 0.388 |
| `val_geom_camber_rc` | 71.231 | 1.267 | 0.586 |
| `val_geom_camber_cruise` | 45.380 | 0.489 | 0.323 |
| `val_re_rand` | 60.556 | 0.887 | 0.450 |
| **avg** | **59.321** | 0.826 | 0.437 |

Per-split test (best epoch 12):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 54.255 | 0.640 | 0.381 |
| `test_geom_camber_rc` | 62.688 | 1.212 | 0.548 |
| `test_geom_camber_cruise` | 38.629 | 0.457 | 0.299 |
| `test_re_rand` | 52.089 | 0.752 | 0.416 |
| **avg** | **51.915** | 0.765 | 0.411 |

Notes:
- n_hidden scaled 160→192 (+20%). Params: 3.47M (+28.5% vs 2.70M). VRAM peak: 57.0 GB (vs 42 GB).
- 30-min budget now yields 12 epochs (vs 13 at n_hidden=160) — model still descending at timeout.
- `geom_camber_rc` slightly regressed vs baseline (+0.840 on val, +1.449 on test), but 3/4 splits improved on both val and test.
- Schedule mismatch: T_max=13 calibrated for 15 epochs but run stops at 12 — cosine still has headroom, recalibrating T_max is a candidate next experiment.
- Improvement vs round-8 baseline: -2.2% on val (59.321 vs 60.685), -1.1% on test (51.915 vs 52.498).
- Metric summary: `models/model-charliepai2f1-edward-n_hidden-192-capacity-probe-20260429-175705/metrics.yaml`
- Reproduce: `cd target/ && python train.py --agent charliepai2f1-edward --experiment_name "charliepai2f1-edward/n_hidden-192-capacity-probe"`

## Previous best (round-8 winner — merged 2026-04-29)

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **60.685** (epoch 13/13, still descending) | #1183 | Cautious AdamW on round-7 EMA-curriculum baseline |
| `test_avg/mae_surf_p` | **52.498** (4 splits, all finite MAE) | #1183 | `test_geom_camber_cruise` vol_loss=inf but MAE valid |

Per-split val (best epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 62.017 | 0.666 | 0.394 |
| `val_geom_camber_rc` | 70.391 | 1.255 | 0.581 |
| `val_geom_camber_cruise` | 49.179 | 0.514 | 0.331 |
| `val_re_rand` | 61.151 | 0.893 | 0.452 |
| **avg** | **60.685** | 0.832 | 0.440 |

Per-split test (best epoch 13):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `test_single_in_dist` | 53.846 | 0.648 | 0.386 |
| `test_geom_camber_rc` | 61.239 | 1.203 | 0.543 |
| `test_geom_camber_cruise` | 40.012 | 0.463 | 0.297 |
| `test_re_rand` | 54.897 | 0.760 | 0.422 |
| **avg** | **52.498** | 0.769 | 0.412 |

Notes:
- `test_geom_camber_cruise` vol_loss=inf/surf_loss=nan is a pre-existing dataset issue (extreme residuals in 1 sample); MAE is valid.
- Best checkpoint is epoch 13 (model still descending under 30-min wall-clock cap).
- Cautious AdamW: subclasses `torch.optim.AdamW`, overrides `step()` to apply `mask = (u * grad > 0)` sign-agreement filter, then rescales by `1/mask.mean()` to preserve update norm. Mask fraction stabilizes at 0.628–0.644 after warmup.
- v4 ran on commit `886685f` (post-#1198 EMA curriculum, pre-#1221 wider FiLMNet). Even without #1221 stacked, val=60.685 < 61.114.
- Improvement vs round-7 baseline: -0.7% on val (60.685 vs 61.114), -0.9% on test (52.498 vs 52.989). All 4 val splits improved.
- Cautious AdamW super-additive stacking history: v1 -3.5%, v2 -10.5%, v4 -19.9% — consistent pattern.
- Metric summary: `models/model-charliepai2f1-edward-cautious-adamw-v4-20260429-163908/metrics.yaml`

## Previous best (round-7 winner — merged 2026-04-29)

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
| Round-7 winner: Wider FiLMNet MLP 512 | 61.114 | 52.989 | #1221 ← merged |
| Round-8 winner: Cautious AdamW | 60.685 | 52.498 | #1183 ← merged |
| Round-9 winner: n_hidden=192 capacity probe | 59.321 | 51.915 | #1244 ← merged |
| Round-10 winner: Sobolev surface gradient loss (w=10) | **59.121** | **51.170** | #1236 ← merged |

Round-1→Round-10 cumulative improvement: **-55.9% on val, -61.3% on test**.

## Default config (`train.py` at HEAD, post-merge of #1236)

| Setting | Value |
|---|---|
| Optimizer | **CautiousAdamW** (sign-agreement mask + 1/mask.mean() rescale), lr=5e-4, weight_decay=1e-4 |
| Scheduler | LinearLR warmup (1 ep, 5e-7 → 5e-4) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Batch size | 4 |
| Surf weight (loss) | 10.0 |
| Surf grad weight | 10.0 (Sobolev surface gradient auxiliary loss) |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` ≈ 15 effective epochs with AMP) |
| Sampler | WeightedRandomSampler (balanced across 3 domains) |
| Loss | MSE on normalized targets, vol + surf_weight·surf; **online EMA per-sample curriculum weighting** (ema_alpha=0.3, temp=0.3, 3-ep warmup) |
| AMP | bfloat16 autocast + GradScaler + clip_grad_norm(max_norm=1.0) |
| Model | Transolver, **n_hidden=192**, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **RFF on (x,z) n_freq=32 sigma=1.0**, **SwiGLU FFN**, **FiLM domain conditioning (wider: 512 hidden)** |
| Params | ~3.47M (+28.5% vs 2.70M at n_hidden=160) |
| VRAM | ~57 GB (vs 42 GB at n_hidden=160) |

Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
Test-time metric for paper: `test_avg/mae_surf_p`.

## Reproduce

```
cd target/ && python train.py --agent <student> --experiment_name "<student>/baseline-default"
```

(All defaults; do NOT pass `--lr`, `--batch_size`, `--surf_weight`, etc.)
