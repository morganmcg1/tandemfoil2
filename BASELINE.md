# Baseline Tracker — TandemFoilSet CFD Surrogate

Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## 2026-04-29 10:50 — PR #1088: Increase surf_weight from 10 to 25 for surface MAE focus

- **Student**: charliepai2f2-edward
- **Branch**: charliepai2f2-edward/surf-weight-sweep-25
- **Change**: `surf_weight: 10.0 → 25.0` (single line change in Config dataclass)

### Best Validation Metrics (epoch 13/50, 30-min timeout)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** (PRIMARY) | **127.6661** |
| val_avg/mae_vol_p | 139.9394 |
| val_avg/mae_surf_Ux | 2.2548 |
| val_avg/mae_surf_Uy | 0.9431 |
| val_avg/mae_vol_Ux | 5.8663 |
| val_avg/mae_vol_Uy | 2.6935 |

Per-split surface pressure MAE:

| Split | mae_surf_p | mae_vol_p |
|-------|------------|-----------|
| val_single_in_dist | 157.82 | 178.70 |
| val_geom_camber_rc | 135.65 | 146.43 |
| val_geom_camber_cruise | 99.26 | 112.71 |
| val_re_rand | 117.94 | 121.91 |

### Test Metrics (3 of 4 splits clean; test_geom_camber_cruise NaN due to upstream corrupted GT sample)

| Split | mae_surf_p |
|-------|------------|
| test_single_in_dist | 137.04 |
| test_geom_camber_rc | 122.18 |
| test_geom_camber_cruise | NaN (upstream data bug) |
| test_re_rand | 117.39 |
| 3-split avg | 125.54 |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~131s/epoch)
- Peak GPU memory: 42.12 GB
- Metrics JSONL: `target/models/model-charliepai2f2-edward-surf-weight-25-20260429-095003/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=5e-4, wd=1e-4, batch=4, CosineAnnealingLR(T_max=50)

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-edward --experiment_name "charliepai2f2-edward/surf-weight-25"
# with surf_weight=25.0 set in Config dataclass
```

---

## 2026-04-29 12:20 — PR #1091: Add stochastic depth (drop_path 0→0.1) for OOD generalization

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/stochastic-depth-regularization
- **Change**: DropPath linear schedule (0.0→0.1 across 5 TransolverBlocks) + budget-aware CosineAnnealingLR (T_max estimated from warm-up timing, eta_min=1e-6) + surf_weight=25.0 + NaN-safe eval workaround

### Best Validation Metrics (epoch 13/50, 30-min timeout)

| Metric | Value | vs prior baseline |
|--------|-------|-------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **121.89** | **-5.78 (-4.5%)** |
| val_avg/mae_surf_Ux | 1.97 | — |
| val_avg/mae_surf_Uy | 0.89 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 154.41 | 130.06 |
| geom_camber_rc | 128.38 | 118.20 |
| geom_camber_cruise | 95.98 | 79.45 |
| re_rand | 108.78 | 110.64 |
| **avg** | **121.89** | **109.59** |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~135s/epoch); best at epoch 13
- Budget-aware cosine schedule: T_max=11 estimated after 2 warm-up epochs (1529s remaining / 135.4s per epoch), eta_min=1e-6
- Peak GPU memory: 42.1 GB (single H100, batch=4)
- Params: 0.66 M (DropPath adds no parameters)
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-sd-cosine-budget-aware-20260429-114039/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=5e-4, wd=1e-4, batch=4, surf_weight=25.0, DropPath(0.0→0.1 linear), budget-aware CosineAnnealingLR

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-nezuko --experiment_name "charliepai2f2-nezuko/sd-cosine-budget-aware"
# with DropPath linear schedule 0.0→0.1, budget-aware CosineAnnealingLR, surf_weight=25.0
```

---

## 2026-04-29 13:10 — PR #1098: Grad clip + higher LR (1e-3) for stable fast convergence

- **Student**: charliepai2f2-tanjiro
- **Branch**: charliepai2f2-tanjiro/grad-clip-higher-lr-rebased
- **Change**: lr 5e-4→1e-3 + grad_clip=1.0 (on top of PR #1091 baseline: DropPath 0→0.1 + budget-aware cosine + surf_weight=25 + NaN-safe eval)

### Best Validation Metrics (epoch 14/50, 30-min timeout)

| Metric | Value | vs prior baseline |
|--------|-------|-------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **100.41** | **-21.48 (-17.6%)** |
| val_avg/mae_vol_p | 120.81 | — |
| val_avg/mae_surf_Ux | 1.4978 | — |
| val_avg/mae_surf_Uy | 0.7422 | — |
| val_avg/mae_vol_Ux | 4.9283 | — |
| val_avg/mae_vol_Uy | 2.2782 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 120.68 | 104.32 |
| geom_camber_rc | 111.80 | 98.04 |
| geom_camber_cruise | 75.99 | 63.06 |
| re_rand | 93.15 | 88.91 |
| **avg** | **100.41** | **88.58** |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30); best at epoch 14
- Budget-aware cosine schedule: T_max=11 after 2-epoch warmup, eta_min=1e-6 (inherited from PR #1091)
- Peak GPU memory: 42.11 GB (single H100, batch=4)
- Params: 662,359
- Metrics JSONL: `target/models/model-charliepai2f2-tanjiro-grad-clip-higher-lr-rebased-20260429-123256/metrics.jsonl`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (all baseline)
- Training: lr=1e-3, wd=1e-4, batch=4, grad_clip=1.0, surf_weight=25.0, DropPath(0.0→0.1 linear), budget-aware CosineAnnealingLR

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-tanjiro --experiment_name "charliepai2f2-tanjiro/grad-clip-higher-lr-rebased" --grad_clip 1.0
# lr=1e-3 is the new default in Config; --grad_clip 1.0 required
```

---

## 2026-04-29 15:15 — PR #1184: BF16 AMP on current best stack: more epochs within 30-min budget

- **Student**: charliepai2f2-askeladd
- **Branch**: charliepai2f2-askeladd/bf16-amp-on-current-stack
- **Change**: BF16 mixed-precision (`torch.autocast` + `dtype=torch.bfloat16`) on top of PR #1098 best config (lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware CosineAnnealingLR + surf_weight=25). No GradScaler. 27% per-epoch speedup: 135s/ep → 98.5s/ep, enabling 19 epochs within the 30-min budget vs 14.

### Best Validation Metrics (epoch 19/50, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1098) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **89.00** | **-11.41 (-11.4%)** |
| val_avg/mae_vol_p | ~103.4 (per-split avg) | — |
| val_avg/mae_surf_Ux | ~1.276 | — |
| val_avg/mae_surf_Uy | ~0.649 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 105.41 | 89.40 |
| geom_camber_rc | 101.20 | 89.89 |
| geom_camber_cruise | 65.37 | 54.72 |
| re_rand | 84.04 | 76.37 |
| **avg** | **89.00** | **77.59** |

### Context
- 19 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~98.5s/epoch); best at epoch 19
- BF16 autocast in train + eval forward passes; loss computed in FP32 (.float() cast after forward)
- Budget-aware CosineAnnealingLR re-estimated T_max=16 from BF16 epoch time → cosine fully anneals to eta_min=1e-6 by epoch 19
- Peak GPU memory: 32.95 GB (RTX PRO 6000 Blackwell, 96 GB total)
- Params: 662,359
- Metrics JSONL (run 1): `target/models/model-charliepai2f2-askeladd-bf16-amp-on-current-stack-20260429-134807/metrics.jsonl`
- Metrics YAML (run 1): `target/models/model-charliepai2f2-askeladd-bf16-amp-on-current-stack-20260429-134807/metrics.yaml`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-askeladd \
  --experiment_name "charliepai2f2-askeladd/bf16-amp-on-current-stack" \
  --grad_clip 1.0
# BF16 autocast in train.py; lr=1e-3, surf_weight=25.0, DropPath(0→0.1), budget-aware cosine, NaN guard
```

---

## 2026-04-29 15:35 — PR #1195: OneCycleLR superconvergence: replace cosine anneal within 14-epoch budget

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/onecyclelr-superconvergence
- **Change**: Replace budget-aware CosineAnnealingLR with `OneCycleLR(max_lr=1.2e-3, pct_start=0.3, total_steps=5625, div_factor=25, final_div_factor=1e4, anneal_strategy='cos', cycle_momentum=False)` stepping per-batch. All other settings unchanged from PR #1098 stack (lr=1e-3 base, grad_clip=1.0, DropPath 0→0.1, surf_weight=25, batch=4).

### Best Validation Metrics (epoch 14/50, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1184) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **97.0209** | +8.02 (this PR built on PR #1098 stack, not BF16) |
| val_avg/mae_vol_p | 110.5413 | — |
| val_avg/mae_surf_Ux | 1.4259 | — |
| val_avg/mae_surf_Uy | 0.6977 | — |
| val_avg/mae_vol_Ux | 4.4613 | — |
| val_avg/mae_vol_Uy | 2.1546 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 114.4687 | 98.7505 |
| geom_camber_rc | 107.1299 | 95.8526 |
| geom_camber_cruise | 73.8442 | 62.4759 |
| re_rand | 92.6409 | 85.8037 |
| **avg** | **97.0209** | **85.7207** |

### Context
- 14 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~135s/epoch); best at epoch 14
- OneCycleLR peak at epoch ~4.5 (step 1687 of 5625), deep anneal to lr=2.65e-05 by epoch 14
- Peak GPU memory: 42.12 GB
- Params: 662,359
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-onecyclelr-superconvergence-20260429-145319/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-nezuko-onecyclelr-superconvergence-20260429-145319/metrics.yaml`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2

### Reproduce
```bash
cd target/ && python train.py --agent charliepai2f2-nezuko \
  --experiment_name "charliepai2f2-nezuko/onecyclelr-superconvergence" \
  --grad_clip 1.0
# OneCycleLR stepping per-batch; lr=1e-3, surf_weight=25.0, DropPath(0→0.1)
```

---

## Notes on NaN in test_geom_camber_cruise

One corrupted GT sample (`000020.pt`) in `.test_geom_camber_cruise_gt/` has NaN in the pressure channel. `data/scoring.py:accumulate_batch` propagates this NaN because `NaN * 0.0 = NaN` in IEEE float — the mask does not fully guard it. Since `data/scoring.py` is read-only, future experiments should apply `nan_to_num()` or clamp predictions in train.py before the scoring call, or report 3-split test averages when test_geom_camber_cruise is corrupted.

---

## 2026-04-29 16:23 — PR #1211: BF16 AMP + OneCycleLR combined on current best stack

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/bf16-onecycle-combined
- **Change**: Combine BF16 AMP (from PR #1184) with OneCycleLR (from PR #1195) on the same stack. BF16 gives ~26% faster epochs (98–100s vs ~135s), enabling 19 epochs in 30 min. OneCycleLR sized for 19 epochs: `max_lr=1.2e-3, pct_start=0.3, total_steps=7125, div_factor=25, final_div_factor=1e4`.

### Best Validation Metrics (epoch 19/19, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1184) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **80.53** | **-8.47 (-9.5%)** |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| single_in_dist     | 91.33 | 80.43 |
| geom_camber_rc     | 91.01 | 83.65 |
| geom_camber_cruise | 60.42 | 50.04 |
| re_rand            | 79.36 | 70.80 |
| **avg**            | **80.53** | **71.23** |

### Context
- 19 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~99s/epoch); best at epoch 19/19 (final)
- OneCycleLR peak at epoch ~5.7, fully annealed to lr≈4.9e-9 by epoch 19
- Peak GPU memory: 32.96 GB (RTX PRO 6000 Blackwell, 96 GB total)
- Params: 662,359
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-bf16-onecycle-combined-20260429-154838/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-nezuko-bf16-onecycle-combined-20260429-154838/metrics.yaml`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-nezuko \
  --experiment_name "charliepai2f2-nezuko/bf16-onecycle-combined" \
  --grad_clip 1.0
# BF16 autocast; OneCycleLR max_lr=1.2e-3, pct_start=0.3; surf_weight=25.0, DropPath(0→0.1)
```

---

## 2026-04-29 19:31 — PR #1264: Lightweight FiLM Re conditioning: shared generator on BF16 stack

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/lightweight-film-re-conditioning
- **Change**: Add `SharedFiLMGenerator` MLP (1→128→1280) producing (γ, β) pairs for all 5 TransolverBlocks. FiLM applied post-norm on each block's hidden state: `fx = fx * (1 + γ) + β`. Input: `log_re_sample = x_input[:, 0, 13].unsqueeze(-1)` (standardized log(Re)). All other settings unchanged from PR #1211 baseline (BF16 AMP + OneCycleLR + grad_clip=1.0 + DropPath 0→0.1 + surf_weight=25).

### Best Validation Metrics (epoch 18/19, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1211) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **74.36** | **-6.17 (-7.7%)** |
| val_avg/mae_surf_Ux | 1.0708 | — |
| val_avg/mae_surf_Uy | 0.5392 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p | val baseline | test baseline |
|-------|---------------:|----------------:|-------------:|--------------:|
| single_in_dist     | **82.09** | **71.62** | 91.33 | 80.43 |
| geom_camber_rc     | **85.31** | **79.90** | 91.01 | 83.65 |
| geom_camber_cruise | **55.67** | **46.22** | 60.42 | 50.04 |
| re_rand            | **74.38** | **65.89** | 79.36 | 70.80 |
| **avg**            | **74.36** | **65.91** | **80.53** | **71.23** |

### Context
- 18 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~105s/epoch); best at epoch 18/18 (final)
- ~6.6% per-epoch slowdown from FiLM forward overhead (105s vs 98.5s); fits 18 instead of 19 epochs
- OneCycleLR annealed to lr=1.66e-05 by epoch 18 — effectively converged within budget
- Peak GPU memory: 35.44 GB (vs baseline 32.95 GB; +2.5 GB from FiLM activations)
- **Params: 827,735 (+165K vs 662,359 baseline)** — `SharedFiLMGenerator`: `Linear(1,128)` + `Linear(128,1280)` ≈ 256 + 165,120
- Branch commit: 98a3053
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-lightweight-film-re-conditioning-20260429-185626/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-nezuko-lightweight-film-re-conditioning-20260429-185626/metrics.yaml`
- Model config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (unchanged)

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-nezuko \
  --experiment_name "charliepai2f2-nezuko/lightweight-film-re-conditioning" \
  --grad_clip 1.0
# SharedFiLMGenerator post-norm FiLM; BF16 AMP; OneCycleLR max_lr=1.2e-3; surf_weight=25.0; DropPath(0→0.1)
```

---

## 2026-04-29 21:15 — PR #1300: Fourier positional features: replace raw (x,z) with sinusoidal encodings

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/fourier-pos-features
- **Change**: Add `FourierPosEncoder` (8 octaves → 32 features per point) replacing raw `(x,z)` coordinates in `Transolver.preprocess` MLP. Preprocess MLP input grows from `fun_dim+2` to `fun_dim+32`. FiLM Re-conditioning, BF16 AMP, OneCycleLR, DropPath, grad_clip=1.0, surf_weight=25 all preserved from PR #1264 baseline.

### Best Validation Metrics (epoch 18/18, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1264) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **71.80** | **-2.56 (-3.4%)** |
| **test_avg/mae_surf_p** | **62.59** | **-3.32 (-5.0%)** |
| val_avg/mae_surf_Ux | 1.0198 | — |
| val_avg/mae_surf_Uy | 0.5444 | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p | val baseline | test baseline |
|-------|---------------:|----------------:|-------------:|--------------:|
| single_in_dist     | **72.78** | **63.89** | 82.09 | 71.62 |
| geom_camber_rc     | **84.26** | **74.44** | 85.31 | 79.90 |
| geom_camber_cruise | 56.03 | 47.11 | 55.67 | 46.22 |
| re_rand            | **74.14** | **64.91** | 74.38 | 65.89 |
| **avg**            | **71.80** | **62.59** | **74.36** | **65.91** |

### Context
- 18 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~105s/epoch); best at epoch 18/18 (final)
- BF16 AMP + OneCycleLR (max_lr=1.2e-3, pct_start=0.3) + DropPath(0→0.1) + grad_clip=1.0 + surf_weight=25 + FiLM Re-conditioning — full PR #1264 stack
- Peak GPU memory: 35.83 GB (+0.4 GB vs baseline)
- **Params: 835,415 (+7,680 vs 827,735 baseline)** — preprocess MLP first linear grew from `(fun_dim+2)*256=6144` to `(fun_dim+32)*256=13824`; `FourierPosEncoder` is parameter-free (buffers only)
- Branch commit: 4499683
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-fourier-pos-features-20260429-203437/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-nezuko-fourier-pos-features-20260429-203437/metrics.yaml`

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-nezuko \
  --experiment_name "charliepai2f2-nezuko/fourier-pos-features" \
  --grad_clip 1.0
# FourierPosEncoder(n_octaves=8) replacing raw (x,z); FiLM Re-conditioning; BF16 AMP; OneCycleLR max_lr=1.2e-3; surf_weight=25.0; DropPath(0→0.1)
```

---

## 2026-04-29 23:10 — PR #1314: Fourier pos enc: normalize coords + concat raw (x,z) for aliasing fix

- **Student**: charliepai2f2-nezuko
- **Branch**: charliepai2f2-nezuko/fourier-coord-norm-concat
- **Change**: Two correctness fixes to `FourierPosEncoder`: (1) normalize input coords to `[-1,1]` via `coord_scale=3.0` before frequency encoding, eliminating aliasing in upper octaves; (2) concatenate raw `(x,z)` alongside 32-D Fourier features (NeRF recipe) to restore the DC channel. Preprocess MLP input grows from 54-D to 56-D (`2 raw + 32 Fourier + 22 other`). Full PR #1300 stack preserved unchanged.

### Best Validation Metrics (epoch 18, 30-min timeout)

| Metric | Value | vs prior baseline (PR #1300) |
|--------|-------|------------------------------|
| **val_avg/mae_surf_p** (PRIMARY) | **69.07** | **-2.73 (-3.8%)** |
| **test_avg/mae_surf_p** | **59.63** | **-2.96 (-4.7%)** |
| val_avg/mae_surf_Ux | 0.911 (single_in_dist) | — |
| val_avg/mae_surf_Uy | 0.488 (single_in_dist) | — |

Per-split surface pressure MAE:

| Split | val mae_surf_p | test mae_surf_p | val baseline | test baseline |
|-------|---------------:|----------------:|-------------:|--------------:|
| single_in_dist     | 72.49 | 62.58 | 72.78 | 63.89 |
| geom_camber_rc     | 83.24 | 72.60 | 84.26 | 74.44 |
| geom_camber_cruise | **51.69** | **42.11** | 56.03 | 47.11 |
| re_rand            | **68.87** | 61.24 | 74.14 | 64.91 |
| **avg**            | **69.07** | **59.63** | **71.80** | **62.59** |

### Context
- 18 epochs / 50 configured (hit SENPAI_TIMEOUT_MINUTES=30 at ~105s/epoch); best at epoch 18/18 (final, still descending)
- Largest gains on OOD geometry (`geom_camber_cruise` -7.7% val, -10.6% test) and OOD Re (`re_rand` -7.1% val, -5.7% test) — consistent with the aliasing-fix hypothesis
- Peak GPU memory: ~35.8 GB (essentially identical to PR #1300; only +2 input dims to first preprocess linear)
- **Params: 836,183 (+768 vs 835,415 baseline)** — wider preprocess Linear(56→256) vs Linear(54→256)
- `coord_scale=3.0` is a hard-coded constant; no new CLI flag needed
- Metrics JSONL: `target/models/model-charliepai2f2-nezuko-fourier-coord-norm-concat-20260429-220329/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-nezuko-fourier-coord-norm-concat-20260429-220329/metrics.yaml`

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-nezuko \
  --experiment_name "charliepai2f2-nezuko/fourier-coord-norm-concat" \
  --grad_clip 1.0
# FourierPosEncoder(n_octaves=8, coord_scale=3.0); raw (x,z) CONCATENATED with 32-D Fourier → 56-D preprocess input
# Full PR #1264/PR #1300 stack: FiLM Re-conditioning, BF16 AMP, OneCycleLR(max_lr=1.2e-3, pct_start=0.3), DropPath(0→0.1), surf_weight=25
```

---

## Baseline #13 — PR #1341: torch.compile + fix OneCycleLR schedule for compiled throughput

**Date:** 2026-04-29
**Branch:** `charliepai2f2-edward/torch-compile-schedule-fix`
**Primary metric:** `val_avg/mae_surf_p = 54.23` | `test_avg/mae_surf_p = 46.13`

### Key changes
- `torch.compile(model, mode='reduce-overhead')` enabled — ~50% per-epoch speedup on Blackwell GPU (PyTorch 2.10 / Triton 3.6)
- `ONECYCLE_PER_EPOCH_SEC_ESTIMATE` corrected from 100s → **55.0s** to match actual compiled throughput
- This fix allows OneCycleLR to see correct `total_steps = epochs × steps_per_epoch` where epochs ≈ 33 (vs 18 before)
- ~83% more training epochs in the same 30-min wall-clock budget → significantly better convergence
- **Peak GPU memory: 23.93 GB** (-33% vs 35.8 GB baseline) — compile fuses ops and reduces activation memory
- Params: 835,927

### Results

| Split | val mae_surf_p | test mae_surf_p | val prev best | test prev best |
|-------|---------------:|----------------:|--------------:|---------------:|
| single_in_dist     | 57.17 | 49.17 | 72.49 | 62.58 |
| geom_camber_rc     | 68.23 | 59.05 | 83.24 | 72.60 |
| geom_camber_cruise | **36.40** | **30.11** | 51.69 | 42.11 |
| re_rand            | **55.12** | **46.18** | 68.87 | 61.24 |
| **avg**            | **54.23** | **46.13** | **69.07** | **59.63** |

Improvement vs prior best: **-21.5% val, -22.6% test**

### Context
- 33 epochs in ~30 min (steady-state ~50–58 s/epoch compiled vs ~105 s/epoch eager)
- Gains across all splits; largest absolute gains on OOD geometry (`geom_camber_cruise`) and OOD Re (`re_rand`)
- Full stack: FourierPosEncoder(n_octaves=8, coord_scale=3.0) + raw (x,z) concat (56-D) + SharedFiLMGenerator(1→128→1280) + BF16 AMP + OneCycleLR(max_lr=1.2e-3, pct_start=0.3, total_steps=7125) + DropPath(0→0.1) + grad_clip=1.0 + surf_weight=25
- Metrics JSONL: `target/models/model-charliepai2f2-edward-torch-compile-schedule-fix-20260429-233514/metrics.jsonl`
- Metrics YAML: `target/models/model-charliepai2f2-edward-torch-compile-schedule-fix-20260429-233514/metrics.yaml`

### Reproduce
```bash
cd target/ && python train.py \
  --agent charliepai2f2-edward \
  --experiment_name "charliepai2f2-edward/torch-compile-schedule-fix" \
  --grad_clip 1.0
# torch.compile(model, mode='reduce-overhead') + ONECYCLE_PER_EPOCH_SEC_ESTIMATE=55.0
# Full stack: FourierPosEncoder(8-octave, coord_scale=3.0) + FiLM Re-conditioning + BF16 AMP
# OneCycleLR(max_lr=1.2e-3, pct_start=0.3) + DropPath(0→0.1) + surf_weight=25
```
