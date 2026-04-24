# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #34 — frieren: slice_num=16 sweep (sn∈{8,16,24} vs sn=32 anchor)**

- **val_avg/mae_surf_p: 60.581** (best single-seed, run `moydqx8l`, seed=0, ep=23); 2-seed mean **61.813**
- **test_avg/mae_surf_p: 54.640** (best single-seed, seed=0); 2-seed mean **55.316**
- W&B runs: `moydqx8l` (s=0), `tkfnon33` (s=1) — group `frieren/slice-num-lower`
- Best epoch: 23 (both seeds hit best val at final epoch — headroom remains)

### Per-split val surface-p MAE (best single-seed `moydqx8l`, s=0, ep=23)

| Split | val mae_surf_p | vs PR #27 (sn=32) |
|-------|----------------|-------------------|
| val_single_in_dist | 71.23 | −8.0% |
| val_geom_camber_rc | 69.87 | −8.6% |
| val_geom_camber_cruise | 40.81 | −17.9% |
| val_re_rand | 60.41 | −7.3% |
| **val_avg** | **60.581** | **−9.8%** |

### Per-split test surface-p MAE (best single-seed `moydqx8l`, s=0, best-val checkpoint)

| Split | test mae_surf_p | vs PR #27 3-seed mean |
|-------|-----------------|----------------------|
| test_single_in_dist | 64.43 | −8.1% |
| test_geom_camber_rc | 64.58 | −9.0% |
| test_geom_camber_cruise | 35.41 | −17.4% |
| test_re_rand | 54.14 | −7.8% |
| **test_avg** | **54.640** | **−10.0%** |

### Noise calibration (from PR #34)

- Anchor (sn=32, σ=0.7, SwiGLU) 3-seed std (ddof=1): **1.650 val** (reproduces PR #27 baseline exactly).
- Winner (sn=16) 2-seed std: **1.742 val** (comparable to anchor std — win is structural, not a lucky snapshot).
- 2-seed merge criterion: winner 2-seed mean ≤ **67.025 val** (68.687 − 1× anchor std 1.650).
  - sn=16 2-seed mean **61.813** passes (5.21 pts below gate — decisive).
- sn=8 single-seed: 62.476 val / 54.677 test; trailing-5 mean **64.05** (lowest of any run). Follow-up needed with 2 seeds.
- sn=24 is UNSTABLE: 2-seed std 6.066 (3.7× anchor std). Not a candidate.

### Current default config (post-merge)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| grad_accum | 4 (effective bs=16) |
| amp | True (bf16 autocast) |
| surf_weight | 1.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| **slice_num** | **16** ← new (PR #34) |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 0.7 |
| ffn | SwiGLU |

**Note on code defaults:** some Config dataclass defaults in `train.py` still reflect pre-merge values (`loss_type="mse"`, `amp=False`, `grad_accum=1`, `fourier_features="none"`, `fourier_m=10`, `fourier_sigma=1.0`, `swiglu=False`). The current merged recipe requires explicit flags. Verification via `--debug` run + W&B config inspection is **mandatory** before committing to full sweeps.

**Note on slice_num:** sn=16 gives 28.5 GB VRAM (vs sn=32's 31.6 GB) and ~78 s/epoch (vs ~89 s), enabling ~23 epochs in 30-min budget. Both sn=16 seeds hit best val at the final epoch — more epochs would likely improve further. The floor is not yet found; sn=8 (trailing-5 64.05) has the lowest loss surface of any run tested.

Reproduce (best single-seed winner):
```bash
cd target && python train.py \
    --agent <student> \
    --loss_type l1 \
    --surf_weight 1 \
    --amp true \
    --grad_accum 4 \
    --batch_size 4 \
    --fourier_features fixed \
    --fourier_m 160 \
    --fourier_sigma 0.7 \
    --swiglu \
    --slice_num 16 \
    --seed 0 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-24 — PR #34: frieren slice_num=16 sweep (sn∈{8,16,24} vs sn=32 anchor)

- **val_avg/mae_surf_p: 60.581 (best seed, s=0) / 61.813 (2-seed mean)** (previous: 67.186 / 68.687, PR #27)
- **test_avg/mae_surf_p: 54.640 (best seed) / 55.316 (2-seed mean)** (previous: 58.358 / 60.680, PR #27)
- W&B runs: `moydqx8l` (s=0), `tkfnon33` (s=1) — group `frieren/slice-num-lower`
- Change: `--slice_num 16` (was 32). σ=0.7, SwiGLU, AMP, grad_accum=4, L1, sw=1, Fourier-m=160 unchanged.
- Delta: −9.8% val / −6.5% test (best seed vs prior best seed); 2-seed mean −10.0% val / −8.8% test.
- Decisive pass: 2-seed mean 61.813 is 5.21 pts below merge gate (67.025). sn=16 2-seed std (1.742) matches anchor std (1.650) — structural win, not a lucky snapshot.
- Gain uniform across all splits; largest on `val_geom_camber_cruise` (−17.9%) — consistent with stronger pooling regularization hypothesis.
- VRAM: 28.5 GB (−9.8% vs sn=32); per-epoch: ~78s (−12%); budget: ~23 epochs vs 21. Both seeds still improving at final epoch.
- sn=8 (trailing-5=64.05) is intriguing — floor not yet found. Follow-up needed.

### 2026-04-24 — PR #27: nezuko slice_num=32 sweep (σ=0.7 + SwiGLU + sn=32)

- **val_avg/mae_surf_p: 67.186 (best seed, s=2) / 68.687 (3-seed mean)** (previous: 69.845 / 70.667, PR #24)
- **test_avg/mae_surf_p: 58.358 (best seed) / 60.680 (3-seed mean)** (previous: 62.778 / 62.691, PR #24)
- W&B runs: `szq21j7r` (s=0), `cmmj8l21` (s=1), `nrba5yg8` (s=2) — group `nezuko/slice-num-sigma07`
- Change: `--slice_num 32` (was 64). σ=0.7, SwiGLU, AMP, grad_accum=4, L1, sw=1, Fourier-m=160 unchanged.
- Delta: −3.8% val / −3.2% test (best seed vs prior); 3-seed mean −2.7% val / −3.2% test.
- All three 2-seed sub-selections of sn=32 pass merge criterion vs anchor 2-seed mean (70.667 − 1.162 std).
- Trailing-5 epoch mean also favors sn=32 by ~8 val pts — win is structural, not just a snapshot artifact.
- Gain concentrated on `val_geom_camber_rc` (−7.3%), `val_re_rand` (−4.8%), `val_single_in_dist` (−3.6%); slight regression on `cruise` (+1.9%).
- VRAM: 31.6 GB vs anchor 37.8 GB (−16%); per-epoch: ~89s vs ~111s (−20%); budget: 21 epochs vs 17.

### 2026-04-24 — PR #24: alphonse σ × SwiGLU sweep (σ=0.7 + SwiGLU)

- **val_avg/mae_surf_p: 69.845 (best seed) / 70.667 (2-seed mean)** (previous: 73.660, PR #20)
- **test_avg/mae_surf_p: 62.778 (best seed) / 62.691 (2-seed mean)** (previous: 63.983, PR #20)
- W&B runs: `flgrjmte` (s=1), `j12mrpeb` (s=0) — both in `alphonse/sigma-swiglu`
- Change: `--fourier_sigma 0.7` (was 1.0 in PR #20). SwiGLU, AMP, grad_accum=4, L1, sw=1, Fourier-m=160 all unchanged.
- Delta: −5.2% val / −2.0% test (best seed vs prior).
- **First merge under strict 2-seed multi-seed protocol.** 2-seed mean 70.67 vs 2-seed anchor mean 73.92 = 3.25 pts gap, ~9× anchor std (0.362). Decisively outside noise.
- Independently verified fern's crashed σ=0.7 compound claim from PR #20: seed=0 reproduced 71.489 bit-exactly.
- **σ landscape is SHARP at 0.7, not a flat basin.** σ=0.8 (79.14) and σ=0.9 (77.99) regress to far worse than σ=1.0. Follow-up needed: sweep {0.5, 0.55, 0.6, 0.65, 0.75} to find true minimum.

### 2026-04-24 — PR #20: fern Fourier σ=1 + SwiGLU feedforward

- **val_avg/mae_surf_p: 73.660** (previous: 84.737, PR #7); test: 63.983.
- W&B run: `eg6i88yf`. Change: SwiGLU FFN replaces GELU-MLP in every TransolverBlock.
- Delta: −13.1 % val / −15.0 % test. Huge architectural jump.

### 2026-04-23 — PR #7: alphonse Fourier PE fixed σ=1 m=160

- **val_avg/mae_surf_p: 84.737** (previous: 88.268, PR #12); test: 75.244.
- W&B run: `91z1948k`. Change: Random Fourier Features on (x,z) coords.

### 2026-04-23 — PR #12: fern AMP + grad_accum=4

- **val_avg/mae_surf_p: 88.268** (previous: 93.127, PR #11); test: 79.733.
- W&B run: `n68w9q7o`. Change: bf16 autocast + grad_accum=4. +5 epochs per 30-min budget.

### 2026-04-23 — PR #11: frieren surf_weight=1 on L1

- **val_avg/mae_surf_p: 93.127** (previous: 103.036, PR #3).
- W&B run: `yt7eup38`. Change: sw=10 → sw=1 under L1.

### 2026-04-23 — PR #3: frieren L1 loss

- **val_avg/mae_surf_p: 103.036** (previous: no baseline).
- W&B run: `w2jsabii`. Change: MSE → L1.

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four val splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint on four test splits. Scoring patch (commit 7d71abd) excludes `test_geom_camber_cruise/000020.pt`'s +Inf sample.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `kagent_v_students`.
2. Update this file with new metric, PR number, W&B run link.
3. Commit on advisor branch.

**Multi-seed requirement (from round 9):** merge claims < 5% require 2-seed anchors. Winner 2-seed mean must beat current 2-seed anchor mean by > 1σ of anchor spread.
