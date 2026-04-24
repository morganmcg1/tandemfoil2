# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #27 — nezuko: slice_num=32 sweep on σ=0.7 + SwiGLU recipe**

- **val_avg/mae_surf_p: 67.186** (best single-seed, run `nrba5yg8`, seed=2); 3-seed mean **68.687**
- **test_avg/mae_surf_p: 58.358** (best single-seed, seed=2); 3-seed mean **60.680**
- W&B runs: `szq21j7r` (s=0), `cmmj8l21` (s=1), `nrba5yg8` (s=2) — group `nezuko/slice-num-sigma07`
- Best epoch: 18 (s=0), 19 (s=1), 20 (s=2)

### Per-split val surface-p MAE (best single-seed `nrba5yg8`, s=2)

| Split | val mae_surf_p | vs PR #24 |
|-------|----------------|-----------|
| val_single_in_dist | 77.43 | −3.6% |
| val_geom_camber_rc | 76.45 | −7.3% |
| val_geom_camber_cruise | 49.72 | +1.9% |
| val_re_rand | 65.14 | −4.8% |
| **val_avg** | **67.186** | **−3.8%** |

### Per-split test surface-p MAE (3-seed mean, best-val checkpoint)

| Split | test mae_surf_p (3-seed mean) | vs PR #24 |
|-------|-------------------------------|-----------|
| test_single_in_dist | 70.14 | −5.7% |
| test_geom_camber_rc | 70.97 | −2.6% |
| test_geom_camber_cruise | 42.89 | −0.4% |
| test_re_rand | 58.72 | −3.1% |
| **test_avg** | **60.680** | **−3.2%** |

### Noise calibration (from PR #27)

- Anchor (sn=64, σ=0.7, SwiGLU) 2-seed std (ddof=1): **1.162 val** (wider than PR #24's 0.362 — recipe-dependent).
- Winner (sn=32) 3-seed std: **1.650 val**. Multi-seed (≥3) protocol important for sn=32.
- 2-seed merge criterion: winner 2-seed mean ≤ **69.345 val** (68.687 − 1.162 × 3-seed std caveat: use anchor std).
  - All three 2-seed subsets of sn=32 pass (min: 67.803, max: 69.437).

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
| **slice_num** | **32** ← new (PR #27) |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 0.7 |
| ffn | SwiGLU |

**Note on code defaults:** some Config dataclass defaults in `train.py` still reflect pre-merge values (`loss_type="mse"`, `amp=False`, `grad_accum=1`, `fourier_features="none"`, `fourier_m=10`, `fourier_sigma=1.0`, `swiglu=False`). The current merged recipe requires explicit flags — always pass the full flag list below. Verification via `--debug` run + W&B config inspection is **mandatory** before committing to full sweeps.

**Note on slice_num:** sn=32 gives −16% VRAM vs sn=64 (31.6 GB vs 37.8 GB) and ~20% faster per epoch (~89s vs ~111s), enabling 21 epochs in the 30-min budget vs 17 for sn=64. The per-epoch speed gain is a real throughput advantage.

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
    --slice_num 32 \
    --seed 2 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

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
