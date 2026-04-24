# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #24 — alphonse: σ × SwiGLU fine sweep (σ=0.7 + SwiGLU winner)**

- **val_avg/mae_surf_p: 69.845** (best single-seed, run `flgrjmte`); 2-seed mean **70.667**
- **test_avg/mae_surf_p: 62.778** (best single-seed); 2-seed mean **62.691**
- W&B runs: `flgrjmte` (seed=1), `j12mrpeb` (seed=0)
- Best epoch: 17 (both seeds)
- **First strict multi-seed merge on this track.**

### Per-split val surface-p MAE (best single-seed `flgrjmte`)

| Split | val mae_surf_p | vs PR #20 |
|-------|----------------|-----------|
| val_single_in_dist | 80.3 | −1.3% |
| val_geom_camber_rc | 82.5 | −10.7% |
| val_geom_camber_cruise | 48.8 | −3.0% |
| val_re_rand | 67.8 | −3.8% |
| **val_avg** | **69.845** | **−5.2%** |

### Per-split test surface-p MAE (best single-seed)

| Split | test mae_surf_p |
|-------|-----------------|
| test_single_in_dist | ≈72 |
| test_geom_camber_rc | ≈73 |
| test_geom_camber_cruise | ≈43 |
| test_re_rand | ≈62 |
| **test_avg** | **62.778** |

### Noise calibration (from PR #24)

- Anchor (σ=1, SwiGLU) 2-seed std (ddof=1): **0.362 val** — ~20× tighter than pre-SwiGLU m=160 band (σ ≈ 8 pts).
- The SwiGLU recipe stabilizes seed variance dramatically. Multi-seed protocol remains mandatory (noise is σ-dependent: σ=0.8/0.9 show std 3-4 val, suggesting optimization pathology in that band).

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
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| **fourier_sigma** | **0.7** ← new (PR #24) |
| ffn | SwiGLU |

**Note on code defaults:** some Config dataclass defaults in `train.py` still reflect pre-merge values (`loss_type="mse"`, `amp=False`, `grad_accum=1`, `fourier_features="none"`, `fourier_m=10`, `fourier_sigma=1.0`, `swiglu=False`). The current merged recipe requires explicit flags — always pass the full flag list below. This footgun has hit 8 consecutive students; verification via `--debug` run + W&B config inspection is now mandatory before committing to full sweeps.

Reproduce:
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
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

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
