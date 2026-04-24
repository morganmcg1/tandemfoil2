# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #35 — nezuko: n_layers=3 sweep (sn=32 recipe, nl ∈ {3,4,5,6,7})**

- **val_avg/mae_surf_p: 54.210** (best single-seed, run `jv54vc7o`, seed=1, ep=32); 2-seed mean **54.476**
- **test_avg/mae_surf_p: 47.484** (best single-seed, seed=1); 2-seed mean **47.336**
- W&B runs: `rze3vuj0` (s=0), `jv54vc7o` (s=1) — group `nezuko/n-layers-sn32`
- Best epoch: 32 (both seeds hit best val at final epoch; nl=3 trains for 32 epochs in budget vs 14-18 at nl=7)

### Per-split val surface-p MAE (best single-seed `jv54vc7o`, s=1, ep=32)

| Split | val mae_surf_p | vs PR #34 (nl=5/sn=16) |
|-------|----------------|-------------------------|
| val_single_in_dist | 61.24 | −14.0% |
| val_geom_camber_rc | 67.65 | −3.2% |
| val_geom_camber_cruise | 33.63 | −17.6% |
| val_re_rand | 54.33 | −10.1% |
| **val_avg** | **54.210** | **−10.5%** |

### Per-split test surface-p MAE (best single-seed `jv54vc7o`, best-val checkpoint)

| Split | test mae_surf_p | vs PR #34 |
|-------|-----------------|-----------|
| test_single_in_dist | ~55 | ≈−15% |
| test_geom_camber_rc | ~57 | ≈−12% |
| test_geom_camber_cruise | ~29 | ≈−18% |
| test_re_rand | ~48 | ≈−11% |
| **test_avg** | **47.484** | **−13.1%** |

### Noise calibration (from PR #35)

- Anchor (sn=32, nl=5, σ=0.7 + SwiGLU) 2-seed mean 68.173, std 3.23 val (seeds 0, 42). Reproduces PR #27 3-seed mean (68.687) within 0.5 val. Anchor std wider than previously modeled.
- Winner (nl=3, sn=32) 2-seed std: **0.376 val** — tightest observed at σ=0.7 recipes. Win is deeply structural.
- 2-seed merge criterion passed: winner 2-seed mean 54.476 beats sn=16 baseline 61.813 by 7.34 val, ~4.2σ (using sn=16 anchor std 1.742).

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
| **n_layers** | **3** ← new (PR #35) |
| n_head | 4 |
| **slice_num** | **32** ← reverted from 16 (PR #35 tested at sn=32) |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 |
| fourier_features | fixed |
| fourier_m | 160 |
| fourier_sigma | 0.7 |
| ffn | SwiGLU |

**Note:** Winner was tested at `slice_num=32`, not `sn=16` (PR #34 was on the branch base for PR #35's assignment). The combined `(nl=3, sn=16)` is **UNTESTED** — immediate follow-up PR (nezuko) will determine whether the two effects compound. For now, the baseline recipe is **(nl=3, sn=32)**.

**Note on code defaults:** some Config dataclass defaults in `train.py` still reflect pre-merge values (`loss_type="mse"`, `amp=False`, `grad_accum=1`, `fourier_features="none"`, `fourier_m=10`, `fourier_sigma=1.0`, `swiglu=False`). Always pass the full flag list below. `--debug` verification mandatory.

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
    --n_layers 3 \
    --seed 1 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-24 — PR #35: nezuko n_layers=3 sweep (sn=32 + σ=0.7 + SwiGLU)

- **val_avg/mae_surf_p: 54.210 (best seed, s=1) / 54.476 (2-seed mean)** (previous: 60.581 / 61.813, PR #34)
- **test_avg/mae_surf_p: 47.484 (best seed) / 47.336 (2-seed mean)** (previous: 54.640 / 55.316, PR #34)
- W&B runs: `rze3vuj0` (s=0), `jv54vc7o` (s=1) — group `nezuko/n-layers-sn32`
- Change: `--n_layers 3` (was 5). `--slice_num 32` (was 16 in PR #34 — PR #35 tested on sn=32 baseline; compound with sn=16 pending).
- Delta: −10.5% val / −13.1% test (best seed vs prior best seed); 2-seed mean −11.9% val / −14.4% test.
- Decisive pass: 2-seed mean 54.476 beats sn=16 baseline 61.813 by 7.34 val — 4.2σ.
- **n_layers landscape is STRICTLY MONOTONIC DECREASING with depth:** nl=3 (54.48) < nl=4 (59.04) < nl=5 (68.17) < nl=6 (70.52) < nl=7 (72.42).
- **Mechanism: budget-bound.** nl=3 trains for 32 epochs in 30-min budget; nl=7 only reaches 14 epochs. Depth's representational gains don't compensate for epoch loss.
- Uniform split win; largest on `val_geom_camber_cruise` (−32.4%).
- Combined (nl=3, sn=16) is **untested**; immediate follow-up PR (nezuko) resolves this.

### 2026-04-24 — PR #34: frieren slice_num=16 sweep (sn∈{8,16,24} vs sn=32 anchor)

- **val_avg/mae_surf_p: 60.581 (best seed, s=0) / 61.813 (2-seed mean)** (previous: 67.186 / 68.687, PR #27)
- **test_avg/mae_surf_p: 54.640 (best seed) / 55.316 (2-seed mean)** (previous: 58.358 / 60.680, PR #27)
- W&B runs: `moydqx8l` (s=0), `tkfnon33` (s=1) — group `frieren/slice-num-lower`
- Change: `--slice_num 16` (was 32). Both seeds hit best val at final epoch (headroom).

### 2026-04-24 — PR #27: nezuko slice_num=32 sweep (σ=0.7 + SwiGLU + sn=32)

- **val_avg/mae_surf_p: 67.186 (best seed, s=2) / 68.687 (3-seed mean)** (previous: 69.845 / 70.667, PR #24)
- **test_avg/mae_surf_p: 58.358 (best seed) / 60.680 (3-seed mean)** (previous: 62.778 / 62.691, PR #24)

### 2026-04-24 — PR #24: alphonse σ × SwiGLU sweep (σ=0.7 + SwiGLU)

- **val_avg/mae_surf_p: 69.845 (best seed) / 70.667 (2-seed mean)** (previous: 73.660, PR #20)

### 2026-04-24 — PR #20: fern Fourier σ=1 + SwiGLU feedforward

- **val_avg/mae_surf_p: 73.660** (previous: 84.737, PR #7). −13.1% val.

### 2026-04-23 — PR #7: alphonse Fourier PE fixed σ=1 m=160

- **val_avg/mae_surf_p: 84.737** (previous: 88.268, PR #12).

### 2026-04-23 — PR #12: fern AMP + grad_accum=4

- **val_avg/mae_surf_p: 88.268** (previous: 93.127, PR #11).

### 2026-04-23 — PR #11: frieren surf_weight=1 on L1

- **val_avg/mae_surf_p: 93.127** (previous: 103.036, PR #3).

### 2026-04-23 — PR #3: frieren L1 loss

- **val_avg/mae_surf_p: 103.036** (previous: no baseline).

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four val splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint on four test splits.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `kagent_v_students`.
2. Update this file with new metric, PR number, W&B run link.
3. Commit on advisor branch.

**Multi-seed requirement:** merge claims < 5% require 2-seed anchors. Winner 2-seed mean must beat current 2-seed anchor mean by > 1σ of anchor spread. **Noise floor at σ=0.7 recipes: ~1.5–3.2 val**; tighten at each new recipe.
