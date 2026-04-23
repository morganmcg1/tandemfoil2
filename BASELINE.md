# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**PR #12 — fern: Throughput scaling — AMP + grad accumulation to unlock more epochs**
- **val_avg/mae_surf_p: 88.268** (lower is better)
- W&B run: `n68w9q7o` (`fern/sw1-amp-accum4`)
- Best epoch: 19 (timeout-bounded at ~31.5 min; AMP unlocked +5 epochs vs baseline)
- test_avg/mae_surf_p: **79.733** (test_geom_camber_cruise +Inf bug now patched)

### Per-split val surface-p MAE (best checkpoint, epoch 19)

| Split | mae_surf_p | vs PR #11 |
|-------|-----------|-----------|
| val_single_in_dist | 104.50 | −2.3% |
| val_geom_camber_rc | 100.70 | −5.1% |
| val_geom_camber_cruise | 65.19 | −11.0% |
| val_re_rand | 82.69 | −4.0% |
| **val_avg** | **88.268** | **−5.2%** |

### Per-split test surface-p MAE (best checkpoint)

| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 94.07 |
| test_geom_camber_rc | 90.90 |
| test_geom_camber_cruise | 56.18 |
| test_re_rand | 77.78 |
| **test_avg** | **79.733** |

### Current default config (post-merge)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| grad_accum | **4** (effective bs=16) |
| amp | **True** (bf16 autocast, no GradScaler) |
| surf_weight | 1.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=total_optimizer_steps) |
| loss | L1 (abs, vol + surf_weight × surf) in normalized space |

Reproduce:
```bash
cd target && python train.py \
    --agent <student> \
    --loss_type l1 \
    --surf_weight 1 \
    --amp true \
    --grad_accum 4 \
    --batch_size 4 \
    --wandb_name "<student>/<experiment>"
```

---

## Baseline history

### 2026-04-23 — PR #12: fern throughput scaling (AMP bf16 + grad_accum=4)

- **val_avg/mae_surf_p: 88.268** (previous: 93.127, PR #11)
- W&B run: `n68w9q7o` (group: `fern/throughput-amp-sw1`)
- Change: Added `--amp true` (bf16 autocast) + `--grad_accum 4` (eff_bs=16). AMP cuts per-epoch time 132s→100s (+28% throughput), unlocking epoch 19 vs epoch 14 under the same 30-min budget. Grad-accum at eff_bs=16 via 4 micro-batches compresses noisy gradient steps without inflating VRAM (~33 GB peak).
- Delta: −5.2% vs previous baseline (93.127). Uniform improvement across all 4 val splits (largest: camber_cruise −11.0%).
- test_avg/mae_surf_p now finite (79.733) — test_geom_camber_cruise +Inf scoring bug patched in this PR.

### 2026-04-23 — PR #11: frieren fine surf_weight sweep on L1 (surf_weight=1)

- **val_avg/mae_surf_p: 93.127** (previous: 103.036, PR #3)
- W&B run: `yt7eup38` (group: `frieren/l1-surf-weight-sweep`)
- Change: surf_weight reduced from 10 → 1. Under L1 loss, volume supervision is load-bearing for surface-pressure prediction — excessive surface upweighting starves the shared feature extractor of volume gradient. sw=1 wins on every channel (surface and volume) simultaneously.
- Delta: −9.62% vs previous baseline (103.036).
- Wins on 3 of 4 splits: −20.0% in_dist, −9.5% camber_rc, +4.5% camber_cruise (slight regression), −6.0% re_rand.
- Test 3-split avg (excl. NaN): 91.58 (sw=1) vs 105.48 (sw=10 control). Consistent with val.

### 2026-04-23 21:40 — PR #3: frieren Huber/L1 loss reformulation

- **val_avg/mae_surf_p: 103.036** (previous: no baseline on this track)
- W&B run: `w2jsabii` (group: `frieren/loss-reformulation-v2`)
- Change: L1 loss in normalized space instead of MSE. surf_weight=10 unchanged.
- Delta: −21.9% vs MSE baseline at same run budget (131.985).
- Wins uniformly: −27.7% in_dist, −29.8% camber_rc, −39.5% camber_cruise, −32.5% re_rand.

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four validation splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same quantity, computed from the best-val checkpoint on the four held-out test splits. Currently blocked by +Inf bug in `test_geom_camber_cruise/000020.pt`.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here, the advisor:

1. Squash-merges the winning PR into `kagent_v_students`.
2. Updates this file with the new metric, PR number, and W&B run link.
3. Commits the update on the advisor branch.
