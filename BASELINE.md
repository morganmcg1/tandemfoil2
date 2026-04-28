# Baseline — willow-pai2e-r3

Advisor branch: `icml-appendix-willow-pai2e-r3`
W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
Primary metric: `val_avg/mae_surf_p` (lower is better). Test mirror: `test_avg/mae_surf_p`.

---

## Founding baseline (round 1 — no hypothesis PR merged yet)

**Commit baseline via PR #807** (NaN-safe masked accumulation bug fix — landed 2026-04-28).
All subsequent runs produce finite `test_avg/mae_surf_p` numbers.

### Default model config (unmodified `train.py`)

- n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~1M params)
- AdamW lr=5e-4, wd=1e-4, batch_size=4, surf_weight=10.0
- 50-epoch cosine annealing (effective ~14 epochs at 30-min timeout)

### Best round-1 val metric (single seed)

| Run | Branch | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|-----|--------|--------------------|----------------------|-------|
| `8cvp4x6r` | thorfinn/unmodified-baseline (matched) | **122.15** | 118.01† | Best unmodified-model result; from PR #762 matched comparison |
| `thnnvgaw` | edward/lr-warmup-cosine v1 | 135.89 | null (re-eval pending) | |
| `t0xgo0zv` | frieren/fourier-re-encoding v1 | 141.25 | null (re-eval pending) | |
| `zaqz12qi` | alphonse/channel-weighted-surface-loss v1 | 146.10 | **130.90** (re-eval via PR #807) | |

† 3-split average (test_geom_camber_cruise NaN pre-fix); post-fix re-eval not yet run.

Single-seed run-to-run noise band: **122–146** at 14-epoch budget (~±10%).

### Founding test baseline (clean number for paper-facing comparisons)

`test_avg/mae_surf_p = 130.90` (W&B run `zaqz12qi`, alphonse channel-weighted v1, re-evaluated with fixed scorer in PR #807)

### Beat-threshold for round 2+

Future PRs must achieve **`val_avg/mae_surf_p < 122.15`** to demonstrate improvement above the round-1 noise band.
For a merge decision: any val_avg below 122.15 merges; gains <5% at single seed will be flagged for multi-seed confirmation.

---

## 2026-04-28 21:28 — PR #814: Huber surface loss (delta=1.0)

**New best — merged 2026-04-28.**

- **val_avg/mae_surf_p: 103.13** (−15.6% vs prior baseline 122.15)
- **test_avg/mae_surf_p: 92.99** (−29% vs founding test 130.90)
- **W&B run:** `at52zeu5` (askeladd, huber-surf-loss v1)
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 123.94 | 109.10 |
| `geom_camber_rc` | 111.30 | 98.88 |
| `geom_camber_cruise` | 81.66 | 69.84 |
| `re_rand` | 95.62 | 94.17 |
| **avg** | **103.13** | **92.99** |

- **Config:** Default model + Huber surface loss (delta=1.0, surf_weight=10.0, AdamW lr=5e-4, wd=1e-4)
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --wandb_group huber-surf-loss \
    --wandb_name v1 \
    --agent willowpai2e3-askeladd
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 103.13`

---

## 2026-04-28 22:20 — PR #761: L1 (MAE) surface loss aligned with metric

**New best — merged 2026-04-28.**

- **val_avg/mae_surf_p: 92.63** (−10.2% vs prior baseline 103.13)
- **test_avg/mae_surf_p: 82.83** (−10.9% vs prior test 92.99)
- **W&B run:** `tirux1y1` (tanjiro, l1-surface-mae-loss v1-rebased)
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 109.65 | 96.33 |
| `geom_camber_rc` | 101.17 | 90.80 |
| `geom_camber_cruise` | 72.37 | 61.90 |
| `re_rand` | 87.33 | 82.29 |
| **avg** | **92.63** | **82.83** |

- **Config:** Default model + L1 surface loss (`torch.where(surf_mask, abs_err, 0).sum() / surf_mask.sum()`), surf_weight=10.0, AdamW lr=5e-4, wd=1e-4, --epochs 14
- **Key finding:** L1 beats Huber(delta=1.0) by 10.2% val / 10.9% test. Pure linear gradient on all surface residuals outperforms Huber's smooth-near-zero quadratic for this dataset's heavy-tailed pressure distribution.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --surf_weight 10.0 \
    --wandb_group l1-surface-mae-loss \
    --wandb_name v1-rebased \
    --agent willowpai2e3-tanjiro
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 92.63`

---

*This file is updated after each merge. Entries are cumulative — do not delete prior entries.*
