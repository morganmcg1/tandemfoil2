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

## 2026-04-28 23:35 — PR #815: FiLM conditioning each Transolver block on log(Re)

**New best — merged 2026-04-28.**

- **val_avg/mae_surf_p: 82.77** (−10.6% vs prior baseline 92.63)
- **test_avg/mae_surf_p: 72.27** (−12.7% vs prior test 82.83)
- **W&B run:** `mfjoux5g` (thorfinn, film-re-conditioning v2-on-l1)
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 95.54 | 81.63 |
| `geom_camber_rc` | 91.38 | 82.02 |
| `geom_camber_cruise` | 64.90 | 53.62 |
| `re_rand` | 79.26 | 71.82 |
| **avg** | **82.77** | **72.27** |

- **Config:** Default model + FiLM per-block conditioning on log(Re) + L1 surface loss. FiLMLayer: 1→32 SiLU →2×n_hidden per block, zero-init, (1+γ)·h+β post-block modulation. surf_weight=10.0, AdamW lr=5e-4, wd=1e-4, --epochs 14. +42,560 params (+6.4%).
- **Key finding:** FiLM and L1 stack constructively (orthogonal mechanisms: loss-shape vs hidden-state Re modulation). FiLM gives largest gains on Re-stratified and cruise splits (`val_re_rand` −9.2%, `val_geom_camber_cruise` −10.3%), exactly as predicted by the regime-modulation hypothesis.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --wandb_group film-re-conditioning \
    --wandb_name v2-on-l1 \
    --agent willowpai2e3-thorfinn
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 82.77`

---

## 2026-04-29 — PR #909: Pre-block FiLM (condition attention input on log(Re))

**New best — merged 2026-04-29.**

- **val_avg/mae_surf_p: 81.55** (−1.5% vs prior baseline 82.77)
- **test_avg/mae_surf_p: 72.40** (+0.2% vs prior test 72.27)
- **W&B run:** `x7hi1qun` (thorfinn, film-pre-block v1)
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 93.70 | 84.20 |
| `geom_camber_rc` | 92.70 | 82.91 |
| `geom_camber_cruise` | 62.62 | 51.93 |
| `re_rand` | 77.16 | 70.54 |
| **avg** | **81.55** | **72.40** |

- **Config:** Default model + FiLM applied pre-block (before attention/MLP) instead of post-block. FiLM modulation `(1+γ)·h + β` applied to block *input* in `Transolver.forward`; `TransolverBlock.forward` receives plain `fx` with no gamma/beta. FiLMLayer unchanged (1→32 SiLU→2×n_hidden, zero-init). surf_weight=10.0, AdamW lr=5e-4, wd=1e-4, --epochs 14. Params: 704,919 (unchanged).
- **Key finding:** Pre-block FiLM yields a small but clean val win (+1.5%). The Re-targeted splits benefit as predicted (`val_re_rand` −2.10, `val_geom_camber_cruise` −2.28). Mixed per-split signal on in-dist and geom_camber_rc suggests input-side Re-conditioning is more useful for OOD-Re regimes than for in-distribution geometry. Test is flat (+0.2%), which is consistent with single-seed noise. Mechanism: pre-block conditioning modulates Q/K/V attention computation directly (regime-aware attention patterns) vs post-block modulation which only scales block outputs.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --wandb_group film-pre-block \
    --wandb_name v1 \
    --agent willowpai2e3-thorfinn
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 81.55`

---

## 2026-04-29 — PR #910: Re-stratified batch sampling

**New best — merged 2026-04-29.**

- **val_avg/mae_surf_p: 79.54** (−2.4% vs prior baseline 81.55)
- **test_avg/mae_surf_p: 70.26** (−3.0% vs prior test 72.40)
- **W&B run:** `wakfw4uy` (nezuko, re-stratified-sampling v1)
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 84.70 | 73.79 |
| `geom_camber_rc` | 92.95 | 83.50 |
| `geom_camber_cruise` | 63.49 | 52.45 |
| `re_rand` | 77.02 | 71.29 |
| **avg** | **79.54** | **70.26** |

- **Config:** Default model + FiLM pre-block + L1 surface loss + Re-stratified batch sampling (`--re_stratify`). Round-robin sampler builds each batch from 5 Re quintile buckets; batch log(Re) std=0.652 (vs ~0 under random). surf_weight=10.0, AdamW lr=5e-4, wd=1e-4, --epochs 14. Params: 704,919 (unchanged).
- **Key finding:** Re-stratified batches compound with FiLM+L1. Largest gain on `single_in_dist` (−11.4% val) — stratification equalizes the pressure-scale gradient bias that random sampling creates (high-Re samples dominate gradient by absolute pressure magnitude under L1). Re-targeted splits (`re_rand`, `cruise`) also improved. Only `geom_camber_rc` slightly regressed (+1.7% val).
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --re_stratify \
    --wandb_group re-stratified-sampling \
    --wandb_name v1 \
    --agent willowpai2e3-nezuko
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 79.54`

---

## 2026-04-29 — PR #961: SwiGLU MLP (replace GELU MLP with Swish-gated linear unit)

**New best — merged 2026-04-29.**

- **val_avg/mae_surf_p: 62.20** (−21.8% vs prior baseline 79.54)
- **test_avg/mae_surf_p: 55.04** (−21.6% vs prior test 70.26)
- **W&B run:** `sv9ktfk3` (alphonse, swiglu-mlp v1)
- **Note:** 12/14 epochs completed (30-min env timeout; +12.6% per-epoch cost). Best checkpoint from epoch 12 — already 25.5% better than baseline at same epoch 12.
- **Per-split:**

| Split | val surf_p | test surf_p |
|---|---|---|
| `single_in_dist` | 74.96 | 65.07 |
| `geom_camber_rc` | 73.39 | 67.47 |
| `geom_camber_cruise` | 42.66 | 35.67 |
| `re_rand` | 57.81 | 51.93 |
| **avg** | **62.20** | **55.04** |

- **Config:** Default model + SwiGLU MLP (replaces GELU MLP in each TransolverBlock; 3 linear layers: w_gate, w_up, w_down with mlp_ratio=2) + FiLM pre-block + L1 surface loss + Re-stratified sampling. +0.17M params (+24%; total 0.87M). surf_weight=10.0, AdamW lr=5e-4, wd=1e-4, --epochs 14 (12 completed before timeout).
- **Key finding:** SwiGLU bilinear gating (`silu(gate) * up`) is the largest single-PR improvement in this round (+21.8%). Gains are across all splits but largest on OOD-extrapolation splits (`geom_camber_cruise` −32.8%, `re_rand` −24.9%), consistent with bilinear forms helping the model express higher-order feature interactions needed for out-of-distribution flow regimes. Wall-clock penalty is real (+12.6%/epoch → 12/14 epochs at budget); worth tuning with smaller `mlp_ratio` (4/3) in a follow-up.
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --epochs 14 \
    --wandb_group swiglu-mlp \
    --wandb_name v1 \
    --agent willowpai2e3-alphonse
  ```
- **Beat-threshold going forward:** `val_avg/mae_surf_p < 62.20`

---

*This file is updated after each merge. Entries are cumulative — do not delete prior entries.*
