# SENPAI Research Results — charlie-pai2d-r5

## 2026-04-27 23:30 — PR #293: L1 loss in normalized space (alignment with MAE eval metric) — **MERGE (winner)**

- Branch: `charliepai2d5-edward/l1-loss`
- Hypothesis: replace MSE with L1 in normalized space; MAE-aligned with the eval metric, more robust to high-Re outliers.

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **101.868** |
| `val_single_in_dist/mae_surf_p` | 125.264 |
| `val_geom_camber_rc/mae_surf_p`  | 108.034 |
| `val_geom_camber_cruise/mae_surf_p` |  75.262 |
| `val_re_rand/mae_surf_p` | 98.912 |
| `test_avg/mae_surf_p` (4-split, with NaN) | NaN |
| `test_avg/mae_surf_p` (3 clean splits) | **102.606** |
| `test_single_in_dist/mae_surf_p` | 113.966 |
| `test_geom_camber_rc/mae_surf_p` |  99.998 |
| `test_geom_camber_cruise/mae_surf_p` | NaN (data bug) |
| `test_re_rand/mae_surf_p` | 93.854 |

Metric summary: `models/model-l1-loss-20260427-223415/metrics.yaml`

### Analysis

Pure L1 swap, no other changes. Training was numerically clean from epoch 1 (no Huber fallback needed). Validation `val_avg/mae_surf_p` descended monotonically across all 14 reached epochs (266 → 209 → 184 → 171 → 161 → 135 → 142 → 140 → 125 → 124 → 112 → 107 → 106 → 102) and was still trending down at the 30-min timeout. Edward did detective work and identified a pre-existing data + scoring bug that affects the round: `test_geom_camber_cruise` sample 20 has 761 non-finite values in the `p` channel of GT, and `data/scoring.accumulate_batch` computes `err = (pred - y).abs()` *before* masking, which lets NaN propagate into the per-channel sums. Same pattern hit fern (#296) and thorfinn (#305). Read-only constraint on `data/scoring.py` means the fix has to be flagged for the human team or solved via a sanitization pre-step in `train.py`.

### Decision

Merge — clear round-1 winner. New baseline `val_avg/mae_surf_p = 101.87`, 3-clean-split `test_avg/mae_surf_p = 102.61`. The cruise NaN is a pre-existing artifact, not L1's fault, and edward's stability investigation confirmed the model itself produces only finite predictions on that split.

---

## 2026-04-27 23:30 — PR #305: Finer attention: slice_num 64→128, n_head 4→8 — **CLOSE**

- Branch: `charliepai2d5-thorfinn/slices-heads-2x`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 8/50) | **160.676** |
| `val_single_in_dist/mae_surf_p`     | 219.613 |
| `val_geom_camber_rc/mae_surf_p`     | 179.649 |
| `val_geom_camber_cruise/mae_surf_p` | 108.617 |
| `val_re_rand/mae_surf_p` | 134.825 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **162.22** |

Metric summary: `models/model-slices-heads-2x-20260427-223358/metrics.yaml`

### Analysis

Per-epoch wall time was ~252 s vs ~131 s for edward / fern — almost exactly 2× the baseline cost. Inside the 30-min `SENPAI_TIMEOUT_MINUTES` cap this gives only 8 epochs vs 14. Worse, the test split exposed the dim_head=16 instability the PR pre-warned about: model produced non-finite predictions on at least one cruise test sample, `surf_loss=NaN` and `vol_loss=+Inf` on that split. Even granting that the model is far from converged at epoch 8, the per-epoch unit economics make this a poor fit for the current timeout regime.

### Decision

Close. The configuration is fundamentally too slow per epoch to compete with the loss-formulation winners, and the dim_head=16 fragility makes test scoring unreliable. The natural fallback (`n_hidden=192` to restore dim_head=24) overlaps with askeladd's running PR #290, so reassigning thorfinn to a non-overlapping hypothesis is the better use of the slot.

---

## 2026-04-27 23:30 — PR #296: Linear warmup then cosine, peak lr 1e-3 — **REQUEST CHANGES (send back)**

- Branch: `charliepai2d5-fern/lr-warmup-1e3`

### Results

| metric | value |
|---|---:|
| `val_avg/mae_surf_p` (best ep 14/50) | **137.319** |
| `val_single_in_dist/mae_surf_p`     | 175.812 |
| `val_geom_camber_rc/mae_surf_p`     | 150.559 |
| `val_geom_camber_cruise/mae_surf_p` |  99.339 |
| `val_re_rand/mae_surf_p` | 123.565 |
| `test_avg/mae_surf_p` (4-split) | NaN |
| `test_avg/mae_surf_p` (3 clean) | **136.998** |

Metric summary: `models/model-lr-warmup-1e3-20260427-223514/metrics.yaml`

### Analysis

The hypothesis is reasonable but the schedule isn't matched to the budget: `cosine T_max = MAX_EPOCHS - warmup_epochs = 45`, while only 14 epochs were ever run. So warmup occupied epochs 1–5, and epochs 6–14 ran at near-peak LR (~9.4e-4 → 8.2e-4) — effectively a "warmup + plateau at ~1e-3" run rather than the intended warmup+decay. `val_avg/mae_surf_p` was still descending at the timeout. We can't tell whether the schedule helps until cosine actually decays into the wall budget.

### Decision

Send back — set `--epochs 14` so cosine T_max scales to the actually-reachable budget and we get a clean read on the schedule. Same student branch, same hypothesis, just a one-line config tweak.
