# SENPAI Research Results — willow-pai2e-r5

---

## 2026-04-28 19:51 — PR #732: Scale Transolver to n_hidden=256, n_layers=7, n_head=8

- **Branch:** `willowpai2e5-alphonse/larger-model-capacity` (closed)
- **W&B run:** `pkyat9dy` — group `larger-model-capacity`
- **Hypothesis:** Scaling Transolver from 0.93M to 3.01M params (n_hidden=256, n_layers=6, n_head=8 after n_layers=7 OOM fallback) improves fitting and OOD generalization.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist`      | 184.36 | 169.38 |
| `val_geom_camber_rc`      | 162.75 | 152.78 |
| `val_geom_camber_cruise`  | 130.79 | **NaN** |
| `val_re_rand`             | 141.88 | 147.91 |
| **avg**                   | **154.95** | **NaN** |

Best epoch = 6 (final epoch before 30-min timeout). Training trajectory (val_avg): 245.66 → 210.36 → 202.81 → 184.30 → 184.74 → 154.95.

### Commentary & Conclusions

- **Non-conclusive under our compute budget.** The 30-min SENPAI_TIMEOUT_MINUTES cap allowed only 6 of 50 scheduled epochs. The CosineAnnealingLR (T_max=50) was running at ~94% initial LR — effectively zero LR decay. Trajectory shows rapid, unfinished descent; we don't know where the model would converge.
- **5.2× throughput penalty.** Each epoch took ~5.2 min vs ~1.5 min baseline. A matched-wall-clock baseline (128/5/4) would do ~20 epochs; this 3.01M-param model got 6.
- **NaN on `test_geom_camber_cruise/mae_surf_p`.** The undertrained large model produced non-finite pressure predictions on at least one hard test sample. Val numbers are finite because no val sample was numerically unstable. This is a model-stability issue, not a data issue.
- **Bug identified in `data/scoring.py`.** `accumulate_batch` skips samples with non-finite ground truth but not non-finite predictions — a single NaN prediction node poisons the entire split's channel sum. data/ is read-only for student PRs; flagging for follow-up.
- **Reference data point recorded:** val_avg/mae_surf_p = 154.95 (first run; model ~1/3 trained). Will compare against in-flight Wave-1 runs once they complete.
- **Decision:** Closed (premature — NaN test metric, no matched-budget baseline comparison, 5.2× throughput cost unjustified without fair comparison).

### Next step for this direction

Revisit capacity scaling after (1) in-flight Wave-1 runs give us a baseline-architecture val number, (2) BF16 mixed-precision is available to reduce per-epoch time, and (3) gradient accumulation is verified to allow n_layers=7. Alphonse reassigned to `film-re-conditioning` (PR #796).

---

## 2026-04-28 20:15 — PR #763: Add physics-informed distance features to Transolver input

- **Branch:** `willowpai2e5-thorfinn/feature-engineering` (merged)
- **W&B run:** `072wo9xb` — group `feature-engineering`
- **Hypothesis:** Appending dist_to_surface, log(1+dist_to_surface), and is_tandem as derived node features provides physically grounded inductive bias for boundary-layer gradients and tandem vs. single-foil regime.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 177.0157 | 148.3098 |
| `val_geom_camber_rc` | 157.4591 | 145.5500 |
| `val_geom_camber_cruise` | 105.7864 | 91.0172 |
| `val_re_rand` | 125.4113 | 121.3623 |
| **avg** | **141.4181** | **126.5598** |

Best epoch = 12 of 50 (30-min timeout). AUG_X_DIM=27, +768 params.

### Commentary

- **First PR with clean test_avg across all 4 splits.** Student correctly diagnosed the `test_geom_camber_cruise` NaN as a dataset-side issue (sample 20 has 761 NaN y[:, 2] entries) and implemented a NaN-safe `evaluate_split` workaround in `train.py` since `data/` is read-only. Critical fix — all future runs now report finite test_avg.
- **Decision: Merged.** Beats #732 reference (154.95), produces clean test metrics. Features are physically motivated and low-risk.
- Became new baseline at val_avg=141.42 before fern's #737 landed.

---

## 2026-04-28 20:25 — PR #737: Add 5-epoch linear warmup + peak lr=1e-3 before cosine decay

- **Branch:** `willowpai2e5-fern/lr-warmup-cosine` (merged)
- **W&B run:** `5b22tecz` — group `lr-warmup-cosine`
- **Hypothesis:** Linear LR warmup (1e-4 → 1e-3, 5 epochs) followed by cosine decay to eta_min=1e-6 stabilises early training of slice-attention projections and allows a higher peak LR without instability.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 149.241 | 126.021 |
| `val_geom_camber_rc` | 146.033 | 129.155 |
| `val_geom_camber_cruise` | 96.362 | NaN (pre-fix; ~91 expected) |
| `val_re_rand` | 119.852 | 115.590 |
| **avg** | **127.872** | NaN (3-split avg ≈ 123.59) |

Best epoch = 14 of 50 (30-min timeout). LR at epoch 14 ≈ 8e-4 (barely into cosine decay).

### Commentary

- **Decision: Merged (current best).** val_avg=127.87 beats new baseline of 141.42 after #763 merged. Schedule change is orthogonal to features change — both merged cleanly.
- **Critical gap identified:** T_max=50 with a 30-min (~14 epoch) budget means the LR never decays properly. Model was still at 80% of peak LR when cut off, trajectory still steeply descending (136→128 at epochs 13→14). Budget-matched re-run (epochs=14, warmup=2) assigned to fern as PR #809.
- **test_avg is NaN** because #737 ran BEFORE #763's NaN fix was in the merged branch. Future runs will be clean.

---

## 2026-04-28 20:30 — PR #733: Increase slice_num from 64 to 256 for richer physics decomposition

- **Branch:** `willowpai2e5-askeladd/more-slices` (closed)
- **W&B run:** `8l3pbq6x` — group `more-slices`
- **Hypothesis:** Quadrupling slice tokens (64→256) gives the model finer-grained physics decomposition for boundary layer / wake / freestream separation.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 185.28 | 161.17 |
| `val_geom_camber_rc` | 179.72 | 160.89 |
| `val_geom_camber_cruise` | 111.53 | NaN (pre-fix) |
| `val_re_rand` | 129.46 | 130.14 |
| **avg** | **151.50** | NaN (3-split avg ≈ 150.73) |

Best epoch = 8 of 50 (30-min timeout). slice_num=256 fits in 82.3 GB VRAM. Per-epoch wall-time: 4.2 min vs 2.2 min for slice_num=64.

### Commentary

- **Decision: Closed (regression).** val_avg=151.50 vs current baseline 127.87 = 18.5% regression, well above the 5% close threshold.
- **Throughput penalty is structural, not tunable.** ~2× per-epoch cost meant 8 epochs vs ~14 for baseline-config runs in the same 30-min cap. Val curve still steeply falling at cutoff (161→151 epochs 7→8), so even granting "256 needs more time," it's strictly worse for our compute budget. Would need a budget extension or a separate throughput improvement to revisit.
- Student also flagged the same `test_geom_camber_cruise` NaN issue thorfinn diagnosed in #763 — but that's already fixed in the current merged baseline. Confirms the bug is global (not slice-num-specific).
- Askeladd reassigned to `mixed-precision-bf16` (PR #811) — directly attacks the throughput constraint this PR exposed.
