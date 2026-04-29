# SENPAI Research Results

## 2026-04-29 10:48 — PR #1097: More physics slices: slice_num 64 → 128
- Branch: `charliepai2f1-frieren/slice-num-128`
- Hypothesis: Doubling Transolver `slice_num` from 64 → 128 gives finer learnable mesh partitions and more capacity in the slice-routing path; predicted -3% to -6% on `val_avg/mae_surf_p`.
- Predicted overhead: a couple of percent wall clock; reality was much worse.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 10) | **162.562** |
| `test_avg/mae_surf_p` (3 finite splits) | 166.409 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN (cruise GT corruption — see PR #1095 NaN-fix) |
| Epochs run | 11 / 50 (timeout-bound) |
| Peak VRAM | 54.5 GB (massive headroom) |
| Wall clock | ~30 min (~173 s/epoch) |
| Metrics file | `models/model-charliepai2f1-frieren-slice-num-128-20260429-095421/metrics.jsonl` |

### Per-split val (epoch 10)

| Split | mae_surf_p |
|---|---|
| `val_single_in_dist` | 201.42 |
| `val_geom_camber_rc` | 196.92 |
| `val_geom_camber_cruise` | 116.38 |
| `val_re_rand` | 135.53 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 183.40 |
| `test_geom_camber_rc` | 179.35 |
| `test_geom_camber_cruise` | NaN (cruise GT corruption — only sample 20 has Inf in p) |
| `test_re_rand` | 136.48 |

### Analysis & conclusions

- 162.56 at epoch 10 is **+21%** vs. the provisional baseline (133.89) — but both are timeout-bound and frieren's run reached only epoch 10/11 vs. edward's 13/14, so the comparison is undertrained on both ends. Val curve was non-monotonic (e7→165.7, e8→171.5, e9→189.9, e10→162.6, e11→193.3), suggesting bs=4 + slice_num=128 has high gradient noise.
- **Critical underutilization.** Peak VRAM 54.5 GB of 95 GB available — frieren can roughly double bs without OOM, which would both reduce per-epoch wall clock and stabilize gradients.
- **Frieren independently rediscovered the `data/scoring.py` NaN bug** (already patched on advisor branch via PR #1095 review).
- **Sent back** with: keep slice_num=128, raise bs to 8 (or higher targeting 80%+ VRAM), add output pressure clamping to prevent fp32 overflow on cruise. Goal: 20+ epochs and finite test_avg.

## 2026-04-29 10:47 — PR #1100: Wider model + larger batch: n_hidden=256, batch_size=8
- Branch: `charliepai2f1-tanjiro/wider-bs8`
- Hypothesis: Wider Transolver (n_hidden 128 → 256, ~4× params) + larger batch (4 → 8) for better gradient stats; predicted -5% to -10% on `val_avg/mae_surf_p`.
- Reality: bs=8 OOMed (91.94 GB used + 3.36 GB needed); bs=6 also OOMed; bs=5 with `expandable_segments` ran.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 7) | **165.304** |
| `test_avg/mae_surf_p` (3 finite splits) | 168.10 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN (cruise vol_loss = +Inf — fp32 overflow) |
| Epochs run | 8 / 50 (timeout-bound; ~3.78 min/epoch) |
| Peak VRAM | 92.39 GB (very tight against 95 GB ceiling) |
| Params | 2.60M (~4× baseline 0.65M) |
| Metrics file | `models/model-charliepai2f1-tanjiro-wider-bs5-20260429-100402/metrics.jsonl` |

### Per-split val (epoch 7)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 197.69 | 2.58 | 1.02 |
| `val_geom_camber_rc` | 182.87 | 3.48 | 1.30 |
| `val_geom_camber_cruise` | 126.37 | 1.79 | 0.76 |
| `val_re_rand` | 154.28 | 2.71 | 1.00 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 180.46 |
| `test_geom_camber_rc` | 175.05 |
| `test_geom_camber_cruise` | NaN (vol_loss = +Inf, fp32 overflow on cruise) |
| `test_re_rand` | 148.77 |

### Analysis & conclusions

- 165.30 val_avg at epoch 7/50 is **+23%** vs. provisional baseline (133.89), but only 8 vs. 14 epochs — undertrained. Val_avg trajectory: e1→222 → e7→165 → e8→226 (oscillation suggests we're already near the noisy regime under bs=5 + this width).
- **Width is the wrong knob with this MLP shape.** mlp_ratio=2 + n_hidden=256 + ~242K nodes × bs=5 → 92 GB VRAM. The MLP intermediate dominates activation memory.
- **Test pressure overflow on cruise.** Cruise samples produce predictions large enough that squared error overflows fp32. This is independent of the cruise GT NaN — it's a model output stability issue.
- **Sent back** with: keep n_hidden=256, drop mlp_ratio 2 → 1 (halves MLP activation), bs=6 (fall back to 4 if OOM), add output pressure clamping. Goal: 20+ epochs and finite test_avg.

## 2026-04-29 10:34 — PR #1095: Per-channel surface loss: 4x weight on pressure channel
- Branch: `charliepai2f1-edward/pressure-channel-weight`
- Hypothesis: Reweight surface MSE channels `(Ux, Uy, p) = (1, 1, 4)` to bias capacity toward the ranked metric (`mae_surf_p`).
- Predicted delta: -5% to -10% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 13) | **133.892** |
| `test_avg/mae_surf_p` over 3 finite test splits | 132.106 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN — see scoring bug |
| Epochs run | 14 / 50 (timeout-bound) |
| Peak VRAM | 42.13 GB |
| Wall clock | 31.0 min |
| Metrics file | `models/model-charliepai2f1-edward-pressure-channel-weight-20260429-095436/metrics.jsonl` |

### Per-split val (epoch 13)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 160.697 | 2.575 | 1.071 |
| `val_geom_camber_rc` | 145.212 | 3.435 | 1.301 |
| `val_geom_camber_cruise` | 102.659 | 1.934 | 0.781 |
| `val_re_rand` | 127.000 | 2.870 | 1.053 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 140.951 |
| `test_geom_camber_rc` | 133.153 |
| `test_geom_camber_cruise` | NaN (one sample with NaN p in y, see bug) |
| `test_re_rand` | 122.213 |

### Analysis & conclusions

- Run trained stably; val curve trended down monotonically (230.9 → 133.89 over 14 epochs).
- This is the first round-1 number on the board; with no prior baseline on this branch it provisionally sets the bar at **133.892** until other in-flight PRs return.
- **Confounded normalization.** The instructed formula divides by `ch_w.sum()=6`, softening aggregate surface signal by ~3× vs. the unweighted variant. The student correctly flagged this — the run is partially a "lower effective surf_weight" experiment, not a pure pressure-channel-boost. Sending PR back with corrected normalization (`/ ch_w.mean()` keeps aggregate surface contribution constant).
- **Critical bug found in `data/scoring.py`.** Mask-by-multiply propagates NaN through `(NaN × 0) = NaN`, producing NaN in `test_avg/mae_surf_p` whenever any test sample has non-finite y. One sample in `test_geom_camber_cruise` triggers it. Fix applied to advisor branch (`torch.where`-based masking).
- Still 14/50 epochs runs (timeout-bound). Suggests architectural/loss changes that don't slow the per-epoch wall clock will compound with the training-discipline experiments in flight.
