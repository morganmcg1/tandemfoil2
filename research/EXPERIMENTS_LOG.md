# SENPAI Research Results — willow-pai2d-r5

Per-PR experiment log. New entries are appended chronologically; the latest entries are at the top.

## 2026-04-28 00:30 — PR #338: LR warmup + peak 1e-3 cosine — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-frieren/lr-warmup-cosine` (sits on pre-#336 commit, slice_num=64)
- Two-config sweep (`lr=5e-4` control vs `lr=1e-3` main), both with 2-epoch linear warmup + cosine T_max=48
- Both timeout-bound at epoch 14/50 (cosine arm only ~25% engaged)

| Run | val_avg/mae_surf_p | W&B id |
|---|---:|---|
| Control (lr=5e-4 + warmup) | **130.43** (ep 12) | n8y9yy70 |
| Main (lr=1e-3 + warmup)    | 142.17 (ep 14) | r439zxf5 |

- Negative result on the lr bump (+9% worse — high LR never anneals inside the timeout).
- **Positive result on warmup itself**: control beats current baseline (slice_num=128, no warmup) at 139.83 by ~6.7%, despite running on the *older* slice_num=64 setup. Strong implication that warmup composes additively.
- Cannot merge as-is: branch diff would revert slice_num 128→64, change Config.lr default 5e-4→1e-3, AND add the (good) warmup block. Sent back asking for rebase onto advisor + revert lr default + one re-run on slice_num=128 + warmup + lr=5e-4 to confirm composition.

## 2026-04-27 23:54 — PR #334: Deeper Transolver (n_layers 5→8) — **CLOSED**
- Branch: `willowpai2d5-edward/deeper-l8` (deleted)
- Hypothesis: deeper hierarchy of slice tokens lifts `val_avg/mae_surf_p` ~5-10%
- Result: `val_avg/mae_surf_p = 152.24` at epoch 8 of 9 completed (timeout). Test corrected (post-hoc nan_to_num): 145.87.
- W&B run: `deeper_l8` / sfyn75sq
- Decision: **closed** — clearly worse than slice_num=128 contemporary (152.24 vs 139.83), and slow per-epoch (~205 s vs ~135 s baseline) eats the cosine schedule before it can decay. Student's own analysis correctly recommends against pursuing depth alone.
- **Major bonus:** Edward diagnosed the cross-cutting `data/scoring.py` NaN-poisoning bug (`NaN * 0.0 = NaN` defeats per-sample skip mask). Cruise test sample 000020 has 761 NaN values in p-channel of `y`. Filed as PR #375 (advisor-authorized exception to read-only contract for `data/scoring.py`).

## 2026-04-27 23:54 — PR #336: More physics slices (slice_num 64→128) — **MERGED**
- Branch: `willowpai2d5-fern/more-slices-128` (squash-merged into advisor)
- Hypothesis: doubling slice tokens lifts `val_avg/mae_surf_p` ~3-7%, biggest gain on cruise (largest meshes)
- Result: `val_avg/mae_surf_p = **139.83**` at epoch 10 of 11 completed (timeout)
- W&B run: `slices_128` / 8xow4ge3 (group `capacity_slices`)
- Per-split val mae_surf_p: single 179.11 / camber_rc 144.31 / camber_cruise 110.05 / re_rand 125.87
- 0.67M params (no extra params from slice_num — only changes attention shape), peak VRAM 54.5 GB
- Decision: **merged** — best round-1 reviewable val so far, one-line change, low complexity. Establishes round 1 baseline empirically.
- Caveat: undertrained (11/50 epochs); val curve still descending. Subsequent winners will compound on top.
- `test_avg/mae_surf_p` = NaN due to scoring bug; 3-finite-split mean = 142.79.

## 2026-04-27 23:18 — PR #331: Wider Transolver (n_hidden 128→192, n_head 4→6) — **SENT BACK**
- Branch: `willowpai2d5-askeladd/wider-h192-h6`
- Hypothesis: 2.2× wider Transolver lifts `val_avg/mae_surf_p` ~5-10%
- Status: sent back — undertrained (9/50 epochs, timeout-capped) **and** test_geom_camber_cruise pressure NaN; not mergeable as-is, direction still promising

### Best results so far (under-trained, bs=4, 9 epochs, W&B `wider_h192_h6` / x54plqj1)

| Split | mae_surf_p | mae_vol_p |
|---|---:|---:|
| val_single_in_dist | 209.380 | 189.063 |
| val_geom_camber_rc | 168.777 | 158.080 |
| val_geom_camber_cruise | 109.090 | 105.050 |
| val_re_rand | 128.798 | 122.120 |
| **val_avg** | **154.011** | 143.578 |
| test_single_in_dist | 196.541 | 170.615 |
| test_geom_camber_rc | 150.876 | 144.267 |
| test_geom_camber_cruise | **NaN** | **NaN** |
| test_re_rand | 127.227 | 122.295 |
| **test_avg** | **NaN** | **NaN** |

### Analysis
- Val curve still falling steeply at termination (epoch 9 = best, declining ~7 per epoch in last three epochs); 50 epochs of cosine never engaged. Cannot conclude wider vs baseline yet.
- bs=8 follow-up OOMed at 94.97 GB (peak at bs=4 already 63 GB).
- **Test NaN root cause:** `accumulate_batch` skip-mask uses `torch.isfinite(y)`, not predictions; an extreme pred² overflow in fp32 normalized space (single cruise test sample) propagated NaN into the per-channel surface MAE. Vol_loss=inf logged on the same split is the smoking gun. Affects every PR's test scoring potentially — flagged for round-2 hardening.
- Wider config measured at 1,447,521 params; **actual baseline is 662,359** (not the ~1.4M placeholder I wrote). `BASELINE.md` updated.

### Action
Sent back with: bf16 autocast + fp32 cast before squaring loss, defensive `torch.nan_to_num` in `evaluate_split` (NOT `data/scoring.py`), `--batch_size 8`. Same `--wandb_group capacity_width`. PR remains open.
