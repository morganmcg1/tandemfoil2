# SENPAI Research Results — willow-pai2d-r5

Per-PR experiment log. New entries are appended chronologically; the latest entries are at the top.

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
