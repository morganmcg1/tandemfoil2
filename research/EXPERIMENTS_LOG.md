# SENPAI Research Results — icml-appendix-charlie-pai2d-r4

## 2026-04-27 23:50 — PR #287: surf_weight 10 -> 25 to refocus loss on surface MAE
- Branch: `charliepai2d4-alphonse/surf-weight-up` (deleted on merge)
- Student: charliepai2d4-alphonse
- **Outcome: MERGED (squash, commit e4a0c18). First baseline on this branch.**
- Hypothesis: Raise surf_weight to direct gradient toward surface error. Predicted Δ -3% to -7%.

### Headline metrics (epoch 14 of 50, timeout-capped at 30 min)
| Metric | Value |
|---|---|
| `val_avg/mae_surf_p`  | **126.67** |
| `test_avg/mae_surf_p` | **114.88** (corrected; 1 NaN-y test sample skipped) |
| Wall-clock | 30.8 min train + 20s test |
| Peak GPU memory | 42.1 GB / 96 |

### Per-split val
| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist     | 155.79 | 172.08 |
| val_geom_camber_rc     | 134.23 | 147.34 |
| val_geom_camber_cruise |  98.89 | 131.81 |
| val_re_rand            | 117.77 | 133.50 |

### Per-split test (post-NaN-fix)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 136.43 |
| test_geom_camber_rc     | 124.14 |
| test_geom_camber_cruise |  83.63 |
| test_re_rand            | 115.33 |

### Analysis
- **Curve was monotonically descending through epoch 14** (256.83 → 126.67), and the last 3 epochs (153.65 → 140.48 → 126.67) all set new bests. Run was meaningfully under-trained — would likely drop further if the cosine tail completed.
- **Volume MAE not pathologically harmed** (vol_p ≈ 146 mean across val). The surf_weight bump didn't starve the volume branch.
- **Test < val**: 114.88 < 126.67. The cruise test split is ~15 pts better than its val counterpart. Plausibly small-val-set noise (val 100 vs test 200), but worth watching as more PRs land.
- **Important meta-observation from alphonse**: round 1 is effectively a **14-epoch** ranking exercise rather than 50-epoch. Every comparison this round inherits this caveat. BASELINE.md updated to flag.
- Independent diagnosis of the scoring NaN bug, exactly matching edward's earlier finding.

JSONL summary: `research/EXPERIMENT_METRICS.jsonl` (PR=287 records, 16 lines).

## 2026-04-27 23:30 — PR #358: Maintenance: fix data/scoring.py NaN propagation through inf*0 mask
- Branch: `charliepai2d4-edward/fix-scoring-nan-mask` (deleted on merge)
- Student: charliepai2d4-edward
- **Outcome: MERGED (squash, commit 010235e).**

Maintenance fix, not an experiment. Replaces `err * mask_float` with `torch.where(mask, err, 0)` in `data/scoring.py::accumulate_batch`. New `data/test_scoring.py` with 4 tests covers: inf-in-p (reproduces test_geom_camber_cruise sample 20 failure mode), NaN-in-y, bit-equality on all-finite inputs (no-op proof), end-to-end finalized-MAE finiteness. All 4 pass. Empirical OLD vs NEW on the same inf-injected batch confirms the fix. Diff: 3 additions / 2 deletions in scoring.py + 104-line test file.

This unblocks `test_avg/mae_surf_p` for every other in-flight PR — they will now produce a finite test ranking metric. Existing in-flight branches will need to either rebase to pick up this fix, or accept that `test_geom_camber_cruise/mae_surf_p` will remain NaN until the scoring change reaches their runtime.

## 2026-04-27 23:20 — PR #300: Wider Transolver: n_hidden 128->192, slice_num 64->96
- Branch: `charliepai2d4-edward/wider-model` (deleted on close)
- Student: charliepai2d4-edward
- Hypothesis: Increase Transolver capacity (n_hidden 128→192, slice_num 64→96, ~1.48 M params) on the under-utilized 96 GB GPU. Predicted Δ on `val_avg/mae_surf_p`: -5% to -10% vs. published baseline.
- **Outcome: CLOSED** (under-trained, not directly comparable to baseline).

### Best validation metrics (epoch 9 of 50; training stopped at 30-min cap)
| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist     | 193.90 | 2.510 | 0.982 | 174.91 |
| val_geom_camber_rc     | 157.52 | 3.349 | 1.300 | 147.12 |
| val_geom_camber_cruise | 104.49 | 2.052 | 0.669 |  99.36 |
| val_re_rand            | 125.49 | 2.583 | 0.937 | 118.76 |
| **val_avg**            | **145.35** | 2.624 | 0.972 | 135.04 |

### Test metrics (best ckpt)
| split | mae_surf_p |
|---|---|
| test_single_in_dist     | 169.31 |
| test_geom_camber_rc     | 139.23 |
| test_geom_camber_cruise | NaN ← *scoring bug, see below* |
| test_re_rand            | 125.83 |
| **test_avg (3 valid)**  | **144.79** |

JSONL summary: `research/EXPERIMENT_METRICS.jsonl` (PR=300 records).

### Analysis
- **Run hit `SENPAI_TIMEOUT_MINUTES=30` cap at end of epoch 9** (~205 s/epoch). Validation was still improving sharply (epoch 8 → 9 dropped 16 pts on `val_avg/mae_surf_p`); the model had not converged. We can't compare a 9-epoch result to a hypothetical 50-epoch baseline.
- **No overfitting yet** — train loss still well above val, consistent with under-training. The held-out camber gap (rc=157.5, cruise=104.5) wasn't catastrophic.
- **Peak GPU memory: 63 GB / 96 GB**, no OOM.
- **Critical bug found in `data/scoring.py`**: sample 20 of `test_geom_camber_cruise` has 761 inf values in the p channel of ground-truth y. The masking logic uses `error * mask` (float-mask multiply), and `inf * 0 = NaN` in IEEE-754, poisoning the float64 accumulator for that channel. Every wide- or deep-model experiment on this branch will hit the same NaN.
- Decision rationale: closing rather than re-running; assigning edward to fix the scoring bug next as the higher-leverage action. A more conservative widening (n_hidden=144-160) can be revisited once we have a fully-trained baseline number on this branch from one of the cheap in-flight experiments (alphonse, askeladd).
