# SENPAI Research Results — icml-appendix-charlie-pai2d-r4

## 2026-04-28 00:40 — PR #304: Deeper Transolver: n_layers 5->8 with DropPath 0.1
- Branch: `charliepai2d4-fern/deeper-model-droppath` (deleted on close)
- Student: charliepai2d4-fern
- **Outcome: CLOSED** (per-epoch wall-clock too high → only 9/50 epochs; worse than baseline at equal-epoch).

### Headline (epoch 8 of 9 completed, timeout-capped)
| Metric | Value | vs. baseline |
|---|---|---|
| `val_avg/mae_surf_p` | 159.62 | +50% vs PR #308 (106.40) |
| `val_avg/mae_surf_p` (epoch 8 equal-budget) | 159.62 | +11% vs nezuko #308 epoch 8 (143.61, online weights) |
| `test_avg/mae_surf_p` | NaN¹ → 163.23 (3 splits) | — |
| Per-epoch time | **210 s** | vs nezuko 141 s — **~1.5× slower** |
| Peak GPU memory | 64.5 GB | within budget |

¹ Branch predates PR #358 scoring fix; same inf-y sample 20 in test_geom_camber_cruise.

### Per-split val (epoch 8)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     | 199.65 |
| val_geom_camber_rc     | 178.49 |
| val_geom_camber_cruise | 118.37 |
| val_re_rand            | 141.97 |

### Analysis
- **Depth + DropPath integration is healthy** — no NaN, no divergence, val curve descended monotonically through epoch 8.
- **Throughput is the kill criterion (again)**. Same lesson as edward's #300 (wider) and tanjiro's #309 (more slices): per-epoch cost above ~150s makes experiments uncompetitive on absolute val_avg in the 30-min cap.
- **Worse at equal-epoch**: fern epoch 8 (159.62) vs nezuko epoch 8 online (143.61) — depth doesn't even win where wall-clock is matched.
- **Independent diagnosis** of the inf-y bug — sixth student to hit it. All resolved by PR #358.
- **Parking depth as round-2** — once throughput-recovery PRs land (#372 bf16, #382 larger batch), n_layers=8 fits more comfortably.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=304 records, 10 lines).

## 2026-04-28 00:25 — PR #308: EMA (decay 0.999) + grad clip max_norm 1.0 — **NEW BASELINE**
- Branch: `charliepai2d4-nezuko/ema-grad-clip` (deleted on merge)
- Student: charliepai2d4-nezuko
- **Outcome: MERGED (squash, commit 5bdb284). NEW BASELINE: val_avg/mae_surf_p = 106.40, -16.2% vs PR #287.**

### Headline metrics (epoch 13 of 14, EMA-evaluated)
| Metric | Value | vs #287 baseline | vs published |
|---|---|---|---|
| `val_avg/mae_surf_p` (EMA) | **106.40** | **-16.2%** | best on branch |
| `test_avg/mae_surf_p` (EMA) | **93.99**  | -18.2% (vs #287 114.88) | best on branch |
| Wall-clock | ~33 min total | comparable | |
| Peak GPU memory | 42.1 GB | unchanged | |

### Per-split val (epoch 13, EMA)
| Split | mae_surf_p | mae_vol_p |
|---|---|---|
| val_single_in_dist     | 130.44 | 133.99 |
| val_geom_camber_rc     | 119.63 | 120.78 |
| val_geom_camber_cruise |  80.75 |  74.44 |
| val_re_rand            |  94.78 |  91.91 |

### Per-split test (post-fix scoring, EMA)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     | 112.78 |
| test_geom_camber_rc     | 103.87 |
| test_geom_camber_cruise |  66.35 |
| test_re_rand            |  92.98 |

### Analysis
- **Monotonically descending**: 322 → 106.40 across 13 epochs, every epoch a new best. No instability.
- **EMA pays off late**: from epoch 10 onward EMA is consistently 13-20 units better than online weights (nezuko's own diagnostic). Epochs 1-4 the EMA lags online (decay=0.999 warmup).
- **Crucial caveat — clipping is a hidden lr dampener.** `max_norm=1.0` clips 100% of batches; pre-clip gn_mean ≈ 50-100 vs threshold 1.0. So the optimizer is doing essentially unit-norm SGD on top of AdamW. The 16% gain therefore cannot be cleanly attributed to EMA alone. **Ablations queued.**
- **Compounds with #287**: surf_weight=25 (alphonse) and EMA+clip (nezuko) are independent changes; combination is a clear round-2 candidate.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=308 records, 14 lines).

## 2026-04-28 00:20 — PR #307: 5-epoch linear warmup + cosine schedule, peak lr 1e-3
- Branch: `charliepai2d4-frieren/warmup-cosine-1e3` (deleted on close)
- Student: charliepai2d4-frieren
- **Outcome: CLOSED** (val_avg=134.58, ~26% worse than the new baseline 106.40).

### Headline (epoch 13 of 14, timeout-capped)
| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` | 134.58 |
| `test_avg/mae_surf_p` | 123.24 |

### Analysis
- **Warmup itself is stable** — 1e-3 with 5-epoch linear warmup did not diverge; only a small blip at epoch 5 when full lr fires (recovers in one epoch).
- **Cosine never decays in the budget**: T_max=50 means lr only drops 1.0 → 0.924 across 14 epochs. The "warmup→peak→decay" cycle the hypothesis predicted never plays out — this run is essentially constant-peak-lr.
- **Below baseline anyway**: even with stable warmup at 2× peak lr, the run lags the prior baseline (PR #287, 126.67) and far behind the new baseline (PR #308, 106.40).
- Independent diagnosis of the same inf-y test bug (now fixed in PR #358).

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=307 records, 15 lines).

## 2026-04-28 00:05 — PR #310: Per-channel surface loss weights: 3x weight on surface pressure
- Branch: `charliepai2d4-thorfinn/per-channel-surf-weights` (deleted on close)
- Student: charliepai2d4-thorfinn
- **Outcome: CLOSED** (+13.0% regression on val_avg/mae_surf_p).
- Hypothesis: weight surface-p 3× over surface-Ux/Uy in the loss to bias optimization toward the metric. Predicted Δ -3% to -8%.

### Headline metrics (epoch 13/14, both runs timeout-capped at 14 epochs)
| Metric | thorfinn baseline-ref (sw=10, p_w=1) | thorfinn 3x p (sw=10, p_w=3) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 130.54 (epoch 12) | **147.56** (epoch 13) | **+13.0% worse** |
| `val_avg/mae_surf_Ux` | 2.13 | 2.88 | +35.2% |
| `val_avg/mae_surf_Uy` | 0.92 | 1.09 | +18.7% |
| `val_avg/mae_vol_p` | 131.24 | 153.05 | +16.6% |
| `test_avg/mae_surf_p` | NaN¹ | NaN¹ | — |

¹ Branch predates PR #358 scoring fix; same inf-y sample 20 in test_geom_camber_cruise.

### Per-split val (3x − baseline)
| split | mae_surf_p Δ |
|---|---|
| val_single_in_dist | +7.0% |
| val_geom_camber_rc | +36.9% |
| val_geom_camber_cruise | -1.5% |
| val_re_rand | +8.0% |

### Analysis
- **Hypothesis cleanly disproved**: 3 of 4 val splits regressed materially, only `val_geom_camber_cruise` saw a tiny improvement (-1.5%). Surface velocities also degraded — the gradient-mass-starvation story (heavy weight on p starves Ux/Uy learning, which feeds back into pressure via Navier-Stokes coupling) is the most plausible explanation.
- **Useful side data:** thorfinn's own baseline-ref run (surf_weight=10, p_weight=1) gave val_avg=130.54 at epoch 12, **independently confirming** that alphonse's PR #287 (val_avg=126.67 at surf_weight=25) is a ~3% improvement over surf_weight=10 at the same wall-clock.
- **Lesson:** with the existing 10× surf_weight, layering more p-specific weight on top dominates the gradient signal too aggressively in a 14-epoch budget regime. If revisited, try `surf_weight=5, surf_p_weight=2` so total surface-p effective weight matches baseline.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=310 records, 15 lines).

## 2026-04-27 23:55 — PR #309: More slice tokens: slice_num 64->128, n_head 4->8
- Branch: `charliepai2d4-tanjiro/more-slices` (deleted on close)
- Student: charliepai2d4-tanjiro
- **Outcome: CLOSED** (slower per epoch and not better at equal-epoch comparison).
- Hypothesis: doubling slice tokens + heads (with halved head_dim) gives more physical-regime "slots". Predicted Δ -3% to -7%.

### Headline metrics (epoch 6 of 50, timeout-capped at 8 epochs / 33.4 min)
| Metric | Value | vs. alphonse #287 baseline |
|---|---|---|
| `val_avg/mae_surf_p`  | 168.47 (epoch 6) | +33% vs 126.67 — but unfair due to fewer epochs |
| `val_avg/mae_surf_p` (epoch 8 equal-budget) | 170.54 | +4.5% vs alphonse epoch 8 (163.24); alphonse used surf_weight=25 |
| `test_avg/mae_surf_p` | 154.97 (post-fix scoring) | — |
| Per-epoch time | **250 s** | vs alphonse 131 s — **~1.9× slower** |
| Peak GPU memory | 82.3 GB | within budget but high |

### Analysis
- **Throughput is the kill criterion.** At 250 s/epoch only 8 epochs fit the 30-min cap (vs alphonse's 14). Even granting the full architectural advantage at equal-epoch (which doesn't show up — tanjiro is *worse* at epoch 8), the total run is fewer-epoch-and-no-better.
- **Late-epoch instability:** epoch 6→7 jumped 168→197 then 170 at epoch 8. Plausibly head_dim=16 (n_head=8 with n_hidden=128) is too narrow.
- **Independent bug diagnosis** matched edward's earlier finding on `data/scoring.py` (`inf * 0 = NaN`). Tanjiro applied a workaround in `train.py::evaluate_split` to drop bad samples before scoring; the proper fix landed in PR #358.
- **Tanjiro's suggested decomposition** (slice-count alone vs head-count alone) is the right experimental design IF we ever return to this axis. Parking for now since the binding constraint is throughput, not slice count.

JSONL: `research/EXPERIMENT_METRICS.jsonl` (PR=309 records, 9 lines).

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
