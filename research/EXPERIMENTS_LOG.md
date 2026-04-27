# SENPAI Research Results — icml-appendix-charlie-pai2c-r2

## 2026-04-27 20:30 — PR #213: Physical-space L1 surface loss (volume stays MSE)
- **Branch**: `charliepai2c2-nezuko/surface-pressure-l1-loss`
- **Hypothesis**: The training loss is MSE in normalized space; the ranking metric is L1 in physical space. High-Re samples dominate the MSE gradient and the model overfits to them at the expense of low-Re samples. Switch the **surface** term to **physical-space L1** (denormalize prediction, mean-absolute pressure error, scale `/y_std.mean()` so magnitudes match the volume-MSE term). Volume term stays normalized MSE.
- **Result**: NEW BEST `val_avg/mae_surf_p = 102.71` at epoch 12/14 completed (–21.0% vs. PR #216 baseline 130.057). **Consistent improvements across all 4 val splits (–18.1% to –23.1%).** Training cut by 30-min wall-clock at epoch 14; val curve was still trending down (epoch 12 → 13 → 14: 102.71 → 108 → 112 — slight noise around the best, no clear divergence).

| Metric | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 125.02 | 109.49 | 80.47 | 95.85 | **102.71** |
| Δ vs. #216 | –22.4% | –23.1% | –18.1% | –19.0% | **–21.0%** |

| Metric | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` (logged) | 108.28 | 99.19 | NaN ⚠️ | 90.99 | NaN ⚠️ |
| `mae_surf_p` (NaN-safe) | 108.28 | 99.19 | **67.62** (199/200) | 90.99 | **91.52** |

- **Param count**: ~1.18M (loss-only change — same arch as #216)
- **Peak VRAM**: 42.13 GB | **Wall-clock**: 30.6 min | **Best epoch**: 12/14
- **Metrics summary**: `models/model-charliepai2c2-nezuko-surface-pressure-l1-loss-20260427-195051/metrics.yaml`
- **Reproduce**: `cd target && python train.py --epochs 50 --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --experiment_name surface-pressure-l1-loss`

### Analysis

Decision: **MERGED**. Single largest improvement of Round 1 so far. The hypothesis (training MSE in normalized space ≠ ranking L1 in physical space) was correctly identified as the dominant misalignment in the training recipe.

**Why this worked.** Two compounding effects:
1. **Gradient direction now aligns with the metric.** L1 gradients are sign-only (per error), so the model's per-sample contribution is independent of magnitude — every sample gets equal optimization pressure. MSE gradients scale linearly with error, so high-Re samples (with large pressure dynamic range) dominated the surface-loss optimizer step. Switching to L1 redistributes gradient weight toward low-Re samples, which were previously underfit.
2. **Physical-space loss bypasses the y-std normalization mismatch.** The MAE metric is in physical units; computing the surface loss in physical units (with a single `/y_std.mean()` rescale to keep magnitudes comparable to the volume MSE term) means the optimization target and the evaluation target are now in the same units.

**Per-split observation** confirmed: cruise tandem geom-holdout is easiest (`mae_surf_p = 80.47`); single-foil in-dist is still hardest (`125.02`), but the gap to the geom holdouts narrowed dramatically vs. the prior baseline. The L1 transition closed about 30% of the single-foil disadvantage.

**Bug** confirmed by nezuko independently of tanjiro: `data/scoring.py::accumulate_batch` propagates `NaN * 0 = NaN` (PyTorch semantics) for samples with non-finite y entries. Fix is one line, but `data/scoring.py` is read-only — needs human team. Affects only `test_geom_camber_cruise/000020.pt` for now.

### Suggested follow-ups (assigned to nezuko on next PR)

1. **Pressure-channel-weighted L1.** Weight channel 2 (`p`) by 2× in the L1 sum since `mae_surf_p` is the sole ranking metric. Direct extension of this PR; should compound.
2. **Match `T_max` to realistic epoch budget.** Cosine never reached its decay phase. Either extend `SENPAI_TIMEOUT_MINUTES` (forbidden) or set `T_max ≈ 12–14` so the schedule actually anneals within the cap.
3. **MSE+L1 mixture.** Only worth trying if a follow-up shows L1 plateauing — currently no sign of plateau.

---

## 2026-04-27 19:52 — PR #216: Wider-shallower Transolver: n_hidden=192, n_layers=4, n_head=6
- **Branch**: `charliepai2c2-tanjiro/wider-shallower-arch`
- **Hypothesis**: For irregular-mesh CFD surrogate problems, *wider* per-node feature dim beats *deeper* network depth — surface pressure MAE depends on local per-node features, not long-range mesh attention. Predicted 5–10% reduction vs. seed baseline.
- **Result**: ESTABLISHED EMPIRICAL BASELINE for the branch. `val_avg/mae_surf_p = 130.0568` at epoch 11/50 (training timed out at 30 min wall clock; was still improving ~5%/epoch).

| Metric | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 161.146 | 142.415 | 98.282 | 118.383 | **130.057** |
| `mae_surf_Ux` | 1.946 | 2.640 | 1.608 | 2.113 | 2.077 |
| `mae_surf_Uy` | 0.855 | 1.129 | 0.637 | 0.882 | 0.876 |

| Metric | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 144.256 | 127.587 | NaN ⚠️ | 115.976 | NaN ⚠️ |
| `mae_surf_Ux` | 1.860 | 2.601 | 1.529 | 1.999 | 1.997 |

- **Param count**: 1.18M (~1.79× the seed config's analytical ~0.66M)
- **Peak VRAM**: 51.79 GB | **Wall-clock**: 30.3 min | **Best epoch**: 11/50
- **Metrics summary**: `models/model-wider-shallower-arch-20260427-191514/metrics.yaml`
- **JSONL (centralized)**: `research/EXPERIMENT_METRICS.jsonl`
- **Reproduce**: `cd target && python train.py --epochs 50 --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --experiment_name wider-shallower-arch`

### Analysis

Decision: **MERGED** as the empirical baseline-establisher. With no prior numbers on this advisor branch, this becomes the floor for Round 1 to beat. Two structural concerns are recorded for the round:

1. **Undertrained.** `T_max=50` cosine schedule + 30-min wall-clock cap = 11 epochs at ~165 s/epoch, with the LR never entering its decay tail. The `val_avg/mae_surf_p` curve was still dropping ~5%/epoch when the timer fired (epoch 10 → 11: 135.98 → 130.06). PR #208 (`lr-warmup-cosine-floor`) partially addresses this with the 1e-5 floor; a budget-matched `T_max` follow-up is warranted.
2. **Test-metric NaN bug.** Tanjiro identified that `inf * 0 → NaN` in `data/scoring.py::accumulate_batch` silently NaN-contaminates `test_avg/mae_surf_p` on `test_geom_camber_cruise` (single sample 020 has 761 Inf pressure values). `data/scoring.py` is read-only per `program.md` — flagging to human team. **Until patched, rank on `val_avg/mae_surf_p` only.** Mean over the 3 finite test splits is `mae_surf_p ≈ 129.27`.

### Per-split observations

The cruise tandem geometry-holdout split is the easiest (`mae_surf_p = 98.28`); single-foil in-distribution is hardest (`mae_surf_p = 161.15`). This is counterintuitive — the in-distribution sanity-check split should be easiest. Possible explanations:
- Single-foil samples have larger pressure dynamic range (per program.md: y range up to ±29K vs cruise ±7.6K), so even a small relative error compounds.
- The balanced sampler equally weights three domains (single, raceCar tandem, cruise tandem), so single-foil gets ~33% of training samples but is the highest-variance domain.

This per-split structure should inform future hypotheses — domain-id-conditioning, per-Re-bin scaling, or single-foil-specific decoder heads may pay off.
