# SENPAI Research Results

## Branch: icml-appendix-charlie-pai2c-r3

## 2026-04-27 21:00 — PR #193: Vanilla baseline anchor — establish reference metrics

- **Branch**: charliepai2c3-alphonse/vanilla-baseline-anchor
- **Hypothesis**: Run the default Transolver (no code changes) as a reference anchor. Reveals the true baseline performance (with EMA already enabled in train.py) so future experiments have a clean comparison point.
- **Results**:

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 162.74 | 2.39 | 0.91 | 168.85 | 6.12 | 2.38 |
| val_geom_camber_rc     | 141.71 | 2.95 | 1.20 | 140.96 | 6.09 | 3.04 |
| val_geom_camber_cruise | 107.43 | 1.67 | 0.66 | 103.37 | 4.40 | 1.64 |
| val_re_rand            | 114.97 | 2.20 | 0.93 | 114.78 | 4.99 | 2.22 |
| **val avg**            | **131.71** | **2.30** | **0.92** | **131.99** | **5.40** | **2.32** |
| test_single_in_dist    | 153.51 | 2.25 | 0.90 | 157.11 | 5.64 | 2.26 |
| test_geom_camber_rc    | 127.75 | 2.83 | 1.14 | 129.58 | 5.89 | 2.91 |
| test_geom_camber_cruise| NaN | 1.72 | 0.60 | NaN | 4.34 | 1.51 |
| test_re_rand           | 116.76 | 2.11 | 0.89 | 114.90 | 4.92 | 2.15 |
| **test avg**           | **NaN** | **2.23** | **0.88** | **NaN** | **5.20** | **2.21** |

- **Metric summary**: `models/model-vanilla-baseline-anchor-20260427-194339/metrics.jsonl`
- **Best epoch**: 11 of 14 (timed out at 30-min wall clock, ~132 s/epoch)
- **Peak VRAM**: 42.11 GB
- **Params**: 662,359 (~0.66M)

**Commentary**: Vanilla baseline beats the EMA-only run (#209) at 131.71 vs 133.66. This reveals that EMA alone (decay=0.999) doesn't help much — the default train.py already includes EMA, so both runs had EMA. The real baseline with EMA is 131.71. Test NaN on test_geom_camber_cruise is expected (pre-NaN-fix run; val metrics unaffected). Best checkpoint was at epoch 11, with validation oscillating in epochs 12-14 due to cosine schedule still being far from T_max=50.

**Key insight**: EMA is always-on in the base train.py — "EMA experiment" PR #209 and "vanilla" PR #193 both used EMA. The val difference (133.66 vs 131.71) may be noise or batch randomness. Both runs timed out at 14 epochs of 50 configured.

**Decision**: MERGED. Establishes the true baseline at val_avg/mae_surf_p=131.71. Updates BASELINE.md.

---

## 2026-04-27 20:18 — PR #209: EMA weight averaging (decay=0.999) — smoother generalization

- **Branch**: charliepai2c3-nezuko/ema-weight-averaging
- **Hypothesis**: EMA of model weights (decay=0.999) maintained throughout training and used for all validation, checkpoint selection, and test evaluation. Expected 0.5–2% improvement with zero extra compute at training time, especially for OOD camber splits.
- **Results**:

| split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |
|---|---:|---:|---:|---:|---:|---:|
| val_single_in_dist     | 171.74 | 2.02 | 0.88 | 162.44 | 5.95 | 2.49 |
| val_geom_camber_rc     | 146.90 | 3.18 | 1.12 | 143.01 | 6.39 | 3.07 |
| val_geom_camber_cruise | 100.14 | 1.46 | 0.60 | 100.80 | 4.28 | 1.56 |
| val_re_rand            | 115.87 | 2.17 | 0.84 | 114.32 | 5.11 | 2.18 |
| **val avg**            | **133.66** | **2.21** | **0.86** | **130.14** | **5.43** | **2.32** |
| test_single_in_dist    | 143.91 | 1.92 | 0.81 | 140.24 | 5.60 | 2.26 |
| test_geom_camber_rc    | 132.09 | 3.05 | 1.06 | 130.20 | 6.22 | 2.93 |
| test_geom_camber_cruise|  85.50 | 1.32 | 0.55 |  88.35 | 4.11 | 1.43 |
| test_re_rand           | 116.84 | 1.99 | 0.83 | 116.30 | 4.91 | 2.10 |
| **test avg**           | **119.58** | **2.07** | **0.82** | **118.77** | **5.21** | **2.18** |

- **Metric summary**: `target/models/model-charliepai2c3-nezuko-ema-weight-averaging-20260427-192048/metrics.jsonl`
- **Epochs run**: 14/50 (timed out at 30-min wall clock, ~132 s/epoch, still improving monotonically)
- **Peak VRAM**: 42.11 GB

**Commentary**: This is the first result on the icml-appendix-charlie-pai2c-r3 track, so it establishes the baseline. val_avg/mae_surf_p=133.66 at epoch 14 with a monotonically decreasing trajectory — had the run not timed out, it would have continued improving. The OOD camber-cruise split performs significantly better (100.14) than the in-dist split (171.74), which is a surprising and encouraging sign.

**Critical bug fix (merged independently)**: The student identified a NaN-poisoning bug in `evaluate_split` caused by 1 sample in `test_geom_camber_cruise` having 761 NaN GT pressure values. The original `data/scoring.py` intended to mask these out, but `(NaN * 0) = NaN` in IEEE arithmetic caused the entire split accumulator to become NaN. Fix: sanitize GT before multiplication, proactively zero padding-position predictions. This is a no-op on all val splits and 3/4 test splits. Fix is now part of `train.py` on the advisor branch.

**Decision**: MERGED. First result establishes baseline. Bug fix is critical and now protects all future experiments.
