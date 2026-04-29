# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1258 — lr=1.5e-4 finer LR sweep on full FiLM+Fourier+warmup+T_max=100 config (charliepai2f3-nezuko)

**Primary metric:** `val_avg/mae_surf_p = 33.1552`

**Configuration:** Lion optimizer (lr=1.5e-4) + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=100, single full-cycle decay) + 5-epoch linear warmup (start_factor=0.0333) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + epochs=100 + Fourier positional encoding on (x,z) with freqs=(1,2,4,8,16,32,64) + FiLM global conditioning (scale+shift per TransolverBlock conditioned on Re/AoA/NACA regime vector, DiT/AdaLN-Zero init)

**Note:** Training cut at ep66/100 by 30-min wall-clock timeout — model still strictly improving at cutoff (LR ~0.35× peak = 5.25e-5 at ep66). Lion sign-based optimizer benefits from lower peak LR (1.5e-4 vs 3e-4): smaller, less noisy parameter updates over entire wall-clock budget. Consistent improvements across all 4 val splits and all 4 test splits. The optimal Lion LR on this schedule appears to be below 2e-4; frieren's lr=2e-4 result (PR #1250) still pending.

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 32.1133 |
| val_geom_camber_rc | 47.2012 |
| val_geom_camber_cruise | 17.1896 |
| val_re_rand | 36.1165 |
| **val_avg** | **33.1552** |

**Test split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 27.8899 |
| test_geom_camber_rc | 43.2971 |
| test_geom_camber_cruise | 14.3845 |
| test_re_rand | 26.8917 |
| **test_avg** | **28.1158** |

**Training:** ~30 min (wall-clock timeout at ep66/100), best epoch 66 (still improving), batch_size=4, n_params: 252,487

**Metrics path:** `target/models/model-charliepai2f3-nezuko-lr-1p5e-4-tmax100-warmup5-100ep-20260429-184943/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 True --surf_weight 28.0 --optimizer lion --lr 1.5e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 100 --warmup_epochs 5 --warmup_start_factor 0.0333 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4 --epochs 100 --fourier_pos_enc --fourier_freqs 1 2 4 8 16 32 64
```

Note: Halving LR from 3e-4 to 1.5e-4 improves generalization with Lion sign-based optimizer. All 4 val splits and all 4 test splits improved. Largest gains on val_geom_camber_cruise (−6.77%) and test_single_in_dist (−5.87%). Best epoch=66 was the final epoch reached (wall-clock limited) — model was still improving.

## Merge History

### 2026-04-29 — PR #1258: lr=1.5e-4 finer LR sweep on full FiLM+Fourier+warmup+T_max=100 config (charliepai2f3-nezuko)
- Previous: `val_avg/mae_surf_p = 34.3851` (PR #1226, lr=3e-4)
- New best: `val_avg/mae_surf_p = 33.1552` (improvement: −1.2299, −3.58%)
- Test: `test_avg/mae_surf_p = 28.1158` (improvement vs previous test_avg 29.0050: −0.8892, −3.07%)
- Student: charliepai2f3-nezuko
- Key finding: Halving LR from 3e-4 to 1.5e-4 improves generalization with Lion sign-based optimizer. Sign-based optimizer benefits from lower peak LR: smaller, less noisy parameter updates over entire wall-clock budget. All 4 val splits and all 4 test splits improved. Largest gains on val_geom_camber_cruise (−6.77%) and test_single_in_dist (−5.87%). Best epoch=66 (wall-clock limited) — model still improving at cutoff.

### 2026-04-29 — PR #1226: Extended training 100ep + T_max=100 + warmup=5 on FiLM+Fourier baseline (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 35.8406` (PR #1208, T_max=75, 75ep, no warmup)
- New best: `val_avg/mae_surf_p = 34.3851` (improvement: −1.4555, −4.06%)
- Test: `test_avg/mae_surf_p = 29.0050` (improvement vs previous test_avg 29.9226: −0.9176, −3.07%)
- Student: charliepai2f3-frieren
- Key finding: T_max=100 (single full-cycle cosine) with 5-epoch warmup extends productive training deeper into the wall-clock budget — LR still ~0.35× peak at ep66 cutoff. All 4 val splits and all 4 test splits improved. test_avg now at 29.005.

### 2026-04-29 — PR #1208: Extended training 75ep + T_max=75 on FiLM+Fourier baseline (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 37.0739` (PR #1175, FiLM+Fourier+warmup, T_max=45, 50 epochs)
- New best: `val_avg/mae_surf_p = 35.8406` (improvement: −1.2333, −3.33%)
- Test: `test_avg/mae_surf_p = 29.9226` (improvement vs previous test_avg 31.3474: −1.4248, −4.55%)
- Student: charliepai2f3-frieren
- Key finding: T_max=75 (single full-cycle cosine) is the primary driver — at ep49 alone this yielded val_avg=37.21 (−6.86% vs PR #1104 baseline), confirming LR horizon matters more than epoch count. Wall-clock timeout cut at ep57/75; model still improving steeply. Best epoch = 57 = last epoch, no plateau. All 4 val splits improved. Test_avg broke below 30 for the first time.

### 2026-04-29 — PR #1175: LR warmup + single-decay cosine on FiLM+Fourier baseline (charliepai2f3-thorfinn)
- Previous: `val_avg/mae_surf_p = 39.9450` (PR #1104, FiLM+Fourier, T_max=15 multi-cycle, no warmup)
- New best: `val_avg/mae_surf_p = 37.0739` (improvement: −2.8711, −7.19%)
- Test: `test_avg/mae_surf_p = 31.3474` (improvement vs previous test_avg 33.5761: −2.2287, −6.64%)
- Student: charliepai2f3-thorfinn
- Key finding: 5-epoch linear warmup (start_factor=1/30 → 3e-4) stabilizes Lion optimizer initialization. Single-decay cosine (T_max=45) avoids LR restarts that bounce model out of good basins. Best epoch=50 — model still improving at timeout, indicating more training could help. Both val and test improved across all 4 splits. Run 1 (warmup=10, multi-cycle T_max=15) also beat baseline at val_avg=38.9454, confirming warmup is the key driver.

### 2026-04-29 — PR #1196: Single-decay cosine schedule (T_max=50) on Fourier pos enc baseline (charliepai2f3-frieren)
- Context: Tested against PR #1148 baseline (val_avg=43.9575); current best is PR #1104 (val_avg=39.9450 with FiLM)
- Result: `val_avg/mae_surf_p = 42.4863` (vs stale baseline −1.4712, −3.35%; vs current best: +2.5413, regresses from 39.9450)
- Test: `test_avg/mae_surf_p = 35.6687`
- Key finding: Single-decay cosine (T_max=50) confirms the LR-cycling failure mode. This improvement on the non-FiLM branch is real but the FiLM path (PR #1104) is already significantly ahead. The T_max=50 improvement should be adopted in the FiLM branch experiments.
- Note: Merged against stale baseline — current best remains PR #1104 at val_avg=39.9450.

### 2026-04-29 — PR #1104: FiLM global conditioning: inject Re/AoA/NACA via scale+shift (charliepai2f3-edward)
- Previous: `val_avg/mae_surf_p = 43.9575` (PR #1148, Fourier freqs=(1,2,4,8,16,32,64))
- New best: `val_avg/mae_surf_p = 39.9450` (improvement: −4.0125, −9.13%)
- Test: `test_avg/mae_surf_p = 33.5761` (improvement vs previous test_avg 37.4541: −3.8780, −10.35%)
- Student: charliepai2f3-edward
- Key finding: FiLM global conditioning (scale+shift per TransolverBlock conditioned on 11-dim physics regime vector) delivers a decisive −9.13% improvement on val and −10.35% on test. Best epoch 49/50 — model not yet converged; extended training is the obvious next direction. All 4 val splits improved. DiT/AdaLN-Zero initialization key to stability. Param count increases from 184,903 to 252,487 (+67,584, +36.5%).

### 2026-04-29 — PR #1148: Extended Fourier freqs on (x,z): freqs=(1,2,4,8,16,32,64) (charliepai2f3-askeladd)
- Previous: `val_avg/mae_surf_p = 44.4154` (PR #1106, Fourier pos enc freqs=(1,2,4,8,16))
- New best: `val_avg/mae_surf_p = 43.9575` (improvement: −0.4579, −1.03%)
- Student: charliepai2f3-askeladd
- Key finding: freqs=(1,2,4,8,16,32,64) beats baseline; adding freq=128 regresses (Nyquist aliasing near mesh resolution)

### 2026-04-29 — PR #1106: Fourier positional encoding on (x,z) (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 47.3987` (PR #1093, compound baseline)
- New best: `val_avg/mae_surf_p = 44.4154` (improvement: −2.9833, −6.29%)
- Student: charliepai2f3-frieren
- Also included: NaN fix for test_geom_camber_cruise (non-finite GT entries in sample 20 masked via y_finite guard in evaluate_split)

### 2026-04-29 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- Previous: `val_avg/mae_surf_p = 47.7385` (charlie-pai2e-r5 reference)
- New best: `val_avg/mae_surf_p = 47.3987` (improvement: −0.3398)
- Student: charliepai2f3-alphonse
