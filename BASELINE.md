# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1104 — FiLM global conditioning: inject Re/AoA/NACA via scale+shift (charliepai2f3-edward)

**Primary metric:** `val_avg/mae_surf_p = 39.9450`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=15) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + Fourier positional encoding on (x,z) with freqs=(1,2,4,8,16,32,64) + FiLM global conditioning (scale+shift per TransolverBlock conditioned on Re/AoA/NACA regime vector, DiT/AdaLN-Zero init)

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 38.3034 |
| val_geom_camber_rc | 56.1374 |
| val_geom_camber_cruise | 22.9918 |
| val_re_rand | 42.3473 |
| **val_avg** | **39.9450** |

**Test split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 32.0588 |
| test_geom_camber_rc | 49.6238 |
| test_geom_camber_cruise | 18.9013 |
| test_re_rand | 33.7205 |
| **test_avg** | **33.5761** |

**Training:** ~22.8 min, 50 epochs (best epoch 49), batch_size=4, Peak VRAM: 9.89 GB, n_params: 252,487

**Metrics path:** `target/models/model-charliepai2f3-edward-film-extended-fourier-rebased-20260429-140420/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 True --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4 --fourier_pos_enc --fourier_freqs 1 2 4 8 16 32 64
```

Note: FiLM conditioning injects global physics scalars (dims 13–23 of x: log(Re), AoA1, NACA1(3d), AoA2, NACA2(3d), gap, stagger) as learned scale+shift (γ, β) applied to each TransolverBlock. The FiLMConditioner uses DiT/AdaLN-Zero initialization so all blocks start as identity and FiLM activates gradually. The Fourier positional encoding appends sin(f*pi*xy) and cos(f*pi*xy) for freqs in (1,2,4,8,16,32,64), expanding spatial input from 2-dim to 30-dim. Input feature dim becomes 52. Adding freq=128 regresses (Nyquist aliasing).

## Merge History

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
