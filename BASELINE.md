# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1106 — Fourier positional encoding on (x,z) + NaN fix (charliepai2f3-frieren)

**Primary metric:** `val_avg/mae_surf_p = 44.4154`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=15) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + Fourier positional encoding on (x,z) with freqs=(1,2,4,8,16)

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 45.6222 |
| val_geom_camber_rc | 58.5071 |
| val_geom_camber_cruise | 26.7073 |
| val_re_rand | 46.8250 |
| **val_avg** | **44.4154** |

**Test split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 37.8511 |
| test_geom_camber_rc | 53.2684 |
| test_geom_camber_cruise | 21.5381 |
| test_re_rand | 36.0350 |
| **test_avg** | **37.1732** |

**Training:** ~21.3 min, 50 epochs, batch_size=4, Peak VRAM: 9.32 GB, n_params: 182,855

**Metrics path:** `target/models/model-charliepai2f3-frieren-fourier-pos-enc-rebased-20260429-110704/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4 --fourier_pos_enc
```

Note: The Fourier positional encoding appends `sin(f*pi*xy)` and `cos(f*pi*xy)` for freqs in (1,2,4,8,16) to the (x,z) dims, expanding the spatial input from 2-dim to 22-dim. Input feature dimension becomes 44.

## Merge History

### 2026-04-29 — PR #1106: Fourier positional encoding on (x,z) (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 47.3987` (PR #1093, compound baseline)
- New best: `val_avg/mae_surf_p = 44.4154` (improvement: −2.9833, −6.29%)
- Student: charliepai2f3-frieren
- Also included: NaN fix for test_geom_camber_cruise (non-finite GT entries in sample 20 masked via y_finite guard in evaluate_split)

### 2026-04-29 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- Previous: `val_avg/mae_surf_p = 47.7385` (charlie-pai2e-r5 reference)
- New best: `val_avg/mae_surf_p = 47.3987` (improvement: −0.3398)
- Student: charliepai2f3-alphonse
