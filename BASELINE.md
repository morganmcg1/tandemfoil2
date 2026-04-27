# SENPAI Research Baseline — TandemFoilSet (icml-appendix-willow-pai2-r3)

Primary ranking metric: `val_avg/mae_surf_p` (val) and `test_avg/mae_surf_p` (test).
Lower is better. Both are equal-weight means of surface pressure MAE across the
four splits, computed in the original (denormalized) target space.

## Current branch baseline (`icml-appendix-willow-pai2-r3`)

The advisor branch ships `target/train.py` in its vanilla form: MSE loss, AdamW
lr=5e-4, surf_weight=10, n_hidden=128, n_layers=5, n_head=4, slice_num=64,
mlp_ratio=2, batch_size=4, 50 epochs, no AMP, no Fourier PE. The vanilla
configuration on this cluster has not yet been measured — the first review
round will pin it.

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | TBD (pending Round 1 anchor) | — |
| `test_avg/mae_surf_p` | TBD (pending Round 1 anchor) | — |

## External reference (prior research, different cluster)

Prior work on a sibling research track (`senpai-kagent-v-students`) established
a "proven recipe" combining L1+`surf_weight=1`, AMP bf16+`grad_accum=4`,
Fourier PE σ=0.7 m=160 + FiLM(log Re), SwiGLU FFN, n_layers=3, slice_num=8.
These numbers are reference-only; they were measured on a different cluster
and are not directly comparable until re-verified on `pai-2`:

| Metric | Vanilla | Recipe |
|--------|---------|--------|
| `val_avg/mae_surf_p` | ~88 | **49.4** (2-seed mean) |
| `test_avg/mae_surf_p` | ~78 | **42.5** (2-seed mean) |

W&B project for this round: `wandb-applied-ai-team/senpai-charlie-wilson-willow-pai2-r3`.

## Reproduce vanilla

```
cd target && python train.py --agent <student-name> \
    --wandb_name <student>/vanilla-anchor
```
