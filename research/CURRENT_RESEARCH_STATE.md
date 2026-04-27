# SENPAI Research State

- **Date**: 2026-04-27
- **Most recent research direction from human researcher team**: None (no human issues found)
- **Current research focus**: Initial round of experiments on the TandemFoilSet CFD surrogate task. The baseline is a Transolver model (n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2) with AdamW optimizer (lr=5e-4), surf_weight=10, cosine LR schedule. Primary metric: val_avg/mae_surf_p (surface pressure MAE, lower is better).

## Current Research Themes

1. **Hyperparameter tuning**: Learning rate, surface weight, weight decay — identifying the best basic training config
2. **Architecture exploration**: Width (n_hidden), depth (n_layers), slice_num, attention head count
3. **Loss formulation**: L1 vs L2, per-channel pressure up-weighting, surface vs volume loss balance
4. **Regularization**: Dropout, gradient clipping, EMA weight averaging
5. **Feature engineering**: Fourier positional encoding, physics-aware features
6. **Optimization**: AMP/bf16 for throughput, alternative optimizers

## Potential Next Research Directions

- Surface-only loss with stronger pressure channel weighting
- Deeper models (n_layers=6-8) with larger hidden dim
- Fourier positional encodings for spatial position features
- SwiGLU feedforward activations
- L1 loss (MAE) aligned with evaluation metric
- asinh target transform to handle heavy-tailed Re distribution
- Per-channel decoder heads (separate prediction for Ux, Uy, p)
- EMA weight averaging during training
- Larger batch via gradient accumulation with AMP
- Mixed-resolution training with curriculum

## Key Constraints

- VRAM: 96 GB per GPU, meshes up to 242K nodes
- Timeout: SENPAI_TIMEOUT_MINUTES wall clock + --epochs limit
- Epochs cap (from env): controls max training duration
- Data loaders are read-only
