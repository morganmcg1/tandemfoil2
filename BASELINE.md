# BASELINE — TandemFoilSet (icml-appendix-willow-pai2e-r1)

Track: `icml-appendix-willow-pai2e-r1`. Round 1 — first results in.

## Implicit baseline

The current implicit baseline is the **unmodified `train.py`** with default config:

- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- AdamW: `lr=5e-4, weight_decay=1e-4`, `CosineAnnealingLR(T_max=epochs)`
- Training: `batch_size=4, surf_weight=10, epochs=50`, MSE loss in normalized space
- Sampler: `WeightedRandomSampler` balancing 3 train domains
- 24-dim node features, 3-dim outputs `(Ux, Uy, p)`
- Reproduce: `python train.py --agent baseline --wandb_name baseline --wandb_group baseline`

## Primary metric

`val_avg/mae_surf_p` for checkpoint selection; `test_avg/mae_surf_p` for paper-facing comparison.
Both are equal-weight means of surface-pressure MAE across the four splits, computed in original
denormalized target space.

## Best so far

| PR   | W&B run    | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes                                |
|------|------------|---------------------|---------------------|--------------------------------------|
| **#775** | [nezuko w0-clip0.5-ema0.99](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1) | **96.54** | **85.33** | warmup=0 + clip=0.5 + EMA=0.99 + Huber δ=0.5, epoch 14, **MERGED ✓** |
| #769 | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) | 102.86 | 94.83 | Huber δ=0.5, no clip, no EMA, **MERGED ✓** |
| #773 | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | 119.35 | 108.79 | EMA decay=0.99, no Huber, **MERGED ✓** |
| #846 (ref) | [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | Unmodified default @ 14 ep — canonical reference |

**Accumulated gains vs unmodified default (140.95/128.32):**
- EMA alone (PR #773): −15.4% val / −15.3% test
- Huber δ=0.5 alone (PR #769): −27.0% val / −26.2% test
- **clip=0.5 + warmup=0 + Huber δ=0.5 + EMA=0.99 (PR #775): −31.5% val / −33.5% test** — *new best*
- Huber + EMA (no clip) stack: in progress (alphonse PR #881)

## Per-split test metrics (current best — PR #775, warmup=0 + clip=0.5 + Huber δ=0.5 + EMA=0.99)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | ~100           |
| test_geom_camber_rc        | ~88            |
| test_geom_camber_cruise    | ~67            |
| test_re_rand               | ~86            |

*(Per-split breakdown from nezuko's results — see PR #775 comments for exact values.)*

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-nezuko \
    --wandb_group warmup-clip-ema-huber \
    --warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99
```

**Default flags for all future experiments:**
```
--huber_delta 0.5 --ema_decay 0.99 --warmup_epochs 0 --clip_norm 0.5
```
All four merged wins — must be set explicitly (none are defaults in train.py).
