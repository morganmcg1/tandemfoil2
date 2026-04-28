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
| **#769** | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) | **102.86** | **94.83** | Huber δ=0.5, no EMA, epoch 14, **MERGED ✓** |
| #773 | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | 119.35 | 108.79 | EMA decay=0.99, no Huber, **MERGED ✓** |
| #846 (ref) | [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | Unmodified default @ 14 ep — canonical reference |

**Pending wins (not yet baseline):**
- PR #775 (nezuko warmup5-clip0.5): val_avg=115.01, test_avg=101.64 — originally beat EMA-only baseline; re-running with --ema_decay 0.99 to compare against new Huber baseline.

**Accumulated gains vs unmodified default (140.95/128.32):**
- EMA alone (PR #773): −15.4% val / −15.3% test
- Huber δ=0.5 alone (PR #769): **−27.0% val / −26.2% test** — *largest single win so far*
- Huber + EMA stack: **untested — in progress** (alphonse PR in flight)

## Per-split test metrics (current best — PR #769, Huber δ=0.5, no EMA)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | 109.68         |
| test_geom_camber_rc        |  99.91         |
| test_geom_camber_cruise    |  76.12         |
| test_re_rand               |  93.62         |

Biggest gains vs EMA baseline: rc −17.8%, re_rand −14.7% — exactly the heavy-tailed OOD splits the Huber hypothesis targeted.

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-alphonse \
    --wandb_group huber-delta-sweep-v2 --wandb_name huber-delta0.5-v2 \
    --huber_delta 0.5
```

**For future experiments: always include both `--huber_delta 0.5` and `--ema_decay 0.99`** (two independent merged wins that are expected to stack). Neither is default — both must be set explicitly.
