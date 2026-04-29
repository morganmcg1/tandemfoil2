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
| **#860** | [qfsfasvc](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/qfsfasvc) | **75.94** | **65.86** | OneCycle T=16 + slice=32 + 4-way (δ=0.5+clip+w0+EMA), FP32, epoch 16, **MERGED ✓** |
| #959 | [j7zko7ml](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/j7zko7ml) | 79.82 | 70.00 | BF16 + δ=0.1 + EMA=0.99 (slice=64), epoch 18, **MERGED ✓** |
| #862 | [jsat9zk5](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jsat9zk5) | 82.64 | 73.02 | slice=32 + 4-way stack (δ=0.5+EMA+clip+w0), epoch 16, **MERGED ✓** |
| #881 | [jej4y8gt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jej4y8gt) | 85.23 | 76.64 | Huber δ=0.1 + EMA=0.99, no clip/warmup, **MERGED ✓** |
| #775 | [h22uwyy3](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/h22uwyy3) | 96.54 | 85.33 | warmup=0 + clip=0.5 + Huber δ=0.5 + EMA=0.99, **MERGED ✓** |
| #769 | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) | 102.86 | 94.83 | Huber δ=0.5, no clip, no EMA, **MERGED ✓** |
| #773 | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) | 119.35 | 108.79 | EMA decay=0.99, no Huber, **MERGED ✓** |
| #846 (ref) | [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | Unmodified default @ 14 ep — canonical reference |

**Accumulated gains vs unmodified default (140.95/128.32):**
- EMA alone (PR #773): −15.4% val / −15.3% test
- Huber δ=0.5 alone (PR #769): −27.0% val / −26.2% test
- clip=0.5 + warmup=0 + Huber δ=0.5 + EMA=0.99 (PR #775): −31.5% val / −33.5% test
- Huber δ=0.1 + EMA=0.99 (PR #881): −39.5% val / −40.3% test
- slice=32 + 4-way stack (PR #862): −41.4% val / −43.1% test
- BF16 + δ=0.1 + EMA=0.99 (PR #959): −43.4% val / −45.4% test
- **OneCycle T=16 + slice=32 + 4-way (PR #860): −46.1% val / −48.7% test** — *new best*

**Stack note:** PR #860 (val=75.94) is on FP32 + δ=0.5 + clip + w0 + EMA + OneCycle T=16 + slice=32. PR #959 (val=79.82) is on BF16 + δ=0.1 + EMA + slice=64. These are parallel winning paths — the combination (BF16 + slice=32 + δ=0.1 + clip + w0 + EMA + OneCycle T=16) is the obvious next milestone.

**Key insight from PR #860 R3:** `--onecycle_total_epochs` must match the actual epoch budget for the stack. slice=32 completes ~16 epochs in 30 min (faster than slice=64's 14 epochs). Using T=14 on slice=32 truncates 2 epochs and wastes budget.

**PR #957 (alphonse):** confirmed clip+warmup=0 stacks at δ=0.1 (−4.7% val on slice=64 FP32). OneCycle T=16 + slice=32 + δ=0.1 + clip is the next major combination to test.

## Per-split test metrics (current best — PR #860, OneCycle T=16 + slice=32 + 4-way)

| Split                      | test/mae_surf_p |
|----------------------------|----------------|
| test_single_in_dist        | **74.13**      |
| test_geom_camber_rc        | **78.70**      |
| test_geom_camber_cruise    | **46.29**      |
| test_re_rand               | **64.31**      |

Massive gains vs PR #959 (83.00/80.46/48.43/68.12): all four splits improved.

## Reproduce best checkpoint

```bash
cd target/
python train.py --agent willowpai2e1-thorfinn \
    --wandb_group schedule-alignment-v3 --wandb_name onecycle-slice32-T16-fullstack \
    --warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99 \
    --slice_num 32 \
    --scheduler onecycle --peak_lr 1e-3 --pct_start 0.3 --onecycle_total_epochs 16
```

**Minimum required flags for ALL future experiments:**
```
--use_bf16                            (1.353× throughput, 21% less VRAM — from PR #959)
--slice_num 32                        (clear architectural win — from PR #862)
--huber_delta 0.1 --ema_decay 0.99   (δ=0.1 floor confirmed — PRs #881, #957)
--warmup_epochs 0 --clip_norm 0.5    (clip+w0 stacks at δ=0.1 — PR #957; confirmed on δ=0.5 in PR #862)
--scheduler onecycle --peak_lr 1e-3 --pct_start 0.3 --onecycle_total_epochs 16
                                      (OneCycle T=16 for slice=32 budget — from PR #860)
```

**Note:** `--onecycle_total_epochs` should match the actual epoch budget for the slice/BF16 combo. At slice=32 + BF16, expect ~20-22 epochs in 30 min — use `--onecycle_total_epochs 20` (or observe and adjust).
