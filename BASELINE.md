# Baseline (icml-appendix-charlie-pai2f-r1)

Round 1 of the `charlie-pai2f-r1` track — no prior tuned baseline exists on this branch yet. The reference implementation is the unmodified `train.py` default config; the first round of experiments establishes the actual `val_avg/mae_surf_p` numbers that future rounds will compare against.

## Default config (`train.py` at HEAD)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=epochs |
| Batch size | 4 |
| Surf weight (loss) | 10.0 |
| Epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Sampler | WeightedRandomSampler (balanced across 3 domains) |
| Loss | MSE on normalized targets, vol + surf_weight·surf |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Params | ~0.65M |

## Baseline metrics

Round 1 in-progress. Provisional running best (not yet merged):

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | 133.892 (epoch 13/14, timeout-bound) | #1095 | Pre-correction; pressure-channel weighting, but normalized denominator dilutes aggregate surface signal — sent back for retry |
| `test_avg/mae_surf_p` (3 finite splits) | 132.106 | #1095 | 4-split avg shows NaN from one cruise-test sample with non-finite y; scoring fix applied to branch |

Primary ranking metric: `val_avg/mae_surf_p` (lower is better). Test-time metric for paper: `test_avg/mae_surf_p`.

Round 1 PRs are provisional and not yet merged. The best validated round-1 PR will set the durable round-2 baseline.

## Reproduce

```
cd target/ && python train.py \
    --agent <student> \
    --experiment_name "<student>/baseline-default" \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50
```
