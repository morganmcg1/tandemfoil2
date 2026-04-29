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

To be populated after round 1. Each student PR's `models/<exp>/metrics.jsonl` contributes a numerical `val_avg/mae_surf_p` and `test_avg/mae_surf_p`. The best result among round 1 PRs becomes the round-2 baseline.

Primary ranking metric: `val_avg/mae_surf_p` (lower is better). Test-time metric for paper: `test_avg/mae_surf_p`.

## Reproduce

```
cd target/ && python train.py \
    --agent <student> \
    --experiment_name "<student>/baseline-default" \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50
```
