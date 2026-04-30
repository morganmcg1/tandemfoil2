# ML Intern TandemFoilSet-Balanced Benchmark — Replicate r3 Summary

**Branch**: `mlintern-pai2-72h-v4-r3`
**W&B group**: `mlintern-pai2-72h-v4-r3` (project `wandb-applied-ai-team/senpai-v1-ml-intern`)
**Pod start**: 2026-04-30 09:59:43 UTC
**Hard deadline**: 2026-05-03 09:59:43 UTC (72h)

## Strategy

Hill-climb on the Transolver baseline by running quick parallel sweeps across all 8 RTX PRO 6000 GPUs, doubling the budget per run as winners emerge. Each sweep narrows on what worked in the previous one. Optimize `val_avg/mae_surf_p`; preserve `test_avg/mae_surf_p` reporting at the end of every run.

## Headline results (best config so far)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Time |
|-----|-------------------:|---------------------:|-----:|
| s1-baseline (vanilla repo defaults) | 138.7 | – | 25 m |
| s1-small3l-l1 | 72.4 | – | 25 m |
| s2-small3l-l1-warmup | 46.5 | 40.5 | 55 m |
| s3-l1-warmup-amp-90 | 35.9 | **30.5** | 90 m |
| s4-amp-ema-90 | 33.2 | **28.3** | 90 m |
| s5-decay9995-3h | 30.6 | **25.77** | 3 h |
| s5-best-3h | 30.6 | **25.78** | 3 h |
| s6-* (8 runs, 6 h each) | TBD — running | TBD | 6 h |

## What works (in order discovered)

1. **Reduce capacity → small Transolver**: `n_layers=3, n_head=1, slice_num=16, n_hidden=128, mlp_ratio=2` (~0.56 M params) crushes the 5L baseline on this 1 499-sample dataset. Vanilla 5L overfits.
2. **L1 loss directly** (instead of MSE on normalized targets) — directly optimizes the MAE objective. Huber(δ=0.1) ≈ L1, Huber(0.3) noticeably worse.
3. **Linear warmup of 5 epochs** before cosine LR — small (~3 %) but consistent.
4. **bf16 mixed precision (`use_amp=True`)** — ~2× throughput with no quality loss. Allows ~150 epochs in 90 min vs ~60 without AMP. Single biggest gain (~22 % improvement).
5. **EMA(decay=0.999)** of model weights, used for val + best checkpoint selection. Smooths checkpoint noise from 4×100 val samples and gives ~4 % MAE drop.
6. **Longer training**: 25 m → 55 m → 90 m → 3 h all monotonically improved best val MAE. Likely keeps improving to 6 h+ given the residual loss slope.

## What doesn't work / no measurable lift

- Larger batch size (bs=8) without LR rescaling — slightly worse with cosine schedule pinned to epochs
- Higher pressure-channel weight (`p_weight=5`) — diverges
- Higher surf-loss weight (`surf_weight=20`) — within noise of default 10
- Random Fourier Features on (x, z) — neutral or slightly worse for sigma ∈ {1, 2, 4}
- Multiscale slice_num (`32,16,8`) — neutral
- More layers (n_layers=4) on this dataset/budget — overfits
- Wider model (n_hidden=160, n_hidden=192) — slower per-epoch, no quality gain at fixed wall time
- Lower LR (2e-4) — slower convergence at fixed budget
- Heavier weight_decay+dropout — kills performance
- Grad clip 1.0 — within noise (might help slightly with longer training; sweep 6 retesting clip0.5)

## Code/data changes credited to this replicate (`train.py` only; data/ untouched)

- Configurable architecture/loss/training flags: `--n_layers --n_hidden --n_head --slice_num --mlp_ratio --dropout --loss_type --p_weight --huber_delta --grad_clip --warmup_epochs --timeout_min --seed --use_amp --rff_sigma --rff_B_size --ema_decay --slice_nums`
- L1 / MSE / Huber loss switch via `loss_type`, with optional pressure-channel reweight
- Optional bf16 autocast in the training loop (`use_amp`)
- Optional Fourier features for (x, z) coords inside the model (`rff_sigma > 0`)
- Optional per-layer slice_num list (`slice_nums="32,16,8"`)
- Optional EMA model via `torch.optim.swa_utils.AveragedModel`; EMA weights drive val + best-checkpoint logging when `ema_decay > 0`
- NaN-safe local accumulator `_accumulate_batch_safe`: numerically identical to `data/scoring.accumulate_batch` on NaN-free batches but avoids the `0 × NaN = NaN` propagation that turned the test cruise split's MAE into NaN (one cruise test sample, `000020.pt`, has NaN pressure on ~0.34 % of nodes, which the original scorer skips at the sample level but the masked multiply still carries through). `data/scoring.py` is untouched.

## GPU usage strategy

- 8 × NVIDIA RTX PRO 6000 Blackwell, 96 GB each
- One training process per GPU pinned with `CUDA_VISIBLE_DEVICES=N`
- Sweep 1 (25 m), sweep 2 (55 m), sweep 3 (90 m), sweep 4 (90 m), sweep 5 (180 m), sweep 6 (360 m) — 8 parallel runs per sweep
- Models are tiny (0.56 M params, ~50 GB peak with AMP); compute is dominated by the 80–240 K node padded batches.

## Best command (current)

```bash
CUDA_VISIBLE_DEVICES=0 python ./train.py \
  --n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 \
  --loss_type l1 --warmup_epochs 5 \
  --use_amp True --ema_decay 0.999 \
  --epochs 600 --timeout_min 360 \
  --agent ml-intern-r3 \
  --wandb_group mlintern-pai2-72h-v4-r3 \
  --wandb_name "mlintern-pai2-72h-v4-r3/<name>"
```

## Per-run records

See `MLINTERN_RESULTS.jsonl` for one JSON object per run with status, best val MAE, test avg MAE, per-split test MAEs, n_params, and W&B run id.

## Next recommendation

After sweep 6 (8 × 6 h, finishing around T + 14 h), pick the best 1–2 configs and run a single 12–15 h final job on the same hardware. With AMP throughput we'll see ~1 200 epochs there, which is roughly 4× more training than sweep 5 — and sweep 1→5 has shown roughly monotonic improvement with training time.

If the multi-seed sweep 6 runs spread within ±1 % MAE, a simple model-average of the 4 best EMA checkpoints (predictions averaged at evaluation) is the cheapest +1–2 % to grab on the final test eval.
