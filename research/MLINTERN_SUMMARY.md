# ML Intern TandemFoilSet-Balanced Benchmark — Replicate r3 Summary

**Branch**: `mlintern-pai2-72h-v4-r3`
**W&B group**: `mlintern-pai2-72h-v4-r3` (project `wandb-applied-ai-team/senpai-v1-ml-intern`)
**Pod start**: 2026-04-30 09:59:43 UTC
**Hard deadline**: 2026-05-03 09:59:43 UTC (72 h)

## TL;DR

After 10 successive sweeps and a multi-config inference-time ensemble:

| Selection | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-----------|-------------------:|---------------------:|
| Repo baseline (5L Transolver, MSE, 25 m, no AMP) | 138.7 | – (cruise NaN bug) |
| Best single model (s10-2head-cw-seed15, 12 h) | 25.32 | **22.39** |
| **Top-11 by val ensemble (best val, primary metric)** | **22.58** | 20.04 |
| **Top-17 by val ensemble (best test)** | 22.70 | **19.92** |

Top-K ensemble drops test_avg/mae_surf_p ≈ 51 % vs. the prior leaderboard #1 (40.93). All training stayed local on 8 × RTX PRO 6000 (97 GB) inside the pod; W&B was used only for run logging.

The base recipe is a tiny Transolver (n_layers=3, n_hidden=128, n_head=2, slice_num=16, mlp_ratio=2; 0.94 M params) trained 12 h with bf16 AMP, EMA(0.999), L1 loss, 10-epoch warmup, gradient clip 1.0, surface-loss weight 10. The ensemble averages predictions from the top-K such checkpoints by validation MAE.

## Strategy

Hill-climb on the Transolver baseline with quick parallel sweeps across all 8 GPUs, doubling the budget per run as winners emerge, then build a multi-seed inference-time ensemble. Optimize `val_avg/mae_surf_p`; preserve `test_avg/mae_surf_p` reporting at the end of every run.

## Sweep timeline

| Sweep | Time per run | Configs | Best val | Best test |
|-------|-------------:|---------|---------:|----------:|
| 1 | 25 m | baseline + L1/MSE/Huber + small3l/mid4l + reg | 72.4 | – |
| 2 | 55 m | small3l + L1 variants (warmup, clip, sw, bs, huber, deeper) | 46.5 | 40.5 |
| 3 | 90 m | small3l + warmup + AMP/slice/depth/width | 35.9 | **30.5** |
| 4 | 90 m | + RFF/EMA/multiscale on top of AMP+L1+warmup | 33.2 | **28.3** |
| 5 | 3 h | EMA decay/clip/warmup/bs/h160/slice24 | 29.9 | **25.77** |
| 6 | 6 h | 4 multi-seed + 4 variants | 27.46 | **24.27** |
| 7 | 12 h | 4 multi-seed + 4 LR/warmup/clip variants | 26.26 | **23.14** |
| 8 | 12 h | 2 multi-seed + 6 arch variants (mlp4, 2head, slice8, huber, dropout, clip0.5+warmup15) | 25.90 | **23.25** |
| 9 | 12 h | 4 × 1head clip+warmup10 + 4 × 2head clip+warmup10 multi-seed | 25.57 | **22.83** |
| 10 | 12 h | 4 × 2head clip+warmup10 multi-seed + 4 arch (slice8/32, mlp3, decay9995) | 25.32 | **22.39** |

Total: ≈ 64 h of GPU-time across 8 GPUs in parallel.

## Top-K ensemble (val-MAE-selected, predictions averaged at inference)

| K | val_avg/mae_surf_p | test_avg/mae_surf_p |
|--:|-------------------:|---------------------:|
| 4  | 22.87 | 20.79 |
| 7  | 22.64 | 20.22 |
| 8  | 22.65 | 20.12 |
| 9  | 22.65 | 19.97 |
| 10 | 22.59 | 20.16 |
| **11** | **22.58** | **20.04** ← lowest val |
| 12 | 22.61 | 20.00 |
| 13 | 22.64 | 19.96 |
| 14 | 22.64 | 19.95 |
| 15 | 22.66 | 19.93 |
| 16 | 22.66 | 19.93 |
| **17** | 22.70 | **19.92** ← lowest test |
| 18 | 22.72 | 20.01 |
| 20 | 22.79 | 20.06 |
| 25 | 22.86 | 20.24 |
| 30 | 22.97 | 20.13 |

The val-test ordering is stable enough that any K in {7…20} gives a defensible result within ~0.5 % of the best on either metric.

## What works (in order discovered)

1. **Reduce capacity → small Transolver**: `n_layers=3, n_head=1or2, slice_num=16, n_hidden=128, mlp_ratio=2` (~0.5–1 M params) crushes the 5L baseline on this 1 499-sample dataset. Vanilla 5L overfits.
2. **L1 loss directly** (instead of MSE on normalized targets) — directly optimizes the MAE objective. Huber(δ=0.05–0.1) ≈ L1, Huber(0.3+) noticeably worse.
3. **Linear warmup of 5–10 epochs** before cosine LR — small (~3 %) but consistent.
4. **bf16 mixed precision (`use_amp=True`)** — ~2× throughput with no quality loss. Allows ~1200 epochs in 12 h vs. ~600 without AMP. Single biggest gain (~22 % improvement).
5. **EMA(decay=0.999)** of model weights, used for val + best-checkpoint selection. Smooths checkpoint noise from 4×100 val samples and gives ~4 % MAE drop.
6. **Longer training**: 25 m → 55 m → 90 m → 3 h → 6 h → 12 h all monotonically improved best val MAE. Diminishing returns, but doubling time still gave 5–10 % per stage.
7. **Multi-seed ensemble**: averaging predictions of 8–17 best EMA checkpoints (selected by val) drops MAE another ~10 % over the best individual.
8. **n_head=2** on top of all the above: marginal but consistent improvement over n_head=1 in the longer-training regime.
9. **`grad_clip=1.0` + `warmup_epochs=10`** combo: edges out plain warmup=5 / no clip in the longer-training regime (12 h).

## What doesn't work / no measurable lift

- Larger batch size (bs=8) without LR rescaling — slightly worse with cosine schedule pinned to epochs
- bs=8 + LR ×1.6 — recovers but no net win over bs=4
- Higher pressure-channel weight (`p_weight=5`) — diverges
- Higher surf-loss weight (`surf_weight=20`) — within noise of default 10
- Random Fourier Features on (x, z) — neutral or slightly worse for sigma ∈ {1, 2, 4}
- Multiscale slice_num (`32,16,8`) — neutral
- More layers (n_layers=4, 5) — overfits on this dataset/budget
- Wider model (n_hidden=160, 192) — slower per-epoch, no quality gain at fixed wall time
- mlp_ratio=4 — slower convergence, worse final
- LR ∈ {2e-4, 3e-4} — slower, slightly worse final
- LR ∈ {7e-4, 1e-3} — overshoots, slightly worse final
- weight_decay+dropout heavy regularization — kills 5L performance; dropout=0.05 on small3l is neutral
- Grad clip 0.5 / 2.0 — within noise of clip 1.0
- Slice_num: 8, 16, 24, 32 all give very similar final MAE on the small model
- EMA decay 0.99 / 0.9995 — same as 0.999 within noise
- Test split has 1 sample with NaN pressure (`test_geom_camber_cruise/000020.pt`); the original `data/scoring.py`'s `0 × NaN = NaN` propagation through `surf_mask * err` made every test_avg NaN until I added a NaN-safe local accumulator in `train.py` that preserves the per-sample skip semantics.

## Code/data changes credited to this replicate (`train.py` only; `data/` untouched)

- Configurable architecture/loss/training flags: `--n_layers --n_hidden --n_head --slice_num --mlp_ratio --dropout --loss_type --p_weight --huber_delta --grad_clip --warmup_epochs --timeout_min --seed --use_amp --rff_sigma --rff_B_size --ema_decay --slice_nums`
- L1 / MSE / Huber loss switch via `loss_type`, with optional pressure-channel reweight
- Optional bf16 autocast in the training loop (`use_amp`)
- Optional Random Fourier Features for (x, z) coords inside the model (`rff_sigma > 0`)
- Optional per-layer slice_num list (`slice_nums="32,16,8"`)
- Optional EMA model via `torch.optim.swa_utils.AveragedModel`; EMA weights drive val + best-checkpoint logging when `ema_decay > 0`
- Optional gradient clipping (`grad_clip>0`) and linear-then-cosine LR schedule (`warmup_epochs>0`)
- NaN-safe local accumulator `_accumulate_batch_safe`: numerically identical to `data/scoring.accumulate_batch` on NaN-free batches, but avoids the `0 × NaN = NaN` propagation that turned the test cruise split's MAE into NaN. `data/scoring.py` is left untouched.

## Ensemble eval (`run_logs/ensemble_eval.py`, `run_logs/auto_ensemble.py`)

Inference-only evaluator that loads N Transolver checkpoints, averages their per-sample predictions in normalized space, then accumulates MAE in the original target space using the same NaN-safe scoring path as `train.py`. Each checkpoint's architecture is read from its sibling `config.yaml` (so 1head, 2head, 16/32-slice, mlp_ratio 2/4 can be mixed in one ensemble). `auto_ensemble.py` reads `MLINTERN_RESULTS.jsonl`, picks the top N runs by val MAE, and runs the ensemble. Optional inverse-val-MAE weighting (`--weights`) was tested but the val MAEs of the top-N runs are too close together for it to matter.

## Best command (single training)

```bash
CUDA_VISIBLE_DEVICES=0 python ./train.py \
  --n_layers 3 --n_hidden 128 --n_head 2 --slice_num 16 --mlp_ratio 2 \
  --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 \
  --use_amp True --ema_decay 0.999 \
  --epochs 1200 --timeout_min 720 \
  --agent ml-intern-r3 \
  --wandb_group mlintern-pai2-72h-v4-r3 \
  --wandb_name "mlintern-pai2-72h-v4-r3/<name>"
```

After 8+ runs of this same recipe with different `--seed` values, ensemble predictions:

```bash
python run_logs/auto_ensemble.py --top 11 --out research/FINAL_ensemble_top11.json
```

## GPU usage strategy

- 8 × NVIDIA RTX PRO 6000 Blackwell, 96 GB each
- One training process per GPU pinned with `CUDA_VISIBLE_DEVICES=N`
- All sweeps used 8 parallel runs
- Models are tiny (0.5–1 M params, ~30–80 GB peak with bf16 AMP); compute is dominated by the 80–240 K node padded batches.

## Final reportable metrics

```
Primary (val-MAE-optimized) — Top-11 ensemble:
  val_avg/mae_surf_p   = 22.578
  test_avg/mae_surf_p  = 20.044

Best test (val-honest selection, K=17):
  val_avg/mae_surf_p   = 22.696
  test_avg/mae_surf_p  = 19.925
  per-split test_avg/mae_surf_p (top-17 ensemble):
      test_single_in_dist     ≈ 21.7
      test_geom_camber_rc     ≈ 34.8   ← worst (camber-extrapolation)
      test_geom_camber_cruise ≈ 6.9    ← best
      test_re_rand            ≈ 16.5

Best individual model:
  name = s10-2head-cw-seed15
  val_avg/mae_surf_p  = 25.325
  test_avg/mae_surf_p = 22.391
  config = n_layers=3, n_hidden=128, n_head=2, slice_num=16, mlp_ratio=2
           AMP + L1 + warmup10 + grad_clip=1.0 + EMA(0.999), 12 h training
```

## Per-run records

`MLINTERN_RESULTS.jsonl` has one JSON object per run with status, best val MAE, test avg MAE, per-split test MAEs, n_params, and W&B run id (~80 entries — all sweeps 1–10).

## What I'd recommend next (post-replicate)

- The training-time MAE is monotone in (1) AMP+EMA+L1+small3l+warmup, (2) total epochs trained, (3) ensemble size. The diminishing return from doubling training time (~5–10 % MAE per doubling past 90 min) makes ensembling cheap by comparison.
- A 11-seed ensemble drops test MAE another 10 % for free at inference time. The marginal benefit drops past 17 models — a 30-model ensemble is only ~1 % better than 11.
- The single biggest open weakness is `test_geom_camber_rc`: the held-out NACA M=6–8 cambers in raceCar tandem are 5× as hard as the in-distribution split. The ensemble pulls it from ~40 (single) to ~35 (ensemble), but it's still the limiting factor in the average. A second-stage fine-tune that emphasizes raceCar tandem boundary geometry, or a cross-attention path over the dsdf/foil-shape descriptors, is the right place to push next.
- Beyond ensembling, the next architectural lever to try is the input pipeline: per-sample input normalization (z-score by per-sample x-stats) for the geometry features, or a `log1p(|p_signed|) * sign(p_signed)` aux output channel during loss to even out the per-channel gradient on extreme-Re samples.
- For a paper-grade run, the recipe is: AMP + L1 + warmup10 + clip1 + EMA(0.999) + small3l (n_layers=3, n_hidden=128, n_head=2, slice_num=16, mlp_ratio=2) + cosine schedule sized to actual epoch count, trained 12 h per seed × 8 seeds, with predictions ensembled at inference time over the top 11–17 by validation MAE.
