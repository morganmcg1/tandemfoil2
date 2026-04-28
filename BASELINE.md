# TandemFoilSet — Current Best Baseline

**Branch:** `icml-appendix-willow-pai2d-r4`
**Last updated:** 2026-04-28 (after PR #442 merged)

## Current best — Round 0, PR #442 (thorfinn H12 EMA × FiLM, Run F)

| Metric | Value | Δ vs prior baseline (PR #404) |
|--------|-------|-------------------------------|
| `val_avg/mae_surf_p` (raw) | 119.36 | matches PR #404 exactly (clean reproduction) |
| **`val_avg/mae_surf_p` (EMA)** | **109.19** | **−8.5%** |
| **`test_avg/mae_surf_p`** (eval source: ema) | **98.47** | **−8.4%** |
| `test/test_single_in_dist/mae_surf_p` | 111.60 | −7.5% |
| `test/test_geom_camber_rc/mae_surf_p` | 112.42 | −6.7% |
| `test/test_geom_camber_cruise/mae_surf_p` | 69.59 | −13.8% |
| `test/test_re_rand/mae_surf_p` | 100.26 | −7.4% |
| Params | 0.75M | unchanged (EMA shadow doesn't add params; just memory for the deepcopy) |

- **W&B run:** [`gc57edp6`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r4/runs/gc57edp6) (`willowpai2d4-thorfinn/h12-ema-on-film`)
- **Best epoch:** 13 of 13 actually trained (run hit 30-min wall clock)
- **Within-run EMA lift:** 8.5% (val_raw 119.36 → val_ema 109.19), monotonically EMA > raw at every EMA-eval epoch starting from epoch 1

## What changed from prior baseline (PR #404)

The merged code on `icml-appendix-willow-pai2d-r4` now includes:

1. **EMA of model weights for evaluation.** Maintains a shadow `ema_model` initialized as `copy.deepcopy(model)`. After each `optimizer.step() + scheduler.step()`, applies `p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)` (with a warmup of `ema_warmup_steps=100` steps where the shadow simply copies the live model). At validation time, evaluates BOTH the raw and EMA models and selects whichever's `val_avg/mae_surf_p` is lower for checkpoint selection. The selected model is the one used for the end-of-run test evaluation and the W&B artifact upload.
2. **`--ema_eval_every` flag** (default 2) halves the EMA-eval frequency to recover the dual-eval cost (~1 training epoch in our 30-min budget). On odd-numbered epochs the EMA eval is skipped and the most-recent EMA val is carried forward.
3. **Cumulative from PR #404 + #344:** Re-conditional FiLM modulation (γ, β per-block from log(Re)), linear warmup + per-step cosine-to-zero schedule, defensive `nan_to_num` in `evaluate_split`, `--seed` CLI flag.

## Recommended training command (reproduces current best)

```bash
cd target/ && python train.py \
    --agent <student-name> \
    --film_re True \
    --use_ema True --ema_decay 0.99 --ema_eval_every 2 \
    --epochs 25 \
    --lr 7e-4 \
    --weight_decay 5e-4 \
    --seed 123 \
    --wandb_name "<student-name>/<experiment-tag>"
```

**Important note: EMA × FiLM is the new baseline configuration.** The compound is uniform across all four test splits. Future hypothesis comparisons should layer their changes ON TOP of `--use_ema True --ema_decay 0.99 --ema_eval_every 2 --film_re True --weight_decay 5e-4 --seed 123` — these are the merged-baseline defaults that should not be overridden unless the hypothesis specifically tests them.

## Setup recap

| Setting | Value |
|---------|-------|
| Model | Transolver + Re-conditional FiLM (~0.75M params) |
| Optimizer | AdamW, weight_decay=**5e-4** (from PR #404) |
| Batch size | 4 |
| Loss | MSE in normalized space, surf_weight=10 |
| Schedule | Linear warmup (5%) + cosine-to-zero, per-step (`LambdaLR`) |
| Epochs (default) | 50, capped by `SENPAI_TIMEOUT_MINUTES=30` (~13–14 actually fit) |
| EMA decay | **0.99** (half-life ~0.2 epoch — fast tracking + noise smoothing) |
| EMA eval frequency | **every 2 epochs** (recovers dual-eval cost) |
| Recommended `--lr` | 7e-4 (from PR #344; default still 5e-4) |
| Recommended `--seed` | 123 (for variance-check reproducibility; default None) |
| Primary metric | `val_avg/mae_surf_p` — selects best of raw vs EMA |
| Paper metric | `test_avg/mae_surf_p` — uses the model that produced the best val |

## Validation/test splits

- `val_single_in_dist` / `test_single_in_dist` — random holdout from single-foil (sanity)
- `val_geom_camber_rc` / `test_geom_camber_rc` — held-out front foil camber (raceCar M=6-8)
- `val_geom_camber_cruise` / `test_geom_camber_cruise` — held-out front foil camber (cruise M=2-4)
- `val_re_rand` / `test_re_rand` — stratified Re holdout across tandem domains

## Round-0 history

- **PR #344 (edward H2):** linear warmup + per-step cosine + NaN fix → val=120.97, test=109.92.
- **PR #404 (edward H11):** Re-conditional FiLM + wd=5e-4 → val=119.36, test=107.54 (Run E, seed=123).
- **PR #442 (thorfinn H12):** EMA decay=0.99 + every-other-epoch eval → val_ema=109.19, test=98.47 (Run F, seed=123 on FiLM-merged baseline).

## Key methodological tooling shipped this round

- **`--seed` CLI flag** (PR #404) — enables seed-controlled comparisons. PR #442's Run F demonstrated 4-sig-fig reproducibility of PR #404's baseline.
- **`--ema_decay` and `--ema_eval_every` flags** (PR #442) — EMA is now a default-on compounding lever. Future PRs touching optimization/loss/regularization should layer on top rather than re-test EMA.
- **Defensive `nan_to_num` in `evaluate_split`** (PR #344) — robust against `test_geom_camber_cruise` sample 20's `-inf` GT.
