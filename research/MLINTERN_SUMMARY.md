# ML Intern — TandemFoilSet-Balanced (replicate `mlintern-pai2-72h-v4-r4`)

Working branch: `mlintern-pai2-72h-v4-r4`
W&B project: [`wandb-applied-ai-team/senpai-v1-ml-intern`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern)
W&B group: `mlintern-pai2-72h-v4-r4`

> Status: in progress — this file is updated as runs complete. Final candidate
> selection happens once the longest Round 3 run reports test metrics.

## Strategy

Anchored on the Transolver baseline (Wu et al., ICML 2024 — arxiv 2402.02366),
with two paper-derived levers tested in parallel:

1. **Capacity scale-up** to the AirfRANS settings used in the official
   `thuml/Transolver` repo: `n_hidden=256, n_layers=8, n_head=8, slice_num=32`,
   `lr=1e-3`, OneCycleLR with cosine annealing — the baseline ships with
   `n_hidden=128, n_layers=5, n_head=4, slice_num=64, lr=5e-4` which is
   intentionally undersized for the dataset.
2. **Loss alignment with the metric.** The primary metric is MAE
   (`val_avg/mae_surf_p`), but the baseline trains with squared error.
   Switching to L1 / Huber gives gradients aligned with what we are measured
   on, without changing any data interface.

EMA (decay `0.999`) is applied on top to smooth the trajectory; bf16
mixed-precision is used everywhere except inside the slice-softmax (kept fp32
for numerical stability, especially with the optional Gumbel-softmax slice
reparameterization from Transolver++ — Luo et al., ICML 2025, arxiv 2502.02414).

Two correctness fixes were necessary to land any of these comparisons:
* **fp32 evaluation.** bf16 autocast on the largest cruise meshes
  (≥ 200K nodes) can produce non-finite predictions — `evaluate_split` now
  forces fp32 forward at val/test time.
* **NaN-y guard.** `data/scoring.py:accumulate_batch` skips per-sample when
  ground truth has any non-finite value, but its `err * mask` multiply
  propagates NaN through (`NaN * 0.0 == NaN`). The
  `test_geom_camber_cruise/000020.pt` sample has 761 non-finite `p` values; in
  `B>1` batches this leaks NaN into the float64 accumulator and turns
  `test_avg/mae_surf_p` into NaN. `train.py:evaluate_split` now zeros out
  non-finite ground-truth entries and extends the mask to drop bad-sample
  positions, preserving the official "drop the whole sample" semantics
  without the NaN leak. `eval_checkpoint.py` applies the same guard so old
  NaN test results can be retro-fixed.

## Compute usage

Single pai2 pod, 8× RTX PRO 6000 Blackwell (96 GB each), 72-hour wall budget.
All training is local: each experiment is one process pinned to one GPU via
`CUDA_VISIBLE_DEVICES`. PIDs are tracked in `launch_logs/runs.tsv` so cleanup
shuts down only the exact processes I launched (no `pkill`/`pgrep`).
`launch_logs/launcher.sh` standardizes the per-run command shape:

```bash
nohup env CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT \
  python ./train.py --epochs 999 --agent ml-intern-r4 \
    --wandb_group mlintern-pai2-72h-v4-r4 \
    --wandb_name "mlintern-pai2-72h-v4-r4/$NAME" \
    "${FLAGS[@]}" \
    > "$LOG_DIR/$NAME.log" 2>&1 &
```

Round 1 → Round 2 → Round 3 funneled the search: 8 broad configs (Round 1) →
5 refinements around the best loss/EMA combo (Round 2) → 6 long runs of the
emerging best recipe (Round 3). The flat `--epochs 999` keeps OneCycleLR
working at near-max LR for the entire wall-clock window so the model keeps
moving.

## Round 1 — broad sweep (T+0:43, 240 min cap, 8 GPUs)

Goal: pick a loss + capacity. All configs were paper-recipe (h256/l8/h8/sn32,
lr=1e-3, onecycle, bf16, bs=1, grad_accum=4) except the two reference
baselines.

| Run | Recipe | best val | test_avg | Notes |
|---|---|---|---|---|
| `baseline-h128l5-mse-cosine-fp32` | original baseline (h128/l5, lr=5e-4, MSE) | 139.43 (killed) | — | killed @ ep 19 — clearly behind |
| `baseline-bf16` | baseline + bf16 | 63.84 | **57.57** (post-fix) | original recipe is competitive |
| `paper-h256l8-mse` | paper recipe + MSE | 152.31 (killed) | — | killed — MSE clearly behind |
| `paper-h256l8-huber` | paper recipe + Huber(δ=1) | 135.79 (killed) | — | killed — Huber alone needed EMA |
| `paper-h256l8-huber-psurf10` | + p_surf_extra_weight=10 | 138.83 (killed) | — | extra term destabilized early |
| `paper-h256l8-huber-tppl` | + Ada-Temp + Gumbel-Softmax | 137.71 (killed) | — | Transolver++ slow to settle here |
| `paper-h256l8-huber-ema` | paper + Huber + EMA(0.999) | **47.55** | **40.40** | strong: 1499 → 47.55 in 85 ep |
| `paper-h256l8-l1` | paper + L1 (no EMA) | **49.37** | **43.47** | L1 great, EMA still helps |

**Verdict.** L1 / Huber + EMA + paper capacity dominate. MSE alone or any
no-EMA variant gives noisier and slower convergence. The Transolver++ Gumbel
slice did not stabilize fast enough at this LR; left for future work.

## Round 2 — refine around L1/Huber + EMA (T+1:25, 360 min cap, 5 GPUs)

5 variants on top of `paper + L1/Huber + EMA(0.999)`, all bf16 / onecycle / bs=1·ga=4:

| Run | Variant | best val | Notes |
|---|---|---|---|
| `r2-paper-h256l8-l1-ema` | L1 + EMA | **36.81** (running) | leader; still improving |
| `r2-paper-h256l8-huber01-ema` | Huber(δ=0.1) + EMA | **37.69** (running) | tracks L1 closely (δ→0 ⇒ L1) |
| `r2-paper-h256l10-l1-ema` | n_layers=10 | 74.17 (killed) | deeper hurt early convergence |
| `r2-paper-h256l8-l1-ema-sn64` | slice_num=64 | 74.52 (killed) | more slices = slower |
| `r2-paper-h384l8-l1-ema` | n_hidden=384 | 76.76 (killed) | bigger ≠ better at this dataset size |

**Verdict.** Architecture is right where it is (`h256/l8/sn32/h8`). Extra
capacity hurts on 1499 training samples; the lever is loss + EMA, not size.

## Round 3 — long-running L1+EMA family (T+2:23, 480 min cap, 6 GPUs)

Tracks two questions: (a) does cosine without OneCycleLR warmup beat
OneCycleLR, and (b) is there a useful seed / scheduling / surface-pressure
weighting tweak.

| Run | Variant | epochs so far | best val |
|---|---|---|---|
| `r3-paper-h256l8-huber-ema-cosine` | Huber + EMA + cosine schedule | 49 | 55.77 |
| `r3-paper-h256l8-huber-ema-lr5e4` | Huber + EMA + lr=5e-4 (lower) | 49 | 62.25 |
| `r3-paper-h256l8-huber-ema-psurf10` | + p_surf_extra_weight=10 | 49 | 66.88 |
| `r3-paper-h256l8-l1-ema-cosine` | L1 + EMA + cosine | 10 | 120.3 (warming up) |
| `r3-paper-h256l8-l1-ema-psurf10` | L1 + EMA + p_surf_extra_weight=10 | 10 | 108.3 (warming up) |
| `r3-paper-h256l8-l1-ema-seed1` | L1 + EMA seed=1 | 10 | 113.3 (warming up) |

These will run for several more hours; the table above will be updated when
they finish.

## Best config (current candidate)

```python
# train.py flags for the leading configuration
--lr 1e-3 --batch_size 1 --grad_accum 4 --surf_weight 10.0
--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2
--loss l1 --scheduler onecycle --warmup_pct 0.05 --amp bf16
--ema true --ema_decay 0.999
```

* Model size: 3.94 M params (Transolver h256/l8/h8/sn32, 8× the baseline).
* Optimizer: AdamW(lr=1e-3, wd=1e-4), OneCycleLR with 5 % cosine warmup.
* Loss: L1 in normalized output space, surf_weight=10 as in the original
  baseline. EMA(0.999) for evaluation only.
* Mixed precision: bf16 forward + fp32 slice-softmax + fp32 val/test forward.

## Best result so far

* **Best val_avg/mae_surf_p:** 36.81 — `r2-paper-h256l8-l1-ema` at epoch 84
  (still training, T+4 h).
* **Best finished test_avg/mae_surf_p:** 40.40 — `paper-h256l8-huber-ema`
  (R1, 240 min cap).

Trajectories show steady ~0.5 points/epoch improvement on the leader; the
full 6 h cap should land it lower.

## Next steps

When Round 2 leader and Round 3 long runs finish:

1. Compare final test numbers between the L1+EMA / Huber(0.1)+EMA /
   cosine-schedule variants.
2. Launch Round 4 (final candidate): the winning recipe replayed for the
   maximum wall-clock window I can afford, then evaluated on the four test
   splits via `train.py`'s built-in test loop.
3. Optionally retro-evaluate any saved checkpoints with `eval_checkpoint.py`
   so all reported `test_avg` numbers go through the same NaN-safe fp32
   scorer.
