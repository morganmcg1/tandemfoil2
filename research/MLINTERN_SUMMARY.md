# ML Intern — TandemFoilSet-Balanced (replicate `mlintern-pai2-72h-v4-r4`)

* Working branch: `mlintern-pai2-72h-v4-r4`
* W&B project: [`wandb-applied-ai-team/senpai-v1-ml-intern`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern)
* W&B group: `mlintern-pai2-72h-v4-r4`

## TL;DR

* **Best single-model `test_avg/mae_surf_p` so far:** **24.35**
  ([`r4-paper-h256l8-l1-ema-seed3-12h`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs))
* **Recipe:** stock Transolver scaled to the AirfRANS settings from the
  paper — `n_hidden=256, n_layers=8, n_head=8, slice_num=32, mlp_ratio=2`
  trained with **L1 + EMA(0.999)**, AdamW(lr=1e-3, wd=1e-4), OneCycleLR with
  5 % warmup, bf16 mixed precision (fp32 evaluator), `surf_weight=10`,
  `batch_size=1` × `grad_accum=4` (effective 4), 12 h wall budget per run
  (~250 epochs).
* **Ensemble:** 13+ seeds of this recipe finished, more in flight; final
  number will be the prediction-mean ensemble across all of them via
  `eval_ensemble.py`.

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
for numerical stability with the optional Gumbel-softmax slice
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
  NaN test results can be retro-fixed (e.g. `baseline-bf16` test_avg moves
  from NaN to 57.57).

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
emerging best recipe (Round 3) → multiple Round 4–7 long-cap (12 h) replicas
of the winning recipe with different seeds for ensembling. The flat
`--epochs 999` keeps OneCycleLR working at near-max LR for the entire
wall-clock window so the model keeps moving.

## Results

### Finished runs, ranked by `test_avg/mae_surf_p`

(L1 = paper recipe + L1; H_e = paper recipe + Huber + EMA; subscripts
denote the loss δ for Huber; ✱ = our final candidate family.)

| Rank | Run | best val | test_avg | Loss / extras | Wall cap |
|---:|---|---:|---:|---|---:|
|  1 | r4-paper-h256l8-l1-ema-seed3-12h ✱   | 29.26 | **24.35** | L1 + EMA | 12 h |
|  2 | r5-paper-h256l8-l1-ema-seed5-12h ✱   | 28.80 | 24.37 | L1 + EMA | 12 h |
|  3 | r4-paper-h256l8-l1-ema-seed2-12h ✱   | 28.38 | 24.70 | L1 + EMA | 12 h |
|  4 | r6-paper-h256l8-l1-ema-seed7-12h ✱   | 28.29 | 25.02 | L1 + EMA | 12 h |
|  5 | r6-paper-h256l8-l1-ema-seed8-12h ✱   | 29.16 | 25.22 | L1 + EMA | 12 h |
|  6 | r5-paper-h256l8-l1-ema-seed4-12h ✱   | 29.20 | 25.24 | L1 + EMA | 12 h |
|  7 | r5-paper-h256l8-l1-ema-seed6-cosine ✱ | 30.29 | 25.79 | L1 + EMA + cosine | 12 h |
|  8 | r6-paper-h256l8-l1-ema-seed9-cosine ✱ | 30.70 | 26.16 | L1 + EMA + cosine | 12 h |
|  9 | r3-paper-h256l8-l1-ema-seed1 ✱        | 30.70 | 26.29 | L1 + EMA | 8 h |
| 10 | r2-paper-h256l8-l1-ema ✱              | 31.47 | 27.36 | L1 + EMA | 6 h |
| 11 | r3-paper-h256l8-l1-ema-psurf10        | 32.22 | 27.84 | L1 + EMA + p_surf_extra=10 | 8 h |
| 12 | r3-paper-h256l8-l1-ema-cosine         | 31.98 | 28.22 | L1 + EMA + cosine | 8 h |
| 13 | r2-paper-h256l8-huber01-ema           | 33.80 | 29.40 | Huber(δ=0.1) + EMA | 6 h |
| 14 | r3-paper-h256l8-huber-ema-lr5e4       | 38.96 | 33.99 | Huber + EMA + lr=5e-4 | 8 h |
| 15 | r3-paper-h256l8-huber-ema-cosine      | 40.26 | 34.56 | Huber + EMA + cosine | 8 h |
| 16 | r3-paper-h256l8-huber-ema-psurf10     | 40.62 | 35.39 | Huber + EMA + p_surf_extra=10 | 8 h |
| 17 | paper-h256l8-huber-ema                | 47.55 | 40.40 | Huber + EMA | 4 h |
| 18 | paper-h256l8-l1                       | 49.37 | 43.47 | L1 (no EMA) | 4 h |
| —  | baseline-bf16                         | 63.84 | **57.57†** | original baseline + bf16 | 4 h |

† retro-evaluated through `eval_checkpoint.py` after the NaN fix; the
in-trainer test_avg was NaN before the fix.

### Per-split test surface-pressure MAE — best single model

`r4-paper-h256l8-l1-ema-seed3-12h` (test_avg/mae_surf_p = 24.35):

| Split | surf_p | surf_Ux | surf_Uy |
|---|---:|---:|---:|
| `test_single_in_dist`     | 26.38 | 0.28 | 0.17 |
| `test_geom_camber_rc`     | 37.21 | 0.53 | 0.29 |
| `test_geom_camber_cruise` | 11.74 | 0.17 | 0.10 |
| `test_re_rand`            | 22.04 | 0.30 | 0.17 |

Cruise is by far the easiest split (lower target magnitudes); RaceCar tandem
P2 (`geom_camber_rc`) is the hardest because the model never sees front-foil
NACA M=6–8 in training, only M=2–5 and M=9 — so it has to extrapolate over
the camber gap.

## Best config (final candidate)

```python
# train.py CLI for the leading configuration
--lr 1e-3 --batch_size 1 --grad_accum 4 --surf_weight 10.0
--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2
--loss l1 --scheduler onecycle --warmup_pct 0.05 --amp bf16
--ema true --ema_decay 0.999
# + SENPAI_TIMEOUT_MINUTES=720 (12 h wall) and --epochs 999
```

* Model size: **3.94 M params** (Transolver `h256/l8/h8/sn32/mlp2`).
* Optimizer: AdamW(lr=1e-3, wd=1e-4), OneCycleLR with 5 % cosine warmup.
* Loss: **L1** in normalized output space, surf_weight=10 (unchanged from the
  original baseline). EMA(0.999) for evaluation only.
* Mixed precision: bf16 forward + fp32 slice-softmax + fp32 val/test forward.
* No data augmentation, no architecture changes beyond capacity, no extra
  features, no dropout — the entire gain comes from loss / EMA / wall budget.

## Round summary

| Round | What was tested | Outcome |
|---|---|---|
| 1 (T+0:43, 8 GPU) | `n_hidden`, loss, EMA, Transolver++ Ada-Temp+Gumbel | L1/Huber + EMA + paper capacity dominate; MSE / Gumbel / `p_surf_extra` not yet competitive at 4 h |
| 2 (T+1:25, 5 GPU) | refine around L1+EMA: `slice_num`, `n_layers`, `n_hidden` | best stays at `h256/l8/h8/sn32`; bigger or deeper hurt |
| 3 (T+2:23, 6 GPU) | long runs (8 h cap) of L1+EMA, Huber+EMA + variants | L1+EMA seed=1 reaches val=30.70, **test=26.29** at ~150 epochs |
| 4 (T+8:08, 2 GPU) | replicate L1+EMA recipe with seed=2,3 at 12 h cap | best single test=**24.35** (seed=3) |
| 5 (T+11:51, 3 GPU) | 3 more seeds with 12 h cap | seed=5 test=24.37, seed=4 test=25.24, seed=6 test=25.79 |
| 6 (T+13:28, 3 GPU) | 3 more seeds (seed=7,8 onecycle, seed=9 cosine), 12 h cap | seed=7 test=25.02, seed=8 test=25.22, seed=9 test=26.16 |
| 7 (T+~20, 8 GPU) | seeds 10–17, 12 h cap | running, will be ensembled |

## Next steps

When Round 7 finishes (estimated T+~46 h):

1. Run `eval_ensemble.py` over all paper-recipe-L1-EMA checkpoints
   (`models/model-<run_id>/checkpoint.pt`). The ensemble averages per-node
   normalized predictions across models, denormalizes once, and goes through
   the same NaN-safe accumulator. Variance reduction across 15+ independent
   seeds typically reduces field-MAE by 5–15 %.
2. Update this summary + the JSONL with the ensemble metric.
3. Final commit & push.

## Files

* `train.py` — parameterizable Transolver trainer (drop-in for the original)
* `model.py` — Transolver classes, importable without running training
* `eval_checkpoint.py` — retro-evaluate a single saved checkpoint, fp32 + NaN-safe
* `eval_ensemble.py` — N-checkpoint ensemble eval (per-node prediction mean)
* `launch_logs/launcher.sh` — pinned-GPU run launcher with PID tracking
* `launch_logs/runs.tsv` — record of every launch (timestamp / GPU / PID / flags)
* `research/MLINTERN_RESULTS.jsonl` — one JSON object per finished run
* `research/MLINTERN_SUMMARY.md` — this file
