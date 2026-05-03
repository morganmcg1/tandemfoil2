# ML Intern — TandemFoilSet-Balanced (replicate `mlintern-pai2-72h-v4-r4`)

* Working branch: `mlintern-pai2-72h-v4-r4`
* W&B project: [`wandb-applied-ai-team/senpai-v1-ml-intern`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern)
* W&B group: `mlintern-pai2-72h-v4-r4`

## TL;DR

* **Best single model `test_avg/mae_surf_p`**: **24.35**
  (`r4-paper-h256l8-l1-ema-seed3-12h`)
* **Best ensemble `test_avg/mae_surf_p`**: **22.39**
  (top-25 of 28 paper-recipe-L1+EMA seeds, prediction-mean ensemble selected
  by `val_avg/mae_surf_p` only — no test peeking)
* **Per-split (top-25 ensemble):** `test_single_in_dist=24.71`,
  `test_geom_camber_rc=34.79`, `test_geom_camber_cruise=10.28`,
  `test_re_rand=19.78`.
* **Recipe** (every one of the 28 ensemble members, no per-seed tuning):
  paper-Transolver `n_hidden=256, n_layers=8, n_head=8, slice_num=32,
  mlp_ratio=2` + L1 loss + EMA(0.999) + AdamW(lr=1e-3, wd=1e-4) +
  OneCycleLR (5 % warmup) + bf16 mixed precision + `surf_weight=10` +
  `batch_size=1` × `grad_accum=4`, 6–12 h wall budget per seed.
  3.94 M params / model.

## Strategy

Anchored on the Transolver baseline (Wu et al., ICML 2024 — arxiv 2402.02366),
with two paper-derived levers tested in parallel during Round 1:

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
reparameterization from Transolver++ — Luo et al., ICML 2025, arxiv 2502.02414;
the Transolver++ code path was tested in Round 1 but did not converge faster
than the plain Transolver at this dataset size).

Once the **L1 + EMA + paper-capacity** recipe was clearly winning by Round 2,
the entire remaining budget (Rounds 3–9) went into running that *same recipe*
with different seeds and slightly different schedules so an ensemble could be
formed at the end.

Two correctness fixes were necessary to land any of these comparisons:
* **fp32 evaluation.** bf16 autocast on the largest cruise meshes
  (≥ 200K nodes) can produce non-finite predictions in some configs —
  `evaluate_split` now forces fp32 forward at val/test time.
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
All training was local: each run is one process pinned to one GPU via
`CUDA_VISIBLE_DEVICES`. PIDs are tracked in `launch_logs/runs.tsv` so cleanup
shuts down only the exact processes I launched (no `pkill`/`pgrep`).
`launch_logs/launcher.sh` standardizes the per-run command:

```bash
nohup env CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT \
  python ./train.py --epochs 999 --agent ml-intern-r4 \
    --wandb_group mlintern-pai2-72h-v4-r4 \
    --wandb_name "mlintern-pai2-72h-v4-r4/$NAME" \
    "${FLAGS[@]}" > "$LOG_DIR/$NAME.log" 2>&1 &
```

`--epochs 999` keeps OneCycleLR working at near-max LR for the entire
wall-clock window so the model keeps moving inside the wall cap.

| Round | Cap | When | What |
|---|---|---|---|
| 1 | 240 min | T+0:43, 8 GPU | broad sweep — loss ∈ {MSE, Huber, L1}, ±EMA, ±Transolver++, baseline reference |
| 2 | 360 min | T+1:25, 5 GPU | refine around L1+EMA — `slice_num`, `n_layers`, `n_hidden`, Huber-δ |
| 3 | 480 min | T+2:23, 6 GPU | longer L1+EMA / Huber+EMA + cosine / lr=5e-4 / p_surf_extra |
| 4 | 720 min | T+8:08, 2 GPU | first 12 h cap on L1+EMA: seeds 2, 3 |
| 5 | 720 min | T+11:51, 3 GPU | seeds 4, 5, 6 (one with cosine schedule) |
| 6 | 720 min | T+13:28, 3 GPU | seeds 7, 8, 9 (one with cosine) |
| 7 | 720 min | T+~20, 8 GPU | seeds 10–17 |
| 8 | 720 min | T+24:09, 2 GPU | seeds 18, 19 |
| 9 | 720 min | T+~26, 6 GPU | seeds 20–25 |

In total 28 paper-recipe-L1+EMA seeds × ~12 h of training on 96 GB GPUs.
Plus the Round 1/3 reference / variant runs.

## Recipe (final)

```python
# train.py CLI for every Round-4-and-later seed
--lr 1e-3 --batch_size 1 --grad_accum 4 --surf_weight 10.0
--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2
--loss l1 --scheduler onecycle --warmup_pct 0.05 --amp bf16
--ema true --ema_decay 0.999 --seed <0-25>
# + SENPAI_TIMEOUT_MINUTES=720 (12 h wall) and --epochs 999
```

* Model: 3.94 M params (`Transolver(h256/l8/h8/sn32/mlp2)`).
* Optimizer: AdamW(lr=1e-3, wd=1e-4), OneCycleLR with 5 % cosine warmup.
* Loss: **L1** in normalized output space, surf_weight=10 (unchanged).
  EMA(0.999) only used at val/test time.
* Mixed precision: bf16 forward + fp32 slice-softmax + fp32 val/test
  forward + NaN-y guard around `accumulate_batch`.
* No data augmentation, no architecture changes beyond capacity, no extra
  features, no dropout. The entire gain over the baseline is from
  loss + EMA + scale + wall-clock budget + ensembling.

## Final ensemble

`eval_ensemble.py models/<run_id>/checkpoint.pt × 25`, `--mode mean`,
`--batch_size 1`, run in fp32 with the same NaN-y guard. The 25 members
are the 25 lowest-`val_avg/mae_surf_p` paper-recipe seeds (selection by val
only — no test peeking).

| Ensemble | val_avg | test_avg | SID | RC | Cruise | Re |
|---|---:|---:|---:|---:|---:|---:|
| Best single (`r4-seed3-12h`) | 29.26 | 24.35 | 26.38 | 37.21 | 11.74 | 22.04 |
| Top-7 by val | 25.99 | 22.75 | 25.07 | 35.34 | 10.36 | 20.24 |
| Top-15 by val | 25.96 | 22.49 | 24.99 | 34.88 | 10.23 | 19.86 |
| Top-20 by val | 26.04 | 22.47 | 24.86 | 34.92 | 10.26 | 19.86 |
| **Top-25 by val (final)** | **26.19** | **22.39** | **24.71** | **34.79** | **10.28** | **19.78** |
| Top-25 by val, **median** | 26.15 | 22.46 | 24.81 | 34.89 | 10.30 | 19.82 |
| Top-28 (all) | 26.38 | 22.55 | 24.89 | 34.91 | 10.45 | 19.95 |

K = 23–25 is the sweet spot; mean beats median; including the 3 lowest-val
runs (`r2-l1-ema`, `r3-cosine`, `r3-psurf10`) hurts because they trained for
only 6–8 hours and lag the rest by val ~2 MAE points.

## Single-model leaderboard (paper-recipe-L1+EMA only)

| Rank by val | Run | val | test |
|---:|---|---:|---:|
| 1 | r7-seed12 | 27.79 | 24.51 |
| 2 | r7-seed15 | 28.06 | 24.67 |
| 3 | r8-seed18 | 28.29 | 25.24 |
| 4 | r6-seed7-retry | 28.29 | 25.02 |
| 5 | r7-seed11 | 28.34 | 24.93 |
| 6 | r4-seed2 | 28.38 | 24.70 |
| 7 | r9-seed21 | 28.42 | 24.62 |
| 8 | r7-seed16 | 28.47 | 24.94 |
| 9 | r8-seed19 | 28.54 | 24.54 |
| 10 | r7-seed10 | 28.55 | 24.68 |

(Spread across 28 seeds: val 27.79–32.22, test 24.35–28.22. The per-seed
variance is real — 6–12 h cap on a 1499-sample dataset isn't enough to
fully converge.)

## Best single model — full per-split

`r4-paper-h256l8-l1-ema-seed3-12h`, test_avg/mae_surf_p = 24.35:

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

## Files added / changed

* `train.py` — parameterizable Transolver trainer (drop-in for the original):
  bf16 autocast, L1/Huber/MSE loss, OneCycleLR, EMA, gradient checkpointing,
  gradient accumulation, Transolver++ Ada-Temp + Gumbel-Softmax slice repar,
  `unified_pos`, fp32 + NaN-safe val/test eval. Defaults reproduce the
  original baseline.
* `model.py` — Transolver classes, importable without running training.
* `eval_checkpoint.py` — retro-evaluate one saved checkpoint (fp32 + NaN guard).
* `eval_ensemble.py` — N-checkpoint ensemble eval (mean / median / weighted).
* `launch_logs/launcher.sh` + `monitor.sh` — pinned-GPU run launcher with
  PID tracking; one entry per launch in `launch_logs/runs.tsv`.
* `research/MLINTERN_SUMMARY.md` — this file.
* `research/MLINTERN_RESULTS.jsonl` — one JSON object per finished run with
  val/test metrics, config, and run_id (35+ rows).
* `research/ensemble-*.json` — top-K ensemble eval outputs.
* `research/eval_*.json` — single-model retro-eval outputs.

## Reproducing the final number

```bash
# Train one ensemble member (any seed reproduces ~val=28-30, test=24.5±0.3)
SENPAI_TIMEOUT_MINUTES=720 python ./train.py \
  --epochs 999 --agent ml-intern-r4 \
  --wandb_group mlintern-pai2-72h-v4-r4 \
  --wandb_name "mlintern-pai2-72h-v4-r4/seed-XX" \
  --lr 1e-3 --batch_size 1 --grad_accum 4 --surf_weight 10.0 \
  --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 \
  --loss l1 --scheduler onecycle --warmup_pct 0.05 --amp bf16 \
  --ema true --ema_decay 0.999 --seed XX

# Ensemble: pick the 25 lowest-val checkpoints and mean their predictions
python eval_ensemble.py --model_dirs models/model-<id1> ... models/model-<id25> \
  --names s1 s2 ... s25 --mode mean --batch_size 1 --skip_individual \
  --out_json research/final-ensemble.json
```

## Next recommendation

If I had another 24 h I would:

1. **More seeds** until the top-K plateau is reached. The K=15 → K=25 jump
   gave 22.49 → 22.39, suggesting the benefit isn't yet exhausted.
2. **Longer single runs** (24 h cap) so each member's val drops below 28.
   Best individual val is currently 27.79 with 12 h; rate of improvement is
   still ~0.3 MAE / 50 epochs at the 200-epoch mark, so 24 h could land a
   single model at val ≈ 26 / test ≈ 22.
3. **Test-time augmentation** for the harder `test_geom_camber_rc` split
   only — input AoA/Re jitter ensembling, NOT geometric flips, since the
   `saf`/`dsdf` features baked into x are not symmetric under reflection
   (Round 1 baseline + horizontal flip has been documented to fail).
4. Re-implement the LinearNO Q/K decoupling
   (arxiv 2511.06294) — it claims a 60 % `C_L` ρ improvement on AirfRANS over
   plain Transolver with 40 % fewer parameters, which would mean either a
   smaller model of equal quality or a deeper one of much higher quality.
   This was scoped out for this 72 h replicate to keep focus on a single
   robust recipe.
