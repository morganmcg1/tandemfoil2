# ML Intern Summary — TandemFoilSet-Balanced (mlintern-pai2-24h-v3-r4)

## TL;DR

| Configuration | val_avg/mae_surf_p | test_avg_3splits/mae_surf_p\* |
|---|---|---|
| Original repo baseline (Transolver 1M, MSE, 30 min cap, leaderboard ref) | ~38–80 (varied) | 39–90 |
| **Best single model** (`baseline-l1-200-s4`, phase 3, 191 epochs) | **29.000** | **29.946** |
| **Best 8-model ensemble** (`ensemble_top8`, mixed phase 2 + 3 baselines) | **25.313** | **25.681** |

\*The `test_geom_camber_cruise` split has one ground-truth file
(`000020.pt`) with 761 `+inf` values in the pressure channel. The shared
`data/scoring.py` accumulator (read-only) propagates that into a `NaN` in
`test_geom_camber_cruise/mae_surf_p`, which then poisons
`aggregate_splits` so the printed `test_avg/mae_surf_p` is `nan` for
*every* model. To compare models fairly I report `test_avg_3splits` =
mean of `mae_surf_p` over the three clean splits
(`test_single_in_dist`, `test_geom_camber_rc`, `test_re_rand`). All
intra-split numbers in this report are computed with the unmodified
`data/scoring.py`. See **Caveats** below.

W&B project: <https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern>
W&B group: `mlintern-pai2-24h-v3-r4`

## Strategy

The TandemFoilSet-Balanced benchmark predicts surface pressure (and velocity) on
irregular 2D meshes (74K–242K nodes/sample, 1499 train / 4×100 val / 4×200 test
samples) with a Transolver baseline. Primary metric is `val_avg/mae_surf_p`.

Working from the Transolver paper (arXiv:2402.02366) and AirfRANS
(2212.07564), I ran three phases on 8× RTX PRO 6000 96GB GPUs:

### Phase 1 — Architecture & loss exploration (6 h, 8 GPUs)

8 candidates, each on its own GPU. Tested L1 vs Huber vs MSE, surface-weight
tuning, p-only surface boost, learning-rate sweep, depth-vs-width tradeoff.
Anchor architecture was *medium* Transolver: `n_hidden=192, n_layers=8,
n_head=8, slice_num=32, mlp_ratio=4` (~3M params, paper-style). One slot used
the original repo baseline (`128/5/4/64/2`, ~1M params) with L1 loss as a
control.

**Result:** the small 1M-parameter baseline + L1 loss + 5-epoch warmup +
grad clip 0.5 won decisively at val 37.80, vs every medium variant in the
48–58 range. With only 1499 training samples, the larger 3M-param model was
data-starved within the 6 h budget. p_extra_weight, surf_weight=30, lr=2e-3,
and the `160/12` deep variant all *hurt* relative to plain L1.

### Phase 2 — Scale up the winner (9 h, 8 GPUs)

Took the phase-1 winner (small 1M baseline + L1) and added a *proper* cosine
LR schedule with `T_max = epochs` (phase 1 used `T_max=999` so cosine barely
decayed). Tested 3 seeds, Huber loss, longer training (300 ep), `lr=1e-3`,
batch size 8, plus one medium variant (`192/8/8/32/4` with proper schedule).

**Result:** all small-baseline variants reached val 29–31 — already 7–10
points better than phase 1's best (37.80). `lr=1e-3` slightly edged plain
`5e-4`. Huber tracked L1 within 0.4 points. The medium variant landed at
33.4 — narrowed the gap but still lost to the small baseline.

### Phase 3 — Ensemble seeds (7 h, 8 GPUs)

Grew the seed bank for ensembling: 5 more L1 baseline seeds (s2, s3, s4,
s5, s7), 2 more Huber seeds, 1 lr1e3 seed.

**Result:** new best single seed: `baseline-l1-200-s4` at val 29.000.
Top-8 ensemble (best 8 baseline checkpoints across phase 2 + phase 3)
reached val **25.313**.

## Single-model leaderboard

Top 12 by `val_avg/mae_surf_p` (lower = better). `test_3` is the
3-split test average.

| # | Run | Phase | Loss | Epochs | val | test_3 |
|---|---|---|---|---|---|---|
| 1 | baseline-l1-200-s4 | 3 | l1 | 191 | 29.000 | 29.946 |
| 2 | baseline-l1-200-s2 | 3 | l1 | 192 | 29.019 | 30.389 |
| 3 | baseline-l1-300 | 2 | l1 | 211 | 29.076 | n/a (manually killed) |
| 4 | baseline-l1-200-lr1e3 | 2 | l1 | 200 | 29.234 | **29.215** |
| 5 | baseline-l1-200-s7 | 3 | l1 | 192 | 29.294 | 29.981 |
| 6 | baseline-l1-200-s42 | 2 | l1 | 200 | 29.388 | 30.064 |
| 7 | baseline-l1-200-lr1e3-s1 | 3 | l1 | 192 | 29.396 | 29.845 |
| 8 | baseline-huber-200-s42 | 3 | huber | 192 | 29.512 | 30.239 |
| 9 | baseline-l1-200-s5 | 3 | l1 | 191 | 29.563 | 29.756 |
| 10 | baseline-huber-200 | 2 | huber | 200 | 29.606 | 31.205 |
| 11 | baseline-l1-200-s1 | 2 | l1 | 200 | 29.997 | 30.211 |
| 12 | baseline-l1-200 | 2 | l1 | 200 | 30.485 | 29.915 |

Best single test_3: `baseline-l1-200-lr1e3` at **29.215**.

## Ensemble eval

Top-8 by val score, averaged in the model's normalized output space
before denormalization (matches the organizer's scorer):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 22.404 | 0.202 | 0.172 |
| val_geom_camber_rc | 38.586 | 0.529 | 0.305 |
| val_geom_camber_cruise | 12.617 | 0.164 | 0.110 |
| val_re_rand | 27.646 | 0.359 | 0.207 |
| **val_avg** | **25.313** | 0.313 | 0.198 |
| test_single_in_dist | 22.568 | 0.237 | 0.174 |
| test_geom_camber_rc | 35.430 | 0.509 | 0.286 |
| test_geom_camber_cruise | nan (data) | 0.146 | 0.095 |
| test_re_rand | 19.048 | 0.265 | 0.169 |
| **test_avg_3splits** | **25.681** | 0.289 | 0.181 |

Comparison: bigger doesn't help.

| Ensemble size | val_avg/mae_surf_p |
|---|---|
| 1 (best single) | 29.000 |
| Top-3 | 26.000 |
| Top-5 | 25.590 |
| **Top-8** | **25.313** |
| All-15 (more variance) | 25.398 |
| Diverse-7 (L1+Huber mix) | 25.468 |

Best 8-model ensemble checkpoints (W&B run IDs):
`e36a6sul, w3wv0vn4, nllydbsv, nqhtgwvl, wgpoz0z2, 8vjis52t, 8jljpawl, w3xtde7o`.

## Code changes I want credited

- **`train.py`**: extended the `Config` dataclass with new flags
  (`loss_type`, `huber_beta`, `p_extra_weight`, `ux_weight`, `uy_weight`,
  `p_weight`, `n_hidden`, `n_layers`, `n_head`, `slice_num`,
  `mlp_ratio`, `dropout`, `optimizer`, `warmup_epochs`, `grad_clip`,
  `min_lr`, `seed`). Loss is now a per-element residual function
  (`mse | l1 | huber(beta)`) with optional per-channel weights, optional
  pressure-only surface boost (`p_extra_weight`), AdamW or Adam,
  linear-warmup → cosine LR scheduler, and optional gradient clipping.
- **`launchers/`**: parallel launchers and analysis utilities — all
  read-only on the data side.
  - `phase1.sh`, `phase2.sh`, `phase3.sh`: 8-GPU parallel launchers, one
    job per GPU, explicit `CUDA_VISIBLE_DEVICES` per subprocess
  - `status.py`: parses live log files into a sorted leaderboard
  - `parse_runs.py`: dumps per-run summary JSONL (used to populate
    `MLINTERN_RESULTS.jsonl`)
  - `find_best_checkpoints.py`: returns the top-K saved checkpoints
    by val for use with `ensemble_eval.py`
  - `ensemble_eval.py`: loads N checkpoints and averages predictions in
    the model's normalized output space, then denormalizes and computes
    val + test MAE through the unmodified `data.scoring` helpers

`data/` files were **not modified** (read-only contract). The
`pyproject.toml` was not modified — no new packages were needed.

## GPU usage strategy

8 GPUs, all on the local pai2 pod (no remote compute used).

- All three phases launched 8 *independent* single-GPU training jobs in
  parallel via `nohup` + `CUDA_VISIBLE_DEVICES=$i`. PIDs were tracked in
  `logs/<phase>.pids` so I could kill specific runs (e.g.
  `baseline-l1-300` to free its GPU for phase 3) without broad
  `pkill`-style cleanup.
- Per-run wall-clock caps via `SENPAI_TIMEOUT_MINUTES`:
  - Phase 1: 360 min (6 h)
  - Phase 2: 540 min (9 h)
  - Phase 3: 420 min (7 h)
- The cap leaves room for `train.py`'s own end-of-run test eval and W&B
  artifact upload.

Each run's `models/model-<run.id>/checkpoint.pt` is the best-by-val
state-dict, saved every time `val_avg/mae_surf_p` improves. Those
on-disk checkpoints are what `ensemble_eval.py` consumes.

## How to reproduce

```bash
# 1. Train 8 phase-3 baseline-L1 seeds (≈7 h on 8× 96 GB GPUs)
bash launchers/phase3.sh

# 2. Pick top 8 checkpoints by val and run ensemble eval
python launchers/find_best_checkpoints.py --top 8 --arch baseline > best.json
CUDA_VISIBLE_DEVICES=0 python launchers/ensemble_eval.py \
  --checkpoints \
    models/model-e36a6sul/checkpoint.pt \
    models/model-w3wv0vn4/checkpoint.pt \
    models/model-nllydbsv/checkpoint.pt \
    models/model-nqhtgwvl/checkpoint.pt \
    models/model-wgpoz0z2/checkpoint.pt \
    models/model-8vjis52t/checkpoint.pt \
    models/model-8jljpawl/checkpoint.pt \
    models/model-w3xtde7o/checkpoint.pt \
  --batch_size 4 --out_json ensemble_top8.json
```

## Caveats

1. **`test_geom_camber_cruise/000020.pt`** has 761 `+inf` values in the
   pressure channel of `y`. `data/scoring.py` (read-only) does
   `(err * mask)` where `err = (pred - y).abs()`; with `y=+inf`,
   `err=+inf` for those nodes, and `inf * 0_double = nan`. So
   `mae_surf_p` for that test split is `nan` for *every* trained model
   in this repo. The same `aggregate_splits` then produces
   `test_avg/mae_surf_p = nan`. I report a 3-split test average that
   excludes that one bad split for fair comparison. The phase-1 baseline
   in this repo's previous-leaderboard table was likely affected the
   same way.
2. `baseline-l1-300` was killed at epoch 211 (val 29.076) so I could
   reuse its GPU for phase 3 — its inline test eval did not run, hence
   `test_3 = n/a`. The checkpoint is still on disk and is included in
   the top-8 ensemble.
3. `parse_runs.py`'s split-line regex didn't match `nan`/`inf`, so its
   per-split test averages quietly excluded the cruise split. The
   numbers I report under `test_3` are computed by re-aggregating only
   the clean splits.

## Best validation metric

**val_avg/mae_surf_p = 25.313** (top-8 ensemble of 1M-param Transolver
baselines, all `n_hidden=128 n_layers=5 n_head=4 slice_num=64 mlp_ratio=2`).

## Best test metric

**test_avg/mae_surf_p (3 clean splits) = 25.681**, ensemble.
Single best test_3 was `baseline-l1-200-lr1e3` at **29.215** — shows
ensembling buys ~3.5 points on test. The 4-split
`test_avg/mae_surf_p` is `nan` for all configurations because of the
`+inf` pressure values in `test_geom_camber_cruise/000020.pt`.

## Next recommendation

If a future agent inherits this branch and wants to push lower:

1. **Get the cruise NaN fixed upstream**, or workaround in
   `data/scoring.py` (zero out `err` where the mask is False *before*
   the multiplication, or `torch.nan_to_num(err, posinf=0, neginf=0,
   nan=0)`). This is the only thing blocking a clean
   `test_avg/mae_surf_p`.
2. **Train 4–8 more baseline seeds** with the exact same recipe
   (`128/5/4/64/2`, L1, lr 5e-4, wd 1e-4, bs 4, warmup 5, cosine
   T_max=epochs=200). Ensemble keeps improving with more seeds — top-8
   is 0.7 better than top-3, top-15 wasn't quite as good but still
   better than top-3 (variance issue, not capacity).
3. **Try the stochastic-weight-averaging (SWA) trick** — avg weights
   from the last 10–20 epochs of one run. Cheap, often gives ~5–10%.
   Compatible with ensembling.
4. **AirfRANS-style boundary-layer feature**: append a per-node soft
   boundary-layer mask using the existing surface flag and an SDF-like
   distance feature derived from `dsdf` (already in input dims 4–11).
   MARIO (2505.14704) reports ~10× MSE improvement on AirfRANS with this.
5. The medium-Transolver (3M params) was data-starved in 80–200
   epochs. With a longer schedule (e.g. 400 epochs and a
   stronger weight decay) it might overtake the small baseline. I had
   one slot in phase 2 (`medium-l1-80`) at val 33.42 — worth a
   2-day run.
