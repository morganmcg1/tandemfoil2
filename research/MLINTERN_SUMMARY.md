# ML Intern — TandemFoilSet-Balanced Replicate r5

* **Branch:** `mlintern-pai2-r5`
* **W&B group:** `mlintern-pai2-r5`
* **W&B project:**
  [`wandb-applied-ai-team/senpai-v1-ml-intern`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern)
* **Hardware:** 8 × NVIDIA RTX PRO 6000 Black (96 GB), local `pai2` pod
* **Wall-clock budget:** 12 h
* **Compute strategy:** parallel single-GPU jobs (one `python train.py` per
  GPU, pinned with `CUDA_VISIBLE_DEVICES`).  Across the 12 h I ran 6 staggered
  waves of 4–8 jobs each, totalling about 50 training runs.

## Task and metric

[TandemFoilSet-Balanced](../README.md) ([SPLITS.md](../data/SPLITS.md)) — given
24 input features per mesh node (geometry, flow conditions), predict the
velocity `(Ux, Uy)` and pressure `p` field at every node of irregular CFD
meshes ranging from 74 K to 242 K nodes.  Primary ranking metric is the
**equal-weight mean surface-pressure MAE across four val splits**:
`val_avg/mae_surf_p`.  The same statistic is reported on the four test splits
as `test_avg/mae_surf_p` for the paper-facing number.

## Headline numbers (best non-debug run)

| Quantity | Value |
|---|---|
| **Best `val_avg/mae_surf_p`** | **50.07** |
| **Best `test_avg/mae_surf_p`** | **42.61** |
| Run name | `mlintern-pai2-r5/w6-onecycle-widemid-40ep` |
| W&B run | [`5hn9jko4`](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/runs/5hn9jko4) |
| Best epoch | 37 / 40 (timeout cut at 165 min after ep 37) |
| Wall clock | 167 min |
| GPU | a single RTX PRO 6000 Black (~ 42 GB peak) |

### Per-split breakdown of the best checkpoint

| Split | val surf p | val Ux | val Uy | test surf p | test Ux | test Uy |
|---|---:|---:|---:|---:|---:|---:|
| `single_in_dist` | 52.67 | 0.64 | 0.35 | 45.34 | 0.62 | 0.34 |
| `geom_camber_rc` | 65.83 | 1.06 | 0.52 | 56.35 | 1.00 | 0.48 |
| `geom_camber_cruise` | 32.36 | 0.46 | 0.26 | 26.68 | 0.43 | 0.24 |
| `re_rand` | 49.41 | 0.78 | 0.40 | 42.07 | 0.66 | 0.35 |
| **avg** | **50.07** | 0.74 | 0.38 | **42.61** | 0.68 | 0.35 |

The model struggles most on `geom_camber_rc` (held-out raceCar front-foil
camber M=6-8) and is strongest on `geom_camber_cruise`, consistent with
cruise meshes being smoother (no ground effect, freestream BCs).

### Winning recipe (commit, command)

* Code on the branch (latest commit in this PR) adds CLI flags for the
  Transolver++ tricks ([Wu et al. 2025, 2502.02414](https://arxiv.org/abs/2502.02414))
  and exposes optimiser knobs without changing the data loader contract.
* Reproducible launch:

```bash
SENPAI_TIMEOUT_MINUTES=720 python train.py \
  --epochs 40 --timeout_min 165 \
  --n_hidden 192 --n_layers 6 --n_head 6 --slice_num 64 \
  --batch_size 2 \
  --ada_temp True --gumbel True --grad_clip 1.0 \
  --surf_weight 20 --lr 0.001 --schedule onecycle \
  --agent ml-intern-r5 --wandb_group mlintern-pai2-r5 \
  --wandb_name "mlintern-pai2-r5/w6-onecycle-widemid-40ep"
```

## Strategy

I ran the experimental loop entirely on the local pai2 pod (per the compute
policy — no remote jobs).  Each wave used all 8 GPUs in parallel, with
30–130 min runtime per job depending on the question being asked.

### Wave 1 — diverse hypothesis screen (33 min each, ~ 14 epochs)
Eight different one-axis perturbations of the baseline (default Transolver,
SGD-cosine, `surf_weight=10`).  Tested:
1. Default baseline (control)
2. Paper-scale architecture (`n_hidden=256, n_layers=8, n_head=8, slice_num=32`)
3. Wide-mid (`n_hidden=192, n_layers=6, n_head=6, slice_num=64`)
4. Deeper-only (`n_layers=8`, default width)
5. Transolver++ (Ada-Temp + Gumbel-Softmax slice routing)
6. `surf_weight=20`
7. `lr=1e-3` + 2 warmup epochs
8. `batch_size=8`

Winner: **Transolver++** (val=106.7), then `surf_weight=20` and `lr=1e-3+warmup` (~118).
Big batch (bs=8) plateaued at val=142 — fewer gradient updates per epoch
hurt at fixed wall clock.

### Wave 2 — compose the wave-1 winners (60 min each, ~ 20 epochs)
Eight runs combining the top wave-1 axes plus extending training time.
Winner: **`tpp-triple` = ada_temp + gumbel + grad_clip=1 + surf_weight=20 +
lr=1e-3 + warmup=2 + cosine** at val=88.7, **test=78.9**.

### Wave 3 — refine around the triple compound (60–95 min, 20–30 epochs)
Tested per-channel pressure weight (1.5, 2, 5, 10), `mlp_ratio=4`,
`slice_num=128`, OneCycleLR vs cosine, and the wide-mid arch.

* Per-channel pressure weighting **hurt** (`pw>=2` strictly worse than no reweight).
* `mlp_ratio=4` and `slice_num=128` did not help.
* **OneCycleLR** with the same total budget tied or beat cosine+warmup.
* **30 cosine epochs** outperformed 20 onecycle epochs at the same recipe
  (val 74.9 vs 84.1, test 64.9 vs 73.1) — more training is the biggest
  remaining lever.

### Wave 4–6 — long training of the winner (60–165 min, 25–50 epochs)
Pushed the schedule out to 30, 40, and 50 epochs on default arch and the
wide-mid arch under both schedules.

* `cosine + 40 ep + default` → val=63.6, test=55.9
* `onecycle + 30 ep + default` → val=71.4, test=61.6
* `cosine + 30 ep + widemid` → val=65.0, test=56.9
* `onecycle + 30 ep + widemid` → val=56.4, test=49.4
* `cosine + 50 ep + default` → val=61.4, test=54.1
* `cosine + 40 ep + widemid` → val=54.2, test=47.4
* `onecycle + 40 ep + widemid` → **val=50.07, test=42.61** (best)

The best run combines:
* The wider Transolver (`192 / 6 / 6 / 64`, ~1.5× params of the baseline) —
  better capacity for 240K-node meshes than the default 128/5/4/64.
* Transolver++ Ada-Temp + Rep-Slice (Gumbel-Softmax) routing — replaces
  the standard slice softmax with a learnable per-point temperature plus
  Gumbel-noise sampling, making the soft assignment to physical-state
  tokens both eidetic and adaptive (paper says +13–30% on industrial CFD;
  here it is the single biggest single-axis win).
* `surf_weight=20` — twice the default; targets the metric directly.
* `lr=1e-3` matches the canonical Transolver recipe (default 5e-4 was
  under-LR'd).
* OneCycleLR over 30 epochs on this dataset converged faster than
  cosine+warmup at the same epoch budget once the model is wide enough.
* `grad_clip=1.0` was needed for numerical stability with Gumbel noise.

## Other issues fixed in this replicate

`data/scoring.accumulate_batch` skips a sample with any non-finite ground
truth via `err * mask` where mask is bool — but `0.0 * nan == nan` in IEEE
754, so a single non-finite y entry contaminates the entire split MAE.
The `test_geom_camber_cruise` split has one sample (idx 20) with 761
NaN pressure values in the ground truth, which made every wave-1 run
report NaN for `test_avg/mae_surf_p` even when the model itself was
finite.

Because `data/` is read-only per `program.md`, I sanitise predictions and
ground-truth in `train.py`'s `evaluate_split` before calling
`accumulate_batch`: zero out mask rows for samples with any non-finite
y, replace non-finite y/pred with zeros, and clip predictions to ±100 in
normalised space (a wide bound that no real value approaches because
y_norm ~ N(0, 1)).  This restores finite test metrics for every run
without changing the scoring semantics on the rest of the data.

## Compute usage

* Wall clock: ~ 7 h of training + ~ 30 min for setup, monitoring, and
  reporting.  ~ 5 h still remained on the 12 h Kubernetes kill switch
  when this summary was first drafted.
* GPU-hours: ~ 50 successful training runs × ~0.5 h each ≈ 25 GPU-hours,
  spread across 8 GPUs in 6 waves.  Steady utilisation 90–100 %.
* No remote compute used — all training stayed in the pai2 pod per the
  compute policy.

## Top 10 leaderboard

Sorted by `test_avg/mae_surf_p` (lower better).

| Rank | Run | val | **test** | Recipe |
|---:|---|---:|---:|---|
| 1 | `w6-onecycle-widemid-40ep` | 50.07 | **42.61** | tpp + sw20 + lr1e-3 + onecycle + widemid + 40 ep, bs=2 |
| 2 | `w6-cosine-widemid-40ep` | 54.24 | 47.43 | tpp + sw20 + lr1e-3 + cosine+warmup + widemid + 40 ep, bs=2 |
| 3 | `w5-onecycle-widemid-30ep` | 56.44 | 49.38 | tpp + sw20 + lr1e-3 + onecycle + widemid + 30 ep, bs=2 |
| 4 | `w5-cosine-50ep` | 61.41 | 54.05 | tpp + sw20 + lr1e-3 + cosine+warmup + 50 ep |
| 5 | `w5-final-40ep` | 63.55 | 55.94 | tpp + sw20 + lr1e-3 + cosine+warmup + 40 ep |
| 6 | `w5-onecycle-40ep` | 64.29 | 56.87 | tpp + sw20 + lr1e-3 + onecycle + 40 ep |
| 7 | `w5-cosine-widemid-30ep` | 64.97 | 56.86 | tpp + sw20 + lr1e-3 + cosine+warmup + widemid + 30 ep, bs=2 |
| 8 | `w4-onecycle-30ep` | 71.36 | 61.64 | tpp + sw20 + lr1e-3 + onecycle + 30 ep |
| 9 | `w3-triple-30ep` | 74.93 | 64.92 | tpp + sw20 + lr1e-3 + cosine+warmup + 30 ep |
| 10 | `w4-onecycle-widemid` | 75.28 | 66.30 | tpp + sw20 + lr1e-3 + onecycle + widemid + 16 ep, bs=2 |

The full per-run breakdown is in `MLINTERN_RESULTS.jsonl`.

## What did NOT work

* **`bs=8`, `bs=12`** — fewer gradient updates per epoch hurt at fixed
  wall clock.  `bs=4` was the sweet spot for the default arch; `bs=2` was
  only used because widemid/paperscale ran out of memory at higher batch.
* **Per-channel pressure weighting** (`surf_w_p > 1`) — strictly hurt at
  any value tried (1.5, 2, 5, 10).  The denormalised pressure already
  dominates absolute error; adding a multiplier on top destabilises the
  loss balance.
* **`mlp_ratio=4` and `slice_num=128`** — extra capacity in the wrong
  direction.  Slice count and FFN ratio of 64 / 2 were already enough.
* **`surf_weight=30`** — pushing the surface emphasis past 20 helped
  early but regressed by epoch 30. 20 is the right setpoint.
* **Paper-scale arch alone** (`256/8/8/32`) — too large for `bs=2` to
  converge in the available wall clock; 14 epochs got val=80, while
  widemid `192/6/6/64` got val=75 in the same time.

## Recommendations for the next replicate

1. **Run the winning config for longer** — observed scaling:
   * `onecycle + widemid + 30 ep` → test=49.38
   * `onecycle + widemid + 40 ep` → test=42.61 (-14 % from +33 % more epochs)

   Extrapolating, `onecycle + widemid + 50 ep` should land near
   test ≈ 38, and 60 ep near test ≈ 35 (likely diminishing returns
   beyond there).  Each additional 10 epochs costs ~ 40 minutes on a
   single GPU at bs=2.
2. **Add Reynolds conditioning as a DiT-style token** instead of just an
   input feature.  Published work on transonic-wing surrogates
   ([2511.21474](https://arxiv.org/abs/2511.21474)) uses adaLN-style
   conditioning on flow conditions and reports robust OOD-Re generalisation
   — directly relevant to the `val_re_rand` / `test_re_rand` split.
3. **Try LRSA** ([2604.03582](https://arxiv.org/abs/2604.03582)) —
   replaces Physics-Attention with standard SDPA on M latent tokens.  On
   AirfRANS it reports 8.5× lower surface RelL2 vs Transolver and is
   BF16-stable, so we could double batch size and halve wall clock.
4. **Per-domain sample-specific output heads** — the train data has
   three structurally different domains (raceCar single, raceCar tandem,
   cruise tandem) with order-of-magnitude different y stds.  A multi-head
   decoder conditioned on `is_tandem` and `is_cruise` from the geometry
   features (dims 18-21 of x) could recover that signal cheaply.
5. **Random node subsampling** as in the canonical Transolver
   AirfRANS recipe (32 K nodes / sample / epoch).  This is mathematically
   equivalent to a much larger effective batch size and would let us run
   the paper-scale architecture (`256/8/8/32`) at `bs=4+`.
6. **Stochastic Weight Averaging** over the last 5 epochs of the cosine
   tail — usually free 2–5 % on transformer-style training and trivial
   to add to `train.py`.

## Files

* `train.py` — extended with CLI knobs and Transolver++ tricks; the
  default-flag run reproduces the original baseline.
* `launch_run.sh` — small wrapper that pins `CUDA_VISIBLE_DEVICES` and
  redirects stdout to a log file under `logs/`.
* `research/harvest.py` — parses every log under `logs/` into a JSON
  record and writes `MLINTERN_RESULTS.jsonl`.
* `research/MLINTERN_RESULTS.jsonl` — one record per run with the val/test
  metrics, per-split breakdown, n_params, and W&B URL.
* `logs/mlintern-pai2-r5__*.log` — full stdout+stderr for every run.
* `session_logs/` — ML Intern's own conversation/tool-call artifacts;
  preserved per the contract.
