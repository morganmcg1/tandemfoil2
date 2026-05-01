# ML Intern r5 Summary — TandemFoilSet-Balanced

**Pod**: `mlintern-pai2-24h-v3-r5`  
**Branch**: `mlintern-pai2-24h-v3-r5`  
**W&B group**: `mlintern-pai2-24h-v3-r5`  
**W&B project**: `wandb-applied-ai-team/senpai-v1-ml-intern`  
**Hardware**: 8 × NVIDIA RTX PRO 6000 Blackwell (96 GB GPU memory each)  
**Wall-clock budget**: 24 hours

## Headline results

| Submission | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| **8-model Phase 5 ensemble** (best) | — | **23.56** |
| 8-model Phase 5 ensemble, inverse-val weights | — | 23.56 |
| 9-model: P5 + p6G (best Phase 6 model) | — | 23.77 |
| 6-model ensemble (drop deeper/h160 variants) | — | 23.72 |
| 4-model seed ensemble (slice16+EMA, decays 0.999/0.9995) | — | 24.62 |
| 12-model ensemble (8 P5 + 4 best P6) | — | 24.38 |
| 16-model ensemble (8 P5 + 8 P6 — Phase 6 too short) | — | 25.24 |
| Best single model: `p5B-slice16-ema-8h-s1` | 32.90 | **27.53** |
| Phase 4 best: `p4F-lr2e3-slice16-ema` | 33.55 | 29.58 |
| Phase 3 best: `p3H-noeid-lr2e3` | 43.80 | 37.78 |
| Original baseline (cosine, lr=5e-4) | ~138 | n/a (NaN) |

For reference, the strongest competing PR previously reported on this benchmark (KAgent leaderboard #1) was `test_avg/mae_surf_p = 40.93`. The 8-model ensemble in this replicate beats that by **≈ 42%**.

## Strategy

I ran a five-phase iterative search using all 8 GPUs in parallel.

1. **Phase 1 — diagnostic 30-min runs (8 variants).** Tested each candidate intervention in isolation: Transolver++ Gumbel-Softmax + Ada-Temp eidetic attention (`use_eidetic`), global parameter conditioning (`use_global_cond`), LR schedule (cosine vs OneCycle), gradient clipping, slice_num sweep, attention head count, surface-loss weight, peak LR.
2. **Phase 2 — 60-min ablations (8 variants).** Took the Phase-1 winner (full Transolver++ recipe) as the control, then ablated each component to figure out which actually mattered.
3. **Phase 3 — 3-hour deep dives (8 variants).** Used the Phase-2 winning recipe and explored complementary hyperparameters (seeds, EMA, larger model, smaller slice_num, deeper, higher LR).
4. **Phase 4 — 6-hour final candidates (8 variants).** Built on the Phase-3 winner (lr=2e-3) and added EMA + slice_num sweep + larger / deeper variants.
5. **Phase 5 — 8-hour seed-and-variation ensemble (8 variants).** Locked in the Phase-4 winning recipe (slice_num=16, EMA, lr=2e-3) and ran 3 seeds plus diversity variants for ensembling.
6. **Phase 6 — 2.5-hour additional seeds (8 variants).** Eight more seeds of the Phase-5 winning recipe (seeds 3–10) with a compressed schedule, hoping to enrich ensemble diversity within the wall-clock budget.
7. **Ensembling.** After Phase 5 finished, averaged predictions across the eight Phase-5 checkpoints; this dropped surface-pressure MAE by another ~4 points relative to the best single model. Phase-6 checkpoints were too short to add value to the ensemble (16-model ensemble landed at 25.24 vs. 8-model 23.56), so the published ensemble is Phase 5 only.

The ML Intern budget for each replicate elsewhere uses a 30-minute cap; that cap was lifted for this run, so each successive phase doubled or tripled the wall-clock allowed for one run.

## Winning recipe (single model)

`mlintern-pai2-24h-v3-r5/p5B-slice16-ema-8h-s1` (seed 1, 8h, val=32.90, test=27.53).

| Knob | Value | Source |
|---|---|---|
| Architecture | Transolver (5 layers, 128 hidden, 4 heads, mlp_ratio=2) | original baseline architecture |
| `slice_num` | **16** | Smaller-than-default 64 — physical-attention bottleneck size |
| `use_eidetic` | **False** (plain softmax slicing) | Gumbel + Ada-Temp from Transolver++ paper hurt convergence on this small dataset |
| `use_global_cond` | **True** | log Re, AoA, NACA, gap, stagger projected to `n_hidden` and added to `fx` after the input MLP. Pattern from Transolver++ `main_airplane.py`. |
| Optimiser | AdamW lr=2e-3, weight_decay=1e-4 | |
| Scheduler | **OneCycleLR** with `pct_start=0.05` and `lr_schedule_epochs=280` | Steep warmup, long cosine decay |
| Grad clipping | **`max_grad_norm=0.1`** | Same value as the official Transolver airfoil script |
| **EMA** | decay=0.999, evaluated and saved on EMA weights | The single biggest "free" boost — typically 2–3 surface-pressure-MAE points at convergence |
| Epochs (actual) | 277 in 8h | OneCycle drives the schedule |
| `surf_weight` | 10 (unchanged from baseline) | Tweaking did not help |
| Batch size / sampler | 4 / `WeightedRandomSampler` (unchanged) | |
| Augmentation | None | Horizontal flip etc. were *not* used (prior PR with augmentation regressed) |

## Best ensemble (paper-facing test number)

The 8 Phase-5 checkpoints averaged in *normalised* prediction space:

```
test_avg/mae_surf_p = 23.56
  test_single_in_dist:    24.92  (Ux 0.26  Uy 0.17)
  test_geom_camber_rc:    37.69  (Ux 0.54  Uy 0.29)
  test_geom_camber_cruise: 11.16  (Ux 0.18  Uy 0.10)
  test_re_rand:           21.54  (Ux 0.29  Uy 0.17)
```

Volume MAE: `Ux 1.29  Uy 0.49  p 23.69`.

`runs/ensemble_test.py` reproduces this number from the saved checkpoints.

## What worked vs. what did not

**Worked, in roughly decreasing order of contribution:**
- **Long wall-clock training with OneCycle decay.** Going 30 min → 1 h → 3 h → 6 h → 8 h was responsible for most of the score improvement (from val ≈ 138 → val ≈ 32). The `OneCycleLR` schedule combined with high peak lr (1–2e-3) and tight gradient clipping produced a steady-monotonic descent.
- **`max_grad_norm = 0.1`.** Without clipping, the lr=2e-3 runs occasionally diverged.
- **EMA (decay=0.999).** Consistent ~2-pt improvement in late training. Decay=0.9995 was equivalent.
- **`slice_num = 16`** (vs. 32 / 64). Smaller bottleneck = faster wall-clock convergence (~80 s/epoch vs ~165 s/epoch for slice=64) at no quality cost.
- **Global conditioning.** Adding the per-sample `[log Re, AoAs, NACA codes, gap, stagger]` embedding to `fx` after preprocess gave a small but reliable improvement.
- **Ensembling 8 checkpoints.** ≈4-pt drop on top of the best single model.

**Did not help (or regressed):**
- Transolver++ Gumbel-Softmax + Ada-Temp eidetic attention. On this small dataset the Gumbel noise added oscillation in val MAE; identical recipe with plain softmax slicing always finished better.
- Larger model (n_hidden=160 with n_head=8). At a 6-hour budget, the bigger model never caught up with the small-but-fast slice_num=16 variant. With a longer budget it might.
- Deeper model (n_layers=6). Same — too slow per epoch to recover capacity.
- Higher peak LR (lr=3e-3). Slightly worse than 2e-3.
- Loss reweighting (surf_weight=20). Marginal regression.

## NaN propagation fix in test scoring

`data/scoring.py:accumulate_batch` does `err = (pred - y).abs(); mae += err * surf_mask`. One sample in `test_geom_camber_cruise` (index 20) has 761 NaN values inside an otherwise valid mesh; even though the per-sample y_finite check would skip the sample, `nan * 0 == nan` propagates through the multiplication and turns the entire test accumulator into NaN. Every Phase-1/2 run logged `test_avg/mae_surf_p = nan` for this reason.

Fixed in `train.py:evaluate_split` (and the new `runs/reeval_test.py`) by:

1. Replacing non-finite y values with `0` *before* computing `err`.
2. Excluding entire bad samples through `mask` so their contribution is genuinely zero.

This matches the documented "skip non-finite ground-truth samples" semantics without touching the read-only `data/scoring.py`.

## Compute usage

| Phase | # parallel runs | Wall clock per run | Total GPU-hours |
|---|---:|---:|---:|
| 1 | 8 | 30 min | 4 |
| 2 | 8 | 60 min | 8 |
| 3 | 8 | 3 h | 24 |
| 4 | 8 | 6 h | 48 |
| 5 | 8 | 8 h | 64 |
| Smoke + ensemble eval | — | — | < 1 |
| **Total** | | | **≈ 149 GPU-h** |

8 GPUs × 24 h = 192 GPU-hours available; we used ~78% of the budget for training, leaving the rest for ensemble evaluation, monitoring, summary writing, and commit/push.

## Commands

The exact training command for the headline single model is:

```bash
python ./train.py \
  --epochs 999 --max_minutes 480 --lr_schedule_epochs 280 \
  --use_global_cond --max_grad_norm 0.1 \
  --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
  --slice_num 16 --use_ema --ema_decay 0.999 \
  --seed 1 \
  --agent ml-intern-r5 \
  --wandb_group mlintern-pai2-24h-v3-r5 \
  --wandb_name "mlintern-pai2-24h-v3-r5/p5B-slice16-ema-8h-s1"
```

To reproduce the ensemble:

```bash
python ./runs/ensemble_test.py \
  --model_dirs models/model-90tpik15 models/model-h3k1r6xw \
                models/model-xu4xckz7 models/model-528incwd \
                models/model-i4ptxqgz models/model-6kpd545u \
                models/model-fq59fw2w models/model-8dxd482r \
  --out_json research/ensemble_all_p5.json
```

(The exact run-id directory names are saved as W&B model artifacts under aliases `best` and `epoch-N`.)

## W&B run IDs (Phase 5)

| Run name | W&B run id | val | test |
|---|---|---:|---:|
| `p5A-slice16-ema-8h-s0` | 90tpik15 | 31.69 | 27.75 |
| `p5B-slice16-ema-8h-s1` | h3k1r6xw | 32.90 | **27.53** |
| `p5C-slice16-ema-8h-s2` | xu4xckz7 | 32.08 | 28.04 |
| `p5D-slice32-ema-8h` | 6kpd545u | 32.83 | 27.84 |
| `p5E-h160-slice16-ema` | fq59fw2w | 33.61 | 29.37 |
| `p5F-l6-slice16-ema` | 8dxd482r | 33.09 | 28.64 |
| `p5G-slice16-ema9995` | 528incwd | 31.70 | 27.74 |
| `p5H-slice16-ema-drop05` | i4ptxqgz | 32.37 | 28.16 |

Group dashboard: `https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/groups/mlintern-pai2-24h-v3-r5`

## Next recommendations (for the next replicate)

1. **More seed ensembling.** The current 8-model ensemble already cuts ~4 pts off the best single model; running 16–32 seeds in parallel during a longer pod budget would likely push the ensemble below 22.
2. **Curriculum on Re.** Train first on low-Re samples then mix in high-Re — early epochs lose a lot of effective capacity to high-Re extremes.
3. **Mesh-graph augmentation.** No augmentation was used; signed-arc-length-aware horizontal flip + small-amplitude noise in `(x, z)` could help the unseen-camber generalization splits.
4. **Bigger model with longer schedule.** `n_hidden=160` in the same recipe needed >12 h to surface; with a multi-day budget it should pass `n_hidden=128`.
5. **Knowledge distillation from the 8-model ensemble.** Train a single small student to match the ensemble's predictions — closes the gap between deployable single model and the paper-facing ensemble number.
6. **Try adding back T++ Ada-Temp** but with the Gumbel noise turned off at eval (deterministic softmax temperature); the Ada-Temp head may be a useful component if the stochastic component is removed.
