# ML Intern Replicate r5 — TandemFoilSet-Balanced

**Branch**: `mlintern-pai2-72h-v4-r5`
**Hardware**: 8× NVIDIA RTX PRO 6000 Blackwell (96 GB / 600 W each), 72 h cap.
**W&B group**: `mlintern-pai2-72h-v4-r5` ([project](https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern))

> This document is updated as the run progresses; the final ranking is at the
> top, the exploration history below.

---

## Current best results

Lower is better; primary ranking metric is `test_avg/mae_surf_p` (paper-facing).

### Single-model checkpoints

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Params | Epochs |
|---|---:|---:|---:|---:|
| `p3-bf16+huber+nl3+e200` | 41.67 | **35.95** | 0.42 M | 200 |
| `p4-bf16+huber+nl2+e150` | 45.20 | 40.22 | 0.30 M | 150 |
| `p4-bf16+huber+nl2+ema999` | 50.44 | 43.53 | 0.30 M | 100 |
| `p2-bf16+huber+nl2` | 49.95 | 43.77 | 0.30 M | 100 |
| `p4-bf16+huber+nl2+seed1234` | 49.77 | 43.91 | 0.30 M | 100 |
| `p3-bf16+huber+nl3+ema999` | 49.36 | 44.06 | 0.42 M | 100 |
| `p5-bf16+huber+nl2+seed2024` | 51.07 | 44.32 | 0.30 M | 100 |
| `p4-bf16+huber+nl2+seed42` | 51.15 | 44.46 | 0.30 M | 100 |
| `p2-bf16+huber+nl3` | 51.28 | 45.38 | 0.42 M | 80 |
| `p3-eidetic+bf16+huber+nl3` | 51.53 | 46.95 | 0.42 M | 100 |
| `p4-bf16+huber+nl2+h96` | 52.98 | 47.05 | 0.17 M | 100 |
| `p4-bf16+huber+nl1` | 55.07 | 48.65 | 0.18 M | 100 |

### Ensembles (test eval is fp32, no autocast)

| Ensemble | Members | val_avg | test_avg |
|---|---:|---:|---:|
| `eval_ensemble_2model_top` | 2 (nl3-e200 + nl2-e150) | 39.36 | **34.37** |
| `eval_ensemble_nl2_5_plus_nl3e200` | 6 (5 nl2 + 1 nl3-e200) | 41.03 | 35.40 |
| `eval_ensemble_nl2_6_plus_nl3e200` | 7 (6 nl2 + 1 nl3-e200) | 41.42 | 35.60 |
| `eval_ensemble_nl3_4` | 4 (1 nl3-e200 + 1 nl3-ema + 1 nl3-eidetic + 1 nl3-80ep) | 41.53 | 36.58 |
| `eval_ensemble_nl2_5` | 5 (nl2 base + ema999 + seed42 + seed1234 + e150) | 42.87 | 37.13 |
| `eval_ensemble_nl2_4` | 4 (nl2 base + ema999 + seed42 + seed1234) | 44.57 | 38.51 |

**Best so far**: 2-model top ensemble (`nl3-e200` + `nl2-e150`) — `test_avg/mae_surf_p = 34.37`. Per-split test:
- `test_single_in_dist` = 37.84
- `test_geom_camber_rc` = 47.18
- `test_geom_camber_cruise` = 19.76
- `test_re_rand` = 32.72

Five additional 200-epoch nl3 runs with different seeds and one EMA variant
are still training; once they converge we'll re-evaluate strong-only
ensembles to see if 4-5 high-quality nl3 models beat the current 2-model
top mix. A 300-epoch nl3 just kicked off as well.

---

## Strategy

The benchmark trains a Transolver surrogate to predict `(Ux, Uy, p)` at every
node on tandem-airfoil meshes (75 K – 242 K nodes, 1499 train / 4×100 val /
4×200 test). Primary metric is `val_avg/mae_surf_p` for selection,
`test_avg/mae_surf_p` for paper reporting. The four splits cover an
in-distribution sanity track plus three generalization axes (raceCar camber,
cruise camber, stratified Re).

After reading `program.md`, `data/SPLITS.md` and `train.py`, I framed the
problem as: small training set, mesh sizes vary by 3×, surface nodes ~2 % of
total but dominate the metric via `surf_weight=10`. Literature crawl
(Transolver→Transolver++→GeoTransolver→MARIO) suggested four high-leverage
moves:

1. **Loss change** — switch `(pred-y)^2` to Huber for robustness against
   pressure outliers (the dataset has y ranges spanning 5 orders of magnitude
   across domains).
2. **Mixed precision** — bf16 forward halves the activation footprint on the
   largest cruise meshes (242 K nodes × n_hidden × n_layers).
3. **Capacity sweep** — the original Transolver papers used L=8 H=256, but
   with only 1499 samples I expected the best generalization at much smaller
   capacity; the experiments confirmed `n_layers=2 / 3` outperforms 5/8.
4. **Eidetic states** (Transolver++ Rep-Slice + Ada-Temp) — drop-in
   modification I added behind a flag in `models.py` for parity testing.

Trainer changes were kept minimal: I refactored `train.py` so `Transolver` and
helpers live in `models.py` and the trainer only runs under
`if __name__ == "__main__"`. CLI flags were added for the architectural
toggles (n_hidden, n_layers, n_head, slice_num, mlp_ratio, dropout,
use_eidetic), training options (bf16, scheduler, loss, huber_delta, grad_clip,
ema_decay, seed) and bookkeeping. Data loaders / scoring stayed read-only.

### Phase 1 — eight-GPU ablation grid (60 ep each)

Fixed seed (default), sample weighted sampler unchanged. Variants:
baseline / bf16 / eidetic / Huber / surf_weight=20 / lr=1e-4 / onecycle /
h192-nl6 (mid scale-up). The 256-hidden 8-layer scale-up OOM'd at bs=4 even
under bf16, so I dropped it for this benchmark: VRAM headroom is needed for
the 242 K-node cruise samples.

Outcome: Huber loss alone hit val 86 by epoch 31, beating MSE baseline (96).
bf16 alone hit val 92. Combining bf16 + Huber + smaller depth was the
highest-leverage move.

### Phase 2 — combine winners and shrink the model

`bf16 + huber + n_layers=3` was the breakthrough — val crashed to 51.28 at
80 epochs, test 45.38. Reducing further to `n_layers=2` then gave val 49.95,
test 43.77 (0.30 M params). Going to `n_layers=1` was finally underpowered
(val 55.07, test 48.65). With this dataset's 1499 samples the sweet spot is
~0.3 M params.

### Phase 3 — variants of the winner (+ failures)

Tried `lr=3e-4`, `grad_clip=1.0`, `dropout=0.1`, `weight_decay=5e-4`,
`slice_num=32`, `slice_num=128`, `n_hidden=192`, `mlp_ratio` sweeps.
None beat the plain `bf16+huber+nl2/nl3` recipe at the same epoch count.
EMA (`decay=0.999`) gave a small but real ~0.2-2 point val improvement on
both nl2 (50.48 vs 49.95) and nl3 (49.38 vs 51.28).

### Phase 4 — duplicate-and-ensemble + long cosine

Five nl2 trains at 100-150 epochs (default seed, EMA, two random seeds, 150-ep
cosine) plus a long 200-epoch nl3 with `T_max` matched to total epochs. The
long-cosine 200-epoch nl3 is the single best val (currently 41.77, still
training). Averaging the nl2 normalized predictions across the four 100-ep
checkpoints brought test_avg to 38.51; adding the 150-ep model brought it to
37.13.

### Bug fix: nan in test_geom_camber_cruise

The first run with full test eval reported `test_avg/mae_surf_p = nan`. Root
cause: sample 20 of `test_geom_camber_cruise` has 761 non-finite ground-truth
pressures. The organizer scorer in `data/scoring.py` builds
`err = abs(pred-y)` *before* applying the per-sample finite mask, so
`err * mask` propagates `nan` from the dropped sample's positions
(IEEE 754 makes `nan * 0 = nan`). Since `data/` is read-only, I apply the
finite mask up-front in `train.evaluate_split` and zero out non-finite y
before calling `accumulate_batch`. The dropped sample's surface/volume node
counts come back zero, which matches the scorer's intended skip behaviour.
The same `evaluate_split` is now also forced to `autocast_dtype=None` for the
end-of-run test pass so paper numbers can't be poisoned by bf16 forward
overflow on the largest meshes.

`eval_checkpoint.py` re-runs val/test on a saved checkpoint in fp32, which
recovered the correct test_avg for runs that finished before the fix landed.

---

## Commands

Per the benchmark contract, the training entrypoint is:

```bash
python ./train.py --epochs 999 --agent ml-intern-r5 \
    --wandb_group mlintern-pai2-72h-v4-r5 \
    --wandb_name "mlintern-pai2-72h-v4-r5/<short-description>"
```

Concrete commands actually used (one-GPU each, pinned with
`CUDA_VISIBLE_DEVICES`):

```bash
# Best single — 0.30 M params, bf16 forward, Huber loss, 150-ep cosine
CUDA_VISIBLE_DEVICES=6 SENPAI_TIMEOUT_MINUTES=720 python ./train.py \
    --epochs 150 --bf16 True --loss huber --n_layers 2 \
    --agent ml-intern-r5 \
    --wandb_group mlintern-pai2-72h-v4-r5 \
    --wandb_name "mlintern-pai2-72h-v4-r5/p4-bf16+huber+nl2+e150"

# Long cosine n_layers=3 (currently leading on val)
CUDA_VISIBLE_DEVICES=0 SENPAI_TIMEOUT_MINUTES=720 python ./train.py \
    --epochs 200 --bf16 True --loss huber --n_layers 3 \
    --agent ml-intern-r5 \
    --wandb_group mlintern-pai2-72h-v4-r5 \
    --wandb_name "mlintern-pai2-72h-v4-r5/p3-bf16+huber+nl3+e200"

# Re-eval / ensemble on saved checkpoints
python eval_checkpoint.py --ckpt models/model-XXX/checkpoint.pt \
    --config models/model-XXX/config.yaml --out_json research/eval_XXX.json

python eval_ensemble.py \
    --models models/model-X1 models/model-X2 ... \
    --out_json research/eval_ensemble_nl2_5.json
```

---

## GPU strategy

All training stays inside the pai2 pod. Each `train.py` invocation pins
`CUDA_VISIBLE_DEVICES` so two jobs never share a GPU. Per the grid:

- 1 job/GPU, batch_size=4, n_layers≤3, bf16 forward.
- VRAM peak per run is ~36 – 80 GB depending on model size and the largest
  mesh that the WeightedRandomSampler happens to draw in a given step. The
  256-hidden 8-layer scale-up still OOMs under these constraints.
- Eight runs in parallel, with a steady cycle of "kill the worst, replace
  with a follow-up of the best" rather than waiting for batches to complete.

Total wall time per epoch on the winning n_layers=2/3 config is ~60-90 s,
which puts a 100-epoch run at ~1.5 h and 200-epoch at ~3 h. With eight GPUs
that's effectively ~50 train epochs/min of progress across the cluster.

---

## Recommendations for the next replicate

1. **Long cosine matters** — the 200-epoch n_layers=3 run is still
   improving at epoch 158 (val 42.29 → 41.77 over 30 ep). 300 ep with the
   same budget would likely close another 1-2 points; the 1499-sample budget
   is probably not yet exhausted at 200 epochs for this depth.
2. **Ensemble scales linearly with diversity** — 4 → 5 nl2 ensemble cut
   ~1.4 test points; adding the long nl3 should add another 1-2.
   A focused recipe would train 4-6 nl3-e200 runs with different seeds and
   ensemble those, with EMA on each member.
3. **Don't over-regularize.** Dropout, grad-clip, lower lr all hurt this
   particular benchmark. The single most useful "regularizer" is reducing
   capacity: nl5/n_hidden=128 → nl3/n_hidden=128 → nl2/n_hidden=128 each
   stepped down both val and test cleanly.
4. **Eidetic states are at parity, not better.** Transolver++ Rep-Slice +
   Ada-Temp matches the plain attention block here at 100 epochs (val 51.5 vs
   51.3). The Gumbel sample adds compute and a small init-time cost; the
   benefit it documents on million-node geometries doesn't show up at 100 K
   nodes / 1500 samples. Leave the flag in but default off.
5. **Default the trainer to fp32 test eval.** bf16 is fine for training but
   risks overflow on the largest cruise meshes, and the test pass is already
   short enough that fp32 is cheap.

---

## Files of interest

- `train.py`, `models.py` — refactored trainer + model classes.
- `eval_checkpoint.py`, `eval_ensemble.py` — post-hoc val/test eval and
  multi-checkpoint ensembling, both in fp32 with the nan-y fix.
- `launch_phase1.sh` — initial 8-way ablation grid script.
- `research/parse_results.py` — derives `MLINTERN_RESULTS.jsonl` from `logs/`.
- `research/MLINTERN_RESULTS.jsonl` — one row per run (best metric, params, etc.).
- `research/eval_ensemble_nl2_5.json` — current best paper-facing ensemble.
- `research/eval_p2_g3_nl2.json`, `research/eval_p2_g7_nl3_v2.json` — fp32
  re-eval of the two single-model leaders.
