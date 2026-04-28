# SENPAI Research Results — willow-pai2d-r1

## 2026-04-28 00:21 — PR #359 (merged, new baseline): bf16 autocast on forward + loss

- branch: `willowpai2d1-alphonse/bf16-autocast` (deleted on merge)
- hypothesis: throughput, not capacity, is the binding constraint at 30-min
  cap. bf16 autocast on forward + loss should shorten per-epoch wall enough
  to actually exercise the cosine schedule and improve val_avg/mae_surf_p.
  Predicted -3% to -10%.

### Results

| Metric | Value | Δ vs prior baseline (PR #312) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **121.8478** (epoch 16 of 19 completed) | **−15.5%** |
| `test_avg/mae_surf_p` | **111.1495** | **−15.3%** |
| Per-epoch wall | 96–99 s (mean ~97 s) | **−26%** |
| Epochs completed | 19 / 50 | +5 epochs |
| Peak GPU memory | 32.9 GB / 96 GB | **−22%** (~63 GB headroom) |
| Optimizer steps | 19 × 375 = 7,125 | n/a (≈+36% vs baseline 5,250) |
| W&B run | `ot9decu8` (`bf16-bsz4`) | — |

Per-split val (best ckpt): single 141.24 / rc 130.28 / cruise 99.83 / re_rand 116.04.
Per-split test: single 123.73 / rc 121.54 / cruise 85.65 / re_rand 113.68.

### Analysis & conclusions

- **Merged as new round-1 baseline.** BASELINE.md updated.
- bf16 autocast was the single highest-value experiment of the round so far —
  prediction of "throughput first, then capacity" is now solidly evidenced.
- The 26% per-epoch speedup is below the upper end of the predicted 1.5–2×;
  the model is small (0.66M params reported by the printed banner) so
  CPU-side dataloader/normalization is a meaningful share of the step.
  Plenty of compute upside remains.
- No bf16 numerical issues observed. All forward steps and per-split val
  losses were finite. The single `loss=NaN` print on `test_geom_camber_cruise`
  is from `train.py::evaluate_split`'s normalised-loss accumulator (which
  doesn't filter non-finite-y) — same cosmetic bug fern flagged on PR #360,
  unrelated to bf16.
- **Headroom snapshot post-merge**: 63 GB of GPU memory free, 31 epochs of
  unused budget per run if the schedule could decay properly. Capacity
  scale-up experiments that were untestable at the old throughput are now
  feasible.

## 2026-04-28 00:02 — PR #360 (closed): Larger batch (bsz=8, lr=7.07e-4)

- branch: `willowpai2d1-fern/batch-size-8-lr-scaled` (deleted on close)
- hypothesis: doubling batch size from 4→8 with sqrt-scaled lr gives a
  meaningful per-epoch wall-clock speedup, letting more cosine schedule run
  inside the 30-min cap. Predicted -2% to -7%.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **148.7170** (epoch 13 of 14 completed) |
| `test_avg/mae_surf_p` | 136.3675 |
| Per-epoch wall | 129.5 s (vs baseline 131 s — **-1.1%, no win**) |
| Optimizer steps | 2444 (vs baseline 5250 — **-53%**) |
| Peak GPU memory | 84.2 GB / 96 GB |
| Stability | clean (no NaN, lr=7.07e-4 was safe) |
| W&B run | `6977miuh` (`bsz8-lr7e-4`) |

vs baseline: val_avg **+3.12%**, test_avg **+3.95%**. Below 5% close
threshold, but ruled-out direction with strong diagnosis.

### Analysis & conclusions

- **Closed.** Negative result, but high information value.
- **Trainer is not kernel-launch-bound at bsz=4.** Doubling B doubled HBM
  traffic and padded-node count, so per-step compute roughly doubled →
  per-epoch time barely changed → fewer gradient updates → worse
  convergence.
- **Padding waste is the real bottleneck.** `pad_collate` pads to max-N in
  the batch; with cruise meshes at ~210K and raceCar single at ~85K, every
  random-composition batch is padded to ~210K nodes regardless of how many
  small samples it contains. Activations scaled near-linearly with B (42 →
  84 GB).
- **Redirect:** fern's own followup — *domain-bucketed batch sampler* — is
  the right next move; assigned as PR #384.
- The wider+deeper scale-up is parked until throughput is unblocked.
- Cosmetic NaN: train.py's `evaluate_split` still uses `(pred-y_norm)**2`
  with no finite-y filter, so the printed test/cruise normalised loss
  shows NaN. Doesn't affect MAE rankings (those go through the patched
  accumulator). Not fixed.

## 2026-04-27 23:15 — PR #312: Round-1 reference baseline (Transolver default config)

- branch: `willowpai2d1-alphonse/baseline-default`
- hypothesis: establish a clean reference number on the advisor branch by
  running the default Transolver config unchanged.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **144.2118** (epoch 10 of 14 completed) |
| `test_avg/mae_surf_p` | **131.1823** (best val checkpoint) |
| Wall time | 30-min `SENPAI_TIMEOUT_MINUTES` binding (~131 s/epoch) |
| Peak GPU memory | 42.1 GB / 96 GB |
| W&B run | `x33nmv34` ([link](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/x33nmv34)) |

Per-split val (best checkpoint): single 169.70 / rc 170.34 / cruise 110.70 /
re_rand 126.11. Per-split test: single 150.39 / rc 155.05 / cruise 93.29 /
re_rand 125.99.

### Analysis & conclusions

- **Merged** as round-1 baseline. BASELINE.md updated.
- **Bug found and fixed.** Alphonse diagnosed a `0 * Inf = NaN` poisoning
  in `data/scoring.py` triggered by `test_geom_camber_cruise/000020.pt`
  (the only Inf-pressure sample in the corpus). I cherry-picked the
  documented fix into commit `b78f404` on the advisor branch — sibling PRs
  now report finite `test_avg/mae_surf_p`.
- **Throughput is the binding constraint** at the current model size.
  Only 14 of 50 epochs ran; the cosine schedule barely decayed; VRAM is
  >50 GB underutilised. AMP/bf16, larger batch, or `torch.compile` are the
  obvious first knobs and should be the highest-priority next experiment.
- Among val splits, cruise is easiest, then re_rand, then rc/single. Val
  and test rankings agree, which is a useful sanity check on the four-track
  split design.

## 2026-04-27 23:22 — PR #321 (round 1, sent back): 5-epoch LR warmup + cosine to 0 with peak lr=1e-3

- branch: `willowpai2d1-frieren/lr-warmup-and-higher-peak`
- hypothesis: warmup + higher peak lr (1e-3) improves over default no-warmup
  cosine from 5e-4. Predicted -2% to -5% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **148.38** (epoch 14 of 14 completed) |
| `test_avg/mae_surf_p` | NaN (scoring bug + non-finite cruise pred) |
| Wall time | 30-min cap binding (~132 s/epoch) |
| Peak GPU memory | 42.1 GB / 96 GB |
| W&B run | `4ba8w3wb` (`warmup5-peak1e3`) |

vs baseline (PR #312, 144.21): **+2.9% regression**. Under the 5% close
threshold; sent back, not closed.

### Analysis & conclusions

- **Sent back** for variation: peak=7e-4 instead of 1e-3.
- Frieren caught a real bug in my pseudocode (`LinearLR` requires
  `end_factor ≤ 1`; my literal version would also leave cosine annealing
  from `1e-5` to `0` due to base_lrs capture). Their reimplementation is
  semantically correct.
- The peak=1e-3 caused a val regression at epochs 6-7 right after warmup
  ended (val: 178 → 254 → 259 → 178), which is exactly what the student
  flagged as a likely problem. peak=7e-4 should be calmer.
- Schedule never reached cosine tail (only 14/50 epochs ran, lr at end
  was still ~9e-4). Throughput PRs (#359 bf16, #360 bsz=8) will let
  warmup+cosine actually be evaluated to convergence in a future round.
- Frieren also separately reported non-finite *predictions* on a cruise
  test sample (vol_loss=inf in normalized space). The scoring fix in
  b78f404 removes the test-cruise NaN from the *scoring*-side, but a
  blown-up prediction is a separate model-stability concern that should
  be calmer with a lower peak lr.

## 2026-04-27 23:16 — PR #318: Wider+deeper Transolver (h=192, L=6, heads=6, slices=96)

- branch: `willowpai2d1-fern/wider-deeper-transolver` (deleted on close)
- hypothesis: scale Transolver up — h=192, L=6, heads=6, slice_num=96 —
  predicted -3% to -8% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| Best `val_avg/mae_surf_p` | **175.8511** (epoch 3 of 7 completed) |
| `test_avg/mae_surf_p` | **NaN** (epoch-3 checkpoint produced non-finite p on a cruise test sample, *and* pre-fix scoring NaN-poisoning was active) |
| Wall time | 30-min cap binding (~275 s/epoch, 2.1× baseline cost) |
| Peak GPU memory | 83.8 GB / 96 GB |
| Param count | 1.72 M |
| W&B run | `rzn96bqj` (`h192-l6-h6-s96`) |

vs baseline (PR #312): val_avg = 175.85 vs 144.21 → **+22% regression**.

### Analysis & conclusions

- **Closed.** Result is well past the >5% close threshold, but the
  underlying hypothesis (more capacity ⇒ better generalization) was *not
  fairly tested*: only 7 of 50 epochs ran in the budget, the model was still
  oscillating downward, and the cosine LR barely moved.
- Fern's analysis was excellent: hypothesis untestable at this throughput,
  not falsified.
- **Redirect:** the right next step is throughput improvement first
  (AMP/bf16, larger batch given 50+ GB headroom, possibly `torch.compile`),
  *then* a half-step scale-up. I've assigned that as fern's round-1.5
  experiment.
