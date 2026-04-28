# SENPAI Research Results — willow-pai2d-r1

## 2026-04-28 01:54 — PR #314 (sent back): SmoothL1 / Huber loss (β=1.0)

- branch: `willowpai2d1-edward/huber-loss` (in flight as draft after send-back)
- hypothesis: replace MSE with SmoothL1 (β=1.0) to align loss with the MAE
  metric and bound gradients on high-Re outliers. Predicted -2 to -5%.

### Results (vs bf16 baseline, but BEFORE FF merge)

| Metric | Value | vs PR #359 (bf16) | vs PR #327 (FF, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **104.2658** (epoch 17 of 19) | **−14.4%** | **−2.5% (still beats FF baseline!)** |
| `test_avg/mae_surf_p` | **92.1301** | **−17.1%** | **−4.8%** |
| Per-epoch wall | ~97 s | ≈baseline | ≈baseline |
| Peak GPU memory | ~95.7 GB transient (epochs 1-2), ~47 GB steady | +14 GB steady | +14 GB steady |
| Epochs completed | 19 / 50 | same | same |
| W&B run | `czpoam0v` (`smoothl1-beta1`) | | |

### Per-split val (epoch 17 best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| val_single_in_dist | 136.57 | 1.83 | 0.82 | 131.62 |
| val_geom_camber_rc | 112.14 | 2.13 | 0.96 | 114.45 |
| val_geom_camber_cruise | 75.12 | 0.98 | 0.55 | 71.05 |
| val_re_rand | 93.23 | 1.56 | 0.75 | 89.04 |
| **val_avg** | **104.27** | 1.62 | 0.77 | 101.54 |

### Analysis & conclusions

- **Sent back, not merged.** Edward rebased onto pre-FF advisor branch
  (tip 0069451) but the FF merge happened at the same time (f17992d).
  Edward's run is essentially Huber+bf16 with no FF. Even so, the result
  (104.27) is *already below the FF baseline* (106.92) — Huber alone is
  a stronger lever than FF.
- The right test is **Huber on top of FF** (orthogonal mechanisms): FF
  improves spatial frequency representation, Huber bounds gradients on
  high-Re outliers. Predicted to stack to val_avg ~90.
- Memory anomaly (transient 95.7 GB peak in epochs 1-2) is from the
  caching allocator; SmoothL1 has more autograd intermediates than MSE.
  Steady-state +14 GB. Means we can't safely scale batch_size with Huber
  without instrumentation — flagged for future PRs.
- Per-split improvement is **uniform** (unlike pressure-weighting which
  was heavily split-dependent). All four splits improve substantially.

## 2026-04-28 01:54 — PR #313 (closed, rebased run): Pressure-weighted MSE on bf16

- branch: `willowpai2d1-askeladd/pressure-channel-loss-weight` (deleted on close)
- hypothesis: same as original #313 (pressure-weighted MSE 5x p), but rebased
  onto bf16 baseline.

### Results

| Metric | Value | vs PR #359 (bf16, rebase target) | vs PR #327 (FF, current baseline) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | 122.5350 (epoch 19 of 19) | **+0.56%** (flat) | **+14.6%** (regression) |
| `test_avg/mae_surf_p` | 112.7175 | +1.41% | +16.4% |
| W&B run | `wf3av0ps` | | |

### Per-split deltas vs bf16 baseline

| Split | Δ |
|---|---|
| val_single_in_dist | +4.19% |
| val_geom_camber_rc | **+5.26%** (sign flip — was −14% pre-bf16) |
| val_geom_camber_cruise | −5.65% |
| val_re_rand | −3.78% |

### Analysis & conclusions

- **Closed.** Excellent diagnosis from askeladd: pressure weighting and
  bf16 are *not* orthogonal — both addressed the same underlying issue
  (high-Re pressure samples dominating gradient). Once bf16 closed the
  rc-camber gap (170.34 → 130.28 between PR #312 and PR #359), the
  additional pressure weighting just starves Ux/Uy. Surface Ux/Uy errors
  went +25-40% across every split — that's the smoking gun.
- The orthogonality assumption in my send-back was wrong. Documented for
  the future.
- Followup queued and assigned: surface-only pressure weighting (askeladd
  PR #451). Restricting (1,1,5) to surf_loss only might preserve cruise /
  re_rand wins without volume Ux/Uy damage.

## 2026-04-28 01:50 — PR #327 (merged, NEW BASELINE): Sinusoidal Fourier features for (x, z), K=8

- branch: `willowpai2d1-tanjiro/fourier-features-positions` (deleted on merge)
- hypothesis: concatenate sinusoidal Fourier features
  `[sin(2^k π x), cos(2^k π x), sin(2^k π z), cos(2^k π z)] for k=0..7` to
  the per-node feature vector. Predicted -2 to -6%.

### Results

| Metric | Value | Δ vs prior baseline (PR #359, bf16) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **106.9223** (epoch 19 of 19) | **−12.2%** |
| `test_avg/mae_surf_p` | **96.8186** | **−12.9%** |
| Per-epoch wall | 97-100 s (mean ~98) | ≈baseline (FF compute trivial) |
| Peak GPU memory | 33.3 GB / 96 GB | +0.4 GB (negligible) |
| Param count | 0.67 M | +0.01 M |
| Epochs completed | 19 / 50 | same |
| W&B run | `nbyicdne` (`ff-K8-bf16`) | — |

### Per-split surface MAE (val, vs prior bf16 baseline)

| Split | Δ |
|---|---|
| val_single_in_dist | **−17.0%** (141.24 → 117.22) |
| val_geom_camber_rc | −3.3% (130.28 → 125.94) |
| val_geom_camber_cruise | **−19.6%** (99.83 → 80.26) |
| val_re_rand | −10.2% (116.04 → 104.27) |

### Analysis & conclusions

- **Merged. New round baseline.** Largest single win of the round so far.
- Tanjiro rebased onto current advisor branch before running so this is
  apples-to-apples vs the bf16 baseline. Excellent practice.
- FF computed in fp32 *before* the bf16 autocast scope so sin/cos aren't
  quantised. Right composition pattern.
- Train loss drops faster per epoch with FF: by epoch 5 train_surf is ~0.41
  vs ~0.50 for bf16-baseline at the same point. Canonical Tancik et al.
  (2020) Fourier-features behaviour.
- **Per-split asymmetry is informative:** cruise and single-in-dist
  improve ~17–20%, val_re_rand 10%, but rc-camber held-out only 3-4%. The
  rc holdout is bottlenecked by *camber → pressure mapping* under unseen
  geometry, not by spatial-frequency representation. Targeted future
  experiment direction.
- Followup directions queued: Gaussian random Fourier features (Tancik's
  variant, often beats deterministic ladder), K sweep, FF on saf/dsdf.

## 2026-04-28 01:03 — PR #393 (closed): Half-step capacity scale-up on bf16 (h=160, L=5, heads=5, slices=80)

- branch: `willowpai2d1-alphonse/halfstep-capacity-on-bf16` (deleted on close)
- hypothesis: with bf16 throughput unlocked + ~63 GB headroom, the bigger
  model that fern's #318 couldn't test is finally feasible. Predicted -2 to -7%.

### Results

| Metric | Value | vs PR #359 (bf16 baseline) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **131.05** (epoch 11 of 14) | **+7.55%** |
| `test_avg/mae_surf_p` | 114.89 | +3.37% |
| Per-epoch wall (mean) | 135 s | +39% |
| Peak GPU memory | 45.6 GB / 96 GB | +38.6% |
| Param count | 1.02 M | +54% |
| Epochs completed | 14 / 50 | -5 |
| W&B run | `nbkqn78z` | — |

### Analysis & conclusions

- **Closed.** val regression past 5% threshold, but well-diagnosed —
  schedule-budget mismatch dominates: best at epoch 11 of 14 with val
  curve still descending; lr was still ~41% of peak at the timeout because
  `T_max=50` doesn't decay in 14 epochs.
- test_avg only +3.37% (vs val +7.55%) — the bigger model isn't
  fundamentally worse, just under-converged.
- **Capacity hypothesis parked, not abandoned.** Once fern's #407 cosine
  T_max alignment lands, retest capacity against the proper schedule.
- Per-epoch wall scales roughly linearly with parameter count (1.55× params
  → 1.39× wall) — no compute pathology.

## 2026-04-28 00:49 — PR #384 (closed): Domain-bucketed batch sampler

- branch: `willowpai2d1-fern/domain-bucketed-sampler` (deleted on close)
- hypothesis: bucket batches by domain so each batch is homogeneous in mesh
  size, cutting padding waste from `pad_collate`. Predicted ~1.2-1.5×
  per-epoch speedup; predicted -2% to -6% on val_avg.

### Results

| Metric | Value | vs PR #359 baseline (bf16) |
|---|---|---|
| Best `val_avg/mae_surf_p` | **125.91** (epoch 15 of 16) | **+3.3%** (worse) |
| `test_avg/mae_surf_p` | 115.40 | +3.8% (worse) |
| Per-epoch wall (mean) | **115.2 s** | **+17%** (slower!) |
| Peak GPU memory | **42.1 GB** | **+28%** (more) |
| Epochs completed | 16 / 50 | -3 |
| W&B run | `00fl62dc` | — |

vs old (pre-bf16) baseline: −12.7%, but ranking is now against the new
post-bf16 baseline.

### Analysis & conclusions

- **Closed.** Hypothesis falsified — bucketing made throughput *worse*, not
  better. Two-mechanism explanation from fern is convincing:
  1. **CUDA caching-allocator fragmentation.** Cycling between 3 max_n
     shapes (4×210K, 4×127K, 4×85K) defeats the allocator's pool reuse —
     the old WeightedRandomSampler had ~80% of batches padding to ~210K so
     the allocator settled on a single dominant pool. Bucketing forces
     three pools.
  2. **Dataloader pipeline mismatch.** GPU step time varies ~2.5× across
     domains while pad_collate worker time is roughly constant; bucketing
     breaks the worker-GPU pipeline overlap.
- The +28% memory regression is consistent with allocator fragmentation;
  the +17% wall regression is consistent with both mechanisms.
- **Throughput-via-sampler is ruled out** as a quick win on this trainer.
  Future throughput attempts should look at `torch.compile`, attention
  flavor swaps, or gradient checkpointing (when scaling capacity), not
  sampler shape.
- Followups parked: length-bucket only the cruise outlier, sort-and-bucket
  NLP-style, memory_history diagnostic. None promising enough to assign
  immediately.

## 2026-04-28 00:20 — PR #313 (sent back): Pressure-channel-weighted MSE (5x p)

- branch: `willowpai2d1-askeladd/pressure-channel-loss-weight`
- hypothesis: weight pressure 5× in the per-channel MSE to align loss with
  the metric. Predicted -3% to -8%.

### Results (pre-bf16 run)

| Metric | Value | vs PR #312 (old baseline) | vs PR #359 (NEW baseline 121.85) |
|---|---|---|---|
| Best `val_avg/mae_surf_p` | **138.1556** (epoch 14 of 14) | **−4.20%** | **+13.4%** |
| `test_avg/mae_surf_p` | 125.9776 | −3.97% | +13.3% |
| W&B run | `gxcli1lf` (`p-weight-5x`) | | |

### Per-split deltas (val, vs old baseline)

| Split | Δ vs baseline |
|---|---|
| val_single_in_dist | **+2.4%** (regressed) |
| val_geom_camber_rc | **−14.0%** (huge OOD geometry win) |
| val_geom_camber_cruise | −5.4% |
| val_re_rand | +1.2% (flat) |

### Analysis & conclusions

- **Sent back, not merged or closed.** Win on prior baseline was clean
  (−4.2% on val_avg) but the run preceded the bf16 merge (PR #359). Vs the
  new bf16 baseline (121.85), this is +13.4% — but pressure weighting and
  bf16 are orthogonal, so expected behavior is they stack.
- **Action:** rebase onto post-bf16 advisor branch and re-run.
- Excellent per-split characterisation: pressure weighting helps most where
  pressure dominates the surface-error budget (geom_camber_rc, the hardest
  split). Slight regression on the easier in-dist split because relative
  velocity-channel gradient drops.
- Suggested followups (queued): sweep over weights (1,1,3 / 1,1,5 / 1,1,8),
  decoupled surf vs vol channel weights.

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
