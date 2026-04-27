# SENPAI Research Results — willow-pai2d-r1

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
