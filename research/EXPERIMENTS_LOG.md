# SENPAI Research Results — charlie-pai2d-r3

## 2026-04-28 00:35 — PR #298 (CLOSED, positive on MSE / loses to L1): 8-freq Fourier positional features
- Branch: `charliepai2d3-nezuko/fourier-pos-features` (deleted on close)
- Hypothesis: Fourier positional encoding of `(x, z)` at 8 octave-spaced
  frequencies addresses MLP/attention spectral bias against high-frequency
  content of low-d inputs; predicted −2% to −8%.
- Config: MSE surface loss (pre-L1 advisor), all other knobs at defaults.
  +1.2% params (32 extra input channels at the first preprocess MLP).

### Headline (best-val checkpoint, epoch 13/14)

| Metric | This PR | vs PR #306 (MSE peer, 135.20) | vs PR #280 (L1, 102.64) |
|--------|--------:|------------------------------:|------------------------:|
| `val_avg/mae_surf_p`  | 116.62 | **−13.7% (win on MSE)** | +13.6% (loses to L1) |
| `test_avg/mae_surf_p` | 105.85 | **−14.1%** | +8.3% |
| Peak GPU memory       | 42.36 GB | — | — |
| Param count           | 670,551 | +1.2% (+8,192 weights) | — |
| Epochs in 30-min cap  | 14/50 | — | — |

### Per-split val (best epoch 13)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 138.75 |
| val_geom_camber_rc     | 122.13 |
| val_geom_camber_cruise |  95.00 |
| val_re_rand            | 110.60 |

### Decision

**Closed.** The hypothesis was validated on the MSE baseline assigned to
this branch (−13.7% on val), but L1 surface loss landed mid-round and
became the new baseline at val 102.64 — a much larger lever than Fourier
features. Per the merge criterion (must beat current baseline), this PR
does not merge. Per the close criterion (>5% regression vs current
baseline), it is technically closeable — but the regression is an
artefact of the merge-order race, not a failure of the lever.

**The lever is on the round-4 candidate list** as L1 + Fourier features.
The student is the right person to run that compose test (already owns
the code). They've been re-assigned to test exactly that.

**Useful per-split insight**: student observed the worst val split was
`val_single_in_dist` (138.75), inverting their prediction that
single-in-dist would benefit most from Fourier features. The split
ranking is dominated by raceCar high-Re extremes, not by spectral bias
on input position — which points round 5 toward applying Fourier
features to `log(Re)` and other scalar inputs (student's own follow-up #2).

**On bug fix**: student's pred-side workaround in `evaluate_split` is
redundant on the current advisor branch — the GT-side fix landed as
commit `2eb5c7f` per thorfinn/alphonse's earlier identification.
Student validated the merged approach (their option (2) recommendation
matches the merged fix exactly).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:30 — PR #288 (CLOSED): 3-epoch warmup + cosine to 1e-5, peak lr=1e-3
- Branch: `charliepai2d3-fern/lr-warmup-peak1e3` (deleted on close)
- Hypothesis: warmup unlocks higher peak LR; cosine to `eta_min=1e-5`
  preserves late-training fine-tune; predicted −2% to −5%.
- Config: MSE surface loss (pre-L1 advisor), `bs=4`, `lr=1e-3`,
  3-epoch LinearLR warmup with `start_factor=0.1`, then
  `CosineAnnealingLR(T_max=47, eta_min=1e-5)`.
- Diff: ~6 lines of imports + scheduler swap in `train.py`.

### Headline (best-val checkpoint, run 3 of 3 seeded re-runs)

| Metric | This PR | vs PR #306 baseline (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|-----------------------------:|---------------------------------:|
| `val_avg/mae_surf_p` (best, epoch 13/14) | 147.50 | +9.1% | +43.7% |
| `test_avg/mae_surf_p` | 130.55 | +6.0% | +33.5% |
| Peak GPU memory       | 42.11 GB | — | — |
| Per-epoch wallclock   | ~131 s   | — | — |

### Cross-run variance (3 unseeded runs)

| run | best epoch | best val_avg/mae_surf_p |
|-----|---------:|-----------------------:|
| v1  | 12       | 136.88 |
| v2  | 9        | 145.12 |
| v3 (canonical) | 13 | 147.50 |
| **mean ± std** | — | **143.2 ± 5.7** |

Even the **best** run (136.88) does not beat the prior MSE baseline
(135.20). Cross-run std ~5.7 is large enough to swamp ~5% schedule
effects — flagged as round-4 infra debt (seed pinning).

### Decision

**Closed.** >5% regression on the primary ranking metric across 3 seeded
re-runs. The student's analysis nailed the failure mode: this is a
long-horizon optimizer change being evaluated under a short-horizon
wallclock cap. Three structural problems compound:

1. Warmup eats 21% of the actual epoch budget (3/14, vs the 6% the
   schedule was designed for).
2. Higher peak LR amplifies seed noise at bs=4 — bouncy descent shows
   the optimizer can't settle in 11 post-warmup epochs.
3. `eta_min=1e-5` is irrelevant — the LR is still ~9e-4 at the timeout.

The corrective experiment (matched-cosine `T_max=14`) is being run by
askeladd in PR #389 on the L1 baseline. A "1-epoch warmup + matched
cosine" variant would be a reasonable round-5 PR if #389 wins.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:30 — PR #292 (CLOSED): slice_num 64 → 128
- Branch: `charliepai2d3-frieren/slice-num-128` (deleted on close)
- Hypothesis: doubling PhysicsAttention slice tokens halves the
  per-token mesh-node neighborhood and lets the slice basis represent
  finer flow structure; predicted −3% to −8%.
- Config: MSE surface loss (pre-L1 advisor), all other knobs at
  defaults. Single-line diff: `slice_num=128`.

### Headline (best-val checkpoint, epoch 9/11)

| Metric | This PR | vs PR #306 baseline (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|-----------------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 149.08 | +10.3% | +45.3% |
| `test_avg/mae_surf_p` | 136.85 | +11.1% | +40.0% |
| Peak GPU memory       | 54.5 GB | — | — |
| Param count           | 0.67 M | +2% vs slice_num=64 | — |
| Epochs in 30-min cap  | 11/50 | — | — |

### Per-split val (best epoch 9)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 193.75 |
| val_geom_camber_rc     | 160.05 |
| val_geom_camber_cruise | 109.81 |
| val_re_rand            | 132.70 |

### Decision

**Closed.** The student's variance observation is the key takeaway: a
separate identical-config run hit val 142.76 at epoch 11 vs 149.08 here
— ~4% spread from sampler/init noise alone, comparable to the predicted
effect size. With only 11 of 50 epochs and the cosine never decaying,
the signal-to-noise ratio for slice count vs noise was too low to
attribute anything cleanly.

slice_num=128 is **not ruled out** for round 4 — it just needs either
tighter variance control (seeded runs) or a much larger expected effect.
The +2% param bump and 54.5 GB peak memory at slice_num=128 confirm
plenty of headroom for slice_num=256 once throughput is unlocked.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` —
branch deleted on close.

---

## 2026-04-28 00:18 — PR #283 (CLOSED): Wider+deeper Transolver (h=192, l=6, head=6, slices=96)
- Branch: `charliepai2d3-askeladd/wider-deeper-h192-l6-s96` (deleted on close)
- Hypothesis: scale capacity along 4 axes simultaneously; predicted −3% to −8%.
- Config: `bs=4`, `lr=5e-4`, all other knobs at defaults.

### Headline (best-val checkpoint, epoch 7/7)

| Metric | This PR | vs original baseline (PR #306, 135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|---------------------------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 166.64 | +23.3% | +62.4% |
| `test_avg/mae_surf_p` | 155.95 | +26.6% | +59.6% |
| Per-epoch wallclock   | 278 s  | 2.1× slower than baseline shape | — |
| Peak GPU memory       | 83.84 GB | / 96 GB cap → 12 GB headroom | — |
| Epochs in 30-min cap  | 7/50   | half of baseline shape | — |
| Param count           | 1.72 M | +1.0 M vs baseline | — |

### Per-split val (best epoch 7)

| split | mae_surf_p |
|-------|-----------:|
| val_single_in_dist     | 198.42 |
| val_geom_camber_rc     | 183.45 |
| val_geom_camber_cruise | 140.69 |
| val_re_rand            | 144.01 |

### Decision

**Closed.** >5% regression on val_avg/mae_surf_p vs both prior baselines.
Per-epoch the bigger model is **12% better** than the round-3 baseline shape
at matched epoch index (epoch 7: 166.64 vs 188.54), so the architecture has
genuine merit — but the 2.1× per-epoch slowdown halves the cosine-anneal
budget under the 30-min cap, wiping out the per-epoch gain. Compute
starvation is structural; revisits blocked on throughput infra
(mixed-precision / activation checkpointing).

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` — branch
was deleted on close before metrics could be cherry-picked into the
advisor branch. Headline numbers above are from the PR results comment.

### Round-4 implications

- Joint scaling on 4 axes is too coarse for clean attribution. Per-axis
  PRs (frieren #292 slice_num, in flight) will give cleaner signals.
- Drop "wider+deeper" from round-4 candidate set until throughput infra
  lands — at that point a single-axis bigger-model PR is justified.
- Student also flagged a pred-side `evaluate_split` y-finite bug fix
  worth keeping in mind if any future PR produces NaN test averages
  from clean GT (current scoring fix only handles GT-side non-finite).

---

## 2026-04-28 00:15 — PR #366 (CLOSED): mlp_ratio 2 → 4
- Branch: `charliepai2d3-thorfinn/mlp-ratio-4` (deleted on close)
- Hypothesis: doubling MLP per-token capacity inside each TransolverBlock;
  predicted −3% to −8% on val_avg/mae_surf_p.
- Config: bs=8 → bs=6 (OOM at bs=8 with the wider MLP), `lr=7.07e-4`
  (kept from baseline PR #306 instructions).

### Headline (best-val checkpoint, epoch 11/13)

| Metric | This PR | vs PR #306 (135.20) | vs L1 baseline (PR #280, 102.64) |
|--------|--------:|--------------------:|---------------------------------:|
| `val_avg/mae_surf_p`  | 144.70 | +7.0%  | +41.0% |
| `test_avg/mae_surf_p` | 132.44 | +7.5%  | +35.5% |
| Peak GPU memory       | 78.16 GB | — | — |
| Epochs in 30-min cap  | 13/50  | −1 epoch vs baseline | — |
| Param count           | 0.99 M | +0.33 M  | — |

### Per-split val (best epoch 11) — **revealing pattern**

| split | This PR | vs PR #306 baseline |
|-------|--------:|--------------------:|
| val_single_in_dist     | 176.83 | **−7.0% (improved)** |
| val_geom_camber_rc     | 159.32 | +15.1% |
| val_geom_camber_cruise | 112.82 | +15.2% |
| val_re_rand            | 129.81 | +13.6% |
| **val_avg**            | 144.70 | +7.0% |

### Decision

**Closed.** >5% regression on val_avg/mae_surf_p vs both prior baselines.

The split pattern is the takeaway: in-distribution improved (single-foil
better fit), every OOD axis regressed. Classic generalisation-gap shift —
extra MLP capacity is being spent memorising training-distribution
structure that doesn't transfer to held-out cambers / Re. Validation
peaked at epoch 11 then degraded (144.70 → 159.04 → 171.83 across epochs
11→12→13), confirming overfit before the cosine could anneal.

Per-epoch metrics not centralised in `EXPERIMENT_METRICS.jsonl` — branch
was deleted on close. Headline numbers above are from the PR results
comment.

### Round-4 implications

- mlp_ratio=4 dropped from candidate set per the standalone-loss rule.
- Two follow-up directions remain interesting and would justify their own
  PRs if revisited: (a) `mlp_ratio=4` only in last 1-2 blocks (asymmetric
  capacity), (b) `mlp_ratio=4` paired with stronger regularisation
  (`dropout=0.05` or `weight_decay=2e-4`) to test whether the
  generalisation gap closes.
- The OOM at bs=8 with +0.33 M params is a useful VRAM-headroom signal:
  the bs=8 MSE baseline (PR #306) was running close to the limit.

---

## 2026-04-28 00:03 — PR #280: L1 surface loss to align gradient with reported MAE metric
- Branch: `charliepai2d3-alphonse/l1-surface-loss`
- Hypothesis: switching the surface loss from MSE to L1 (volume MSE
  unchanged) aligns the gradient with the reported `val_avg/mae_surf_p`
  metric and is more robust to the heavy-tailed high-Re samples.
- Config: `bs=4`, `lr=5e-4`, all other knobs at defaults; only loss changed.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best |
|------:|-------------------:|:----:|
| 1  | 244.06 | * |
| 4  | 198.10 | * |
| 8  | 131.46 | * |
| 11 | 113.55 | * |
| 13 | 105.84 | * |
| 14 | **102.64** | * |

Best epoch 14 (the final epoch); curve was still descending at termination.
Stopped at epoch 14 by the 30-min timeout. Full per-epoch metrics committed
at `models/model-charliepai2d3-alphonse-l1-surface-loss-20260427-223604/`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint, epoch 14)

| split | val mae_surf_p | test mae_surf_p (NaN-safe) |
|-------|---------------:|---------------------------:|
| single_in_dist     | 121.18 | 109.80 |
| geom_camber_rc     | 125.01 | 114.60 |
| geom_camber_cruise |  73.22 |  79.92 |
| re_rand            |  91.14 |  86.58 |
| **avg**            | **102.64** | **97.73** |

### Analysis

- **Wins on all four val splits vs the prior baseline (PR #306, val 135.20).**
  Biggest improvement on the hardest split, `val_single_in_dist`: 121.18 vs
  190.14 (−36%). The high-Re raceCar singles dominated the surface error
  before; L1 cuts that error sharply.
- −24.1% on `val_avg/mae_surf_p`, −20.6% on `test_avg/mae_surf_p`. Test < val
  on three of four splits → no overfit.
- The bs=4 / lr=5e-4 config used here is *different* from the prior baseline
  (bs=8 / lr=7.07e-4 / MSE). So the headline 24% win conflates L1 vs MSE
  with bs=4 vs bs=8. Per the bs-only test (PR #306 vs unknown bs=4 MSE),
  the bs effect was at most ~5%; L1 carries the rest.
- Peak memory only 42.13 GB at bs=4 — round 4 has plenty of room to push
  bs and capacity in combination with L1.

### Decision

**Merged** as the new round 3 baseline. Old baseline (PR #306, val 135.20)
becomes round-3 reference 1. New baseline `val_avg/mae_surf_p = 102.64`,
`test_avg/mae_surf_p = 97.73`. The seven other in-flight r3 PRs branched off
the pre-L1 advisor; their results need to clear 102.64 (val) to be winners.
Several are likely orthogonal to L1 and useful for round 4 composition even
if they don't beat the new baseline outright.

---

## 2026-04-27 23:26 — PR #306: Batch size 8 with sqrt LR scaling (lr=7.07e-4)
- Branch: `charliepai2d3-thorfinn/batch8-sqrt-lr`
- Hypothesis: doubling `batch_size` to 8 with √2-scaled LR (`5e-4 → 7.07e-4`)
  reduces gradient noise without changing the data budget; tests whether
  gradient quality alone improves convergence within the 30-min wallclock.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best | sec | peak_mem |
|------:|-------------------:|:----:|----:|---------:|
| 1  | 264.88 | * | 134.5 | 84.2 GB |
| 2  | 215.25 | * | 129.2 | 84.2 GB |
| 5  | 212.78 | * | 130.0 | 84.2 GB |
| 7  | 188.54 | * | 129.9 | 84.2 GB |
| 8  | 155.47 | * | 128.8 | 84.2 GB |
| 11 | 142.97 | * | 129.2 | 84.2 GB |
| 13 | **135.20** | * | 129.7 | 84.2 GB |
| 14 | 142.03 |   | 127.1 | 84.2 GB |

Best epoch 13/14. Stopped at epoch 14 by 30-min timeout (cosine T_max was
50 → never reached the tail). Full per-epoch metrics committed at
`models/model-charliepai2d3-thorfinn-batch8_sqrt_lr-20260427-223454/metrics.jsonl`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint)

| split | val mae_surf_p | test mae_surf_p (corrected) |
|-------|---------------:|----------------------------:|
| single_in_dist     | 190.14 | 173.01 |
| geom_camber_rc     | 138.39 | 120.22 |
| geom_camber_cruise |  97.95 |  82.83 |
| re_rand            | 114.32 | 116.53 |
| **avg**            | **135.20** | **123.15** |

### Analysis

- Run is stable; bs=8 + lr=7.07e-4 fits comfortably (peak 84.2 GB of 96 GB).
- The val curve was still descending at termination (epoch 13 = 135.20 vs
  epoch 11 = 142.97), so this is a **partially-trained model on a truncated
  cosine** — not a converged result.
- Test < val on three of four splits (single, rc, cruise) → no overfit.
- Cruise track (97.95 / 82.83) is by far the easiest; single-in-dist is the
  hardest (190.14 / 173.01) — high-Re raceCar singles dominate the surface
  error.
- **Critical infrastructure bug found and fixed:** `data/scoring.py` had an
  `Inf*0=NaN` reduction bug on the test path (a single sample in
  `test_geom_camber_cruise` has 761 Inf values in its pressure GT). Fix
  applied as advisor commit `2eb5c7f` (attribution to thorfinn).

### Decision

**Merged** as the round 3 measured baseline. No prior r3 baseline existed,
so this becomes the reference for the seven other in-flight r3 PRs. Per-PR
follow-ups for round 4: bs=12 (√3 LR scaling) if "larger batch" wins;
`--epochs ≤ 14` to actually decay cosine inside the wallclock cap;
activation checkpointing for bs=16+. These are recorded in
`research/CURRENT_RESEARCH_STATE.md`.
