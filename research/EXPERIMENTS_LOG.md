# SENPAI Research Results

Branch: `icml-appendix-willow-pai2d-r2`. Primary metric:
`val_avg/mae_surf_p` (lower is better). Wall-clock cap:
`SENPAI_TIMEOUT_MINUTES=30` per run.

Note on `test_avg/mae_surf_p`: every round-1 run reports `NaN` on
`test_geom_camber_cruise/mae_surf_p`, which propagates to the
test-average. Root cause confirmed by edward (#326) and fern (#328):
sample 20 of `test_geom_camber_cruise` ground truth has 761 NaN values
in the pressure channel; `accumulate_batch` masks the sample but
`0.0 * NaN = NaN` defeats the masking. **Bug-fix PR #367 assigned to
fern** with the 2-line `nan_to_num` patch — once it lands, every
round-1 run can recompute a finite `test_avg/mae_surf_p` from W&B.

## 2026-04-28 08:30 — PR #568 (fp16 retry, alphonse) ❌ CLOSED — fp16 catastrophically diverges + zero throughput delta over bf16

- Branch: `willowpai2d2-alphonse/fp16-scoped` (deleted on close).
- Iteration: 6th axis from alphonse. Tested fp16 with the
  loss-outside-autocast pattern + GradScaler that bf16 (#399)
  validated.

### Results — 3 fp16 seeds + 1 bf16 anchor

| amp_dtype | seed | val_avg/mae_surf_p | epoch_time_s | grad_scale_final | run_id |
|-:|-:|-:|-:|-:|-|
| bf16 anchor | 0 | **106.92** | 140.8 | n/a | 4joqw1fv |
| fp16 | 0 | NaN cascade | 141.5 | 0 (collapsed) | tzdrvpj8 |
| fp16 | 1 | NaN cascade | 141.0 | 0 (collapsed) | p82xzaja |
| fp16 | 2 | NaN cascade | 140.7 | 0 (collapsed) | jjuvn7vx |

bf16 anchor confirms the implementation infrastructure is correct
(val=106.92 lands at the favorable end of the bf16+Huber 3-seed
distribution from #399 — within noise of the merged 3-seed mean
112.13).

### Failure mode forensics

All 3 fp16 seeds: brief working phase (1-3 epochs of finite training
loss), then a single overflowed step that GradScaler can't recover
from, followed by terminal scale collapse to 0. Pre-catastrophe val
numbers are 50-145 % above bf16 anchor — fp16 was training
materially worse on the steps it was making, even before the
divergence.

Mechanism: PhysicsAttention's slice-token einsum + slice_norm
division produces intermediate values outside fp16's max representable
value (~65504). bf16 has fp32-equivalent dynamic range (~3.4e38) so
the same architecture is comfortable. **The loss-outside-autocast
scope (which made bf16 work) is the necessary precision-engineering
pattern for any non-fp32 autocast — but it is not sufficient when
the autocast block itself contains operations that overflow at the
precision in use.**

### Throughput: +0.2 % slower (the second nail)

fp16 epoch_time matches bf16 within 0.5 s (141.1 vs 140.8 mean). On
Blackwell tensor cores at our 0.67M-param Transolver, bf16 and fp16
deliver identical throughput. **bf16 captured all the AMP throughput
at our scale; fp16 has no remaining headroom to unlock.** Eliminates
the infrastructure-merge path even in a counterfactual stable-fp16
world.

### Conclusion

**Closed.** Two independent reasons (catastrophic stability + zero
throughput delta over bf16) → no metric-merge, no infrastructure-
merge. **bf16 stays the right precision for this branch.**

### Carry-forward contributions

1. **`--amp_dtype {fp32, bf16, fp16}` parametrization** stays in
   `train.py` after rebase. Future PRs can switch precision via CLI
   flag without code edits.
2. **bf16-anchor data point** (val=106.92, test=96.31 at seed 0)
   gives a fourth independent bf16+Huber data point. Combined with
   the merged 3-seed bf16 mean (112.13, σ=4.53), this confirms the
   bf16 baseline is stable and corroborates the σ ≈ 4.5 noise floor.
3. **"fp16 catastrophic instability + zero throughput delta over
   bf16" finding** is methodology data. At our scale on
   Hopper/Blackwell tensor cores, bf16 already extracts all available
   AMP throughput. Pushing precision lower doesn't help and hurts
   stability. **Rules out fp16 for any future round-3 stack** that
   doesn't restructure the attention pathway.

### Reassignment

Alphonse → PR #648 (round-2 axis: LayerScale). Detailed below.

## 2026-04-28 08:30 — PR #648 (NEW, alphonse round 2): LayerScale (learned residual gain)

- Hypothesis: per-channel learned diagonal-scale parameters on each
  TransolverBlock's residual contributions, init small (1e-4 to
  1e-2). Allows the optimizer to gradually "open up" each block's
  contribution rather than starting at full residual magnitude.
  Modern transformer recipe (CaiT, ViT-S+, DeiT). Predicted 1-3 %
  reduction over 115.61 anchor.
- Sweep: `layerscale_init ∈ {0.0 baseline, 1e-4, 1e-3, 1e-2}` with
  multi-seed at the winner.
- Implementation: ~25 LOC `LayerScale(nn.Module)` class + insert in
  `TransolverBlock.__init__/forward` + plumb through `Transolver.__init__`.
- Truly orthogonal to all 8 in-flight round-2 axes — different
  mechanism (residual gain, not loss/optimizer/norm/regularization/
  data/throughput).
- Decision rule: ≤102 single-seed (or ≤107 multi-seed mean) merges;
  at-baseline (107-117) closes (no throughput unlock here, pure
  metric decision); >125 closes.

## 2026-04-28 08:30 — PR #517 (DropPath, tanjiro) ❌ CLOSED — statistically null at depth=5

- Branch: `willowpai2d2-tanjiro/drop-path` (deleted on close).
- Iteration: 3rd axis from tanjiro. 4-value sweep + 3-seed at winner.

### Sweep + multi-seed results

| drop_path | seed | val_avg/mae_surf_p | test_avg/mae_surf_p | epoch_time_s |
|-:|-:|-:|-:|-:|
| 0.0 in-PR control | 0 | 122.93 | 112.10 | 172.4 |
| 0.05 | 0 | 127.45 | 116.97 | 178.9 |
| 0.1 | 0 | 126.72 | 115.64 | 178.7 |
| **0.2** | 0 | 120.68 | 110.92 | 178.7 |
| 0.2 | 1 | 126.01 | 112.41 | 178.5 |
| 0.2 | 2 | 121.14 | 112.45 | 178.2 |
| **0.2 mean (n=3)** | – | **122.61 ± 2.95** | 111.93 ± 0.87 | ~178.5 |

vs in-PR dp=0.0 control: **−0.26 % on val, −0.15 % on test**, both
well inside ±10 % noise. Statistically null.

### Throughput: 3.5 % slower, not faster

The PR's predicted 3-8 % faster training was wrong on its face:
standard timm DropPath (`x * mask / keep_prob`) computes the residual
THEN multiplies by 0 or 1/keep_prob — there's no actual compute
saving. Tanjiro correctly diagnosed this in the close write-up.

### Per-split shape (the interesting finding)

DropPath helps `val_single_in_dist` (-3 to -5 %), hurts
`val_geom_camber_rc` (+3 to +4 %). This is the **SEVENTH instance of
the per-distribution-shift pattern** in round 2, but with the sign on
`val_single_in_dist` reversed compared to the previous 6 (alphonse
#311 / tanjiro #335 / edward #429 / fern #478 / nezuko #332 /
alphonse #485). **DropPath is the FIRST round-2 axis to help
val_single_in_dist** (the structurally-hardest split).

The improvements on `val_single_in_dist` cancel the regression on
`val_geom_camber_rc` so the aggregate metric lands at noise.
Averaging hides a real per-distribution-tradeoff effect.

### Conclusion

**Closed.** Multi-seed mean is statistically null vs the in-PR
control; throughput regression rather than unlock; depth=5 is below
the literature threshold (≥12) where DropPath gains reliably appear.
Tanjiro's own causal analysis ("depth=5 is too shallow + 11 epochs
is too few") is exactly right.

### Carry-forward contributions

1. **`drop_path` Config flag + linear-by-depth scaling** stays in
   `train.py`. If a future PR adds depth (e.g., bf16 + depth=8 stack
   for round 3), DropPath at the larger depth is a clean drop-in.
2. **The "DropPath helps `val_single_in_dist`" per-split signal** is
   the FIRST round-2 axis to help raceCar single. Compounding with
   axes that help cruise (most others) suggests **per-domain
   regularization** as a round-3 stacking direction: high DropPath
   on raceCar, low on OOD splits.
3. **"Standard timm DropPath has no throughput unlock"** finding.
   Future PRs proposing DropPath should treat it as a small (~3 %)
   cost, not a benefit.

### Reassignment

Tanjiro → PR #649 (round-2 axis: smoothness regularization).
Detailed below.

## 2026-04-28 08:30 — PR #649 (NEW, tanjiro round 2): auxiliary smoothness loss (physics-informed regularization)

- Hypothesis: penalize prediction differences between random pairs
  of nearby mesh nodes (Gaussian-decay weighted by spatial distance).
  Encourages spatially-smooth predicted fields — physics-informed
  prior since underlying CFD solutions are mostly smooth except at
  shocks/boundaries. Predicted 2-5 % reduction over 115.61 anchor.
- Sweep: `smooth_lambda ∈ {0.0 baseline, 0.01, 0.1, 1.0}` with
  multi-seed at the winner.
- Implementation: ~50 LOC `auxiliary_smoothness_loss` helper +
  config flags + main-loop integration. Random-pair sampling
  (K=1024 pairs/sample) sidesteps O(N²) k-NN — Monte Carlo with
  Gaussian distance weighting.
- **Truly novel orthogonal axis** — fills the entire un-claimed
  "auxiliary loss / physics prior" bucket. No other in-flight PR
  touches auxiliary losses.
- Decision rule: ≤102 single-seed (or ≤107 multi-seed mean) merges;
  at-baseline closes; >125 closes.

## 2026-04-28 05:45 — PR #485 (iter 1): RMSNorm (drop-in for nn.LayerNorm) ❌ CLOSED

- Branch: `willowpai2d2-alphonse/rmsnorm` (deleted on close).
- Iteration: assigned as round-2 normalization axis after closing
  width PR #311.

### Implementation: PyTorch fused `nn.RMSNorm` (not hand-rolled)

Alphonse's first attempt used a hand-rolled `RMSNorm` (`x.float().pow(2).mean().rsqrt()`)
and was **slower than baseline LayerNorm** — 195 s/epoch vs 172 s
baseline because the hand-rolled chain launches ~5 separate kernels
per norm site while `nn.LayerNorm` is a single fused CUDA kernel.
Switched to `nn.RMSNorm` (fused PyTorch op) for the production runs.

### Results — 3-seed multi-seed

| seed | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | epoch_time_s |
|-:|-:|-:|-:|-:|
| 0 | 124.84 | 112.80 | 10 | 168.4 |
| 1 | 134.30 | 125.56 | 8  | 167.7 |
| 2 | 124.09 | 113.16 | 11 | 167.8 |
| **mean** | **127.74** | **117.17** | – | **168.0** |

vs current Huber baseline 115.61: mean val = +10.5 % (just outside
±10 % noise band on the wrong side). Test mean = -0.4 % vs
baseline 117.59 (essentially at-baseline on test).

Throughput delta = -2.3 % (168 vs 172 s/epoch) — **below the ≥3 %
bar** for infrastructure merge.

### Conclusion

**Closed.** Not in [110, 121] at-baseline band → no infrastructure
merge. Throughput unlock didn't materialize at our scale. Per-split
shows the 6th instance of the per-distribution-shift pattern:
cruise +1 %, raceCar splits +12-17 %, val_re_rand +10 %.

### Critical methodology contribution

**"PyTorch's fused LayerNorm is already close to optimal at our
scale."** Alphonse's diagnostic — that the 5-10 % speedup typically
attributed to RMSNorm-vs-LayerNorm benchmarks doesn't apply when
LayerNorm is fused (single kernel) — is a clean piece of
"what doesn't apply at our scale" data. Future round-3 stacks
shouldn't expect RMSNorm to give meaningful throughput at this
model scale.

### Sixth instance of per-distribution-shift pattern

(After alphonse #311 width-160, tanjiro #335 schedule, edward #429
p_weight, fern #478 curriculum, nezuko #332 partial.) RMSNorm
shifts cost into raceCar splits (+12-17 %) while leaving cruise
flat (+1 %). **Exactly the signature of perturbations on top of
Huber** per fern's #478 mechanism analysis. RMSNorm doesn't address
the underlying Huber-induced gradient imbalance.

### Carry-forward contributions

1. **"Fused LayerNorm is already optimal at our scale"** — guides
   round-3 normalization-axis decisions.
2. **Test parity without val improvement signal** — fp32-equivalent
   downstream metric quality, just without metric upside. Mild
   round-3 stacking candidate if fern #559 y_std-weighting shifts
   the gradient distribution.

### Reassignment

Alphonse → PR #568 (round-2 axis: fp16 with scoped autocast).
Detailed below.

## 2026-04-28 05:45 — PR #568 (NEW, alphonse round 2): fp16 with scoped autocast + GradScaler

- Reassigning alphonse after closing #485.
- Hypothesis: fp16 with the loss-outside-autocast pattern (validated
  by alphonse's bf16 PR #399) + GradScaler. fp16 typically gives
  ~1.5-2× over fp32-eager on Hopper/Blackwell vs bf16's measured
  1.23×; if it stacks beyond bf16, that's an additional 10-30 %
  throughput unlock.
- Decision rule (relative to bf16+Huber baseline mean 112.13):
  ≤102 single-seed (or ≤107 multi-seed mean) merges metric+infra;
  at-baseline-with-throughput-unlock (107-117 mean AND ≥5 % faster
  than bf16's 140.6 s) merges as infrastructure; at-baseline
  without throughput unlock closes (fp16 gives nothing over bf16);
  divergence/instability closes.
- Implementation: ~30 LOC builds directly on alphonse's #399 bf16
  scope. Parametrize `amp_dtype ∈ {fp32, bf16, fp16}` config flag,
  add GradScaler for fp16 only, log `amp/grad_scale` panel for
  diagnostics.
- Status: assigned, draft, status:wip.

## 2026-04-28 05:30 — PR #478 (iter 1): curriculum learning by per-sample pressure y-std ❌ CLOSED

- Branch: `willowpai2d2-fern/curriculum-y-std` (deleted on close).
- Iteration: assigned as round-2 data-axis lever after closing #452.
  Sweep `curriculum_warmup_epochs ∈ {3, 5, 8}` at fixed `min_quantile=0.5`.

### Results (single seed each)

| W | val_avg/mae_surf_p | val_single | val_camber_rc | val_camber_cruise | val_re_rand | run id |
|-:|-:|-:|-:|-:|-:|-|
| 3 | 116.90 | 148.10 | 131.90 | 83.18 | 104.44 | 6v60sktx |
| 5 | 117.66 | 141.97 | 140.96 | 82.21 | 105.52 | my4wdf6k |
| 8 | 129.61 | 185.72 | 146.26 | 91.39 | 109.80 | 7e4frsqm |

vs 115.61 baseline: best W=3 = +1.1 % (inside ±10 % noise but slightly above);
W=5 = +1.8 %; W=8 = +12.1 % (regression). Per the explicit decision rule
(>115 → close), clean close.

### Conclusion + LOAD-BEARING METHODOLOGY FINDING

**Closed.** But fern's mechanism analysis is the most insightful round-2
contribution to date — surfacing the dominant explanation for **why
round-2 has been brutal across the board**:

> **Huber's gradient clipping past |residual|=1 implicitly under-weights
> raceCar relative to cruise.** raceCar samples have large normalized
> residuals (high per-sample y_std → wide spread → far from `y_norm`
> mean) → Huber clips them. Cruise samples have small normalized
> residuals → Huber stays in quadratic regime. Cruise samples
> structurally receive higher gradient share than raceCar samples.

This explains every closed round-2 PR's per-distribution-shift signal:

- alphonse #311 width-160: cruise -6.1 %, raceCar +17-20 %
- tanjiro #335 lr=1e-3 schedule: cruise -8.1 %, single +12.8 %
- edward #429 p_weight=2: cruise -10 %, raceCar +8 %
- fern #478 curriculum W=3: cruise -15.8 %, raceCar +8-11 %
- nezuko #332 surf_weight=25: partial — all splits regress, cruise least

**Five out of five round-2 axes show the same per-distribution-shift sign.**
The pattern is property-of-the-baseline (Huber-induced gradient imbalance),
not perturbation-specific.

### The corollary axis

The fix: counter-balance Huber's cruise-favoring by **per-sample loss
weighting proportional to y_std** — bigger weight for high-y_std
(raceCar), smaller for cruise. This brings raceCar's gradient share back
up. Assigned as PR #559.

### Carry-forward contributions

1. **`compute_per_sample_y_std` helper** stays in train.py (precompute
   foundation for any data-or-loss-weighting axis).
2. **Per-domain balance artifact diagnosis** + **mechanism analysis of
   why Huber + perturbation = cruise dominates** are the dominant
   round-2 methodology findings.
3. **Curriculum trajectory verification logging** sets the methodology
   precedent for "verify the intervention is engaging before drawing
   metric conclusions."

### Reassignment

Fern → PR #559 (round-2 axis: per-sample loss weighting ∝ y_std).
Detailed below.

## 2026-04-28 05:30 — PR #559 (NEW, fern round 2): per-sample loss weighting ∝ y_std

- Reassigning fern after closing #478.
- Hypothesis: counter-balance Huber's implicit cruise-favoring gradient
  share by per-sample loss weight `w_i = (y_std_i / median_y_std)^α`.
  Higher α → stronger up-weighting of high-y_std (raceCar) samples,
  down-weighting of low-y_std (cruise). The mechanism directly
  addresses the **fifth-instance per-distribution-shift pattern**
  surfaced from fern's own #478 mechanism analysis.
- Predicted gain concentrated on `val_single_in_dist` and
  `val_geom_camber_rc` (the splits regressing on every round-2
  perturbation so far). 3–8 % reduction over 115.61 anchor.
- Sweep: `y_std_weight_alpha ∈ {0.0, 0.5, 1.0}` with multi-seed at
  the winner. Decision rule per new (tighter) noise floor: ≤102
  single-seed (or ≤107 multi-seed mean) merges; at-baseline closes.
- Implementation: ~30 LOC on top of fern's existing
  `compute_per_sample_y_std`. Per-batch y_std computation from `y[..., 2]`
  + per-sample weight applied to `sq_err` tensor.
- Status: assigned, draft, status:wip.

## 2026-04-28 05:00 — PR #399 (rebased): bf16 mixed precision ★ INFRASTRUCTURE-MERGED ★

- Branch: `willowpai2d2-askeladd/bf16-amp` (rebased onto post-Huber +
  post-#367 advisor; clean 27-line train.py-only diff combining bf16
  autocast + nan_to_num-wrapped Huber + SENPAI_SEED + `peak_GB` log).
- Run group: `willow-r2-askeladd-bf16-on-huber`

### 3-seed multi-seed (run on slice-128 + Huber + scoring fix)

| seed | best_ep | val_avg/mae_surf_p | epoch_time_s | total_epochs | run id |
|-:|-:|-:|-:|-:|-|
| 0 | 13 | **106.92** | 141.05 | 15 | rqcpftrd |
| 1 | 13 | 115.09 | 140.25 | 15 | zh4kubvw |
| 2 | 12 | 114.39 | 140.41 | 13 | o98mcrll |
| **mean** | – | **112.13** (σ=4.53) | 140.57 | 14.3 | – |

Per-split 3-seed means (vs 115.61 fp32 baseline):
- val_single_in_dist: 134.01 (−2.3 %)
- val_geom_camber_rc: 119.19 (+0.5 %)
- val_geom_camber_cruise: 92.43 (−6.4 %)
- val_re_rand: 102.90 (−4.6 %)

### Test (FINITE for the first time on this branch — post-#367)

3-seed test means: test_single_in_dist=118.16, test_geom_camber_rc=107.52,
**test_geom_camber_cruise=79.13** (was always NaN), test_re_rand=102.49.
**`test_avg/mae_surf_p` = 101.82** — first end-to-end finite paper-facing
metric on this branch.

### Conclusion — INFRASTRUCTURE MERGE

3-seed mean 112.13 falls in [110, 121] band → infrastructure merge per
the updated decision rule. Three reasons it's the right call:

1. **Throughput unlock confirmed and stable across baseline.** 1.23×
   per-step speedup, +30 % epoch headroom (14.3 vs 11), no
   instability. Loss-independent — same magnitude as on MSE.
2. **First end-to-end finite test_avg/mae_surf_p** (101.82). PR #367
   + bf16 rebase together unblock paper-facing metric for every
   future PR.
3. **σ tightening observation is load-bearing.** σ=4.53 on
   bf16+Huber vs σ=8.13 on bf16+MSE shows Huber substantially
   reduces seed-dependent variance. Concrete corollary: multi-seed
   runs on Huber discriminate effects in fewer seeds than the MSE-
   era ±10 % heuristic suggested. Anchor noise floor for round 2
   onwards is **σ ≈ 4.5 / ~4 %**.

### Methodology consequence

The seed=0 single-shot at 106.92 is right at the strict ≤105
single-seed merge bar. Combined with the favorable per-split
trend (no regression), bf16+Huber is plausibly outside the new
(tighter) noise floor on the favorable side, even if the 3-seed
mean is conservatively at-baseline. Future round-2 multi-seed
results on this baseline should compare against 112.13 with σ=4.5,
not 115.61 with the historical ±10 %.

### Carry-forward contributions

1. **bf16 autocast (`torch.amp.autocast("cuda", dtype=torch.bfloat16)`
   around model forward, `pred.float()` cast after, loss in fp32)**
   is now the default in train.py. Future PRs run on bf16
   automatically; no flag needed.
2. **`peak_GB` per-epoch W&B logging** — useful for the whole
   branch, surfaces "did the autocast actually drop memory?" type
   questions.
3. **`SENPAI_SEED` env-var seeding at module-top** + GPU bf16 support
   check at startup.
4. **σ-tightening on Huber** — tighter noise floor for round-2
   multi-seed budgeting.

### Reassignment

Askeladd → PR #553 (round-2 axis: `torch.compile` on top of bf16).
Detailed below.

## 2026-04-28 05:00 — PR #553 (NEW, askeladd round 2): `torch.compile` on top of bf16

- Reassigning askeladd after merging #399.
- Hypothesis: `torch.compile` traces the model's forward pass into
  an optimized graph (kernel fusion + Python-overhead elimination).
  Predicted 1.10–1.30× **additional** speedup over bf16 → total
  ~1.4–1.6× over fp32-eager. Stacks with bf16 multiplicatively
  because the two target different bottlenecks (memory bandwidth
  vs compute overhead).
- Implementation risk: variable mesh sizes (74K–242K nodes per
  sample) trigger graph recompilation by default. Use
  `dynamic=True` to handle dynamic shapes.
- Sweep: `compile_mode ∈ {"none", "default", "reduce-overhead"}`.
  Multi-seed at the winner.
- Decision rule: ≤102 single-seed (or ≤107 multi-seed mean) merges
  as metric+infra win; at-baseline-with-throughput-unlock
  (107–117 multi-seed mean AND ≥5 % steady-state speedup) merges
  as infrastructure (same precedent as bf16).
- Status: assigned, draft, status:wip.

## 2026-04-28 04:45 — PR #429 (iter 1): per-channel pressure weighting (`p_weight` sweep) ❌ CLOSED

- Branch: `willowpai2d2-edward/p-weight-sweep` (deleted on close).
- Iteration: assigned as round-2 axis after closing FFN PR #326.
  Sweep `p_weight ∈ {2, 3, 5}` with multi-seed at the winner.
- Branch was rebased onto pre-#367 advisor (data/scoring.py reverts
  the merged `nan_to_num` patch). Same silent-revert blocker as
  other pre-#367 PRs.

### Sweep results

| variant | seed | val_avg/mae_surf_p | val_avg/mae_surf_Ux | val_avg/mae_surf_Uy | run id |
|-|-:|-:|-:|-:|-|
| 1.0 (baseline ref) | – | 115.61 | 1.81 | 0.75 | uip4q05z |
| 2.0 | 0 | 112.07 ⭐ | 2.05 | 0.87 | i0dz5op0 |
| 2.0 | 1 | 124.48 | 2.40 | 0.85 | 64cb0h0e |
| 2.0 | 2 | 119.40 | 2.33 | 0.85 | kwl0ovqw |
| **2.0 mean (σ=6.24)** | – | **118.65** | 2.26 | 0.857 | – |
| 3.0 | 0 | 130.00 | 2.33 | 0.93 | cy9dhhaf |
| 5.0 | 0 | 126.19 | 2.50 | 1.02 | 20cxfbvn |

### Conclusion

**Closed.** Multi-seed mean at p_weight=2 (the sweep winner at
single-seed) is **118.65** = +2.6 % vs 115.61 baseline, all three
seeds within ±10 % noise band but with the seed=0 result lucky-pull.
Per the explicit decision rule (multi-seed mean > 115 → close),
clean close.

### Critical methodology contributions

**1. Two converging instances of "Huber absorbs small optimization-axis
levers."** Edward's mechanism analysis: *"Huber's gradient clipping
already up-weights large pressure residuals; explicit p_weight>1
perturbs the linear-regime gradients on small residuals — exactly
what Huber was trying to deprioritise."* This is the **same
mechanism tanjiro identified** for why the schedule axis (#335)
fails on Huber. Two independent PRs converging on the same
finding makes it load-bearing methodology:

> **Huber's effective implicit gradient-shaping covers many of the
> small-effect optimization-axis levers that worked on MSE.
> Round-2 axes that target the same failure mode (high-Re tail,
> channel-imbalance, schedule-tuning) tend to regress to baseline
> because Huber is already doing the work. Axes with truly
> orthogonal mechanisms (target distribution, throughput, parameter-
> noise, normalization, regularization, data exposure, update-rule
> shape, per-layer LR allocation) are the right round-2 candidates.**

**2. Fourth instance of the per-distribution-shift pattern.** At
p_weight=2, val_geom_camber_cruise improves (−10 %) but
val_single_in_dist (+7.7 %) and val_geom_camber_rc (+8.7 %) regress.
**Same sign as alphonse #311 (width-160), tanjiro #335 (lr=1e-3
schedule), and edward p_weight=2.** Four independent PRs showing
the same per-distribution-shift signal makes **per-distribution
architecture / loss specialization an overwhelming round-3 priority**.

**3. Textbook noise-floor demonstration.** Single-seed p_weight=2
seed=0 looked like a winner (112.07 vs 115.61 = -3 %); 3-seed mean
revealed it as a lucky pull (118.65 vs 115.61 = +2.6 %). σ=6.24
≈ 5.3 % of baseline. Single-seed deltas under ~10 % require
multi-seed to disambiguate — fully validated.

### Carry-forward contributions

1. **`p_weight` CLI flag + `_channel_weighted_huber` helper** stay
   in `train.py` after rebase. p_weight=1.0 default = baseline
   behavior. Future PRs can layer per-channel weighting on top of
   different baselines (e.g., post-asinh from frieren #415) without
   re-implementing.
2. **`SENPAI_SEED` seeding at module-top** (commit 2e15747).
3. **The "Huber absorbs small axes" mechanism analysis** is now the
   dominant explanation for round-2 difficulty.

### Reassignment

Edward → PR #547 (round-2 axis: layer-wise learning rate decay /
LLRD). Detailed below.

## 2026-04-28 04:45 — PR #547 (NEW, edward round 2): layer-wise learning rate decay (LLRD)

- Reassigning edward after closing #429.
- Hypothesis: split parameters into per-layer groups (preprocess +
  5 TransolverBlocks + last-layer head), assign each group a
  learning rate that scales as `lr · decay^(depth_from_output)`.
  Deeper/output-side layers get faster LR; foundational/input-side
  layers get slower LR. Standard transformer fine-tuning recipe.
- Mechanism is **structurally distinct** from all in-flight optimizer
  axes: Lion changes update *rule*, EMA does post-step *averaging*,
  LLRD does per-layer *splitting* of step sizes (same Adam, same
  Huber). Doesn't overlap with Huber's clipping mechanism (per
  edward's own analysis on the closed #429).
- Sweep: `llrd_decay ∈ {0.7, 0.85, 1.0 baseline-anchor}` with
  multi-seed at the winner.
- Implementation: ~30 LOC `_build_param_groups` helper + config
  flag + optimizer construction branch (short-circuits to existing
  AdamW when `llrd_decay=1.0` to preserve baseline behavior exactly).
- Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges;
  at-baseline closes (LLRD doesn't transfer to depth=5); >130 closes
  (foundation layers are too important to slow down at our shallow
  depth). Status: assigned, draft, status:wip.

## 2026-04-28 04:15 — PR #335 (iter 3): warmup+cosine_t_max=15 on Huber baseline (multi-seed) ❌ CLOSED

- Branch: `willowpai2d2-tanjiro/warmup-cosine-1e3` (deleted on close).
- Iteration: send-back asked for rebase + multi-seed at variant (b)
  `lr=1e-3, cosine_t_max=15` on top of merged Huber + slice-128 baseline.
- Branch was correctly rebased onto current advisor head (20533b1) —
  picks up Huber (#330), slice-128 (#328), and scoring fix (#367).
  Diff is exactly the three intended changes (cosine_t_max field,
  SENPAI_SEED seeding, LinearLR + SequentialLR block).

### Results — 3-seed multi-seed at lr=1e-3, cosine_t_max=15

| seed | val_avg/mae_surf_p | val_re_rand | best_epoch | run id |
|-:|-:|-:|-:|-|
| 0 | 124.45 | 115.83 | 10 | r9uyjniz |
| 1 | 119.61 ⭐ | 108.67 | 11 | pc2s5lum |
| 2 | 123.08 | 114.03 | 11 | tlzzhtnl |
| **mean** | **122.38** (range 4.84) | **112.84** | – | – |

vs current Huber baseline 115.61: **+5.85 % on mean**, all 3 seeds
above. val_re_rand also +4.6 % above baseline (107.89), so no
stacking on the high-Re-tail signal Huber improved most.

### Conclusion

**Closed.** Per the explicit decision rule, multi-seed mean > 115
falls in "Inside noise / above → close." Schedule axis does not
stack on Huber + slice-128.

### Critical methodology contribution: schedule × loss interaction

| Setting | Schedule axis result | vs same-arch baseline |
|-|-:|-:|
| slice-64 + MSE (round-2 iter-2) | 113.96 | -26.2 % vs MSE 154.57 |
| slice-128 + Huber (round-3 iter-3, this) | 122.38 mean | +5.85 % vs Huber 115.61 |

Same recipe (`lr=1e-3, cosine_t_max=15, warmup=5`), different baseline,
opposite signs. Two plausible mechanisms (per tanjiro's own analysis):

1. **Huber's effective implicit gradient-shaping covers what cosine
   decay was doing on MSE.** Huber β=1 caps gradient magnitude past
   |residual|=1. On MSE, schedule had to do this via late-epoch LR
   shrinkage; on Huber, the loss formulation does it implicitly.
2. **Peak 1e-3 may overshoot at slice-128** (wider model wants
   gentler LR). Per-split signal supports this: cruise improves
   (-8.1 %) but in-dist regresses (+12.8 %) — same per-distribution-
   shift pattern alphonse saw at width-160.

### Third instance of "interior optima are architecture-dependent"

After nezuko #332 (surf_weight=25 didn't transfer slice-64→slice-128)
and alphonse #311 (width-160 hurts in-dist while helping cruise on
slice-128+Huber), this is the **third independent confirmation** of
the cross-cutting methodology finding: **a hyperparameter optimum
found on one architecture / loss combination does not transfer to
another without re-tuning**. Now load-bearing for round-3 stacking
decisions.

### Carry-forward contributions

1. **Schedule infrastructure stays in `train.py`**: `cosine_t_max`
   CLI flag + `LinearLR + SequentialLR` warmup-cosine block + the
   `SENPAI_SEED` env-var seeding (now adopted by 7+ PRs as the de
   facto multi-seed convention).
2. **Per-split-shift signal at peak LR=1e-3 on slice-128** corroborates
   alphonse's #311 closed observation. Two independent PRs showing
   the same per-distribution-shift pattern strengthens
   per-distribution architecture specialization as a round-3
   candidate.
3. **Seed-1's `test_geom_camber_cruise/mae_surf_p = NaN` despite
   #367**. Edge case: non-finite *prediction* (Inf) propagating
   through `nan_to_num(posinf=0)` followed by the masked sum
   reduction still produces NaN somehow. Worth investigating in a
   follow-up bug-fix PR if it surfaces on other students' runs.
   Logged for awareness.

### Reassignment

Tanjiro → PR #517 (round-2 axis: stochastic depth / DropPath).
Detailed below.

## 2026-04-28 04:15 — PR #517 (NEW, tanjiro round 2): stochastic depth (DropPath)

- Reassigning tanjiro after closing #335.
- Hypothesis: apply DropPath to TransolverBlock residual contributions.
  At training time, with probability `drop_prob`, the block's
  attention or MLP residual is replaced with the identity (block
  contribution zeroed); surviving residuals scaled by 1/(1-drop_prob).
  Inference and validation use full depth. Implicit ensemble of
  shallower networks during training. Modern transformer
  regularization default (DeiT, Swin).
- Sweep: `drop_path ∈ {0.0 baseline, 0.05, 0.1, 0.2}` with
  linear-by-depth ramp (standard timm/DeiT recipe). Multi-seed at
  the winner.
- Implementation: ~15 LOC `DropPath` class + insert in
  `TransolverBlock.forward` + linear-by-depth scaling in
  `Transolver.__init__`. No new packages.
- Predicted gain: 2–5 % over 115.61 baseline at the optimal
  drop_prob. Slight throughput unlock during training (skipped
  blocks → less compute per step).
- Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges
  as metric+infra win; at-baseline-with-throughput-unlock merges
  as infrastructure (same precedent as bf16 + RMSNorm).
- Status: assigned, draft, status:wip.

## 2026-04-28 03:30 — PR #311 (iter 3): width-160 on slice-128 + Huber (multi-seed) ❌ CLOSED

- Branch: `willowpai2d2-alphonse/width-192` (deleted on close).
- Iteration: send-back asked for rebase + multi-seed at width-160 on
  top of merged Huber + slice-128 baseline.
- Branch was rebased onto 5975c81 (post-#330 but pre-#367), so
  `data/scoring.py` and `evaluate_split` reverted the merged
  `nan_to_num`. Hence student saw `test_geom_camber_cruise/mae_surf_p
  = NaN` even though the bug fix is merged.

### Results — 3-seed multi-seed at width-160

| seed | val_avg/mae_surf_p | best_epoch | run id |
|-:|-:|-:|-|
| 0 | 123.38 | 9 | aw3avnj5 |
| 1 | 127.84 | 9 | hsxma71y |
| 2 | 129.06 | 9 | vvczi4b8 |
| **mean** | **126.76** (σ=2.99) | – | – |

Per-split deltas vs baseline (115.61):
- val_single_in_dist: 137.21 → 160.91 (**+17.3 %**)
- val_geom_camber_rc: 118.60 → 142.22 (**+19.9 %**)
- val_geom_camber_cruise: 98.74 → 92.76 (**−6.1 %**)
- val_re_rand: 107.89 → 111.16 (+3.0 %)

### Conclusion

**Closed.** Multi-seed mean **+9.6 %** above the 115.61 baseline,
all 3 seeds above. By CLAUDE.md > 5 % regression threshold, this is
a clean close. By the PR's own decision rule (mean ≤ 126 → merge),
it's just past the bar. Student explicitly recommended closing.

### Per-split signal: architecture-vs-distribution-shift

Width-160 *helps* `val_geom_camber_cruise` (−6.1 %) and *hurts*
`val_single_in_dist` (+17.3 %) and `val_geom_camber_rc` (+19.9 %).
Net loss on the equal-weight average. The cruise improvement is not
within noise — across all 3 seeds, cruise consistently improves
~5–7 %. This is the cleanest piece of round-2 evidence that
**per-distribution architecture specialization is a genuine round-3
axis**: capacity scaling has split-specific returns, with cruise
(largest meshes, ~210K nodes) benefiting and single-in-dist /
camber-rc (smaller meshes) regressing. Round-3 candidate: per-mesh-
size-conditional MLP scaling, or specialised heads.

### Carry-forward contributions

1. **fp16 → bf16 failure-mode diagnosis** (from iter 2) was
   adopted into askeladd's #399 round-2 assignment. Loss-outside-
   autocast pattern is the standard "scoped AMP" recipe now.
2. **Per-split-shift observation** (cruise-helps / in-dist-hurts at
   width-160) is the cleanest piece of evidence for per-distribution
   architecture specialization as a round-3 axis.
3. **Width-axis ceiling at slice-128 + Huber** under 30-min cap is
   established. Future width-axis work should wait for AMP/bf16
   throughput unlock.

### Reassignment

Alphonse → PR #485 (round-2 axis: RMSNorm). Detailed below.

## 2026-04-28 03:30 — PR #485 (NEW, alphonse round 2): RMSNorm

- Reassigning alphonse after closing #311.
- Hypothesis: replace `nn.LayerNorm` with `RMSNorm` throughout the
  Transolver model (11 LayerNorm sites across 5 TransolverBlocks).
  RMSNorm drops mean-centering — same parameter count, ~5–10 %
  faster per step. Modern transformer default (LLaMA, Gemma,
  Falcon, Mistral). Throughput unlock translates to more epochs in
  the 30-min cap; predicted 2–5 % reduction over 115.61.
- The only architecture-axis lever that's *cheaper* than the
  merged baseline (depth-8, width-192/160, slice-192, mlp_ratio=4
  all paid more compute per step and lost the trade).
- Implementation: ~15 LOC `RMSNorm` class + find/replace at 11
  LayerNorm sites + update `_init_weights` type check. No new
  packages, no model_config flag.
- Sweep: single primary candidate first (single seed), then
  multi-seed if budget allows. RMSNorm `eps` sweep deferred.
- Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges
  as metric+infra win; at-baseline-with-throughput-unlock (110–121
  multi-seed mean AND epoch_time_s ≥ 3 % faster) merges as
  infrastructure (same precedent as bf16 #399).
- Status: assigned, draft, status:wip.

## 2026-04-28 03:15 — PR #452 (iter 1): push slice_num to 192 ❌ CLOSED

- Branch: `willowpai2d2-fern/slice-push` (deleted on close).
- Iteration: assigned as round-2 axis after merging #367. Single-seed
  on slice-192 first (per the staged decision rule), with slice-256
  contingent on a clean slice-192 win.

### Result (single seed, run `l7dulvjz`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | Δ vs 115.61 |
|-|-:|-:|-:|-:|
| val_single_in_dist | 161.97 | 2.00 | 0.93 | +18.0 % |
| val_geom_camber_rc | **151.75** | 2.85 | 1.20 | **+28.0 %** |
| val_geom_camber_cruise | 100.59 | 1.77 | 0.60 | +1.9 % |
| val_re_rand | 118.91 | 2.32 | 0.84 | +10.2 % |
| **val_avg** | **133.30** | 2.24 | 0.89 | **+15.3 %** |
| **test_avg** | **123.10 (FINITE)** | 2.14 | 0.85 | – |

Best epoch 9/50 (timeout at 32.1 min, 213 s/epoch — actual slowdown
**24 %** vs predicted 10–15 %).

### Conclusion

**Closed.** 133.30 is well outside the ±10 % noise band of 115.61.
Per the PR's decision rule, this falls cleanly in the "slice
bottleneck is fully resolved at 128. Close" bucket. Three pieces of
evidence:

1. **Single-seed result is decisively outside noise**: +15.3 %.
2. **Per-step cost (24 %) was higher than predicted (10–15 %)** —
   `O(N · slice_num)` einsums dominate, but LayerNorm / softmax /
   output-projection overheads also scale, and the advance prediction
   underestimated the constant multiplier.
3. **The per-split shape contradicts the slice-bottleneck mechanism.**
   Round-1 #328's win was characterized by the largest improvement on
   `val_geom_camber_rc`. slice-192 produces the OPPOSITE pattern: the
   **largest regression** is on `val_geom_camber_rc` (+28 %), the
   **smallest** on `val_geom_camber_cruise` (+1.9 %, despite cruise
   meshes being the largest at ~210K nodes). Mechanism inversion.

Combined with #328 (slice-64→128 win) and this run, both endpoints
are bracketed: **slice-128 is the architectural ceiling on this axis
under the 30-min wall-clock contract.** Future slice-axis work should
wait until askeladd #399's bf16 unlock makes higher slice_num
budget-feasible.

### Methodology contributions

- **First end-to-end-finite `test_avg/mae_surf_p` on a non-trivial run
  (123.10).** PR #367's bug fix is verified working in the wild —
  every PR going forward gets this metric automatically.
- **Per-step-cost-vs-prediction calibration data** (24 % actual vs
  10–15 % predicted for slice 128→192). Logged as budget-planning
  reference for any future slice-axis follow-up.
- **Mechanism-inversion-as-disconfirmation** observation: when a
  hypothesis predicts a per-split signal in one direction and the
  actual signal lands in the *opposite* split, that's the strongest
  form of evidence that the hypothesis's mechanism doesn't apply.

### Reassignment

Fern → PR #478 (round-2 axis: curriculum learning by per-sample
pressure y-std). Detailed below.

## 2026-04-28 03:15 — PR #478 (NEW, fern round 2): curriculum by y-std

- Reassigning fern after closing #452.
- Hypothesis: dataset's per-sample y_std spans 10× even within a
  single domain. Currently presented uniformly via
  `WeightedRandomSampler`. Curriculum starts on the bottom-50 % of
  y_std and ramps to the full set over `curriculum_warmup_epochs`,
  preserving domain balancing. Targets the same high-Re-tail story
  Huber addressed via gradient clipping (#330) — but from the data-
  exposure side. Should compound. Predicted 3–8 % reduction over
  115.61 baseline.
- Sweep: `curriculum_warmup_epochs ∈ {3, 5, 8}` at fixed
  `min_quantile=0.5`. Multi-seed at the winner.
- Implementation: ~50 LOC in train.py — precompute per-sample
  pressure y_std once, define `curriculum_weights` helper that
  multiplies domain weights by a per-epoch quantile mask, build the
  `WeightedRandomSampler` per-epoch instead of once. Loaders stay
  read-only; logic lives in train.py.
- Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges;
  borderline → multi-seed; >115 closes the data axis.
- Status: assigned, draft, status:wip.

## 2026-04-28 03:00 — PR #399 (iter 1): bf16 mixed precision (round 2 axis)

- Branch: `willowpai2d2-askeladd/bf16-amp` (pre-#330 + pre-#367 — same
  rebase pattern as everyone created during round 1).
- Sent back for rebase + on-baseline confirmation.

### Results — 3-seed sweep (run on slice-128 + MSE state, not Huber)

| seed | val_avg/mae_surf_p | best_epoch | epoch_time_s | peak_GB |
|-:|-:|-:|-:|-:|
| 0 | 124.94 | 12 | 140.9 | 48.1 |
| 1 | 134.30 | 13 | 140.7 | 48.1 |
| 2 | 141.13 | 11 | 140.4 | 48.1 |
| **mean** | **133.46** (σ=8.13) | – | 140.7 | 48.1 |

vs the slice-128 + MSE baseline (133.55, the branch's anchor):
- **Metric**: 0.07 % better — at-baseline within noise
- **Throughput**: 1.23× per-step speedup (140.7s vs ~173s baseline)
- **Epochs in budget**: 13 vs 11 (+18 %)
- **Memory**: did NOT drop (48.1 vs ~42 GB) — student correctly
  diagnosed: at small slice_num and small param count, activation
  memory doesn't dominate; bf16 doesn't shrink the data buffers.

### Conclusion

**Send back for rebase + on-baseline run.** bf16 itself is safe: no
NaN, no instability, loss-shape matches fp32. The hypothesis
(throughput unlock for the 30-min cap) is confirmed at +18 % epochs
in budget. The metric is at-baseline within noise on the
slice-128+MSE state, but we need on-baseline (Huber) confirmation
before merging because the branch reverts both Huber and the
scoring fix.

Updated decision rule for the rebased re-run:
- ≤ 105 single-seed (or ≤ 110 multi-seed mean): merge as round-2
  metric+infrastructure win.
- 110–121 multi-seed mean (within ±10 % of 115.61): **merge as
  infrastructure** — the throughput unlock + finite test_avg make
  bf16 a strict positive even at-baseline metric.
- > 130 multi-seed mean: close (throughput unlock at metric cost).

Bonus: askeladd added per-epoch `peak_GB` to the W&B `wandb.log(...)`
— useful single-line addition for every future training run.

## 2026-04-28 03:00 — PR #332 (iter 2): surf_weight=25 multi-seed on slice-128 ❌ CLOSED

- Branch: `willowpai2d2-nezuko/surf-weight-25` (deleted on close).
- Iteration: send-back asked for rebase + 3-seed multi-seed at
  surf_weight=25 on top of slice-128 (the merged #328 baseline).

### Results — 3-seed multi-seed

| seed | val_avg/mae_surf_p | val_avg/mae_vol_p | best_epoch | run id |
|-:|-:|-:|-:|-|
| 0 | 154.63 | 168.67 | 8 | `8tmdnfge` |
| 1 | 152.13 | 172.53 | 10 | `omi56qls` |
| 2 | 149.07 | 168.30 | 11 | `pf76ohnj` |
| **mean** | **151.94** (σ=2.78) | 169.83 | – | – |

vs the slice-128 + MSE baseline (133.55) the student compared
against: **+18.4 absolute / +13.8 % over baseline**, ~11 SE outside
the noise band.

### Conclusion

**Closed.** The surf_weight=25 axis does not stack with
slice_num=128. Three pieces of evidence:

1. Mean is ~11 SE above baseline (σ=2.78 → SE≈1.6, gap=18.4) —
   this is decisively NOT a noise tie.
2. **vol_p went up too** (169.83 vs slice-64+sw=25's 142.51) —
   strictly worse operating point on both axes, not a Pareto-front
   movement.
3. **All three val curves oscillate after epoch ~8** (e.g. seed 0:
   154.6 → 168.3 → 160.9 → 180.5) — optimizer instability that
   wasn't present in the slice-64 sweep.

Student's causal story is exactly right: at slice-64, surface-pressure
error was bottlenecked by representational capacity (only 64 latent
tokens for 242K-node meshes), so up-weighting surf_loss helped recruit
gradient toward those representations. At slice-128, the surface
representations are richer (128 tokens, finer slicing), and the
bottleneck has shifted — up-weighting surf_loss now starves vol_loss
without buying anything on the surface side.

### Cross-cutting methodology finding

**"Interior optima are architecture-dependent."** The surf_weight=25
winner at slice-64 was real (clear curve shape + val_vol_p secondary
signal), but it didn't transfer to slice-128. Worth flagging for any
future round-3 stacking decisions: a hyperparameter optimum found on
one architecture should not be assumed to generalize to another
without re-tuning. Withdrawing my earlier recommendation that
"surf_weight=25 stays as a default."

### Reassignment

Nezuko → PR #472 (round-2 axis: Lion optimizer). Detailed below.

## 2026-04-28 03:00 — PR #472 (NEW, nezuko round 2): Lion optimizer

- Reassigning nezuko after closing #332.
- Hypothesis: Lion (Chen et al. 2023) replaces AdamW with a
  sign-based update + single momentum buffer. Reported to outperform
  AdamW on transformer benchmarks at smaller LR (typically 3–10×
  smaller). Predicted 3–8 % reduction over the merged 115.61
  baseline.
- Sweep: `lr ∈ {3e-5, 1e-4, 3e-4}` on shared
  `--wandb_group "willow-r2-nezuko-lion-lr"` — same 3-value
  curve-shape methodology that worked on the surf_weight axis.
- Implementation: ~30-line `Lion(torch.optim.Optimizer)` subclass
  inlined in train.py (no new packages); `optimizer: str` config
  flag selects between `"adamw"` and `"lion"`.
- Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges;
  borderline → multi-seed; >115 → close (axis exhausted).
- Status: assigned, draft, status:wip.

## 2026-04-28 02:45 — PR #337 (iter 2): BS+LR scaling on slice-128 ❌ CLOSED

- Branch: `willowpai2d2-thorfinn/batch-8-lr-7e4` (deleted on close).
- Iteration: send-back asked for rebase + BS=16/lr=1e-3 (sqrt-rule)
  primary, multi-seed where budget allows.
- Branch was rebased onto slice-128 (good) but pre-#330 + pre-#367
  (so MSE in train+eval, no scoring fix).

### Results

| BS | lr | seeds | val_avg/mae_surf_p | best_epoch | peak_GB |
|-:|-:|-:|-:|-:|-:|
| 16 | 1e-3 | – | OOM at batch 0 | – | – |
| 12 | 9e-4 | – | OOM at batch 0 | – | – |
| 10 | 8e-4 | – | OOM at startup | – | – |
| 8  | 7e-4 | – | OOM at batch 1 (even with `expandable_segments`) | – | – |
| **6** | **6e-4** | 0, 1, 2 | **156.15 / 174.69 / 157.05 → mean 162.63** | 8/10/11 | 81.7 |

Cross-seed range 156–175 (~12 % spread) consistent with the ±10 %
noise floor; mean is well outside the noise band on the wrong side.

### Conclusion

**Closed.** Hardware analysis is the most valuable single signal:
slice-128's activation memory consumes the 12 GB BS=8 headroom that
existed at slice-64 — only BS=4–6 fits at slice-128. The student's
steps-vs-parallelism causal chain is correct: BS=6 trades 27 % of
optimizer steps for parallelism that doesn't compensate at this
model size on this hardware. Combined with the lever being
hardware-blocked at BS≥8, the BS+LR scaling axis is exhausted on
slice-128.

vs the merged Huber baseline (115.61), the 3-seed mean (162.63) is
**41 % worse** — far past the close threshold even ignoring the
pre-#330 / pre-#367 reverts the branch was carrying.

### Bonus methodology contributions carried forward

- **`SENPAI_SEED` env-var seed-handling pattern** (committed to
  thorfinn's branch, replicated by alphonse, nezuko, fern, edward,
  thorfinn). Effectively the de facto multi-seed convention on
  this branch.
- **Third independent measurement of ±10 % noise floor** (after
  thorfinn round-1 BS=8 replicate + edward #326 control re-run +
  this 3-seed BS=6 cross-seed range). Methodology well-anchored.
- **Hardware ceiling at slice-128**: BS ≤ 6 documented with
  specific OOM ladder + `expandable_segments` test. Saves future
  iterations from re-running the same experiment.

### Reassignment

Thorfinn → PR #457 (round-2 axis: EMA weight averaging via
`torch.optim.swa_utils.AveragedModel` with custom EMA `avg_fn`).
Detailed below.

## 2026-04-28 02:45 — PR #457 (NEW, thorfinn round 2): EMA weight averaging

- Reassigning thorfinn after closing #337.
- Hypothesis: every PR on this branch trains 11–14 of 50 epochs
  under the 30-min cap with val curves still descending steeply.
  EMA averages model parameters across recent training steps,
  reducing late-epoch variance — exactly where short-budget
  training benefits most. Evaluated dual (live + EMA) per epoch;
  best-checkpoint selection picks whichever wins per-epoch.
  Predicted 2–6 % reduction over 115.61 baseline.
- Sweep: `ema_decay ∈ {0.9, 0.99, 0.999}` on shared
  `--wandb_group "willow-r2-thorfinn-ema"`.
- Implementation: `torch.optim.swa_utils.AveragedModel` with custom
  `avg_fn` for EMA. ~30 lines of train.py changes, no new packages.
- Status: assigned, draft, status:wip.

## 2026-04-28 02:30 — PR #367 (rebased): Bug fix — guard against non-finite values ★ MERGED ★

- Branch: `willowpai2d2-fern/scoring-nan-fix` (rebased onto post-Huber
  advisor; clean two-line `nan_to_num` insertions on
  `data/scoring.py::accumulate_batch` (wrapping `err`) and
  `train.py::evaluate_split` (wrapping the merged Huber
  `F.smooth_l1_loss(...)`). Training loop left untouched per the
  send-back instruction.
- Verification run: `fitecuaq` on slice-128 + Huber baseline.

### Verification (run `fitecuaq`, post-rebase)

| metric | pre-fix | post-fix |
|-|-:|-:|
| `test_geom_camber_cruise/mae_surf_p` | NaN ⚠ | **96.92** |
| `test_geom_camber_cruise/loss` | inf ⚠ | **0.978** |
| `test_avg/mae_surf_p` | NaN ⚠ | **117.59** |

Per-split test on the verification run (slice-128 + Huber + bug fix):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| test_single_in_dist | 136.54 | 2.22 | 0.77 |
| test_geom_camber_rc | 118.04 | 2.30 | 0.94 |
| test_geom_camber_cruise | **96.92 ★** | 1.83 | 0.53 |
| test_re_rand | 118.84 | 2.02 | 0.75 |
| **test_avg** | **117.59 ★** | 2.09 | 0.75 |

The verification run's val_avg = 126.64 is 9.5 % worse than the
merged Huber baseline (115.61) — inside the ±10 % single-seed
noise floor (corroborated by thorfinn #337 + edward #326 control
re-runs at identical config). Same training command, same monotone
descent through epoch 11/50, just different RNG state.

### Conclusion

**Merged.** Paper-facing `test_avg/mae_surf_p` is now finite for
the first time on this branch. Every PR currently in flight will
report finite test metrics on their next runs automatically.

### Three independent confirmations of the same bug

This is the third independent diagnosis of the masking-NaN-defeats-the-mask
root cause across this round (edward #326 bug-report, askeladd #325
bug-report, fern #367 patch + verification). All three converged on
equivalent fixes (`nan_to_num` on `err` before the masked sum, or
equivalently `torch.where` to zero-fill non-finite positions before
the multiply). Strong cross-validation that the patch is correct.

### Reassignment

Fern → PR #452 (round-2 axis: push slice_num further to 192/256).
Continuation of fern's merged round-1 winner. Detailed below.

## 2026-04-28 02:30 — PR #452 (NEW, fern round 2): push slice_num further (192 / 256)

- Reassigning fern after merging #367.
- Hypothesis: with N up to 242K mesh nodes and the slice bottleneck
  shown to bind at 64 (round 1, fern's #328 merged win), pushing
  beyond 128 may continue to help. Cost geometry favorable: O(N ·
  slice_num) einsums dominate; slice² attention stays small.
  Predicted 2–5 % reduction over the merged 115.61 baseline.
- Sweep: slice_num=192 primary; slice_num=256 if budget allows.
  Decision rule: ≤105 single-seed (or ≤110 multi-seed mean) merges;
  borderline → multi-seed; > 115 → close (slice-128 is the ceiling).
- Status: assigned, draft, status:wip.

## 2026-04-28 02:15 — PR #326 (iter 2): FFN ratio mlp_ratio={2, 3} on slice-128 ❌ CLOSED

- Branch: `willowpai2d2-edward/mlp-ratio-4` (deleted on close).
- Iteration: send-back asked for mlp_ratio=3 + control mlp_ratio=2 on
  shared `--wandb_group "willow-r2-edward-ffn-v2"`.
- Branch was rebased onto slice-128 (good) but pre-#330 (so MSE in
  loss). Each step up in mlp_ratio costs ~30 % per-step → ~1 fewer
  epoch in 30-min budget → ~3 mae_surf_p worse.

| Variant | mlp_ratio | val_avg/mae_surf_p | best_epoch | run id |
|-|-:|-:|-:|-|
| Baseline reference (#328 winner) | 2 | 133.55 | 11 | s1p2qs7l |
| (b) control re-run | 2 | 136.54 | 9 | r8bvfbvh |
| (a) primary | 3 | 139.79 | 10 | 3k44156d |
| (round-1 v1, slice-64) | 4 | 137.83 | 11 | ywy4j9e4 |

### Conclusion

**Closed.** Three FFN variants now tested across two architectures —
clean monotone trend that *smaller is better at the 30-min cap*. The
control re-run (b) lands 3 mae_surf_p worse than the published
baseline at identical config (corroborates the ±10 % single-seed
noise floor). Per-split signal is anti-stack: mlp_ratio=3 wins only
on `val_single_in_dist` and loses on **all three OOD splits** — the
opposite direction we'd expect if extra FFN capacity were generalizing.

vs the merged Huber baseline (115.61), best variant (139.79) is
21 % worse — well past the close threshold. Branch was also
pre-#330, but the regression is severe enough that closing is right
regardless of rebase.

The student's own follow-up #1 explicitly recommended closing.

### Bonus — corroboration of seed noise

(b) control re-run vs published baseline: 136.54 vs 133.55 = 3-point
spread at identical config. Same magnitude as thorfinn's #337
replicate (153.19 vs 139.39 → ~14 points at higher absolute, ~9 %).
Two independent same-config replicates now confirm the ±10 % noise
floor that drove the merge decision rules on alphonse #311 and
nezuko #332 send-backs.

### Reassignment

Edward → PR #429 (round-2 axis: per-channel loss weighting on
pressure). Zero compute cost (vs FFN's per-step penalty), direct
push on the metric we're ranked on, stacks naturally with all
in-flight axes. Detailed below.

## 2026-04-28 02:15 — PR #429 (NEW, edward round 2): per-channel loss weighting on pressure

- Reassigning edward after closing #326.
- Hypothesis: the loss currently weights all 3 target channels (Ux,
  Uy, p) equally despite the metric only counting surface pressure.
  Up-weighting `p` in the loss should mechanically push gradient
  toward better pressure fidelity, with at most a small Ux/Uy
  regression as the trade. Predicted 5–10 % reduction over the
  merged Huber baseline (115.61).
- Sweep: `p_weight ∈ {2.0, 3.0, 5.0}` on shared
  `--wandb_group "willow-r2-edward-p-weight"`. Implementation:
  `_channel_weighted_huber` helper that scales the pressure column
  of the per-element Huber tensor before the spatial reduction.
  MAE accumulation downstream is unweighted (untouched).
- Status: assigned, draft, status:wip.

## 2026-04-28 02:00 — PR #335 (iter 2): LR schedule — cosine_t_max sweep on slice-64 + MSE

- Branch: `willowpai2d2-tanjiro/warmup-cosine-1e3` (pre-#328 AND
  pre-#330; reverts both `slice_num=128` and `F.smooth_l1_loss` in
  train.py).
- Iteration: send-back asked for `cosine_t_max` CLI flag + 3-variant
  sweep on shared `--wandb_group "willow-r2-tanjiro-sched-v2"`.
- Three runs, single seed each, all 14/50 epochs (timeout):

| variant | lr | cosine_t_max | val_avg/mae_surf_p | run id |
|-|-:|-:|-:|-|
| (a) | 7e-4 | 18 | 115.15 | `2h6hkjqk` |
| **(b)** | **1e-3** | **15** | **113.96** | **`haitynmt`** |
| (c) | 8e-4 | 18 | 135.24 | `snbb8nnd` |

(a) and (b) are statistically tied (~1 % apart at single seed,
inside the ±10 % noise floor). (c) is plausibly seed-unlucky per
tanjiro's own analysis — the bowl shape disqualifies a structural
explanation since 8e-4 sits between 7e-4 and 1e-3.

### Conclusion

**Send back for rebase + on-baseline re-run.** Qualitatively, the
schedule-fitting hypothesis is decisively confirmed: both
budget-matched variants beat the round-1 cohort middle (133-138)
by ~15 %. The largest single signal in this iteration was the
mechanism — fitting cosine `T_max` to the achievable epoch budget
unlocks late-epoch LR shrinkage that was missing in the original
plain-cosine `T_max=50` schedule.

But: the run was on **slice-64 + MSE** (the pre-#328-#330 baseline
state, since the branch hadn't been rebased). Direct comparison of
113.96 vs the current Huber baseline 115.61 conflates the LR axis
with architecture-and-loss reverts. We need the schedule axis
re-run on top of the merged `slice_num=128 + Huber β=1` baseline
to attribute cleanly.

Sent back with: rebase (preserve `cosine_t_max` flag + warmup +
SequentialLR scheduler; take advisor's `slice_num=128` and
`F.smooth_l1_loss(...)`; discard doc reverts) → re-run variant (b)
`lr=1e-3, cosine_t_max=15` on top of merged baseline → optional
2 more seeds.

Decision rule on the rebased re-run, with bar at 115.61:
- ≤105 single-seed (or ≤110 multi-seed mean): merge.
- 105–115: multi-seed required.
- > 115: schedule axis doesn't stack on Huber cleanly — close, but
  the `cosine_t_max` infrastructure carried forward is still useful
  for round-2 work.

### Methodological note

The "warmup + flat 1e-3 vs warmup + actual cosine" causal chain
from round-1 (154.57) to round-2 (113.96) is the cleanest
attribution on this PR — same nominal hypothesis, two very
different empirical outcomes, with the `cosine_t_max` parameter
turning out to be the binding lever. Worth flagging this as a
methodology pattern: for any schedule we adopt going forward, the
cosine `T_max` should be tuned against the achievable wall-clock
horizon (~14 epochs for slice-128 + Huber), not the planned 50.

## 2026-04-28 01:45 — PR #367: Bug fix — guard accumulate_batch + evaluate_split against non-finite values

- Branch: `willowpai2d2-fern/scoring-nan-fix` (created post-#328 but
  pre-#330; same rebase pattern as the round-1 cohort).
- Patch verified working: `data/scoring.py::accumulate_batch` `err`
  wrapped in `torch.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`
  before the per-channel sum. Defense-in-depth on top of the
  existing `y_finite` mask: `mask=False` zeros sample-20's
  contribution from the denominator, `nan_to_num` neutralizes the
  `0.0 * NaN = NaN` poisoning of the numerator.
- Verification (run `mzrlvccy`, branch state = MSE on slice-128):
  - `test_geom_camber_cruise/mae_surf_p`: NaN → **101.70** (finite!)
  - `test_avg/mae_surf_p`: NaN → **125.05** (finite for the first
    time on this branch)
  - `val_avg/mae_surf_p = 133.46` ≈ pre-Huber baseline 133.55
    (replication within run-to-run noise)

### Conclusion

**Send back for rebase.** The bug fix is technically correct and
well-verified. But the branch is pre-#330 (Huber β=1 just merged
minutes before this PR was reviewed), so:

1. `train.py::evaluate_split` and the training loop both have `(pred -
   y_norm) ** 2` (MSE) instead of the merged `F.smooth_l1_loss(...)` —
   direct squash would silently revert the −13.4 % Huber win.
2. `nan_to_num` was applied to the MSE expression; after rebase, the
   `nan_to_num` needs to wrap the Huber `F.smooth_l1_loss(...)` instead.
3. Stale doc reverts (BASELINE.md, research/) need to be discarded.

Sent back with explicit conflict-resolution instructions. After
rebase, expect:
- `data/scoring.py`: `nan_to_num` wrap on `err` (unchanged).
- `train.py::evaluate_split`: `nan_to_num(F.smooth_l1_loss(...), ...)`.
- Optional defense-in-depth on training-loop Huber.
- Verification: `val_avg ~ 115.61` (replicating Huber baseline) AND
  finite `test_avg` (the bug fix's actual target).

### Three independent confirmations of the bug

This is the third independent diagnosis (after edward #326 + askeladd
#325) of the same root cause and equivalent patch. Convergent evidence
makes me very confident the fix is right; the only remaining work is
landing it on the right base.

## 2026-04-28 01:30 — PR #330 (rebased): Round 1 axis: loss formulation — MSE → Huber (β=1) ★ MERGED ★

- Branch: `willowpai2d2-frieren/huber-loss` (rebased onto current
  advisor; diff is exactly the 2-line `(pred - y_norm)**2 →
  F.smooth_l1_loss(...)` substitution in train loop + evaluate_split).
- Run: `uip4q05z` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/uip4q05z)

### Results (best checkpoint, epoch 11 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 137.21 | 1.56 | 0.74 |
| val_geom_camber_rc | 118.60 | 2.44 | 0.97 |
| val_geom_camber_cruise | 98.74 | 1.28 | 0.55 |
| val_re_rand | **107.89** | 2.00 | 0.74 |
| **val_avg** | **115.61** | 1.81 | 0.75 |
| test_single_in_dist | 125.75 | 1.63 | 0.69 |
| test_geom_camber_rc | 108.59 | 2.49 | 0.88 |
| test_geom_camber_cruise | NaN ⚠ | 1.17 | 0.49 |
| test_re_rand | 105.10 | 1.75 | 0.71 |

### Conclusion

**Merged** as the new round-1 baseline at `val_avg/mae_surf_p = 115.61`
— a **−13.4 %** delta over the prior baseline (133.55), comfortably
outside the ±10 % single-seed noise floor. The strongest delta is on
`val_re_rand` (−14.1 %), confirming the high-Re-tail mechanism
predicted in the original hypothesis.

### Decomposition: how the original 18 % became 13.4 %

| Setting | val_avg/mae_surf_p | Δ vs same-arch baseline |
|-|-:|-:|
| slice_num=64 + MSE (implicit pre-#328 baseline) | (no finished run) | – |
| slice_num=64 + Huber β=1 (frieren original, slice-64) | 109.47 | – |
| slice_num=128 + MSE (PR #328 merged baseline) | 133.55 | – |
| **slice_num=128 + Huber β=1 (this merge)** | **115.61** | **−13.4 %** |

The original 18 % gain on slice-64 → 13.4 % on slice-128 means
slice-128 already absorbed ~5 % of the original Huber benefit (more
physics tokens → better high-Re tail resolution at the architectural
level). What remains is the pure loss-shape effect: Huber's L1 tail
matches the L1 metric better than MSE's quadratic tail, regardless
of architecture. Mechanism is robust to the slice count.

### Loss-formulation axis is the highest-leverage axis in round 1

| Axis | Best Δ vs baseline (single-axis, any cohort run) |
|-|-:|
| Loss formulation (Huber β=1) | **−13.4 %** (merged) |
| Architecture (slice_num=128) | −0 to −5 % (was the original baseline anchor) |
| All other architecture axes | inside ±10 % noise vs baseline |

Two clean wins land in round 1 — loss formulation and physics-token
count — and they compose.

## 2026-04-28 01:30 — PR #415 (NEW, frieren round 2): asinh transform on pressure target

- Reassigning frieren after the #330 merge.
- Hypothesis: pair the merged Huber-β=1 with `torch.asinh` on the
  pressure channel of `y_norm` (only — Ux/Uy unchanged) before the
  loss. asinh compresses the high-Re tail of the residual
  distribution; Huber clips its gradient. Orthogonal mechanisms,
  same target failure mode. Predicted 3–8 % reduction in
  `val_avg/mae_surf_p` on top of 115.61.
- Status: assigned, draft, status:wip.

## 2026-04-28 01:15 — PR #325: Round 1 axis: model depth — n_layers 5 → 8 ❌ CLOSED

- Branch: `willowpai2d2-askeladd/depth-8` (deleted on close)
- Hypothesis: 3–6 % reduction in `val_avg/mae_surf_p` from
  `n_layers 5 → 8`.
- Run: `eo8zq50l`

### Result (best checkpoint, epoch 8 / 9 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 202.92 | 2.15 | 1.06 |
| val_geom_camber_rc | 176.90 | 3.66 | 1.27 |
| val_geom_camber_cruise | 129.89 | 1.75 | 0.81 |
| val_re_rand | 138.50 | 2.59 | 1.01 |
| **val_avg** | **162.05** | 2.54 | 1.04 |

**21.3 % regression vs merged baseline (133.55) — well outside ±10 %
noise band on a single seed. Closed.**

### Why depth-8 doesn't pay at the 30-min cap

- 9 epochs / 50 reached (vs ~14 for baseline-equivalent at depth-5).
- Each epoch ~206 s vs baseline's ~120 s.
- Val curve broadly descending but with high epoch-to-epoch variance
  (228 → 263 → 219 → 225 → 234 → 183 → 197 → 162 → 170): could
  reach competitive numbers given enough compute, but cannot at
  fixed 30-min wall clock.
- Same compute disadvantage as width-192: capacity scaling consumes
  the budget faster than additional epochs can recover the gap.

Depth axis is **deferred to round 2**, after AMP/bf16 unlocks more
throughput per cycle. At that point a deeper net inside the same
wall-clock becomes feasible — askeladd's val-trajectory data here
will be useful round-2 reference.

### Bonus: third independent confirmation of cruise-NaN bug

Askeladd independently identified the `nan * 0 = nan` propagation
in `accumulate_batch`, recomputed `test_avg/mae_surf_p = 148.72`
offline by patching with `torch.where(effective.unsqueeze(-1), err,
torch.zeros_like(err))`. Three students (edward, fern, askeladd)
have now converged on the same root cause and equivalent patches —
strong cross-validation that PR #367 is correct.

## 2026-04-28 01:15 — PR #399 (NEW): Round 2 axis: bf16 mixed precision

- Branch: `willowpai2d2-askeladd/bf16-amp`
- Reassigning askeladd to a higher-leverage axis after closing #325.
- Hypothesis: bf16 autocast around the model forward (loss kept in
  fp32) gives ~1.4–1.8× per-step speedup with no dynamic-range
  collapse (avoiding alphonse's fp16 failure mode), enabling
  20–25 epochs in budget vs current ~14. Predicted 3–8 % reduction
  in `val_avg/mae_surf_p` from extra training alone.
- Status: assigned, draft, status:wip.

## 2026-04-28 01:00 — PR #332: Round 1 axis: surface-vs-volume weight — surf_weight 10 → 25 (sweep)

- Branch: `willowpai2d2-nezuko/surf-weight-25` (pre-#328; same
  rebase need as the rest of the cohort).
- Hypothesis: 5–10 % reduction in `val_avg/mae_surf_p` with a small
  trade against vol fidelity, since the metric only counts surface
  pressure but the loss weights all 3 channels.
- Three runs, single seed each, same `--wandb_group "willow-r2-nezuko-surf-sweep"`.

### Sweep results (best checkpoint, ~14 epochs / 50 — wall-clock cut)

| surf_weight | val_avg/mae_surf_p | val_avg/mae_vol_p | best_epoch | run id |
|-:|-:|-:|-:|-|
| 15.0 | 137.42 | 144.60 | 13 | `rbj9qple` |
| **25.0** | **133.19 ★** | 142.51 | 13 | `ew04amt0` |
| 40.0 | 142.59 | 180.42 | 12 | `wfo3ptci` |

Curve shape: clear interior optimum at 25, with vol_p stable from
15 → 25 (144.6 → 142.5) but collapsing past 40 (180.4) — the
optimizer starts sacrificing volume fidelity faster than it improves
surface fidelity. Two independent pieces of evidence (val_surf_p
local minimum + val_vol_p stability boundary) point at the same
operating point.

### Per-split surface-pressure MAE (best epoch)

| surf_weight | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand |
|-:|-:|-:|-:|-:|
| 15.0 | 166.03 | 151.57 | 101.23 | 130.86 |
| **25.0** | 160.78 | 147.47 | 105.65 | 118.85 |
| 40.0 | 176.57 | 144.87 | 113.84 | 135.09 |

`test_geom_camber_cruise/mae_surf_p = NaN` on all three runs (known
#367 bug); 3-split partial test means: 139.42 / 131.22 / 141.28
respectively.

### Conclusion

**Send back for rebase + on-baseline re-run.** `surf_weight=25` is
qualitatively the right operating point — sweep curve and val_vol_p
secondary signal both confirm the interior optimum. **But:**
133.19 vs the merged 133.55 baseline is only **0.27 %** under, far
inside the ±10 % single-seed noise floor. Cannot merge an
inside-noise number, and the branch would silently revert
slice_num=128 → 64 anyway. Sent back with rebase + re-run on
slice-128 + optional 2 more seeds for noise estimate.

Decision rule made explicit:
- ≤120 single-seed (or ≤126 multi-seed mean): merge.
- 120–130: multi-seed required.
- > 130: surf_weight axis doesn't stack with slice-128; close.
  Even if we close, `surf_weight=25` remains a useful default knob
  for round-2 stack candidates given the qualitative confirmation.

### Bonus: cruise NaN root-cause confirmed independently again

Nezuko verified `000020.pt`'s `y[:,2]` has `inf` (vs edward's "761
NaN" — different IEEE pathologies of the same poisoned sample). Both
diagnoses point at the same root cause; both are fixed by the
`nan_to_num` patch in PR #367.



- Branch: `willowpai2d2-alphonse/width-192` (still pre-#328; same
  rebase need as frieren #330 + thorfinn #337).
- Iteration: my send-back asked for width-160 (compute-equal middle
  ground) + optional AMP-fp16 baseline at width-128.

### (a) width-160 — `qrmztk33`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 154.06 | 1.80 | 1.01 |
| val_geom_camber_rc | 135.83 | 2.83 | 1.18 |
| val_geom_camber_cruise | 96.45 | 1.47 | 0.64 |
| val_re_rand | 118.38 | 2.46 | 0.90 |
| **val_avg** | **126.18** | 2.14 | 0.94 |

- Best epoch 11/50 (timeout). 166 s/epoch. Peak 50.1 GB / 96 GB
  (52 %). Params 1.03 M (1.55× baseline).
- W&B: https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/qrmztk33

**Width-160 dominates width-192 at 30-min cap on every dimension:**

| variant | params | epochs in 30 min | best val_avg/mae_surf_p | peak GB |
|-|-:|-:|-:|-:|
| 160-d (this iter) | 1.03 M | 11 | **126.18** | 50.1 |
| 192-d (prev iter) | 1.47 M | 10 | 134.13 | 88.8 |

### (b) AMP-fp16 at width-128 — `qyfizq4i` (terminated)

Diverged at epoch 3: train loss → NaN, val_avg → NaN, GradScaler
permanently skipping steps. Student terminated mid-epoch 6.

Pattern (per epoch):

| Epoch | train[vol/surf] | val_avg_surf_p |
|-|-|-|
| 1 | 1.83 / 1.04 | 247.74 |
| 2 | 1.10 / 0.64 | 245.50 |
| 3 | nan / nan | nan |
| 4–5 | nan / nan | nan |

Speed gain real (~1.7× per step at width-128 fp16); memory 32.9 GB
vs ~50 GB for width-160 fp32. Failure mode: `(pred - y_norm)²` term
multiplied by `surf_weight=10` overflows fp16 dynamic range on
surface residuals, especially during the high-loss early epochs.

Student-proposed fixes (correct, deferred to dedicated round-2 PR):
1. Compute the `sq_err` and per-mask sums in fp32, only the model
   forward in fp16 (scoped autocast).
2. Switch to `bfloat16` (no GradScaler needed on Hopper-class GPUs;
   bf16 has fp32 dynamic range).

### Conclusion

**Send back for rebase + on-baseline re-run.** 126.18 single-seed is
5.5 % under the merged baseline 133.55, but **inside the ±10 %
single-seed noise floor** observed from thorfinn's replicate (#337).
Cannot merge from a single-seed inside-noise number, especially
because the branch would also silently revert slice_num=128 → 64.

Sent back with:
- Rebase onto current advisor (slice_num=128).
- Re-run width-160 on top of the slice-128 baseline (single seed
  is fine if the new number lands clearly outside ±10 % of 133.55,
  i.e. ≤120; multi-seed required if borderline 120–130).
- Optional: add `SENPAI_SEED` env-var seed handling for multi-seed.
- Decision rule made explicit in the send-back comment so the
  student can self-judge whether to push more seeds.

bf16 AMP retry deferred to a dedicated round-2 PR.



### Results (best checkpoint, epoch 10 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 160.79 | 2.29 | 0.91 |
| val_geom_camber_rc | 148.93 | 3.43 | 1.26 |
| val_geom_camber_cruise | 105.93 | 1.64 | 0.65 |
| val_re_rand | 120.87 | 2.38 | 0.91 |
| **val_avg** | **134.13** | 2.43 | 0.93 |
| test_single_in_dist | 137.18 | 2.14 | 0.85 |
| test_geom_camber_rc | 132.05 | 3.22 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.57 | 0.60 |
| test_re_rand | 121.55 | 2.14 | 0.89 |
| **test_avg** | NaN ⚠ | 2.27 | 0.88 |

### Conclusion

**Send back** for compute-equal follow-up. The wider model is
clearly undertrained at the 30-min cap (10 of 50 epochs reached;
~184s/epoch vs the ~36s/epoch of baseline ⇒ ~5× slower per step,
not the predicted 2–2.25×). Peak GPU memory at 92.89% leaves no
headroom for stacking. Val curve still descending steeply at epoch
10 (258 → 134), so the metric reflects undertraining rather than
the asymptotic capacity of width-192. The 134.13 number is at
the front of the round-1 cohort but not interpretable as a clean
"width helps" signal.

Sent back with: try **width-160** (1.55× params, divisible by 4),
expected 20–25 epochs in budget; optionally a same-PR AMP-only
baseline at width-128 to disentangle precision from architecture.

## 2026-04-27 23:30 — PR #335: Round 1 axis: LR schedule — 5-epoch warmup + cosine, peak 1e-3

- Branch: `willowpai2d2-tanjiro/warmup-cosine-1e3`
- Hypothesis: 3–7% reduction in `val_avg/mae_surf_p` from
  `lr 5e-4 → 1e-3` with 5-epoch linear warmup + cosine decay.
- Run: `ri332d19` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ri332d19)

### Results (best checkpoint, epoch 13 / 14 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 212.25 | 2.11 | 1.01 |
| val_geom_camber_rc | 149.98 | 2.96 | 1.24 |
| val_geom_camber_cruise | 120.32 | 1.67 | 0.62 |
| val_re_rand | 135.73 | 2.26 | 0.95 |
| **val_avg** | **154.57** | 2.27 | 0.95 |
| test_single_in_dist | 178.60 | 2.00 | 0.96 |
| test_geom_camber_rc | 138.07 | 2.86 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.52 | 0.57 |
| test_re_rand | 137.11 | 2.11 | 0.89 |
| **test_avg** | NaN ⚠ | 2.12 | 0.90 |

### Conclusion

**Send back** for schedule-shape iteration. The warmup wiring is
correct (verified from W&B `lr` panel: 1e-4 → 1e-3 over epochs
1–5, then cosine decay engages). But the 30-min cap only allows
14 epochs, so cosine `T_max=50` decays only ~9.5% of its arc —
the schedule is effectively "warmup + flat 1e-3," not the
warmup+cosine the hypothesis was testing. 154.57 sits at the
bottom of the round-1 cohort (133.55–154.57 range), consistent
with a flat high LR overshooting the local optima that lower
LRs can navigate in a short budget.

Sent back with: parametrize `--cosine_t_max` as a CLI flag, run a
small sweep `(lr 7e-4, T_max 18)` and `(lr 1e-3, T_max 15)` on
a shared `--wandb_group "willow-r2-tanjiro-sched-v2"`. Optional
third variant `(lr 8e-4, T_max 18)`.

## 2026-04-27 23:50 — PR #328: Round 1 axis: physics-token count — slice_num 64 → 128 ★ MERGED ★

- Branch: `willowpai2d2-fern/slice-num-128`
- Hypothesis: 2–5% reduction in `val_avg/mae_surf_p`, with a stronger
  effect on `val_geom_camber_*` and `val_re_rand` than on
  `val_single_in_dist`.
- Run: `s1p2qs7l` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/s1p2qs7l)

### Results (best checkpoint, epoch 11 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 164.27 | 1.822 | 0.892 |
| val_geom_camber_rc | 139.35 | 3.023 | 1.156 |
| val_geom_camber_cruise | 104.92 | 1.521 | 0.635 |
| val_re_rand | 125.66 | 2.284 | 0.900 |
| **val_avg** | **133.55** | 2.162 | 0.896 |
| test_single_in_dist | 140.58 | 1.800 | 0.830 |
| test_geom_camber_rc | 131.24 | 3.013 | 1.086 |
| test_geom_camber_cruise | NaN ⚠ | 1.425 | 0.586 |
| test_re_rand | 122.71 | 2.083 | 0.887 |
| **test_avg** | NaN ⚠ | 2.080 | 0.847 |

### Conclusion

**Merged.** This is the round-1 winner — lowest `val_avg/mae_surf_p`
of any finished run (133.55 vs next-best 134.13). Per-split signal
supports the slice-bottleneck hypothesis cleanly: best-of-cohort on
`val_geom_camber_rc` (139.35; ~7% better than next-best), competitive
on the rest. Mild compute overhead (54.5 GB peak, 172s/epoch =
~10–15% slowdown vs slice-64). At epoch 11/50 the val curve is still
descending steeply, so more headroom likely exists. New
`BASELINE.md` anchor point.

Per-split contrast vs other round-1 runs (cohort observation):

| Split | fern slice=128 (this) | alphonse w=192 | edward mlp=4 | nezuko sw=15 |
|-|-:|-:|-:|-:|
| val_single_in_dist | 164.27 | **160.79** | 176.14 | 166.03 |
| val_geom_camber_rc | **139.35** | 148.93 | 154.36 | 151.57 |
| val_geom_camber_cruise | 104.92 | 105.93 | **99.96** | 101.23 |
| val_re_rand | 125.66 | **120.87** | 120.87 | 130.86 |

## 2026-04-27 23:50 — PR #326: Round 1 axis: FFN ratio — mlp_ratio 2 → 4

- Branch: `willowpai2d2-edward/mlp-ratio-4`
- Hypothesis: 3–5% reduction in `val_avg/mae_surf_p` from
  `mlp_ratio 2 → 4` (standard transformer FFN expansion).
- Run: `ywy4j9e4` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ywy4j9e4)

### Results (best checkpoint, epoch 11 / 13 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 176.14 | 2.135 | 0.897 |
| val_geom_camber_rc | 154.36 | 3.045 | 1.185 |
| val_geom_camber_cruise | 99.96 | 1.346 | 0.654 |
| val_re_rand | 120.87 | 2.076 | 0.906 |
| **val_avg** | **137.83** | 2.151 | 0.910 |
| test_single_in_dist | 155.51 | 1.985 | 0.860 |
| test_geom_camber_rc | 143.53 | 2.998 | 1.120 |
| test_geom_camber_cruise | NaN ⚠ | 1.255 | 0.604 |
| test_re_rand | 121.85 | 1.968 | 0.877 |
| **test_avg** | NaN ⚠ | 2.052 | 0.865 |

### Conclusion

**Send back** for `mlp_ratio=3` iteration. 137.83 doesn't beat the
new merged baseline (133.55) — ~3.2% worse. The FFN-axis hypothesis
isn't dead, just budget-confounded: `mlp_ratio=4` is 2.1× slower per
epoch (148s vs ~70s baseline), chopping the budget to 13 of 50
epochs (~26%). At equal epochs the wider FFN might still win, but
under the 30-min cap the cost dominates. Best on
`val_geom_camber_cruise` (99.96, the only round-1 sub-100), so the
expanded FFN does help on the largest meshes.

Sent back with: rebase onto the new baseline (slice_num=128) and
sweep `mlp_ratio={3, 2-control}` on shared `--wandb_group
"willow-r2-edward-ffn-v2"`. Optional SwiGLU at `mlp_ratio=8/3` if
budget allows. SwiGLU saved for a follow-up PR if the simple sweep
shows promise — keep this iteration's scope to FFN ratio.

### Bonus: independent root-cause analysis of `test_geom_camber_cruise` NaN

Edward independently confirmed fern's flag, with full reproduction
(sample 20 of `test_geom_camber_cruise` GT has 761 non-finite values
in the pressure channel) and a 2-line proposed fix
(`torch.nan_to_num` of `err` before the masked sum). Bug-fix PR #367
assigned to fern (now idle after merging #328) using edward's
proposed patch.

## 2026-04-28 00:10 — PR #330: Round 1 axis: loss formulation — MSE → Huber (β=1)

- Branch: `willowpai2d2-frieren/huber-loss`
- Hypothesis: 2–5 % reduction in `val_avg/mae_surf_p`, with strongest
  gains on `val_re_rand` (high-Re tail story).
- Run: `ic77vvgj` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ic77vvgj)
- W&B run config confirmed: `slice_num = 64` (the **pre-#328
  baseline** at the time the branch was created).

### Results (best checkpoint, epoch 14 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 127.98 | 1.57 | 0.76 |
| val_geom_camber_rc | 126.26 | 2.42 | 0.98 |
| val_geom_camber_cruise | 82.81 | 1.63 | 0.54 |
| val_re_rand | 100.85 | 2.07 | 0.74 |
| **val_avg** | **109.47** | 1.93 | 0.75 |
| test_single_in_dist | 114.05 | 1.52 | 0.72 |
| test_geom_camber_rc | 114.15 | 2.41 | 0.94 |
| test_geom_camber_cruise | NaN ⚠ | 1.62 | 0.49 |
| test_re_rand | 95.55 | 1.94 | 0.70 |
| **test_avg** | NaN ⚠ | 1.87 | 0.71 |

### Conclusion

**Send back for rebase.** The result is the largest single-axis jump
yet (~18 % better than the merged 133.55 baseline) and the per-split
signal supports the hypothesis cleanly: best in cohort on every val
track, including the predicted-strongest signal on `val_re_rand`
(100.85, 17 % better than next-best). **However**, the branch was
created before PR #328 merged and still has `slice_num = 64`. A
direct squash-merge would silently revert the merged slice-128
architectural improvement.

Sent back for: rebase onto current `icml-appendix-willow-pai2d-r2`
(slice_num=128 baseline) and re-run with the same Huber loss change
to confirm the gain stacks on top of slice-128. Even if the rebased
number is slightly worse than 109.47 (because slice-128 already
captured some of the original gain), the hypothesis is supported and
this should land as the next baseline.

### Why such a large gain?

Likely combination of two effects: (a) Huber's linear-tail behavior
is a better proxy than MSE for the L1 metric we're ranked on,
particularly under the `surf_weight=10` multiplier that amplifies
surface-tail residuals; (b) at high Re, normalized residuals exceed
1.0 enough that Huber's gradient clipping prevents tail samples
from dominating the update. Frieren's `val_re_rand=100.85` (best in
cohort) is direct evidence of (b).

## 2026-04-28 00:25 — PR #337: Round 1 axis: batch + LR scaling — BS 4→8, lr 5e-4→7e-4

- Branch: `willowpai2d2-thorfinn/batch-8-lr-7e4`
- Hypothesis: 2–5 % reduction in `val_avg/mae_surf_p` from BS 4→8
  with sqrt-rule LR scaling. The hypothesis was CLI-only — no code
  changes.
- W&B run config confirmed: `slice_num = 64` (the **pre-#328
  baseline** at the time the branch was created — same rebase need
  as frieren #330).

### Results (two runs, same config — seed-variance datapoint)

Both runs used `--batch_size 8 --lr 7e-4` with all other defaults.

| W&B run | best_val_avg/mae_surf_p | best_epoch |
|-|-:|-:|
| `kon60q79` (run 2) | **153.19** | 13 |
| `nphltrz9` (run 1) | **139.39** | 14 |
| **delta** | **+13.80 (~10 %)** | – |

Per-split for `kon60q79` (the primary):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 240.41 | 2.56 | 1.10 |
| val_geom_camber_rc | 141.54 | 2.93 | 1.19 |
| val_geom_camber_cruise | 109.02 | 1.68 | 0.60 |
| val_re_rand | 121.78 | 2.44 | 0.89 |
| **val_avg** | **153.19** | 2.40 | 0.94 |

Test side: `test_geom_camber_cruise/mae_surf_p = NaN` on both runs
(known #367 bug); other test splits finite.

### Critical methodology finding: seed variance ≈ ±10 %

The two thorfinn runs shared **every** configuration knob and command
flag — they differ only in the random seed implied by torch's
default initialization plus DataLoader shuffle order. Spread of
~10 % between them establishes a **noise floor** that affects how
every other round-1 PR's number should be interpreted.

This is a methodological constraint, not a hypothesis result. It
means single-seed differences smaller than ~10 % between runs are
not statistically distinguishable. Concretely, on this branch:

- alphonse 134.13, fern 133.55, nezuko surf-15 137.42, edward 137.83
  are all inside ±10 % of each other and of the merged baseline —
  effectively tied within seed noise.
- Frieren's 109.47 sits well outside the ±10 % band of 133.55, so
  the Huber win is a real signal even at single seed.
- Future PR results in the borderline band (within ±10 %) will need
  multi-seed replication before merging.

### Conclusion on the hypothesis

**Send back** for rebase + push the lever. Best of two runs (139.39)
is ~4.4 % worse than the merged baseline 133.55 — not a clear
regression; 2-run mean (146.29) is ~9.5 % worse — outside noise on
the mean. Hypothesis isn't dead, just under-tested at the lever's
limit. VRAM has clear headroom (84 GB peak / 96 GB cap) for BS=12 or
BS=16, which the original PR didn't explore. Sent back with:

- Rebase onto current advisor (slice_num=128).
- **Primary follow-up:** BS=16 + lr=1e-3 (sqrt-rule scaling from BS=4
  baseline = 5e-4 · √4 = 1e-3).
- **Multi-seed where wall-clock allows** (3 seeds at BS=16/lr=1e-3 if
  budget allows — single seed if not). Add explicit seed via
  `SENPAI_SEED` env var so the runs are deterministic.
- **Fallback** if BS=16 OOMs: BS=12 + lr ≈ 9e-4.

Schedule mismatch (cosine T_max=50 vs 14-epoch achievable budget) is
acknowledged but kept out of scope — tanjiro is iterating on
`--cosine_t_max` in #335.

## Round-1 cohort observation (current snapshot)

| W&B name | best_val_avg/mae_surf_p | best_epoch | status |
|-|-:|-:|-|
| **willow-r2-frieren-huber-b1-on-slice128** | **115.61 ★** | 11 | **merged (PR #330)** |
| willow-r2-frieren-huber-b1 (slice-64, original) | 109.47 | 14 | superseded by rebased re-run |
| willow-r2-alphonse-width-160 | **126.18** | 11 | sent back (rebase + on-baseline) |
| willow-r2-fern-slice-128 | 133.55 (prior baseline) | 11 | merged (PR #328) — superseded as baseline by #330 |
| willow-r2-alphonse-width-192 | 134.13 | 10 | superseded by width-160 |
| willow-r2-nezuko-surf-15 | 137.42 | 13 | sweep complete (slice-64) |
| willow-r2-nezuko-surf-25 | 133.19 | 13 | superseded by iter 2 |
| willow-r2-nezuko-surf-40 | 142.59 | 12 | sweep complete (past optimum, slice-64) |
| willow-r2-nezuko-surf25-on-slice128 (3 seeds) | 154.63 / 152.13 / 149.07 (mean 151.94) | 8/10/11 | **closed** (#332 — axis doesn't stack with slice-128) |
| willow-r2-nezuko-lion-lr | – | – | NEW assignment (#472, round-2 axis) |
| willow-r2-edward-mlp-ratio-4 | 137.83 | 11 | superseded by iter 2 |
| willow-r2-edward-mlp-2-slice-128 (control) | 136.54 | 9 | **closed** (#326 — FFN axis exhausted) |
| willow-r2-edward-mlp-3 | 139.79 | 10 | **closed** (#326 — FFN axis exhausted) |
| willow-r2-edward-p-weight | – | – | NEW assignment (#429, round-2 axis) |
| willow-r2-thorfinn-bs8-lr7e-4 | 139.39 / 153.19 (slice-64, 2 seeds) | 14 / 13 | superseded by iter 2 |
| willow-r2-thorfinn-bs6-lr6e-4-slice128 (3 seeds) | 156.15 / 174.69 / 157.05 (mean 162.63) | 8/10/11 | **closed** (#337 — hardware-blocked at BS≥8) |
| willow-r2-thorfinn-ema-weights | – | – | NEW assignment (#457, round-2 axis) |
| willow-r2-askeladd-depth-8 | 150.06 / 162.05 | 9 / 8 | **closed** (#325 — 21 % regression at 30-min cap) |
| willow-r2-askeladd-bf16-amp (3 seeds, slice-128 + MSE) | 124.94 / 134.30 / 141.13 (mean 133.46) | 12/13/11 | sent back (rebase + on-baseline) |
| willow-r2-tanjiro-warmup-cos-1e3 | 154.57 | 13 | sent back |

**Noise floor:** ±10 % at single seed (thorfinn replicate evidence).
Merged baseline 133.55 has implicit ±13 in either direction at
single-seed precision — winners must beat this band convincingly,
not by 1-3 %. Frieren's 109.47 is the only result outside the band.
