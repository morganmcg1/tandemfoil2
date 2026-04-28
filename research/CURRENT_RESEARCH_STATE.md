# SENPAI Research State — charlie-pai2d-r3

- **Date:** 2026-04-27
- **Advisor branch:** `icml-appendix-charlie-pai2d-r3`
- **Round:** 3 (8 idle students at start)

## Programme target

Beat the unmodified Transolver baseline on TandemFoilSet, ranked by
`val_avg/mae_surf_p` (val) and `test_avg/mae_surf_p` (test). The four val/test
tracks measure (a) in-distribution single-foil sanity, (b) unseen front-foil
camber for raceCar tandem, (c) unseen front-foil camber for cruise tandem,
(d) stratified Re holdout across all tandem domains. A useful change should
help across at least three of the four tracks.

## Round 3 focus

**Current measured baseline (merged 2026-04-28 05:24):**
PR #534 (fern) — **L1 + spatial FF + EMA(0.997) + matched cosine
+ lr=7.5e-4 + grad clipping**. `val_avg/mae_surf_p = 78.60`,
`test_avg/mae_surf_p = 67.77`. Wins on every val and test split.
Ninth merge of round 3.

**Cumulative round-3 improvement: −41.9% on val, −45.0% on test**
from PR #306 reference.

**Caveat**: PR #534 was branched off pre-#506 advisor (FF=8). The
post-merge advisor has FF=12 (from #506) + EMA=0.997 (from this PR).
The actual joint config (FF=12 × EMA=0.997 + everything else) is
**untested** but expected to land ≤ 78.60 since FF=12 was a +1.57%
lever in PR #506.

**Schedule × averaging interference RESOLVED**: PR #476 (matched
cosine × EMA(0.999)) was destructive (+1.1% val); PR #534 (matched
cosine × EMA(0.997)) is clean compose (−0.25% / −1.97%). The fix
is shorter EMA window (1000 steps → 333 steps), ending averaging
before cosine tail weight-collapse.

**Round-3 baseline lineage:**
| Round | best val | best test | lever | Δ vs prior |
|-------|---------:|----------:|-------|----:|
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial FF | **−10.5% / −17.0%** |
| PR #389 |  90.90 |  80.84 | + matched cosine (CLI) | −1.06% / −0.33% |
| PR #447 |  82.97 |  73.58 | + EMA(0.999) | **−8.7% / −9.0%** |
| PR #461 |  80.28 |  70.92 | + lr=7.5e-4 (CLI) | **−3.2% / −3.6%** |
| PR #462 |  80.06 |  70.04 | + grad clipping max_norm=1.0 | −0.27% / −1.24% |
| PR #506 |  78.80 |  69.13 | + NUM_FOURIER_FREQS=12 | −1.57% / −1.30% |
| **PR #534** | **78.60** | **67.77** | **+ EMA_DECAY=0.997 (schedule × EMA fix)** | **−0.25% / −1.97%** |

**Round-3 proven levers (cumulative, eight stacked)**:
1. L1 surface loss (PR #280)
2. 8→12-freq spatial FF (PR #400 → PR #506)
3. Matched cosine `--epochs 14` (PR #389, CLI)
4. EMA-of-weights, decay=0.999→0.997 (PR #447 → PR #534)
5. Peak LR `lr=7.5e-4` (PR #461, CLI)
6. Gradient clipping max_norm=1.0 (PR #462)
7. NUM_FOURIER_FREQS=12 (PR #506) — refinement of lever #2.
8. **EMA_DECAY=0.997** (PR #534) — refinement of lever #4.

Recommended reproduce: `python train.py --epochs 14 --lr 7.5e-4`.

**Round-5 priorities** (refreshed):
- **Tighter clipping bracket** (`max_norm ∈ {0.5, 0.1}`) — pre-clip
  grad norms still ~27× threshold at cosine tail (edward PR #462).
  edward PR #499 in flight at max_norm=0.5.
- **L1-volume compose test on post-#462 advisor** — tanjiro PR #492
  in flight.
- **DropPath / stochastic depth** — mechanistically-different
  regulariser; the wd × FF and beta2 × FF compose failures (PRs #437,
  #446) were both magnitude-based regularisers. thorfinn PR #501 in
  flight.
- **wd=5e-4 compose** on post-#462 advisor — frieren validated the
  sweet spot in PR #469. frieren PR #500 in flight (compose with full
  6-lever stack).
- **Spatial FF frequency-count bracket** (NUM_FOURIER_FREQS=12) —
  nezuko's own follow-up to PR #400; the input-encoding axis that
  worked (vs log(Re) FF which didn't, PR #432).
- **Auxiliary log-pressure target** — `val_single_in_dist` is now
  93.59 (closing fast) but still the worst split. Heavy-tail compress.

**Closed PRs (2026-04-28):**
   - PR #283 — askeladd (wider+deeper): val 166.64 (+62% vs L1 baseline);
     compute-starvation under wallclock cap is structural.
   - PR #366 — thorfinn (mlp_ratio=4): val 144.70 (+41% vs L1 baseline);
     in-dist improved, all OOD axes regressed → generalisation-gap shift.
   - PR #292 — frieren (slice_num=128): val 149.08 (+45% vs L1 baseline);
     ~4% seed-noise floor swamped the predicted effect — not ruled out
     for round 4 with seeded runs / larger slice bumps.
   - PR #288 — fern (warmup + lr=1e-3): val 147.50 (+44% vs L1 baseline);
     mean of 3 seeded runs 143.2±5.7 — schedule mismatched with wallclock
     cap (warmup eats 21% of actual epoch budget).
   - PR #298 — nezuko (Fourier features 8-freq): **val 116.62 — −13.7%
     vs MSE peer (PR #306) but +13.6% vs current L1 baseline**. Lever
     validated on its assigned baseline; loses to the merged L1 baseline
     by merge-order timing. Re-assigned as L1 + Fourier compose test.
   - PR #390 — thorfinn (L1 + bs=8 + sqrt-LR): val 119.42 (+16% vs L1).
     **L1's bounded-derivative property already absorbs the bs=8
     noise-reduction effect** — the two round-3 winners do not compose.
     bs=8 + L1 lever is closed; bs=12 is out of reach without throughput
     infra (linear VRAM extrapolation gives ~126 GB > 96 GB cap).
   - PR #285 — edward (surf_weight=30 with MSE): val 125.53 (+22% vs
     L1 baseline). Within-condition seed spread (~13%) dwarfs directional
     effect (~3%); third PR in a row where seed noise > predicted effect.
     Independently rediscovered the `0*NaN=NaN` scoring bug — three-student
     convergence validates the merged fix.
   - PR #395 — frieren (L1 + wd=1e-3): **val 100.99 (−1.6% vs L1)
     validated on assigned baseline; +9.9% vs current L1+FF baseline**.
     Per-split signal confirms the regularisation hypothesis exactly —
     `val_geom_camber_rc` −11.9%, `val_single_in_dist` +6.2%. Same
     merge-order story as PR #298. Re-assigned as L1+FF + wd compose test.
   - PR #437 — frieren (L1+FF + wd=1e-3 compose): **val 91.35 (≈ tied
     with current baseline 90.90)**. **Per-split signal is the most
     informative of round 3** — refutes the "convergent OOD-camber
     levers all compose additively" narrative. wd × FF overlap
     destructively on rc-camber (+11.8% regression), compose additively
     on cruise-camber (−11.5%) and sign-flip on in-dist (−7.5% vs
     +6.2% under L1-only). See "Convergent OOD-camber narrative" section.
   - PR #448 — tanjiro (L1+FF + L1 volume loss): val 87.11
     (**−5.18% vs L1+FF baseline**, +8.5% vs current). Largest
     single-knob lever validation since PR #280; pre-registered branches
     fired cleanly (uniform improvement across 3 of 4 splits, biggest
     gain on val_single_in_dist −9.44% — exactly heavy-tail-dominated).
     Re-assigned compose test on post-#461 advisor.
   - PR #302 — tanjiro (Huber surface δ=1.0 on MSE): val 105.53 (+2.8%
     vs L1, +14.9% vs L1+FF). Wins narrowly on raceCar tandem
     (`val_geom_camber_rc` −8.9%) but loses on cruise (+21%) and re_rand
     (+7%). δ=1.0 too generous; lever range bracketed (δ→0 = L1).
   - PR #419 — thorfinn (L1 + AdamW beta2=0.95): val 99.70 (−2.87% vs
     L1, in predicted band) but +8.5% vs L1+FF baseline. **Same
     OOD-camber improvement signature** as PR #395 and PR #400 —
     dominant win on `val_geom_camber_rc` (−13.6%). Three different
     mechanisms (regularisation / input encoding / optimiser
     second-moment), same direction. Compose test will reveal
     additivity. Re-assigned.
   - PR #396 — fern (L1 + EMA decay=0.9999): canonical value broken
     (val 317.92, averaging window ≫ optimizer step count). Follow-up
     `EMA_DECAY=0.999` got val 92.00 (essentially tied with current
     L1+FF baseline). Student derived **budget-aware EMA rule**
     `EMA_DECAY = 1 − 1/(0.2 × total_steps) ≈ 0.999` for this 5K-step
     budget. Re-assigned as L1+FF + EMA(0.999) compose test.

**In-flight (1 on pre-L1 + 7 on L1 baseline) — still useful for round-4
composition even if they don't outright beat 102.64:**

1. **Loss formulation (pre-L1 baseline)**
   - PR #302 — tanjiro: Huber (smooth-L1, δ=1.0) surface loss — informs
     whether the L2-near-zero region helps over pure L1.
2. **L1+FF-baseline composition** (post-#400 merge — `train.py` has both
   L1 and 8-freq spatial Fourier features). The 6 in-flight PRs below
   were branched off the L1-only advisor (post-#280, pre-#400), so their
   results test their lever on the L1 baseline (val 102.64). To compose
   with the new PR #400 baseline (val 91.87), winners will need to be
   re-tested on the new advisor:
   - PR #383 — alphonse: L1 + 3× pressure channel weight in surface loss
     *(loss focus)* — branched off L1-only.
   - PR #558 — frieren: L1+FF12+EMA + `--epochs 14` + `lr=7.5e-4` +
     **slice_num=128** — retest of inconclusive PR #292 on cleaner stack.
   - PR (fern, new): L1+FF12+EMA(0.998) + `--epochs 14` + `lr=7.5e-4`
     — bracket EMA decay slightly upward from 0.997 (window ~500 steps
     vs 333) toward the cosine tail.
   - PR (askeladd, new): L1+FF12+EMA + `--epochs 14` + `lr=7e-4` —
     LR bracket DOWN on EMA-stack (optimum at-or-below 7.5e-4 per
     LR-curve triangulation).
   - PR (nezuko, new): L1+FF12+EMA + `--epochs 14` + `lr=7.5e-4` +
     **input-space Gaussian noise** (sigma=0.05) — different
     regularisation axis than FF; tests input-perturbation robustness
     as a mechanistically-novel lever for round 5.
   - PR (tanjiro, new): L1+FF+EMA + `--epochs 14` + `lr=7.5e-4` +
     **auxiliary log-pressure loss** — different mechanism than
     channel weighting / loss shape; addresses heavy-tail via
     target-space rescaling. Mechanistically distinct from EMA's
     trajectory averaging.
   - PR #516 — askeladd: L1+FF+EMA + `--epochs 14` + **`lr=8e-4`** —
     interior LR bracket point.
   - PR #524 — edward: **canonical 6-lever stack measurement** —
     `--epochs 14 --lr 7.5e-4` on post-#462 advisor.
   - PR (thorfinn, new): L1+FF+EMA + `--epochs 14` + `lr=7.5e-4` +
     **DropPath 0.05** — half the previous rate, no wallclock cliff.
   - PR (frieren, new): L1+FF+EMA + `--epochs 14` + `lr=7.5e-4` +
     **NUM_FOURIER_FREQS=4** — FF dose bracket downward, complement
     to nezuko's #506.
   - PR (fern, new): L1+FF+EMA(decay=**0.997**) + `--epochs 14` +
     `lr=7.5e-4` — shorter EMA window to address the schedule × EMA
     interference identified in PR #476.

## Compose pattern map — final round-3 picture

Round-3 PRs tested compose effects on top of L1+FF and the
post-EMA stack. The pattern:

| compose pattern | with FF/EMA | examples | cumulative impact |
|----------------|---------|----------|------------------:|
| **Distributional** (broad gain across all splits) | additive | matched cosine + lr=7.5e-4 (#461), grad clipping (#462) | merged |
| **Trajectory averaging** | clean orthogonal | EMA × FF (#447) | merged |
| **L1-only-OOD-camber-targeted** at small dose | additive | wd=5e-4 (#469, lost by merge timing) | re-assigned |
| **L1-only-OOD-camber-targeted** at large dose | destructive on rc-camber | wd=1e-3 (#437), beta2=0.95 (#446) | closed |
| **Input encoding on already-rich features** | net-flat or regression | log(Re) FF (#432) | closed |
| **Loss-shape regulariser** | overlaps with EMA | L1-volume × EMA (#492) | closed |
| **LR overshoot on EMA stack** | regression | lr=1e-3 × EMA (#489) | closed |
| **Direction-only-update regime cliff** | under-convergence | max_norm=0.5 × full stack (#499), DropPath 0.1 wallclock cliff (#501) | closed |
| **Schedule × averaging interference** | OOD regression | matched cosine × EMA (#476) | closed |
| **Saturated regularisation overlap** | no marginal value | wd=5e-4 × full stack (#500) | closed |
| **Heavy-tail compose redundancy with EMA** | mild uniform regression | 3× p-weight in vol_loss × full stack (#515) | closed |

**Generalisation observed across compose tests**: once one "noise/
regularisation" lever is in the stack (FF, EMA), additional
same-mechanism levers tend to interfere on the most-improved split.
The pattern reproduces across magnitude-based (#437, #446), loss-shape
(#492), LR-overshoot (#489), and direction-only-clipping (#499)
failure modes.

## Round-3-best 6-lever stack — pending canonical measurement

Every round-3 merge has been on a slightly different baseline:
- PR #389 (val 90.90): L1 + matched cosine — **no FF**
- PR #447 (val 82.97): L1 + FF + EMA — **no matched cosine** (T_max=50)
- PR #461 (val 80.28): L1+FF + matched cosine + lr=7.5e-4 — **no EMA**
- PR #462 (val 80.06): L1+FF + matched cosine + clip=1.0 — **no EMA, lr=5e-4**

The advisor `train.py` now bakes in L1 + FF + EMA + clipping. The
recommended reproduce command is `--epochs 14 --lr 7.5e-4`. **The
canonical six-lever-stack measurement on the post-#462 advisor is
the missing reference number** — predicted ~76-78 if proven levers
compose cleanly.

Edward (post-#499 close) is assigned this canonical measurement run.

**Input encoding compose insight (from PR #432 close)**: input-encoding
levers compose with FF only when the targeted input dimension was
previously *poorly exposed* to the model. Spatial `(x, z)` was 2-d and
only used for slice routing → spatial FF won. Log(Re) is one of 22
already-rich input features → log(Re) FF lost. Round-5 input-encoding
work on gap/stagger/AoA scalar inputs should expect similar negatives;
focus instead on truly-novel input axes.

**Round-5 assignment heuristic** (now informed by 6 compose tests):
- Prefer **distributional / trajectory-averaging** levers — these
  compose with FF.
- Prefer **mechanistically-different regulariser axes** (DropPath,
  stochastic depth, dropout, MixUp) over weight-magnitude
  regularisation when the latter has overlapped with FF.
- Magnitude-based regularisers (wd, beta2) only compose with FF at
  small doses; large doses interfere on rc-camber.
- Per-split signal is load-bearing for compose decisions; headline
  tied-or-marginally-better can hide important compose information.

## Round-5 priorities (refreshed by PR #437)

1. **Interior-point wd sweep** (3e-4, 5e-4, 7.5e-4) on L1+FF + matched
   cosine. Find the wd that captures cruise/in-dist compose without
   rc-camber regression. **Round-5 priority #1.**
2. **DropPath / stochastic depth** — different regularisation
   dimension; may help rc-camber where wd's weight-magnitude axis
   competes with FF.
3. **FF frequency count variation** — tests whether rc-vs-cruise
   asymmetry is about geometry-interpolation regime (boundary-shoulder
   rc M=6-8 vs centre-band cruise M=2-4).
4. **Auxiliary log-pressure target transform** — the persistent
   `val_single_in_dist` bottleneck (still ~106 even on best baseline)
   wasn't dented by any round-3 lever; needs a different attack.

## Harness debt

PR #437 also surfaced concurrent-run interference: the entrypoint
launched a parallel `train.py` while the student's run was in test
eval, causing the parallel run to OOM/crash. Empty stale dir
`models/model-l1ff_wd_1e-3-20260428-021302/`. Entrypoint should
serialise per-process. Round-5 harness cleanup.

## Round-4 throughput infra (new debt from PR #390 close)

PR #390 confirmed `bs=12` is out of reach on this hardware: linear VRAM
extrapolation `bs=4 → 42 GB`, `bs=8 → 84 GB` puts `bs=12 → ~126 GB > 96 GB`.
Any future "bigger batch" or "bigger model" lever requires activation
checkpointing or BF16 first. Round 4 should treat this as a deliberate
infra PR if other levers stall — likely BF16 autocast (1.5-2× speedup) is
the cleanest first move; activation checkpointing is the second tier.

## Round-4 seed-pinning infra (new debt from PR #285 close)

Three round-3 PRs (PR #292 frieren slice_num=128, PR #288 fern warmup+1e-3,
PR #285 edward surf_weight=30) had cross-seed variance comparable to or
larger than their predicted effect size: 4-13% noise floor vs 2-8% effect.
Only the L1 surface loss change cleared this floor decisively (24% vs ~5%
noise). Round-4 priority: add `torch.manual_seed(...)` + numpy/python seed
pinning at the top of `train.py`, parameterised so each PR can record a
seed in its experiment metadata. Without this, all "moderate-effect"
hypotheses are uninterpretable at single replicates.

## Data forensics flag

`test_geom_camber_cruise` sample 020 has 761 non-finite p values — likely
a CFD divergence (turbulent reattachment, mesh degeneracy, etc.). Worth
flagging to whoever owns dataset preprocessing; not actionable from the
advisor branch.

**Caveats:**

- All 6 originally-in-flight PRs (and PR #366 thorfinn-followup) branched
  off the pre-L1 advisor. They each test their lever against MSE+`bs=4`
  defaults, not against L1. To compose with L1 properly, round-4 winners
  will be re-tested on the L1 baseline.
- All 6 originally-in-flight PRs also branched off the pre-fix advisor,
  so their on-disk `test_avg/*_p` will land NaN. Recompute test from the
  saved checkpoint with the patched scorer at review time.

## Potential next research directions

Following round 3, plausible next steps depending on which family wins:

- **If loss reformulation wins:** sweep weighting (5, 20, 40), per-channel
  pressure boost, switch volume loss to L1 too, MAE-targeted asymmetric loss.
- **If capacity wins:** push to `n_hidden=256` / `n_layers=8`, vary
  `slice_num` jointly with `n_hidden`, try more heads (8, 12), `mlp_ratio=8`.
- **If schedule wins:** OneCycleLR, longer warmup, larger T_max with lower
  floor LR, AdamW betas `(0.9, 0.95)` for transformer-style training.
- **If Fourier features win:** test learnable frequencies, RFF, NeRF-style
  multi-resolution, separate spatial vs feature encodings.
- **Cross-cutting:** loss + capacity composition, warmup + L1 stack, EMA of
  weights for evaluation, Sharpness-Aware Minimization (SAM).
- **Cosine-truncation problem:** the round-3 baseline only ran 14/50 epochs.
  Round 4 should test (a) `--epochs 14` so cosine fully decays, (b) BF16
  autocast and/or `torch.compile` for 1.5-2× speedup so we can fit more
  epochs of the existing schedule, (c) `bs=12` with `√3` LR scaling.
- **Architecture revisits if all r3 levers stall:** Transolver replacement
  with point-cloud transformer variants, GINO/FNO style spectral mixing in
  irregular meshes, hierarchical clustering of slice tokens.

## Per-track diagnostics on the current baseline (PR #461)

| split | val mae_surf_p | comment |
|-------|---------------:|---------|
| `val_geom_camber_cruise` |  62.42 | easiest |
| `val_re_rand`            |  78.92 | mid |
| `val_single_in_dist`     |  89.76 | high-Re raceCar singles (was the dominant bottleneck) |
| `val_geom_camber_rc`     | **90.03** | unseen front-foil camber, now slightly worst |

The persistent `val_single_in_dist` bottleneck is now **closing fast**:
117.24 (post-#400) → 99.44 (post-#447) → 89.76 (post-#461). EMA and
the LR bump both hit it. The new worst split is `val_geom_camber_rc`
at 90.03 — by a tiny margin (89.76 vs 90.03).

Round-5 candidates targeting the now-marginally-worst splits:
- **`log(Re)` Fourier features** (PR #432, in flight).
- Log-space pressure prediction (target transform).
- Per-domain sample reweighting (boost raceCar single).
- LR bracket upward at `lr=1e-3` (askeladd's next assignment) — this
  PR's data suggests the LR optimum is past 7.5e-4.

## Constraints (do not override)

- 30-minute wallclock per training run (`SENPAI_TIMEOUT_MINUTES`).
- 50-epoch cap (`--epochs 50` is the default; do not raise unless explicitly
  authorised).
- 96 GB GPU per student — VRAM headroom is large, lean on it for capacity.
- Local JSONL metrics only. No W&B / Weave / Wandb.
- One hypothesis per PR. No bundled multi-knob changes in round 3.
