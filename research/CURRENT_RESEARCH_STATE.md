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

**Current measured baseline (merged 2026-04-28 02:28):**
PR #389 (askeladd) — **L1 + matched cosine schedule** (`--epochs 14`,
T_max=14). `val_avg/mae_surf_p = 90.90`, `test_avg/mae_surf_p = 80.84`.
Three-seed reproducibility ~1% spread. **First round-3 result with
effect size large enough to clearly clear the seed-noise floor.**

**Important caveat**: PR #389 was branched off the pre-FF advisor, so
the measured 90.90 reflects L1+matched-cosine *without* FF. The
advisor `train.py` retains FF from PR #400 — running the post-merge
advisor with `--epochs 14` is **untested** but expected to land below
90.90 if FF and matched cosine compose. The 90.90/80.84 numbers should
be treated as approximately tied with PR #400's 91.87/81.11 (both
configurations are good, ~1% gap is below seed noise).

**Round-3 baseline lineage:**
| Round | best val | best test | lever | Δ vs prior |
|-------|---------:|----------:|-------|----:|
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| PR #400 |  91.87 |  81.11 | + 8-freq spatial Fourier features | **−10.5% / −17.0%** |
| **PR #389** | **90.90** | **80.84** | **+ matched cosine `--epochs 14`** | **−1.06% / −0.33%** (≈ tied with FF) |

**Round-3 proven levers (cumulative)**:
1. L1 surface loss (PR #280) — loss formulation aligned with metric.
2. 8-freq spatial Fourier features (PR #400) — spectral bias mitigation.
3. Matched cosine `--epochs 14` (PR #389) — let LR fully decay.

**Round-4 priority**: confirm the L1+FF + matched-cosine compose
result. Round-5 candidate stacking depends on which compose tests
return wins.

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
   - PR #432 — nezuko: L1+FF + **`log(Re)` Fourier features** *(input
     encoding extension)* — on post-#400 advisor.
   - PR #437 — frieren: L1+FF + `weight_decay 1e-3` *(regularisation
     compose)* — on post-#400 advisor.
   - PR #446 — thorfinn: L1+FF + AdamW(beta2=0.95) *(optimiser compose)*
     — on post-#400 advisor.
   - PR #447 — fern: L1+FF + EMA(decay=0.999) *(weight averaging compose
     with budget-aware decay)* — on post-#400 advisor.
   - PR #448 — tanjiro: L1+FF + L1 volume loss *(loss formulation
     extension — does L1 dominance extend to volume?)* — on post-#400
     advisor.
   - PR (askeladd, new): L1+FF + `--epochs 14` + `lr=7.5e-4` —
     student-suggested follow-up: now that cosine actually anneals,
     `lr=5e-4` is conservatively low; modest LR bump tests headroom on
     the matched schedule *(schedule × lr)*.
   - PR (edward, new): L1+FF + `--epochs 14` + grad clipping `max_norm=1.0`
     — three-lever stack (FF + matched cosine + clipping) tests whether
     the "convergent OOD-camber" levers compose additively *(stability ×
     schedule × FF compose)*.

## Convergent OOD-camber generalisation signal — now 5 levers

Five round-3 PRs hit the **same per-split signature**: dominant win on
`val_geom_camber_rc` (the unseen-front-foil-camber raceCar tandem track),
flat-or-mild on the other splits.

| PR | lever | `val_geom_camber_rc` Δ |
|----|-------|-----------------------:|
| #400 (merged) | spatial Fourier features | −20.8% |
| #389 (merged) | matched cosine | **−19.4%** |
| #423 (closed) | gradient clipping | −15.0% |
| #419 (closed) | AdamW(beta2=0.95) | −13.6% |
| #395 (closed) | weight_decay=1e-3 | −11.9% |

Five independent mechanisms — input encoding, schedule, optimiser
second-moment, regularisation, stability — same direction of effect.
Whatever's bottlenecking `val_geom_camber_rc` is responsive to "make
optimisation more effective in the limited budget we have" rather than
to any one specific intervention.

Round-4 compose tests (#437 wd, #432 log(Re), #446 beta2, #447 EMA,
edward grad clipping) will reveal whether the levers each hit
independent paths to the same generalisation gain (additive — round-5
stack of all five) or share a common dynamic (diminishing — only one
or two of the five matter once combined). Two of the five (FF and
matched cosine) are now baked in as baseline, so the remaining
compose tests measure marginal gain on top of those.

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

## Per-track diagnostics on the current baseline (PR #400)

| split | val mae_surf_p | comment |
|-------|---------------:|---------|
| `val_geom_camber_cruise` |  68.61 | easiest — low-Re cruise meshes |
| `val_re_rand`            |  82.64 | mid — stratified Re, all tandem domains |
| `val_geom_camber_rc`     |  98.99 | unseen front-foil camber, raceCar |
| `val_single_in_dist`     | **117.24** | hardest — high-Re raceCar singles |

`val_single_in_dist` is still the dominant bottleneck even after L1+FF —
the gap to cruise (1.71×) actually *widened* from the L1 baseline (1.65×).
High-Re raceCar singles are the persistent failure mode.

Round-4/5 candidates that target this regime specifically:
- **`log(Re)` Fourier features** ← assigning to nezuko this round.
- Log-space pressure prediction (target transform).
- Re-aware output rescaling (per-sample scalar gain).
- Per-domain sample reweighting in the WeightedRandomSampler (boost
  raceCar single).

## Constraints (do not override)

- 30-minute wallclock per training run (`SENPAI_TIMEOUT_MINUTES`).
- 50-epoch cap (`--epochs 50` is the default; do not raise unless explicitly
  authorised).
- 96 GB GPU per student — VRAM headroom is large, lean on it for capacity.
- Local JSONL metrics only. No W&B / Weave / Wandb.
- One hypothesis per PR. No bundled multi-knob changes in round 3.
