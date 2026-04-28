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

**Current measured baseline (merged 2026-04-28 01:30):**
PR #400 (nezuko) — **L1 surface loss + 8-frequency Fourier positional
features for `(x, z)`**. `val_avg/mae_surf_p = 91.87`,
`test_avg/mae_surf_p = 81.11`. Wins on **all four** val splits and all
four test splits vs the prior L1 baseline. Test gain (−17%) larger than
val gain (−10.5%) is real generalisation evidence.

**Round-3 baseline lineage:**
| Round | best val | best test | lever | Δ vs prior |
|-------|---------:|----------:|-------|----:|
| PR #306 | 135.20 | 123.15 | bs=8 + sqrt-LR (MSE) | reference |
| PR #280 | 102.64 |  97.73 | + L1 surface loss | **−24.1%** |
| **PR #400** | **91.87** | **81.11** | **+ 8-freq spatial Fourier features** | **−10.5% val, −17.0% test** |

**Round-3 dominant lever (1 of 2 stacked):** loss formulation (L1) +
input-encoding spectral compensation (Fourier features). Two
independent levers, attacking different failure modes (heavy-tailed
gradients vs spectral bias on input position).

**Current bottleneck**: `val_single_in_dist` at 117.24 (vs 68.61
cruise, 82.64 re_rand, 98.99 rc camber). High-Re raceCar singles —
the FF lever helped least there. Round-5 priorities should target this
regime specifically.

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
     *(loss focus)*
   - PR #389 — askeladd: L1 + matched cosine schedule (`--epochs 14`) so
     the schedule fully decays inside the 30-min cap *(schedule)*
   - PR #395 — frieren: L1 + `weight_decay 1e-4 → 1e-3` *(regularisation)*
   - PR #396 — fern: L1 + EMA of weights for evaluation *(weight averaging)*
   - PR #419 — thorfinn: L1 + AdamW(beta2=0.95) — modern transformer
     optimiser config for noisier gradients *(optimiser)*
   - PR #423 — edward: L1 + gradient clipping `max_norm=1.0` *(stability)*
   - PR (nezuko, new): L1 + spatial FF + **`log(Re)` Fourier features**
     — extends the proven FF lever to the scalar log-Re input, targets
     the cross-regime axis where `val_re_rand` improved less than camber
     splits *(input encoding extension)*

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
