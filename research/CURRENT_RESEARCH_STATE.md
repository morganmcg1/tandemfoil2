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

**Current measured baseline (merged 2026-04-28 00:03):**
PR #280 (alphonse) — **L1 surface loss**, `bs=4`, `lr=5e-4`, all other
defaults. `val_avg/mae_surf_p = 102.64`, `test_avg/mae_surf_p = 97.73`
(NaN-safe re-eval). Wins on **all four** val splits vs the prior baseline,
including −36% on the hardest split (`val_single_in_dist`).

**Previous baseline (round 3 reference 1):**
PR #306 (thorfinn) — `bs=8, lr=7.07e-4` MSE, val 135.20 / test 123.15.
Established the round-3 starting reference. The L1 win (−24% val) is much
larger than the bs/lr win it superseded.

**Round-3 dominant lever:** loss formulation. Aligning the gradient with
the reported MAE metric is the highest-impact change found so far.

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

**In-flight (2 on pre-L1 + 6 on L1 baseline) — still useful for round-4
composition even if they don't outright beat 102.64:**

1. **Loss formulation (pre-L1 baseline)**
   - PR #302 — tanjiro: Huber (smooth-L1, δ=1.0) surface loss — informs
     whether the L2-near-zero region helps over pure L1.
   - PR #285 — edward: `surf_weight` 10 → 30 (with MSE) — informs how much
     of the L1 win is just heavier surface gradient signal.
2. **L1-baseline composition (post-#280 merge, with L1 already in
   `train.py`):**
   - PR #383 — alphonse: L1 + 3× pressure channel weight in surface loss
     *(loss focus)*
   - PR #389 — askeladd: L1 + matched cosine schedule (`--epochs 14`) so
     the schedule fully decays inside the 30-min cap *(schedule)*
   - PR #390 — thorfinn: L1 + bs=8 + `lr=7.07e-4` — composes the two
     merged round-3 winners *(optimization)*
   - PR #395 — frieren: L1 + `weight_decay 1e-4 → 1e-3` *(regularisation)*
   - PR #396 — fern: L1 + EMA of weights for evaluation *(weight averaging)*
   - PR (nezuko, new): L1 + 8-freq Fourier positional features for `(x, z)`
     *(input encoding compose test)*

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

## Per-track diagnostics on the current baseline

The round-3 baseline shows strongly uneven per-split surface MAE:

| split | val mae_surf_p | comment |
|-------|---------------:|---------|
| `val_geom_camber_cruise` |  97.95 | easiest — low-Re cruise meshes |
| `val_re_rand`            | 114.32 | mid — stratified Re, all tandem domains |
| `val_geom_camber_rc`     | 138.39 | unseen front-foil camber, raceCar |
| `val_single_in_dist`     | **190.14** | hardest — high-Re raceCar singles |

The high-Re raceCar singles are the dominant error. Round-4 candidates
that target this specifically: log-space pressure prediction, Re-aware
output rescaling, an explicit log-Re embedding beyond the raw `log(Re)`
feature.

## Constraints (do not override)

- 30-minute wallclock per training run (`SENPAI_TIMEOUT_MINUTES`).
- 50-epoch cap (`--epochs 50` is the default; do not raise unless explicitly
  authorised).
- 96 GB GPU per student — VRAM headroom is large, lean on it for capacity.
- Local JSONL metrics only. No W&B / Weave / Wandb.
- One hypothesis per PR. No bundled multi-knob changes in round 3.
