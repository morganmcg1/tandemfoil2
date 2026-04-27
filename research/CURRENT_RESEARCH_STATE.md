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

Open the round with a **breadth-first sweep** across orthogonal levers.

**Measured baseline (merged 2026-04-27 23:26):**
PR #306 (thorfinn) — `bs=8, lr=7.07e-4`, `val_avg/mae_surf_p = 135.20`,
`test_avg/mae_surf_p = 123.15` (NaN-safe re-eval). Stopped at epoch 14/50
by 30-min timeout — cosine never decayed to its tail. Critical scoring fix
also landed on the advisor branch (commit `2eb5c7f`) to avoid Inf×0=NaN
poisoning of `test_avg/*_p`.

**In-flight (7) — single-knob changes, ordered by family:**

1. **Loss formulation that targets the reported MAE metric directly**
   - PR #280 — alphonse: L1 surface loss
   - PR #302 — tanjiro: Huber (smooth-L1, δ=1.0) surface loss
   - PR #285 — edward: surf_weight 10 → 30
2. **Model capacity in the directions the architecture is parameterised for**
   - PR #283 — askeladd: n_hidden=192, n_layers=6, n_head=6, slice_num=96
   - PR #292 — frieren: slice_num 64 → 128 (single-knob)
3. **Optimisation schedule**
   - PR #288 — fern: 3-epoch warmup + cosine to 1e-5, peak lr=1e-3
4. **Input representation**
   - PR #298 — nezuko: 8-frequency Fourier positional encoding for x/z

**New assignment (thorfinn, post-merge):** mlp_ratio 2 → 4 (single-knob
capacity bump on a different axis from askeladd / frieren).

In-flight PRs branched off the pre-fix advisor, so their on-disk
`test_avg/*_p` will land NaN. Recompute test from the saved checkpoint
with the patched scorer — or rebase if the student is happy to absorb the
extra wallclock.

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
