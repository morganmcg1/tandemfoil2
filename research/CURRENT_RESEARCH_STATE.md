# SENPAI Research State

- **Date:** 2026-04-27
- **Track:** `icml-appendix-willow-pai2c-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-r5`
- **Most recent direction from human researcher team:** None (no open issues on this track)

## Current baseline

**PR #227 (Huber surface loss, surf_weight=10):** `val_avg/mae_surf_p = 112.1574`

| Split | `val_*/mae_surf_p` |
|---|---|
| `val_single_in_dist`     | 147.6473 |
| `val_geom_camber_rc`     | 112.0473 |
| `val_geom_camber_cruise` |  89.5626 |
| `val_re_rand`            |  99.3723 |
| **avg** | **112.1574** |

W&B run: `6bylngu8`

## Key constraints / learnings

- **30-min timeout limits training to ~13–14 epochs** (fp32). Schedule T_max must align to ~13–14 achievable epochs, not configured 50.
- **Batch size does not help throughput** — Transolver attention is mesh-size-dominated (N=74K–242K), not batch-dominated. bs=8 was same speed as bs=4. Closed dead end (PR #228).
- **NaN/Inf poisoning of cruise test split** is a recurring issue — one sample produces Inf pressure prediction, poisoning `test_geom_camber_cruise/mae_surf_p`. Mitigation: add `nan_to_num` guard in `evaluate_split`.
- **surf_weight=25 with MSE is worse** than surf_weight=10 with Huber. OOD generalization needs volume coherence; over-weighting surface harms volume scaffold. Optimal surf_weight likely 10–20.
- **Warmup with 5-epoch budget** was harmful (14 epochs total = 36% warmup = too much). Shorter warmup (2 ep) or no warmup preferred.

## Active PRs (WIP)

| PR | Student | Hypothesis | Axis |
|----|---------|------------|------|
| #263 | tanjiro | bf16 autocast throughput unlock | throughput → more epochs |
| #260 | askeladd | `n_hidden` 128 → 256 (wider model) | capacity (width) |
| #259 | nezuko | Pure L1 on surface (β→∞ limit of Huber) | loss form sweep |
| #229 | thorfinn | `n_layers` 5 → 7 (deeper model) | capacity (depth) |
| #224 | fern | 2-epoch warmup + aligned cosine (v2) | LR schedule (corrected) |
| #223 | edward | `slice_num` 64 → 128 | inductive bias (physics tokens) |
| #265 | frieren | Huber + `surf_weight` 10 → 15 (clean isolation) | loss weighting |
| #266 | alphonse | `lr` 5e-4 → 1e-3 with cosine (faster convergence under timeout) | learning rate |

## Recently closed PRs

- **PR #225** (frieren, surf_weight=25, MSE): CLOSED — worse than baseline (+9.8%). Confounded by loss-form mismatch (MSE vs Huber). Follow-up: PR #265.
- **PR #184** (alphonse, baseline anchor, MSE default): CLOSED — established MSE reference at 134.09, confirms Huber (PR #227) wins by 16.4%.
- **PR #228** (tanjiro, bs=8 sqrt-LR): CLOSED — 33% worse; bs scaling doesn't help throughput on Transolver.

## Highest-priority hypotheses to test next (after current wave)

1. **Compound winners**: If width (askeladd #260) and/or depth (thorfinn #229) beat baseline, combine with Huber surface loss and optimal surf_weight.
2. **bf16 + architecture growth**: If tanjiro's #263 unlocks more epochs via bf16, test wider/deeper model that OOMs at fp32 batch_size=4.
3. **Per-channel pressure weighting**: Add separate weight for the `p` channel within surface loss (p is the primary metric; Ux/Uy are not ranked). Requires train.py code edit.
4. **EMA weight averaging (decay=0.999)**: Addresses observed last-epoch bounce-back (alphonse: 134.09 @ ep13 → 174.38 @ ep14). EMA should smooth prediction stability and improve best-checkpoint quality.
5. **surf_weight=20**: Follow-up if sw=15 (frieren #265) beats baseline, test sw=20 to find the sweet spot.
6. **Learning rate scaling**: If alphonse's #266 (lr=1e-3) beats baseline, test lr=7e-4 as a moderate midpoint and lr=1.5e-3 as an aggressive upper.
7. **Symmetry data augmentation (vertical flip)**: Physics-respecting Uy sign-flip augmentation roughly doubles training data. Zero-cost throughput gain.
8. **Unified positional encoding** (`unified_pos=True, ref=8`): Transolver built-in feature currently disabled; worth a single test after simpler levers exhausted.

## Plateau watch

Not triggered — baseline improvement still being found (Huber PR #227 merged). Monitor after current wave of 8 PRs lands.
