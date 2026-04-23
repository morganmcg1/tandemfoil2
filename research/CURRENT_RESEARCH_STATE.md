# SENPAI Research State

- **Updated:** 2026-04-23 21:10
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Student Status

| Student | Status | Current PR |
|---------|--------|-----------|
| frieren  | WIP | #3: Loss reformulation — Huber/L1 sweep |
| fern     | WIP | #4: Capacity scaling — width/depth/slice-num |
| tanjiro  | WIP | #5: Channel-weighted loss — upweight surf_p |
| nezuko   | WIP | #6: LR + schedule — warmup/floor/OneCycle |
| alphonse | WIP | #7: Fourier PE + FiLM(log Re) |
| edward   | WIP | #8: EMA of weights + gradient clipping |
| thorfinn | WIP | #9: Pressure target reparameterization — asinh/robust/per-domain |

**Idle students:** none.

---

## PRs Ready for Review

None.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Branch |
|----|---------|-----------|--------|
| #3 | frieren  | Huber/L1 vs MSE — close MSE/MAE mismatch on heavy-tailed p  | `frieren/loss-reformulation-v2` |
| #4 | fern     | Scale n_hidden, n_layers, slice_num on 96 GB GPUs            | `fern/capacity-scaling-v2` |
| #5 | tanjiro  | Channel-region re-weighting; upweight surf_p aggressively    | `tanjiro/channel-weighted-loss` |
| #6 | nezuko   | Peak LR + warmup + LR floor + OneCycle sweep                 | `nezuko/lr-schedule-sweep` |
| #7 | alphonse | Random Fourier PE on (x,z) + FiLM conditioning on log(Re)    | `alphonse/fourier-pe-film-re` |
| #8 | edward   | EMA of weights (0.999/0.9999) × grad-clip (0/0.5/1.0) grid    | `edward/ema-gradclip-stability` |
| #9 | thorfinn | asinh, robust, per-domain target normalization for p channel  | `thorfinn/pressure-target-reparam` |

Each PR is an 8-GPU sweep in a single `--wandb_group`, one run per GPU, the first GPU of each sweep running the vanilla Transolver baseline for anchoring.

---

## Baseline

**No baseline yet.** The GPU-0 run in every round-1 PR establishes anchor `val_avg/mae_surf_p` on this track. Eight independent baseline runs (one per student) will also give us a natural measure of seed variance — useful for setting a "significantly better" threshold on round-2 reviews.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 21:10.

---

## Current Research Focus and Themes

This is round 1 on a fresh advisor branch. Seven hypothesis families span the seven orthogonal "obvious first" levers:

1. **Loss shape** (frieren) — match the metric, tame the heavy tails.
2. **Model capacity** (fern) — the 0.7 M-param baseline is almost certainly under-parameterized for 96 GB GPUs.
3. **Task-aligned loss weights** (tanjiro) — the metric only ranks surface-p, so weight it.
4. **Optimizer schedule** (nezuko) — LR + warmup + LR floor + OneCycle.
5. **Conditioning & positional features** (alphonse) — Fourier PE on (x,z) + FiLM on log(Re) to address spatial sharpness and per-Re scale drift.
6. **Training stability** (edward) — EMA + grad-clip. Cheap, stackable, rarely hurts.
7. **Target reparameterization** (thorfinn) — asinh / robust / per-domain for the heavy-tailed p channel.

Every round-1 PR is expected to produce a within-PR baseline vs. variants comparison; the best variant from each PR becomes a candidate for merging. The seven PRs together should also let us rank *categories* of improvement — useful for picking round-2 priorities.

---

## Potential Next Research Directions and Themes

Round-2 candidates from `research/RESEARCH_IDEAS_2026-04-23_21:00.md` that would stack cleanly on round-1 winners:

### Physics-informed priors
- **Panel-method residual learning** — predict viscous correction on top of an analytical potential-flow baseline. Highest expected OOD lift but substantial complexity.
- **Divergence-free regularization** on volume nodes via finite-difference.
- **Kutta condition** at the trailing edge.

### Data / augmentation
- **Horizontal-flip augmentation** with Uy sign-flip — physics-exact for both raceCar (ground effect) and cruise. Free doubling of effective train set; expected medium-large lift.
- **Low-Re → high-Re curriculum** — start training on easier samples, anneal to harder.
- **Synthetic single-foil pretraining** from panel method.

### Architecture
- **Cross-attention decoder head** specialised for surface nodes while a shared trunk does the heavy lifting.
- **Near-surface volume band** three-tier weighting — boundary layer resolution directly drives surface pressure fidelity.
- **Neural-operator variants** (FNO, GINO, Perceiver IO).

### Training signal
- **Spectral loss** (Fourier components of the pressure field on the surface) to complement pointwise MAE.
- **Distillation** from an ensemble of round-1 winners.

### Compounding plan for round 2

Best winner from {frieren, tanjiro, thorfinn} (loss/target family) merges first to get the loss landscape right. Then compound with best winner from {fern, alphonse} (capacity/conditioning). Then re-run round-1 stability tweaks (edward, nezuko) against the new baseline — these tend to become *more* effective against a better-tuned model. At that point, introduce round-2 physics/augmentation ideas on top.
