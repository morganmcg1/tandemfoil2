# SENPAI Research State

- **Updated:** 2026-04-23 22:20
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 103.036** (PR #3, `frieren/loss-l1-sw10`, W&B run `w2jsabii`)
- Per-split val: in_dist=133.19 | camber_rc=117.21 | camber_cruise=70.11 | re_rand=91.63
- Config: L1 loss, surf_weight=10, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4

**Scoring bug (GH #10) FIXED** — cherry-picked from thorfinn's PR #9 (commit 7d71abd). `test_avg/mae_surf_p` now produces finite numbers across the track.

**Leading unmerged candidate:** thorfinn's asinh-s458 at **100.034 val / 90.26 test** (on MSE) — to be validated on L1 after rebase.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP | #11: L1 + fine surf_weight sweep | `frieren/l1-surf-weight-sweep` |
| fern     | WIP | #12: Throughput — AMP + grad accumulation | `fern/throughput-amp` |
| tanjiro  | WIP (r2) | #5: Channel-weighting on top of L1 | `tanjiro/channel-weighted-loss` |
| nezuko   | WIP (r2) | #6: 3-seed LR replay + WSD on L1 | `nezuko/lr-schedule-sweep` |
| alphonse | WIP | #7: Fourier PE + FiLM on log(Re) | `alphonse/fourier-pe-film-re` |
| edward   | WIP (r2) | #8: EMA + grad-clip (wider threshold sweep) on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r2) | #9: asinh-on-L1 compound + 3-seed replay | `thorfinn/pressure-target-reparam` |

**Idle students:** none. Zero idle GPUs.

---

## PRs Ready for Review

None.

---

## PRs In Progress (`status:wip`)

### Round 2 (rebased on L1 baseline + scoring fix)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #11 | frieren  | Fine surf_weight sweep under L1 (sw ∈ {1,2,3,5,10,15,20,30}) | Beat 103.036 |
| #12 | fern     | AMP (bf16) + grad accumulation to unlock 25–35 epochs vs current 14 | Beat 103.036 |
| #5  | tanjiro  | Channel-weight fine sweep (psurf ∈ {14,17,20,23,27}) + vol_weight on L1 | Beat 103.036 |
| #6  | nezuko   | 3-seed replay of LR floor + WSD scheduler on L1 | Beat 103.036 |
| #8  | edward   | EMA + wider grad-clip sweep ({1,5,10,50}) on L1 | Beat 103.036 |
| #9  | thorfinn | asinh-on-L1 + 3-seed replay around s=458 | Beat 103.036 |

### Round 1 still in flight

| PR | Student | Hypothesis |
|----|---------|-----------|
| #7  | alphonse | Fourier PE on (x,z) + FiLM conditioning on log(Re) |

PR #7 started before L1 merged; verdict will be evaluated vs 103.036 when complete.

---

## Recent Decisions Log

- **2026-04-23 21:40** Merged PR #3 (frieren L1): baseline set to 103.036.
- **2026-04-23 21:40** Closed PR #4 (fern capacity): undertrained, not undercapacity — reassigned to throughput (PR #12).
- **2026-04-23 21:40** Sent back PR #5 (tanjiro), PR #6 (nezuko) for L1 rebase.
- **2026-04-23 22:20** Cherry-picked scoring.py fix from PR #9 (commit 7d71abd). Closed GH issue #10.
- **2026-04-23 22:20** Sent back PR #8 (edward): clip fires 100% — need higher thresholds; rebase on L1.
- **2026-04-23 22:20** Sent back PR #9 (thorfinn): conflict with L1 merge; rebase + compound sweep on L1. **The asinh-s458 MSE result beats L1 baseline by 2.9% — if it replicates on L1 it'll be the new baseline.**

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 22:20.

---

## Current Research Focus and Themes

**Highest-EV experiments currently in flight:**

1. **thorfinn #9 r2 (asinh-on-L1)** — If asinh mechanism is orthogonal to L1 (as theory predicts), compound gain could push baseline into the 95–98 range. Target reparameterization is a genuinely new axis; the MSE result showed large OOD effects (win on test splits).

2. **fern #12 (throughput/AMP)** — The hidden bottleneck. Every round-1 run stopped at epoch 14/50 due to 30-min wall-clock. Unlocking 25–35 epochs compounds with every other improvement. This is the force-multiplier experiment.

3. **alphonse #7 (Fourier PE + FiLM)** — Feature conditioning. Expected to help most on OOD camber + re_rand splits. Fourier features address spatial sharpness (boundary layer), FiLM addresses per-Re scale drift.

**Medium-EV:**

4. **edward #8 r2 (EMA + wider clip)** — EMA 0.999 is orthogonal to loss choice and should add ~3–5% on top of L1. Higher clip thresholds test whether clip has residual value on L1.

5. **frieren #11 (fine surf_weight)** — L1-sw10 ≈ L1-sw20 at round 1 — the optimum may lie below 10.

6. **tanjiro #5 r2 (channel-weighting on L1)** — psurf=20 gave −4.9% on MSE; need to see if it compounds.

7. **nezuko #6 r2 (schedule + seeds)** — disambiguate whether min_lr=1e-5 effect is real or noise.

---

## Potential Next Research Directions (Round 3+)

Ranked candidates from `research/RESEARCH_IDEAS_2026-04-23_21:00.md`:

### Not yet assigned, high EV
- **Horizontal-flip augmentation with Uy sign-flip** — physics-exact 2× data doubling. Cheap, orthogonal to everything in flight.
- **Domain-id as input feature (one-hot or FiLM)** — the right way to use the per-domain signal that broke in thorfinn's normalization experiment.
- **Channel-aware asinh** — asinh only on p, keep Ux/Uy z-score. Probably cleaner than the current global reparameterization.
- **Near-surface volume-band weighting** (3-tier: far-vol / near-vol / surf) — boundary-layer resolution directly drives surface pressure fidelity.

### Mid-term architectural
- **Cross-attention decoder with surface/volume query separation** — specialised surface head.
- **Sample-wise z-score with Re-predicted scale** — addresses 10× per-sample y_std variance directly.

### Longer-term physics-informed
- **Panel-method residual learning** — predict viscous correction on top of potential-flow prior.
- **Divergence-free regularization** + Kutta condition at trailing edge.

### Compounding plan after current round 2 lands
If asinh-on-L1 (thorfinn) and throughput (fern) both merge:
1. Re-run Fourier/FiLM on the improved + longer-epoch baseline.
2. Test EMA at 0.9999 (longer EMA horizon now feasible with more epochs).
3. Introduce horizontal-flip augmentation as standalone PR (high expected lift).
4. Channel-weighting + asinh combined experiment.
