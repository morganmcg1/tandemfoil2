# SENPAI Research State

- **Updated:** 2026-04-23 23:35 (round 4 continued)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 88.268 / test_avg/mae_surf_p = 79.733** (PR #12, `fern/sw1-amp-accum4`, W&B run `n68w9q7o`)
- Per-split val: in_dist=104.50 | camber_rc=100.70 | camber_cruise=65.19 | re_rand=82.69
- Config: **L1 loss, surf_weight=1, AMP (bf16), grad_accum=4**, bs=4, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4
- Best epoch: 19 (AMP unlocked +5 epochs vs PR #11's 14)

**Round-4 takeaway:** AMP + accum=4 gives a ~5% direct improvement AND unlocks +5 epochs per 30-min budget, which compounds with every subsequent experiment. **First finite `test_avg/mae_surf_p` on this track.**

**Key prior insights still binding:**
- L1 > MSE (PR #3), surf_weight=1 under L1 > sw=10 without AMP (PR #11)
- **New (PR #14 r4):** sw=2 beats sw=1 under L1+AMP+accum=2. The surface-weight optimum has shifted right with throughput improvements — retest at accum=4 pending.
- Scoring bug (GH #10) fixed (commit 7d71abd) — `test_avg/mae_surf_p` now produces finite numbers.
- Fourier σ=1 on L1 monotonic in m (m=40 hit 93.25, m not saturated; PR #7 r3b)
- Asinh + L1 compounds within sw=10 (−9 pts); open question whether it also compounds on sw=1.
- **Seed noise floor is ~9%** on single-seed runs (frieren's sw=1 no-AMP anchor vs PR #11 anchor). **Multi-seed required for any claim <5% improvement.**

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r4b) | #14: sw>1 sweep at eff_bs=16 + 2-seed anchor | `frieren/surf-weight-sub1` |
| fern     | WIP (r4) | #16: Capacity scaling on AMP baseline (h/l/s sweep) | `fern/capacity-on-amp` |
| tanjiro  | WIP (r3b) | #15: H-flip augmentation (physics-confirmed, running next) | `tanjiro/horizontal-flip-augmentation` |
| nezuko   | WIP (r3b) | #6: LR floor + WSD on **sw=1** (rerun) | `nezuko/lr-schedule-sweep` |
| alphonse | WIP (r3b) | #7: Fourier m-extend {40,80,160} on **sw=1**, no FiLM | `alphonse/fourier-pe-film-re` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r3b) | #9: asinh + **sw=1** 3-seed compound | `thorfinn/pressure-target-reparam` |

**Idle students needing assignment:** none. All 7 GPUs occupied.

## ⚠️ Round-3 systemic footgun

Argparse default for `surf_weight` is still `10.0` in train.py. PR #11 established sw=1 as the winning recipe but did NOT change the default (it passed `--surf_weight 1` at runtime). **Three of four round-3b submissions (PRs #6, #7, #9) silently ran on sw=10** because students rebased the code but didn't realize the flag change was load-bearing. Every send-back now explicitly asks for `--surf_weight 1`. Future assignment templates should include this flag in the reproduce command.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #16 | fern     | Capacity scaling on AMP baseline (width/depth/slice sweep) | Beat **88.268** |
| #14 | frieren  | sw>1 sweep {1, 1.5, 2, 3, 5, 10} × AMP + grad_accum=4 + 2-seed anchor | Beat **88.268** |
| #15 | tanjiro  | Horizontal-flip augmentation (x-flip, physics-confirmed) on sw=1 | Beat **88.268** |
| #6  | nezuko   | Rerun: WSD@1e-3 + floor=1e-5 stacked on sw=1 (2-seed) | Beat **88.268** |
| #7  | alphonse | Fourier m-saturation {40, 80, 160} at σ=1 on sw=1, no FiLM | Beat **88.268** |
| #8  | edward   | EMA 0.999 + wider grad-clip ({1, 5, 10, 50}) on L1 sw=1 | Beat **88.268** |
| #9  | thorfinn | asinh-on-sw=1 3-seed compound @ s∈{250,300,350,458} | Beat **88.268** |

**Note:** Six of seven in-flight PRs were assigned against the 93.127 target. Now that AMP merged as 88.268, they all have a harder bar. Most WIP experiments will rebase onto AMP on their next iteration; the asinh compound (thorfinn) and Fourier m-saturation (alphonse) are the most likely candidates to clear 88.268.

---

## Recent Decisions Log

- **2026-04-23 21:40** Merged PR #3 (frieren L1): baseline set to 103.036.
- **2026-04-23 21:40** Closed PR #4 (fern capacity): undertrained, not undercapacity — reassigned to throughput (PR #12).
- **2026-04-23 22:20** Cherry-picked scoring.py fix from PR #9 (commit 7d71abd). Closed GH issue #10.
- **2026-04-23 22:20** Sent back PR #8 (edward): wider clip thresholds; rebase on L1.
- **2026-04-23 22:20** Sent back PR #9 (thorfinn): conflict with L1 merge; rebase + compound sweep.
- **2026-04-23 22:35** Sent back PR #7 (alphonse): Fourier σ=1 wins on MSE, needs L1 rebase + per-block FiLM.
- **Round 3:** Merged PR #11 (frieren surf_weight=1): **new baseline 93.127** (−9.62% vs prior).
- **Round 3:** Closed PR #5 (tanjiro channel-weighting): dead end under L1. Tanjiro idle.
- **Round 3:** Sent back PR #12 (fern AMP): 93.29 no longer beats new baseline; rerun with sw=1.
- **Round 3b (22:50):** Sent back PRs #6 (nezuko), #7 (alphonse), #9 (thorfinn) — all ran on sw=10 argparse default, missed the PR #11 runtime flag. Sent back #15 (tanjiro) with physics sign correction (x-flip: negate Ux not Uy; student caught the assignment bug).
- **Round 3b insight:** Asinh + L1 compound confirmed (−9 pts within sw=10). Seed variance at s=458 is std 2.07 — any effect < 3% needs multi-seed confirmation going forward.
- **Round 3b insight:** Fourier m is NOT saturated on L1. Monotonic m=10→20→40 val improvement (97.51→94.11→93.25). m=80+ is the obvious next step.
- **Round 3b insight:** Per-block FiLM alone is a regressor (−9.2% on sw=10); dropping it from next alphonse sweep.
- **Round 4 (23:00): Merged PR #12 (fern AMP + accum=4): new baseline 88.268 val / 79.733 test (−5.2% on val, −13.2% on test vs PR #11).** AMP unlocks +5 epochs per 30-min budget. First finite test metric on track.
- **Round 4 (23:05):** Assigned fern PR #16 (capacity scaling on AMP) — the clean capacity sweep that round 1's PR #4 couldn't run because of wall-clock bottleneck.
- **Round 4b (23:30):** Sent back PR #14 (frieren). Sub-1 surf_weight confirmed dead (2-experiment replication with PR #12). But **sw=2 wins the AMP arm at val 89.27** — reverses PR #11's sw=1 optimum. Ran at grad_accum=2 not grad_accum=4, so not a clean test of current baseline — retesting at eff_bs=16 with 2-seed anchor. Also tightened multi-seed requirement from <3% → <5% effects, because frieren's no-seed anchor varied 9% from PR #11's published number.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 (round 3).

---

## Current Research Focus and Themes

The research has established a strong new operating point: L1 loss + surf_weight=1. The key insight is that under L1, volume supervision is equally load-bearing to surface supervision for the pressure metric — upweighting surface starves the shared trunk. This is the opposite of the MSE-era intuition.

**Highest-EV in-flight experiments (vs new 88.268 baseline):**

1. **alphonse #7 r4 (Fourier σ=1, m ∈ {40, 80, 160} on sw=1+AMP)** — Fourier m was NOT saturated at m=40 on sw=10 (monotonic m=10→20→40 win pattern). Combined with AMP's extra 5 epochs, m=80–160 should produce the strongest compound gain; most likely to beat 88.268.

2. **thorfinn #9 r4 (asinh-on-sw=1+AMP 3-seed compound)** — The decisive orthogonality test. If asinh and sw=1 stack, expect val ≈ 82–85. If redundant, ≈ 88–90 and we close asinh as a dead end under the new recipe.

3. **fern #16 r4 (capacity scaling on AMP)** — Now properly evaluable because AMP+accum unblocked the epoch budget. Width=192 or 256 is the canonical first bet.

4. **tanjiro #15 (horizontal-flip augmentation)** — Physics-exact 2× data, orthogonal to every other change. Expected 3–5% on OOD splits.

**Medium-EV:**

5. **edward #8 r2 (EMA + wider clip on L1 sw=1)** — EMA 0.999 is orthogonal to loss choice; clip threshold {5, 10, 50} probes the informative regime (round 1 showed all threshold ≤1 clip 100% of steps).

6. **nezuko #6 r3b (WSD + LR floor stacked on sw=1)** — WSD@1e-3 and floor=1e-5 both replicate on L1. GPU 5 tests the stack (floor+WSD).

7. **frieren #14 (sub-1 surf_weight + AMP compound)** — probes sw ∈ {0.25, 0.5, 1} on AMP. Likely confirms sw=1 is the optimum given PR #12 already probed sw=0.5 and sw=0.25 (both regress).

---

## Potential Next Research Directions (Round 4+)

### Immediate high-EV (ready to assign once students idle)
- **sw sub-1 micro-sweep** (sw ∈ {0.25, 0.5}) — frieren's suggestion: edge of sweep, true minimum may be below 1. Quick 2-run check.
- **Horizontal-flip augmentation with Uy sign-flip** — physics-exact 2× data doubling. ~80 LOC. Orthogonal to all current changes.
- **Near-surface volume-band weighting** (3-tier: far-vol / near-vol / surf) — boundary-layer nodes need more gradient; sw=1 treats all volume equally. Target: 3–5% further gain.
- **SwiGLU feedforward replacement** — replace standard MLP in each TransolverBlock with SwiGLU (~30 LOC). Known wins in transformers. Orthogonal.

### Mid-term architectural
- **Cross-attention decoder with surface/volume query separation** — dedicated surface head at ~120 LOC. Directly targets the primary metric.
- **Attention temperature schedule** — anneal hardcoded `self.temperature=0.5` from high→0.5 in first 20% of training. ~10 LOC.
- **Sample-wise per-sample normalization** — addresses 10× per-sample y_std variance; moderate complexity.
- **Coordinate-based slice assignment** (slice projection from (x,z,is_surface) only) — more interpretable spatial partition; may improve OOD.

### Physics-informed
- **Kutta condition soft enforcement at trailing edge** — identify TE node, add |p_upper - p_lower| aux loss. ~150 LOC.
- **Learnable dsdf aux head** — predict signed distance as aux target; forces trunk to encode geometry cleanly. ~40 LOC.
- **Panel-method residual learning** — model predicts viscous correction on top of thin-airfoil prior. High impact, high complexity.

### Compounding plan after in-flight PRs land
If asinh (thorfinn) and AMP (fern) both merge, rerun with combined config as new baseline. Then:
1. H-flip augmentation as standalone PR.
2. Fourier + FiLM on the compounded config.
3. Capacity scaling (h=256) now feasible with AMP+accum — VRAM headroom exists.
