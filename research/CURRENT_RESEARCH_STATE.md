# SENPAI Research State

- **Updated:** 2026-04-24 00:10 (round 5 start)
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
| frieren  | WIP (r4b) | #14: sw>1 sweep at eff_bs=16 + 2-seed anchor | `frieren/surf-weight-subunit-plus-amp` |
| fern     | WIP (r4) | #16: Capacity scaling on AMP baseline (h/l/s sweep) | `fern/capacity-on-amp` |
| tanjiro  | WIP (r5) | #17: In-distribution input jitter (AoA/logRe/gap) | `tanjiro/input-feature-jitter` |
| nezuko   | WIP (r5) | #6: AMP rebase + fixed WSD stack test + cosine@1e-3 control | `nezuko/lr-schedule-sweep` |
| alphonse | WIP (r3b) | #7: Fourier m-extend {40,80,160} on sw=1, no FiLM | `alphonse/fourier-pe-film-re` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r5) | #18: Cross-attention surface decoder head | `thorfinn/cross-attn-surface-decoder` |

**Idle students needing assignment:** none. All 7 GPUs occupied.

## ⚠️ Systemic footgun — two rounds deep

Argparse defaults for `surf_weight`, `amp`, `grad_accum` are stale. Merged-winner recipes (PR #11 sw=1, PR #12 AMP+grad_accum=4) were runtime-flag changes, not code-default changes.

**Round 3b (sw=1 flag):** PRs #6, #7, #9 silently ran on sw=10.
**Round 5 (AMP + grad_accum flags):** PRs #6 (nezuko), #9 (thorfinn), #15 (tanjiro) silently ran pre-AMP. All three branches forked BEFORE PR #12 merged and never picked up AMP/grad_accum.

Every new assignment now:
1. Leads with explicit `--loss_type l1 --surf_weight 1 --amp true --grad_accum 4`
2. Requires a `--debug` sanity-check run with verification that W&B config shows `amp=True, grad_accum=4` BEFORE committing to the full sweep
3. Pins seeds and requires 2-seed anchors for any claim <5%

**Noise floor:** ~9% on single-seed pre-AMP runs (frieren sw=1 no-AMP ≈ 102 vs PR #11's 93.13; thorfinn s=350 σ=6.74). AMP+grad_accum likely tightens this.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #16 | fern     | Capacity scaling on AMP baseline (width/depth/slice sweep) | Beat **88.268** |
| #14 | frieren  | sw>1 sweep {1, 1.5, 2, 3, 5, 10} × AMP + grad_accum=4 + 2-seed anchor | Beat **88.268** |
| #17 | tanjiro  | In-distribution input jitter (AoA/logRe/gap) on AMP + sw=1 | Beat **88.268** |
| #6  | nezuko   | AMP rebase + fixed WSD stack + cosine@1e-3 control | Beat **88.268** |
| #7  | alphonse | Fourier m-saturation {40, 80, 160} at σ=1 on sw=1, no FiLM | Beat **88.268** |
| #8  | edward   | EMA 0.999 + wider grad-clip ({1, 5, 10, 50}) on L1 sw=1 | Beat **88.268** |
| #18 | thorfinn | Cross-attention surface decoder head on AMP + sw=1 | Beat **88.268** |

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
- **Round 5 (00:00):** Three pre-AMP rebase failures (PRs #6, #9, #15). Closed PR #9 (thorfinn asinh — partial redundancy with sw=1 confirmed, 5 rounds deep, reassigned to cross-attention surface decoder PR #18). Closed PR #15 (tanjiro hflip — AoA sign-flip creates OOD samples, clean negative, reassigned to in-distribution input jitter PR #17). Sent back PR #6 (nezuko — AMP rebase + WSD min_lr bug fix + retune phase splits 10/30/60 for realized 14-epoch budget).
- **Round 5 insight:** Asinh × sw=1 partial redundancy (both rebalance surface↔volume residuals). The asinh direction is now well-mapped. Architectural approaches (surface decoder, capacity) should take precedence over further loss-weight tuning.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 (round 3).

---

## Current Research Focus and Themes

The research has established a strong new operating point: L1 loss + surf_weight=1. The key insight is that under L1, volume supervision is equally load-bearing to surface supervision for the pressure metric — upweighting surface starves the shared trunk. This is the opposite of the MSE-era intuition.

**Highest-EV in-flight experiments (vs 88.268 baseline):**

1. **alphonse #7 (Fourier σ=1, m ∈ {40, 80, 160} on sw=1+AMP)** — Fourier m was NOT saturated at m=40 on sw=10 (monotonic m=10→20→40 win pattern). With AMP's 19 epochs, m=80–160 should produce the strongest compound gain; most likely to beat 88.268.

2. **thorfinn #18 (cross-attention surface decoder)** — Fresh architectural direction. Primary metric is surface-p; dedicated decoder head for surface nodes (Perceiver IO-style) directly targets it. No prior experiment has tried this architectural angle. Researcher-agent top-5 idea.

3. **fern #16 (capacity scaling on AMP)** — The clean capacity sweep unblocked by AMP+grad_accum. Width=192 or 256 is the canonical first bet; previously failed at pre-AMP due to wall-clock constraint.

4. **tanjiro #17 (input jitter: AoA + logRe + gap)** — In-distribution augmentation that doesn't hit hflip's AoA-OOD trap. Physics-safe, 40 LOC, expected 2–4% on OOD splits.

**Medium-EV:**

5. **nezuko #6 (AMP rebase + fixed WSD + cosine@1e-3 control)** — WSD implementation bug fixed (min_lr now threads into lambda), phase splits retuned for 14-epoch realized budget. GPU 6 is the corrected stack test.

6. **edward #8 (EMA + wider clip on L1 sw=1)** — EMA 0.999 is orthogonal to loss choice; clip thresholds {5, 10, 50} probe the informative regime (round 1 showed clip ≤1 fires 100% of steps, effectively unit-normalizing gradients).

7. **frieren #14 (sw>1 at eff_bs=16 + 2-seed)** — tests whether sw=2's AMP-arm win replicates at grad_accum=4. Same mechanism as PR #11 but at the right batch-size regime.

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
