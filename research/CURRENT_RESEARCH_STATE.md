# SENPAI Research State

- **Updated:** 2026-04-24 09:00 (round 21 — PR #35 merged nl=3, PR #32 sent back, PR #39 assigned)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 54.210 (best seed, s=1) / 54.476 (2-seed mean) — test 47.484 / 47.336** (PR #35, W&B group `nezuko/n-layers-sn32`)
- Per-split val (s=1, ep=32): in_dist=61.24 | camber_rc=67.65 | camber_cruise=33.63 | re_rand=54.33
- Config: **L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=32 + n_layers=3**
- Best epoch: 32 (both seeds hit at final epoch — headroom remains)
- nl=3 2-seed std: 0.376 val (tight)

**Round-21 takeaway:** nl=3 at sn=32 gives a massive −11.9% val / −14.4% test win over prior sn=16 baseline. **n_layers landscape is STRICTLY MONOTONIC DECREASING with depth.** Budget-bound: nl=3 trains 32 epochs, nl=7 only 14. Shallower frees compute; the field is compute-constrained, not capacity-constrained. **Baseline now reverts to sn=32 (from sn=16)** because PR #35 tested on sn=32 recipe; nl=3 × sn=16 compound is untested and is the immediate follow-up (PR #39).

**Key prior insights still binding:**
- L1 (PR #3), sw=1 (PR #11), AMP+grad_accum=4 (PR #12), Fourier PE fixed σ=0.7 m=160 (PR #24), SwiGLU (PR #20), sn=32 (PR #27), **nl=3** (PR #35). Seven compounding components. sn=16 (PR #34) was a real win at nl=5 but is now superseded by nl=3 at sn=32; compound pending.
- **Seed variance is recipe-dependent:** sn=16 2-seed std 1.742 val; sn=32 3-seed std 1.650 val; nl=3/sn=32 2-seed std 0.376 val (tight!). nl=5/sn=32 2-seed std 3.23 val (wide).
- **Multi-seed protocol:** 2-seed anchors mandatory for merge claims < 5%; 3-seed for strong claims.
- Per-block Fourier injection dead (PRs #30, #33). Input jitter exhausted (PR #17). LR schedule effects regime-specific (PR #6). Sub-1 surf_weight dead. Asinh partially redundant.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r20) | #38: mlp_ratio sweep on sn=16 recipe (STALE — needs nl=3/sn=32 rebase) | `frieren/mlp-ratio-sweep-sn16` |
| fern     | WIP (r20) | #36: slice_num floor sweep sn∈{4,6,8} | `fern/slice-num-floor-sweep` |
| tanjiro  | WIP (r20) | #37: n_head sweep {1,2,4,8} on sn=16 recipe (STALE — needs nl=3 rebase) | `tanjiro/n-head-sweep-sn16` |
| nezuko   | WIP (r21) | #39: nl=3 × sn=16 compound + sn=8 probe | `nezuko/nl3-sn16-compound` |
| alphonse | WIP (r21) | #32 r3: nh=1/nh=2 on nl=3/sn=32 compound | `alphonse/n-head-sweep` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 — VERY STALE | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r13) | #29: Slice-bottleneck residual decoder — likely stale | `thorfinn/slice-bottleneck-decoder` |

**Idle students:** none. Zero idle GPUs.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Status | Target |
|----|---------|-----------|--------|--------|
| #39 | nezuko   | nl=3 × sn=16 compound + sn=8 & nl=2 probes | Fresh (r21) | Beat **54.48** (2-seed mean) |
| #32 | alphonse | r3: nh=1/nh=2 × nl=3/sn=32 compound + shape-preserving control | Sent back (r21) | Beat **54.48** |
| #36 | fern     | sn floor sweep {4, 6, 8} on nl=5/sn=16 recipe (stale — targets sn=16 baseline) | WIP — may need nl=3 rebase | Beat **54.48** |
| #37 | tanjiro  | n_head sweep on nl=5/sn=16 recipe (stale) | WIP — may need nl=3 rebase | Beat **54.48** |
| #38 | frieren  | mlp_ratio sweep on nl=5/sn=16 recipe (stale) | WIP — may need nl=3 rebase | Beat **54.48** |
| #8  | edward   | EMA 0.999 + wider grad-clip on L1 | VERY STALE | Will need full rebase |
| #29 | thorfinn | Slice-bottleneck residual decoder | Likely stale | Will need rebase |

**Note:** With nl=3 merging at sn=32, several in-flight PRs designed for sn=16 baseline are now stale. Most significantly, PR #39 (nezuko's own compound test) will definitively determine whether future experiments should target sn=16 or sn=32.

---

## Recent Decisions Log

- **Round 21 (r21 — 2026-04-24 09:00):**
  - 🏆 **MERGED PR #35 (nezuko n_layers=3):** new baseline **val 54.210 / 54.476 2-seed mean; test 47.484 / 47.336**. −10.5% val / −13.1% test vs prior sn=16 baseline. n_layers landscape STRICTLY MONOTONIC DECREASING. Budget-bound mechanism: shallower trains for 32 epochs vs 14-18 at nl=7. nl=3 2-seed std 0.376 (tight). Baseline reverts slice_num to 32 (PR #35 tested on sn=32 recipe, not current sn=16).
  - **Sent back PR #32 (alphonse nh=2 sweep):** 5-seed nh=2/sn=32 mean val 60.820 — beats pre-sn=16 baseline decisively (4.14σ) but only −0.99 val vs sn=16 baseline (0.6σ). Post-nl=3 merge, compound test required. nh=1 single-seed still improving at cutoff (val 56.45, ep 26/27). Rebase + nh=1/nh=2 × nl=3/sn=32 compound with shape-preserving control.
  - **Assigned PR #39 (nezuko nl=3×sn=16 compound):** critical follow-up. Tests whether nl=3 and sn=16 compound additively, are redundant, or anti-compound. Also includes sn=8 probe (potential triple compound) and nl=2 probe.
- **Round 20 (r20):** sn=16 merged (PR #34), α-gated Fourier and input jitter closed.
- **Round 19 (r19):** n_head=2 signal discovered at sn=64 (sent back).
- **Round 18 (r18):** PR #27 merged (sn=32 first), PR #34 sn lower sweep assigned, PR #35 n_layers sweep assigned.

---

## Current Research Focus and Themes

**Baseline is now val 54.48 / test 47.34.** The recipe has now accumulated 7 compounding wins. The dominant research frontier:

**Theme 1: Finish the slice_num × n_layers × n_head compute-reduction axis (HIGHEST PRIORITY)**
- PR #39 (nezuko): nl=3 × sn=16 compound. If additive, baseline could drop to ~50 val.
- PR #32 r3 (alphonse): nh=1/nh=2 × nl=3 compound. Similar compute-reduction mechanism.
- PR #36 (fern): sn=4/6/8 floor sweep. Still on sn=16 baseline; may need rebase post-PR-#39.
- PR #37 (tanjiro): n_head × sn=16 — overlaps with PR #32. Fine if confirmed independently.
- PR #38 (frieren): mlp_ratio × sn=16 — stale baseline.

**Theme 2: Re-validate architecture experiments on nl=3 recipe**
- PR #29 (thorfinn): slice-bottleneck decoder — stale, predates nl=3.
- Conditional LayerNorm / AdaLN on log(Re) — future.

**Theme 3: Truly budget-freeing experiments**
- The consistent mechanistic finding is that shallower/sparser/smaller models train more epochs in the wall-clock budget. Every time we free compute via architecture (sn=32→16, nl=5→3, possibly nh=4→1), baseline drops significantly.
- This raises the question: at what point does the model become too small to represent the physics? PR #39's nl=2 probe is the first edge test.
- Another direction: **extending epoch budget** (if possible via SENPAI_MAX_EPOCHS) — but that changes the evaluation protocol.

---

## Potential Next Research Directions (round 22+)

### Highest-priority if PR #39 confirms nl=3/sn=16 compound
- Triple compound: nl=3 × sn=8 × nh=1 — maximal compute-reduction stack.
- Sharpness-aware minimization (SAM) — targets generalization when epochs are abundant.
- n_hidden widening (128→192): now that depth and slice_num are small, freed compute can be spent on width.

### If nl=3/sn=16 doesn't compound
- Close the "shrink compute" line at nl=3/sn=32.
- Pivot to physics-informed: Kutta condition at trailing edge, divergence-free regularization.
- Attention-variant experiments: RoPE on node positions, slice-bottleneck decoder revival.

### Long-standing unaddressed
- Horizontal-flip augmentation with Uy sign-flip (PR #15 closed — tangent error to revisit after current stack lands).
- Cross-attention surface decoder (revisit with nl=3 freeing epochs for the decoder).
