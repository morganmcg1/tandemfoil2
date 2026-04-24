# SENPAI Research State

- **Updated:** 2026-04-24 03:30 (round 12 — post PR #24 merge)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 69.845 (best single-seed) / 70.667 (2-seed mean) — test 62.778 / 62.691** (PR #24, `flgrjmte` / `j12mrpeb`)
- Per-split val (best seed): in_dist≈80.3 | camber_rc≈82.5 | camber_cruise≈48.8 | re_rand≈67.8
- Config: **L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU FFN**
- Best epoch: 17 (both seeds)

**Round-12 takeaway:** σ=0.7 + SwiGLU is a sharp optimum, not a flat basin. σ=0.8/0.9 regress (79–80 val) with high seed variance (std 3–4) — optimization pathology in that band. Under the SwiGLU recipe, seed variance at well-tuned σ is extremely tight (anchor std 0.362 val — ~20× tighter than pre-SwiGLU). First merge under strict multi-seed protocol.

**Round-9 takeaway (still binding):** SwiGLU compound with Fourier PE delivered −13.1 % val / −15.0 % test vs PR #7.

**Key prior insights still binding:**
- L1 > MSE (PR #3), sw=1 under L1 (PR #11), AMP + grad_accum=4 +5 epochs (PR #12), Fourier PE fixed σ=1 m=160 (PR #7), SwiGLU FFN (PR #20). Five compounding components.
- **Seed variance floor: σ ≈ 8 pts on no-pinned-seed runs; ~2.5% on seeded serial runs; 4-5% on 8-parallel-run IO-contended runs.**
- **Multi-seed protocol:** 2-seed anchors mandatory for merge claims < 5% (established in rounds 7–8 from frieren + alphonse + nezuko data).
- Schedule effects don't transfer across regimes (nezuko's 3-round finding). Capacity scaling is epoch-budget bound (fern PR #4/#16). sub-1 surf_weight is dead. Asinh is partially redundant with sw=1.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r10) | #26: Sample-wise normalization with Re-predicted scale | `frieren/sample-wise-renorm` |
| fern     | WIP (r9) | #25: SwiGLU refinements (mlp2 head + mlp_ratio) | `fern/swiglu-refinements` |
| tanjiro  | WIP (r12) | #17 r3: Gap σ-scan on SwiGLU baseline + tandem-gated AoA | `tanjiro/input-feature-jitter` |
| nezuko   | WIP (r11) | #27: slice_num sweep on merged recipe | `nezuko/slice-num-sweep` |
| alphonse | WIP (r12) | #28: Fine σ sweep {0.5, 0.55, 0.6, 0.65, 0.7, 0.75} + 2-seed | `alphonse/sigma-fine-sweep` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r9) | #23: Zero-init residual surface decoder | `thorfinn/zeroinit-residual-decoder` |

**Idle students:** none. Zero idle GPUs.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #26 | frieren  | Sample-wise normalization with Re-predicted scale | Beat **70.67** (2-seed mean) |
| #25 | fern     | SwiGLU refinements (mlp2 head SwiGLU + mlp_ratio expansion) | Beat **70.67** |
| #17 | tanjiro  | r3: Gap σ-scan on SwiGLU baseline + tandem-gated AoA | Beat **70.67** |
| #27 | nezuko   | slice_num sweep {32, 48, 64, 96, 128, 192} + 2-seed anchor | Beat **70.67** |
| #28 | alphonse | Fine σ sweep {0.5, 0.55, 0.6, 0.65, 0.75} + 2-seed at winner | Beat **70.67** |
| #8  | edward   | EMA 0.999 + wider grad-clip on L1 | Beat **70.67** |
| #23 | thorfinn | Zero-init residual surface decoder | Beat **70.67** |

---

## Recent Decisions Log

- **Round 1–6:** L1 (PR #3), sw=1 (#11), AMP+grad_accum=4 (#12), Fourier PE m=160 σ=1 (#7) merged — compound 84.737 baseline.
- **Round 7 (r7):** Closed PR #14 (sw>1 exhausted). Seed floor on seeded serial: 2.5%.
- **Round 8 (r8):** Closed PR #6 (LR schedule 3-round exhaustion — regime-specific artefacts). Sent back #17 (gap-only jitter signal real). Assigned #22 (temp annealing).
- **Round 12 (r12 — 2026-04-24 03:30):**
  - 🏆 **MERGED PR #24 (alphonse σ × SwiGLU):** new baseline **69.845 best-seed / 70.667 2-seed-mean val; 62.778 / 62.691 test**. −5.2% val vs PR #20 under strict multi-seed protocol (first merge under the new rule). 2-seed anchor std only 0.362 val — SwiGLU recipe stabilizes seed variance ~20× vs pre-SwiGLU. Winner is ~9 σ_anchor outside noise. Verified fern's crashed σ=0.7 claim bit-exactly at seed=0 (71.489).
  - **σ landscape finding:** σ=0.7 is a SHARP minimum. σ=0.8 (79.14) and σ=0.9 (77.99) regress with high seed variance (std 3–4 val) — optimization pathology in that band.
  - **Sent back PR #17 (tanjiro):** 8th stale-rebase (pre-SwiGLU). σ=0.04 gap-jitter is the new peak at seed=42 (89.84 on pre-SwiGLU) but r5 σ=0.02 peak (89.88) was seed-lucky (3-seed mean 94.22). Rebase + focused σ ∈ {0.04, 0.06, 0.08} + tandem-AoA probe.
  - **Assigned PR #28 (alphonse):** fine σ sweep {0.5, 0.55, 0.6, 0.65, 0.7, 0.75} + 2-seed at candidate winner. Natural follow-up on the sharp-minimum finding.

- **Round 11 (r11 — 2026-04-24 02:00):**
  - **Closed PR #22 (nezuko temperature annealing):** hypothesis rejected at 2-seed level. Anneal 2-seed mean (91.06) was +0.7 val WORSE than anchor 2-seed mean (90.36); effect buried in 8-val anchor spread. After release, per-head temperatures drift back to [0.3, 0.7] around original 0.5 — init was already optimal. Branch pre-SwiGLU (7th consecutive stale-rebase). Student execution exemplary (grad-freeze during anneal was a smart deviation). Reassigned to PR #27 (slice_num sweep on merged recipe).
  - **Protocol observation:** 7 consecutive students with stale-rebase on pre-merge branches. Mandatory --debug verification step now lead-item in every new assignment.

- **Round 10 (r10 — 2026-04-24 01:45):**
  - **Closed PR #21 (frieren BL weighting):** uniform regression across all tier weights (monotonic w_near ∈ {2,3,5}: 105.6→118.9). Student's self-diagnosis correct — per-tier-mean formulation sums 3 volume-tier means, effectively downweighting surface. Branch pre-SwiGLU (6th stale-rebase). Loss-weighting landscape now mapped across 4 PRs (#3, #11, #14, #21). Reassigned to #26 (sample-wise renorm with Re-predicted scale) — new territory.

- **Round 9 (r9):**
  - **Merged PR #20 (fern SwiGLU + σ=1):** new baseline **73.660 val / 63.983 test** (−13.1% val / −15.0% test). Huge win. Unified SwiGLU implementation with 2/3 hidden-width to preserve param count. Anisotropic per-coord σ regressed; σ=1 is the robust optimum among completed runs.
  - **Student's σ=0.7+SwiGLU compound claim (val 71.49) was based on crashed W&B runs** — unverified, not merged. Re-assigned to alphonse as verified multi-seed sweep (PR #24).
  - **Closed PR #18 (thorfinn decoder):** catastrophic (+175% worse). Fresh-parameter head can't catch trunk in 15 epochs. Salvage path: zero-init residual decoder (assigned as PR #23).
  - **Closed PR #19 (alphonse m-extension):** m=160 saturation confirmed; learnable B loses at this budget. Key finding: **seed variance at m=160 is σ ≈ 8 pts; 84.737 was ~1σ below config-mean**. Multi-seed protocol now mandatory.

---

## Most Recent Research Direction from Human Team

No human issues received.

---

## Current Research Focus and Themes

**Highest-EV in-flight experiments (vs new 73.66 baseline):**

1. **alphonse #24 (σ × SwiGLU seeded sweep)** — verifies whether fern's crashed σ=0.7+SwiGLU claim (71.49) is real or noise. First experiment to run under the strict 2-seed merge protocol. If σ<1 robustly wins, new baseline ~71. If σ=1 confirmed optimal, we close σ-tuning as a direction.

2. **thorfinn #23 (zero-init residual surface decoder)** — PR #18 showed the decoder architecture is sound but the training protocol is wrong. Zero-init residual structure means the decoder starts invisible, can only improve. Targets surface pressure directly.

3. **fern #25 (SwiGLU refinements)** — natural next step after PR #20. mlp2 head SwiGLU + mlp_ratio expansion {2, 3, 4}. Expected +1–4% per component.

**Medium-EV:**

4. **frieren #21 (near-surface volume-band weighting)** — 3-tier loss using dsdf to define BL band. Physics-motivated; orthogonal to everything.

5. **nezuko #22 (attention temperature annealing)** — fresh architectural direction after 3 rounds of LR schedule work. Low complexity.

6. **tanjiro #17 (gap-jitter σ-scan on Fourier baseline)** — gap-only signal replicated last round at −5.89 pts vs anchor. Needs verification on current baseline with tandem-gating.

7. **edward #8 (EMA + clip on L1)** — long-running (r2); EMA is orthogonal to loss choice.

---

## Potential Next Research Directions (Round 10+)

### High-EV if round 9 lands winners
- **Warm-start from merged checkpoint** for architectural experiments (e.g., if thorfinn's zero-init residual decoder still underperforms, try warm-starting from the PR #20 checkpoint).
- **Per-block Fourier injection** — condition mid-network features with frequency-varying codes (alphonse's suggested follow-up from PR #19).
- **Pressure-gradient-weighted L1 loss** — focus loss on nodes where surface pressure varies steeply (boundary-layer regions inferred from dsdf or from |∇p| in training batches).

### Novel physics-informed
- **Kutta condition soft enforcement** at trailing edge (~150 LOC).
- **Panel-method residual learning** — high complexity but direct OOD lift potential.
- **Divergence-free regularization** on volume nodes via finite-difference.

### Scale
- **Capacity scaling on merged recipe** (h=128 → 192 with SwiGLU) — may finally succeed now that per-epoch efficiency is higher. Budget still marginal; needs ~20+ epochs at h=192.
