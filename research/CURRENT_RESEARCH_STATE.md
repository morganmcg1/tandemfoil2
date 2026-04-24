# SENPAI Research State

- **Updated:** 2026-04-24 07:00 (round 18 — PR #27 merged, PR #34 assigned, PR #35 assigned)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 67.186 (best single-seed, seed=2) / 68.687 (3-seed mean) — test 58.358 / 60.680** (PR #27, W&B group `nezuko/slice-num-sigma07`)
- Per-split val (seed=2): in_dist=77.43 | camber_rc=76.45 | camber_cruise=49.72 | re_rand=65.14
- Config: **L1 + sw=1 + AMP + grad_accum=4 + Fourier PE fixed m=160 σ=0.7 + SwiGLU + slice_num=32**
- Best epoch: 18 (s=0), 19 (s=1), 20 (s=2)
- VRAM: 31.6 GB (freed ~16% vs sn=64); per-epoch: ~89s (−20%); budget: ~20 epochs in 30-min

**Round-17 takeaway:** PR #30 (fern per-block Fourier re-injection) catastrophically regressed (+47% val, +55% test) despite textbook zero-init. Key finding: zero-init alone is NOT "can only help" — post-step-0 learning dynamics can grow the projector harmfully. Reassigned to α-gated variant (PR #33).

**Round-16 takeaway:** σ axis fully mapped {0.5–1.0} across 9 values. σ=0.7 is a narrow well; σ=0.65/0.75/0.8/0.9 all regress. Secondary basin at σ=0.60 (2-seed mean 72.13). **Noise floor at σ=0.7 is wider: std=1.162 val (3.2× the PR-#24 anchor std of 0.362).**

**Round-15 takeaway:** PR #27 merged (new best: 67.186 val / 58.358 test). sn=32 gives −16% VRAM and −20% per-epoch vs sn=64, enabling ~20 epochs in budget. sn lower sweep continues via PR #34 (frieren).

**Key prior insights still binding:**
- L1 > MSE (PR #3), sw=1 under L1 (PR #11), AMP + grad_accum=4 (PR #12), Fourier PE fixed σ=0.7 m=160 (PR #24), SwiGLU FFN (PR #20), slice_num=32 (PR #27). Six compounding components.
- **Seed variance floor at σ=0.7:** std=1.162 val (3-seed; use this for merge decisions going forward, not the old 0.362 from σ=1 anchor).
- **Multi-seed protocol:** 2-seed anchors mandatory for merge claims < 5%.
- Schedule effects don't transfer across regimes. Asinh partially redundant with sw=1. sub-1 surf_weight dead. Per-block Fourier injection catastrophic (zero-init insufficient guard).

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r18) | #34: slice_num lower sweep sn∈{8,16,24} vs sn=32 anchor | `frieren/slice-num-lower-sweep` |
| fern     | WIP (r17) | #33: α-gated per-block Fourier (magnitude-constrained, zero-init α) | `fern/alpha-gated-pbf` |
| tanjiro  | WIP (r12) | #17 r3: Gap σ-scan on SwiGLU baseline + tandem-gated AoA | `tanjiro/input-feature-jitter` |
| nezuko   | WIP (r18) | #35: n_layers depth sweep on sn=32 recipe {3,4,5,6,7} | `nezuko/n-layers-sweep-sn32` |
| alphonse | WIP (r16) | #32: n_head sweep + 3-seed anchor recalibration | `alphonse/n-head-sweep` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r13) | #29: Slice-bottleneck residual decoder (PhysicsAttention) | `thorfinn/slice-bottleneck-decoder` |

**Idle students:** none. Zero idle GPUs.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #34 | frieren  | slice_num lower sweep sn∈{8,16,24} vs sn=32 anchor (3-seed) | Beat **67.186** (best seed) / **68.687** (3-seed mean) |
| #33 | fern     | α-gated per-block Fourier (per-block scalar α_i init=0, magnitude-constrained) | Beat **67.186** |
| #17 | tanjiro  | r3: Gap σ-scan on SwiGLU baseline + tandem-gated AoA | Beat **67.186** |
| #35 | nezuko   | n_layers depth sweep {3,4,5,6,7} on sn=32 recipe + 2-seed anchor recalibration | Beat **67.186** |
| #32 | alphonse | n_head sweep {2, 4, 8, 16} + 3-seed anchor recalibration | Beat **67.186** (threshold recalibration this PR) |
| #8  | edward   | EMA 0.999 + wider grad-clip on L1 | Beat **67.186** |
| #29 | thorfinn | Slice-bottleneck residual decoder (PhysicsAttention, zero-init) | Beat **67.186** |

---

## Recent Decisions Log

- **Round 1–6:** L1 (PR #3), sw=1 (#11), AMP+grad_accum=4 (#12), Fourier PE m=160 σ=1 (#7) merged — compound 84.737 baseline.
- **Round 7 (r7):** Closed PR #14 (sw>1 exhausted). Seed floor on seeded serial: 2.5%.
- **Round 8 (r8):** Closed PR #6 (LR schedule 3-round exhaustion). Sent back #17 (gap-only jitter signal real). Assigned #22 (temp annealing).
- **Round 18 (r18 — 2026-04-24 07:00):**
  - **No new PRs for review.** All 7 students now active.
  - **Assigned PR #35 (nezuko n_layers depth sweep):** First n_layers sweep on the fully-merged recipe (sn=32). Hypothesis: sn=32's −20% per-epoch speedup enables deeper models (n_layers=6,7) to converge within budget. Sweep n_layers∈{3,4,5,6,7} with 2-seed at n_layers=5 anchor. Also recalibrates the noise floor post-sn=32 merge.
  - **Baseline updated** to PR #27 result: val 67.186 best / 68.687 mean; test 58.358 / 60.680. All WIP PR targets updated accordingly.
- **Round 17 (r17 — 2026-04-24 06:00):**
  - **Closed PR #30 (fern per-block Fourier re-injection):** catastrophic regression (+47% val, +55% test) despite textbook zero-init implementation. **Key finding: zero-init alone is NOT a "can only help" guarantee — post-step-0 learning dynamics can grow the projector in a harmful direction.** Shared projector > per-block (103.93 vs 108.05, ~5σ) — reversed from NeRF/FiLM intuition. Largest regression on `val_single_in_dist` (+68%), not OOD splits → core representation disrupted. mlp_ratio=3 doesn't rescue. Reassigned to PR #33 (α-gated variant — per-block learnable scalar α_i init=0 as ControlNet-style magnitude gate; by construction mechanism can only activate if strictly helpful).

- **Round 16 (r16 — 2026-04-24 05:30):**
  - **Closed PR #28 (alphonse fine σ sweep):** σ=0.7 confirmed robust optimum. σ axis now fully mapped across 9 values {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0}. **Important landscape findings:**
    - Pathology band (divergence + best_ep=11) extends on BOTH sides of σ=0.7 — σ=0.65 joins σ=0.8/0.9 in the unstable band. σ=0.7 is a narrow well.
    - Secondary basin at σ=0.60 (2-seed mean 72.13, std 0.666) — viable fallback but clearly worse than σ=0.7.
    - **Variance at σ=0.7 is σ-dependent:** 2-seed std is **1.162 val**, ~3.2× the 0.362 anchor std measured at σ=1 in PR #24. **Current merge threshold (0.362) is too tight for σ=0.7 decisions.**
  - **Assigned PR #32 (alphonse n_head sweep + 3-seed anchor recalibration):** dual-purpose sweep. Recalibrate noise floor with 3-seed anchor (updates merge threshold lab-wide). Sweep untested n_head knob {2, 4, 8, 16} with 2-seed at candidates.

- **Round 15 (r15 — 2026-04-24 05:00):**
  - **Sent back PR #27 (nezuko slice_num):** advisor-side error — my assignment body specified `--fourier_sigma 1.0` because PR #24 (σ=0.7) hadn't merged yet at round 11. Nezuko ran faithfully to spec. Their results are directionally real: sn=32 2-seed mean 71.996 beats σ=1 anchor by 1.92 val on both seeds (4/4 seed-paired comparisons favor sn=32). But against the current σ=0.7 baseline (70.667), sn=32 is +1.33 val above — not merge-eligible. Sent back with focused re-run on σ=0.7: sn={32, 48, 96} + 3-seed at sn=32 candidate. Cliff at sn≥128 dropped (epoch-budget-bound at current wall-clock).
  - **PR #31 (frieren post-hoc rescale):** student preemptively marked ready to ask a clarification question. Review agent answered (option b for 1ch: preserve per-channel ratios via multiplicative correction `pred * y_std_global * (y_std_pred / y_std_global_geomean)`) and sent back to WIP. Student now unblocked.
  - **Lab-wide observation:** round-11 stale-rebase was actually **my specification error** (assignment body had the wrong σ for the eventual baseline). Correction: when a merge lands during the assignment cycle, the advisor must either (a) retroactively update open-assignment bodies, or (b) explicitly tell students to treat their branch as baseline-at-assignment-time. Haven't done either; adding this as a process note.

- **Round 14 (r14 — 2026-04-24 04:30):**
  - **Closed PR #25 (fern SwiGLU refinements):** mlp2-SwiGLU destabilizes training (2-seed std 3.21 val, 15× anchor noise — student correctly diagnosed the gated w3 × trunc_normal(0.02) init without residual depth scaling). mlp_ratio=3 is a flat plateau (73.35 vs 73.66); mr=4 regresses (+1.62) and costs 3 epochs. Compound (mr=3 + mlp2-SwiGLU) is the worst run. Branch pre-PR #24 (10th consecutive stale-rebase). Reassigned to PR #30 (per-block Fourier re-injection).
  - **Closed PR #26 (frieren sample-wise renorm):** decisive negative (+90 val vs in-PR anchor, ~248σ regression). Scale head learned log(Re) → log(y_std) cleanly (R² ≈ 0.9) but main-model retraining under shifted target distribution doesn't fit in 30-min budget — uniform 2–2.5× regression. Important mechanistic finding: λ_scale sweep architecturally uninformative because scale_head and main model share no parameters. Reassigned to PR #31 (post-hoc scale correction — student's own follow-up #3, inference-only, no retraining cost).
  - **Lab-wide observation:** 10 consecutive students have had stale-rebase issues on pre-merge branches. The merged-recipe flags (`--fourier_sigma 0.7`, `--swiglu`, `--amp true`, `--grad_accum 4`) are the persistent footgun; students rebase code but not runtime-flag assumptions. Debug-run verification step is the only reliable mitigation.

- **Round 13 (r13 — 2026-04-24 04:00):**
  - **Closed PR #23 (thorfinn zero-init residual decoder):** mechanism verified (surf_delta_step0=0 confirmed) but **budget-bound** — decoder ~2× slower than trunk, only reaches ep 6–9 vs anchor ep 17. ControlNet analogy fails when trunk is still training. Branch pre-PR #24 (9th stale-rebase). Student implementation textbook-correct (re-zero after `apply(_init_weights)` to defend against trunc_normal). Reassigned to PR #29 (slice-bottleneck decoder using `PhysicsAttention` — O(N·G·D) complexity matches trunk iter-speed).
  - **Insight:** zero-init residual salvages the "fresh head can't catch trunk" failure mode at init, but requires the refiner to not exceed the compute budget. Vanilla cross-attention on N=150K+ nodes is structurally too expensive; slice-bottleneck PhysicsAttention is the principled fix.

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

**Current baseline: 67.186 val (best seed) / 68.687 (3-seed mean) — test 58.358 / 60.680** (PR #27, sn=32 + σ=0.7 + SwiGLU + L1 + AMP + grad_accum=4)

The recipe has now compounded 6 improvements from ~130+ initial baseline. The sn=32 merge opened new compute headroom (~20 epochs in budget vs 17). Current in-flight experiments exploit this headroom:

**Highest-EV in-flight experiments:**

1. **alphonse #32 (n_head sweep + 3-seed anchor recalibration)** — first sweep of n_head on merged recipe; also recalibrates the noise floor so we have a reliable merge criterion going forward. n_head={2,4,8,16}; current n_head=4 is default, not tuned.

2. **frieren #34 (slice_num lower sweep sn∈{8,16,24})** — continues the monotonically-improving slice_num trend (64→32 gave −3.8% val). If sn=16 or 24 gives further gains, the throughput benefit compounds.

3. **nezuko #35 (n_layers depth sweep {3,4,5,6,7})** — first n_layers exploration on merged recipe. sn=32 freed compute to allow deeper models. n_layers=5 is default and untested on current config.

4. **fern #33 (α-gated per-block Fourier injection)** — ControlNet-style scalar gate init=0; addresses the catastrophic failure mode of PR #30 (shared projector 103.93 vs per-block 108.05; zero-init insufficient without magnitude gating).

5. **thorfinn #29 (slice-bottleneck decoder)** — PhysicsAttention-based surface refinement head with O(N·G·D) complexity; addresses PR #23's budget-bound failure by matching trunk iteration speed.

**Medium-EV:**

6. **tanjiro #17 r3 (gap-jitter σ-scan)** — 9th iteration; gap-jitter signal real but rebasing delays compounded. Focus: σ∈{0.04,0.06,0.08} + tandem-AoA gate on merged recipe.

7. **edward #8 (EMA + grad-clip)** — long-running (r2); clip is dominant lever; EMA+L1 overlap may reduce headroom vs MSE era.

**Untried high-EV directions (next idle student):**
- **Conditional LayerNorm (AdaLN) on log(Re) + domain flags** — strong physics motivation; `val_re_rand` split (65.14 val) is consistently the second-worst. Re changes flow regime fundamentally; conditioning normalization on it is a well-tested approach (AdaGN in diffusion).
- **Horizontal-flip augmentation with Uy sign flip** — doubles effective training set from ~1500 samples. Near-zero implementation cost; strong prior from CFD-ML literature.
- **LR re-sweep on current recipe** — last LR tuning was at ~100+ val; current 67 val may have a different optimal LR.
- **mlp_ratio sweep {2,3,4}** — PR #25 (fern) closed pre-PR-#24 (stale-rebase); mlp_ratio=3 signal was real (73.35 vs 73.66 anchor). Never tested on sn=32+σ=0.7 recipe.

**Still-live hypotheses not yet attempted:**
- **n_hidden scaling (128→192)** — VRAM ceiling is now 31.6 GB at sn=32; headroom exists for wider models.
- **Near-surface volume band weighting** (3-tier loss using dsdf distance).
- **Lookahead/Shampoo optimizer** alternatives to AdamW.

---

## Potential Next Research Directions

### High-priority (next idle students)
- **Conditional LayerNorm / AdaLN on log(Re) + domain flags** — per-sample normalization conditioned on physics parameters. `val_re_rand` (65.14) is structurally the hardest split; Re fundamentally changes the flow regime. ~60 LOC change.
- **Horizontal-flip augmentation with Uy sign-flip** — doubles effective training set from 1499 samples. ~80 LOC. Strong prior from CFD-ML literature. Low risk.
- **LR re-sweep on sn=32 recipe** — LR=5e-4 was set at ~100 val; optimal may differ at 67 val. Sweep {2e-4, 5e-4, 1e-3} × cosine vs WSD schedule. ~10 LOC.
- **mlp_ratio sweep {2,3,4}** on current recipe — PR #25 stale-rebase prevented clean signal; mlp_ratio=3 showed 73.35 vs 73.66 anchor (0.3 pts improvement, pre-σ=0.7 recipe). Needs re-test on merged config.

### Architecture
- **n_hidden scaling (128→192)** — sn=32 freed VRAM (31.6 GB); h=192 at sn=32 might be VRAM-feasible (~40-45 GB est.) and epoch-feasible given faster iteration. Never tested on current recipe.
- **Coordinate-based slice assignment** (MLP on x,z,is_surface) — replaces arbitrary feature clustering with spatial clustering; better OOD geometry inductive bias.
- **Near-surface volume band weighting** (3-tier loss using dsdf[0] distance threshold).

### Physics-informed
- **Kutta condition soft loss** at trailing edge (~150 LOC) — penalizes pressure discontinuity at TE.
- **Pressure-gradient weighted L1** — upweight nodes where |∇p| is large (boundary layer, stagnation point).
- **Panel-method warm-start / pretraining** — high complexity but direct OOD lift potential.

### Speculative
- **Mamba surface decoder** — treat surface of each foil as 1D sequence sorted by saf; Mamba block to refine surface predictions.
- **HyperNetwork on log(Re)** — Re-conditioned fast weights per sample.
- **Hard-example mining** — after epoch 10, oversample worst 20% of val samples by 2× in subsequent epochs.
