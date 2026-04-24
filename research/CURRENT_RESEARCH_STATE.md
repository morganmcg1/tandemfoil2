# SENPAI Research State

- **Updated:** 2026-04-24 (round 7 — post PR #14 close)
- **Advisor branch:** `kagent_v_students`
- **Research tag:** `kagent-v-students-20260423-2055`
- **W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current Baseline

**val_avg/mae_surf_p = 84.737 / test_avg/mae_surf_p = 75.244** (PR #7, `alphonse/sw1-fr-s1-m160`, W&B run `91z1948k`)
- Per-split val: in_dist=103.90 | camber_rc=94.07 | camber_cruise=61.58 | re_rand=79.40
- Per-split test: in_dist=90.58 | camber_rc=83.39 | camber_cruise=54.37 | re_rand=72.63
- Config: **L1 loss, surf_weight=1, AMP (bf16), grad_accum=4, Fourier PE fixed σ=1 m=160**, bs=4, n_hidden=128, n_layers=5, slice_num=64, lr=5e-4
- Best epoch: 18 (timeout-bounded)

**Key insights binding the current operating point:**
- L1 > MSE (PR #3), surf_weight=1 under L1 (PR #11), AMP+accum=4 unlocks +5 epochs (PR #12)
- **NEW (PR #7, round 6):** Fourier PE (fixed B, σ=1, m=160) gives −4.0% val / −5.6% test vs AMP baseline. Tancik-2020 mechanism confirmed: spatial bandwidth resolves boundary-layer gradients.
- **m-curve is U-shaped:** m=20 (85.39) nearly ties m=160 (84.74); m=40/80 regress. Non-monotonic structure — seed variance (~5 pts on test) partially accounts for this.
- **σ=1 is the robust sweet spot:** σ=10 harmful, σ=1.5 marginal at m=80.
- **FiLM (per-block log(Re) conditioning) consistently net-negative at 17–18 epoch budget.** Dropped.
- **Capacity scaling is epoch-budget bound:** PR #4 and PR #16 both confirm — negative correlation r<−0.95 between n_params and epochs completed at 30-min budget.
- **Seed noise floor on seeded AMP+accum=4 (round-7 measurement):** 2-seed spread at sw=1 is **2.38 val (2.5%), 1.86 test (2.2%)** — much tighter than the ~9% pre-seed floor. Multi-seed mandatory for claims under ~5%.

---

## Student Status

| Student | Status | PR | Branch |
|---------|--------|----|--------|
| frieren  | WIP (r7) | #21: Near-surface volume-band weighting (3-tier loss) | `frieren/near-surface-volume-band` |
| fern     | WIP (r6) | #20: Fourier σ fine-sweep at m=160 + SwiGLU FFN | `fern/fourier-sigma-fine-swiglu` |
| tanjiro  | WIP (r5) | #17: In-distribution input jitter (AoA/logRe/gap) | `tanjiro/input-feature-jitter` |
| nezuko   | WIP (r5) | #6: AMP rebase + fixed WSD stack + cosine@1e-3 control | `nezuko/lr-schedule-sweep` |
| alphonse | WIP (r6) | #19: Fourier m-extension {160, 320, 640} + learnable B + multi-seed | `alphonse/fourier-m-extension-learnable` |
| edward   | WIP (r2) | #8: EMA + grad-clip on L1 | `edward/ema-gradclip-stability` |
| thorfinn | WIP (r5) | #18: Cross-attention surface decoder head | `thorfinn/cross-attn-surface-decoder` |

**Idle students needing new assignments:** none. Zero idle GPUs.

---

## PRs Ready for Review

None at this time.

---

## PRs In Progress (`status:wip`)

| PR | Student | Hypothesis | Target |
|----|---------|-----------|--------|
| #21 | frieren  | Near-surface volume-band 3-tier loss weighting | Beat **84.737** |
| #20 | fern     | Fourier σ fine-sweep + SwiGLU feedforward | Beat **84.737** |
| #19 | alphonse | Fourier m-extension {160, 320, 640} + learnable B | Beat **84.737** |
| #17 | tanjiro  | In-distribution input jitter (AoA/logRe/gap) on AMP + sw=1 | Beat **84.737** |
| #6  | nezuko   | AMP rebase + fixed WSD + cosine@1e-3 control | Beat **84.737** |
| #8  | edward   | EMA 0.999 + wider grad-clip ({1, 5, 10, 50}) on L1 sw=1 | Beat **84.737** |
| #18 | thorfinn | Cross-attention surface decoder head on AMP + sw=1 | Beat **84.737** |

---

## Recent Decisions Log (round 6)

- **2026-04-23 (r6):** Merged PR #7 (alphonse Fourier PE m=160): **new baseline 84.737 val / 75.244 test** (−4.0% val, −5.6% test vs prior 88.268).
- **2026-04-23 (r6):** Closed PR #16 (fern capacity scaling): dead end confirmed second time. Capacity scaling is epoch-budget bound. All scaled-up variants underperformed baseline anchor; peak at h384-l5-s64 = 65.4% worse. VRAM was never the constraint.
- **2026-04-24 (r7):** Closed PR #14 (frieren sw>1 at eff_bs=16). sw=2's round-4 win compressed from −11.8% at eff_bs=8 to −1.0% sub-1σ at eff_bs=16 — **grad-accum-specific effect, not loss-specific**. sw direction exhausted across 2 rounds. Noise floor on seeded AMP+accum=4: **2.5% val, 2.2% test** (multi-seed threshold for significance). Branch was pre-Fourier — any post-Fourier sw sweep would be a new hypothesis.
- **2026-04-24 (r7):** Assigned frieren PR #21 (near-surface volume-band weighting) — 3-tier loss using dsdf to define BL band; orthogonal to all current improvements.
- **Prior rounds:** See EXPERIMENTS_LOG.md for full history.

---

## Most Recent Research Direction from Human Team

No human issues received as of 2026-04-23 (checked round 6).

---

## Current Research Focus and Themes

**New operating point:** L1 + sw=1 + AMP (bf16) + grad_accum=4 + **Fourier PE fixed σ=1 m=160**. This is a 4-component recipe that compounds across rounds.

**Primary open questions to drive next experiments:**
1. **Can Fourier m be extended further?** m=160 wins but m=20 nearly ties. Is there a true saturation point above m=160? Or is the U-shaped curve an artefact of single-seed variance? m=320 or a multi-seed test of m=160 vs m=20 is the highest-priority diagnostic.
2. **Can Fourier features be made learnable at higher budget?** learnable B underperformed in early rounds at 12-13 epochs. At 18-19 epochs under AMP, learnable B has had more time. This is the next mechanism to probe.
3. **Does σ have a meaningful optimum below 1?** σ=0.5 nearly tied σ=1 in round 3b (93.85 vs 93.59 at the time). With the new baseline recipe, σ ∈ {0.3, 0.5, 0.7, 1.0} is worth a targeted sweep.
4. **Are there complementary architectural improvements?** Fourier PE only touches the input encoding. Architecture improvements to the attention mechanism (temperature, slice assignment) or output head (surface decoder) are orthogonal.
5. **Can σ/m be conditioned on the coordinate scale?** If σ is the only knob, a per-coordinate learnable scale might outperform a global σ.

---

## Potential Next Research Directions (Round 6+)

### Immediate high-EV (for fern and alphonse)

- **Fourier m extension + learnable B comparison (alphonse):** m ∈ {160, 320, 640} on fixed B + learnable B at m=160 + multi-seed test. Answer: where does m saturate and does learnable B beat fixed at 19 epochs? Use `--wandb_group alphonse/fourier-m-ext`.

- **Fourier σ fine-sweep at m=160 + coordinate-specific σ (fern):** σ ∈ {0.3, 0.5, 0.7, 1.0, 1.3} at m=160; also test σ_x ≠ σ_z (separate bandwidths for chord vs thickness direction). Hypothesis: chord (x) needs lower σ (smoother), thickness (z) needs higher (sharper BL structure). Use `--wandb_group fern/fourier-sigma-fine`.

- **Attention temperature annealing (fern or alphonse):** Anneal `self.temperature` from T₀=1.5 → 0.5 over first 30% of training via `--temp_warmup`. ~10 LOC. Inspired by how soft attention early in training lets the model explore slice assignments before hardening. Low-cost, orthogonal.

- **SwiGLU feedforward in TransolverBlock:** Replace standard MLP with SwiGLU. ~30 LOC. Well-documented wins in transformers; orthogonal to Fourier/loss changes.

### Mid-term architectural
- **Near-surface volume-band weighting:** 3-tier loss (far-vol / near-vol / surf) — BL nodes get extra gradient. dsdf features already encode proximity-to-surface; use them to define the near-surface band mask.
- **Learnable slice assignment (coordinate-based):** Replace random/learned slice tokens with coordinate-projected queries using (x, z, is_surface). More interpretable spatial partition; may improve OOD.
- **Sample-wise per-sample normalization or re-centering:** Addresses 10× per-sample y_std variance. Keep global mean/std as baseline but add a per-sample re-centering step for the surface pressure channel.
- **Kutta condition soft enforcement:** aux loss on |p_upper - p_lower| at trailing edge node. Physics-informed, ~150 LOC, directly targets the physical constraint the model is implicitly learning.

### Compounding plan
If PRs #14 (frieren sw>1), #17 (tanjiro jitter), #6 (nezuko WSD), #8 (edward EMA), or #18 (thorfinn cross-attn) merge, update the baseline and assign the next-round students to compound with the accumulated improvements.
