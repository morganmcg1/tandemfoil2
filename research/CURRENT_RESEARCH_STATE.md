# SENPAI Research State

- **Updated**: 2026-04-29 (PR #971 MERGED — new best val=54.70/test=48.15 default seed; askeladd → next: 3rd-seed verification)
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students assigned in PRs, but only 8 pods deployed (alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn). 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap. PRs #842/#843/#844 are zombie assignments — see caveat below.

## Current baseline (MERGED)

**PR #971 (askeladd, LR warmup + relative_mae default) — MERGED 2026-04-29** ← NEW BEST
- `val_avg/mae_surf_p` = **54.70** at best epoch 49 (default seed `1xfcb5h5`)
- Per-split test: single=67.22, rc=60.38, cruise=23.79, re_rand=41.20
- `test_avg/mae_surf_p` = **48.15** (all 4 splits finite)
- Config: compound base + new defaults (no flags needed): `loss_type="relative_mae"`, `warmup_epochs=5`, lr=2e-3, bs=16, compile=True
- Schedule: `SequentialLR(LinearLR(start=0.05) for 5ep → CosineAnnealingLR(T_max=45, eta_min=1e-6))`
- Wall: 22.4 min / 50 epochs
- ⚠️ Seed-swap caveat: paired seed42 `9a9di1dz` landed at val=67.28 / test=57.80. Spread narrowed from 27→13 but still material — 3rd seed is the next priority. **Do NOT pin PYTHONHASHSEED=42** when reproducing the new baseline.

**Prior baseline**: PR #821 (askeladd, tooling stack) — val=55.90, test=49.64 (superseded)
**Earlier**: PR #840 (edward, rel MAE) — val=64.16, test=55.73 (superseded)
**Earlier**: PR #783 (fern, Huber δ=1.0) — val=75.93 (superseded)

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP |
| askeladd | #1008 | 3rd-seed (seed7) at warmup=5 + warmup sweep warmup_epochs∈{3,10} | variance (seed budget) | WIP |
| edward   | #940 | Relative MAE ε sweep: ε ∈ {1e-3, 1e-2, 1e-1} vs default 1e-6 | loss (ε tuning) | WIP |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |
| thorfinn | #865 | AdamW weight decay sweep: wd=1e-5 and wd=0 on Huber base | optimization (regularization) | WIP |
| tanjiro  | #864 | Bugfix: sanitize GT y in evaluate_split (cruise NaN poison fix) | infrastructure | WIP |
| nezuko   | #866 | EMA model weights for val/test eval (decay=0.999) | optimization (eval smoothing) | WIP |

**Note on PRs #840–#844**: Assigned against old compound anchor (96.80). Compare against new baseline (64.16) when they finish.

**Idle-detection caveat (2026-04-29)**: The entrypoint harness reports 6 "idle" students (alphonse, fern, frieren, nezuko, tanjiro, thorfinn) because it queries `student:willowpai2e2-<name>` while their PRs (#853, #854, #855, #864, #865, #866) use the short-form `student:<name>` label. **Do NOT re-assign these students** — verify with `gh pr list --base $ADVISOR_BRANCH` before treating any "idle" report as actionable.

**Zombie-PR caveat (2026-04-29)**: PRs #842 (stark), #843 (himmel), #844 (charlie) reference students whose pods are NOT deployed in the willow-pai2e-r2 cluster (verified via `kubectl get deployments -l app=senpai`). Only 8 willowpai2e2 student pods exist: alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn. These 3 PRs have been WIP for ~4.5h with no compute to pick them up. The hypotheses (SwiGLU, grad-clip, mlp_ratio=4) are still scientifically valid and could be re-routed to a deployed student if one becomes idle. Effective active GPU count: **8**, not 11.

**PR #940 (edward, ε sweep)**: ε=1e-6 (default) may over-weight cruise/low-magnitude samples, starving rc/single splits. Testing ε ∈ {1e-3, 1e-2, 1e-1} to soften the small-denominator dominance and recover the 84.10 rc / 77.07 single headroom.

## Key events this review pass

1. **PR #971 (askeladd, LR warmup + flip relative_mae default) MERGED** — NEW BEST: val=54.70/test=48.15 (default seed `1xfcb5h5`, best epoch 49). 2-seed spread narrowed from 27.07 → 12.58, mean improved 69.43 → 60.99 (−8.4 pts). Per-split wins on rc, cruise, re_rand; single regresses ~3 pts. **Seed-swap discovery**: under warmup, default seed is now best (was seed42 in round-3). Askeladd → 3rd-seed verification + warmup-length sweep (next assignment).

2. **PR #821 (askeladd, tooling stack) MERGED** — prior baseline: val=55.90/test=49.64 (seed42). Full tooling stack on advisor branch: AMP/bf16, bs=16, lr=2e-3, torch.compile, NaN-safe eval. Superseded by #971.

3. **PR #840 (relative MAE) MERGED** as earlier baseline (val=64.16, test=55.73 — superseded by #821).

4. **PR #900 (edward, loss curriculum) CLOSED**: Hard Huber→rel-MAE curriculum rejected. 10ep (+0.38 val, +1.73 test) and 20ep (+1.54, +1.95) both regress. Root causes: optimizer-reset stall at switch-over, plus Huber pre-training builds high-Re biased representations. Edward reassigned to ε sweep (#940).

## Current research focus

**Warmup landed. Variance is materially lower but not zero.** PR #971 narrowed the 2-seed spread from 27 → 13 and improved mean by 8.4 pts. Best test=48.15 puts us within 7.2 pts of the reference target 40.93. The seed-swap behavior (default seed best under warmup, opposite of round-3) is real — basin assignment per seed effectively re-randomized under the new schedule. **Single-seed comparisons are no longer reliable** for ranking small architectural deltas.

**Top-of-stack priority is variance reduction.** Until we have ≥ 3 seeds at the new baseline, hypothesis PRs that beat 54.70 by less than ~5 pts on a single seed cannot be confidently ranked. A 3rd seed for the new baseline is the immediate next step (askeladd's pending assignment).

**Secondary push is loss / regularization combinations on top of warmup.** All hypothesis PRs (#853, #854, #855, #864, #865, #866, #940) were branched before warmup landed, so they will collide with the warmup default. They should still work (warmup composes with loss/regularization changes) but their effect sizes will need to be measured against the 54.70 anchor, not the older 55.90 or 64.16 numbers.

Current open questions:
1. Does a 3rd seed at the warmup baseline confirm the 12.58 spread or widen it? (askeladd next assignment)
2. Does longer warmup (3 vs 5 vs 10 epochs) further damp variance, or does 5 already eat too much of the cosine budget?
3. Does ε tuning (1e-6 → 1e-2/1e-1) help rc/single at cruise's expense? (PR #940 in-flight, branched pre-warmup)
4. What happens with the full hyperparam stack (Huber δ #853, surf_weight #855, grad clip #843, EMA #866) on top of warmup defaults?
5. Can n_hidden=192 + AMP/bs=16 fit in budget? (VRAM was 49.8 GB; n_hidden 128→192 adds ~40% params; should still fit at 96 GB)
6. Can we break below the prior-round reference of 40.93?

## Settled facts from this round

- **Relative MAE > Huber MSE**: per-sample scale normalization gives −14.7% on top of Huber's −21.6%. Total loss gain from loss reformulation: −36.3% from anchor.
- **Cruise split is the loss-reform beneficiary**: 40.13 val / 32.35 test — best OOD split, confirming the scale-equalization mechanism.
- **All test splits finite after relative-MAE**: The relative loss suppresses extreme pressure predictions that caused the cruise Inf bug.
- **Slice floor at sn=16**: sn=4 (val=98.25) and sn=8 (val=92.5) both regress.
- **Mean-centering is load-bearing**: RMSNorm decisive negative (val=109.17).
- **Loss reformulation beats architecture tweaks**: GeGLU, Fourier PE, RMSNorm all regress. Loss-first principle confirmed.

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — PR #787, val=100.12, decisive negative
- GeGLU activation in FFN — PR #782, val=94.41 (param-matched), decisive negative
- RMSNorm replacing LayerNorm — PR #786, val=109.17, decisive negative
- FiLM conditioning — failed in prior round
- OneCycleLR — PR #784 round 2, val=92.25; gradient-step-limited
- slice_num=4 — PR #841, val=98.25, floor at sn=16
- n_hidden=192 without AMP — throughput-blocked; re-test after #821 lands

## Pending new assignments (all students active)

All 11 students now have active WIP PRs. Next priority assignments (for when students finish):

1. **Relative MAE + full 50-epoch run (post-AMP)** — once PR #821 merges with lr=2e-3, run edward's config for a full 50 epochs. Most likely to push val below 60.
2. **ε sweep for relative MAE** (in-flight as PR #940) — ε ∈ {1e-3, 1e-2, 1e-1}.
3. **Domain-adaptive Huber δ** — different δ per domain (cruise vs rc vs single vs re_rand). Per-domain residual distributions are structurally different (verified by split metrics).
4. **n_hidden=192 + relative-MAE** — width scaling after AMP/bf16 lands.
5. **Cosine annealing LR schedule** — add CosineAnnealingLR on top of relative-MAE base; may squeeze extra performance in final 10 epochs.
6. **Relative MAE + surf_weight tuning** — compound the surf_weight sweep (#855) with the relative MAE loss rather than Huber base.

## Potential longer-horizon directions

- **Curriculum learning**: train on single-foil first, then add tandem. Motivated by the split disparity (cruise best with loss reform, single worst).
- **PerceiverIO cross-attention decoder**: if plateau persists at ~60.
- **Physics-constrained output layer**: divergence-free velocity prior for Ux/Uy channels.
- **Graph attention network**: compare against Transolver if plateau persists.
- **Multi-fidelity training**: use lower-resolution CFD samples to pre-train.
