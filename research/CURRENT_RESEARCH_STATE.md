# SENPAI Research State

- **Updated**: 2026-04-29 06:30 (PR #1048 MERGED — n_hidden=192 new best; reference target gap 7.22 → 3.22 pts; edward freed for next assignment)
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students assigned in PRs, but only 8 pods deployed (alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn). 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap. PRs #842/#844 are zombie assignments — see caveat below.

## Current baseline (MERGED)

**PR #1048 (edward, n_hidden=128 → 192 architecture width) — MERGED 2026-04-29** ← NEW BEST
- `val_avg/mae_surf_p` = **50.61** at best epoch 49 (default seed `ovkjhjyo`)
- Per-split test: single=58.21, rc=57.94, cruise=21.65, re_rand=38.79
- `test_avg/mae_surf_p` = **44.15** (all 4 splits finite)
- Config: compound base widened — `n_hidden=192` (was 128); `loss_type=relative_mae`, `warmup_epochs=5`, lr=2e-3, bs=16, compile=True
- 2-seed corridor: spread val=5.63 / test=4.18 — **halved** from n_hidden=128's 12.58/9.65
- Wall: 30.6 min / 50 epochs (right at the 30 min harness cap; peak VRAM 73.6 GB / 96 GB)
- Param count: 1.24M (was 0.56M; ~2.2×)
- Reference target gap: prior-round PR #32 hit test=40.93 → **3.22 pts away** (was 7.22)

**Prior baseline**: PR #971 (askeladd, LR warmup + relative_mae default) — val=54.70, test=48.15 (superseded)
**Earlier**: PR #821 (askeladd, tooling stack) — val=55.90, test=49.64 (superseded)
**Earlier**: PR #840 (edward, rel MAE) — val=64.16, test=55.73 (superseded)
**Earlier**: PR #783 (fern, Huber δ=1.0) — val=75.93 (superseded)

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP |
| askeladd | #1047 | gradient clipping (max_norm 0.5 / 1.0) × 2 seeds on warmup baseline | optimization (variance) | WIP |
| edward   | (idle) | PR #1048 MERGED — needs new assignment | — | available |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP (zombie — no pod) |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP (zombie — no pod) |
| thorfinn | #865 | AdamW weight decay sweep: wd=1e-5 and wd=0 on Huber base | optimization (regularization) | WIP |
| tanjiro  | #864 | Bugfix: sanitize GT y in evaluate_split (cruise NaN poison fix) | infrastructure | WIP |
| nezuko   | #866 | EMA model weights for val/test eval (decay=0.999) | optimization (eval smoothing) | WIP |

**Note on PRs #853–#866**: Branched against old compound + Huber base. Compare against current baseline (val=50.61 / test=44.15 at n_hidden=192) when they finish — the deltas they should target shifted twice during their lifetime (96.80 → 64.16 → 54.70 → 50.61). Their underlying hypotheses (loss δ tuning, accum, surf weighting, EMA, weight decay) are still valid levers but their measured effect sizes will be relative to the new wider baseline.

**Idle-detection caveat (2026-04-29)**: The entrypoint harness reports "idle" students (alphonse, fern, frieren, nezuko, tanjiro, thorfinn) because it queries `student:willowpai2e2-<name>` while their PRs use the short-form `student:<name>` label. **Do NOT re-assign these students** — verify with `gh pr list --base $ADVISOR_BRANCH` before treating any "idle" report as actionable.

**Zombie-PR caveat (2026-04-29)**: PRs #842 (stark), #844 (charlie) reference students whose pods are NOT deployed in the willow-pai2e-r2 cluster (verified via `kubectl get deployments -l app=senpai`). Only 8 willowpai2e2 student pods exist: alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn. PR #843 (himmel grad-clip) was closed and re-routed to askeladd. PRs #842/#844 (SwiGLU, mlp_ratio=4) remain valid hypotheses for future re-routing. Effective active GPU count: **8**, not 11.

## Key events this review pass

1. **PR #1048 (edward, n_hidden=128 → 192) MERGED** — NEW BEST: val=50.61 / test=44.15 (default seed `ovkjhjyo`, best epoch 49). All four test splits improve. Seed corridor halves (12.58 → 5.63 val), confirming the wider model has a friendlier optimization landscape. Wall climbed 22.4 → 30.6 min — width changes are now budget-bounded; further width must be paired with throughput offsets. Reference target gap closed from 7.22 → **3.22 pts**.

2. **PR #1008 (askeladd, 3rd-seed + warmup sweep) CLOSED — informational.** Confirmed 3-seed corridor (under n_hidden=128) at val-spread=12.58, test-spread=9.65. Confirmed warmup_epochs=5 is the right default. Single-seed screening convention adopted.

3. **PR #940 round 2 (edward, ε=1e-3 + warmup) CLOSED — settled negative.** ε=1e-3 does not compose with warmup. Edward's diagnosis: ε=1e-3 was protecting against early-training instability under the old tooling; under warmup, it just becomes a regularization term that biases toward absolute MAE and erases useful signal.

4. **PR #843 (himmel grad-clip) CLOSED — zombie re-route.** Hypothesis re-routed to askeladd's PR #1047.

5. **PR #971 (askeladd, LR warmup + flip relative_mae default) MERGED** — superseded by #1048; was the new-best at the time of merge: val=54.70/test=48.15 (default seed `1xfcb5h5`).

## Current research focus

**Width landed. Reference target is within reach.** PR #1048 closed the gap to PR #32's prior-round target (test=40.93) from 7.22 → 3.22 pts. The seed corridor halved as a side effect — variance is no longer the top-of-stack blocker. Single-seed screening remains valid (now with a tighter 5.63-pt spread) for hypotheses with default-seed deltas above ~3 val pts.

**Top-of-stack priorities, in order**:

1. **Schedule extension on the wider model.** Best epoch is still 48–49/50 even at 1.24M params. The model is signal-positive at the very end of cosine decay, so any way to unlock more compute (longer cosine budget, bf16 throughput tuning, faster kernels) should yield further wins. **Wall is at 30.6 min vs the 30 min cap** — width-only path is essentially closed; throughput is the gating constraint. This makes grad accum (#854) and any compile/kernel optimization unusually high-value because they directly translate into more epochs at this width.

2. **Compose surviving hypothesis levers (loss / regularization / EMA / surf_weight) on top of n_hidden=192 baseline.** All currently-WIP hypothesis PRs (#853, #854, #855, #864, #865, #866) were branched against the n_hidden=128 baseline. Their hypotheses still hold but they must be re-anchored to val=50.61 / test=44.15. None should be closed pre-emptively; review them on completion against the updated baseline.

3. **Architecture levers we haven't yet tested under wider baseline**: depth (n_layers), heads, mlp_ratio, slice_num, decoder design. SwiGLU and mlp_ratio=4 hypotheses (#842/#844) remain valid for re-routing now that we have an idle GPU.

Current open questions (refreshed):
1. **Can we close the remaining 3.22 pts to test=40.93 with schedule + width**? Best epoch still descending — `--epochs 60` with throughput offset, or longer cosine, is the most direct path.
2. **Does grad accum (#854) actually buy throughput at bs=16 with compile**? If yes, it unlocks `--epochs 60` on n_hidden=192 within the 30 min cap.
3. **Does width × depth trade-off (n_hidden=192, n_layers=2) reclaim throughput at minimal cost**? Tay et al. say width usually wins but worth a single-seed screen on irregular meshes.
4. **n_hidden=256**: ~50 s/epoch projected → blows 30 min cap on 50 epochs but feasible at 30–35. Worth testing whether width-scaling continues monotonically.
5. **Loss / surf_weight / EMA combinations** on the wider model — same hypotheses, different anchor.
6. **Decoder lift (PerceiverIO cross-attn, physics-aware output head)** — if width-scaling hits a hard ceiling.

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
- n_hidden=192 without AMP — throughput-blocked; **PROVEN with AMP via PR #1048 (now baseline)**
- Hard Huber→relative_mae curriculum — PR #900, optimizer-reset stall + feature-bias mismatch
- ε ≠ 1e-6 with warmup — PR #940 round 2, settled negative; ε=1e-3 + warmup regresses single_in_dist
- warmup_epochs=10 — PR #1008, +10.07 val due to under-decay; warmup_epochs=5 is the floor

## Pending new assignments

Edward is freed by the merge of PR #1048; all other deployed students remain WIP. Next priority assignments:

1. **Schedule extension on n_hidden=192**: longer cosine budget — `--epochs 60` if throughput offset is found, or a tighter `T_max` to concentrate decay near the end. Best epoch is 49/50 → schedule, not capacity, is the bottleneck.
2. **Throughput unlock for n_hidden=192**: torch.compile mode tuning, fused kernels, or grad accum at bs=8 (only if it actually speeds up — usually it doesn't). Pairs naturally with (1) — if you can shave 4–5 min off the 50-epoch wall, the 30 min cap permits longer training.
3. **Width × depth trade**: `n_hidden=192, n_layers=2` (param-balanced) — single-seed screen.
4. **n_hidden=256**: projected wall 50–55 min for 50 epochs → must drop to ~30 epochs or relax timeout. Worth a single-seed screen at 30 epochs to check whether width-scaling continues monotonically.
5. **Loss / surf_weight / EMA on the wider baseline**: re-run the in-flight hypotheses (#853, #854, #855, #866) on top of n_hidden=192 if they prove sensitive to baseline width on completion.
6. **Decoder lift**: PerceiverIO cross-attention or a physics-aware output head — held in reserve for when the immediate width/schedule levers are exhausted.

## Potential longer-horizon directions

- **Curriculum learning**: train on single-foil first, then add tandem. Motivated by the split disparity (cruise best with loss reform, single worst).
- **PerceiverIO cross-attention decoder**: if plateau persists at ~60.
- **Physics-constrained output layer**: divergence-free velocity prior for Ux/Uy channels.
- **Graph attention network**: compare against Transolver if plateau persists.
- **Multi-fidelity training**: use lower-resolution CFD samples to pre-train.
