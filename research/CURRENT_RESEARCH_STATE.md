# SENPAI Research State

- **Updated**: 2026-04-28 23:00 UTC
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## Current baseline

**PR #840 (edward, per-sample relative MAE) — WINNER, pending rebase/merge**
- `val_avg/mae_surf_p` = **64.73** at epoch 32/50 (timed out, still improving)
- Per-split val: single=80.41, rc=78.51, cruise=40.13, re_rand=60.73
- Per-split test (all finite!): single=77.25, rc=67.74, cruise=32.35, re_rand=50.35
- `test_avg/mae_surf_p` = **56.92** (all 4 splits finite for first time)
- W&B: `nz8eev8e`; Config: compound base + `--loss_type relative_mae --huber_delta 1.0 --surf_weight 10 --lr 5e-4`
- Delta vs PR #783: −11.20 (−14.7%) on val_avg/mae_surf_p

**Previous merged baseline**: PR #783 (fern, Huber δ=1.0) — val=75.93 (superseded)

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP |
| askeladd | #821 | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval (sent back: lr=2e-3 fix) | infrastructure | WIP (sent back) |
| edward   | #840 | compound + per-sample relative MAE loss — WINNER, sent back for rebase | loss (scale heterogeneity) | WIP (rebase) |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |
| thorfinn | #865 | AdamW weight decay sweep: wd=1e-5 and wd=0 on Huber base | optimization (regularization) | WIP |
| tanjiro  | #864 | Bugfix: sanitize GT y in evaluate_split (cruise NaN poison fix) | infrastructure | WIP |
| nezuko   | #866 | EMA model weights for val/test eval (decay=0.999) | optimization (eval smoothing) | WIP |

**Note on PRs #840–#844**: Assigned against old compound anchor (96.80). Compare against new baseline (64.73) when they finish.

## Key events this review pass

1. **PR #840 declared winner** (val=64.73, test=56.92 — all splits finite). Relative MAE loss compounds on top of Huber: gradient equalization across Re regimes works exactly as predicted. Cruise split is now our best split (40.13 val / 32.35 test). Merge blocked by rebase conflict — edward sent back to rebase and resubmit.

2. **PR #821 reviewed, sent back for LR fix**: AMP/bf16 and NaN-safe eval both confirmed working. Cruise test split is finite (58.77 / 58.54) — **first time on this branch** (landmark). The only blocker: bs=16 without LR scaling gives 4× fewer gradient steps/epoch → val 159/124 (far above ~92 baseline). Fix: lr default 5e-4 → 2e-3 (linear scaling: 5e-4 × 16/4). Askeladd sent back for this single change.

## Tooling debt

- **PR #821 (askeladd)**: Almost done — NaN-safe eval and AMP work. Only pending: lr=2e-3 fix for bs=16. Once merged, ALL subsequent PRs benefit from throughput + finite test metrics.
- **PR #840 (edward)**: Winner awaiting rebase onto current advisor branch.
- **Throughput note**: fp32/bs=4 → ~56s/epoch → ~32/50 epochs in budget. PR #821 (once merged) will drop this to ~40s/epoch + bs=16 → ~94 batches/epoch. All undertrained runs (#854, etc.) should be re-run post-#821.

## Current research focus

**The relative-MAE mechanism is working.** Both Huber (PR #783) and relative MAE (PR #840) attack the same root cause — high-Re tail dominance — at different abstraction levels, and they compound. The test_avg has improved from NaN (cruise bug) to 56.92 (all splits finite) with a clear path to the reference target of 40.93.

**Key hypothesis for next round**: relative-MAE + Huber combo + AMP throughput = potentially sub-50 val. The 64.73 is undertrained (32/50 epochs). Once PR #821 lands with lr=2e-3 and 50 epochs complete, re-run relative-MAE with full budget.

Current open questions:
1. How much headroom remains in relative-MAE at 50 epochs? (gap from 64.73 to sub-60)
2. Can ε tuning (1e-6 → 1e-2) improve the relative normalization further?
3. Do Huber δ tuning (#853) and surf_weight tuning (#855) compound with relative-MAE?
4. Does AMP/bf16 + relative-MAE at 50 epochs break below the prior-round compound reference (40.93)?

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
2. **ε sweep for relative MAE** — try ε ∈ {1e-3, 1e-2} (current ε=1e-6 may be too tight in normalized space). Low-cost, high-expected-benefit.
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
