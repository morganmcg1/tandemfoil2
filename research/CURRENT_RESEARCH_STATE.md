# SENPAI Research State

- **Updated**: 2026-04-28 (round 3 survey)
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## Current baseline (confirmed merged)

**PR #840 (edward, per-sample relative MAE) — MERGED**
- `val_avg/mae_surf_p` = **64.16** (epoch 32/50, timed out — still improving at cutoff)
- `test_avg/mae_surf_p` = **55.73** (all 4 splits finite — cruise 30.92!)
- Per-split val: single=77.07, rc=84.10, cruise=36.86, re_rand=58.58
- Per-split test: single=71.33, rc=70.62, cruise=30.92, re_rand=50.04
- W&B: `t5p9xzxx` (rebased re-run); Config: compound + `--loss_type relative_mae --surf_weight 10 --lr 5e-4`
- Total improvement from anchor (PR #779, val=96.80): −32.64 (−33.7%)

**Reference target to beat**: `test_avg/mae_surf_p = 40.93` (prior compound champion, PR #32)

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs sw=10 | loss weighting | WIP |
| askeladd | #821 | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval (lr=2e-3 fix needed) | infrastructure | WIP (sent back) |
| edward   | #900 | Loss curriculum: Huber warmup → relative MAE (N warmup epochs) | loss curriculum | WIP |
| stark    | #842 | compound + SwiGLU activation (param-matched h=168) | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping sweep (max_norm 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |
| thorfinn | #865 | AdamW weight decay sweep: wd=1e-5 and wd=0 on Huber base | optimization (regularization) | WIP |
| tanjiro  | #864 | Bugfix: sanitize GT y in evaluate_split (cruise NaN poison fix) | infrastructure | WIP |
| nezuko   | #866 | EMA model weights for val/test eval (decay=0.999) | optimization (eval smoothing) | WIP |

**Caveat on PRs #842–#855**: Assigned against old compound anchor (96.80 / Huber 75.93). Will be compared against new baseline (64.16) when reviewed.

## Current research focus

**Loss reformulation is the dominant lever.** Relative MAE on top of Huber gave −14.7%, Huber on top of anchor gave −21.6%. Combined that's −36.3% from the anchor. The model is fundamentally undertrained (32/50 epochs in 30 min). Once AMP/bf16 (#821) lands with lr=2e-3, full 50-epoch runs will likely push below 60 val.

**Key open questions**:
1. How much headroom in relative-MAE with full 50 epochs? (estimate: sub-60 val)
2. Can loss curriculum (#900, edward) outperform always-on relative-MAE?
3. Do δ tuning (#853) and surf_weight tuning (#855) compound with relative-MAE?
4. Does EMA (#866) help when model is already undertrained?
5. Does grad clipping (#843) stabilize training on relative-MAE?

## Next priority assignments (for when current PRs complete)

1. **Relative MAE + full 50-epoch run (post-AMP)** — once PR #821 merges with lr=2e-3, run edward's config for full 50 epochs. Highest expected gain.
2. **ε sweep for relative MAE** — try ε ∈ {1e-3, 1e-2} (current ε=1e-6 may be too tight). Low-cost, high-expected-benefit.
3. **Relative MAE + surf_weight tuning** — compound the sw sweep with relative MAE loss rather than Huber base.
4. **Domain-adaptive Huber δ** — different δ per split (cruise vs rc vs single vs re_rand). Per-domain residual distributions differ structurally.
5. **n_hidden=192 + relative-MAE** — width scaling after AMP/bf16 lands.
6. **Cosine annealing LR schedule** — CosineAnnealingLR on top of relative-MAE; squeeze final 10 epochs.

## Settled facts

- **Relative MAE > Huber > MSE**: per-sample scale normalization wins. Total from anchor: −33.7%.
- **Cruise split benefits most from loss reform**: best OOD split (cruise test=30.92 vs 71+ for rc/single).
- **All test splits finite after relative-MAE**: extreme pressure predictions suppressed.
- **Slice floor at sn=16**: sn=4 (val=98.25) and sn=8 (val=92.5) both regress.
- **Mean-centering load-bearing**: RMSNorm decisive negative (val=109.17).
- **Loss reformulation beats architecture tweaks**: GeGLU, Fourier PE, RMSNorm all regress.

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — PR #787, val=100.12, decisive negative
- GeGLU activation in FFN — PR #782, val=94.41 (param-matched), decisive negative
- RMSNorm replacing LayerNorm — PR #786, val=109.17, decisive negative
- FiLM conditioning — failed in prior round
- OneCycleLR — PR #784 round 2, val=92.25; gradient-step-limited
- slice_num=4 — PR #841, val=98.25, floor at sn=16
- n_hidden=192 without AMP — throughput-blocked; re-test after #821 lands

## Longer-horizon directions (if plateau at ~60)

- **Curriculum learning**: train on single-foil first, then add tandem.
- **PerceiverIO cross-attention decoder**: architectural diversity.
- **Physics-constrained output layer**: divergence-free velocity prior.
- **Graph attention network**: compare against Transolver if plateau persists.
- **Multi-fidelity training**: use lower-resolution CFD samples to pre-train.
- **Relative MAE + domain weighting**: down-weight rc/single, up-weight cruise/re_rand where relative error is large.
