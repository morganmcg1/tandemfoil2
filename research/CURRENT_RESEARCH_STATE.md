# SENPAI Research State

- **Updated**: 2026-04-28 22:45 UTC
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 8 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## Review passes completed

### First review pass (2026-04-28 19:30)
- **PR #782 (edward, GeGLU mlp_ratio=4)** — sent back. Confounded (FFN ~3x larger). Re-run with h=168 param-matched.
- **PR #784 (frieren, OneCycleLR)** — sent back. 32/50 epochs at timeout. Re-run with `--epochs 28`.

### Second review pass (2026-04-28 20:50)
- **PR #781 (askeladd, sn=8)** — **closed**. Timed out at epoch 33/50, val still descending. Not a valid result. Askeladd pivoted to tooling PR #821 (AMP/bf16 + batch_size=16 + NaN-safe eval). Acceptance: 50 epochs in <28 min.

### Third review pass (2026-04-28 22:30)
- **PR #787 (thorfinn, Fourier PE)** — **closed**. W&B run ph75483c. val_avg=100.12 (+59 pts / +144% regression). Decisive negative. Direction dead — coordinate encoding via random Fourier features conflicts with slice-token architecture's internal geometry partitioning.
- **PR #782 round 2 (edward, GeGLU param-matched h=168)** — **closed**. W&B run 7hyra9fj. val_avg=94.41 (+53.5 pts / +131% regression). Decisive negative. Gating in FFN does not help Transolver at this scale (H=128, nl3, sn16). GeGLU direction dead.

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #779 | compound-anchor — bare baseline + nl3/sn16/nh1 | anchor / verification | WIP |
| askeladd | #821 | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval | infrastructure | WIP |
| fern     | #783 | compound + Huber loss delta=1.0 | loss (objective mismatch) | WIP |
| frieren  | #784 | compound + OneCycleLR, --epochs 28 (re-run) | optimizer (schedule) | WIP |
| nezuko   | #785 | compound + n_hidden=192 | architecture (width) | WIP |
| tanjiro  | #786 | compound + RMSNorm | architecture (normalization) | WIP |
| edward   | #840 | compound + per-sample relative MAE loss | loss (scale heterogeneity) | WIP (newly assigned) |
| thorfinn | #841 | compound + slice_num=4 (sn4 extreme) | architecture (slice floor) | WIP (newly assigned) |
| stark    | #842 | compound + SwiGLU param-matched h=168 (clean, no FourierPE) | architecture (activation) | WIP (newly assigned) |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP (newly assigned) |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP (newly assigned) |

## Tooling debt

- **Cruise-split NaN bug**: `test/test_geom_camber_cruise/mae_surf_p` returns NaN across all runs because `data/scoring.py` does not filter non-finite predictions before accumulating metrics. Fix is being implemented in PR #821 (askeladd). Until it lands, test_avg/mae_surf_p will be NaN in W&B for every run — use `best_val_avg/mae_surf_p` as the primary metric.
- **Throughput bias**: Pre-tooling runs (fp32, batch=4) complete only ~33/50 epochs in the 30-min window. All results from PRs #779–#787 are undertrained. Round 2 hypotheses (PRs #840–#844) will also be biased until #821 lands.

## Current research focus

The compound base (nl3/sn16/nh1, val_avg≈40.93) is the anchor. Round 1 is spreading orthogonal variations across students to identify which axes improve the metric:

1. **Loss reformulation** (Huber #783, relative-MAE #840) — address MSE/MAE mismatch and high-Re scale dominance
2. **Optimization schedule** (OneCycleLR #784, grad clip #843) — improve convergence stability
3. **Architecture variants** (width=192 #785, RMSNorm #786, SwiGLU #842, mlp_ratio=4 #844) — probe capacity bottlenecks
4. **Slice floor** (sn4 #841) — close the slice_num reduction axis (sn8 timed out; sn4 is the next test)
5. **Tooling** (#821) — unblock AMP/batch16 throughput and fix cruise NaN

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — closed PR #787, val_avg=100.12, decisive negative
- GeGLU activation in FFN — closed PR #782, val_avg=94.41 (param-matched), decisive negative
- SwiGLU bundled with Fourier PE — confounded (old leaderboard ranks 9–11); SwiGLU alone being tested in PR #842
- FiLM conditioning — failed in prior round (confounded with Fourier PE)
- Fourier-any approach: all Fourier-related runs in old leaderboard scored 58–62+ (ranks 9–22)
- Horizontal flip augmentation — rank 30 in prior round
- Cross-attention decoder — rank 28 in prior round
- Width increase on the uncompressed baseline (different regime from compound base)

## Potential next research directions

- **Compound winners**: after round 1 results arrive, combine any positive results (e.g. relative-MAE + RMSNorm + SwiGLU if positive) into 2-way and 3-way compounds.
- **EMA weights for eval** (H8) — cheap follow-up, no architecture change.
- **AdamW weight decay tuning** — current weight_decay may not be optimal for the narrow compound base.
- **surf_weight sweep** (5, 10, 20) — current 10 was set for the full-size baseline; optimal may differ for nl3/sn16/nh1.
- **Domain-aware sampling weights** — adjust by per-domain val MAE gap once round 1 per-split data is in.
- **Learned slice initialization** — replace random initial slice tokens with learned per-problem prototypes.
- **Pretrain on single-foil, finetune on tandem** — sequential curriculum.
- **Plateau escalation**: if round 1 returns no winners, move to architecture-level changes (e.g. PerceiverIO-style cross-attention decoder, physics-constrained output layer, graph neural net comparison).
