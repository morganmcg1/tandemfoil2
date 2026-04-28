# SENPAI Research State

- **Updated**: 2026-04-28 19:30 UTC
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 8 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## First review pass (2026-04-28 19:30)

- **PR #782 (edward, GeGLU mlp_ratio=4)** — sent back. Confounded design (FFN ~3x larger than baseline). Re-run will use mlp_ratio≈8/3 to match FFN params. Best val_avg/mae_surf_p 109.89 / offline test_avg ~106.
- **PR #784 (frieren, OneCycleLR epochs=50)** — sent back. Schedule never completed (32/50 epochs at timeout, LR floor never reached). Re-run will use `--epochs 28` to fit the 30-min budget. Best val_avg/mae_surf_p 91.72 / offline test_avg 81.62.
- **Tooling debt**: cruise-split `+Inf` in target makes `test_avg/mae_surf_p` log as NaN for every run on this branch. Need to cherry-pick the `nan_to_num` guard from PR #797 (r4 branch) into `evaluate_split`. Until then students must compute clean test_avg offline.

## Current research focus

Fresh advisor branch — no PRs merged yet. The plan for round 1 is to **stress-test
the prior round's compound winner** on this fresh branch, while spreading
mostly-orthogonal additive variations across the 8 students. The compound
(`n_layers=3, slice_num=16, n_head=1` on `n_hidden=128, mlp_ratio=2`) was the
prior round's best config (`test_avg/mae_surf_p = 40.927` per
`target/README.md`). One student (alphonse) is the anchor — they run the
default baseline AND the compound to give us a reproducible reference.
The other seven each layer one independent variation on top of the compound.

## Round 1 assignments (8 WIP PRs)

| Student | PR | Hypothesis | Axis | Predicted delta |
|---------|----|-----------|------|-----------------|
| alphonse | #779 | compound-anchor — bare baseline + nl3/sn16/nh1 compound | anchor / verification | matches prior ~40.9 |
| askeladd | #781 | compound + slice_num=8 | architecture (slice floor) | uncertain (sn8 was rank 2 in prior round) |
| edward   | #782 | compound + GeGLU activation, mlp_ratio=4 | architecture (gating) | −2 to −5 |
| fern     | #783 | compound + Huber loss δ=1.0 | loss (objective mismatch) | −2 to −6 |
| frieren  | #784 | compound + OneCycleLR with 5% warmup | optimizer (schedule) | −1 to −4 |
| nezuko   | #785 | compound + n_hidden=192 | architecture (width) | −2 to −5 |
| tanjiro  | #786 | compound + RMSNorm | architecture (normalization) | −1 to −3 |
| thorfinn | #787 | compound + Fourier feature PE on (x,z) | features (positional) | high variance — orthogonal axis |

## Potential next research directions

- **Round 2 — combine winners** of round 1 in 2× and 3× compounds (e.g. compound + GeGLU + RMSNorm + OneCycle if all individually positive).
- **Loss reformulations**: relative MAE per-sample (researcher H9), channel-balanced loss, log-cosh, log-magnitude weighting for high-Re samples.
- **Conditioning on Re/AoA**: FiLM blocks injecting Reynolds + AoA as global modulation (was a failure with sigma1.0 Fourier in prior round, but FiLM-only has not been cleanly tested on this base).
- **Slice-token architecture variants**: PerCN cross-attention to slice tokens as a decoder, learned slice-token initialization, dynamic slice_num.
- **Capacity sweep at compound**: n_hidden × n_layers grid (e.g. nl4/n_hidden=192 vs nl5/n_hidden=160) once we know whether width or depth helps.
- **EMA weights for evaluation** (researcher H8) — cheap follow-up.
- **Gradient clipping** (researcher H10) — if any training runs show unstable loss curves.
- **Domain-aware sampling weights**: adjust sample weights based on val performance gap per domain (start once we have round 1 data).
- **Pretrain on single-foil, finetune on tandem** — sequential curriculum exploiting easy → hard subset structure.

## Risks and watch items

- The compound base assumption is unverified on this branch; alphonse's anchor is the test. If the compound doesn't reproduce, round 2 will need to recompute baseline before further compounding.
- High-Re samples drive ~12× larger residuals than low-Re — Huber and relative-MAE hypotheses target this directly.
- 30-min wall clock + 50 epochs may not always complete; watch elapsed-time logs in PR comments.
