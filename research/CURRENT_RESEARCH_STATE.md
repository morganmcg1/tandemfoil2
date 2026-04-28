# SENPAI Research State

- **Updated**: 2026-04-28 20:55 UTC
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 8 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## First review pass (2026-04-28 19:30)

- **PR #782 (edward, GeGLU mlp_ratio=4)** — sent back. Confounded design (FFN ~3x larger than baseline). Re-run will use mlp_ratio≈8/3 to match FFN params. Best val_avg/mae_surf_p 109.89 / offline test_avg ~106.
- **PR #784 (frieren, OneCycleLR epochs=50)** — sent back. Schedule never completed (32/50 epochs at timeout, LR floor never reached). Re-run will use `--epochs 28` to fit the 30-min budget. Best val_avg/mae_surf_p 91.72 / offline test_avg 81.62.
- **Tooling debt**: cruise-split non-finite values in target make `test_avg/mae_surf_p` log as NaN for every run on this branch (one cruise sample has 761 -Inf entries in `p`; the multiplicative mask in `data/scoring.py::accumulate_batch` turns `Inf × 0` into NaN, defeating the intended `y_finite` skip). Fix belongs in `train.py::evaluate_split` (sanitize `pred_orig`/`y` before `accumulate_batch`, or pass a masked version of `mask` — `data/scoring.py` is read-only). **Promoted to PR #821, see below.**

## Second review pass (2026-04-28 20:50)

- **PR #781 (askeladd, slice_num=8)** — **closed**. Two seeds tightly clustered (val 92.50 ± 0.10, test 83.3 ± 0.2 offline-clean), but both hit timeout at epoch 33/50 with val still descending. Pattern matches PR #784 (compound + OneCycleLR, also timeout at epoch 32/50 with val=91.72) and PR #782 (compound + GeGLU mlp_ratio=4, timeout at epoch 12/20 because of bigger FFN). **Diagnosis: at fp32 / batch_size=4 the compound base takes ~55 s/epoch on this hardware, so 50 epochs needs ~46 min — far past the 30-min wall clock.** No round-1 hypothesis is testable cleanly until throughput is fixed. Askeladd also delivered a clean root-cause + fix recipe for the NaN bug.
- **Pivot**: closed sn=8 PR (data preserved in EXPERIMENTS_LOG.md), reassigned askeladd to a tooling PR (#821) that bundles AMP/bf16 + `batch_size=16` + NaN-safe sanitize-before-accumulate-in-evaluate_split. Acceptance: 50 epochs of compound base in <28 min wall clock with finite test_avg/mae_surf_p.

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

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #779 | compound-anchor — bare baseline + nl3/sn16/nh1 compound | anchor / verification | WIP |
| askeladd | ~~#781~~ → **#821** | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval | infrastructure | WIP (pivot from sn=8 ablation; sn=8 closed as undertrained) |
| edward   | #782 | compound + GeGLU activation, mlp_ratio≈8/3 (param-matched re-run) | architecture (gating) | WIP (re-run after first send-back) |
| fern     | #783 | compound + Huber loss δ=1.0 | loss (objective mismatch) | WIP |
| frieren  | #784 | compound + OneCycleLR with 5% warmup, --epochs 28 | optimizer (schedule) | WIP (re-run after first send-back) |
| nezuko   | #785 | compound + n_hidden=192 | architecture (width) | WIP |
| tanjiro  | #786 | compound + RMSNorm | architecture (normalization) | WIP |
| thorfinn | #787 | compound + Fourier feature PE on (x,z) | features (positional) | WIP |

**Note on the timeout bias.** The 6 active hypothesis PRs (#779, #782, #783, #785, #786, #787) and frieren's re-run (#784 with --epochs 28) are still running under the pre-tooling fp32/batch=4 throughput. Their results will be biased by undertraining (~33 of 50 epochs completable). The *relative* ordering between hypotheses at a fixed truncated budget is still informative, but absolute numbers will trail the prior-round reference (~40.9) substantially. Round 2 (after #821 lands) will re-run any candidate hypotheses under proper throughput.

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
