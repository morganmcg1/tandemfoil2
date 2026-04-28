# SENPAI Research Results — willow-pai2e-r4

## 2026-04-28 19:55 — PR #752: L1 absolute-error loss aligned with MAE metric — **MERGED**

- Branch: `willowpai2e4-askeladd/l1-loss`
- Student: willowpai2e4-askeladd
- W&B run: [`8lyryo5g`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/8lyryo5g)

**Hypothesis.** Headline metric is MAE, training loss was MSE. Switching the
per-element loss to L1 should align gradients with the metric and stop
high-Re samples (per-sample y_std up to 2077) from dominating optimization
through MSE's quadratic outlier penalty.

**Implementation.** One-line changes in train.py: `(pred - y_norm) ** 2` →
`(pred - y_norm).abs()` in both the train loop and `evaluate_split`. No
other knobs touched.

**Results (epoch 14, run hit 30 min timeout)**

| Metric | L1 (#752) | Comparable MSE round-1 runs |
|---|---|---|
| `val_avg/mae_surf_p` | **101.93** | 124–162 |
| `test_avg/mae_surf_p` | NaN (cruise bug) | NaN |
| 3-split test mean | 100.83 | — |

| Split | val mae_surf_p |
|---|---|
| `val_single_in_dist` | 133.25 |
| `val_geom_camber_rc` | 109.26 |
| `val_geom_camber_cruise` | 76.13 |
| `val_re_rand` | 89.07 |

**Analysis.** L1 dropped val_avg/mae_surf_p ~33% relative to the next-best
MSE run in the round. The improvement is consistent across all four val
splits (largest absolute reduction on `val_single_in_dist`, the heaviest-tail
split — exactly where MSE's outlier bias would hurt the most). Train/val
loss gap stayed flush (~0.276 vs 0.281), so no overfitting. L1 was still
improving at epoch 14 / timeout — there is more headroom inside the same
budget if other levers shorten epoch time.

**Decision.** Merged as the new baseline. All round 1 PRs that ran on MSE
must be re-tested on top of L1 to know whether their levers compound.

**Open issue surfaced.** `test_geom_camber_cruise/vol_loss = Inf` and
`test_geom_camber_cruise/mae_surf_p = NaN` in this run and every round-1 run
the student inspected. Cause: the model emits Inf on at least one cruise-test
sample's `p` channel; this propagates through scoring (since `Inf * 0 = NaN`
during masked-sum aggregation). Blocks `test_avg/mae_surf_p` reporting until
guarded.

## 2026-04-28 19:55 — PR #758: Higher peak LR (1e-3) with 10% warmup — **SENT BACK**

- Branch: `willowpai2e4-tanjiro/lr-1e-3-warmup`
- Student: willowpai2e4-tanjiro
- W&B run: [`7wplj1pg`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/7wplj1pg)

**Hypothesis.** Default `lr=5e-4` is conservative for a transformer-style
model on small data; `1e-3` with 10% linear warmup should converge faster
without divergence.

**Implementation.** SequentialLR(LinearLR warmup over 5 epochs → CosineAnnealingLR
over 45 epochs); peak lr=1e-3. No other knobs.

**Results (epoch 11 best, hit 30 min timeout at epoch 14)**

| Metric | tanjiro lr=1e-3 (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 151.60 |
| `test_avg/mae_surf_p` | NaN (cruise bug) |
| 3-split test mean | 147.14 |
| Best epoch | 11 / 50 |

**Analysis.** Stable training, no NaN spikes. Best val arrived at epoch 11
(0.22 of cap) — the higher peak LR did front-load gains as predicted.
Better than every other MSE round-1 run (151.60 vs 161.74 / 154.81 / 130.87
/ 124.41) but well behind the merged L1 baseline (101.93). The lever is not
a dead end — it just needs to be tested on top of L1.

**Decision.** Sent back. Asked tanjiro to rebase onto the L1 baseline and
re-run with `lr=1e-3 + 10% warmup + L1`. The two changes are orthogonal
in the codebase (loss tensor vs scheduler), so the rebase should be clean.

## 2026-04-28 20:15 — PR #754: Per-channel loss weight: pressure 3x — **SENT BACK**

- Branch: `willowpai2e4-fern/p-channel-3x`
- Student: willowpai2e4-fern
- W&B run: [`jr8nfzbg`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r4/runs/jr8nfzbg)

**Hypothesis.** Pressure is the only ranked channel; weighting per-element MSE
on the `p` channel 3× should focus gradient there at the cost of (acceptable)
slight Ux/Uy degradation.

**Implementation.** `channel_weights = [1.0, 1.0, 3.0]` multiplied into
`(pred - y_norm) ** 2` in train loop and `evaluate_split`. Stock model and
hyperparameters otherwise.

**Results (epoch 14 best, 30.8 min wall, ran on MSE before L1 merge)**

| Metric | fern channel-3x (MSE) |
|---|---|
| `val_avg/mae_surf_p` | 130.87 |
| `test_avg/mae_surf_p` | NaN (cruise bug) |
| 3-split test mean (W&B) | 130.20 |

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 169.98 | 2.47 | 1.08 |
| `val_geom_camber_rc` | 137.29 | 3.34 | 1.30 |
| `val_geom_camber_cruise` | 100.65 | 1.72 | 0.73 |
| `val_re_rand` | 115.55 | 2.63 | 1.00 |

**Analysis.** Worse than the merged L1 baseline (101.93) by ~28%, but the run
was MSE-based (predates the L1 merge). Velocity channels stayed sane: surface
Ux MAE 1.7-3.3, Uy MAE 0.7-1.3 — no qualitative degradation, so the 3× weight
is not destabilizing the velocity field. Per-epoch trajectory still descending
at the cap (146 → 141 → 130.9), so a longer schedule would help, but the
bigger question is whether channel-3x compounds with L1.

**Decision.** Sent back. Asked fern to rebase on L1 and re-run; net
expression `sq_err = (pred - y_norm).abs() * channel_weights[None, None, :]`.
If this compounds with L1, we'll sweep 5×/10× next round.

**Important diagnostic surfaced.** Fern empirically pinned the
`test_geom_camber_cruise` NaN bug to **two** distinct issues:

1. **Model emits non-finite predictions** (`vol_loss = +Inf` in W&B summary)
2. **GT itself has NaN** in `test_geom_camber_cruise/000020.pt` p channel
   (verified via direct file scan — exactly 1 of 200 cruise test samples)

The second finding is critical: even with the model output guarded, scoring
still propagates `NaN * 0 = NaN` from the bad GT sample through the
masked-sum accumulator. PR #797 (askeladd, NaN guard) has been expanded to
handle both bugs — drop samples with non-finite GT before `accumulate_batch`,
in addition to the original `nan_to_num` on `pred`.

## Round 1 status snapshot (2026-04-28 ~20:15)

| PR | Student | Topic | Status |
|----|---------|-------|--------|
| #749 | alphonse | Capacity scale-up | wip |
| #752 | askeladd | L1 loss | merged (baseline 101.93) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #754 | fern | Per-channel pressure 3× | sent back to retest on L1 |
| #755 | frieren | slice_num 64→128 | wip |
| #757 | nezuko | 5% warmup + cosine | wip |
| #758 | tanjiro | lr=1e-3 + 10% warmup | sent back to retest on L1 |
| #760 | thorfinn | batch_size 4→8 | wip |
| #797 | askeladd | NaN/Inf guard (scope expanded) | wip |

## Round 1 status snapshot (2026-04-28 ~20:00)

| PR | Student | Topic | Status |
|----|---------|-------|--------|
| #749 | alphonse | Capacity scale-up | wip |
| #752 | askeladd | L1 loss | **merged** (new baseline) |
| #753 | edward | surf_weight 20/30/50 | wip |
| #754 | fern | Per-channel pressure 3× | wip |
| #755 | frieren | slice_num 64→128 | wip |
| #757 | nezuko | 5% warmup + cosine | wip |
| #758 | tanjiro | lr=1e-3 + 10% warmup | sent back to retest on L1 |
| #760 | thorfinn | batch_size 4→8 | wip |

Next assignment: askeladd → cruise-test NaN/Inf guard fix (unblocks
`test_avg/mae_surf_p` for all future round-1 reviews).
