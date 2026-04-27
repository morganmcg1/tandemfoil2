# SENPAI Research Results

Branch: `icml-appendix-willow-pai2d-r2`. Primary metric:
`val_avg/mae_surf_p` (lower is better). Wall-clock cap:
`SENPAI_TIMEOUT_MINUTES=30` per run.

Note on `test_avg/mae_surf_p`: every round-1 run reports `NaN` on
`test_geom_camber_cruise/mae_surf_p`, which propagates to the
test-average. Root cause confirmed by edward (#326) and fern (#328):
sample 20 of `test_geom_camber_cruise` ground truth has 761 NaN values
in the pressure channel; `accumulate_batch` masks the sample but
`0.0 * NaN = NaN` defeats the masking. **Bug-fix PR #367 assigned to
fern** with the 2-line `nan_to_num` patch — once it lands, every
round-1 run can recompute a finite `test_avg/mae_surf_p` from W&B.

## 2026-04-27 23:30 — PR #311: Round 1 axis: model width — n_hidden 128 → 192

- Branch: `willowpai2d2-alphonse/width-192`
- Hypothesis: 3–7% reduction in `val_avg/mae_surf_p` from
  `n_hidden 128 → 192` (+50% width, ~2.25× params).
- Run: `oahab4iy` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/oahab4iy)

### Results (best checkpoint, epoch 10 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 160.79 | 2.29 | 0.91 |
| val_geom_camber_rc | 148.93 | 3.43 | 1.26 |
| val_geom_camber_cruise | 105.93 | 1.64 | 0.65 |
| val_re_rand | 120.87 | 2.38 | 0.91 |
| **val_avg** | **134.13** | 2.43 | 0.93 |
| test_single_in_dist | 137.18 | 2.14 | 0.85 |
| test_geom_camber_rc | 132.05 | 3.22 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.57 | 0.60 |
| test_re_rand | 121.55 | 2.14 | 0.89 |
| **test_avg** | NaN ⚠ | 2.27 | 0.88 |

### Conclusion

**Send back** for compute-equal follow-up. The wider model is
clearly undertrained at the 30-min cap (10 of 50 epochs reached;
~184s/epoch vs the ~36s/epoch of baseline ⇒ ~5× slower per step,
not the predicted 2–2.25×). Peak GPU memory at 92.89% leaves no
headroom for stacking. Val curve still descending steeply at epoch
10 (258 → 134), so the metric reflects undertraining rather than
the asymptotic capacity of width-192. The 134.13 number is at
the front of the round-1 cohort but not interpretable as a clean
"width helps" signal.

Sent back with: try **width-160** (1.55× params, divisible by 4),
expected 20–25 epochs in budget; optionally a same-PR AMP-only
baseline at width-128 to disentangle precision from architecture.

## 2026-04-27 23:30 — PR #335: Round 1 axis: LR schedule — 5-epoch warmup + cosine, peak 1e-3

- Branch: `willowpai2d2-tanjiro/warmup-cosine-1e3`
- Hypothesis: 3–7% reduction in `val_avg/mae_surf_p` from
  `lr 5e-4 → 1e-3` with 5-epoch linear warmup + cosine decay.
- Run: `ri332d19` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ri332d19)

### Results (best checkpoint, epoch 13 / 14 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 212.25 | 2.11 | 1.01 |
| val_geom_camber_rc | 149.98 | 2.96 | 1.24 |
| val_geom_camber_cruise | 120.32 | 1.67 | 0.62 |
| val_re_rand | 135.73 | 2.26 | 0.95 |
| **val_avg** | **154.57** | 2.27 | 0.95 |
| test_single_in_dist | 178.60 | 2.00 | 0.96 |
| test_geom_camber_rc | 138.07 | 2.86 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.52 | 0.57 |
| test_re_rand | 137.11 | 2.11 | 0.89 |
| **test_avg** | NaN ⚠ | 2.12 | 0.90 |

### Conclusion

**Send back** for schedule-shape iteration. The warmup wiring is
correct (verified from W&B `lr` panel: 1e-4 → 1e-3 over epochs
1–5, then cosine decay engages). But the 30-min cap only allows
14 epochs, so cosine `T_max=50` decays only ~9.5% of its arc —
the schedule is effectively "warmup + flat 1e-3," not the
warmup+cosine the hypothesis was testing. 154.57 sits at the
bottom of the round-1 cohort (133.55–154.57 range), consistent
with a flat high LR overshooting the local optima that lower
LRs can navigate in a short budget.

Sent back with: parametrize `--cosine_t_max` as a CLI flag, run a
small sweep `(lr 7e-4, T_max 18)` and `(lr 1e-3, T_max 15)` on
a shared `--wandb_group "willow-r2-tanjiro-sched-v2"`. Optional
third variant `(lr 8e-4, T_max 18)`.

## 2026-04-27 23:50 — PR #328: Round 1 axis: physics-token count — slice_num 64 → 128 ★ MERGED ★

- Branch: `willowpai2d2-fern/slice-num-128`
- Hypothesis: 2–5% reduction in `val_avg/mae_surf_p`, with a stronger
  effect on `val_geom_camber_*` and `val_re_rand` than on
  `val_single_in_dist`.
- Run: `s1p2qs7l` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/s1p2qs7l)

### Results (best checkpoint, epoch 11 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 164.27 | 1.822 | 0.892 |
| val_geom_camber_rc | 139.35 | 3.023 | 1.156 |
| val_geom_camber_cruise | 104.92 | 1.521 | 0.635 |
| val_re_rand | 125.66 | 2.284 | 0.900 |
| **val_avg** | **133.55** | 2.162 | 0.896 |
| test_single_in_dist | 140.58 | 1.800 | 0.830 |
| test_geom_camber_rc | 131.24 | 3.013 | 1.086 |
| test_geom_camber_cruise | NaN ⚠ | 1.425 | 0.586 |
| test_re_rand | 122.71 | 2.083 | 0.887 |
| **test_avg** | NaN ⚠ | 2.080 | 0.847 |

### Conclusion

**Merged.** This is the round-1 winner — lowest `val_avg/mae_surf_p`
of any finished run (133.55 vs next-best 134.13). Per-split signal
supports the slice-bottleneck hypothesis cleanly: best-of-cohort on
`val_geom_camber_rc` (139.35; ~7% better than next-best), competitive
on the rest. Mild compute overhead (54.5 GB peak, 172s/epoch =
~10–15% slowdown vs slice-64). At epoch 11/50 the val curve is still
descending steeply, so more headroom likely exists. New
`BASELINE.md` anchor point.

Per-split contrast vs other round-1 runs (cohort observation):

| Split | fern slice=128 (this) | alphonse w=192 | edward mlp=4 | nezuko sw=15 |
|-|-:|-:|-:|-:|
| val_single_in_dist | 164.27 | **160.79** | 176.14 | 166.03 |
| val_geom_camber_rc | **139.35** | 148.93 | 154.36 | 151.57 |
| val_geom_camber_cruise | 104.92 | 105.93 | **99.96** | 101.23 |
| val_re_rand | 125.66 | **120.87** | 120.87 | 130.86 |

## 2026-04-27 23:50 — PR #326: Round 1 axis: FFN ratio — mlp_ratio 2 → 4

- Branch: `willowpai2d2-edward/mlp-ratio-4`
- Hypothesis: 3–5% reduction in `val_avg/mae_surf_p` from
  `mlp_ratio 2 → 4` (standard transformer FFN expansion).
- Run: `ywy4j9e4` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ywy4j9e4)

### Results (best checkpoint, epoch 11 / 13 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 176.14 | 2.135 | 0.897 |
| val_geom_camber_rc | 154.36 | 3.045 | 1.185 |
| val_geom_camber_cruise | 99.96 | 1.346 | 0.654 |
| val_re_rand | 120.87 | 2.076 | 0.906 |
| **val_avg** | **137.83** | 2.151 | 0.910 |
| test_single_in_dist | 155.51 | 1.985 | 0.860 |
| test_geom_camber_rc | 143.53 | 2.998 | 1.120 |
| test_geom_camber_cruise | NaN ⚠ | 1.255 | 0.604 |
| test_re_rand | 121.85 | 1.968 | 0.877 |
| **test_avg** | NaN ⚠ | 2.052 | 0.865 |

### Conclusion

**Send back** for `mlp_ratio=3` iteration. 137.83 doesn't beat the
new merged baseline (133.55) — ~3.2% worse. The FFN-axis hypothesis
isn't dead, just budget-confounded: `mlp_ratio=4` is 2.1× slower per
epoch (148s vs ~70s baseline), chopping the budget to 13 of 50
epochs (~26%). At equal epochs the wider FFN might still win, but
under the 30-min cap the cost dominates. Best on
`val_geom_camber_cruise` (99.96, the only round-1 sub-100), so the
expanded FFN does help on the largest meshes.

Sent back with: rebase onto the new baseline (slice_num=128) and
sweep `mlp_ratio={3, 2-control}` on shared `--wandb_group
"willow-r2-edward-ffn-v2"`. Optional SwiGLU at `mlp_ratio=8/3` if
budget allows. SwiGLU saved for a follow-up PR if the simple sweep
shows promise — keep this iteration's scope to FFN ratio.

### Bonus: independent root-cause analysis of `test_geom_camber_cruise` NaN

Edward independently confirmed fern's flag, with full reproduction
(sample 20 of `test_geom_camber_cruise` GT has 761 non-finite values
in the pressure channel) and a 2-line proposed fix
(`torch.nan_to_num` of `err` before the masked sum). Bug-fix PR #367
assigned to fern (now idle after merging #328) using edward's
proposed patch.

## Round-1 cohort observation (current snapshot)

| W&B name | best_val_avg/mae_surf_p | best_epoch | status |
|-|-:|-:|-|
| willow-r2-fern-slice-128 | **133.55 ★** | 11 | merged (PR #328) |
| willow-r2-alphonse-width-192 | 134.13 | 10 | sent back |
| willow-r2-nezuko-surf-15 | 137.42 | 13 | wip (sweep ongoing) |
| willow-r2-edward-mlp-ratio-4 | 137.83 | 11 | sent back |
| willow-r2-thorfinn-bs8-lr7e-4 | 139.39 | 14 | wip |
| willow-r2-askeladd-depth-8 | 150.06 | 9 | wip |
| willow-r2-tanjiro-warmup-cos-1e3 | 154.57 | 13 | sent back |

No unmodified-baseline finished run exists yet. The merged slice-128
run anchors the new baseline at 133.55.
