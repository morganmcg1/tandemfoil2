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

## 2026-04-28 00:10 — PR #330: Round 1 axis: loss formulation — MSE → Huber (β=1)

- Branch: `willowpai2d2-frieren/huber-loss`
- Hypothesis: 2–5 % reduction in `val_avg/mae_surf_p`, with strongest
  gains on `val_re_rand` (high-Re tail story).
- Run: `ic77vvgj` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ic77vvgj)
- W&B run config confirmed: `slice_num = 64` (the **pre-#328
  baseline** at the time the branch was created).

### Results (best checkpoint, epoch 14 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 127.98 | 1.57 | 0.76 |
| val_geom_camber_rc | 126.26 | 2.42 | 0.98 |
| val_geom_camber_cruise | 82.81 | 1.63 | 0.54 |
| val_re_rand | 100.85 | 2.07 | 0.74 |
| **val_avg** | **109.47** | 1.93 | 0.75 |
| test_single_in_dist | 114.05 | 1.52 | 0.72 |
| test_geom_camber_rc | 114.15 | 2.41 | 0.94 |
| test_geom_camber_cruise | NaN ⚠ | 1.62 | 0.49 |
| test_re_rand | 95.55 | 1.94 | 0.70 |
| **test_avg** | NaN ⚠ | 1.87 | 0.71 |

### Conclusion

**Send back for rebase.** The result is the largest single-axis jump
yet (~18 % better than the merged 133.55 baseline) and the per-split
signal supports the hypothesis cleanly: best in cohort on every val
track, including the predicted-strongest signal on `val_re_rand`
(100.85, 17 % better than next-best). **However**, the branch was
created before PR #328 merged and still has `slice_num = 64`. A
direct squash-merge would silently revert the merged slice-128
architectural improvement.

Sent back for: rebase onto current `icml-appendix-willow-pai2d-r2`
(slice_num=128 baseline) and re-run with the same Huber loss change
to confirm the gain stacks on top of slice-128. Even if the rebased
number is slightly worse than 109.47 (because slice-128 already
captured some of the original gain), the hypothesis is supported and
this should land as the next baseline.

### Why such a large gain?

Likely combination of two effects: (a) Huber's linear-tail behavior
is a better proxy than MSE for the L1 metric we're ranked on,
particularly under the `surf_weight=10` multiplier that amplifies
surface-tail residuals; (b) at high Re, normalized residuals exceed
1.0 enough that Huber's gradient clipping prevents tail samples
from dominating the update. Frieren's `val_re_rand=100.85` (best in
cohort) is direct evidence of (b).

## 2026-04-28 00:25 — PR #337: Round 1 axis: batch + LR scaling — BS 4→8, lr 5e-4→7e-4

- Branch: `willowpai2d2-thorfinn/batch-8-lr-7e4`
- Hypothesis: 2–5 % reduction in `val_avg/mae_surf_p` from BS 4→8
  with sqrt-rule LR scaling. The hypothesis was CLI-only — no code
  changes.
- W&B run config confirmed: `slice_num = 64` (the **pre-#328
  baseline** at the time the branch was created — same rebase need
  as frieren #330).

### Results (two runs, same config — seed-variance datapoint)

Both runs used `--batch_size 8 --lr 7e-4` with all other defaults.

| W&B run | best_val_avg/mae_surf_p | best_epoch |
|-|-:|-:|
| `kon60q79` (run 2) | **153.19** | 13 |
| `nphltrz9` (run 1) | **139.39** | 14 |
| **delta** | **+13.80 (~10 %)** | – |

Per-split for `kon60q79` (the primary):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 240.41 | 2.56 | 1.10 |
| val_geom_camber_rc | 141.54 | 2.93 | 1.19 |
| val_geom_camber_cruise | 109.02 | 1.68 | 0.60 |
| val_re_rand | 121.78 | 2.44 | 0.89 |
| **val_avg** | **153.19** | 2.40 | 0.94 |

Test side: `test_geom_camber_cruise/mae_surf_p = NaN` on both runs
(known #367 bug); other test splits finite.

### Critical methodology finding: seed variance ≈ ±10 %

The two thorfinn runs shared **every** configuration knob and command
flag — they differ only in the random seed implied by torch's
default initialization plus DataLoader shuffle order. Spread of
~10 % between them establishes a **noise floor** that affects how
every other round-1 PR's number should be interpreted.

This is a methodological constraint, not a hypothesis result. It
means single-seed differences smaller than ~10 % between runs are
not statistically distinguishable. Concretely, on this branch:

- alphonse 134.13, fern 133.55, nezuko surf-15 137.42, edward 137.83
  are all inside ±10 % of each other and of the merged baseline —
  effectively tied within seed noise.
- Frieren's 109.47 sits well outside the ±10 % band of 133.55, so
  the Huber win is a real signal even at single seed.
- Future PR results in the borderline band (within ±10 %) will need
  multi-seed replication before merging.

### Conclusion on the hypothesis

**Send back** for rebase + push the lever. Best of two runs (139.39)
is ~4.4 % worse than the merged baseline 133.55 — not a clear
regression; 2-run mean (146.29) is ~9.5 % worse — outside noise on
the mean. Hypothesis isn't dead, just under-tested at the lever's
limit. VRAM has clear headroom (84 GB peak / 96 GB cap) for BS=12 or
BS=16, which the original PR didn't explore. Sent back with:

- Rebase onto current advisor (slice_num=128).
- **Primary follow-up:** BS=16 + lr=1e-3 (sqrt-rule scaling from BS=4
  baseline = 5e-4 · √4 = 1e-3).
- **Multi-seed where wall-clock allows** (3 seeds at BS=16/lr=1e-3 if
  budget allows — single seed if not). Add explicit seed via
  `SENPAI_SEED` env var so the runs are deterministic.
- **Fallback** if BS=16 OOMs: BS=12 + lr ≈ 9e-4.

Schedule mismatch (cosine T_max=50 vs 14-epoch achievable budget) is
acknowledged but kept out of scope — tanjiro is iterating on
`--cosine_t_max` in #335.

## Round-1 cohort observation (current snapshot)

| W&B name | best_val_avg/mae_surf_p | best_epoch | status |
|-|-:|-:|-|
| **willow-r2-frieren-huber-b1** | **109.47 ★ candidate** | 14 | sent back (rebase) |
| willow-r2-fern-slice-128 | **133.55 ★** | 11 | merged (PR #328) |
| willow-r2-alphonse-width-192 | 134.13 | 10 | sent back |
| willow-r2-nezuko-surf-15 | 137.42 | 13 | wip (sweep ongoing) |
| willow-r2-edward-mlp-ratio-4 | 137.83 | 11 | sent back |
| willow-r2-thorfinn-bs8-lr7e-4 | 139.39 / 153.19 (2 seeds) | 14 / 13 | sent back (rebase + BS=16) |
| willow-r2-askeladd-depth-8 | 150.06 | 9 | wip |
| willow-r2-tanjiro-warmup-cos-1e3 | 154.57 | 13 | sent back |

**Noise floor:** ±10 % at single seed (thorfinn replicate evidence).
Merged baseline 133.55 has implicit ±13 in either direction at
single-seed precision — winners must beat this band convincingly,
not by 1-3 %. Frieren's 109.47 is the only result outside the band.
