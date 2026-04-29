# SENPAI Research State

- 2026-04-29 (round 10 active — 8 students, 8 WIP in flight; all students assigned)
- No human researcher directives for this branch.
- Track: `charlie-pai2f-r1` (icml-appendix), 8 students, 1 GPU each, 30 min/run, ~12 effective epochs per run (n_hidden=192 with AMP bfloat16).

## Current best baseline

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **59.321** | #1244 (edward n_hidden=192) | epoch 12/12, still descending at cap |
| `test_avg/mae_surf_p` | **51.915** | #1244 | 4 splits, all finite |

Nine winners merged: schedule (#1101) + RFF n_freq=32 (#1138) + SwiGLU FFN (#1160) + FiLM conditioning (#1158) + AMP/n_hidden=160 (#1197) + online EMA curriculum (#1198) + wider FiLMNet 512 (#1221) + Cautious AdamW (#1183) + n_hidden=192 capacity (#1244).
Cumulative improvement: **-55.7% val, -60.7% test** vs starting provisional (133.892 / 132.106).

Per-split val baseline (epoch 12):

| Split | mae_surf_p |
|---|---|
| `val_single_in_dist` | 60.116 |
| `val_geom_camber_rc` | 71.231 |
| `val_geom_camber_cruise` | 45.380 |
| `val_re_rand` | 60.556 |
| **avg** | **59.321** |

## Round 10 — active experiments (all WIP, beat target val < 59.321)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1142 | nezuko | ema-decay-999 | EMA weight averaging (decay=0.999) for variance reduction |
| #1165 | frieren | rff-n64 | RFF n_freq=64; tests capacity ceiling above merged n_freq=32 |
| #1215 | fern | multi-scale-rff | Sent back: needs rerun on n_hidden=192 baseline; try sigma={0.5,2.0} dual bank |
| #1225 | tanjiro | lion-optimizer | Lion (sign-based momentum); three-config sweep A/B/C around lion_lr=1.5e-4 wd=1.0 |
| #1236 | askeladd | surface-gradient-loss | Penalize ∇p errors along foil surface, not just point values |
| #1256 | alphonse | cosine-schedule-recalibration | T_max recalibration (13→11/12) to fix schedule mismatch at 12 epochs/run |
| #1260 | thorfinn | n-hidden-224-capacity-probe | n_hidden=192→224 scaling; ~4.29M params, ~66-70 GB VRAM |
| #1267 | edward | surf-weight-tuning | Sweep surf_weight={5.0,15.0,20.0} vs baseline 10.0; single scalar, cheap win candidate |

## Current research focus

Nine winners stacked on the baseline (val=59.321, test=51.915). Model still descending at the 30-min/12-epoch cap in all recent runs — the time constraint is the primary binding constraint. Key open questions for round 10:

1. **Schedule mismatch fix?** alphonse (#1256) — T_max=13 was calibrated for 15 epochs but n_hidden=192 runs only 12; recalibrate T_max to match.
2. **Next capacity increment?** thorfinn (#1260) — n_hidden=224 (+16.7% width); VRAM ~66-70 GB (within 96 GB budget).
3. **Optimal surf_weight for current stack?** edward (#1267) — surf_weight=10.0 was never re-evaluated; EMA curriculum changes the effective loss balance.
4. **Multi-scale RFF (correct baseline)?** fern (#1215) — dual RFF bank sigma={0.5,2.0} re-run on n_hidden=192 stack.
5. **Surface gradient loss?** askeladd (#1236) — Sobolev-style ∇p penalty on foil surface.
6. **RFF capacity ceiling?** frieren (#1165) — n_freq=64 vs merged n_freq=32.
7. **EMA weight averaging?** nezuko (#1142) — post-convergence variance reduction.
8. **Lion optimizer?** tanjiro (#1225) — sign-based momentum sweep.

## Default config at HEAD (post-#1244 merge)

| Setting | Value |
|---|---|
| Optimizer | CautiousAdamW, lr=5e-4, wd=1e-4 |
| Scheduler | LinearLR warmup (1 ep, 5e-7→5e-4) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Model | Transolver, n_hidden=192, n_layers=5, n_head=4, slice_num=64, ~3.47M params |
| Features | RFF on (x,z) n_freq=32 sigma=1.0; SwiGLU FFN; FiLM conditioning (512 hidden) |
| Training | AMP bfloat16, batch_size=4, surf_weight=10.0, online EMA curriculum (ema_alpha=0.3, temp=0.3, 3-ep warmup) |
| VRAM | ~57/96 GB |
| Throughput | ~12 epochs per 30-min cap (vs 13 at n_hidden=160) |

## Rejected / closed experiments (summary)

| PR | Hypothesis | Outcome | val_avg |
|---|---|---|---|
| #1092 | capacity-scale-up | closed +5.4% | 141.121 |
| #1094 | surf-weight-25 | closed +12.3% | 150.931 |
| #1095 | pressure-channel-weight | closed +7.8% | 117.0 |
| #1096 | huber-vol | closed +6.9% | 143.1 |
| #1097 | slice-num-128 | closed +29.8% | 162.562 |
| #1099 | lr1e-3-warmup5 | closed +7.0% | 143.313 |
| #1159 | aoa-flip | closed +20.3% | 117.5 |
| #1162 | scale-norm-loss | closed +12.8% | 122.4 |
| #1176 | re-stratified-sampler | closed +1.6% vs old baseline | 110.263 |
| #1179 | gradient-norm-loss | closed +16.4% | 114.008 |
| #1205 | lean-film-conditioner-ablation | closed dead end: +22.8% vs 75.750 | 92.986 |
| #1100 | wider-bs8 (n_hidden=256+bs=8) | closed: superseded by merged #1197 | n/a |

## Key learnings

1. **30-min budget is the binding constraint.** n_hidden=192 gives 12 epochs; model still descending at timeout in all recent runs — more epochs would directly help.
2. **Capacity scaling is highly effective.** Three successful capacity PRs merged: n_hidden=160 (#1197, -10.2%), wider FiLMNet-512 (#1221, -8.3%), n_hidden=192 (#1244, -2.2%). Trend still positive.
3. **Loss reweighting hurts when static.** Both per-sample (#1176) and per-node (#1179, #1162) surface weighting regress significantly. Dynamic EMA curriculum (#1198) works because it adapts.
4. **Feature representation is high-leverage.** RFF gave -13.5% — biggest single improvement. Multi-scale spectral encoding is the natural next step.
5. **Architecture activation functions matter.** SwiGLU gave -9.7% with no parameter increase.
6. **Domain conditioning via FiLM is powerful.** FiLM gave -13.8% stacked on top of SwiGLU+RFF.
7. **Schedule calibration matters.** T_max=13 was set for 15 epochs; now that n_hidden=192 runs only 12 epochs, the cosine schedule has unsampled headroom — alphonse's T_max recalibration targets this directly.
8. **surf_weight=10.0 is untuned for the current stack.** Set before EMA curriculum and n_hidden=192 were introduced; may not be optimal now.

## Potential next research directions (post-round-10)

- **Gradient clipping tuning.** clip_grad_norm=1.0 is fixed; relax to 2.0 or remove and see if CautiousAdamW handles it natively.
- **Arc-length / curvature input features.** Surface geometry features for better boundary representation.
- **Physics-aware auxiliary head.** Soft incompressibility constraint (∇·u ≈ 0) as regularization on volumetric output.
- **Attention head tuning.** n_head=4 fixed; try n_head=6 with n_hidden=192 (divisible) or n_head=8.
- **Deeper model (n_layers=7).** Alphonse's #1214 (older run on n_hidden=160) showed promise; revisit on n_hidden=192 stack.
- **SWA (Stochastic Weight Averaging).** Alternative to EMA decay; SWA averaging over last few epochs.
- **Layer-wise learning rate decay (LLRD).** Lower LR on earlier transformer layers, higher on final layers — standard in CV fine-tuning.

## Constraints

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
- Test metric for paper: `test_avg/mae_surf_p`.
