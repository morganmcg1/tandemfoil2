# SENPAI Research State

- 2026-04-29 (round 9 active — 8 students, 7 WIP in flight; edward assigned new experiment PR #1244; all 8 students now assigned)
- No human researcher directives for this branch.
- Track: `charlie-pai2f-r1` (icml-appendix), 8 students, 1 GPU each, 30 min/run, ~13-15 effective epochs per run (with AMP bfloat16).

## Current best baseline

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **60.685** | #1183 (Cautious AdamW) | epoch 13/13, model still descending at cap |
| `test_avg/mae_surf_p` | **52.498** | #1183 | 4 splits, all finite |

Eight winners merged: schedule (#1101) + RFF n_freq=32 (#1138) + SwiGLU FFN (#1160) + FiLM conditioning (#1158) + AMP/n_hidden=160 (#1197) + online EMA curriculum (#1198) + wider FiLMNet 512 (#1221) + Cautious AdamW (#1183).
Cumulative improvement: **-54.5% val, -60.2% test** vs starting provisional (133.892 / 132.106).

Per-split val baseline (epoch 13):

| Split | mae_surf_p |
|---|---|
| `val_single_in_dist` | 62.017 |
| `val_geom_camber_rc` | 70.391 |
| `val_geom_camber_cruise` | 49.179 |
| `val_re_rand` | 61.151 |
| **avg** | **60.685** |

## Round 9 — active experiments (all WIP, beat target val < 60.685)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1244 | edward | n_hidden-192-capacity-probe | n_hidden=160→192; AMP gives VRAM headroom (42/96 GB), ~3.9M params; tests wider capacity benefit |
| #1214 | alphonse | deeper-model-n-layers-7 | n_layers=5→7 on AMP+n_hidden=160 baseline |
| #1215 | fern | multi-scale-rff | Dual RFF bank: sigma=1.0 + sigma=2.0 concat; multi-scale spatial frequency encoding |
| #1236 | askeladd | surface-gradient-loss | Penalize ∇p errors along foil surface, not just point values |
| #1235 | thorfinn | deeper-filmnet-3layer | 3-layer residual FiLM conditioner (512→512→out) |
| #1165 | frieren | rff-n64 | RFF n_freq=64; tests capacity ceiling above merged n_freq=32 |
| #1142 | nezuko | ema-decay-999 | EMA weight averaging (decay=0.999) for variance reduction |
| #1225 | tanjiro | lion-optimizer | Lion (sign-based momentum); three-config sweep A/B/C around lion_lr=1.5e-4 wd=1.0 |

## Current research focus

Eight winners stacked on the baseline (val=60.685, test=52.498). Model still descending at the 30-min/13-epoch cap in all recent runs — the time constraint is the primary binding constraint. AMP bfloat16 provides ~13-15 epochs/30-min; VRAM (42/96 GB) has significant headroom. Key open questions:

1. **Does wider hidden dimension help?** edward (#1244) — n_hidden=192 (+20%) now feasible with AMP VRAM headroom.
2. **Does deeper model help?** alphonse (#1214) — n_layers=7 tests depth vs. width tradeoff.
3. **Multi-scale frequency encoding?** fern (#1215) — dual RFF sigma banks for coarse+fine spatial features.
4. **Physics-aware surface gradient penalty?** askeladd (#1236) — Sobolev-style ∇p loss on foil surface.
5. **Richer FiLM conditioner?** thorfinn (#1235) — 3-layer residual FiLMNet vs. current 1-layer 512-wide.
6. **RFF capacity ceiling?** frieren (#1165) — n_freq=64 vs. merged n_freq=32.
7. **EMA weight averaging?** nezuko (#1142) — post-convergence variance reduction.
8. **Lion optimizer?** tanjiro (#1225) — sign-based momentum; orthogonal to merged Cautious AdamW.

## Default config at HEAD (post-#1183 merge)

| Setting | Value |
|---|---|
| Optimizer | CautiousAdamW, lr=5e-4, wd=1e-4 |
| Scheduler | LinearLR warmup (1 ep) + CosineAnnealingLR (T_max=13, eta_min=5e-6) |
| Model | Transolver, n_hidden=160, n_layers=5, n_head=4, slice_num=64, ~2.70M params |
| Features | RFF on (x,z) n_freq=32 sigma=1.0; SwiGLU FFN; FiLM conditioning (512 hidden) |
| Training | AMP bfloat16, batch_size=4, surf_weight=10.0, online EMA curriculum (ema_alpha=0.3, temp=0.3, 3-ep warmup) |
| VRAM | ~42/96 GB |

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
| #1198 | online-loss-importance-sampling | sent back (29.3% regression vs actual baseline) | 109.125 |
| #1205 | lean-film-conditioner-ablation | closed dead end: +22.8% vs 75.750; nonlinear layer earns its params | 92.986 |
| #1100 | wider-bs8 (n_hidden=256+bs=8) | closed: superseded by merged #1197 (AMP+n_hidden=160) | n/a (never ran) |

## Key learnings

1. **30-min budget is the binding constraint.** AMP now gives ~15 epochs/run; T_max=13 still calibrated to this but model runs longer epochs now.
2. **AMP (bfloat16) + capacity scaling is highly effective.** #1197 gave -10.2% val by running 15 epochs with n_hidden=160 (+53% params, same VRAM).
3. **Loss reweighting hurts.** Both per-sample (#1176) and per-node (#1179, #1162) surface weighting regress significantly. The Transolver's slot attention already performs implicit spatial adaptation.
4. **Dynamic curriculum needs redesign for 30-min.** EMA curriculum (#1198) had 86× weight ratio but needed 20+ epochs; ema_alpha=0.7 too inertial for 13 epochs. Redesign: loss-weighting (not sampler), ema_alpha=0.3, temperature=0.3 scaling, 3-ep warmup.
5. **Feature representation is high-leverage.** RFF gave -13.5% — biggest single improvement. Curvature/arc-length features may compound.
6. **Architecture activation functions matter.** SwiGLU gave -9.7% with no parameter increase.
7. **Domain conditioning via FiLM is powerful.** FiLM gave -13.8% stacked on top of SwiGLU+RFF — confirms per-domain feature modulation helps significantly.
8. **bs↑ does NOT buy more epochs** at default architecture — per-batch time grows linearly, epochs/30-min stays constant.
9. **Run-to-run variance σ≈7 on val metric** — small-effect hypotheses (<3%) need multi-seed confirmation.

## Potential next research directions (post-round-9)

- **Deeper model with AMP.** n_hidden=160 succeeded; try n_hidden=160 + n_layers=7 or 8 next, or n_hidden=192.
- **Multi-scale RFF.** RFF at multiple σ values (e.g., σ=0.5 and σ=2.0 concatenated) — multi-scale frequency basis.
- **Arc-length / curvature input features.** Surface geometry features from student suggestion post-gradient-norm-loss results.
- **Sobolev-style spatial gradient loss.** Penalize wrong pressure gradient vectors ∇p on surface (not just point values).
- **Physics-aware auxiliary head.** Soft incompressibility constraint (∇·u ≈ 0) as regularization.
- **Batch normalization replacement with LayerNorm or RMSNorm.** May stabilize training across geometry splits.

## Constraints

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
- Test metric for paper: `test_avg/mae_surf_p`.
