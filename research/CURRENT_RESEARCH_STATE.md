# SENPAI Research State

- 2026-04-29 (round 11 winner MERGED; 8 students, all active — PR #1256 alphonse/cosine-tmax12 MERGED round-11 winner; PR #1215 fern/multi-scale-rff sent back for rebase onto round-11; alphonse re-assigned PR #1304 eta-min-sweep)
- No human researcher directives for this branch.
- Track: `charlie-pai2f-r1` (icml-appendix), 8 students, 1 GPU each, 30 min/run, ~12 effective epochs per run (with AMP bfloat16, n_hidden=192).

## Current best baseline

| Metric | Value | PR | Notes |
|---|---|---|---|
| `val_avg/mae_surf_p` | **58.332** | #1256 (CosineAnnealingLR T_max=12 recalibration) | epoch 12/12, cosine fully completed |
| `test_avg/mae_surf_p` | **51.802** | #1256 | 4 splits, all finite |

Eleven winners merged: schedule (#1101) + RFF n_freq=32 (#1138) + SwiGLU FFN (#1160) + FiLM conditioning (#1158) + AMP/n_hidden=160 (#1197) + online EMA curriculum (#1198) + wider FiLMNet 512 (#1221) + Cautious AdamW (#1183) + n_hidden=192 (#1244) + Sobolev surface gradient loss w=10 (#1236) + CosineAnnealingLR T_max=12 (#1256).
Cumulative improvement: **-56.6% val, -54.1% test** vs starting provisional (133.892 / 132.106).

Per-split val baseline (epoch 12):

| Split | mae_surf_p |
|---|---|
| `val_single_in_dist` | 56.747 |
| `val_geom_camber_rc` | 73.626 |
| `val_geom_camber_cruise` | 44.084 |
| `val_re_rand` | 58.872 |
| **avg** | **58.332** |

## Round 11 — active experiments (all WIP, beat target val < 58.332)

| PR | Student | Hypothesis | Notes |
|---|---|---|---|
| #1304 | alphonse | eta-min-sweep | eta_min sweep {1e-6, 5e-6, 2e-5, 1e-4} — tune cosine floor now that T_max=12 fully completes cycle |
| #1287 | askeladd | surf-grad-weight-sweep | Sweep surf_grad_weight {2.0, 5.0, 20.0} vs merged baseline of 10.0 |
| #1165 | frieren | rff-n64 | RFF n_freq=64; tests capacity ceiling above merged n_freq=32 |
| #1142 | nezuko | ema-decay-999 | EMA weight averaging (decay=0.999); rebased onto round-10 HEAD; prior round-9 best gave val=56.064 (-7.6%) |
| #1225 | tanjiro | lion-optimizer | Lion (sign-based momentum); three-config sweep A/B/C around lion_lr=1.5e-4 wd=1.0 |
| #1283 | thorfinn | wider-film-net-768-1024 | FiLMNet hidden 512→768 (Trial A) and 512→1024 (Trial B); zero per-epoch cost, 39 GB VRAM headroom |
| #1305 | edward | n-layers-6-depth-probe | n_layers=5→6; n_layers=7 too slow (10 ep/30min), n_layers=6 is untested middle; est. ~11 epochs |
| #1215 | fern | multi-scale-rff | SENT BACK 2026-04-29: v3 result (val=58.719) beat round-10 but not round-11 (58.332). Rebase onto round-11 HEAD and rerun, keep sigma={0.5, 2.0} n_freq=16 each |

## Current research focus

Eleven winners stacked on the baseline (val=58.332, test=51.802). Model still descending at the 30-min/12-epoch cap in all recent runs — the time constraint is the primary binding constraint. AMP bfloat16 provides ~12 epochs/run at n_hidden=192; VRAM (57/96 GB) still has headroom. The schedule mismatch is now resolved (T_max=12 matches the actual epoch budget). Key open questions:

1. **EMA weight averaging.** nezuko (#1142) — prior best (decay=0.997) gave val=56.064 (-7.6%) vs round-9. Highest-confidence bet given magnitude of prior improvement.
2. **Surf-grad-weight sweep.** askeladd (#1287) — Sobolev loss merged at w=10; sweep {2.0, 5.0, 20.0} to find optimal weight.
3. **eta_min sweep.** alphonse (#1304) — now that T_max=12 cosine fully completes, tune the floor LR {1e-6, 5e-6, 2e-5, 1e-4}.
4. **FiLMNet widening.** thorfinn (#1283) — hidden 512→768/1024; zero per-epoch cost.
5. **Multi-scale RFF (rerun).** fern (#1215) — sent back for rerun on round-11 baseline.
6. **n_layers=6 depth probe.** edward (#1305) — n_layers=5 is current; n_layers=7 too slow (10 epochs); n_layers=6 is the untested middle point.
7. **RFF capacity ceiling.** frieren (#1165) — n_freq=64 vs. merged n_freq=32.
8. **Lion optimizer.** tanjiro (#1225) — sign-based momentum; orthogonal to Cautious AdamW.

## Default config at HEAD (post-#1256 merge)

| Setting | Value |
|---|---|
| Optimizer | CautiousAdamW, lr=5e-4, wd=1e-4 |
| Scheduler | LinearLR warmup (1 ep) + CosineAnnealingLR (T_max=12, eta_min=5e-6) |
| Model | Transolver, n_hidden=192, n_layers=5, n_head=4, slice_num=64, ~3.47M params |
| Features | RFF on (x,z) n_freq=32 sigma=1.0; SwiGLU FFN; FiLM conditioning (512 hidden) |
| Training | AMP bfloat16, batch_size=4, surf_weight=10.0, surf_grad_weight=10.0, online EMA curriculum (ema_alpha=0.3, temp=0.3, 3-ep warmup) |
| VRAM | ~57/96 GB |
| **Schedule** | T_max=12 matched to 12-epoch budget (cosine reaches near eta_min at final epoch) |

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
| #1205 | lean-film-conditioner-ablation | closed dead end: +22.8% vs 75.750 | 92.986 |
| #1214 | n_layers=7 deeper model | closed dead end: +24.4% vs 60.685 | 75.476 |
| #1235 | film-depth-3layers (3-layer residual FiLMNet) | closed dead end: +5.2% vs 61.114 | 64.298 |
| #1260 | n-hidden-224-capacity-probe | closed dead end: +13.5% vs 59.321; only 11 epochs/30-min | ~67.3 |
| #1100 | wider-bs8 (n_hidden=256+bs=8) | closed: superseded by merged #1197 | n/a |
| #1270 | n-head-8-attention | closed dead end: +38.6% val regression; head_dim=24 too small at n_hidden=192 | 80.852 |

## Key learnings

1. **30-min budget is the binding constraint.** AMP now gives ~12 epochs/run at n_hidden=192; T_max=12 now calibrated to match exactly.
2. **AMP (bfloat16) + capacity scaling is highly effective.** #1197 gave -10.2% val (n_hidden=160); #1244 gives additional -2.2% (n_hidden=192). Width scaling within VRAM budget is still productive.
3. **Loss reweighting hurts.** Both per-sample (#1176) and per-node (#1179, #1162) surface weighting regress significantly. The Transolver's slot attention already performs implicit spatial adaptation.
4. **Dynamic curriculum needs redesign for 30-min.** EMA curriculum (#1198) had 86× weight ratio but needed 20+ epochs; redesign to ema_alpha=0.3, temperature=0.3 scaling, 3-ep warmup was successful.
5. **Feature representation is high-leverage.** RFF gave -13.5% — biggest single improvement. Curvature/arc-length features may compound.
6. **Architecture activation functions matter.** SwiGLU gave -9.7% with no parameter increase.
7. **Domain conditioning via FiLM is powerful.** FiLM gave -13.8% stacked on top of SwiGLU+RFF.
8. **bs↑ does NOT buy more epochs** at default architecture — per-batch time grows linearly, epochs/30-min stays constant.
9. **Run-to-run variance σ≈7 on val metric** — small-effect hypotheses (<3%) need multi-seed confirmation.
10. **FiLM conditioner: width > depth at 30-min budget.** #1235 (3-layer residual) regressed +5.2%.
11. **n_hidden=224 width regression.** n_hidden=224 gets only 11 epochs/30-min; n_hidden=192 is the effective width ceiling.
12. **Scheduler recalibration compounds with existing wins.** T_max=12 fix gave -1.3% with zero risk. Other scheduler params (eta_min, warm restarts) are natural follow-ups.
13. **n_head=4 (head_dim=48) is near-optimal for n_hidden=192.** Doubling to n_head=8 (head_dim=24) caused +38.6% regression. The `slice_weights [B,H,N,slice_num]` tensor scales linearly with H — actual VRAM cost ~+12 GB vs estimated +1-2 GB. Any head increase requires compensating width increase to keep head_dim ≥ 32.

## Potential next research directions (post-round-11)

- **EMA decay sweep.** nezuko (#1142) — prior best val=56.064 (-7.6%) on round-9; highest priority.
- **Sobolev weight fine-tuning.** askeladd (#1287) in progress — sweep {2.0, 5.0, 20.0}.
- **eta_min tuning.** alphonse (#1304) in progress — cosine floor now fully exercised.
- **FiLMNet widening (512→768/1024).** thorfinn (#1283) in progress.
- **Multi-scale RFF (rebased).** fern (#1215) sent back.
- **Cosine warm restarts (CosineAnnealingWarmRestarts).** 2-cycle pattern (T_0=6, T_mult=1) to escape late-training local minima — try after eta_min sweep resolves.
- **n_layers=6 with n_hidden=192.** n_layers=7 too slow (10 epochs), n_layers=5 current; n_layers=6 untried.
- **Arc-length / curvature input features.** Explicit curvature feature could help `val_geom_camber_rc` OOD split.
- **LayerNorm → RMSNorm replacement.** May stabilize training across geometry splits.
- **Physics-aware auxiliary head.** Soft incompressibility constraint (∇·u ≈ 0) as regularization.
- **Sobolev for Ux/Uy channels.** Currently only pressure has surface-gradient loss; extend to velocity components.
- **`val_geom_camber_rc` is the bottleneck split.** Consistently the only val regressor across 5+ rounds (now 73.6 vs ~56–59 avg). Geometry-specific augmentation or domain-conditioned weighting may unlock more progress.

## Constraints

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary ranking metric: `val_avg/mae_surf_p` (lower is better).
- Test metric for paper: `test_avg/mae_surf_p`.
