# SENPAI Research State
- 2026-04-30 00:00
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: FiLM Re conditioning merged (PR #1264, val=74.36, -7.7% vs prev 80.53). SharedFiLMGenerator conditions all 5 TransolverBlocks on standardized log(Re), improving all 4 val splits. Next push: exploit the ~60 GB unused VRAM (capacity: n_hidden=160, n_layers=6), refine FiLM itself (zero-init output, deeper generator), and close out the pct_start/max_lr schedule sweep on the new baseline.

## Current Baseline

**val_avg/mae_surf_p = 74.36** (PR #1264, charliepai2f2-nezuko, SharedFiLMGenerator + BF16 + OneCycleLR(max_lr=1.2e-3, pct_start=0.3) + grad_clip=1.0 + DropPath 0→0.1 + surf_weight=25 + NaN guard + WeightedRandomSampler, epoch 18/18)

Per-split breakdown (PR #1264 — current best):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| single_in_dist     | 82.09 | 71.62 |
| geom_camber_rc     | 85.31 | 79.90 |
| geom_camber_cruise | 55.67 | 46.22 |
| re_rand            | 74.38 | 65.89 |
| **avg**            | **74.36** | **65.91** |

Stack: BF16 AMP (`torch.autocast`, `dtype=torch.bfloat16`) + OneCycleLR(max_lr=1.2e-3, pct_start=0.3, total_steps=steps_per_epoch * budgeted_epochs) + grad_clip=1.0 + DropPath 0→0.1 + surf_weight=25 + wd=1e-4 + NaN guard + WeightedRandomSampler + SharedFiLMGenerator(log_re → γ/β for all 5 blocks). Model config unchanged: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2.

Throughput: ~105 s/epoch (BF16 + FiLM), 18 epochs in 30 min. Peak GPU memory: 35.44 GB on a 96 GB RTX PRO 6000 Blackwell (~60 GB headroom). Params: 827,735 (SharedFiLMGenerator: Linear(1,128)+Linear(128,1280) = +165K). Schedule anneals to lr=1.66e-05 by epoch 18.

## Key Insights

**1. Throughput dominates capacity in the 30-min budget.** BF16 AMP (PR #1184) was a -11.4% gain purely from more gradient steps. mlp_ratio=3 (PR #1206) and FiLM (PR #1199) and n_hidden=160 attempts all fail when they shave epochs. Rule of thumb: any architectural change that slows the per-epoch wall clock by >20% is fatal unless paired with a throughput recovery (BF16, channels-last, fused ops, gradient accumulation).

**2. Schedule shape matters as much as schedule type.** OneCycleLR (PR #1195) gave an interim best 86.40 over CosineAnnealingLR-budget-aware (89.00). Combining BF16 + OneCycleLR (PR #1211) gave 80.53 — both effects compounded. Open follow-ups: pct_start {0.2} in-flight (PR #1265); max_lr {1.0e-3} in-flight (PR #1278).

**3. The cruise OOD split is now the easiest, rc+single are the hardest.** Surface pressure MAE with FiLM: cruise=55.67 << re_rand=74.38 < single=82.09 < rc=85.31. The hardest splits are still the raceCar-dominated geom_camber_rc and single_in_dist — high-Re, ground effect, negative loading. FiLM improved all splits but the relative ordering is unchanged.

**4. lr=1e-3 + grad_clip=1.0 is the foundational stable-fast-LR combo (MERGED PR #1098).** It enabled every subsequent gain. Lower LR → underfitting; higher LR without clip → divergence on high-Re samples. With OneCycleLR, the equivalent peak is max_lr=1.2e-3.

**5. Per-sample instance normalization fails catastrophically (PR #1129, closed).** Per-channel std down-weights pressure (largest std channel) → 3× regression. Physical unit scales must be anchored globally — the global stats from `data/stats.json` are correct.

**6. DropPath 0→0.1 is optimal; 0→0.2 underfits (PR #1156 closed).** Higher drop rates need more epochs to converge. Anything that reduces effective gradient steps in the 30-min regime is suspect.

**7. wd=1e-4 confirmed optimal (PR #1178 closed); surf_weight=25 confirmed optimal (PR #1182 closed).** Increasing either over-regularized or unbalanced the loss. DropPath provides sufficient regularization.

**8. Re noise augmentation is a closed direction (PR #1204, PR #1261, both closed).** Two confirmed negatives: per-node noise (PR #1204) and per-sample corrected noise (PR #1261) both regressed ~8-10 points. log(Re) is the strongest label-correlated feature; adding noise introduces systematic bias. FiLM conditioning (PR #1264) is the correct way to handle Re variation.

**9. SWA / checkpoint averaging requires diverse checkpoints (PR #1203 closed).** All averaged checkpoints fell in the cosine eta_min tail → averaging produced essentially the same parameter point. Not high priority on the current stack.

**10. FiLM Re conditioning works (PR #1264, merged).** SharedFiLMGenerator MLP(1→128→1280) conditions all 5 TransolverBlocks on standardized log(Re). Applied post-norm: `fx = fx * (1 + γ) + β`. -7.7% vs PR #1211 baseline. All 4 splits improved uniformly. Epoch 1 val=254.4 (high), suggesting zero-init of FiLM output Linear would help early convergence. Val still trending down at epoch 18 — more budget or schedule tuning could yield further gains.

**11. OneCycleLR "beneficial clamping tail" insight (from PR #1262 analysis).** PR #1211 set total_steps based on overestimated epoch time, so the schedule exhausted before the run ended. PyTorch clamps LR at min_lr (≈5e-9) for the remaining ~4 epochs — this acted as a free min-LR finetune tail and was accidentally beneficial. Calibrating total_steps accurately (PR #1262, 94s estimate) removed this effect and regressed. Deliberately exploiting this: use epoch_estimate=150s → ~12 budgeted epochs → ~6 clamping epochs. Worth testing.

**12. Domain re-weighting via RACECAR_BOOST hurts (PR #1263, closed).** Boosting raceCar 2× starved cruise, spiking geom_camber_rc +13.48 points. The WeightedRandomSampler already balances domains by count; over-boosting creates inverse imbalance. The raceCar bottleneck is physics complexity, not sampling frequency.

## Active Experiments (Round 6 — as of 2026-04-29)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #1324 | charliepai2f2-askeladd | Multi-condition FiLM: add AoA as 2nd conditioning channel alongside Re | WIP |
| #1326 | charliepai2f2-fern | Gated foil-2 geometry augmentation: fix structural contamination from PR #1266 | WIP |
| #1327 | charliepai2f2-tanjiro | OneCycleLR deliberate underestimate: free fine-tune tail via LR clamping | WIP |
| #1335 | charliepai2f2-nezuko | Fourier n_octaves 8→12: more high-freq spatial features on PR#1314 stack | WIP |
| #1337 | charliepai2f2-thorfinn | Deeper FiLM generator: 1→128→128→1280 for richer Re conditioning | WIP |
| #1340 | charliepai2f2-alphonse | Fourier-encode scalar Re for FiLM conditioning (8-D vs 1-D) | WIP |
| #1341 | charliepai2f2-edward | torch.compile + fix OneCycleLR schedule for compiled throughput | WIP |
| #1345 | charliepai2f2-frieren | Per-field output heads: separate Ux/Uy/p projections | WIP |

## Merged Winners (Chronological)

1. **PR #1088** (edward, surf_weight 10→25): 127.67
2. **PR #1091** (nezuko, DropPath 0→0.1 + budget-aware CosineAnnealingLR): 121.89 (-4.5%)
3. **PR #1098** (tanjiro, lr=1e-3 + grad_clip=1.0): 100.41 (-17.6%)
4. **PR #1184** (askeladd, BF16 AMP): 89.00 (-11.4%)
5. **PR #1195** (nezuko, OneCycleLR-only): interim
6. **PR #1211** (nezuko, BF16 + OneCycleLR combined): 80.53 (-9.5%)
7. **PR #1264** (nezuko, Lightweight FiLM Re conditioning): **74.36** (-7.7%) — **current baseline**

## Recently Closed (Round 5)

- **PR #1263** (fern, RACECAR_BOOST=2.0): val=89.46, +11.1% — boosting raceCar 2× starved cruise, spiked geom_camber_rc +13.48. Sampling imbalance is already handled; raceCar bottleneck is physics, not frequency.
- **PR #1262** (edward, 20-epoch budget with 94s estimate): val=81.65, +1.12 — calibrating total_steps accurately removed the beneficial clamping tail from PR #1211. Deliberately overestimating epoch time (150s → ~12 budgeted epochs → ~6 clamping epochs) is the correct follow-up.
- **PR #1261** (alphonse, per-sample Re noise std=0.05): val=89.07, +8.54 — second confirmed negative for Re noise. All splits hurt. FiLM is the correct Re-handling approach.
- **PR #1259** (askeladd, max_lr=1.5e-3): val=84.21, +4.6% — higher peak LR caused warmup instabilities; model entered anneal phase at worse parameter point.

## Older closed dead ends (kept for context)

- PR #1207 (batch_size=8): +26% — bigger batch halved gradient steps
- PR #1206 (mlp_ratio=3): +7% — underfitting from added params slowing throughput
- PR #1204 (Re noise per-node): worse than 100.41 baseline — implementation bug
- PR #1203 (checkpoint averaging K=3): SWA=102.98 — diverse checkpoints needed
- PR #1199 (FiLM without BF16): 92.20 — underfit at 13 epochs (PR #1264 fixed this fairly)
- PR #1152 (eta_min=1e-5): 92.28 — eta_min=1e-6 default better
- PR #1087 slice_num=128, PR #1086 n_hidden=192/256, PR #1129 per-sample loss norm, PR #1089/1161 deeper models, PR #1102 mlp_ratio=4, PR #1156 DropPath 0→0.2, PR #1178 wd=1e-3, PR #1144 BF16 (stale), PR #1185 SGDR, PR #1182 surf_weight=50, PR #1143 combined best (superseded), PR #1126/1166 LR warmup variants

## Potential Next Research Directions (sorted by expected value × cost ratio)

**Tier A — schedule / FiLM refinements (cheap, likely positive)**
1. **FiLM zero-init output**: Initialize `SharedFiLMGenerator`'s output Linear(`nn.init.zeros_`) so γ≈0, β≈0 at init → identity behavior at epoch 1 → better early convergence (epoch 1 val was 254.4 without zero-init). Essentially free; nezuko's own suggestion.
2. **Deliberate LR clamping tail**: epoch_estimate=150s → total_steps ≈ steps_per_epoch × 12 → ~6 clamping epochs at min_lr on top of FiLM stack. Combines the PR #1262 insight with the new baseline.
3. **OneCycle pct_start sweep on FiLM stack**: pct_start=0.2 (tanjiro's #1265 tests without FiLM; should be retested on 74.36 stack after merge).
4. **OneCycle max_lr 1.0e-3 on FiLM stack**: askeladd's #1278 tests without FiLM; should confirm the optimal max_lr on the new stack.

**Tier B — capacity unlocked by BF16 VRAM headroom (~60 GB free)**
5. **n_hidden=160 + n_head=5 on FiLM stack**: 25% wider; ~105s+15% = ~121s/epoch ≈ 14-15 epochs — still 14+ epochs with FiLM overhead. Intermediate between 128 (current) and the previously-failed 192 (FP32).
6. **n_layers=6 on BF16+FiLM stack**: deeper at same width; previously failed at FP32 (PR #1161); worth a single retest now that BF16 VRAM headroom is 60 GB.
7. **Gradient accumulation (effective batch=8)**: keep b=4 micro-batches, accumulate 2; gives larger effective batch without halving steps (avoids PR #1207 failure mode).

**Tier C — FiLM architecture variants**
8. **Deeper FiLM generator**: 2-layer MLP (1→128→128→1280) with residual connection inside the generator; nezuko's suggestion; +16K params; may improve Re extrapolation on geom_camber_rc split.
9. **Per-block FiLM generators**: 5 separate small generators instead of one shared; allows each block to specialize its Re conditioning; ~5× more FiLM params but still tiny relative to model size.
10. **FiLM on volume decoder too**: currently only conditioning TransolverBlocks; extend to the output projection MLP for the volume fields (Ux, Uy, p_vol).

**Tier D — OOD-targeted and structural (if Tier A-C plateau)**
11. **Per-field output heads (frieren #1273 in-flight)**: 3 separate decoders for Ux/Uy/p.
12. **Geometry augmentation (thorfinn #1266 in-flight)**: AoA+NACA perturbations for geom splits.
13. **Fourier positional features**: random Fourier features on (x, z) coords concatenated to input; classic INR/NeRF trick for high-frequency spatial fitting on surface pressure peaks.
14. **Cross-attention surface↔volume stream**: dedicated head letting surface tokens attend to nearby volume tokens (k-NN) for sharper surface predictions.
15. **Curriculum on Re**: start with low-Re samples, ramp to full Re distribution; targets re_rand and rc splits.
