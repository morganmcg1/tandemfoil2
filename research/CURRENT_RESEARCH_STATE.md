# SENPAI Research State

- **Date:** 2026-04-29
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 62.20** (W&B run `sv9ktfk3`, alphonse SwiGLU MLP, merged PR #961, 2026-04-29)
- **test_avg/mae_surf_p = 55.04**
- Per-split val: single_in_dist=74.96, geom_camber_rc=73.39, geom_camber_cruise=42.66, re_rand=57.81
- Beat-threshold for new PRs: **val_avg < 62.20** (will drop to < 57.96 once PR #999 RMSNorm merges after rebase)
- **Note:** PR #983 (alphonse, merged 2026-04-29) switched canonical config to mlp_ratio=1 (0.62M params, 14/14 epochs). All pre-SwiGLU WIP branches (#869, #975, #976) — if any shows improvement over old baseline (79.54) but not new (62.20), it will be sent back to rebase onto SwiGLU+ratio=1 HEAD and re-run. Post-SwiGLU assignments (#999, #1016, #1020, #1021) build directly on current HEAD (ratio=1). Beat-threshold: val_avg < 62.20 (leaderboard low from PR #961 snapshot; pending PR #999 RMSNorm merge will drop this to 57.96).

### Pending merge (both strong wins — mechanical rebases only)
- **PR #999 RMSNorm:** val_avg=57.95 (best), 58.30 (mean 2 seeds), test_avg=51.17/51.51. Sent back for rebase. Once merged, beat-threshold drops to <57.96.
- **PR #1016 bf16 mixed precision:** val_avg=58.49, test_avg=51.50, 14/14 epochs in 26.1 min (vs SwiGLU 12/14 in 30 min), bf16 sanity instrumentation clean. Sent back for rebase. Mechanically orthogonal to RMSNorm — both should compound when merged.
- **Compound prediction (both merged):** val_avg ~56-57 if mechanisms additive (RMSNorm gain primarily on `single_in_dist` from cleaner gradient; bf16 gain primarily from recovered cosine epochs); test_avg ~50.

### Prior bests (for reference)
- val_avg/mae_surf_p = 79.54 (nezuko Re-stratified batch sampling, merged PR #910) — superseded by SwiGLU
- val_avg/mae_surf_p = 81.55 (thorfinn pre-block FiLM, merged PR #909) — superseded by Re-stratify
- val_avg/mae_surf_p = 82.77 (thorfinn FiLM v2-on-l1, merged PR #815) — superseded by pre-block FiLM
- val_avg/mae_surf_p = 92.63 (tanjiro L1 surface MAE, merged PR #761) — superseded by FiLM+L1
- val_avg/mae_surf_p = 103.13 (askeladd Huber surf loss, merged PR #814) — superseded by L1

## Founding baseline (round 1 reference)

- val_avg/mae_surf_p = 122.15 (W&B run `8cvp4x6r`, unmodified Transolver)
- test_avg/mae_surf_p = 130.90 (W&B run `zaqz12qi`, re-eval via #807)
- Round-1 noise band: 122–146 (single seed, 14-epoch budget)
- PR #807 (NaN-safe masked accumulation) merged — all future runs produce finite `test_avg`

## Progress summary

| PR | Title | Outcome | val_avg |
|----|-------|---------|---------|
| #807 | NaN-safe scoring fix | **MERGED** (infra) | — |
| #814 | Huber surface loss (delta=1.0) | **MERGED** | 103.13 |
| #761 | L1 surface MAE loss | **MERGED** | 92.63 |
| #815 | FiLM+L1 (per-block Re conditioning, post-block) | **MERGED** | 82.77 |
| **#909** | **Pre-block FiLM (condition attention input on Re)** | **MERGED** | **81.55** |
| **#910** | **Re-stratified batch sampling** | **MERGED — current best** | **79.54** |
| #748 | Transolver 2x capacity | Closed (under-trained) | 203.16 |
| #762 | Boundary-layer features | Closed (−13.3%) | 138.43 |
| #759 | EMA model weights | Closed (wrong-regime) | 124.51 |
| #847 | Huber delta sweep (0.5, 2.0) | Closed — flat; L1 dominates | 102.97 |
| #751 v2 | Dropout 0.05 + drop_path 0.05 on L1 | Closed — within noise | 93.16 |
| #858 | Focal surface loss gamma=0.5/1.0 on L1 | Closed — γ=1.0 +13.4% worse | 92.13 |
| #884 | RevIN — per-sample y normalization | Closed — structural mismatch (+65%) | 152.64 |
| #750 v2-rebased | LR warmup + cosine on FiLM+L1 | Closed — mechanism baked in (+2.78%) | 85.07 |
| #902 | Volume L1 (mirror surface L1 on vol side) | Closed — gradient rebalancing hurts surf_p (+4.2%) | 96.52 |
| #743 v3 | Channel-weighted L1 [1.0,0.5,2.0] on FiLM+L1 | Closed — mechanism falsified on FiLM+L1 (+1.1%) | 83.69 |
| #924 | Per-channel output heads (3 independent decoders) | Closed — slows convergence, loses 1 epoch to timeout (+5.8% vs current best) | 84.16 |
| #936 | Depth scaling n_layers=7 | Closed — wall-clock incompatible (10/14 epochs, +15.3%) | 91.74 |
| #756 v3 | Fourier Re-encoding on FiLM+Re-stratify stack | Closed — mechanism redundant with FiLM (+3.0%) | 81.96 |
| #934 | Layer-targeted FiLM (last 2 blocks only) | Closed — pruning early-block FiLM removes useful capacity (+2.8%) | 81.74 |
| #937 | Dual FiLM (pre-block + post-block per block) | Closed — Re-axis lever saturated, capacity overlap (+3.5%) | 82.36 |
| #952 | Wider single output head (128→256→3) | Closed — decoder capacity not the lever (+2.5%); two probes now falsified | 81.52 |
| #917 | Re-input noise σ sweep {0.02, 0.05, 0.10} | Closed — mechanism real but small (+0.4% on old baseline); Re-axis saturated by Re-stratify on current stack | 83.08 |
| #962 | EMA model weights (decay=0.999) on FiLM+L1+Re-stratify | Closed — wrong-regime smoothing on non-converged trajectory (+13.1% on old baseline; +44.7% vs new SwiGLU best) | 89.99 |
| #970 | Shared FiLM head — rank-reduction probe | Closed — depth-specialization confirmed; per-block FiLM heads carry information, not redundant capacity (+5.0% on old / +34.3% vs new SwiGLU best); 4th FiLM-redistribution falsification | 83.55 |
| #993 | TTA with vertical flip on SwiGLU stack | Closed — equivariance prerequisite violated by dataset asymmetry (~54% half-domain meshes; single-foil AoA strictly negative; stagger strictly positive); flip impl correct; +114% val | 133.00 |
| #969 | Vertical-flip data augmentation (training-time, domain-gated) | Closed — NACA M not flipped → mesh-vs-metadata contradictions; sign-reversed per-split signal (cruise +18.8% worst, not best); two independent y-symmetry falsifications | 86.68 |
| #927 v2 | Per-channel volume L1 (vol_w_p=2.0) v2 rebased onto SwiGLU | Closed — mechanism absorbed by SwiGLU; vol_p gain shrunk −9.0% → −4.2%; surf_p flipped neutral → +4.77% regression; volume-loss-shape direction declared closed at this architecture | 65.59 |
| #983 | SwiGLU mlp_ratio ablation (ratio=1 vs ratio=2) | **MERGED** — canonical config switched to mlp_ratio=1; gating-mechanism is primary driver (~97% of gain), capacity contribution ~3%; 0.62M params, 14/14 epochs | 62.74 (paired; test=55.04) |
| **#1016** | **bf16 mixed precision training (wall-clock unlock + cosine recovery)** | **PENDING REBASE → MERGE** — strong win val_avg=58.49, test_avg=51.50, 14/14 epochs in 26.1 min, sanity instrumentation clean; orthogonal to RMSNorm | **58.49** (pending) |
| **#999** | **RMSNorm replacing LayerNorm (canonical SwiGLU pairing)** | **PENDING REBASE → MERGE** — strong win val_avg=57.95 best / 58.30 mean (2 seeds), −6.3% val / −6.4% test; Pareto win on params+FLOPs+simplicity | **57.95** (pending) |
| **#961** | **SwiGLU MLP — replace GELU MLP with Swish-gated linear unit** | **MERGED — leaderboard low-water-mark** | **62.20** |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| alphonse | #1020 | **Ultra-thin SwiGLU (mlp_ratio=2/3, intermediate=85, 0.51M params) — paper-efficiency extension** | WIP — new 2026-04-29 (post-SwiGLU) |
| askeladd | #976 | AoA-FiLM: extend FiLM input from 1-d log_Re to 3-d (log_Re, AoA1, AoA2) | **SENT BACK 2026-04-29** — pre-SwiGLU mechanism win (val=78.58 < 79.54 old; cruise −7.5%, γ-norms growing); rebase onto SwiGLU+ratio=1 HEAD, paired v2-aoa vs v2-baseline |
| thorfinn | #999 | **RMSNorm replacing LayerNorm — canonical SwiGLU pairing (LLaMA/Mistral-style normalization)** | **PENDING REBASE → MERGE 2026-04-29** — strong win val_avg=57.95 (best), 58.30 (mean 2 seeds), −6.3% val −6.4% test; sent back for mechanical rebase only |
| nezuko | #1021 | **slice_num sweep {32, 64, 96, 128} — physics-attention spatial resolution ablation** | WIP — new 2026-04-29 (post-SwiGLU) |
| fern | #1029 | **Surface loss reweighting by per-node pressure quantile (within-sample, target-value-gated; sweep top10/α=2, top20/α=2, top10/α=3)** | WIP — new 2026-04-29 (post-SwiGLU) |
| edward | #975 | DropPath rate sweep {0.05, 0.10, 0.15} on FiLM+L1+Re-stratify | WIP — new 2026-04-29 |
| frieren | #1016 | **bf16 mixed precision training — wall-clock unlock + cosine recovery** | **PENDING REBASE → MERGE 2026-04-29** — strong win val_avg=58.49, test_avg=51.50, 14/14 epochs in 26.1 min, sanity instrumentation clean; sent back for mechanical rebase only |
| tanjiro | #869 | surf_weight sweep (sw=5 wins on L1 base); v2 rebase onto FiLM+L1 pending | WIP |

## Cross-cutting findings

- **Timeout is the binding constraint (~14 epochs at 30 min).** All assignments include `--epochs 14` so cosine annealing completes.
- **NaN test poisoning FIXED** via PR #807. All future runs produce finite `test_avg/mae_surf_p`.
- **L1 dominates the loss-shape sensitivity curve.** Full ordering confirmed (PRs #761, #814, #847): L1 (92.63) << Huber(0.5) (102.97) ≈ Huber(1.0) (103.13) < Huber(2.0) (106.78). Big lever is Huber→L1 (−9.9%).
- **FiLM stacks cleanly with L1** (PR #815 v2-on-l1: −10.6%). Orthogonal mechanisms confirmed. FiLM gains biggest on Re-stratified and widest-Re-range splits.
- **Pre-block FiLM marginally better than post-block** (PR #909: −1.5% val vs post-block baseline). Mixed per-split: Re-targeted splits (re_rand, cruise) improved, in-dist/rc slightly regressed. Mechanism: pre-block modulates Q/K/V attention computation (regime-aware attention patterns) vs post-block modulation which only scales outputs.
- **Re-stratified batch sampling stacks with FiLM+pre-block** (PR #910: −2.5% val). Largest surprise: single_in_dist −9.6% val (not re_rand as predicted). Gradient equalizes high-Re bias under L1. `--re_stratify` now defaults to True.
- **`geom_camber_rc` is the hardest split** (92.95 val at current best) — consistently the most resistant to improvement. Potential next target.
- **Per-channel vol-L1 (p only) works on vol_p** (PR #927 v1: −9% val_vol_p / −9.4% test_vol_p on FiLM+L1 baseline). Surf_p flat — mechanism orthogonal to surface improvements. v2 rebase onto current stack pending.
- **Depth scaling wall-clock incompatible** (PR #936: n_layers=7 → 185s/epoch +41%, only 10/14 epochs fit in 30-min timeout, +15.3% regression). Not a capacity verdict; would need bf16 or torch.compile to test fairly.
- **Fourier encoding redundant with FiLM** (PR #756 v3: +3.0% on full stack). FiLM is a strict generalization of fixed-frequency input encoding. The Re-axis lever is saturated architecturally; no benefit from input-side encoding.
- **Channel weighting falsified on FiLM+L1 stack** (PR #743 v3: +1.1% worse). FiLM's hidden-state modulation already captures the per-channel gradient lever. Channel weighting was genuine at Huber stage (−3.8%) but FiLM makes it redundant.
- **FiLM redistribution attempts saturated — including rank-reduction** — four independent FiLM-redistribution probes now falsified (PR #934 last-2 FiLM +2.8%, PR #937 dual FiLM +3.5%, PR #756 Fourier +3.0%, PR #970 shared head +5.0%). **The Re-conditioning lever is architecturally saturated** in *both* directions — neither redistribution nor rank-reduction wins. Per-block FiLM specialization carries information (depth-specialized conditioning); the 5 heads aren't redundant capacity. FiLM-axis work is closed for this round; future probes should be on different axes (multi-variable FiLM input PR #976 is the only remaining FiLM-axis lever).
- **Re-input-noise saturated by Re-stratify** (PR #917 σ-sweep). Mechanism confirmed (val_re_rand −2.1% at σ=0.05) but small absolute effect; Re-stratify already achieves val_re_rand=77.02 < σ=0.05's 77.58. The Re-axis input-side smoothing lever is gone.
- **Decoder capacity is not a lever** — two falsifications: PR #924 per-channel heads (+5.8%) and PR #952 wider single head (+2.5%). All 3 channels regress uniformly, decoder isn't bottlenecked at this depth/width budget. **One preserved signal: `geom_camber_rc` improved on PR #952 (−4.2% val).** Suggests rc-bottleneck is representational, not capacity-uniform — open question for future rc-targeted intervention.
- **Training-time weight smoothing is wrong-regime** for this short-budget schedule (PR #962 EMA decay=0.999 +13.1%). 14-epoch budget never enters a stationary regime; EMA / SWA / warmup variants all fail for the same reason (averaging stale non-converged weights).
- **TTA-by-symmetry equivariance prerequisite violated by dataset asymmetry** (PR #993 +114%). Eval-time vflip averaging fails because the dataset is structurally y-asymmetric: ~54% of training samples are half-domain meshes (y > 0 only), single-foil AoA range strictly negative, stagger range strictly positive. Model has no incentive to learn y-equivariance and goes severely OOD on flipped input. **Flip implementation was correct** (column-by-column verified). New finding documented in research/DATASET_ANALYSIS.md. Has implications for PR #969 (training-time vflip-aug) — may need to subset bilateral-mesh samples or pair with re-meshing.
- **FiLM input axis is single-variable now; AoA-FiLM probe (PR #976) opens multi-variable conditioning** as a fresh axis. AoA is a primary flow parameter the model has zero conditioning-awareness of. If AoA-FiLM wins, opens 4-d (Re, AoA1, AoA2, gap) and beyond.
- **SwiGLU is the new canonical MLP architecture** (PR #961, merged 2026-04-29, val 79.54→62.20 −21.8%). All future experiments build on L1 + FiLM-pre + Re-stratify + SwiGLU. Active pre-SwiGLU WIP branches (#869, #927, #969, #975, #976) — if any beats old baseline (79.54) but not new (62.20), they will be sent back to rebase onto SwiGLU HEAD. Post-SwiGLU assignments (#983, #999, #1016) build directly on SwiGLU HEAD.
- **Off-Re axes underway:** physics-attention resolution (slice_num #1021), parameter-efficiency push (ultra-thin SwiGLU #1020), regularization (DropPath #975), conditioning multi-variable (AoA-FiLM #976), within-sample heavy-tail surface reweight (#1029), normalization canonical (RMSNorm #999), wall-clock infrastructure (bf16 #1016). [SwiGLU paper ablation #983 MERGED, vol-channel #927 CLOSED].
- **Volume-loss-shape direction closed at SwiGLU stack** (PR #927 v2): vol_w_p mechanism shrunk −9.0% → −4.2% post-SwiGLU; surf_p flipped neutral → +4.77% regression. SwiGLU's bilinear gating absorbs the channel-residual-balance lever; explicit channel-rebalancing now competes rather than complements. Cross-stack mechanism reversal — node-level reweight is the next refinement direction (fern #1029).
- **Y-symmetry axis fully closed:** two independent probes both falsified on dataset-asymmetry grounds (TTA frieren #993 — equivariance prerequisite; training-aug nezuko #969 — NACA M hidden asymmetry). Documented in DATASET_ANALYSIS.md. To retry: must also flip NACA M (col 15, 19) AND subset to bilateral-mesh samples.
- **Canonical config update:** mlp_ratio=1 is now the merged default. All 14 epochs fit 30-min budget. 0.62M params (−29% vs ratio=2).
- **Focal loss falsified on L1 base** (PR #858): high-error nodes are convergence-bottlenecked, not gradient-bottlenecked.
- **RevIN structurally mismatched** (PR #884): per-sample loss normalization decouples gradient from absolute-MAE metric.
- **LR warmup mechanism baked into baseline** (PR #750): schedule-budget alignment principle survives as a convention.
- **Full vol-L1 falsified** (PR #902): volume bulk is Gaussian-ish far-field where MSE is theoretically optimal.
- **IMPORTANT:** PRs #756 (Fourier), #869 (surf_weight) need to beat **79.54** now. When sent back, always rebase onto current HEAD.

## Potential next research directions

### Active assignments (2026-04-29)
1. **Surface loss reweighting by per-node pressure quantile** — within-sample heavy-tail-targeted gradient direction. Three-run sweep top10/α=2, top20/α=2, top10/α=3. **Assigned → fern PR #1029.**
2. **AoA-FiLM (multi-variable conditioning)** — extend FiLM input from 1-d (log_Re) to 3-d (log_Re, AoA1, AoA2). Tests whether multi-variable conditioning compounds. **Assigned → askeladd PR #976.**
3. **DropPath rate sweep {0.05, 0.10, 0.15}** on full SwiGLU stack. Regularization axis. **Assigned → edward PR #975.**
4. **RMSNorm replacing LayerNorm** — canonical SwiGLU pairing (LLaMA/Mistral-style). Removes mean-centering; preserves scale (the bilinear-gate-relevant statistic). Modest expected gain (−0.5 to −2%); paper-friendly architectural simplification. **Assigned → thorfinn PR #999.**
5. **bf16 mixed precision training** — wall-clock unlock + cosine recovery; once landed, opens width-scaling and longer schedules. **Assigned → frieren PR #1016.**
6. **Ultra-thin SwiGLU (mlp_ratio=2/3, 0.51M params)** — paper-efficiency extension of gating-mechanism thesis. **Assigned → alphonse PR #1020.**
7. **slice_num sweep {32, 64, 96, 128}** — physics-attention spatial resolution ablation. **Assigned → nezuko PR #1021.**
8. **surf_weight rebalancing** — test surf_weight=5 on FiLM+L1 (tanjiro PR #869 v2 rebase pending).

### Closed directions (this round)
- ~~**SwiGLU MLP**~~ — **MERGED (PR #961, val=62.20 −21.8%). New canonical.** Paper ablation **MERGED (PR #983, ratio=1 canonical).** Gating mechanism is primary driver (~97% of gain).
- ~~**Per-channel volume L1 (vol_w_p=2.0)**~~ — **Closed (PR #927 v2): mechanism absorbed by SwiGLU** (+4.77% surf_p; vol_p gain shrunk −9.0% → −4.2%). Volume-loss-shape direction closed at this architecture.
- ~~**Vertical-flip data augmentation**~~ — **Closed (PR #969): NACA M asymmetry breaks mechanism** (+9.0% val; sign-reversed cruise split). Must also flip NACA M cols 15,19 for self-consistency. Not retrying until higher-EV axes exhausted.
- ~~**TTA with vertical flip**~~ — **Closed (PR #993): equivariance prerequisite violated by dataset asymmetry** (+114% val). ~54% half-domain meshes; single-foil AoA strictly negative; stagger strictly positive. Documented in DATASET_ANALYSIS.md.
- ~~**Shared FiLM head (rank-reduction)**~~ — **Closed (PR #970): depth-specialization confirmed** (+5.0%). FiLM-redistribution axis fully saturated (4 probes).

### Off-the-Re-axis ideas (Re-conditioning lever saturated; pivot to fresh axes)
9. **Horizontal-flip + rotation augmentation** — conditional on a successful y-symmetry breakthrough (currently 0/2 probes). Domain-aware augmentation pipeline; need NACA-M and bilateral-mesh subset machinery first.
10. **rc-targeted intervention** — `geom_camber_rc` improved isolated on PR #952 (−4.2%). Probe: rc-aware loss reweighting OR a small geometry-domain conditioning head. Currently the hardest split (74.60 val on SwiGLU stack).
11. **Slice-token reduction / LRSA** — replace S×S (64×64) slice-token self-attention with rank-16 factored. Reduces compute, possibly improves regularization, frees budget for more width. Conditional on slice_num #1021 result.
12. **Width scaling with bf16** — n_hidden=192 plus mixed precision to fit the wall-clock budget that depth-scaling broke. Conditional on bf16 #1016 landing.
13. **Geometric pre-training / SSL** — pretrain on vflip-augmented data with masked-node reconstruction, fine-tune on pressure prediction.
14. **4-d FiLM (Re, AoA1, AoA2, gap)** — conditional on PR #976 (AoA-FiLM) result. If multi-variable wins, extend to 4-d.
15. **Per-node loss reweighting variants** — conditional on fern #1029 result. If quantile-reweight wins: try non-static weight schedules (warmup-then-up), error-bucketed reweighting, surface-curvature reweighting.
16. **Compound stack frontier** — FiLM-pre + Re-stratify + SwiGLU + best-of post-SwiGLU winners (RMSNorm, AoA-FiLM, DropPath, slice_num, surf-quantile, bf16). Round-4 candidate.
