# SENPAI Research State

- **Date:** 2026-04-29
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current best (live)

- **val_avg/mae_surf_p = 57.9550** (W&B run `6krvx540`, thorfinn RMSNorm v2, merged PR #999, 2026-04-29)
- **test_avg/mae_surf_p = 51.1735** (2-seed mean: val=58.30, test=51.51)
- Per-split val: single_in_dist=61.93, geom_camber_rc=72.84, geom_camber_cruise=40.39, re_rand=56.66
- Beat-threshold for new PRs: **val_avg < 57.9550**
- **Note:** PR #999 (thorfinn, merged 2026-04-29) added RMSNorm as canonical normalization. Current canonical stack: SwiGLU(ratio=1) + RMSNorm + FiLM-pre(log_Re) + L1 + Re-stratify, 0.619M params, 14/14 epochs at 148-150s/epoch. All new experiments build on this HEAD.

### Pending re-run on new canonical (RMSNorm+ratio=1 stack)
- **PR #1016 bf16 mixed precision:** Original headline `extyiumn` val_avg=58.49 was on OLD canonical (ratio=2 + LayerNorm). Against new RMSNorm canonical (57.95), that result is +0.94% — not a strong win. Frieren sent back to run one full canonical+bf16 compound: `--epochs 14 --re_stratify --swiglu_ratio 1 --rms_norm --bf16`. Predicted compound: val_avg ~56-57.
- **PR #1021 slice_num=32 (nezuko):** Won −5.0% on SwiGLU+ratio=1 stack (val=59.57). Monotonic trend across {32, 64, 96, 128}. Predicted compound on RMSNorm canonical: ~55 if mechanism additive. Sent back for paired sn32+RMSNorm vs sn64+RMSNorm A/B. Optional sn=16 extension probe.
- **PR #1020 ultra-thin SwiGLU (alphonse):** Pareto-flat at ratio=2/3 (intermediate=85, 0.54M params), val=62.69 vs ratio=1 62.74. Sent back for ratio=2/3+RMSNorm vs ratio=1+RMSNorm paired A/B. Most likely outcome: Pareto-flat merge as new param-efficient canonical.
- **PR #976 AoA-FiLM (askeladd, 2nd send-back):** v2 paired on SwiGLU+ratio=1 gave −0.44% val / −1.67% test (within noise band). γ-norm diagnostic shows AoA orthogonal axis added (Re-only γ unchanged, AoA-only γ adds +0.33). Cross-stack mechanism shrinkage: v1 −1.2% → v2 −0.4% → v3 ? Sent back for paired RMSNorm A/B.
- **PR #1029 surface quantile-reweight (fern):** v1 (top10/α=2) wins −2.6% val on SwiGLU stack (val=61.09), test −2.0% (53.91). Cruise OOD largest gain (-7.7% val, -9.3% test). qr_e_top_over_rest=4.14× confirms heavy-tail mechanism after 14 epochs. Predicted compound on RMSNorm canonical: ~56.5 val (potential new best). Sent back for paired A/B.

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
| **#999** | **RMSNorm replacing LayerNorm (canonical SwiGLU pairing)** | **MERGED 2026-04-29** — val_avg=57.9550 best / 58.30 mean (2 seeds), −6.8% val / −7.0% test; **new canonical config** | **57.9550** |
| **#961** | **SwiGLU MLP — replace GELU MLP with Swish-gated linear unit** | **MERGED — leaderboard low-water-mark** | **62.20** |

## Active WIP PRs

| Student | PR | Hypothesis | Status |
|---------|-----|-----------|--------|
| alphonse | #1020 | **Ultra-thin SwiGLU (intermediate=85, 0.54M params) — paper-efficiency** | **SENT BACK 2026-04-29** — Pareto-flat win on SwiGLU stack (val=62.69 vs 62.74 paired); needs RMSNorm canonical paired A/B |
| askeladd | #976 | AoA-FiLM: extend FiLM input from 1-d log_Re to 3-d (log_Re, AoA1, AoA2) | **SENT BACK 2nd time 2026-04-29** — v2 paired SwiGLU val=61.01 vs 61.28 (Δ−0.44% within noise; test Δ−1.67%); cross-stack γ shrinkage 34% with AoA orthogonal addition; needs v3 RMSNorm paired A/B |
| thorfinn | #1076 | **Coordinate-frame canonicalization (rotate (x,y) by -AoA1) — physics-informed inductive bias targeting camber OOD** | WIP — new 2026-04-29 (replaced #1057 NACA_M FiLM which CLOSED) |
| nezuko | #1021 | **slice_num sweep {32, 64, 96, 128} — physics-attention spatial resolution ablation** | **SENT BACK 2026-04-29** — sn=32 wins decisively on SwiGLU stack (val=59.57, −5.0%); monotonic trend; cruise −10.3%; needs RMSNorm paired A/B + optional sn=16 |
| fern | #1029 | **Surface quantile-reweight (top10/α=2 wins −2.6% val on SwiGLU stack, cruise −7.7%; qr_e_top_over_rest=4.14× confirms heavy-tail mechanism)** | **SENT BACK 2026-04-29** — needs RMSNorm canonical paired A/B; predicted compound ~56.5 val (potential new best) |
| edward | #1080 | **Polynomial cross-features for conditioning vars — explicit nonlinear physics priors (log_Re×NACA_M1, AoA1×NACA_M1, etc.)** | WIP — new 2026-04-29 (replaced #1061 NACA_M-stratify which CLOSED) |
| frieren | #1016 | **bf16 mixed precision training — wall-clock unlock + cosine recovery** | **SENT BACK 2nd time 2026-04-29** — original v1 (val=58.49) was on OLD canonical (ratio=2 + LayerNorm); +0.94% vs new RMSNorm 57.95 — needs canonical+bf16 compound run |
| tanjiro | #1056 | **LR sweep (lr=2e-4, 5e-4 ref, 1e-3) — optimizer axis not touched since round 1; RMSNorm+SwiGLU change loss landscape** | WIP — new 2026-04-29 |

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
- **SwiGLU is the new canonical MLP architecture** (PR #961, merged 2026-04-29, val 79.54→62.20 −21.8%). All future experiments build on L1 + FiLM-pre + Re-stratify + SwiGLU. Active pre-SwiGLU WIP branches (#869, #927, #969, #976) — if any beats old baseline (79.54) but not new (62.20), they will be sent back to rebase onto SwiGLU HEAD. Post-SwiGLU assignments (#983, #999, #1016) build directly on SwiGLU HEAD.
- **Off-Re axes underway:** physics-attention resolution (slice_num #1021), parameter-efficiency push (ultra-thin SwiGLU #1020), data-distribution (NACA_M stratify #1061), conditioning multi-variable (AoA-FiLM #976, NACA_M FiLM #1057), within-sample heavy-tail surface reweight (#1029), optimizer (LR sweep #1056), wall-clock infrastructure (bf16 #1016). [SwiGLU ablation #983 MERGED, vol-channel #927 CLOSED, DropPath #975 CLOSED, surf_weight #869 CLOSED].
- **Volume-loss-shape direction closed at SwiGLU stack** (PR #927 v2): vol_w_p mechanism shrunk −9.0% → −4.2% post-SwiGLU; surf_p flipped neutral → +4.77% regression. SwiGLU's bilinear gating absorbs the channel-residual-balance lever; explicit channel-rebalancing now competes rather than complements. Cross-stack mechanism reversal — node-level reweight is the next refinement direction (fern #1029).
- **surf_weight rebalancing direction closed at SwiGLU+RMSNorm stack** (PR #869 v2): sw=7 → +1.6%, sw=5 → +4.4% on primary metric vs SwiGLU best. Three consistent data points now confirm: gradient-ratio rebalancing (vol-L1 #927, surf_weight #869 sw=5, sw=7) does not stack with SwiGLU's bilinear gating on the primary metric. Vol_p does improve (~5%), but surface accuracy suffers. surf_weight=10.0 is the correct default for current stack.
- **RMSNorm is new canonical normalization** (PR #999, merged 2026-04-29): val_avg=57.95/58.30 mean, test_avg=51.17/51.51 mean; −6.8% val / −7.0% test vs SwiGLU. LLaMA/Mistral-style pairing with SwiGLU bilinear gating. Pareto win: −1,408 params, fewer FLOPs, simpler definition. Two-seed std=0.34 well inside noise band.
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
2. **AoA-FiLM (multi-variable conditioning)** — extend FiLM input from 1-d (log_Re) to 3-d (log_Re, AoA1, AoA2). Sent back for SwiGLU rebase + paired A/B. **Assigned → askeladd PR #976.**
3. ~~**DropPath rate sweep**~~ — **Closed (PR #975 2026-04-29): all 3 rates >81, best epoch = last for all** — DropPath disrupts FiLM modulation chain, delays convergence rather than regularizing. Negative result for ablation table. **Follow-up: edward PR #1061 (NACA_M1-stratified sampling, camber-axis gradient equalization).**
4. ~~**RMSNorm replacing LayerNorm**~~ — **MERGED PR #999 (val=57.9550). Canonical normalization.** −6.8% val / −7.0% test; new beat-threshold <57.9550.
5. **bf16 mixed precision training** — wall-clock unlock + cosine recovery; pending rebase, strong win already confirmed. **Frieren PR #1016.**
6. **Ultra-thin SwiGLU (mlp_ratio=2/3, 0.51M params)** — paper-efficiency extension of gating-mechanism thesis. **Assigned → alphonse PR #1020.**
7. **slice_num sweep {32, 64, 96, 128}** — physics-attention spatial resolution ablation. **Assigned → nezuko PR #1021.**
8. **LR sweep (lr=2e-4, 5e-4 ref, 1e-3)** — optimizer axis untouched since round 1; SwiGLU bilinear gating + RMSNorm change loss landscape curvature. **Assigned → tanjiro PR #1056.**
9. ~~**NACA_M FiLM (geometry-aware conditioning)**~~ — **CLOSED (PR #1057 2026-04-29): val_avg=60.68 +4.7%; γ_naca grows monotonically but cos(γ_re_aligned, γ_naca)→0.997 in deep blocks (NACA absorbed onto Re-axis); held-out M values cannot be extrapolated by linear FiLM projection.** Multi-variable FiLM with held-out geometric axes is now declared closed. **Follow-up: thorfinn → PR #1076 (coordinate-frame canonicalization, input-side physics-informed inductive bias).**

- ~~**DropPath (stochastic depth)**~~ — **Closed (PR #975 2026-04-29): all 3 rates >81, best epoch=last for all, geom_camber_rc regressed** — FiLM modulation chain disruption. Paper-quality ablation negative result.
- ~~**NACA_M FiLM (geometry-aware multi-variable conditioning)**~~ — **Closed (PR #1057 2026-04-29): val_avg +4.7%; γ-norm cosine-similarity diagnostic shows NACA absorbed onto Re-axis in deep blocks (cos→0.997); linear FiLM projection cannot extrapolate to held-out M values.** Multi-variable FiLM with held-out geometric axes declared closed.

### Closed directions (this round)
- ~~**NACA_M1-stratified sampling**~~ — **Closed (PR #1061 2026-04-29): val_avg +6.3%, both camber OOD splits worsen (rc +2.7%, cruise +12.1%).** Mechanism failure: NACA_M is not a FiLM input (gradient equalization on raw features ≠ equalization of conditioning signal); more importantly, held-out M values aren't in training distribution — stratification within seen distribution doesn't expose model to held-out OOD region. Distribution shift ≠ distribution skew. Both data-side (this) and model-side (#1057 FiLM) interventions on NACA_M-stratify axis now closed for held-out-camber OOD. **Follow-up: edward → PR #1080 (polynomial cross-features — input-side nonlinearity for held-out-M extrapolation).**
- ~~**SwiGLU MLP**~~ — **MERGED (PR #961, val=62.20 −21.8%). New canonical.** Paper ablation **MERGED (PR #983, ratio=1 canonical).** Gating mechanism is primary driver (~97% of gain).
- ~~**RMSNorm**~~ — **MERGED (PR #999, val=57.9550 −6.8%). New canonical normalization.** Two seeds, std=0.34.
- ~~**Per-channel volume L1 (vol_w_p=2.0)**~~ — **Closed (PR #927 v2): mechanism absorbed by SwiGLU** (+4.77% surf_p; vol_p gain shrunk −9.0% → −4.2%). Volume-loss-shape direction closed at this architecture.
- ~~**surf_weight rebalancing (sw=5, 7)**~~ — **Closed (PR #869 v2): mechanism absorbed by SwiGLU bilinear gating** (+1.6% and +4.4% regression on primary metric). Vol_p still improves but doesn't offset surface cost. surf_weight=10.0 is optimal for current stack.
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
16. **Compound stack frontier** — FiLM-pre + Re-stratify + SwiGLU + best-of post-SwiGLU winners (RMSNorm, AoA-FiLM, NACA_M FiLM, NACA_M stratify, slice_num, surf-quantile, bf16, LR-opt). Round-4 candidate.
