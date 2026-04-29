# SENPAI Research Results — willow-pai2e-r3

## 2026-04-29 — PR #999 (PENDING REBASE → MERGE): RMSNorm — strong win, two seeds, val_avg=58.30 mean (best 57.95)
- **Branch:** `thorfinn/rmsnorm`
- **Hypothesis:** RMSNorm (scale-only, no mean-centering) is the canonical normalization pairing for SwiGLU. Removing the mean-centering preserves scale (the bilinear-gate-relevant statistic), reduces activation coupling, and gives back ~2 epochs of wall-clock margin inside the 30-min budget.
- **Runs:** W&B `6krvx540` (v2, best) and `82kvdhbn` (v3), both 14/14 epochs, 32.3 min, 54.6 GB peak, group `rmsnorm`

| Split | val v2 (`6krvx540`) | val v3 (`82kvdhbn`) | val mean | val SwiGLU baseline | Δ mean | test v2 | test v3 | test mean | test baseline | Δ test |
|---|---|---|---|---|---|---|---|---|---|---|
| `single_in_dist` | 61.93 | 62.24 | 62.09 | 74.96 | **−17.2%** | 55.77 | 56.09 | 55.93 | 65.07 | **−14.1%** |
| `geom_camber_rc` | 72.84 | 72.34 | 72.59 | 73.39 | −1.1% | 66.36 | 66.61 | 66.48 | 67.47 | −1.5% |
| `geom_camber_cruise` | 40.39 | 40.77 | 40.58 | 42.66 | −4.9% | 32.88 | 34.10 | 33.49 | 35.67 | −6.1% |
| `re_rand` | 56.66 | 59.20 | 57.93 | 57.81 | +0.2% | 49.69 | 50.56 | 50.12 | 51.93 | −3.5% |
| **avg** | **57.9550** | **58.6360** | **58.30** | **62.20** | **−6.3%** | **51.1735** | **51.8385** | **51.51** | **55.04** | **−6.4%** |

Two-seed std (range/2): val_avg=0.34, test_avg=0.33 — well inside noise band.

### Decision: PENDING REBASE → MERGE — strong win (val < 60 threshold cleared); merge conflict on squash-merge due to branch updates since submission
- **Both seeds beat val < 60 strong-win threshold** (57.95, 58.64). Mean 58.30 vs baseline 62.20 = −6.3% val / −6.4% test.
- **`single_in_dist` biggest winner (−17.2% val)** — opposite of prediction that Re-targeted splits would benefit most. Interpretation: SwiGLU's bilinear-gate gain had already captured most Re-extrapolation headroom; RMSNorm's gain comes from cleaner gradient flow on in-distribution variation (scale preservation on the gate input). `geom_camber_rc` minimal (−1.1%); `re_rand` essentially flat on val (+0.2%) but meaningful on test (−3.5%).
- **Pareto win on simplicity:** −1,408 params (remove β terms from 11 LayerNorms), fewer FLOPs, simpler definition. Even a flat result would merge as a Pareto improvement.
- **Thorfinn's implementation used `F.rms_norm()` (PyTorch ≥2.4 fused CUDA kernel)** — numerically equivalent to manual, slightly faster (one kernel launch).
- **Sent back for mechanical rebase** — advisor branch was updated after submission (PR #927 close, #976 sendback, doc commits). No re-run needed; rebase + push + resubmit is sufficient.
- **New canonical config post-merge:** `--rms_norm` added to all future baselines. Beat-threshold updates to val_avg < 57.9550.

---

## 2026-04-29 — PR #976 (SENT BACK): AoA-FiLM v1 — strong mechanism win on pre-SwiGLU stack; rebase onto SwiGLU+ratio=1 HEAD required
- **Branch:** `askeladd/aoa-film`
- **Hypothesis:** Extending FiLM input from 1-d (log_Re) to 3-d (log_Re, AoA1, AoA2) tests whether multi-variable conditioning compounds. AoA is a primary flow parameter modulating stagnation point, lift coefficient, wake interaction.
- **Run:** W&B `tevsj514`, 14/14 epochs, 31.8 min, 44.7 GB peak, group `aoa-film` (pre-SwiGLU stack — `swiglu` key absent from config)

| Split | val pre-SwiGLU baseline | val v1 (`tevsj514`) | val Δ | test pre-SwiGLU baseline | test v1 | test Δ |
|---|---|---|---|---|---|---|
| `single_in_dist` | 84.70 | 87.97 | +3.86% | 73.79 | 76.62 | +3.83% |
| `geom_camber_rc` | 92.95 | 91.32 | −1.76% | 83.50 | 80.86 | −3.16% |
| `geom_camber_cruise` | 63.49 | **58.75** | **−7.46%** | 52.45 | **50.17** | −4.36% |
| `re_rand` | 77.02 | 76.30 | −0.94% | 71.29 | 68.44 | −4.00% |
| **avg** | **79.54** | **78.58** | **−1.21%** | **70.26** | **69.02** | **−1.76%** |

### γ-norm diagnostics (epoch 14, paper-quality mechanism evidence)
| Block | γ-norm full | γ-norm Re-only | γ-norm AoA-only |
|---|---|---|---|
| 0 | 0.302 | 0.280 | 0.266 |
| 1 | 0.622 | 0.415 | 0.504 |
| 2 | 0.854 | 0.627 | 0.493 |
| 3 | 0.827 | 0.753 | 0.496 |
| 4 | 0.887 | 0.805 | 0.499 |
| **mean** | **0.698** | **0.576** | **0.452** |

- AoA-only γ-norm grew monotonically from 0.20 (epoch 1) to 0.45 (epoch 14) — model is actively learning AoA conditioning, not falling back to Re-only.
- Re still dominates deeper blocks (Re-only γ rises 0.28→0.80); AoA stays ~0.50 from block 1 onward — plausibly a global stagnation-point/lift correction applied uniformly across depth.

### Decision: SENT BACK — rebase onto SwiGLU+ratio=1 HEAD; paired A/B v2-aoa vs v2-baseline
- **Mechanism is real** on pre-SwiGLU stack: cruise −7.5% is the largest single-split improvement since FiLM-pre-block landed; AoA γ being learned monotonically; predicted-flat re_rand confirmed (−0.9%).
- **Cannot directly stack on SwiGLU's gain** — bilinear gating (silu * up) and FiLM-pre are both multiplicative on hidden state; could compound or interfere. fern PR #927 cross-stack mechanism reversal (vol_p shrunk −9.0%→−4.2%, surf_p flipped +4.77% post-SwiGLU) is direct precedent for SwiGLU absorbing channel-rebalancing levers.
- **Single-seed noise band on SwiGLU stack ~3.6%** (62.20→64.46 same-day, PR #983) — paired A/B same-day is the only way to robustly separate AoA-FiLM signal from drift on the new baseline.
- **Send-back protocol:** rebase `askeladd/aoa-film` onto current HEAD (SwiGLU+ratio=1+L1+FiLM-pre+Re-stratify all merged), run paired v2-aoa (with `--aoa_film`) and v2-baseline (without) in `wandb_group=aoa-film-on-swiglu`. Re-log γ-norm diagnostics — especially interesting whether γ-norms collapse on SwiGLU (capacity overlap signature).
- **Decision criteria on rebased run:** merge if v2-aoa val_avg < min(v2-baseline, 62.20) AND cruise reproduces direction. Strong-win <60 opens 4-d FiLM (Re, AoA1, AoA2, gap). Send-back-again if within ±0.5 of v2-baseline + cruise improves → wider FiLM hidden Linear(3, 64) is next probe.

---

## 2026-04-29 — PR #927 (CLOSED): Per-channel volume-loss weights — mechanism absorbed by SwiGLU
- **Branch:** `fern/per-channel-vol-l1`
- **Hypothesis:** v1 result on pre-SwiGLU stack: `vol_w_p=2.0` gave −9.0% on `val_avg/vol_p` with neutral surf_p. Fern requested rebase onto SwiGLU (post-PR #961 #983) to test whether the volume-pressure mechanism survived the bilinear-gating MLP.
- **Runs:** W&B `aa6mch9c` (v2-rebased, vol_w_p=2.0) and `qwzljf64` (v2-baseline, vol_w_p=1.0), both 14/14 epochs, group `per-channel-vol-l1-v2`

| Metric | v2-baseline (vol_w_p=1.0) | v2-rebased (vol_w_p=2.0) | Δ v2 | v1 (pre-SwiGLU) |
|---|---|---|---|---|
| **val_avg/mae_surf_p** | **62.60** | **65.59** | **+4.77%** | ~neutral |
| val_avg/mae_vol_p | 0.475 | 0.455 | −4.21% | −9.0% |
| test_avg/mae_surf_p | 56.20 | 58.30 | +3.74% | n/a |

### Decision: CLOSED — mechanism absorbed by SwiGLU; cross-stack mechanism reversal documented
- **The vol_p mechanism shrunk from −9.0% (v1, GELU) to −4.2% (v2, SwiGLU)** — bilinear gating absorbs much of the per-channel residual-balance effect. Surf_p flipped from neutral to clear regression (+4.77%).
- **Fern's paired A/B was the right protocol** — v2-baseline at 62.60 (vs PR #961's 62.20) became a third SwiGLU canonical noise data point alongside 64.46 (PR #983 ratio=2 same-day) — single-seed band ~62.20–64.46, range 3.6%. v2-rebased at 65.59 falls *outside* the noise band, confirming the regression is real not noise.
- **Cross-stack mechanism reversal:** explicit dampening of channel competition is a substitute for SwiGLU's bilinear gating, not a complement. With gating already in place, hard-coded dampening over-constrains.
- **Volume-loss-shape direction declared closed at this architecture.** Future channel-rebalancing should target node-level (within-channel) rather than channel-level reweighting.
- **Follow-up assigned:** fern → PR #1029 (surface loss reweighting by per-node pressure quantile — heavy-tail-targeted, within-sample within-channel reweighting, completely orthogonal to all current stack components).

---

## 2026-04-29 — Assignment: PR #1029 (fern surface quantile-reweight)
- **PR #1029** (`fern/surface-quantile-reweight`): Per-node loss reweighting on surface pressure, gated by absolute target value. CLI flags `--surf_p_quantile_topk` (fraction of top-|p| nodes) and `--surf_p_quantile_alpha` (multiplicative weight). Three-run sweep: v1 (top10, α=2), v2 (top20, α=2), v3 (top10, α=3).
- **Mechanism orthogonal to all current stack components:** L1 (uniform across surface nodes), FiLM-pre (Re conditioning), Re-stratify (between-sample), SwiGLU (MLP architecture), mlp_ratio=1 (capacity). This experiment differentiates *within* sample.
- **Distinct from PR #743 (channel-weighted) and PR #858 (focal):** static within-sample reweighting by `|y_p|`, not by channel and not by current model error. Cleaner attribution than focal.
- **Predicted delta:** −1 to −4% on val_avg (62.20 → 60–61.5). Largest gains expected on `single_in_dist` (highest absolute pressure) and `geom_camber_rc` (currently hardest at val=74.60).

---

## 2026-04-29 — PR #969 (CLOSED): Vertical-flip data augmentation — NACA M hidden asymmetry breaks mechanism
- **Branch:** `nezuko/vflip-augmentation`
- **Hypothesis:** y-mirror incompressible 2D flow symmetry → p=0.5 vflip aug effectively doubles training data for bilateral-mesh samples. Domain-gated (raceCar-single excluded via `min(pos_y) < -0.5` filter).
- **Run:** W&B `0g2fsgwg`, 14/14 epochs, ~29.2 min, 134.9s/epoch (+5% vs baseline), group `vflip-augmentation`, 443/1499 eligible (30%).

| Split | val v1 | val baseline | Δ val | Eligibility |
|---|---|---|---|---|
| `single_in_dist` | 91.58 | 84.70 | +8.1% | 0% (raceCar single) |
| `geom_camber_rc` | 93.08 | 92.95 | +0.1% | ~0% (raceCar) |
| `geom_camber_cruise` | **75.39** | 63.49 | **+18.8%** | ~100% (cruise) |
| `re_rand` | 86.66 | 77.02 | +12.5% | mixed |
| **avg** | **86.68** | **79.54** | **+9.0%** | |

### Decision: CLOSED — NACA M hidden asymmetry; sign-reversed per-split signal confirms mechanism issue (not hyperparameter)
- val_avg = **86.68** (+9.0% on old baseline 79.54). Falls clearly in the `close` zone (val_avg > 82).
- **Sign-reversed per-split signal is the smoking gun:** `geom_camber_cruise` (100% eligible) was the *worst* regression (+18.8% val), not the *best* as predicted. This is a mechanism issue, not a hyperparameter issue — dialing p from 0.5 to 0.3 cannot fix a sign-flipped dominant effect.
- **Root cause identified by nezuko: NACA M (camber magnitude) was not flipped.** After y-flipping pos_y, the mesh shows a foil with camber on the lower surface (M < 0 effectively), but the metadata still reports M ≥ 0 (NACA M unchanged). The model received mesh-vs-metadata contradictions on every flipped cruise sample. FiLM conditioning (which conditions on geometric metadata) is especially vulnerable to this.
- Consistent with frieren's PR #993 finding (TTA-vflip equivariance prerequisite falsified). Two independent y-symmetry probes have now hit the same dataset-asymmetry family.
- **Nezuko's domain-aware gating analysis** (citing Brandstetter et al. LPSDA ICML 2022 on group-action validity) was sharp — correctly excluded raceCar-single ground-effect samples. Gating doesn't rescue the NACA M issue however.
- **Dataset finding confirmed:** raceCar-single domain has ground plane at y=0 (0/599 bilateral). raceCar-tandem: 258/457 bilateral (57%). Cruise: 185/443 bilateral (42%). Total eligible: 443/1499 (30%). Documented in DATASET_ANALYSIS.md.
- **If vflip to be retried:** must also flip NACA M (cols 15 and 19 in 24-d input). This puts M into negative range never seen in training — risk, but required for self-consistency. Not assigning immediately — two vflip slots closed; higher-EV fresh axes available.
- **Follow-up assigned:** nezuko → PR #1021 (slice_num sweep {32, 64, 96, 128} — physics-attention spatial resolution ablation, completely fresh axis).

---

## 2026-04-29 — PR #983 (MERGED): SwiGLU mlp_ratio ablation — gating mechanism is the primary driver
- **Branch:** `alphonse/swiglu-ablation`
- **Hypothesis:** PR #961 SwiGLU gain (+21.8%) — from bilinear gating expressivity or added parameters (+24%)? SwiGLU at mlp_ratio=1 (0.62M params, fewer than GELU baseline's 0.70M) tests gating-only contribution.
- **Runs:** W&B `3m9a8l02` (v2-ratio1) and `2jwarnfe` (v2-ratio2), both 14/14 (ratio=1) and 12/14 (ratio=2), group `swiglu-ablation`

| Run | n_params | Epochs | val_avg | test_avg | s/epoch |
|---|---|---|---|---|---|
| **v2-ratio1 (ablation)** | **0.62M** | **14/14** | **62.74** | **55.04** | **138s** |
| v2-ratio2 (replication) | 0.87M | 12/14 | 64.46 | 57.30 | 152s |
| PR #961 baseline (`sv9ktfk3`) | 0.87M | 12/14 | 62.20 | 55.04 | ~152s |
| old GELU (`wakfw4uy`) | 0.70M | 14/14 | 79.54 | 70.26 | 135s |

### Decision: MERGED — canonical config switched to mlp_ratio=1; gating-mechanism verdict is paper-strong
- **Same-day paired comparison:** ratio=1 (62.74) beats ratio=2 (64.46) by −2.7% val / −3.9% test, with 29% fewer params and 9% faster wall-clock. 
- **Test_avg matches PR #961 EXACTLY** (55.04 = 55.04) — paper-facing metric, strong parity signal.
- **Gating-mechanism verdict:** SwiGLU at 0.62M (fewer than GELU's 0.70M) beats old GELU (79.54) by −21.1% val / −21.7% test. Capacity contribution bounded to ~3% of the SwiGLU gain (|62.74−62.20| / (79.54−62.20) ≈ 3%).
- **Single-seed variance context:** today's ratio=2 drifted 64.46 vs PR #961's 62.20 (+3.6%), showing single-seed noise ~2-3%. The +0.54 drift of ratio=1 vs PR #961 (62.74 vs 62.20) is within that band.
- **Strategic infra win:** mlp_ratio=1 runs 14/14 epochs vs 12/14 for ratio=2, recovering the 2 lost cosine-annealing epochs. Future PRs benefit from fuller LR decay.
- **New canonical config:** `--swiglu_ratio 1` is now the default in merged HEAD. BASELINE.md updated.
- **Leaderboard low-water-mark remains 62.20** (PR #961 snapshot). Future PRs beat 62.20.
- **Follow-up assigned:** alphonse → PR #1020 (ultra-thin SwiGLU mlp_ratio=2/3, intermediate=85, 0.51M params — paper-strong parameter-efficiency extension of the gating-mechanism thesis).

---

## 2026-04-29 — Assignments: PR #1020 (alphonse ultra-thin SwiGLU), PR #1021 (nezuko slice_num sweep)
- **PR #1020** (`alphonse/ultra-thin-swiglu`): SwiGLU with `intermediate_dim=85` (= floor(128 × 2/3), ~0.51M params — fewer than GELU baseline's 0.70M). Tests the gating-mechanism thesis at even lower capacity. Paper-strong: if ratio=2/3 beats old GELU (79.54), the bilinear-gating story holds even at 27% fewer params than GELU baseline.
- **PR #1021** (`nezuko/slice-num-sweep`): slice_num sweep {32, 64, 96, 128} on full SwiGLU+L1+FiLM+Re-stratify stack. First ablation of Transolver's core spatial-resolution hyperparameter. Higher slice_num may help OOD-geometry splits (camber_rc, camber_cruise); sweep identifies the optimal value.

---

## 2026-04-29 — PR #993 (CLOSED): TTA with vertical flip — equivariance prerequisite violated by dataset asymmetry
- **Branch:** `frieren/tta-vflip`
- **Hypothesis:** y-mirror symmetric incompressible 2D flow → eval-time vflip + averaging halves variance of asymmetric error component. Predicted −1 to −3% on val_avg.
- **Run:** W&B `rwhif8qa`, 12/14 epochs, group `tta-vflip`, peak 54.6 GB

| Metric | TTA-on (this run) | TTA-off (same checkpoint) | Baseline (PR #961 SwiGLU) |
|---|---|---|---|
| **val_avg/mae_surf_p** | **133.00** (+114%) | n/a (TTA on at val) | **62.20** |
| test_avg/mae_surf_p | **126.22** (+129%) | **58.09** (+5.5%, ~noise) | **55.04** |

### Decision: CLOSED — equivariance prerequisite violated; flip implementation IS correct
- TTA-on val_avg = **133.00**, +114% vs current best 62.20. Same-checkpoint TTA-off test 58.09 (~baseline noise) — **regression entirely from inference-time mirror pass**, not from training.
- **Flip implementation is the most correct possible** (frieren's offline ablation table). `pos_y`, `saf[1]` (verified y-signed via `corr(pos_y, saf[1])=+0.99`), `dsdf` angle-mirror permutation `[0°, 45°, ..., 315°] → [4, 11, 10, 9, 8, 7, 6, 5]`, AoA1/AoA2 sign-flip, pred_Uy un-flip — all correct. Each component *reduces* pred1↔pred2 disagreement when added.
- **Genuine new dataset finding (frieren's diagnostic):** dataset is structurally y-asymmetric:
  - **~54% of training samples are half-domain meshes** (y > 0 only — CFD efficiency trick exploiting bilateral symmetry).
  - **Single-foil AoA range = `[-9.99°, 0°]`** — only negative; under y-flip → strictly positive AoA, never seen.
  - **Stagger range = `[0, +2.0]`** — strictly positive; under y-flip → negative, never seen.
  - **`dsdf` is angle-mirror permuted, not magnitude** — a non-trivial structural property.
- Model trained on this asymmetric distribution **has no incentive to learn y-equivariance**. Pred(flip(x)) is essentially independent of pred(x) — going severely OOD on flipped input. Averaging halves the squared error of pred(x) but adds a pred(flip(x))-magnitude OOD error that dominates.
- **Per-split signal corroborates:** `geom_camber_cruise` is the most bilateral split (sample 0: 162K positive y, 57K negative) → smallest TTA delta (+47.5 vs single-in-dist +85.0, rc +72.5). The more bilateral the mesh, the closer flipped input is to in-distribution.
- **Implications for PR #969 (nezuko vflip-aug)**: training-time vflip will produce ~OOD configurations (positive single-foil AoA, negative stagger). May need to subset to bilateral samples OR pair with re-meshing. Will discuss on #969 review.
- **Methodology insight (frieren's #4):** TTA-aware checkpoint selection — best-ckpt should be selected on TTA-off val, not TTA-on, to avoid contamination by mirror-pass quality.
- **Updated DATASET_ANALYSIS.md** with the y-asymmetry findings (new section).
- **Follow-up assigned:** frieren → PR #1016 (bf16 mixed precision training — strategic infra unlock + likely direct gain from recovered cosine epochs).

---

## 2026-04-29 — Assignment: PR #1016 (frieren bf16 mixed precision)
- **PR #1016** (`frieren/bf16`): bf16 autocast for forward+backward on full SwiGLU stack; weights/optimizer stay fp32. Targets the wall-clock binding constraint (SwiGLU truncated cosine schedule at 12/14). Predicted ~30-40% per-epoch speedup → recovers 2 lost epochs of LR-decayed training. Strategic infra unlock — reopens depth/width experiments (PR #936 depth-scaling closed for wall-clock-incompat in fp32).

---

## 2026-04-29 — PR #970 (CLOSED): Shared FiLM head — rank-reduction probe of FiLM conditioning manifold
- **Branch:** `thorfinn/shared-film`
- **Hypothesis:** Replace 5 independent per-block FiLM heads with one shared FiLM head reused at all blocks. Tests whether per-block FiLM specialization carries information vs is just redundant capacity. Saves ~34K params (~5% of model). Match-or-better → Pareto win.
- **Run:** W&B (single seed, 14/14 epochs, group `shared-film`)
- val_avg/mae_surf_p = **83.55** vs old baseline (when assigned) 79.54 → **+5.0% regression**; vs current SwiGLU baseline 62.20 → **+34.3% regression**.

### Decision: CLOSED — depth-specialization confirmed; 4th FiLM-redistribution failure
- **Per-block FiLM specialization carries non-trivial information.** The 5 independent FiLM heads aren't redundant — collapsing to a single shared head loses 34K params *and* hurts val_avg by 5% on old baseline. The conditioning manifold is genuinely depth-rank-5, not depth-rank-1.
- **Per-split signal:** regression magnitude grows with the Re-range width of each split — `re_rand` regressed most (where Re distribution is widest, requiring most depth-specialized conditioning). `single_in_dist` (narrow Re range) regressed least. Direct evidence of depth-specialization in the Re-conditioning function.
- **Operator-class argument:** at each block, FiLM head must compose (1+γ_l)·h_l + β_l with the **already-modulated** input from prior blocks. Sharing means same (γ, β) applied at multiple compositional depths — non-equivalent operations even at identical Re. Compounding effect explains why per-split regression scales with Re-range.
- **Fourth FiLM-redistribution probe to fail.** With #934 (last-2 only), #937 (dual FiLM), #756 (Fourier features), and now #970 (shared head) all clustering at +2-5% regression, the FiLM-axis is **architecturally saturated** in the redistribution sense. Further FiLM-redistribution experiments would be redundant.
- **Follow-up assigned:** thorfinn → PR #999 (RMSNorm replacing LayerNorm — canonical SwiGLU pairing). Pivot off FiLM axis to normalization axis. RMSNorm is the LLaMA/Mistral-canonical normalization for SwiGLU stacks; LayerNorm pairing is leaving a small clean win on the table.

---

## 2026-04-29 — Assignment: PR #999 (thorfinn RMSNorm replacing LayerNorm)
- **PR #999** (`thorfinn/rmsnorm`): RMSNorm (root mean square norm — LayerNorm without mean-centering) replaces LayerNorm in all 3 block normalization sites (pre-attn / pre-mlp / pre-head). Canonical pairing with SwiGLU in modern transformers (LLaMA, Mistral, PaLM). RMSNorm preserves scale (the statistic SwiGLU's bilinear gate is sensitive to) while dropping centering (one fewer cross-dimension correlation). Modest expected gain (−0.5 to −2%); paper-friendly architectural simplification with clean methods-section alibi. Stack: L1 + FiLM-pre + Re-stratify + SwiGLU + **RMSNorm**.

---

## 2026-04-29 — PR #962 (CLOSED): EMA model weights on FiLM+L1+Re-stratify (revisit #759 in new regime)
- **Branch:** `frieren/ema-weights`
- **Hypothesis:** EMA decay=0.999 shadow weights for evaluation; revisit prior PR #759 close in correct regime (post-L1+FiLM+Re-stratify).
- **Run:** W&B `qhys5efw`, 13/14 epochs (timeout cut by added raw-val diagnostic at +14s/epoch), 31.3 min, group `ema-weights`, peak 44.7 GB, 0.70M params.

| Split | val baseline | val EMA v1 | Δ_val | test baseline | test EMA v1 | Δ_test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 84.70 | 104.58 | **+23.5%** | 73.79 | 91.25 | +23.7% |
| `geom_camber_rc` | 92.95 | 99.31 | +6.8% | 83.50 | 88.76 | +6.3% |
| `geom_camber_cruise` | 63.49 | 70.30 | +10.7% | 52.45 | 59.76 | +13.9% |
| `re_rand` | 77.02 | 85.76 | +11.4% | 71.29 | 80.63 | +13.1% |
| **avg** | **79.54** | **89.99** | **+13.1%** | **70.26** | **80.10** | **+14.0%** |

### Decision: CLOSED — wrong-regime mechanism (third falsification of "smoothing" mechanism on this stack)
- val_avg = **89.99**, +13.1% vs old baseline 79.54 (and +44.7% vs new SwiGLU baseline 62.20).
- **Mechanism analysis (student's own, correct):** decay=0.999 → ~1000-step memory window ≈ 2.7 epochs. Model val_avg improves 108→85 across that window — EMA is averaging *stale weights from a non-stationary trajectory*, not refining a stationary minimum. The "post-convergence smoothing" regime EMA needs **never exists in this 14-epoch schedule** — model still descending steeply at cutoff.
- EMA-vs-raw gap remained +5.4 mae units at epoch 13, never closed within budget.
- Faster decay (0.99) would make EMA ≈ raw weights (no smoothing). SWA over last-4 epochs would assume convergence in averaging window (epochs 9–10 are pre-convergence on SwiGLU stack — would pull result up). Neither variant fixes the diagnosed root cause.
- **Conclusion:** training-time weight smoothing isn't a lever in this short-budget non-converged regime. Pivot to inference-time augmentation (TTA, frieren PR #993).
- **Follow-up assigned:** frieren → PR #993 (TTA with vertical flip — inference-time symmetry exploitation).

---

## 2026-04-29 — Assignment: PR #993 (frieren TTA with vertical flip)
- **PR #993** (`frieren/tta-vflip`): test-time augmentation with vertical flip (y → -y, Uy → -Uy at eval). Zero training cost; exploits whatever y-mirror symmetry of incompressible 2D flow the model has approximately learned from un-augmented training data. Orthogonal to nezuko #969 (training-time vflip) — composable. Inference-only forward-pass averaging.

---

## 2026-04-29 — PR #961 (MERGED): SwiGLU MLP — replace GELU MLP with Swish-gated linear unit
- **Branch:** `alphonse/swiglu-mlp`
- **Hypothesis:** SwiGLU's bilinear gating (`silu(gate) * up`) lets each MLP output unit express a learnable bilinear form over its inputs, vs GELU's single-path nonlinear projection. Nat-fit for PDE surrogates due to advection-like bilinear structure in flow physics.
- **Run:** W&B `sv9ktfk3`, 12/14 epochs (30-min env timeout; +12.6% per epoch → 152s), group `swiglu-mlp`, 0.87M params (+24%), peak 54.6 GB

| Split | val baseline (Re-strat+FiLM) | val SwiGLU | Δ_val | test baseline | test SwiGLU | Δ_test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 84.70 | **74.96** | −11.5% | 73.79 | **65.07** | −11.8% |
| `geom_camber_rc` | 92.95 | **73.39** | **−21.0%** | 83.50 | **67.47** | **−19.2%** |
| `geom_camber_cruise` | 63.49 | **42.66** | **−32.8%** | 52.45 | **35.67** | **−32.0%** |
| `re_rand` | 77.02 | **57.81** | **−24.9%** | 71.29 | **51.93** | **−27.2%** |
| **avg** | **79.54** | **62.20** | **−21.8%** | **70.26** | **55.04** | **−21.7%** |

### Decision: MERGED — new best, largest single-PR improvement in round 3
- val_avg = **62.20** (−21.8%) vs prior baseline 79.54; test = 55.04 (−21.7%) vs 70.26. Every split improves.
- **Largest gains on OOD-extrapolation splits:** geom_camber_cruise −32.8%, re_rand −24.9%, geom_camber_rc −21.0%. SwiGLU's bilinear forms help most where the model has to extrapolate — consistent with bilinear interaction terms approximating advection-like flow physics (`u·∇u`, pressure-gradient products).
- Wall-clock real (+12.6%/epoch, 12/14 epochs at budget). Epoch 12 SwiGLU (62.20) already 25.5% better than baseline epoch 12 (83.56). Gain is not explained by convergence.
- **Open question for paper:** gain from bilinear gating or added parameters (+24%)? → assigned alphonse PR #983 (mlp_ratio ablation).
- **New beat-threshold: val_avg/mae_surf_p < 62.20**

---

## 2026-04-29 — Assignment: PR #983 (alphonse SwiGLU mlp_ratio ablation)
- **PR #983** (`alphonse/swiglu-ablation`): SwiGLU with `mlp_ratio=1` (intermediate_dim=128 vs 256). Parameter-matched ablation — SwiGLU at ratio=1 has ~49.5K params/block vs GELU at ratio=2 with ~66K/block. **Fewer params** than old GELU baseline. If ratio=1 clearly beats old GELU baseline (79.54), bilinear gating is the primary driver, not capacity. Paper-critical control.

---

## 2026-04-29 — PR #952 (CLOSED): Wider single output head (128→256→3) — capacity vs independence
- **Branch:** `edward/wider-output-head`
- **Hypothesis:** PR #924 (per-channel heads) failed: was the bottleneck channel decoupling or sheer head capacity? Test capacity-only by widening the single channel-coupled head from 128→128→3 to 128→256→3.
- **Run:** W&B `qmrdrbcb`, 14/14 epochs, 31.81 min, group `wider-output-head`, peak 81.2 GB

| Split | val baseline | val v1-x2 | Δ_val | test baseline | test v1-x2 | Δ_test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 84.70 | 92.11 | **+8.7%** | 73.79 | 78.47 | +6.3% |
| `geom_camber_rc` | 92.95 | **89.03** | **−4.2%** | 83.50 | **80.81** | **−3.2%** |
| `geom_camber_cruise` | 63.49 | 65.54 | +3.2% | 52.45 | 55.20 | +5.2% |
| `re_rand` | 77.02 | 79.38 | +3.1% | 71.29 | 72.98 | +2.4% |
| **avg** | **79.54** | **81.52** | **+2.5%** | **70.26** | **71.86** | **+2.3%** |

### Decision: CLOSED — decoder capacity isn't the lever (two-experiment falsification)
- val_avg = **81.52** vs current best 79.54 → **+2.5% regression**. All 3 channels regressed uniformly (Ux +1.7%, Uy +1.0%, p +2.5%); 3/4 splits regressed.
- **Two independent capacity probes now both falsified:** PR #924 (per-chan heads, +5.8%) — capacity + decoupling; PR #952 (wider single head, +2.5%) — capacity only, channel-coupled. Conclusion: decoder isn't bottlenecked by parameters at this depth/width budget.
- Wall-clock healthy (+1.2%, 14/14 epochs done). Convergence wasn't the bottleneck this time — model just doesn't want more head parameters.
- **Interesting preserved signal:** `geom_camber_rc` improved (−4.2% val / −3.2% test) — only split to benefit, and it's the hardest split. Suggests rc-specific bottleneck might be representational rather than capacity-uniform. Logged for future rc-targeted intervention design.
- **Follow-up assigned:** edward → PR #975 (DropPath rate sweep on FiLM+L1+Re-stratify) — pivot to regularization axis.

---

## 2026-04-29 — PR #917 (CLOSED): Re-input noise augmentation σ ∈ {0.02, 0.05, 0.10}
- **Branch:** `willowpai2e3-askeladd/re-input-noise`
- **Hypothesis:** Add small Gaussian noise to log(Re) input during training to force FiLM to learn locally smooth conditioning function, regularizing against memorization of training Re values.
- **Runs:** W&B `012xavg9` (σ=0.02), `jejfszpk` (σ=0.05), `x05pf7uo` (σ=0.10), 14 epochs each, group `re-input-noise`. **Run on OLD baseline (FiLM+L1, val=82.77)**, not current best (79.54).

| σ | val_avg | Δ vs 82.77 | val_re_rand | Δ_re |
|---|---|---|---|---|
| 0.0 (baseline) | 82.77 | — | 79.26 | — |
| 0.02 | 83.28 | +0.6% | 78.71 | −0.7% |
| 0.05 | 83.08 | +0.4% | **77.58** | **−2.1%** |
| 0.10 | 90.44 | +9.3% | 84.00 | +6.0% |

### Decision: CLOSED — mechanism confirmed but small; saturated by Re-stratify on current stack
- σ=0.05 is the cleanest mechanism point — `val_re_rand` improves monotonically 0 → 0.02 → 0.05 (clean dose-response on the targeted split, −2.1%). σ=0.10 collapses (perturbation overpowers FiLM). **The mechanism is real.**
- BUT: net effect on val_avg = +0.4% on old baseline (at-noise-floor); val_re_rand on current stack (with Re-stratify) is already 77.02 — *better* than σ=0.05's 77.58 on old baseline. Re-stratify ate the lever Re-noise was probing.
- **Three FiLM-redistribution attempts on current stack now falsified** (#934, #937, #756) — all clustering at +2-3% regression. Cross-cutting pattern: **Re-axis lever architecturally saturated** by FiLM-pre + Re-stratify. Re-noise lives on same axis and would likely cluster similarly on rebase. Closing rather than rebasing.
- The σ=0.05 → val_re_rand −2.1% data point preserved as evidence that input-noise smoothing helps OOD-Re at the FiLM+L1 stage (pre-Re-stratify world).
- **Follow-up assigned:** askeladd → PR #976 (AoA-FiLM: extend FiLM input from 1-d log_Re to 3-d (log_Re, AoA1, AoA2)) — pivot to multi-variable conditioning, fresh axis.

---

## 2026-04-29 — Assignments: PR #975 (edward DropPath sweep), PR #976 (askeladd AoA-FiLM)
- **PR #975** (`edward/drop-path-sweep`): stochastic depth on FiLM+L1+Re-stratify stack at rates {0.05, 0.10, 0.15} with linear depth ramp. Regularization-axis probe — different from PR #751 v2 (which tested at L1-only base, single rate). On current stack with FiLM forcing every block to participate in Re-conditioning and Re-stratify giving cleaner gradients, DropPath should compose differently. Targets `geom_camber_rc` (hardest split, regularization-amenable).
- **PR #976** (`askeladd/aoa-film`): FiLM input expanded from 1-d (log_Re) to 3-d (log_Re, AoA1, AoA2). Multi-variable conditioning probe — opens fresh axis after Re-axis saturation findings. AoA is a primary tandem-foil flow parameter that the model currently has zero conditioning awareness of. Expected to help geom_camber_* splits (where novel camber × AoA combinations live).
- Both opened off current best (79.54). Beat-threshold: val_avg/mae_surf_p < 79.54.

---

## 2026-04-29 — PR #937 (CLOSED): Dual FiLM (pre-block + post-block per block)
- **Branch:** `willowpai2e3-nezuko/dual-film`
- **Hypothesis:** Two FiLM heads per block — pre-block FiLM modulates attention input (regime-aware Q/K/V), post-block FiLM modulates output magnitude. Should give independent control over attention patterns vs scaling.
- **Run:** W&B (single seed, 14/14 epochs, group `dual-film`)
- val_avg/mae_surf_p = 82.36 vs current best 79.54 → **+3.5% regression**
- **Mechanism falsified:** dual FiLM doubles Re-conditioning capacity (5 × 2 = 10 FiLM heads, ~85K params → +12% params), but the Re-stratify+pre-block stack already saturates the Re-axis modulation lever. Adding redundant capacity slows convergence and amplifies early-epoch errors. Per-split signal: in-dist regressed most (+5%), Re-targeted splits flat — the opposite of what the hypothesis predicted.
- Confirms the emerging pattern across 3 independent FiLM-redistribution attempts (#934 last-2 FiLM, #937 dual FiLM, #756 Fourier): **the Re-conditioning lever is architecturally saturated** at the current pre-block-FiLM design.
- **Follow-up assigned:** nezuko → PR #969 (geometric vertical-flip data augmentation — opens a fresh **geometric-symmetry axis** orthogonal to FiLM/Re-stratify).

---

## 2026-04-29 — PR #934 (CLOSED): Layer-targeted FiLM (last 2 blocks only)
- **Branch:** `willowpai2e3-thorfinn/film-last2`
- **Hypothesis:** Pre-block FiLM on the last 2 blocks only — early blocks process geometry-dominant features, late blocks integrate flow-regime context. Targeted conditioning should reduce over-conditioning on early geometry features.
- **Run:** W&B (single seed, 14/14 epochs, group `film-last2`)
- val_avg/mae_surf_p = 81.74 vs current best 79.54 → **+2.8% regression**
- **Mechanism falsified:** removing FiLM from blocks 0–2 removes useful Re-conditioning capacity rather than reducing "over-conditioning". Early-layer FiLM signals propagate forward and inform attention patterns at every depth — pruning them reduces the effective conditioning budget. Per-split: re_rand and cruise (Re-targeted splits) both regressed, contradicting the "early blocks don't need Re" prediction.
- All 5 blocks contribute to FiLM modulation; pruning by depth was the wrong axis.
- **Follow-up assigned:** thorfinn → PR #970 (shared FiLM head — one head reused at all 5 blocks). This was thorfinn's own suggested follow-up #1 and tests **rank-reduction** of the FiLM conditioning manifold — the right axis for redistribution.

---

## 2026-04-29 — Assignments: PR #969 (nezuko vflip), PR #970 (thorfinn shared-FiLM)
- **PR #969** (`nezuko/vflip-augmentation`): random vertical flip data augmentation (y → -y, Uy → -Uy, p unchanged) at p=0.5 per sample. Exploits known physical symmetry of incompressible 2D flow to effectively double training set on a **geometric** axis. Predicted strongest gains on `geom_camber_rc` (the hardest split, currently val=92.95) and `single_in_dist` (camber asymmetry).
- **PR #970** (`thorfinn/shared-film`): single FiLMLayer reused across all 5 blocks (vs 5 independent heads). Rank-reduction probe — directly tests whether per-block FiLM specialization carries information after the saturation evidence from #934/#937/#756. Saves ~34K params (~5% of model). Match-or-better is mergeable as Pareto win; clean win unlocks more capacity per FiLM head at same param budget.
- Both opened off current best (79.54). Beat-threshold: val_avg/mae_surf_p < 79.54.

---

## 2026-04-29 — PR #910 (MERGED): Re-stratified batch sampling
- **Branch:** `willowpai2e3-nezuko/re-stratified-sampling`
- **Hypothesis:** Partition training set into Re quintiles; ensure every mini-batch contains samples from each quintile (round-robin). Gives FiLM conditioning consistent Re-diverse training signal; also equalizes the high-Re gradient dominance under L1 loss.
- **Run:** W&B `wakfw4uy`, 14 epochs, 31.5 min, group `re-stratified-sampling`, params 704,919 (unchanged)

| Split | val baseline (pre-block FiLM) | val Re-strat | Δ val | test baseline | test Re-strat | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 93.70 | **84.70** | **−9.6%** | 84.20 | **73.79** | **−12.4%** |
| `geom_camber_rc` | 92.70 | 92.95 | +0.3% | 82.91 | 83.50 | +0.7% |
| `geom_camber_cruise` | 62.62 | **63.49** | +1.4% | 51.93 | **52.45** | +1.0% |
| `re_rand` | 77.16 | **77.02** | −0.2% | 70.54 | **71.29** | +1.1% |
| **avg** | 81.55 | **79.54** | **−2.5%** | 72.40 | **70.26** | **−3.0%** |

### Decision: MERGED — new best, Re-stratification compounds with FiLM+L1 stack
- **−2.5% val, −3.0% test** vs pre-block FiLM baseline. Clean improvement.
- Mechanism surprise: largest gain from `single_in_dist` (−9.6% val / −12.4% test), not `re_rand` as predicted. Interpretation: uniform random batching was biasing gradient toward high-Re cluster (which dominates by absolute pressure magnitude under L1); stratification equalizes per-Re gradient even within in-distribution samples.
- `geom_camber_rc` is now the hardest split (92.95 val), marginally plateauing across all interventions.
- Val curve still falling at epoch 14 — `--re_stratify` now defaults to True in all subsequent experiments.
- **New beat-threshold: val_avg/mae_surf_p < 79.54**

---

## 2026-04-29 — PR #743 (CLOSED): Per-channel surface loss — channel weighting v3
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Per-channel weights [1.0, 0.5, 2.0] (Ux, Uy, p) on L1 surface loss aligns gradient with pressure-dominated metric. Had stacked with Huber (−3.8%). Tested on FiLM+L1 stack (v3).
- **Run:** W&B `n5e2f61d`, 14 epochs, 32.1 min, group `channel-weighted-surface-loss v3-l1-base`

| Split | val FiLM+L1 baseline | val chan-weighted v3 | Δ val |
|---|---|---|---|
| `single_in_dist` | 95.54 | 97.01 | +1.5% |
| `geom_camber_rc` | 91.38 | 95.25 | +4.2% |
| `geom_camber_cruise` | 64.90 | 64.25 | −1.0% |
| `re_rand` | 79.26 | 78.24 | −1.3% |
| **avg** | 82.77 | **83.69** | **+1.1% (WORSE)** |

### Decision: CLOSED — mechanism falsified on FiLM+L1 stack
- Channel weighting doesn't stack with FiLM+L1: +1.1% worse than FiLM+L1 baseline, +2.6% worse than current best (79.54).
- FiLM's per-block `(1+γ)·h + β` conditioning already implicitly adapts effective per-channel contributions via regime-aware hidden state modulation. Adding explicit channel weights at loss level is redundant.
- Per-split pattern confirms mechanism overlap: only splits where FiLM is weakest (cruise/re_rand) saw marginal improvement; FiLM-strong splits (single_in_dist, rc) regressed.
- **Conclusion:** Per-channel gradient lever is saturated architecturally by FiLM. Channel weighting was genuine at Huber stage (−3.8%) but FiLM now captures that benefit more efficiently.

---

## 2026-04-29 — PR #936 (CLOSED): Depth scaling n_layers=7 — wall-clock incompatible
- **Branch:** `alphonse/depth-scaling`
- **Run:** W&B `6ugjdhi2`, 10/14 epochs (timeout-cut), 30.9 min, group `depth-scaling`, peak 60.6 GB
- val_avg/mae_surf_p = 91.74 vs current best 79.54 → **+15.3% regression**
- **Mechanism:** budget-incompatible, not capacity verdict. n=7 takes ~185s/epoch (+41% vs baseline ~131s) and only 10/14 epochs fit in 30-min timeout. Val curve still falling monotonically at epoch 10 (220 → 91); model never reached late-cosine annealing where baseline does its big drop (87.7 → 82.8 at ep14).
- Memory was as predicted (60.6 GB ≈ 62.5 GB est, well under 90 GB cap), no instability.
- **Conclusion:** Pure depth scaling cannot be tested fairly under 30-min wall-clock cap. Would need bf16, torch.compile, or grad-checkpointing to be viable.
- **Follow-up assigned:** alphonse → PR #961 (SwiGLU MLP — strict expressivity gain at near-same FLOPs).

---

## 2026-04-29 — PR #756 v3 (CLOSED): Fourier Re-encoding on FiLM+Re-stratify stack
- **Branch:** `willowpai2e3-frieren/fourier-re-encoding`
- **Run (v3):** W&B `g40h7vvp`, 14/14 epochs (clean finish), 31.5 min, group `fourier-re-encoding v3-film-restrat-base`, peak 44.9 GB
- val_avg/mae_surf_p = 81.96 vs current best 79.54 → **+3.0% regression**
- test_avg/mae_surf_p = 73.31 vs 70.26 → +4.3%
- **Mechanism falsified:** Fourier encoding is redundant with FiLM. FiLM injects log(Re) into hidden state at every block via 1→32 SiLU →2×n_hidden non-linear transformations; Re-stratify ensures FiLM sees full Re distribution per batch. A 12-d fixed Fourier basis at the input layer is strictly less expressive than what FiLM is already learning end-to-end.
- Per-split: regressions on the encoding-targeted splits (val_re_rand +4.8%, val_geom_camber_cruise +5.1%) — exactly where input-side Fourier should help. The encoding lever is saturated by FiLM.
- **Conclusion:** Once a conditional-modulation primitive (FiLM) is in the network, input-side fixed-frequency encoding adds only redundant signal + extra preprocess parameters. Matches the literature pattern: NeRF-style input encoding helps when conditioning has no other path; here it doesn't.
- **Follow-up assigned:** frieren → PR #962 (EMA model weights on FiLM+L1+Re-stratify — revisits prior #759 close in correct regime).

---

## 2026-04-29 — PR #924 (CLOSED): Per-channel output heads (Ux/Uy/p — decouple decoder)
- **Branch:** `willowpai2e3-edward/per-channel-head`
- **Run:** W&B `0fik93c4`, 13/14 epochs (timeout-cut), 30.6 min, group `per-channel-head`, peak 46.6 GB
- val_avg/mae_surf_p = 84.16 vs current best 79.54 → **+5.8% regression** (close criterion met)
- Mechanism: 3 independent heads (Linear(128,128) → GELU → Linear(128,1) × 3) vs shared Linear(128,128) → GELU → Linear(128,3). +33K params (~+4.7%) → ~141s/epoch vs baseline ~131s/epoch.
- **Per-epoch revealed nuance:** at epoch 13, per-chan (84.16) was *better* than baseline at epoch 13 (87.68). Baseline did its big drop on epoch 14 (87.68 → 82.77). Per-chan lost epoch 14 to 30-min timeout.
- All 3 channels regressed: surf_p +1.7%, surf_Ux +4.2%, surf_Uy +5.4% (val avg).
- Conclusion: at this budget, decoupling the decoder slows convergence enough to lose 1 effective epoch — net regression. Mechanism interesting but not viable at 30-min timeout.
- **Follow-up assigned:** edward → PR #952 (wider single head 128→256→3) — tests "decoder capacity vs decoder independence" with same param budget but no per-epoch wall-clock cost.

---

## 2026-04-29 — PR #927 (SENT BACK): Per-channel volume loss v1 (L1 on p only, MSE on Ux/Uy)
- **Branch:** `willowpai2e3-fern/per-channel-vol-loss`
- **Run (v1):** W&B `b47cu99c`, 14 epochs, 32.3 min, group `volume-l1-channel v1-p-only`, peak 44.6 GB

| Metric | FiLM+L1 baseline | v1-p-only | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 82.77 | 82.24 | −0.6% |
| **val_avg/mae_vol_p** | 97.33 | **88.60** | **−9.0%** |
| test_avg/mae_surf_p | 72.27 | 72.60 | +0.5% |
| test_avg/mae_vol_p | 86.60 | 78.45 | **−9.4%** |

### Decision: SENT BACK — mechanism works on vol_p but doesn't beat current best on primary metric
- val_avg/mae_surf_p=82.24 doesn't beat current best 79.54 (PR #910 Re-stratified merged after this PR was assigned).
- vol_p gain is real and substantial (−9.0% val / −9.4% test) — heavy-tail mechanism captured on vol_p as predicted.
- Surf_p flat (within noise) on FiLM+L1 baseline; mechanism is orthogonal to surface improvements.
- Largest vol_p gain on `val_geom_camber_cruise` (−16.9%) — channel-localized heavy-tail benefit.
- Ux/Uy preserved (mechanism check passed; MSE on Ux/Uy preserves gradient profile).
- **v2 plan:** rebase onto current HEAD (FiLM pre-block + Re-stratify), re-run with `--vol_l1_p_only`, also run paired `--novol_l1_p_only` baseline for clean A/B on current stack.

---

## 2026-04-29 — PR #756 (SENT BACK): Fourier Re-encoding v2-rebased
- **Branch:** `willowpai2e3-frieren/fourier-re-encoding`
- **Run (v2-rebased):** W&B `zyyswd05`, 13 epochs (timeout at 30.1 min), group `fourier-re-encoding v2-rebased`
- val_avg=88.671 — does not beat current best 79.54
- **Status:** Sent back for v3 rebase onto full FiLM+pre-block+Re-stratify stack. Fourier encoding hypothesis still live — 27.4% improvement from founding baseline, and `val_re_rand=84.87` is the strongest per-split evidence for cross-Re encoding benefit. Needs retesting on current HEAD.

---

## 2026-04-29 — PR #909 (MERGED): Pre-block FiLM conditioning
- **Branch:** `willowpai2e3-thorfinn/film-pre-block`
- **Hypothesis:** Move FiLM modulation from post-block (after attention+MLP residuals) to pre-block (block input). Pre-block conditioning lets Q/K/V compute attention scores on Re-conditioned features, giving regime-aware attention patterns rather than just output rescaling.
- **Run:** W&B `x7hi1qun`, 14 epochs, 32.3 min, group `film-pre-block`, params 704,919 (unchanged from post-block baseline)

| Split | val baseline (post-block) | val pre-block | Δ val | test baseline | test pre-block | Δ test |
|---|---|---|---|---|---|---|
| `single_in_dist` | 95.54 | 93.70 | −1.84 | 81.63 | 84.20 | +2.57 |
| `geom_camber_rc` | 91.38 | 92.70 | +1.32 | 82.02 | 82.91 | +0.89 |
| `geom_camber_cruise` | 64.90 | **62.62** | **−2.28** | 53.62 | **51.93** | −1.69 |
| `re_rand` | 79.26 | **77.16** | **−2.10** | 71.82 | **70.54** | −1.28 |
| **avg** | 82.77 | **81.55** | **−1.22 (−1.5%)** | 72.27 | 72.40 | +0.13 |

### Decision: MERGED — small but mechanistically interpretable win
- **−1.5% val vs FiLM+L1 post-block baseline** (82.77 → 81.55). Meets merge criterion.
- **Test flat (+0.2%)** — single-seed noise territory; not a regression.
- Mechanism confirmed: Re-targeted splits (re_rand −2.10 val/−1.28 test, cruise −2.28/−1.69) improved as predicted by the "regime-aware attention pattern" hypothesis. Pre-block conditioning changes *which* features are attended to, not just how attended features are scaled.
- Mixed signal: `val_geom_camber_rc` +1.32 and `test_single_in_dist` +2.57 regressions. Interpretation: early-block features for in-dist geometry don't need Re-modulation; applying FiLM to all 5 blocks may over-condition the early representation.
- Val curve still falling at epoch 14 (ep13→ep14: −0.56, steep drop ep12→13 of −5.3). Indicates more headroom with either longer training or architectural refinement.
- **New beat-threshold: val_avg/mae_surf_p < 81.55**

---

## 2026-04-28 22:20 — PR #761 (MERGED): L1 surface MAE loss
- **Branch:** `willowpai2e3-tanjiro/l1-surface-mae-loss`
- **Hypothesis:** Replace surface MSE (then Huber) with pure L1 (MAE) to align training objective directly with `mae_surf_p` metric and exploit pressure's heavy-tailed residual distribution. Predicted −5 to −12% gain.
- **Run (v1-rebased):** W&B `tirux1y1`, **14/14 epochs (clean finish, cosine → 0)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 109.65 | 96.33 |
| `*_geom_camber_rc` | 101.17 | 90.80 |
| `*_geom_camber_cruise` | **72.37** | **61.90** |
| `*_re_rand` | **87.33** | **82.29** |
| **avg** | **92.63** | **82.83** |

### Decision: MERGE PENDING — 2nd rebase required (conflict with #814)
- **−10.2% val vs Huber-merged baseline (103.13 → 92.63)** — new branch best.
- **−10.9% test (92.99 → 82.83)** — new best test metric.
- L1 beats Huber(delta=1.0) by 10%: pure linear gradient on all surface residuals outperforms the smooth-near-zero Huber transition for this dataset's heavy-tailed pressure distribution.
- V1-rebased vs v1-timeout: −15.4% val — bulk of gain is from completing the cosine LR schedule (v1 was at ~95% of peak LR at timeout; v1-rebased reached LR=0 at epoch 14). Loss-shape gain is real but proper schedule alignment amplified it.
- `val_re_rand=87.33`, `test_re_rand=82.29` — exceptional regime-generalization numbers.
- Surface loss dominates volume ~7:1 at convergence (tanjiro's own diagnosis). 2nd rebase resolved cleanly — single block swap, no logic change. Merged 2026-04-28.
- **New beat-threshold: val_avg < 92.63**
- **Round-3 follow-up assigned: #869** (`surf_weight=3.0` rebalancing sweep; tanjiro).

---

## 2026-04-28 22:55 — PR #751 v2 (CLOSED): Dropout 0.05 + stoch-depth 0.05 on L1 base
- **Branch:** `willowpai2e3-fern/dropout-stochastic-depth`
- **Hypothesis:** v2 — halve regularization (`dropout=0.05, drop_path_max=0.05`) on the merged L1 advisor branch. Goal: keep v1's OOD-generalization wins without v1's in-dist regression.
- **Run (v2):** W&B `pbsbkt4g`, **14/14 epochs (clean cosine→0)**, peak ~43 GB.

| Split | Baseline (L1, #761) | v2 | Δ |
|---|---:|---:|---|
| `val_single_in_dist` | 109.65 | **107.85** | −1.6% |
| `val_geom_camber_rc` | 101.17 | 101.77 | +0.6% |
| `val_geom_camber_cruise` | 72.37 | 74.77 | +3.3% |
| `val_re_rand` | 87.33 | 88.26 | +1.1% |
| **val_avg** | **92.63** | **93.16** | **+0.6% (within noise)** |
| **test_avg** | **82.83** | **84.20** | **+1.7%** |

### Decision: CLOSE — hypothesis falsified
- v2 within ±10% noise band but in the wrong direction (+0.6% val).
- v1's OOD wins (-5% on geom_camber) **vanished** at half-strength regularization. Per-split sign pattern flipped: in-dist slightly improved, OOD slightly worse.
- **L1 surface loss already captures the OOD-generalization signal** that motivated regularization. Heavy-tailed pressure residuals are now handled by loss shape; regularization had nothing left to add.
- Student's own analysis correctly identified this and offered close-or-ablate. Choosing close: single-mechanism ablation (drop_path-only or dropout-only) is unlikely to beat L1 baseline given the v1→v2 trajectory.
- **Closed 2026-04-28. Next assignment: #902 volume-l1 (mirror surface L1 success on the volume side).**

### Side observation flagged
Fern noted `test_geom_camber_cruise/vol_loss = Infinity` even with #807's torch.where for sq_err. This is because `evaluate_split`'s vol_loss accumulator doesn't have explicit sample-level non-finite-y skip (only #807's element-wise mask). Doesn't affect MAE metrics (which use scoring.py's accumulator). Worth a future hardening pass but not blocking.

---

## 2026-04-28 22:40 — PR #847 (CLOSED): Huber surface loss delta sweep (delta=0.5, 2.0)
- **Branch:** `willowpai2e3-askeladd/huber-delta-sweep`
- **Hypothesis:** Tighten Huber delta from 1.0 to 0.5 (closer to L1) for tighter L1-like gradient on most residuals. Predicted -2 to -5%. Optional delta=2.0 sensitivity probe.
- **Runs:** W&B `2hi8vsek` (delta=0.5) and `ua6tw32g` (delta=2.0), 14/14 epochs both, full cosine cycles.

| delta | val_avg/mae_surf_p | test_avg/mae_surf_p |
|-------|-------------------:|--------------------:|
| 2.0 (probe) | 106.78 | 95.11 |
| 1.0 (#814 baseline) | 103.13 | 92.99 |
| 0.5 (this PR) | **102.97** | **91.21** |
| **L1 (delta→0, merged #761)** | **92.63** | **82.83** |

### Decision: CLOSE
- delta=0.5 just clears the OLD beat-threshold (103.13 → 102.97 = −0.16% on val). But during the run, **PR #761 L1 merged** with val_avg=92.63 — the new baseline.
- delta=0.5 is **+11.2% worse than current best**. Merging it would code-revert from L1 to Huber(0.5), worsening the model.
- **Loss-shape sensitivity curve confirmed monotonic in 0.5-2.0 range** but flat (3.6% spread). The big lever was MSE→Huber (#814: −15.6%); within the Huber regime, delta tuning hits diminishing returns.
- The Huber→L1 step (delta 0.5 → 0) accounts for −9.9%, much bigger than the entire Huber-delta sensitivity range. **Heavy-tailed pressure residual diagnosis confirmed all the way to the L1 limit.**
- Askeladd's analysis was correct on the trend (smaller delta better) but used the *pre-rebase* tanjiro number (109.53) as the L1 reference; the merged L1 is 92.63.
- **Closed 2026-04-28. Next assignment: #884 RevIN output normalization.**

---

## 2026-04-29 00:10 — PR #902 (CLOSED): Volume L1 — mirror surface L1 mechanism on volume side
- **Branch:** `willowpai2e3-fern/volume-l1`
- **Hypothesis:** Switch volume loss from MSE to L1 to mirror PR #761 surface success. Volume pressure has heavy-tailed residuals (boundary layer spikes, wake structure); predicted −2 to −7% on val_avg with bigger gain on val_vol_p.
- **Run (v1):** W&B `p4wzubwe`, **14/14 epochs (clean cosine→0)**, val curve flat by ep 12-13, peak ~92.1 GB. Run on L1 baseline (pre-FiLM).

| Metric | L1 baseline `tirux1y1` | v1 `p4wzubwe` | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p** | **92.63** | **96.52** | **+4.2%** |
| val_avg/mae_vol_p | 103.16 | 101.36 | −1.7% (within noise) |
| **test_avg/mae_surf_p** | **82.83** | **87.14** | **+5.2%** |

### Decision: CLOSE — mechanism falsified
- **Surface metric regresses uniformly** across all 4 splits (single_in_dist +7.4%, geom_camber_rc +3.5%, cruise +2.1%, re_rand +2.8%).
- **Volume gain only +1.7%** — well within ±10% single-seed noise band. The heavy-tailed-volume-pressure hypothesis is not supported.
- Val curve flat by epoch 12-13 — not under-trained.
- **Mechanism: gradient-magnitude rebalancing.** Switching vol loss from MSE (proportional gradient) to L1 (constant gradient) shifts the relative surf/vol contribution at fixed surf_weight=10. Surface optimization suffers because the gradient balance the optimizer was tuned to is broken.
- **Plus** the bulk of volume nodes are far-field/smooth-wake (Gaussian-ish residuals), where MSE is theoretically optimal. L1 wastes gradient on these well-behaved regions; the heavy-tail sliver (BL, stagnation streamlines) is too small to offset the damage.
- Plus: PR #815 FiLM+L1 merged at 82.77; v1 result 96.52 is +16.6% above current best — clear regression even if v1's L1-baseline criteria had been met.
- Student's analysis was decisive and lays out four candidate refinements; we're picking #4 (per-channel volume L1) as next assignment.
- **Closed 2026-04-29. Next assignment: #927 per-channel volume loss** (L1 on p only, MSE on Ux/Uy — isolates heavy-tail to channel where it might matter, preserves MSE balance on Gaussian-ish velocity channels).

---

## 2026-04-29 00:00 — PR #869 v1: surf_weight sweep (sw=3, sw=5) on L1 baseline
- **Branch:** `willowpai2e3-tanjiro/surf-weight-sweep`
- **Hypothesis:** Reduce `surf_weight` from 10.0 → 3.0 (and 5.0 bracket) to rebalance surface/volume gradient ratio. Predicted: −1 to −4% val_avg; volume-side gain larger.
- **Runs:** W&B `bffsf626` (sw=5) and `4q4b5kzs` (sw=3), 14/14 epochs both, ran on L1 baseline (pre-FiLM merge).

| metric | sw=10 baseline | **sw=5** | sw=3 |
|---|---|---|---|
| val_avg/mae_surf_p | 92.63 | **91.16** (−1.6%) | 95.55 (+3.2%) |
| test_avg/mae_surf_p | 82.83 | **81.36** (−1.8%) | 84.32 (+1.8%) |
| val_avg/mae_vol_p | 103.16 | **92.49** (−10.4%) | 94.57 (−8.3%) |

### Decision: SEND BACK FOR REBASE (onto FiLM+L1)
- **Mechanism directionally confirmed:** lowering surf_weight frees gradient budget → volume features develop more fully (val_vol_p −10.4%) → surface predictions also benefit (−1.6%) via shared trunk. Volume gain dominates by 7×, exactly as predicted.
- sw=5 wins on L1 baseline; sw=3 over-corrects on surface side (still in surface-dominant regime).
- BUT: PR #815 FiLM+L1 just merged at 82.77. sw=5 result of 91.16 is +10.1% above current best.
- Mechanism may stack with FiLM (different mechanisms: hidden-state Re modulation vs loss balance) OR overlap (FiLM strengthens volume features, reducing rebalancing benefit). Empirical answer matters.
- **Sent back:** rebase onto FiLM+L1 advisor; re-run sw=5 primary + optional sw=7 bracket (in case FiLM shifts the optimum toward sw=10).
- New beat-threshold for v2: **val_avg < 82.77**.
- **Round-3 follow-ups queued:** per-channel surface weighting (Ux/Uy/p within surface loss), longer LR schedule.

---

## 2026-04-29 00:00 — PR #750 v2-rebased (CLOSED): LR warmup + cosine on FiLM+L1
- **Branch:** `willowpai2e3-edward/lr-warmup-cosine`
- **Hypothesis history:** v1 (lr=1e-3, 50ep schedule, timeout-cut at 14ep) — pre-rebase: −19.6% on weak baseline mean. v2 (lr=2e-3, 14ep matched schedule) — pre-rebase: 111.12 (−9.0% vs founding). v2-rebased on FiLM+L1: see below.
- **Run (v2-rebased):** W&B `gqc006f6`, **14/14 epochs (clean finish)**, peak 99.13 GB on 96 GB card.

| Split | FiLM+L1 baseline | v2-rebased | Δ |
|---|---|---|---|
| `val_single_in_dist` | 95.54 | 97.09 | +1.6% |
| `val_geom_camber_rc` | 91.38 | **98.56** | **+7.9%** |
| `val_geom_camber_cruise` | 64.90 | 64.94 | +0.1% |
| `val_re_rand` | 79.26 | 79.69 | +0.5% |
| **val_avg** | **82.77** | **85.07** | **+2.78%** |
| **test_avg** | **72.27** | **74.33** | **+2.85%** |

### Decision: CLOSE — schedule mechanism baked into baseline
- **+2.78% val regression** (within single-seed noise but worse than baseline).
- **`val_geom_camber_rc` regresses +7.9%** — highest LR (lr=2e-3) interacts poorly with FiLM's per-block γ/β modulation in the OOD-camber regime. The schedule and conditioning mechanism are touching the same training dynamics.
- Student's honest analysis: pre-rebase 19.6% gain came from fixing schedule mismatch (cosine over 50ep but timeout at 14ep). Once `--epochs 14` became standard for all assignments (after PR #761), that gain was folded into baseline.
- The schedule-budget alignment principle survives the close: it's now the convention for every assignment in this branch (use `--epochs 14` so cosine actually anneals).
- Iterating on lower LR (lr=1e-3 or 7.5e-4) is low-EV at this point — would land within ~2-5% of baseline at most, not a mechanistic test.
- **Closed 2026-04-29. Next assignment: #924 per-channel output heads** (decouple Ux/Uy/p decoder pathways).

---

## 2026-04-28 23:50 — PR #884 (CLOSED): RevIN — per-sample y normalization for surface loss
- **Branch:** `willowpai2e3-askeladd/revin-output-norm`
- **Hypothesis:** Per-sample, per-channel target-stat normalization of pred and y before computing surface L1 loss. Goal: equalize per-Re-sample gradient contribution; predicted −3 to −10% with biggest gains on `val_re_rand`.
- **Run (v1):** W&B `99ltqjj3`, **14/14 epochs (clean finish)**, val_avg still descending but converging slowly, peak 42.2 GB.

| Split | val surf_p (RevIN) | val surf_p (L1 baseline) | Δ |
|---|---|---|---|
| `single_in_dist` | 221.78 | 109.65 | **+102%** |
| `geom_camber_rc` | 166.30 | 101.17 | +64% |
| `geom_camber_cruise` | 98.42 | 72.37 | +36% |
| `re_rand` | 124.04 | 87.33 | +42% |
| **val_avg** | **152.64** | **92.63** | **+65%** |
| **test_avg** | **141.19** | **82.83** | **+70%** |

### Decision: CLOSE — structurally mismatched mechanism
- **+65% val / +70% test vs L1 baseline** — not noise; clean failure.
- **Mechanism falsified.** Student's own analysis: "RevIN's per-sample normalization breaks the agreement between optimizer and metric. The gradient now treats every sample as unit-variance, so the model under-fits the high-amplitude samples that the metric weights most."
- The metric (`mae_surf_p`) is in physical units and high-Re-dominated by Re² scaling. Per-sample loss normalization decouples training gradient from metric weighting.
- **Splits regress in proportion to amplitude** (single_in_dist +102% > re_rand +42% > cruise +36%) — diagnostic of the failure mode. High-amplitude splits suffer most because RevIN under-fits them most.
- **Target-metric (`val_re_rand`) regressed +42%** — even on its own predicted-best signal, the mechanism failed.
- RevIN paper assumed scale-invariant metric (relative RMSE, normalized error). For absolute physical-units MAE this approach is structurally incorrect.
- **Closed 2026-04-28. Next assignment: #917 Re-input noise augmentation** (smooth FiLM conditioning via training-time log(Re) perturbation).

### Side observation flagged
Askeladd noted `test_geom_camber_cruise/loss = inf` artifact in `evaluate_split` (single test batch with tiny per-sample y_surf_var amplifying residual). Same vol_loss accumulator NaN-safety gap fern flagged earlier. Doesn't affect MAE metrics but worth a future hardening pass.

---

## 2026-04-28 23:35 — PR #815 v2-on-l1 (MERGED): FiLM conditioning per-block on log(Re) + L1
- **Branch:** `willowpai2e3-thorfinn/film-re-conditioning`
- **Hypothesis:** FiLM (γ, β) per-block conditioning on log(Re), rebased onto post-#761 L1 advisor. Predicted stacking: L1 (loss shape) ⊥ FiLM (hidden-state Re modulation) → both gains compound.
- **Run (v2-on-l1):** W&B `mfjoux5g`, **14/14 epochs (clean finish)**, val_avg still falling at epoch 14 (new best at final epoch), peak 44.6 GB. +42.5K params (+6.4%).

| Split | L1 baseline `tirux1y1` | FiLM v2 `mfjoux5g` | Δ |
|---|---|---|---|
| `val_single_in_dist` | 109.65 | **95.54** | **−12.9%** |
| `val_geom_camber_rc` | 101.17 | **91.38** | **−9.7%** |
| `val_geom_camber_cruise` | 72.37 | **64.90** | **−10.3%** |
| `val_re_rand` | 87.33 | **79.26** | **−9.2%** |
| **val_avg** | **92.63** | **82.77** | **−10.6%** |
| **test_avg** | **82.83** | **72.27** | **−12.7%** |

### Decision: MERGED (2026-04-28) — new branch best
- **−10.6% val (92.63 → 82.77)** — beats advisor's predicted target of 89.85 by 7.9%. Every split improved.
- **−12.7% test (82.83 → 72.27)** — new test best.
- **All 4 val splits improved this time** — including `val_geom_camber_rc` (+9.7%), which regressed in v1b. v1b regression was single-seed noise or a Huber/L1 interaction; L1 removed it.
- `val_re_rand` −9.2%, `val_geom_camber_cruise` −10.3% — FiLM-targeted splits still show strongest relative gains (widest Re ranges), confirming the per-block Re-modulation mechanism.
- L1 + FiLM stack constructively as predicted (orthogonal mechanisms: loss shape vs hidden-state Re conditioning).
- `(1+γ)·h+β` zero-init worked perfectly — stable training throughout, FiLM started as identity.
- **New beat-threshold: val_avg < 82.77**
- **Follow-up assigned: #909** (pre-block FiLM: condition attention input rather than output; thorfinn).

---

## 2026-04-28 23:35 — PR #858 (CLOSED): Focal surface loss gamma=1.0 (L1 base)
- **Branch:** `willowpai2e3-nezuko/focal-surface-loss`
- **Hypothesis:** Apply focal-style node weighting `(per-node-err/mean-err)^gamma` on top of L1 surface loss to concentrate gradient on high-error surface nodes (stagnation, suction peak, leading-edge curvature). Predicted −2 to −7%.
- **Runs (v1-l1-gamma05 and v1-l1-gamma10):** W&B `6f8lwss4` (γ=0.5) and `ulobuh9d` (γ=1.0), 14/14 epochs both.

| gamma | val_avg/mae_surf_p | vs L1 baseline | conclusion |
|---|---|---|---|
| 1.0 | 105.06 | **+13.4%** | substantially worse |
| 0.5 | 92.13 | −0.5% | within noise |
| **L1 baseline (#761)** | **92.63** | — | reference |

Per-split val (gamma=0.5): single_in_dist=105.07, geom_camber_rc=100.80, cruise=74.90, re_rand=87.74.

### Decision: CLOSE — mechanism falsified
- **gamma=0.5 (92.13) is within single-seed noise** — not a reliable improvement. The −0.5% delta is well within the ±10% single-seed noise band.
- **gamma=1.0 is substantially worse** (+13.4%): ~10% of nodes get >2× weight, worst single node ~35×. Noisy minibatch gradients on nodes the model cannot yet drive to zero slows convergence.
- **Mechanism is incorrect at this budget**: high-error nodes are not gradient-bottlenecked; they are capacity/convergence-time-bottlenecked. Focal amplification makes convergence worse, not better.
- Student's own diagnosis nailed it: "amplifying their loss only makes that worse." Clean mechanistic falsification.
- **Additional decisive point:** PR #815 just merged at val_avg=82.77 — gamma=0.5's 92.13 is now +11.3% above new baseline.
- **Focal-weight distribution stats** (student measured): gamma=0.5 max=4.7×, p99=2.4×, frac>2×=2.3%; gamma=1.0 max=34.9×, p99=8.2×, frac>2×=10.3%. Stats confirm the tail is moderately concentrated but not gradient-starved.
- **Closed 2026-04-28. Next assignment: #910 Re-stratified batch sampling.**

---

## 2026-04-28 22:30 — PR #815 v1b: FiLM conditioning per-block on log(Re)
- **Branch:** `willowpai2e3-thorfinn/film-re-conditioning`
- **Hypothesis:** Add FiLM (γ, β) conditioning on log(Re) to each Transolver block — explicit per-layer regime adaptation across Reynolds ranges. Predicted −5 to −15% with biggest gains on Re-stratified and OOD-camber splits.
- **Run (v1b):** W&B `ujkwztbk`, **14/14 epochs (clean finish)**, val_avg still falling at epoch 14, peak 44.6 GB. +42.5K params (+6.4% over baseline). Pre-#761 branch (no L1).

| Split | Baseline `8cvp4x6r` | FiLM v1b | Δ |
|---|---|---|---|
| `val_single_in_dist` | 143.36 | 141.12 | −1.6% |
| `val_geom_camber_rc` | 124.20 | 132.08 | +6.3% |
| `val_geom_camber_cruise` | 109.42 | **93.92** | **−14.2%** |
| `val_re_rand` | 111.63 | **106.87** | **−4.3%** |
| **val_avg** | **122.15** | **118.50** | **−3.0%** |
| **test_avg** | NaN (pre-fix) | **107.76** | n/a |

### Decision: SEND BACK FOR REBASE (onto post-#761 advisor)
- **Hypothesis directionally confirmed.** The two splits FiLM should help most are exactly the ones that improved: `val_re_rand` (Re-stratified holdout) and `val_geom_camber_cruise` (widest Re range, 110K-5M). `−14.2%` on cruise is the largest single-split gain seen on this branch from any architecture experiment.
- **Absolute number doesn't beat new baseline (92.63).** Branch predates #761 L1 merge. v1b at 118.50 is +27.9% above current best, but FiLM is orthogonal to L1's loss-shape change — should stack.
- **Single split regressed** (`val_geom_camber_rc` +6.3% / +9.2% test). Single-seed in narrow-Re-band domain — likely noise but flagged for v2 follow-up.
- val curve still falling at epoch 14 — gain has more to give with rebase.
- `(1+γ)·h+β` zero-init worked: training stable through every epoch, FiLM started as identity.
- **Sent back:** rebase onto current advisor (which has L1), no code changes to FiLM itself, re-run with `--epochs 14`. Predicted: 92.63 × 0.97 ≈ 89.85 if mechanisms stack.

### Round-3 follow-ups queued (post-merge)
1. **Pre-block FiLM** — modulate hidden state *before* attention/mlp blocks (currently after).
2. **Layer-targeted FiLM** — only last 2 blocks; tests whether full per-block is necessary.
3. **Investigate `val_geom_camber_rc` regression** — multi-seed re-run of baseline + FiLM; could be real coupling issue or noise.

---

## 2026-04-28 22:25 — PR #743 v2: Per-channel surface loss [1.0, 0.5, 2.0] on Huber base
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Apply per-channel weights `[Ux=1.0, Uy=0.5, p=2.0]` to the surface loss to align training emphasis with the `mae_surf_p` metric. v1 was blocked by NaN poisoning; v2 adds the Huber base (matching PR #814) with normalized channel weighting.
- **Run:** W&B `2tj1e31r`, **14/14 epochs (clean)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 115.33 | 98.01 |
| `*_geom_camber_rc` | 109.49 | 99.81 |
| `*_geom_camber_cruise` | **77.86** | **66.60** |
| `*_re_rand` | **94.18** | **89.40** |
| **avg** | **99.21** | **88.45** |

### Decision: SEND BACK FOR REBASE (onto post-L1-merge advisor)
- **−3.80% val / −4.88% test vs Huber baseline (103.13 / 92.99)** — genuine compound gain over Huber.
- BUT: new baseline is tanjiro L1 (92.63 / 82.83). Alphonse's 99.21 is +7.1% WORSE than L1 alone.
- Channel weighting on top of Huber is confirmed to stack constructively. Natural v3 question: does it also compound on top of L1?
- Prediction: if same -3.8% stacks on L1 → 92.63 × 0.962 ≈ 89.1 val — would be a meaningful improvement.
- **Sent back** with instructions to: rebase onto post-#761 advisor; replace `F.huber_loss(reduction='none')` with per-element `abs_err * surf_chan_w` in both training loop and evaluate_split.
- New beat-threshold for v3: **val_avg < 92.63**.

---

## 2026-04-28 21:28 — PR #814 (MERGED): Huber surface loss (delta=1.0)
- **Branch:** `willowpai2e3-askeladd/huber-surf-loss`
- **Hypothesis:** Replace MSE surface loss with Huber(delta=1.0) to align training objective with MAE metric and gain robustness against heavy-tailed pressure errors. Predicted -5 to -10%.
- **Run:** W&B `at52zeu5`, **14/14 epochs (clean)**, best ckpt @ epoch 14, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 123.94 | 109.10 |
| `*_geom_camber_rc` | 111.30 | 98.88 |
| `*_geom_camber_cruise` | 81.66 | **69.84** |
| `*_re_rand` | **95.62** | **94.17** |
| **avg** | **103.13** | **92.99** |

### Decision: MERGED (2026-04-28) — new leading winner
- **−15.6% val (122.15 → 103.13)** — well outside noise band; beats tanjiro L1 (109.53).
- **−29% test (130.90 → 92.99)** — spectacular test improvement; founding clean test baseline.
- Hypothesis confirmed: L1-like gradient tail robustness on surface pressure with MSE stability near zero.
- Val still falling at epoch 14 (−1.5 from ep13→14) — headroom remains at longer budget.
- Student noted: `reduction='mean'` implicitly weakens surf_weight by ~3× vs old `sum/N_nodes` form; despite this the metric improved — loss shape is doing the work.
- **New beat-threshold: val_avg < 103.13**
- **Follow-up assigned: #847** (huber-delta-sweep: try delta=0.5 to push closer to L1).

## 2026-04-28 19:55 — PR #743: Per-channel surface loss: 3× weight on pressure
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Boost pressure channel by 3× inside surface loss to align training signal with the `mae_surf_p` ranking metric. `y_std_p ≈ 679`, ~30× larger than `y_std_Ux` and ~70× larger than `y_std_Uy`; uniform-weighted MSE under-emphasizes the metric channel.
- **Run:** W&B `zaqz12qi` (entity `wandb-applied-ai-team`, project `senpai-charlie-wilson-willow-e-r3`)
- **Budget consumed:** 14/50 epochs (hit `SENPAI_TIMEOUT_MINUTES=30` cap; ≈131 s/epoch)

### Results

| Split | val (best ckpt @ epoch 14) | test (best ckpt @ epoch 14) |
|---|---|---|
| `*_single_in_dist` | 196.59 | 166.63 |
| `*_geom_camber_rc` | 156.43 | 141.34 |
| `*_geom_camber_cruise` | **107.40** | **null** |
| `*_re_rand` | 124.01 | 122.96 |
| **avg** | **146.10** | **null (cruise NaN propagates)** |

### Analysis & decision: SEND BACK
- Val side is informative: `val_avg/mae_surf_p = 146.10`, with `cruise = 107.40` the best of the four val splits — consistent with the hypothesis that p-channel boost helps where p dominates surface dynamics.
- **Test side blocks merge.** `test_geom_camber_cruise/mae_surf_p = null` (single non-finite prediction polluting the global accumulator in `accumulate_batch` — `data/scoring.py` only skips on non-finite ground truth, not non-finite preds). Three of four test splits finite. Per CLAUDE.md, NaN/missing on the paper-facing metric blocks adoption.
- Budget reality check: at default `SENPAI_TIMEOUT_MINUTES=30` and current model size, only ~14 epochs fit. **All round-1 PRs are timeout-limited to ~14 epochs**, not 50. Future hypothesis design should account for this — recommend setting `--epochs 14` explicitly so cosine annealing reaches end-of-curve LR rather than mid-curve.
- Sent back with feedback to:
  1. Add a NaN-guard / clamp in `evaluate_split` (`pred = torch.nan_to_num(pred, ...).clamp(-20, 20)` before denormalization) — defends MAE numerics for all future students once merged.
  2. Try softer per-channel weights `[1.0, 0.5, 2.0]` instead of `[1.0, 1.0, 3.0]` — 2× boost on p (closer to variance ratio after surface gating absorbs most of it) plus 0.5× on Uy (over-represented and not in ranking metric).
  3. Set `--epochs 14` explicitly to plan for the timeout.
- Once v2 lands with finite `test_avg/mae_surf_p`, this becomes the founding baseline for the branch.

### Cross-cutting findings (apply to ALL round-1 students)

- **Timeout is the binding constraint, not epoch count.** Plan for 14 epochs, not 50.
- **NaN test poisoning is a real `data/scoring.py` bug, not a model issue.** Identified by askeladd in PR #748 (since closed): `accumulate_batch` does `err * mask` where `NaN * 0 = NaN` in IEEE-754. `test_geom_camber_cruise/sample 20` has 761 NaN values in the **ground truth** `p` channel. This poisons `mae_surf_p` for that split on every run regardless of model. Fix: `torch.where(mask, err, 0)`. Same pattern in `evaluate_split` for `vol_loss`/`surf_loss` produces `Infinity`. Assigned to askeladd as PR #807. Once that lands, all future runs will produce clean `test_avg` numbers.
- **Cruise OOD camber (M=2-4)** is otherwise the most extrapolation-prone test split — already the hardest extrapolation track regardless of the NaN bug.

## 2026-04-28 20:02 — PR #750: Linear warmup + cosine LR schedule (lr=1e-3, wd=5e-4)
- **Branch:** `willowpai2e3-edward/lr-warmup-cosine`
- **Hypothesis:** 500-step linear warmup + per-step cosine to 0, lr=1e-3, wd=5e-4 — should buy 3–8% over plain cosine.
- **Run:** W&B `thnnvgaw`, 14/50 epochs (timeout), best ckpt @ epoch 12, peak 83.6 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 162.00 | 144.47 |
| `*_geom_camber_rc` | 148.21 | 136.08 |
| `*_geom_camber_cruise` | 102.53 | null (scoring bug) |
| `*_re_rand` | 130.84 | 127.33 |
| **avg** | **135.89** | **null** |

### Decision: SEND BACK (v1)
- 1.7% better than student's quoted baseline-mean reference (5 runs, range [124.6, 146.1]) — within noise; the hypothesized 3–8% gain isn't demonstrated.
- Student's diagnosis is exactly right: cosine schedule with `T_max = 50 × 375 ≈ 18.7K steps` but only ~5.2K steps fit in 30 min — never reaches the low-LR fine-tuning regime.
- Sent back: set `--epochs 14` explicitly so cosine actually anneals end-to-end; raise peak LR to `2e-3` (warmup makes higher peaks safe — that's where the standard transformer warmup gain lives).
- (Branch hygiene: edward referenced 5 baseline runs t0xgo0zv/6zc9kq6x/6lj642bf/7qi7tbcy/zaqz12qi as a noise band; only zaqz12qi (alphonse v1) is from this advisor branch, the others are out-of-scope. The variance argument stands; the specific run-IDs do not.)

### v2 Results (2026-04-28 20:50) — `lr=2e-3, epochs=14, warmup_steps=500`
- **Run:** W&B `mv16jwsp`, **14/14 epochs (clean finish, no timeout)**, best ckpt @ epoch 14, peak 94.24 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 133.38 | 116.37 |
| `*_geom_camber_rc` | 120.79 | 112.57 |
| `*_geom_camber_cruise` | **85.96** | null (pre-#807 run) |
| `*_re_rand` | 104.37 | 101.46 |
| **avg** | **111.12** | **null (re-eval pending)** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−9.0% vs founding baseline (122.15 → 111.12)** — well outside round-1 noise band.
- All 4 val splits improved uniformly (-17 to -20% vs v1) — gain is from genuine convergence, not split-specific quirk.
- Best epoch landed exactly at 14/14 with cosine asymptote reached; flat last-3 deltas (113.75 → 111.13 → 111.12) suggest near-asymptote for this config. Suggests v3 with lr=3e-3 or longer budget could buy more.
- **Conflict:** branch predates PR #807; both touched `train.py` near loss accumulation. Sent back to rebase + re-run (same config) to verify gain holds + pick up clean `test_avg`. Once verified ≤~115, will merge immediately.

## 2026-04-28 20:04 — PR #748: Transolver 2x capacity scale-up
- **Branch:** `willowpai2e3-askeladd/transolver-2x-capacity`
- **Hypothesis:** n_hidden=192, n_layers=8, n_head=8, slice_num=128, mlp_ratio=4 (3.42M params) — predicted 5–15% gain.
- **Run:** W&B `p486z24b`, **4/50 epochs only** (timeout), best ckpt @ epoch 3, peak 82.5 GB.
- **val_avg/mae_surf_p = 203.16** (raw); test_avg null (scoring bug); **test_avg corrected = 191.71** (offline re-eval with `torch.where`).

### Decision: CLOSE
- val_avg = 203 vs. round-1 baseline-range ~140 — clear regression at 4/50 epochs.
- Approach not broken — it's that 50-epoch cosine schedule with only 4 epochs done means LR is still ~98% of peak. Model never reached convergence regime where 2× capacity is supposed to help.
- **Critical bug discovery embedded in PR comments**: pinpointed the `data/scoring.py` NaN-mask bug, validated the fix offline. Spawned PR #807 to land the fix as their next assignment.
- For round 2 capacity: budget-matched schedule (`--epochs 4` so cosine completes), or smaller capacity boost (n_layers=6 to fit ~8 epochs) — defer until scoring fix merges.

## 2026-04-28 20:15 — PR #756: Fourier features for log(Re) input encoding
- **Branch:** `willowpai2e3-frieren/fourier-re-encoding`
- **Hypothesis:** Replace scalar `log(Re)` (dim 13) with 16 sin/cos features at 8 freqs `[1, 2, 4, 8, 16, 32, 64, 128]` for richer cross-Re generalization. Predicted 3-7% gain.
- **Run:** W&B `t0xgo0zv`, 14/50 epochs (timeout), best ckpt @ epoch 14, peak 42.3 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 202.03 | 169.41 |
| `*_geom_camber_rc` | 139.03 | 122.76 |
| `*_geom_camber_cruise` | 102.50 | null (scoring bug) |
| `*_re_rand` | 121.42 | 123.83 |
| **avg** | **141.25** | **null** |

### Decision: SEND BACK (v1)
- val_avg=141.25 sits inside the round-1 noise band (other v1 runs: alphonse 146.10, edward 135.89). Predicted 3-7% gain not demonstrable at single-seed.
- val_re_rand=121.42 (the strongest per-split) is suggestive — Fourier-of-log(Re) may help cross-Re generalization. Direction worth iterating on, not abandoning.
- val curve still falling steeply at the cutoff (epochs 11-14: 152→160→160→141) — model under-converged.
- Sent back: (a) concatenate Fourier features instead of replacing dim 13 (preserves smooth scalar path); (b) drop top frequencies, use `[1, 2, 4, 8, 16, 32]` (high freqs cycle below the data's Re resolution); (c) `--epochs 14` explicit so cosine completes.

### v2 Results (2026-04-28 21:04) — concat + 6 freqs + `--epochs 14`
- **Run:** W&B `tg59rxt1`, **14/14 epochs (clean finish)**, best ckpt @ epoch 14, peak 42.3 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 147.07 | 126.45 |
| `*_geom_camber_rc` | 125.55 | 119.19 |
| `*_geom_camber_cruise` | 97.37 | null (pre-#807 run) |
| `*_re_rand` | **110.89** | **111.10** |
| **avg** | **120.22** | **null (re-eval pending)** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−1.6% vs founding baseline (122.15 → 120.22)** — small win but positive signal.
- v2 vs v1: −14.9% improvement; uniform across all 4 val splits and all 3 finite test splits.
- **Strongest signal on `val_re_rand=110.89` and `test_re_rand=111.10`** — exactly the cross-Re generalization track the encoding was designed for. This is the genuine encoding gain.
- Frieren correctly noted bundled changes: (a) concat encoding, (b) drop high freqs, (c) `--epochs 14`. Code-level changes are only (a) and (b); (c) is just a runtime flag so the encoding hypothesis is honestly tested by the diff.
- val curve still descending at cutoff (124.65 → 123.74 → 123.26 → 120.22), under-converged.
- **Conflict:** branch predates PR #807; needs rebase. Sent back to rebase + re-run with same config (`--epochs 14`) to verify gain holds + pick up clean `test_avg`.
- Notable: edward's parallel PR #750 v2 (lr-warmup-cosine) hits val_avg=111.12 — a *better* winner via LR schedule changes. The two are mechanism-orthogonal; both can compound. If edward's PR merges first, frieren's encoding contribution is measured on top of edward's optimizer fix.

## 2026-04-28 21:09 — PR #761: L1 (MAE) surface loss aligned with metric
- **Branch:** `willowpai2e3-tanjiro/l1-surface-mae-loss`
- **Hypothesis:** Replace surface MSE with L1 (MAE) loss to align training objective directly with the `mae_surf_p` ranking metric and provide robustness to the heavy-tailed pressure distribution. Predicted -5 to -12% gain.
- **Run:** W&B `ee9p55qd`, 14/50 epochs (timeout), best ckpt @ epoch 13, peak 42.1 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 149.36 | 128.91 |
| `*_geom_camber_rc` | 107.87 | 100.02 |
| `*_geom_camber_cruise` | 82.64 | **70.62** |
| `*_re_rand` | **98.25** | **94.20** |
| **avg** | **109.53** | **98.44** |

### Decision: WINNER — but SEND BACK FOR REBASE (2026-04-28)
- **−10.3% vs founding baseline (122.15 → 109.53)** — well outside round-1 noise band.
- **Best winner on the branch**: ahead of edward (111.12) and frieren (120.22).
- **First clean test_avg** beating founding baseline (130.90 → 98.44, −24.6%) — tanjiro bundled their own sample-level NaN-guard which works equivalently to #807.
- Hypothesis directly confirmed: L1 metric-alignment + pressure-tail robustness. `val_re_rand=98.25` and `test_re_rand=94.20` are exceptional per-split numbers.
- val curve still improving at epoch 13 (epoch 14 worsened so checkpoint at 13). Suggests headroom on a longer/better-tuned schedule.
- Bundled NaN-guard in `evaluate_split` is redundant with #807's torch.where (and should be dropped on rebase).
- **Conflict:** branch predates #807; touched same loss-computation lines. Sent back to rebase + re-run with `--epochs 14` (cosine T_max alignment, like edward/frieren).
- Once rebased run lands at val_avg ≤ ~115, merge immediately. This is the single biggest gain on the branch.

### Queued follow-ups (post-merge)
1. surf_weight=3.0 — student's own diagnosis: L1 surface gradient currently dominates 10:1 vs vol_loss. Rebalancing should free volume capacity.
2. L1 on p only, MSE on Ux/Uy — pressure-tail alignment without changing dynamics for the better-behaved velocity channels. Possibly stacks with alphonse's channel-weighted-3xp.

---

## 2026-04-28 20:08 — PR #807 (MERGED): Bug fix — NaN-safe masked accumulation
- **Branch:** `willowpai2e3-askeladd/scoring-nan-mask-fix`
- **Type:** Infrastructure bug fix (not a hypothesis experiment)
- **Scope:** `data/scoring.py::accumulate_batch`, `train.py::evaluate_split`, training-loop loss accumulation.

### Results

Fix: replaced `(err * mask).sum()` with `torch.where(mask, err, zero).sum()` in all three locations. Mathematically equivalent for finite y; zeroes out NaN contributions rather than propagating them.

**Re-evaluated checkpoints (no retraining):**

| Run | Old test_avg/mae_surf_p | New (corrected) test_avg/mae_surf_p | Notes |
|-----|------------------------|--------------------------------------|-------|
| `zaqz12qi` (alphonse channel-weighted v1) | null | **130.897** | test_geom_camber_cruise=92.66 |
| `p486z24b` (askeladd transolver-2x) | null | 192.259 | test_geom_camber_cruise=175.62 (under-trained) |

`--debug` smoke test: identical outputs pre/post fix on clean training splits. New `scripts/reeval_artifact.py` helper included.

### Decision: MERGED (2026-04-28)
- Infrastructure fix unblocking finite `test_avg/mae_surf_p` for all future runs.
- Founding baseline established: `zaqz12qi` val_avg=146.10 / test_avg=130.897.
- Also added BASELINE.md to advisor branch anchored by thorfinn's matched-baseline run (8cvp4x6r, val_avg=122.15 — best unmodified model result in round 1).

---

## 2026-04-28 20:31 — PR #762 (CLOSED): Boundary-layer feature: log(Re·|saf|) as input
- **Branch:** `willowpai2e3-thorfinn/boundary-layer-features`
- **Hypothesis:** Add `log(Re·|saf|)` as a 25th input dimension to give an explicit local Re_x boundary-layer signal. Predicted -10 to -25% gain, strongest on `val_re_rand`.
- **Run:** W&B `7qi7tbcy` (BL feature), `8cvp4x6r` (matched baseline), 14/50 epochs (timeout), same GPU back-to-back.

### Results

| Split | Baseline (8cvp4x6r) | +log(Re·|saf|) (7qi7tbcy) | Δ |
|---|---|---|---|
| `val_single_in_dist` | 143.36 | 180.55 | **+25.9%** |
| `val_geom_camber_rc` | 124.20 | 159.73 | **+28.6%** |
| `val_geom_camber_cruise` | 109.42 | **95.96** | -12.3% |
| `val_re_rand` | 111.63 | 117.47 | +5.2% |
| **val_avg** | **122.15** | **138.43** | **+13.3% (WORSE)** |

Test splits: both have test_avg null (pre-fix runs); 3-split avg: baseline 118.01, BL 139.50 (+18.2% worse).

### Analysis & decision: CLOSE
- Consistent negative across 3/4 val splits and all 3 finite test splits. Cruise is the only win.
- **Information redundancy diagnosis (thorfinn):** dim-13 log(Re) and dims 2:3 saf are already in x; MLP preprocess can construct their product for free. Explicit feature likely competes with rather than augments existing capacity.
- **Volume-node saf mismatch:** saf is arc-length on surface nodes but undefined/different off-surface; broadcasting `log(Re·|saf|)` to all nodes injects physically wrong signal for volume nodes.
- Cruise improvement (-12.3%) is real but isolated (100 samples, test NaN-poisoned, single seed) and insufficient to outweigh the other regressions.
- **Exceeds the >5% close threshold** (13.3% regression). Closed 2026-04-28.
- Matched baseline `8cvp4x6r` (val_avg=122.15) promoted to BASELINE.md as best clean unmodified-model result.

### Cross-cutting: thorfinn's matched-baseline methodology
Thorfinn independently identified the data/scoring.py NaN bug (same root cause as askeladd). Excellent experimental practice: ran matched baseline side-by-side, produced full split breakdown, conducted 3-cause analysis. The matched baseline (122.15) reshapes our noise estimate — round-1 noise band is now 122–146, not 135–146.

## 2026-04-28 22:00 — PR #759 (CLOSED): EMA model weights (decay=0.999)
- **Branch:** `willowpai2e3-nezuko/ema-model-weights`
- **Hypothesis:** Apply Exponential Moving Average (decay=0.999) of model weights to reduce noise in surface pressure predictions; an EMA shadow model averages over the stochastic gradient trajectory, producing smoother predictions. Predicted −5 to −10% on val_avg/mae_surf_p.
- **Run:** W&B `qetkdsku`, 14/14 epochs, askeladd Huber-merged baseline (103.13) used as beat-threshold.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | — | — |
| `*_geom_camber_rc` | — | — |
| `*_geom_camber_cruise` | — | — |
| `*_re_rand` | — | — |
| **avg** | **124.51** | **110.63** |

### Analysis & decision: CLOSE
- val_avg=124.51 is **+20.7% worse** than current best (PR #814 Huber, 103.13) and **+1.9% worse** than founding baseline (122.15) — within round-1 noise but in the wrong direction vs. the current bar.
- **Regime mismatch (fundamental diagnosis):** EMA's benefit is maximal in the *converged-but-noisy* regime — late training where parameters have found a basin but stochastic gradients cause high-frequency jitter. At 14-epoch budget, Transolver is still descending the loss curve (the live weights are improving every epoch). EMA shadow weights (decay=0.999 means ~1000-step effective memory) are staler than the current live weights and drag the ensemble toward earlier, worse states rather than smoothing noise around a convergence point.
- **Supporting evidence:** nezuko's own analysis correctly identified this: "the training loss was still decreasing at epoch 14, which suggests the model hadn't fully converged and EMA might have been averaging over a range of improving but not yet optimal weights."
- **EMA is correctly motivated for a longer budget.** At ~50 epochs or with a lower LR tail, this hypothesis should be revisited. At current 14-epoch ceiling, EMA is a hindrance.
- Notable: nezuko independently discovered a NaN-guard variant during implementation — good diagnostic instinct. That NaN-guard is covered by the already-merged PR #807 (torch.where pattern), so no further action needed on that front.
- **Closed 2026-04-28. Next assignment: #858 focal-surface-loss (gamma=1.0).**

## 2026-04-28 22:05 — PR #751: Dropout 0.1 + stochastic depth (regularization)
- **Branch:** `willowpai2e3-fern/dropout-stochastic-depth`
- **Hypothesis:** Add `dropout=0.1` (PhysicsAttention + post-MLP) and linear stochastic depth `0 → 0.1` across the 5 TransolverBlocks. Predicted −3 to −8% with bias toward `geom_camber` OOD splits. Round 1 — pre-#807, pre-#814.
- **Run:** W&B `w5vmv84w` (v1) and `pv21nz5x` (matched baseline), 14/50 epochs (timeout), peak 43.4 GB.

### Results — v1 vs fern's matched baseline (pre-Huber)

| Split | Baseline `pv21nz5x` (val) | v1 `w5vmv84w` (val) | Δ val | Baseline (test) | v1 (test) | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| `single_in_dist` | **163.29** | 182.97 | **+12.0%** | **137.53** | 159.05 | **+15.6%** |
| `geom_camber_rc` | 137.79 | **130.92** | **−5.0%** | 129.92 | **119.45** | **−8.0%** |
| `geom_camber_cruise` | 117.49 | **111.63** | **−5.0%** | 99.89 | **93.99** | **−5.9%** |
| `re_rand` | **128.30** | 129.72 | +1.1% | **127.65** | 128.90 | +1.0% |
| **avg** | **136.72** | **138.81** | **+1.5%** | **123.75** | **125.35** | **+1.3%** |

### Decision: SEND BACK FOR REBASE + WEAKEN REGULARIZATION
- Hypothesis directionally confirmed on the OOD axis: `val_geom_camber_*` improves 5%, `test_geom_camber_*` improves 5.9–8%. That's exactly where regularization should help.
- But the in-distribution `single_in_dist` track regresses 12–15.6% — `dropout=0.1, drop_path_max=0.1` is too strong for a 5-layer network at 1499-sample scale; stochastic depth dropping whole residual paths destabilizes the easy in-dist fit.
- vs current best (PR #814 Huber, 103.13): v1 is +34.6% worse — a regression at the absolute level, BUT branch predates both #807 and #814 so the comparison isn't apples-to-apples.
- **Sent back with feedback to:** (a) rebase onto current advisor (gets Huber baseline); (b) drop the redundant `evaluate_split` y-NaN guard (already covered by #807); (c) halve both regularizers — `dropout=0.05, drop_path_max=0.05`; (d) `--epochs 14` explicit so cosine completes.
- v2 tests whether mild regularization compounds with Huber surface loss or whether the OOD-generalization signal is already saturated by the loss-shape change (Huber alone got `test_geom_camber_cruise` from 99.89 → 69.84).

### Independent NaN-bug discovery (notable)
Fern is the **fourth** student to independently identify the `data/scoring.py` NaN-poisoning bug (askeladd PR #807, tanjiro #761 inline guard, nezuko #759 inline guard, fern #751 inline guard). Fern's `evaluate_split` y-finite-per-sample mask uses a different surface (filter samples vs filter elements), but the underlying diagnosis matches. Cleanly handled at the `data/scoring.py` layer in #807 (now merged), so all student inline guards become redundant on rebase.
