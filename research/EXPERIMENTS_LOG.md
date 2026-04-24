# SENPAI Research Results

## 2026-04-23 21:40 — PR #3: frieren: Huber/L1 loss reformulation to close MSE-vs-MAE gap

- **Branch:** `frieren/loss-reformulation-v2`
- **W&B group:** `frieren/loss-reformulation-v2` (project: `wandb-applied-ai-team/senpai-kagent-v-students`)
- **Hypothesis:** L1/Huber loss in normalized space better matches the MAE ranking metric, and closes the MSE-over-penalizes-outliers gap for heavy-tailed pressure data.

### Results

| Rank | Config | val_avg/mae_surf_p | W&B run |
|------|--------|--------------------|---------|
| 1 | L1 / sw=10 | **103.036** ✓ WINNER | `w2jsabii` |
| 2 | L1 / sw=20 | 103.039 | `fdxlome0` |
| 3 | Huber δ=1.0 / sw=5 | 108.279 | `ugvqg0l4` |
| 4 | Huber δ=0.5 / sw=10 | 111.334 | `3ojxqu4l` |
| 5 | Huber δ=1.0 / sw=10 | 116.860 | `dgqcgzk3` |
| 6 | Huber δ=1.0 / sw=20 | 117.695 | `7fqbh9jz` |
| 7 | Huber δ=2.0 / sw=10 | 118.347 | `usrpv2wh` |
| 8 | MSE / sw=10 (baseline) | 131.985 | `azo8qcpu` |

**Per-split for winner (L1 sw10):**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 133.194 |
| val_geom_camber_rc | 117.209 |
| val_geom_camber_cruise | 70.109 |
| val_re_rand | 91.634 |
| **val_avg** | **103.036** |

### Analysis

- **−21.9% improvement over MSE baseline.** Wins on ALL four val splits — unanimous, not a single-split artifact.
- Huber δ-monotonicity is clean: smaller δ → closer to L1 → better. This independently confirms the mechanism: L1 shape better matches the MAE metric, and linear-not-quadratic loss better handles the heavy-tailed pressure distribution (−29K to +2.7K, per-sample y_std varying 10×).
- L1-sw10 ≈ L1-sw20 (103.036 vs 103.039): flat plateau at the current surf_weight, suggesting we may be at or beyond the optimal point on that axis.
- All runs timeout-bounded at epoch 11–14/50. Models were still improving — throughput is the next leverage point.
- `test_avg/mae_surf_p` is NaN on all runs due to +Inf in `test_geom_camber_cruise/000020.pt`. Filed as GH issue #10.

### Decision: **MERGED** into `kagent_v_students`. New baseline = 103.036 (L1, sw10).

---

## 2026-04-23 21:40 — PR #4: fern: Transolver capacity scaling — width/depth/slice-num sweep

- **Branch:** `fern/capacity-scaling-v2`
- **W&B group:** `fern/capacity-scaling-v2`
- **Hypothesis:** The baseline 0.7M-param model is under-capacity for 96 GB GPUs; scaling n_hidden/n_layers/slice_num should improve metric.

### Results

| Rank | Config (h/l/s) | n_params | val_avg/mae_surf_p | W&B run |
|------|---------------|----------|--------------------|---------|
| 1 | 128/5/64 (BASELINE) | 0.66 M | **126.87** | `nuilgeox` |
| 2 | 192/5/64 | 1.47 M | 147.53 | — |
| 3 | 256/5/128 | 2.62 M | 152.73 | — |
| 4 | 256/5/64 | 2.60 M | 154.00 | — |
| 5 | 256/7/64 | 3.56 M | 154.60 | — |
| 6 | 256/7/128 | 3.58 M | 157.94 | — |
| 7 | 384/5/128 | 5.85 M | 182.35 | — |
| 8 | 384/7/128 | 8.00 M | 185.45 | — |

### Analysis

- **Hypothesis falsified: baseline wins by 16–46%.** Larger models are NOT capacity-limited; they are severely undertrained. Baseline converged to best checkpoint at epoch 14, h384 variants at epoch 4. With a 30-min wall-clock, per-epoch time ≈ linear in (n_hidden × n_layers), so bigger models see only 4–9 epochs vs 14 for the baseline.
- No capacity advantage appears even controlling for this — at equal best_epoch, larger models are worse, consistent with "large model needs longer warmup and more steps than the budget allows."
- slice_num=64→128 at fixed h=256, l=5: val_avg changes only 1.27 (154.00 → 152.73) — lowest-leverage knob.
- **Root cause:** wall-clock throughput is the binding constraint, not parameters. The prerequisite to meaningful capacity comparison is unlocking more epochs per 30-min (AMP, larger batch, fewer samples per epoch).
- bs=4 OOMs on h256/l7 and h384 configs; re-run at bs=2 exacerbates undertraining (fewer tokens/step).

### Decision: **CLOSED.** Fern assigned to PR #12 (throughput/AMP) as prerequisite.

---

## 2026-04-23 21:40 — PR #5: tanjiro: Channel-weighted loss — surf_p upweighting (round 1 / MSE)

- **Branch:** `tanjiro/channel-weighted-loss`
- **W&B group:** `tanjiro/channel-weighted-loss`
- **Hypothesis:** Aggressively re-weighting surface-pressure channel in the loss should improve `val_avg/mae_surf_p`.

### Results (best 4 of 8)

| Rank | Config (w_surf_p / w_surf_U / w_vol) | val_avg/mae_surf_p |
|------|--------------------------------------|--------------------|
| 1 | 20 / 10 / 1 | **128.43** |
| 2 | 10 / 10 / 1 (baseline) | 135.11 |
| 3 | 50 / 10 / 0.1 | 140.58 |
| 4 | 50 / 10 / 1 | 142.17 |

### Analysis

- Sweet spot: `w_surf_p=20` (2× upweight) gives −4.9% vs baseline. Beyond that, optimization degrades globally — cross-channel coupling (p ≈ f(∇·u)) means killing velocity gradients kills pressure accuracy too.
- Crushing vol weight (0.1) severely harms vol_p (127→373) and hurts surf_p — volume provides important regularization.
- Signal is real, but the experiment ran on MSE loss. New track baseline (103.036, L1) makes 128.43 look worse than it is — needs to be retested on top of L1.

### Decision: **SENT BACK** to tanjiro. Rebase on L1, fine sweep psurf ∈ {14,17,20,23,27} + vol ∈ {0.5,1,2}.

---

## 2026-04-23 21:40 — PR #6: nezuko: LR + schedule sweep (round 1 / MSE)

- **Branch:** `nezuko/lr-schedule-sweep`
- **W&B group:** `nezuko/lr-schedule-sweep`
- **Hypothesis:** Higher peak LR + warmup + LR floor improve convergence under 30-min budget.

### Results (top 4 of 8)

| Rank | Config | val_avg/mae_surf_p |
|------|--------|--------------------|
| 1 | 5e-4, no warmup, min_lr=1e-5, cosine | **125.86** |
| 2 | 1e-3, wu5, min_lr=1e-5, cosine | 127.63 |
| 3 | 3e-4, wu3, no floor, cosine | 130.67 |
| 4 | baseline (5e-4, no wu, no floor) | 140.18 |

### Analysis

- LR floor (min_lr=1e-5) appears to help by ~10%, but student correctly flagged this may be noise: at epoch 12 (best checkpoint for top run), LR is ~4.3e-4 for both floor=0 and floor=1e-5 — near-identical at the moment of selection. Effect might be initialization/sampling variance rather than the floor itself.
- Higher LR (2e-3) didn't diverge but was +3.8% worse. Warmup alone unhelpful.
- OneCycle ≈ baseline (worst on in_dist + cruise).
- All experiment on MSE; new baseline (103.036, L1) makes 125.86 obsolete as a merge candidate.

### Decision: **SENT BACK** to nezuko. Rebase on L1, 3-seed disambiguation of floor=1e-5 + WSD scheduler test.

---

## 2026-04-23 22:15 — PR #8: edward: EMA of weights + gradient clipping (round 1 / MSE)

- **Branch:** `edward/ema-gradclip-stability`
- **W&B group:** `edward/ema-gradclip-stability`
- **Hypothesis:** EMA of model weights + gradient clipping stabilize training on heavy-tailed CFD regression.

### Results

| Rank | Config (ema / clip) | val_avg/mae_surf_p | W&B run |
|------|---------------------|--------------------|---------|
| 1 | 0.999 / 1.0 | **103.302** | `jvpser43` |
| 2 | 0 / 0.5 | 109.492 | `72qcolpt` |
| 3 | 0 / 1.0 | 118.120 | `i6dnta6r` |
| 4 | 0.999 / 0 | 128.018 | `6ptzwqgy` |
| 5 | 0 / 0 (baseline) | 137.558 | `lruq653s` |
| 6 | 0.9999 / 0 | 294.597 | `cfvqv705` |
| 7 | 0.9999 / 1.0 | 301.702 | `9cry9jff` |
| 8 | 0.9999 / 0.5 | 302.122 | `0bi9hip6` |

### Analysis

- **Winner ties (+0.26) track L1 baseline 103.036** on MSE. Not a merge, but strong signal — 24.9% improvement over in-PR MSE baseline via stability alone.
- **Clip is the dominant lever.** Clip alone (clip0.5=109.49) captures most of the gain. EMA 0.999 alone = 128.02 (−6.9%).
- **Stacking slightly super-additive**: expected 21.0% from summing isolated gains, observed 24.9%.
- **EMA 0.9999 catastrophically fails** (294–302) — horizon mismatch. At 14 epochs (5250 steps) vs EMA decay-horizon ~10k steps, shadow weights never escape init. Student's diagnosis correct.
- **Grad-clip at 1.0 fires 100% of steps** (median norm = 44, p99 = 330). The optimizer uses unit-norm direction only. Higher thresholds (5, 10, 50) would gate only the tail — that's the informative regime.
- Overlap with L1: grad-clip and L1 both damp heavy-tail gradient signal. May be redundant when combined.

### Decision: **SENT BACK** to edward. Rebase on L1, sweep higher clip thresholds {1, 5, 10, 50}, 2-seed EMA-only read.

---

## 2026-04-23 22:15 — PR #9: thorfinn: Pressure target reparameterization (round 1 / MSE)

- **Branch:** `thorfinn/pressure-target-reparam`
- **W&B group:** `thorfinn/pressure-target-reparam`
- **Hypothesis:** Reparameterizing pressure target (asinh, robust z-score, per-domain z-score) reduces heavy-tail bias in loss.

### Results

| Rank | y_norm | asinh_scale | val_avg/mae_surf_p | test_avg/mae_surf_p |
|------|--------|-------------|---------------------|----------------------|
| 1 | asinh | 458 (per-sample avg y_std) | **100.034** | **90.261** ← first finite test on track |
| 2 | asinh | 500 | 106.292 | 95.169 |
| 3 | asinh | 100 | 108.387 | 97.617 |
| 4 | asinh | 1000 | 111.537 | 100.782 |
| 5 | asinh | 2000 | 118.296 | 105.418 |
| 6 | per_domain (zscore) | — | 129.756 | 142.916 |
| 7 | baseline (zscore) | — | 134.843 | 121.932 |
| 8 | robust (median/MAD) | — | 143.406 | 130.164 |

### Analysis

- **Winner beats track L1 baseline (103.036) by 2.91%** — while still running MSE. Asinh mechanism is **orthogonal to loss shape** → signal should compound with L1 (expected combined gain ≥ 2.9%, possibly more).
- **Clean U-shape over asinh scale:** 458 < 500 < 100 < 1000 < 2000. Optimum near per-sample σ, matching theory — linear regime covers typical pressure, log kicks in for outliers.
- **robust LOSES (+6.4%)** — median/MAD fixes scale-stat robustness but not tail damping. The effective mechanism is **compression of extreme target values**, not resistance to outliers in the divisor.
- **per_domain LOSES (val −3.8%, test +17.2%)**. On `test_re_rand` raceCar MAE jumps 132→266 (+102%). Classic shortcut failure: domain-baked normalization breaks cross-domain transfer. Right way to use domain signal: as input feature (FiLM / one-hot), not normalization.
- **BUG FIX PROVIDED:** student patched `data/scoring.py::accumulate_batch` to zero non-finite samples' y before the subtract, so `(Inf - pred).abs() * 0.0 = NaN` no longer poisons the accumulator. First finite `test_avg/mae_surf_p` on this track (= 90.26 for winner).

### Decision: **PARTIAL MERGE**
- **Scoring.py fix cherry-picked** to advisor branch (commit 7d71abd). GH issue #10 closed. Unblocks test metrics for every in-flight and future PR.
- **asinh hypothesis SENT BACK** for rebase on L1 + 3-seed compound sweep. Couldn't clean-merge the combined PR due to train.py conflict with L1. If thorfinn's L1+asinh replicates ≤103.036, the new baseline will shift.

---

## 2026-04-23 (round 3) — PR #11: frieren: Fine surf_weight sweep on L1 loss

- **Branch:** `frieren/l1-surf-weight-sweep`
- **W&B group:** `frieren/l1-surf-weight-sweep` (project: `wandb-applied-ai-team/senpai-kagent-v-students`)
- **Hypothesis:** L1 loss shifts the optimal surf_weight below 10; volume supervision may be load-bearing for surface-pressure prediction under L1.

### Results

| surf_weight | best_val_avg/mae_surf_p | best_epoch | Δ vs 103.036 | W&B run |
|-------------|-------------------------|------------|--------------|---------|
| **1**       | **93.127**              | 14         | **−9.62%**   | `yt7eup38` |
| 2           | 100.212                 | 14         | −2.74%       | `8570stbe` |
| 5           | 98.819                  | 14         | −4.09%       | `b868hlw6` |
| 3           | 102.901                 | 13         | −0.13%       | `jjrsgjlu` |
| 10 (ctrl)   | 105.762                 | 13         | +2.64%       | `vjr01ox8` |
| 30          | 100.467                 | 13         | −2.49%       | `grygmqwp` |
| 15          | 103.496                 | 14         | +0.45%       | `pluzdzbf` |
| 20          | 111.752                 | 12         | +8.46%       | `xni7zysx` |

**Per-split for winner (sw=1):**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 106.92 |
| val_geom_camber_rc | 106.14 |
| val_geom_camber_cruise | 73.28 |
| val_re_rand | 86.16 |
| **val_avg** | **93.127** |

**Test 3-split avg (excl. NaN):** 91.58 (sw=1) — best across all sw values.

### Analysis

- **−9.62% improvement over L1 baseline.** Monotonically clean: surface pressure error grows with surf_weight 1→20, with a partial recovery at 30.
- **Mechanism confirmed:** under L1 loss, gradient is w·sign(err), not w·err. Surface points are ~2–5% of tokens; they don't need extra weighting to contribute enough gradient. Over-upweighting them starves the shared feature extractor of volume signal. sw=1 wins on every channel simultaneously — surface AND volume.
- **sw=1 is at the edge of the sweep** — the true optimum may lie below 1. Frieren recommends a sw∈{0.25,0.5} micro-sweep.
- All runs hit 14 epochs under 30-min wall-clock cap; models still improving at cutoff.

### Decision: **MERGED** into `kagent_v_students`. New baseline = **93.127** (L1, surf_weight=1).

---

## 2026-04-23 (round 3) — PR #12: fern: Throughput scaling — AMP + grad accumulation

- **Branch:** `fern/throughput-amp`
- **W&B group:** `fern/throughput-amp`
- **Hypothesis:** bf16 AMP (~2× throughput) + grad accumulation unlocks more epochs within 30-min budget, enabling a free improvement at no extra parameters.

### Results

| GPU | Config | Eff BS | Epochs | s/epoch | **val_avg/mae_surf_p** | Δ vs 103.04 | W&B run |
|-----|--------|--------|--------|---------|------------------------|-------------|---------|
| 0 | bs4, no-AMP | 4 | 14 | 132 | 105.99 | +2.9% | `c2m2yrs1` |
| 1 | bs4, AMP | 4 | 18 | 100 | 100.51 | −2.5% | `2s0nfouv` |
| 2 | bs8, no-AMP | 8 | 14 | 130 | 99.77 | −3.2% | `1ih3uvpx` |
| 3 | bs8, AMP | 8 | 17 | 107 | 104.94 | +1.8% | `d4z93j22` |
| **4** | **bs4, AMP, accum=2** | **8** | **19** | **100** | **93.29** | **−9.5%** | **`ny7msqow`** |
| 5 | bs4, AMP, accum=4 | 16 | 19 | 100 | 94.88 | −7.9% | `hizikr4z` |
| 6 | bs8, AMP, accum=2 | 16 | 17 | 107 | 112.24 | +8.9% | `p1cnolrn` |
| 7 | bs8, AMP, accum=4 | 32 | 17 | 106 | 119.99 | +16.5% | `pxyzhof6` |

### Analysis

- **bs4 + AMP + accum=2 (eff_bs=8) reaches 93.29**, beating old baseline (103.036) but not new baseline (93.127, PR #11). 
- **AMP delivers ~24% wall-clock speedup** (132s→100s/epoch at bs=4), unlocking 4–5 extra epochs.
- **Effective batch=8 via accumulation beats true bs=8** by ~6 pts. Mechanism: pad_collate padding waste is proportional to max sample in batch; at bs=4 this is lower, so AMP headroom goes further.
- **Bigger effective batches (>8) hurt** — cosine schedule anneals too fast with fewer optimizer steps; gradient noise suppressed too far.
- **GradScaler not needed for bf16** (full fp32 exponent range). AMP cuts VRAM ~22% at both bs=4 and bs=8.

### Decision: **SENT BACK** to fern. New baseline is now 93.127 (PR #11, surf_weight=1). Rerun with surf_weight=1 + probe sw∈{0.5, 0.25, 2} to confirm AMP compounding works.

---

## 2026-04-23 (round 3) — PR #5: tanjiro: Channel-weighted loss (rounds 1+2)

- **Branch:** `tanjiro/channel-weighted-loss`
- **W&B groups:** `tanjiro/channel-weighted-loss`, `tanjiro/channel-weighted-loss-v2`
- **Hypothesis:** Per-channel upweighting of surf_p in loss (independent of surf_Ux, surf_Uy) improves the primary metric.

### Results (v2 — on L1, grid: w_surf_p ∈ {10,14,17,20,23,27} + vol_weight ∈ {0.5,1,2})

| Config | val_avg/mae_surf_p | Δ vs 103.036 | W&B run |
|--------|---------------------|--------------|---------|
| baseline-sw10 (identity) | **97.874** | **−5.01%** | `28rhghge` |
| psurf17 | 100.301 | −2.65% | `popm00pl` |
| psurf27 | 99.320 | −3.60% | `rzp9b0bd` |
| psurf14 | 102.330 | −0.68% | `8q629ms9` |
| psurf23 | 102.704 | −0.32% | `kojebspb` |
| psurf20-vol2 | 102.941 | −0.09% | `wcfarlvh` |
| psurf20 | 106.811 | +3.66% | `7wtnp082` |
| psurf20-vol0.5 | 105.053 | +1.96% | `t03mduja` |

### Analysis

- **Identity case (sw=10, mathematically equivalent to merged baseline) nominally wins.** The 97.87 vs 103.036 gap is run-to-run variance — student's analysis correct.
- **Channel weighting does not compound with L1.** Under L1, w·sign(err) tilts constant gradient away from coupled channels; MSE-era psurf20 win (+2x) doesn't replicate.
- **Cross-channel regularization is saturated** at the identity weighting under L1 — no grid point beats identity on surf_p.
- Student recommends closing. Assessment agreed.

### Decision: **CLOSED.** Channel weighting explored thoroughly. Dead end for L1 loss. Tanjiro is now idle and needs reassignment.

---

## 2026-04-23 22:35 — PR #7: alphonse: Fourier PE on (x,z) + FiLM conditioning on log(Re) (round 1 / MSE)

- **Branch:** `alphonse/fourier-pe-film-re`
- **W&B group:** `alphonse/fourier-pe-film-re`
- **Hypothesis:** Random Fourier features let coordinate-based MLPs resolve high-frequency boundary-layer structure; FiLM on log(Re) gives per-channel scale modulation to handle 10× per-Re y_std variance.

### Results

| Rank | Config (fourier/σ/m + FiLM) | n_params | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|------|-----------------------------|----------|---------------------|---------------------|---------|
| 1 | fixed/1.0/10 | 667K | **116.73** | **105.67** | `2kxauwgi` |
| 2 | learn/1.0/16 + FiLM | 704K | 116.92 | 107.47 | — |
| 3 | fixed/1.0/10 + FiLM | 701K | 121.05 | 109.05 | — |
| 4 | learn/1.0/16 | 671K | 124.41 | 114.85 | — |
| 5 | baseline (no Fourier, no FiLM) | 662K | 128.87 | 115.39 | — |
| 6 | fixed/10.0/10 | 667K | 129.31 | 120.13 | — |
| 7 | fixed/10.0/10 + FiLM | 701K | 134.54 | 121.67 | — |
| 8 | FiLM only | 696K | 136.09 | 120.89 | — |

### Analysis

- **Winner (Fourier σ=1, m=10, no FiLM) beats in-PR MSE baseline by −9.4% val, −8.4% test** — wins uniformly on most splits, with strongest gains on `val_single_in_dist` (−27%) and `val_geom_camber_rc` (−20%). Classic Tancik-2020 signal: Fourier features let the MLP represent sharper spatial gradients, which matters most at boundary layers and OOD geometries.
- **vs track L1 baseline (103.036): +13% worse.** Not standalone-mergeable, but Fourier is orthogonal to loss shape → expected to compound with L1.
- **σ=10 is too high-frequency** — 129.31 (ties MSE baseline). Optimum is around σ ≤ 1, suggesting fine sweep in σ ∈ {0.5, 1, 2}.
- **FiLM alone HURTS** (+7.2 pts vs baseline). Single-point injection (after preprocess only) is underpowered — doesn't reach the deep layers where Re conditioning matters.
- **Fourier + FiLM: inconsistent.** Helps on learnable-m16 (−7.5), hurts on fixed-σ=1 (+4.3). FiLM behaves like regularization that helps overparameterized paths, hurts lean ones. Real test needs **per-block FiLM** (canonical Perez-et-al placement).
- Student independently implemented the scoring fix (same as commit 7d71abd) and provided finite test metrics.

### Decision: **SENT BACK** to alphonse. Rebase on L1, fine σ sweep {0.5, 1, 2}, m sweep {10, 20, 40}, and proper per-block FiLM (separate MLP per block, γ/β modulation after LayerNorm). Drop σ=10.

**Note:** Student flagged that `test_geom_camber_cruise/000020.pt` has Infs upstream of scoring; filed as low-priority data-quality issue #13 for the human team.

---

## 2026-04-23 22:50 — Round 3b reviews: PRs #6, #7, #9, #15 all SENT BACK

All four reviews of this batch landed in the **same pattern**: strong mechanism signals, but three of four ran on `--surf_weight 10` (the argparse default) instead of the post-PR #11 `--surf_weight 1` track baseline. **Systemic footgun:** students rebased code but didn't pick up the runtime flag change. Flagging in CURRENT_RESEARCH_STATE.md. Every send-back explicitly asks for `--surf_weight 1`.

### PR #6 nezuko (round 2, sw=10): LR floor + WSD on L1

**Winner:** `l1-lr-floor-s42` val=98.19 (sw=10). vs track baseline (93.127): +5.4% — no merge.

**Replicated effects:**
- **min_lr=1e-5 floor:** 3-seed replay (42, 7, 99) = 98.19 / 100.09 / 102.07, mean 100.12 ± 1.94. ~2σ real but modest. Mechanism still unclear (LR at best_epoch is near-identical between floor=0 and floor=1e-5).
- **WSD at lr=1e-3 beats cosine at lr=1e-3 by −6.6%** (101.12 vs 108.28). Cleanest new signal — high peak LR needs flat-plateau schedule.

**Decision:** SENT BACK. Rerun on sw=1 testing the two confirmed effects individually + stacked (GPU 5 = WSD@1e-3 + floor=1e-5 is the combined bet).

### PR #7 alphonse (round 2, sw=10): Fourier + per-block FiLM on L1

**Winner:** `l1-fr-s1-m40` val=93.245 (sw=10). vs track baseline (93.127): +0.13% — nominal tie, no merge.

**Clean signals:**
- **σ=1 is still best on L1** at m=10 (σ=1 < σ=2 < σ=0.5).
- **m is NOT saturated**: val monotonically improves m=10 (97.51) → m=20 (94.11) → m=40 (93.25). Obvious next step is m=80+.
- **Per-block FiLM alone is a regressor** (114.53, +9.2% worse than baseline).
- FiLM + Fourier stacks weakly positive at m=10 (−1.7%) but dominated by simply increasing m.

**Decision:** SENT BACK. Rerun on sw=1, drop FiLM, extend m to {40, 80, 160}. Honest forecast: m=80+ Fourier-only on sw=1 should land val ≈ 88–91.

### PR #9 thorfinn (round 2, sw=10): asinh + L1 compound sweep

**Winner:** `l1-asinh-s350` val=92.676 / test=84.022 (sw=10). vs track baseline (93.127): −0.48% nominal, but single seed and on wrong sw.

**Clean signals:**
- **Asinh compounds with L1 within sw=10:** L1+zscore (101.78) → L1+asinh-s350 (92.68) = −9 val points.
- **Seed variance at s=458 is substantial: val std 2.07** (3 seeds: 97.17, 100.93, 97.56). Round-1's 100.034 was within noise band — **partly coarse-sample luck**.
- **L1 shifts the optimum**: s≈350 beats s=458 (which was optimal on MSE). Possibly lower than 350.

**Open question:** does asinh compound with sw=1, or are the two redundant? Both rebalance surface↔volume residuals ~9 val points. If redundant, stacking gives nothing; if orthogonal, expect val ≈ 85–90.

**Decision:** SENT BACK. sw=1 + 3-seed at s=350 + 2-seed at s=250 + probes at s=300, 458. Best single-round test of the orthogonality question.

### PR #15 tanjiro (round 3): Horizontal-flip augmentation

**No runs.** Student performed a physics sanity check before burning 8 GPU-hours and caught a sign error in the assignment. Correct invariants under x-flip (for this dataset where `Uy` = velocity z-component):
- NEGATE: x(0), saf[0](2), AoA foil1(14), AoA foil2(18), stagger(23), output Ux(0)
- KEEP: z(1), saf[1](3), dsdf(4–11), is_surface(12), log(Re)(13), NACA(15–17, 19–21), gap(22), output Uy(1), output p(2)

Bonus insight: **x-flip is physics-exact for all three domains** (raceCar single, raceCar tandem, cruise tandem) — ground stays at z=0 under x→−x. No conditional flipping required.

**Decision:** SENT BACK with confirmation + green-light to run. Explicitly asked for `--surf_weight 1`.

**Student behavior commendation:** this is exactly the right way to handle a hypothesis with ambiguous physics — verify before burning compute. Worth emulating in future assignments.

---

## 2026-04-23 23:00 — PR #12: fern: Throughput scaling (AMP + grad accumulation) — MERGED

- **Branch:** `fern/throughput-amp`
- **W&B group:** `fern/throughput-amp` (rerun on sw=1)
- **Hypothesis:** AMP (bf16) + grad accumulation unlocks 25–35 epochs per 30-min budget vs baseline's 14, compounding with every other improvement.

### Results

| Rank | Config (amp / bs / accum / sw) | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B run |
|------|--------------------------------|---------------------|---------------------|-----------|---------|
| 1 | AMP / bs=4 / accum=4 / sw=1 | **88.268** ✓ WINNER | **79.733** (first finite test) | 19 | `n68w9q7o` |
| 2 | AMP / bs=4 / accum=2 / sw=1 | 88.770 | 79.641 | 18 | — |
| 3 | AMP / bs=4 / accum=2 / sw=2 | 89.712 | 81.554 | 18 | — |
| 4 | AMP / bs=4 / accum=1 / sw=1 | 90.415 | 80.749 | 18 | — |
| 5 | AMP / bs=4 / accum=2 / sw=0.5 | 93.102 | 84.640 | 17 | — |
| 6 | AMP / bs=4 / accum=2 / sw=0.25 | 96.189 | 87.364 | 17 | — |
| 7 | no-AMP / bs=4 / accum=1 / sw=1 (anchor) | 104.271 | 94.210 | 11 | — |
| 8 | AMP / bs=8 / accum=2 / sw=1 | 113.716 | 103.005 | 15 | — |

### Analysis

- **WINNER beats previous baseline (93.127) by −5.2% (val), −13.2% (test). New baseline: 88.268 val / 79.733 test.**
- **AMP delivers +4–5 epochs per 30-min budget** (14 → 18–19). Only ~24% per-epoch speedup (not theoretical 2×) because padding on variable meshes (74K–242K nodes) dominates compute.
- **Grad accumulation is genuinely free throughput at bs=4.** Time/epoch unchanged; only optimizer-step frequency drops. accum=4 vs accum=2 both ~100s/epoch.
- **bs=8 fails the wall-clock test** (17 epochs vs 19). Padding waste at bs=8 is ~2× bs=4 because a single 242K-node sample forces all 8 samples to pad. Critical insight: **effective batch size matters less than real batch size** — eff_bs=16 via bs=4+accum=4 (winner) far beats eff_bs=16 via bs=8+accum=2 (worst config).
- **sw ∈ {0.25, 0.5}** probe confirms PR #11's finding: sw=1 is the optimum; sub-1 values regress (93 → 96). No surprise, but useful double-check on this new baseline.
- Winner beats baseline uniformly on all 4 val splits, largest gain on `val_geom_camber_cruise` (−11.0%) and `val_geom_camber_rc` (−5.1%). Extra epochs = cleaner OOD generalization.
- **First finite `test_avg/mae_surf_p` on this track:** 79.733. Scoring bug fix (commit 7d71abd) validated.

### Decision: **MERGED** into `kagent_v_students`. New baseline = 88.268 val / 79.733 test. train.py now includes `--amp` and `--grad_accum` flags. Every future PR should use the AMP recipe.

**Throughput unlock is the force multiplier of the round.** It should compound with every other improvement — the next round of capacity/loss/conditioning experiments all inherit +5 epochs of training.

---

## 2026-04-23 23:30 — PR #14: frieren: surf_weight sub-1 micro-sweep + AMP compound (round 4)

- **Branch:** `frieren/surf-weight-sub1` (or similar)
- **W&B group:** `frieren/sw-sub1-amp`
- **Hypothesis:** sub-1 surf_weight might help under L1 by giving the volume gradient more weight; expected to test whether PR #11's sw=1 optimum is a plateau edge.

### Results (8 runs: 4 sw × {no-AMP, AMP+accum=2})

| Rank | Config (sw / AMP / accum) | val_avg | test_avg | W&B run |
|------|---------------------------|---------|----------|---------|
| 1 | sw=2 / AMP / accum=2 | **89.268** | **80.332** | `u511jzaa` |
| 2 | sw=0.5 / AMP / accum=2 | 96.078 | 85.452 | `ydh5xr81` |
| 3 | sw=1 / AMP / accum=2 | 96.969 | 86.169 | `2awyn4bk` |
| 4 | sw=0.25 / AMP / accum=2 | 98.417 | 88.967 | `f4ptf5ez` |
| 5 | sw=2 / no-AMP | 101.155 | 91.768 | `j0u8r174` |
| 6 | sw=1 / no-AMP (anchor) | 102.164 | 92.840 | `wetr25tj` |
| 7 | sw=0.5 / no-AMP | 104.890 | 93.287 | `l2kmvob1` |
| 8 | sw=0.25 / no-AMP | 109.586 | 97.647 | `jwl4hxi1` |

### Analysis

- **Sub-1 surf_weight confirmed as dead end.** This is a two-experiment replication with PR #12's side-probes (sw=0.5 → 93.10, sw=0.25 → 96.19). sw<1 regresses in every configuration tested.
- **AMP lift is 5–12% across all sw values**, biggest at sw=2 (−11.8%). Clean replication of PR #12's main finding.
- **⚡ sw=2 wins the AMP arm at 89.268** — **reverses PR #11's finding**. Under AMP + grad_accum=2 (eff_bs=8), a little more surface weight is the new optimum. The minimum has shifted right.
- **But ran at grad_accum=2, not the merged baseline's grad_accum=4 (eff_bs=16).** Not a clean test against the current 88.268 baseline. Winner is +1.13% WORSE than merged baseline — does not merge.
- **Noise-floor concern:** frieren's no-AMP sw=1 anchor hit 102.164 vs PR #11's published 93.127 at identical config. That's a ~9% gap. **No seed pinning is the issue.** Going forward, multi-seed runs are required for any claim <5% improvement (was <3%).
- **Split story:** sw=2 wins at `val_single_in_dist` (−0.7%) but hurts `val_geom_camber_cruise` (+5.5%). Cruise has highest volume fraction — weighing surface more starves volume precision there. No per-split "surprise win."

### Decision: **SENT BACK** to frieren.

- Close sub-1 direction.
- Refined sweep: sw ∈ {1, 1.5, 2, 3, 5, 10} × AMP + grad_accum=4 (eff_bs=16 to match merged recipe). 2-seed anchor at sw=1 to establish baseline noise floor.
- If sw=2 holds at eff_bs=16 → new baseline candidate (~86–88 possible). If it compresses → sw=1 is the robust optimum and we close this direction.

---

## 2026-04-24 00:00 — Round 5: PRs #6, #9, #15 reviewed

All three round-4 rerun submissions ran on **stale pre-AMP recipe** (branch forked from PR #11 tip, not post-PR-#12). Second round in a row that students have silently missed the merged recipe. Footgun flagged; going forward every assignment leads with explicit `--amp true --grad_accum 4` and requires a `--debug` sanity-check run BEFORE the full sweep.

### PR #6 nezuko (round 4, pre-AMP): LR floor + WSD stack on sw=1

**Winner:** `sw1-floor-s7` val=99.254, test=90.374. vs track baseline 88.268: **+12.45%** — no merge.

**Findings:**
- Floor=1e-5 effect shrinks from −4.7% (sw=10) to −1.5% (sw=1) — ~1σ at sw=1. Weak, replicating but decreasingly valuable as the baseline recipe improves.
- **Stack test (floor+WSD) FAILED BY CONSTRUCTION.** GPU 5 is bit-exact identical to GPU 3 (no-floor+WSD) because `wsd_lambda` doesn't thread `cfg.min_lr`. Plus WSD decay phase starts at epoch 37 of 50; realized runs end at 14. Main bet untested, not falsified.
- **WSD vs cosine at lr=1e-3 not testable** (student repurposed the control GPU slot).
- Seed=99 is a consistent outlier (108.15 here, 102.07 in v2); drop from future sweeps.

**Decision:** SENT BACK. Rebase to pick up AMP + grad_accum=4. Fix `wsd_lambda` to thread `cfg.min_lr`. Retune WSD phase splits to 10/30/60 so decay engages before ep 14. Proper stack test on AMP.

### PR #9 thorfinn (round 4, pre-AMP): asinh × sw=1 orthogonality test

**Winner:** `sw1-asinh-s250-s2` val=89.330, test=79.386. vs track baseline 88.268: **+1.06%** — no merge.

**ORTHOGONALITY TEST RESOLVED: partial redundancy.** Delta compressed from −9 pts (sw=10) to −3 pts (sw=1). Asinh and sw=1 largely rebalance the same surface↔volume residual imbalance; sw=1 already down-weights surface over-representation, leaving little headroom for asinh's tail compression.

**Seed variance at s=350 on sw=1: σ=6.74 (range 12.72), worse than round-3's σ=2.07 on sw=10.** The ~3 pt residual gain is within noise. Optimum shifted from s≈350 (sw=10) to s≈250–300 (sw=1), directionally consistent but not statistically defensible.

**Decision: CLOSED** after 5 rounds of asinh work. Mechanism well-mapped. Thorfinn reassigned to a fresh architectural direction: cross-attention surface decoder (PR #18).

### PR #15 tanjiro (round 3b, pre-AMP + physics-confirmed invariants): Horizontal-flip augmentation

**Anchor (no flip):** val=100.27 (+12 vs baseline, explained by stale branch).  
**Best hflip variant:** p=0.3 val=122.87. **All hflip settings REGRESS.**

**Structural failure:** AoA sign-flip pushes raceCar samples OOD — raceCar AoA ∈ [-10°, 0°] flipped → [0°, +10°] which is absent from val. Hypothesis cleanly refuted.

**Execution quality excellent:** invariants correctly implemented per round-3b physics correction; flip applied per-sample pre-normalization; seed-pinned 3-seed variance tight (std 1.11, 28× noise floor) — confident negative.

**Decision: CLOSED.** Hflip direction dead. Salvage paths (cruise-only, no-AoA-flip) low-EV vs what's already in flight. Tanjiro reassigned to fresh direction: in-distribution input feature jitter (PR #17).

---

## 2026-04-23 — PR #16: fern: Capacity scaling on AMP baseline (n_hidden / n_layers / slice_num) — CLOSED

- **Branch:** `fern/capacity-on-amp`
- **W&B group:** `fern/capacity-on-amp`
- **Hypothesis:** With AMP unlocking ~60 GB VRAM headroom and +5 epochs, larger models (h192/h256/h384, deeper l=7/9, wider slice_num=128) may now win within 30-min budget.

### Results

| GPU | Config | n_params | Peak VRAM | Epochs | best_ep | s/epoch | val_avg/mae_surf_p | test_avg | Δ vs 88.268 | W&B run |
|-----|--------|---------:|----------:|-------:|--------:|--------:|-------------------:|---------:|:-----------:|---------|
| 0 | h128-l5-s64 (anchor) | 0.662 M | 32.9 GB | 19 | 18 | 100 | 91.323 | 82.739 | +3.5% | `2w8sjpna` |
| 1 | h192-l5-s64 | 1.472 M | 43.0 GB | 15 | 14 | 126 | 98.987 | 89.911 | +12.1% | `7139msgc` |
| 2 | h256-l5-s64 | 2.600 M | 53.1 GB | 12 | 12 | 151 | 109.559 | 101.309 | +24.1% | `9i19496l` |
| 3 | h384-l5-s64 | 5.814 M | 73.3 GB | 9 | 7 | 211 | 136.821 | 125.691 | +55.0% | `6fot6dnw` |
| 4 | h128-l7-s64 | 0.905 M | 44.9 GB | 14 | 14 | 137 | 117.531 | 106.144 | +33.2% | `ysipi3af` |
| 5 | h128-l9-s64 | 1.147 M | 56.9 GB | 11 | 9 | 175 | 123.764 | 110.781 | +40.2% | `3e43hk3x` |
| 6 | h128-l5-s128 | 0.673 M | 48.1 GB | 13 | 12 | 144 | 114.222 | 106.178 | +29.4% | `lr8gs0gf` |
| 7 | h256-l7-s128 | 3.585 M | 89.5 GB | 7 | 6 | 267 | 146.034 | 135.127 | +65.4% | `b68xqgjv` |

### Analysis

- **No config beat the baseline.** The anchor underperformed merged 88.268 by +3.5% val — consistent with seed/loader variance (8 concurrent DataLoaders sharing CPU/IO).
- **Epoch budget is the bottleneck, not capacity.** r < −0.95 linear correlation between log(n_params) and epochs_completed. h192 (+122% params) gets only 15 epochs vs anchor's 19 — the epoch deficit exceeds the capacity benefit at every scale.
- **VRAM headroom never hit:** h256-l7-s128 at 89.5 GB is the heaviest config, still 6 GB under 96 GB ceiling. Memory is not the constraint.
- **Sub-linear throughput scaling with capacity:** AMP helps marginally but doesn't offset the per-epoch cost — h192 is +26% s/epoch, h384 is +111% s/epoch.
- **No config had enough epochs to plateau** — all scaled-up runs were still monotonically improving at timeout. h192 would need ~22 epochs to reach 88.27, requiring +46% wall-clock budget.
- **Mirror of PR #4** — this is the second clean confirmation that capacity scaling is epoch-budget bound.

### Decision: **CLOSED.** Capacity scaling direction exhausted under current budget. Negative result is now doubly confirmed. Fern reassigned to new direction.

---

## 2026-04-23 — PR #7: alphonse: Fourier PE on (x,z) — MERGED (round 6)

- **Branch:** `alphonse/fourier-pe-film-re`
- **W&B group:** `alphonse/fourier-sw1` (final rerun)
- **Hypothesis:** Random Fourier Features on (x,z) give the preprocess MLP spatial bandwidth to resolve boundary-layer gradients; m-saturation tested at sw=1 + AMP baseline.

### Final Results (sw1 m-saturation sweep, rebased on L1+AMP+grad_accum=4)

| # | Run | m | σ | seed | best_ep | val_avg | test_avg | Δ val vs 88.268 | W&B run |
|---|-----|---:|----:|----:|--------:|--------:|---------:|:----------------:|---------|
| 1 | `sw1-fr-s1-m160` | 160 | 1.0 | 0 | 18 | **84.737** | **75.244** | **−4.0%** | `91z1948k` |
| 2 | `sw1-fr-s1-m20` | 20 | 1.0 | 0 | 18 | 85.392 | 75.800 | −3.3% | `au5quccl` |
| 3 | `sw1-fr-s1-m10` | 10 | 1.0 | 0 | 17 | 86.041 | 77.737 | −2.5% | `ciemnw3i` |
| 4 | `sw1-fr-s1.5-m80` | 80 | 1.5 | 0 | 17 | 89.220 | 80.786 | +1.1% | `weukxee5` |
| 5 | `sw1-fr-s1-m80` | 80 | 1.0 | 0 | 17 | 89.631 | 82.658 | +1.5% | `pxlzd0sx` |
| 6 | `sw1-fr-s1-m40` | 40 | 1.0 | 0 | 18 | 89.695 | 81.796 | +1.6% | `zt5muqis` |
| 7 | `sw1-fr-s1-m80-seed2` | 80 | 1.0 | 2 | 17 | 90.066 | 77.267 | +2.0% | `q3vyfddr` |
| 8 | `sw1-baseline` | — | — | 0 | 17 | 98.821 | 88.890 | +11.9% | `1wdvy914` |

### Analysis

- **WINNER: m=160 beats baseline (88.268) by −4.0% val / −5.6% test.** Clean merge.
- **m-curve is non-monotonic (U-shaped):** m=20 and m=160 win; m=40 and m=80 regress to near-baseline. Likely redundancy at mid-m: B vectors cluster and the preprocess MLP can't exploit them in 17–18 epochs.
- **σ=1 is the robust sweet spot** (σ=1.5 at m=80 barely helps; σ=10 from round 1 is clearly harmful).
- **Seed variance is substantial on test (~5 pts for same config).** Single-seed test numbers should be read with this noise band in mind.
- **Peak VRAM at m=160: 74.4 GB** — within 96 GB card headroom.
- **FiLM (per-block log(Re) conditioning) dropped** — consistently net-negative at 17–18 epoch budget.

### Decision: **MERGED** into `kagent_v_students`. New baseline = **84.737 val / 75.244 test** (PR #7).
- W&B run: `91z1948k` (alphonse/sw1-fr-s1-m160)
- Fourier PE (fixed B, σ=1, m=160) added as new default config in BASELINE.md.

---

## 2026-04-24 — PR #14 (round 5 rerun): frieren: sw>1 sweep at eff_bs=16 — CLOSED

- **Branch:** `frieren/surf-weight-subunit-plus-amp` (pre-Fourier, stale)
- **W&B group:** `frieren/sw-over-1`
- **Hypothesis:** sw=2's win at grad_accum=2 should replicate at grad_accum=4 (eff_bs=16).

### Results (seeded, 8 runs)

| Rank | Config (sw / seed) | val_avg/mae_surf_p | test_avg |
|------|--------------------|---------------------|----------|
| 1 | sw=2.0 / seed=2 | **92.398** | 83.265 |
| 2 | sw=1.0 / seed=1 | 94.046 | 84.066 |
| 3 | sw=3.0 / seed=1 | 94.262 | 85.453 |
| 4 | sw=1.5 / seed=1 | 94.941 | 85.448 |
| 5 | sw=10 / seed=1 | 96.004 | 87.158 |
| 6 | sw=2.0 / seed=1 | 96.144 | 87.035 |
| 7 | sw=1.0 / seed=2 | 96.427 | 85.930 |
| 8 | sw=5.0 / seed=1 | ~98.4 | — |

### Analysis

- **Winner (sw=2 s=2) is +9.05 % worse than current 84.737 baseline** (branch is pre-Fourier; runs on 88.268 recipe).
- **sw=2 effect collapses at eff_bs=16.** sw=2 2-seed mean (94.27) vs sw=1 2-seed mean (95.24) = +1.0% sub-1σ. The round-4 sw=2 win (−11.8% at eff_bs=8) was grad-accum-specific — eff_bs=16 averaging dissolves the surface-upweight benefit.
- **Flat basin in sw ∈ [1, 3]**; sw ≥ 5 regresses. Optimum did NOT shift back to MSE-era sw=10.
- **Noise floor finding (load-bearing for future rounds):** seeded 2-seed spread at sw=1 under AMP+grad_accum=4 is 2.38 val / 1.86 test — MUCH tighter than the ~9% pre-seed floor. Multi-seed now mandatory for claims under 5%.
- **Seeding infrastructure:** frieren added `--seed` flag independently in fa577eb — already present in advisor branch from alphonse's PR #7 (`seed: int = 0` in Config). Cherry-pick skipped.

### Decision: **CLOSED.** Sw direction exhausted across 2 rounds on pre-Fourier code.

- Any future sw work would need to be *post-Fourier* (Fourier PE rebalances surface/volume gradient flow), but that's a new hypothesis (sw × Fourier interaction), not a continuation of this PR.
- Frieren reassigned to PR #21 (near-surface volume-band weighting) — a fresh loss-weighting direction using dsdf features to tier volume loss by proximity to surface.

---

## 2026-04-24 — PR #6 (round 5 rerun, v4): nezuko: LR + schedule sweep on L1+AMP — CLOSED

- **Branch:** `nezuko/lr-schedule-sweep` (pre-Fourier PE — stale)
- **W&B group:** `nezuko/lr-schedule-amp`
- **Hypothesis:** WSD schedule + LR floor compound with AMP baseline; test floor+WSD stack properly (previous rounds had WSD construction bug).

### Results (seeded 2-anchor sweep, 8 runs)

| Rank | Config (lr / wu / min_lr / sched / seed) | val_avg | test_avg |
|------|------------------------------------------|---------|----------|
| 1 | 5e-4 / 0 / 0 / cos / seed=7 (anchor) | **90.86** | 81.92 |
| 2 | 1e-3 / 5% / 0 / cos / 42 | 92.53 | 82.83 |
| 3 | 5e-4 / 0 / 0 / cos / seed=42 (anchor) | 94.88 | 86.11 |
| 4 | 1e-3 / 10% / 0 / wsd / 42 | 95.74 | 85.57 |
| 4 | 1e-3 / 10% / 1e-5 / wsd / 42 (stack) | 95.74 | 85.57 (bit-exact) |
| 6 | 5e-4 / 0 / 1e-5 / cos / 42 | 95.64 | 85.31 |
| 7 | 5e-4 / 0 / 1e-5 / cos / 7 | 97.56 | 89.58 |
| 8 | 2e-3 / 10% / 0 / wsd / 42 | 98.34 | 88.97 |

### Analysis

- **Branch stale (pre-Fourier PE from PR #7).** Comparison baseline is 88.268, not current 84.737. Winner (90.86) is still +2.9% worse than even the pre-Fourier baseline.
- **CROSS-ROUND SIGN FLIPS — schedule effects don't transfer across regimes:**
  - Floor=1e-5: v2 (sw=10) −4.7% better → v3 (sw=1) −1.5% noise → **v4 (sw=1+AMP) +4.0% worse** (paired-seed mean).
  - WSD vs cosine at lr=1e-3: v2 WSD wins −6.6% → v4 cosine wins −3.4%.
- **Stack test (floor+WSD) failed by construction AGAIN.** WSD bug fixed correctly by student, but at realized 19/50 epochs (38%), training ends before the decay phase begins at 40%. GPU 3 and GPU 5 produce bit-exact identical runs. Cannot test WSD+floor compounding at current budget.
- **Seed variance at anchor: 4.4% val / 5.1% test** — wider than round-7's 2.5%. 8-parallel-run IO contention causing wider variance and slower epochs (16-19 vs expected 19+).
- Student themselves asked for closure in follow-up #1.

### Decision: **CLOSED.**

Three rounds × many variants thoroughly map LR-schedule space. Key takeaways carried forward:
1. Seed-anchor methodology (2-seed for claims <5%).
2. WSD construction bug diagnosis (min_lr threading, phase-split vs realized-budget mismatch).
3. Negative result: schedule effects are regime-specific artefacts, not universal regularizers.

Nezuko reassigned to PR #22 (attention temperature annealing) — fresh architectural direction orthogonal to Fourier PE, AMP, loss shape.

---

## 2026-04-24 — PR #17 (round 5): tanjiro: In-distribution input jitter — SENT BACK

- **Branch:** `tanjiro/input-feature-jitter` (pre-Fourier PE — stale)
- **W&B group:** `tanjiro/input-jitter`
- **Hypothesis:** Small Gaussian jitter on continuous input features (AoA, log(Re), gap, stagger) gives in-distribution augmentation without the AoA-OOD trap that killed PR #15.

### Results (seeded sweep, 8 runs)

| Rank | Config (aoa / logRe / gap / seed) | val_avg | test_avg |
|------|-----------------------------------|---------|----------|
| 1 | gap-only 0.02 (seed=42) | **89.880** | 80.992 |
| 2 | anchor (seed=42) | 95.772 | 87.540 |
| 3 | aoa-only 0.01 (seed=42) | 96.048 | 86.532 |
| 4 | full-stack (seed=7) | 96.574 | 87.430 |
| 5 | anchor (seed=7) | 99.949 | 90.151 |
| 6 | Re-only 0.05 (seed=42) | 100.214 | 91.853 |
| 7 | full-stack (seed=42) | 101.055 | 91.865 |
| 8 | all-2× (seed=42) | 117.403 | 105.371 |

### Analysis

- **Branch pre-Fourier (4th student this round). Winner 89.88 vs 84.737 baseline = +6.1% worse.** But vs pre-Fourier 88.268 baseline, gap-only beats by 1.8 pts — a real signal.
- **Gap/stagger jitter (σ=0.02) is the only real signal** (−5.89 pts vs anchor, winning single_in_dist −13.7 and camber_rc −9.6).
- **AoA/Re jitter at suggested σ: neutral-to-harmful.** Re-only regressed by +4.44.
- **Full stack is DESTRUCTIVE (+11.2 vs gap-only alone).** Mechanism: AoA2/gap/stagger are exactly zero on single-foil samples; adding noise to "0.0" nudges those samples slightly OOD, cancelling the gap-only gain. Student's follow-up #3 correctly identifies this.
- **σ=2× probe catastrophic** (117.40, +21.6) — confirms 2× σ exits the in-distribution manifold.
- **Seed spread: 4.3% at anchor, 4.6% at full stack** (vs calibrated 2.5% floor on single-run). 8-parallel IO contention is a confound.

### Decision: **SENT BACK.**

Refined sweep: gap-only σ-scan ∈ {0.01, 0.02, 0.03, 0.04} + 3-seed at σ=0.02 winner + tandem-gated AoA (follow-up #3), on Fourier baseline. Honest forecast: if gap-jitter is orthogonal to Fourier (different mechanisms: foil separation vs spatial bandwidth), val ~82–84 is achievable. If redundant, ~85.

---

## 2026-04-24 — PR #20: fern: Fourier σ fine-sweep + SwiGLU feedforward — MERGED

- **Branch:** `fern/fourier-sigma-fine-swiglu`
- **W&B group:** `fern/fourier-sigma-swiglu`
- **Hypothesis:** σ fine-sweep at m=160 + SwiGLU replacement of standard MLP in TransolverBlock. Orthogonal mechanisms — spatial bandwidth vs gated FFN.

### Results (verified only)

| Rank | Config (σ / SwiGLU / seed) | val_avg | test_avg | W&B run |
|------|----------------------------|---------|----------|---------|
| 1 | σ=1.0 / SwiGLU / s0 | **73.660** | **63.983** | `eg6i88yf` |
| 2 | σ=1.0 / None / s0 (anchor) | 84.737 | 75.244 | `2jd70vx1` (reproduces PR #7 exactly) |
| 3 | σ_x=0.7 σ_z=1.3 / None | 85.773 | 75.140 | `lnwrwucf` |
| 4 | σ=0.5 / None | 88.203 | 79.330 | `3gro8kvc` |
| 5 | σ_x=0.5 σ_z=1.5 / None | 89.960 | 81.111 | `ejywoanq` |

Unverified (W&B runs crashed, student-reported numbers don't exist as summary keys):
- σ=0.7 / SwiGLU (student claimed 71.49 val; crashed epoch 11)
- σ=0.7 / None (student claimed 86.54; crashed epoch 9)
- σ=1.3 / None (student claimed 87.39; crashed epoch 9)

### Analysis

- **SwiGLU at σ=1.0 wins decisively: −13.1% val / −15.0% test vs PR #7.** Biggest lifts on `val_single_in_dist` (−21.7%) and `val_geom_camber_cruise` (−18.3%). Wins every split on both val and test.
- **Per-coordinate σ anisotropy is a NET LOSS.** Both (σ_x=0.7, σ_z=1.3) and (σ_x=0.5, σ_z=1.5) regress. Isotropic σ=1.0 wins among verified configs.
- **σ=0.5 regresses +4.1 % val.** Confirms σ ≥ ~0.7 is required; low-σ is too-smooth.
- **SwiGLU implementation is textbook-correct:** three projections (gate, up, down), SiLU on gate, element-wise product, 2/3 hidden-width to match GELU MLP param count (744k vs 743k).
- **Peak VRAM: 37.8 GB** (+2.9 GB vs PR #7). Best epoch 17 vs 18 baseline — negligible budget cost.
- **Data-integrity issue:** student reported a σ=0.7+SwiGLU compound winner (val 71.49) that is not verifiable on W&B (run crashed mid-training, no summary keys written). Flagged in merge comment; re-assigned to alphonse PR #24 as a properly-seeded verified sweep.

### Decision: **MERGED** σ=1.0 + SwiGLU. New baseline = 73.660 val / 63.983 test. SwiGLU FFN added to default recipe.

---

## 2026-04-24 — PR #18: thorfinn: Cross-attention surface decoder head — CLOSED

- **Branch:** `thorfinn/cross-attn-surface-decoder` (pre-Fourier, stale)
- **W&B group:** `thorfinn/surface-decoder`
- **Hypothesis:** Dedicated cross-attention decoder for surface nodes (surface queries attend to full hidden state), replacing mlp2 for surface predictions.

### Results

| Rank | Config (L/h/seed) | val_avg | test_avg | W&B run |
|------|-------------------|---------|----------|---------|
| 1 | anchor (no decoder) / s7 | 90.86 | 81.92 | (in-PR anchor) |
| 2 | anchor (no decoder) / s42 | 94.88 | 86.11 | — |
| 3 | L1 / h=4 / s42 | 232.59 | 215.16 | — |
| 4 | L2 / h=4 / s99 | 243.42 | 224.84 | — |
| 5 | L2 / h=4 / s42 | 253.78 | 235.31 | — |
| 6 | L2 / h=4 / s7 | 257.48 | 239.24 | — |
| 7 | L2 / h=8 / s42 | 258.20 | 239.45 | — |
| 8 | L3 / h=4 / s42 | 274.47 | 254.49 | — |

### Analysis

- **Decoder catastrophically fails:** best variant +175% worse than current track baseline (73.66 post-PR #20). Monotone regression with depth (+20–25 pts per layer) and width (+5 pts for h4→h8).
- **3-seed variance at L2/h4: σ=5.88**. Decoder–anchor gap (158 pts) is ~27σ of decoder std and ~39σ of anchor spread. Cleanly outside noise.
- **Mechanism refuted:** decoder hurts in-distribution MOST (`val_single_in_dist` +242%). Fresh untrained head cannot catch trunk's pretrained mlp2 in ~15 epochs. Budget loss (15 vs 19 epochs) is real but not load-bearing.
- **Implementation was clean and correct** — pre-LN cross-attention, per-sample surface gather, padding handled via index restriction, SDPA direct call, residual+FFN, torch.where gating, AMP dtype probe.
- 5th student this round with stale Fourier rebase (branch at `06e898d`). Anchor underperformance (~7% vs 84.737) confirms but isn't load-bearing given decoder loses by 175%.

### Decision: **CLOSED.**

**Salvage path for revival:** zero-init residual decoder. Decoder produces a delta (`preds = vol_preds + is_surface * surf_delta`) with the final output projection zero-initialized. At init, decoder invisible; can only improve from there by construction. Assigned as thorfinn PR #23.

---

## 2026-04-24 — PR #19: alphonse: Fourier m-extension + learnable B — CLOSED

- **Branch:** `alphonse/fourier-m-extension-learnable`
- **W&B group:** `alphonse/fourier-m-ext`
- **Hypothesis:** Find m-saturation point above 160; test learnable vs fixed B; multi-seed m=20 vs m=160 to disambiguate PR #7's U-shaped curve.

### Results

| Rank | Config (feat / m / seed) | val_avg | test_avg | W&B run |
|------|-------------------------|---------|----------|---------|
| 1 | fixed / m=160 / s0 (anchor, reproduces PR #7) | 84.737 | 75.244 | `7kf0h22b` |
| 2 | fixed / m=20 / s0 | 85.392 | 75.800 | `ukzmp8nt` |
| 3 | fixed / m=20 / s1 | 88.383 | 80.760 | `wxrsg9xn` |
| 4 | fixed / m=320 / s0 | 92.926 | 86.281 | `xz8b2lfu` |
| 5 | fixed / m=160 / s1 | 96.668 | 87.075 | `sep9wl6e` |
| 6 | learnable / m=320 / s0 | 97.000 | 87.900 | `vcqxn6u2` |
| 7 | learnable / m=160 / s0 | 100.745 | 91.362 | `4eppk7o6` |
| 8 | fixed / m=640 / s0 | 106.834 | 97.708 | `7q8swcdt` |

### Analysis

- **m=160 is the saturation point — past it is pure regression.** m=320 (+9.7%), m=640 (+26.1%). m=640 also loses 3 epochs to timeout due to more params. Fixed-B m-curve is inverted-U with peak at ~160.
- **Learnable B clearly loses at this budget:** +11.1% at m=160, +4.4% at m=320. Optimizer doesn't have 19 epochs to improve over random σ=1 init; added parameter noise dominates. Kill until budget ≥ 35 epochs.
- **BIGGEST FINDING — seed variance at m=160: σ ≈ 8 pts.** 2-seed means:
  - m=20: mean 86.89 (s0=85.39, s1=88.38)
  - m=160: mean 90.70 (s0=84.74, s1=96.67)
  
  **m=20 and m=160 have overlapping seed distributions.** The "U-shape" from PR #7 likely wasn't real — PR #7's 84.737 was a lucky s=0 tail; the m=160 config's **actual expected performance is ~90.70**, not 84.74. 84.737 is ~0.7σ below config-mean. It's a valid pinned-seed baseline but NOT representative.

### Decision: **CLOSED.**

Methodology win: establishes that **single-seed measurements of advances < ~5% are below noise floor.** Multi-seed protocol becomes mandatory for sub-5% merge claims. Alphonse reassigned to PR #24 (σ × SwiGLU seeded sweep) to run the first experiment under this tightened protocol.

---

## 2026-04-24 — PR #21: frieren: Near-surface volume-band weighting (3-tier BL loss) — CLOSED

- **Branch:** `frieren/near-surface-volume-band` (pre-SwiGLU, stale)
- **W&B group:** `frieren/ns-vol-band`
- **Hypothesis:** Use `dsdf` (dims 4–11) to define a BL band of volume nodes; upweight the BL band (w_near) so gradients emphasize boundary-layer pressure fidelity.

### Results

| Rank | Config (τ_near / w_near / w_far / seed) | val_avg | test_avg | state |
|------|------------------------------------------|---------|----------|-------|
| — | anchor-s0 | running (never completed summary) | — | running |
| 1 | anchor-s1 | 96.67 | 87.08 | finished |
| 2 | τ=0.10 / w_near=3 / w_far=1 / s0 | 102.67 | 91.23 | finished |
| 3 | τ=0.05 / w_near=2 / w_far=1 / s0 | 105.55 | 92.89 | finished |
| 4 | τ=0.05 / w_near=3 / w_far=1 / s1 | 109.18 | 99.19 | finished |
| 5 | τ=0.05 / w_near=3 / w_far=0.5 / s0 | 112.57 | 100.40 | finished |
| 6 | τ=0.05 / w_near=3 / w_far=1 / s0 | running (val ~230 mid-training) | — | running |
| 7 | τ=0.05 / w_near=5 / w_far=1 / s0 | 118.88 | 108.25 | finished |

### Analysis

- **3-tier BL weighting HURTS at all tested weights.** Monotonic regression w_near ∈ {2, 3, 5}: 105.6 → 114 → 118.9. Deweighting far-vol doesn't rescue it.
- **Sanity check PASSED:** frac_near = 8.0% at τ=0.05 (within 5–15% target). τ=0.10 gives 22.7% which is wider than target but physically reasonable. BL band is correctly identified.
- **Student's self-diagnosis (mechanism):** per-tier-mean formulation sums three volume-tier means (far/mid/near), each of comparable magnitude, against ONE surface mean × surf_weight=1 → effectively deweights surface relative to volume. Opposite of the optimization goal. A sum-weighted-per-node loss (∑w_i·|pred - y| / n_vol_total) would have been a cleaner form — but this PR used the mean-per-tier form as specified.
- **Uniform split regression** (+15–27% across all 4 val splits) — hypothesized OOD lift on camber_cruise and re_rand doesn't materialize.
- **Branch is stale (pre-SwiGLU).** All 8 runs on the 84.737 Fourier-only baseline, not the current 73.66 Fourier+SwiGLU baseline. Not load-bearing here (BL weighting regresses by 20% internally, far larger than the 13% SwiGLU gap), but the cumulative pre-merge-rebase pattern across 6 students is a lab-level issue.

### Decision: **CLOSED.**

- Hypothesis disconfirmed under the per-tier-mean formulation.
- Loss-weighting landscape now exhaustively mapped by frieren across 4 PRs: L1 (#3), sw=1 (#11), sw>1 (#14), 3-tier BL (#21).
- Frieren reassigned to PR #26 (sample-wise normalization with Re-predicted scale) — fresh territory, researcher's top-ranked untested idea.

---

## 2026-04-24 — PR #22: nezuko: Attention temperature annealing — CLOSED

- **Branch:** `nezuko/attn-temperature-annealing` (pre-SwiGLU, stale — 7th consecutive)
- **W&B group:** `nezuko/attn-temp-anneal`
- **Hypothesis:** Fixed `self.temperature = 0.5` at init makes slice-assignment softmax sharp from step 0. Anneal from T₀ ∈ {1.5, 2.0} → 0.5 over first 3–5 epochs to let attention explore before hardening.

### Results (ranked by best_val_avg/mae_surf_p, W&B verified)

| Rank | Config (T₀ / anneal / seed) | val_avg | test_avg | state |
|------|------------------------------|---------|----------|-------|
| 1 | 0.5 / 0 / s7 (anchor) | **86.268** | 76.284 | finished |
| 2 | 2.0 / 5 / s42 | 88.085 | 79.010 | finished |
| 3 | 1.5 / 5 / s42 | 90.948 | 83.606 | finished |
| 4 | 2.0 / 0 / s42 | 91.147 | 83.230 | finished |
| 5 | 1.5 / 5 / s7 | 91.164 | 82.958 | finished |
| 6 | 1.5 / 3 / s42 | 94.030 | 85.335 | running (orphan) |
| 7 | 0.5 / 0 / s42 (anchor) | 94.458 | 84.221 | finished |
| 8 | 1.5 / 0 / s42 | 95.099 | 84.953 | finished |

### Analysis

- **Hypothesis rejected at 2-seed significance.** Anneal 2-seed mean at winning config (T=1.5 an=5): 91.056 vs anchor 2-seed mean: 90.363 → **annealing is +0.7 val WORSE on average**. Anchor spread is 8.19 val — effect buried in noise. The raw "winner" (86.268) is the seed=7 ANCHOR, not any anneal variant.
- **Post-release temperatures drift to [0.3, 0.7]** around original 0.5 init — 0.5 was already a good basin. No head diverged to extremes.
- **Trend direction informative despite insignificance:** on s42, longer anneal > shorter anneal > init-only > anchor (94.46 → 95.10 → 94.03 → 90.95 → 88.09). Higher T₀ (2.0 > 1.5) wins by 2.9 val on s42. Directional signal is sensible but inside noise.
- **Student implementation was exemplary:** grad-freeze during anneal (freezes `temperature.requires_grad` during schedule, snaps to 0.5 and re-enables at release) was a smart principled deviation from spec — avoids stale AdamW momentum against overwritten params. Temperature traces verified step-for-step schedule execution.
- **Branch is stale pre-SwiGLU.** All runs on pre-PR #20 recipe (84.737 baseline). Even a real win couldn't be merged without regressing SwiGLU gains.

### Decision: **CLOSED.**

Methodology lessons carried forward:
- 2-seed protocol correctly rejects the hypothesis. First clean use of the multi-seed protocol established round 9.
- Anchor seed variance ~8 val on pre-SwiGLU recipe; sub-5% effects cannot be distinguished.

Nezuko reassigned to PR #27 (slice_num sweep on merged recipe) — fundamental architectural knob never cleanly tested on the current 73.66 baseline.

---

## 2026-04-24 — PR #24: alphonse: σ × SwiGLU fine sweep — MERGED

- **Branch:** `alphonse/sigma-swiglu-sweep`
- **W&B group:** `alphonse/sigma-swiglu`
- **Hypothesis:** Verify fern's crashed σ=0.7+SwiGLU claim (val 71.49) from PR #20. Establish 2-seed noise floor under the merged SwiGLU recipe. First strict multi-seed experiment.

### Results (2-seed summary)

| σ | seed 0 | seed 1 | 2-seed mean | test mean | std(val) |
|---|--------|--------|-------------|-----------|----------|
| **0.7** | 71.489 | **69.845** | **70.667** | **62.691** | 1.162 |
| 1.0 (anchor) | 73.660 | 74.173 | 73.917 | 65.496 | 0.362 |
| 0.8 | 81.468 | 76.817 | 79.142 | 72.212 | 3.289 |
| 0.9 | 80.994 | 74.989 | 77.992 | 69.038 | 4.246 |

### Analysis

- **σ=0.7 2-seed mean (70.67) beats anchor mean (73.92) by 3.25 val, ~9× anchor std (0.362).** Outside noise by a very wide margin.
- **Anchor seed variance under SwiGLU recipe: std 0.362 val — dramatically tighter than pre-SwiGLU m=160 band (σ ≈ 8 val).** SwiGLU recipe stabilizes optimization ~20×. This is the key methodological calibration.
- **σ=0.8 and σ=0.9 regress catastrophically** (79–80) with high seed variance (std 3–4) — optimization pathology in that specific band. σ=0.7 is a **sharp minimum, not a flat basin.**
- **Fern's crashed σ=0.7 claim verified bit-exactly.** seed=0 reproduced 71.489.
- **Per-split uniform win:** σ=0.7 best seed beats σ=1 best on every val and every test split.
- First application of strict multi-seed protocol to merge: worked cleanly, caught fern's crashed claim, established tight noise floor.

### Decision: **MERGED** → new baseline 69.845 best-single / 70.667 2-seed-mean val; 62.778 / 62.691 test.

Follow-up: fine σ sweep {0.5, 0.55, 0.6, 0.65, 0.75} to locate the true minimum (sharp-minimum finding makes this critical). Assigned as PR #28.

---

## 2026-04-24 — PR #17 (round 8 rerun): tanjiro: Gap-only jitter on Fourier baseline — SENT BACK

- **Branch:** `tanjiro/input-feature-jitter` (pre-SwiGLU, 8th consecutive stale-rebase)
- **W&B group:** `tanjiro/gap-jitter-fourier`
- **Hypothesis (r8):** Gap/stagger jitter σ-scan {0.01, 0.02, 0.03, 0.04} + tandem-gated AoA1 probe on the Fourier baseline.

### Results

| Config (σ_gap / seed) | val_avg | test_avg |
|------------------------|---------|----------|
| 0.04 / s42 | **89.838** | 78.513 |
| 0.01 / s42 | 90.940 | 82.501 |
| 0.02 / s7 | 91.235 | 80.487 |
| 0.03 / s42 | 91.579 | 80.657 |
| anchor / s7 | 92.332 | 85.423 |
| anchor / s42 | 94.878 | 83.678 |
| 0.02 / s42 | 95.135 | 84.087 |
| 0.02 / s99 | 96.287 | 87.466 |

### Analysis

- **Branch pre-SwiGLU.** Winner val 89.84 is +16.18 val vs current 69.85 baseline; incompatible recipe.
- **σ=0.04 is the new peak at seed=42** (r5's σ=0.02 winner was seed-lucky). 3-seed spread at σ=0.02: 91.24/95.14/96.29 (mean 94.22, spread 5.05 val).
- **Monotonic trend {0.01 → 0.04}** at seed=42 suggests true peak may be beyond σ=0.04 (edge of grid).
- **OOD win pattern replicates** on pre-SwiGLU Fourier: single_in_dist −8.2, camber_rc −7.5 at σ=0.04 vs anchor-s42.
- **4-parallel still too much IO contention** (5.4% seed spread vs target 2.5%).

### Decision: **SENT BACK.** Rebase + serial pairs + extended σ-scan {0.04, 0.06, 0.08} + isolated tandem-AoA probe.

---

## 2026-04-24 — PR #23: thorfinn: Zero-init residual surface decoder — CLOSED

- **Branch:** `thorfinn/zeroinit-residual-decoder` (pre-PR #24; fourier_sigma=1.0 on all runs — 9th consecutive stale-rebase)
- **W&B group:** `thorfinn/residual-decoder`
- **Hypothesis:** ControlNet-style zero-init residual decoder (`preds = vol_preds + is_surface * surf_delta` with surf_delta output Linear zero-initialized) salvages the PR #18 catastrophic failure mode by starting invisible and only improving.

### Results

| Rank | Config (L/h/seed) | val_avg | test_avg | best_ep |
|------|-------------------|---------|----------|---------|
| 1 | anchor (no decoder) / s0 | **73.660** | 63.983 | 17 |
| 2 | anchor (no decoder) / s1 | 74.173 | 67.009 | 17 |
| 3 | L1/h4/s0 | 88.348 | 80.588 | 9 |
| 4 | L2/h4/s2 | 99.588 | 95.068 | 7 |
| 5 | L2/h4/s1 | 103.016 | 92.502 | 7 |
| 6 | L3/h4/s0 | 110.270 | 99.106 | 6 |
| 7 | L2/h4/s0 | 112.289 | 95.127 | 8 |
| 8 | L2/h8/s0 | 123.279 | 107.423 | 6 |

### Analysis

- **Zero-init verified correctly:** `surf_delta_step0_abs_max = 0.0` on all 6 decoder runs. The student's careful re-zero AFTER `apply(_init_weights)` (to defend against trunc_normal overwrite) is exemplary.
- **Hypothesis falsified by budget, not mechanism.** Decoder is ~2× slower per-iter than trunk even after student's 40× speedup (full-N MHA → surface-only Q). Anchor reaches ep 17; decoders only ep 6–9.
- **ControlNet analogy fails when trunk is still training.** The premise (refiner adds tiny adjustments to a converged base) doesn't hold here: at epochs 1-8 both decoder and anchor are noisy (±13 val), then anchor converges 89→73.66 at ep 9-17 while decoder is cut off. No "track-anchor-then-diverge-downward" pattern.
- **3-seed L2/h4 std: 6.57 val — 18× anchor std (0.36).** Decoder head never converges to stable regime.
- Monotone regression with depth (L1→L2→L3) driven entirely by epoch budget (ep 9→8→6), not architectural capacity.

### Decision: **CLOSED.**

Architectural-decoder direction NOT closed entirely — student's own follow-up #1 (slice-bottleneck decoder using `PhysicsAttention`) is the principled complexity fix. Reassigned as PR #29: slice-bottleneck matches trunk iter-speed at O(N·G·D).
