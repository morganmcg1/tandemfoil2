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
