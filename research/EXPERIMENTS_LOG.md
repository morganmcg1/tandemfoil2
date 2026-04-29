# SENPAI Research Results

## 2026-04-29 00:30 — PR #859: Surface weight scan ↩ SENT BACK (rebase + Huber stack)

- Branch: `willowpai2e1-fern/surf-weight-scan`
- Hypothesis: Increase surf_weight beyond default 10 to put more gradient pressure on the surface MAE objective. Sweep ∈ {10, 20, 50, 100} with EMA decay=0.99.

| surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p | Best epoch | W&B run |
|------------:|-------------------:|--------------------:|:----------:|---------|
| 10 (control) | 122.38 | 109.15 | 12 | [fwpcgiwo](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/fwpcgiwo) |
| **20** | **115.87** | **103.92** | 13 | [c54oko37](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/c54oko37) |
| 50 | 123.34 | 111.88 | 14 | [rlcqbqzw](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/rlcqbqzw) |
| 100 | 129.79 | 119.33 | 14 | [6z7uhk6s](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/6z7uhk6s) |

Per-split test (sw=20): single=117.79, rc=112.11, cruise=78.88, re_rand=106.90  
Won 3/4 splits in val and test; cruise marginally regressed (already easiest).

**Mechanism check (train losses):** vol_loss monotone-degrades with sw (0.43→1.00); surf_loss flat. The gain at sw=20 is a *generalization* effect via EMA, not better surface fit. At sw=50/100, volume-task starvation feeds back into surface error — the failure mode is "boundary-only training under-constrains the volume field."

**Analysis and conclusions:**

sw=20 is the surf_weight optimum under EMA decay=0.99 (no Huber): −5.3% val / −4.8% test within-sweep vs sw=10 control. The non-monotonic curve with optimum at 2× default is consistent with surface-emphasis being a real lever, but volume-task starvation kicks in by sw=50.

**However**, the result is built on the OLD baseline (EMA-only, val=119.35) — the new baseline is PR #769 (Huber δ=0.5, val=102.86) which merged after these runs started. sw=20+EMA at 115.87 does not beat the new baseline, and crucially Huber implicitly upweights surface-vs-volume (since surface residuals are smaller than volume in normalized space, Huber clips volume more aggressively). So the optimum surf_weight under Huber may shift.

**PR sent back for rebase + 4-variant Huber+EMA stack scan** (sw ∈ {10, 15, 20, 30}). If sw=20 still wins under Huber, that's the new baseline ~98-100 val. If the optimum shifts down to 15 (likely under the Huber-already-upweights-surface argument), that's actually an even stronger result.

---

## 2026-04-28 22:45 — PR #769: Huber loss for outlier-robust pressure regression ✓ MERGED

- Branch: `willowpai2e1-alphonse/huber-loss`
- Hypothesis: Replacing MSE with Huber loss (quadratic for small residuals, linear for large) will reduce the outsized influence of high-Re outlier batches on gradient direction, particularly for the pressure channel which has the heaviest-tailed residual distribution.

| Delta | val_avg/mae_surf_p | Δ vs EMA baseline | test_avg/mae_surf_p | Δ vs EMA baseline | Epoch | W&B run |
|------:|-------------------:|:-----------------:|--------------------:|:-----------------:|:-----:|---------|
| **0.5** | **102.86** | **−13.8%** | **94.83** | **−12.8%** | 14 | [hp87pun7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/hp87pun7) |
| 1.0 | 108.49 | −9.1% | 98.25 | −9.7% | 14 | [ph6zmlfa](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ph6zmlfa) |
| 2.0 | 126.53 | +6.0% (worse) | 114.39 | +5.1% (worse) | 11 | [9r4h4s1g](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/9r4h4s1g) |

All three runs: no EMA; compared against EMA-only baseline PR #773 (val=119.35, test=108.79).

Per-split test (δ=0.5): single=109.68, rc=99.91, cruise=76.12, re_rand=93.62  
Biggest gains: rc −17.8%, re_rand −14.7% — exactly the heavy-tailed OOD splits the hypothesis targeted.

**Analysis and conclusions:**

Huber loss works exactly as hypothesized. The monotone ordering (δ=0.5 > 1.0 >> 2.0) and the concentration of gains on OOD heavy-tailed splits (rc, re_rand) confirm the heavy-tail story. δ=2.0 regresses because typical normalized pressure residuals are ≤2 σ-units, so the linear-penalty regime barely activates — essentially MSE in practice.

Key empirical insight: δ=0.5 was still trending downward at epoch 14 (val_avg had dropped 124→119→102 in the last 5 epochs). More epochs would likely yield further gains. The win is also WITHOUT EMA (student confirmed the EMA merge landed after their runs started). Huber+EMA stack is the natural follow-up.

Note: the student independently found and fixed the same NaN propagation bug (in evaluate_split, zeroing pred at masked positions), corroborating the earlier fix in commit 49c55ed. The advisor's nan_to_num guard in current train.py covers both prediction and GT NaN.

**New baseline: val_avg/mae_surf_p = 102.86, test_avg/mae_surf_p = 94.83. Merged into advisor branch.**

---

## 2026-04-28 22:20 — PR #846: Unmodified baseline run (canonical reference) ✓ CLOSED (results recorded)

- Branch: `willowpai2e1-edward/unmodified-baseline`
- Hypothesis: Establish the clean, unmodified default config baseline that all Round 1/2 experiments can compare against.

| W&B run | val_avg/mae_surf_p | test_avg/mae_surf_p | Epoch | Notes |
|---------|:------------------:|:-------------------:|:-----:|-------|
| [bv3x1tp6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/bv3x1tp6) | 140.95 | 128.32 | 14/50 | Timeout hit; no code changes |

Per-split val: single=182.40, rc=162.60, cruise=103.93, re_rand=114.86  
Per-split test: single=157.33, rc=145.12, cruise=90.01, re_rand=120.80  
Per-epoch val_avg converged monotonically to best at epoch 14 (still descending — schedule never annealed below ~4.1e-4).

**Analysis and conclusions:**

Clean reference baseline confirmed: val_avg=140.95, test_avg=128.32 at 14 epochs (30-min wall clock). This validates all Round 1 wins:
- EMA (PR #773) at val=119.35 is **15.4% better** than unmodified default — a real improvement.
- nezuko's clip0.5 (PR #775, no EMA) at val=115.01 is **18.4% better** — even stronger.

Key observation: per-epoch throughput is ~132s, confirming ~14 epochs per 30-min budget. The cosine LR at epoch 14 is still at 4.1e-4 (out of 50-epoch schedule) — strong evidence of schedule mismatch already under investigation by thorfinn PR #860.

Notable: UW configs in PR #771 (val ~123–124 at epoch 14) were better than this clean unmodified baseline by ~17 val units. Most likely explanation is run-to-run variance or accidental beneficial effect of UW's different loss reparameterization — not enough samples to be conclusive.

**PR closed** — no code to merge; results recorded as canonical reference in BASELINE.md.

---

## 2026-04-28 19:30 — PR #771: Learnable per-channel uncertainty weighting (Kendall & Gal 2018)

- Branch: `willowpai2e1-edward/uncertainty-weighting`
- Hypothesis: Replace fixed MSE loss with Kendall-Gal learnable uncertainty weighting. A scalar log-variance per output channel (Ux, Uy, p) is jointly learned with the model. The natural gradient signal redistributes loss capacity toward channels with the highest residual variance, which we expected to be the pressure channel given its larger dynamic range.

| W&B run | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p | Epoch | Notes |
|---------|------------|---------------------|---------------------|-------|-------|
| [1tvvwlux](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/1tvvwlux) | 10 | 123.243 | 111.227 | 14/50 | Timeout hit |
| [6gjtvi4h](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/6gjtvi4h) | 1 | 123.887 | 113.698 | 14/50 | Timeout hit |

**Note:** No unmodified baseline exists yet. These numbers cannot be interpreted as improvement or regression.

**Analysis and conclusions:**

The UW mechanism is fundamentally misaligned with the task objective. Kendall-Gal UW assigns lower effective loss weight to the channel with the *highest* per-element MSE — because high residual variance is interpreted as high aleatoric uncertainty, not as something the model should work harder on. Since the pressure channel has the largest dynamic range (and hence the highest absolute MSE), `log_var[p]` converged to -1.0 while `log_var[Ux/Uy]` converged to ~-2.25, giving pressure approximately 3.5x *less* effective loss weight than velocity. This is precisely backwards: our ranking metric is `mae_surf_p`, so we need more focus on p, not less.

The approach is a mathematical dead-end for this metric and objective. Inverse/fixed channel weighting (upweighting p explicitly) remains a valid idea worth testing separately.

**Bug found and fixed by student (credited to willowpai2e1-edward):**
`test_geom_camber_cruise/000020.pt` contains 761 non-finite pressure node values. The existing per-sample `y_finite` guard in `data/scoring.py` correctly excluded sample 20 from the accumulation mask, but `NaN * 0.0 = NaN` in IEEE-754 meant the error tensor still poisoned the running sum. The same NaN propagated through `y_norm` into the monitoring loss in `train.py`'s `evaluate_split()`. Both bugs were fixed in commit `49c55ed` on the advisor branch with `torch.nan_to_num(err, nan=0.0)` guards. All future runs on this track will have correct `test_geom_camber_cruise` metrics.

**PR closed** as a dead end.

---

## 2026-04-28 21:55 — PR #773: EMA weight averaging (Polyak) for flatter generalization ✓ MERGED

- Branch: `willowpai2e1-fern/ema-weights`
- Hypothesis: Polyak / EMA averaging of model weights (via `torch.optim.swa_utils.AveragedModel`) produces a flatter validation minimum and better OOD generalization, especially on the held-out geometry splits. A sweep of three decay rates was tested.

| Decay  | Best epoch | val_avg/mae_surf_p | EMA Δ vs live model | test_avg/mae_surf_p | W&B run |
|--------|------------|--------------------|---------------------|---------------------|---------|
| **0.99**   | **13/14**  | **119.35**         | **+6.0% over live** | **108.79**          | [5yzk5722](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/5yzk5722) |
| 0.999  | 9/14       | 145.08             | +10.8%              | 132.17              | [t7x9cjha](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/t7x9cjha) |
| 0.9999 | 14/14      | 158.68             | −12.5% (worse!)     | 146.05              | [3otfhs7r](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/3otfhs7r) |

Per-split val improvement at decay=0.99 vs live model: `val_single_in_dist` +5.5%, `val_geom_camber_rc` +4.7%, `val_geom_camber_cruise` +9.4%, `val_re_rand` +5.2%. Gains are consistent across all 4 splits; geometry OOD splits benefit most (consistent with the "flatter basin = better OOD" story).

**Analysis and conclusions:**

EMA works as predicted. decay=0.99 is optimal for the ~14-epoch budget: fast enough to integrate useful signal (4-step half-life ≈ 100 optimizer steps per epoch ≈ ~400-step effective lookback). decay=0.999 was theoretically better (slower, flatter) but needed more epochs than the 30-min budget allowed — it bested the live model at epoch 9 but had only 4 EMA-active epochs post-warmup. decay=0.9999 was far too slow: still anchored to early-training weights at epoch 14.

Key implementation note from student: checkpoint saves `ema_model.module.state_dict()` (inner module, no `module.` prefix) so it loads cleanly into a plain Transolver for eval or further fine-tuning.

**New baseline: val_avg/mae_surf_p = 119.35, test_avg/mae_surf_p = 108.79. Merged into advisor branch.**

---

## 2026-04-28 22:10 — PR #775: Linear LR warmup + gradient norm clipping ↩ SENT BACK (rebase required)

- Branch: `willowpai2e1-nezuko/warmup-grad-clip`
- Hypothesis: Linear LR warmup (5 epochs) + gradient norm clipping stabilises PhysicsAttention's orthogonal slice projections and learnable temperature parameter in early training, where high-Re samples produce large gradients.

| Variant | warmup_epochs | clip_norm | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch | W&B run |
|---------|:---:|:---:|---:|---:|:---:|---|
| **warmup5-clip0.5** | 5 | 0.5 | **115.01** | **101.64** | 14 | [xlo5cmpw](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/xlo5cmpw) |
| warmup5-clip1 | 5 | 1.0 | 118.93 | 107.09 | 13 | [7q09eo6v](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/7q09eo6v) |
| warmup10-clip1 | 10 | 1.0 | 130.60 | 118.81 | 14 | [1rrzrcqe](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/1rrzrcqe) |
| noclip-warmup5 | 5 | 0 | 132.75 | 119.50 | 14 | [n63jhoif](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/n63jhoif) |

Per-split test (warmup5-clip0.5): single=121.68, rc=110.76, cruise=74.29, re_rand=99.82

**Analysis and conclusions:**

`warmup5-clip0.5` beats the EMA baseline (119.35) by **3.6% on val and 6.6% on test** — a clear win. The gain is real but the branch was created before EMA (PR #773) merged into advisor, making the PR CONFLICTING in GitHub. The result is also WITHOUT EMA, so we don't know whether EMA+clip stack.

Key empirical finding: gradient norms pre-clip are consistently large throughout the entire run (median ~60, p95 ~268, 100% of steps clipped). Clipping acts as a **continuous gradient regulariser**, not just early-training stabilization. The cruise split benefited most (74 vs 94 test), consistent with high-Re cruise samples driving the largest gradients.

Warmup alone (noclip-warmup5=132.75) contributes little. Longer warmup (warmup10) is actively harmful — 5 warmup epochs eat into the cosine schedule's convergence window. Clipping is the dominant lever; shorter warmup may even be redundant once clip is in place.

**PR sent back for rebase + EMA stack test.** Requested: rebase onto advisor, re-run `warmup5-clip0.5` with `--ema_decay 0.99`, then sweep clip ∈ {0.1, 0.25, 0.5} + warmup_epochs=0 to measure diminishing returns and EMA interaction.

---

## 2026-04-28 22:10 — PR #774: Wider Transolver (hidden=256, slices=128, heads=8) ✗ CLOSED

- Branch: `willowpai2e1-frieren/wider-model`
- Hypothesis: Increasing model width (128→256), slice count (64→128), and heads (4→8) provides more representational capacity for TandemFoilSet's large irregular meshes.

| Run | n_hidden | slice_num | n_head | BS | Epochs | val_avg | test_avg | W&B |
|-----|:---:|:---:|:---:|:---:|:---:|---:|---:|---|
| `wider-256-128` (full) | 256 | 128 | 8 | 2 | 6 | 164.79 | 154.71 | [o16zl2ma](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/o16zl2ma) |
| `wider-256-64` (width only) | 256 | 64 | 8 | 4 | 7 | 156.84 | 142.88 | [3obqis2j](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/3obqis2j) |
| **`wider-128-128`** (slim+2x slices) | 128 | 128 | 4 | 4 | 11 | **136.02** | **124.11** | [b307a0lc](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/b307a0lc) |

**Analysis and conclusions:**

All three variants are worse than the EMA baseline (119.35). However, the **slim+2x-slices variant won** by 19.8% on test_avg vs the full proposal, despite having ~4× fewer parameters. Key insight: slice doubling is nearly free in params (+5%) but appears to be a much more compute-efficient capacity dial than hidden width under our 30-min budget. More slices = more "physics tokens" for the irregular mesh decomposition.

Width increase is firmly budget-incompatible: wider models need bs=2 (OOM at bs=4) and only reach 6-7 epochs. The 14-epoch budget favours slim models with high slice count.

**Follow-up assigned:** slice scan ∈ {64, 96, 128, 192} at n_hidden=128, n_head=4 with EMA (PR #862), to characterize the slice optimum cleanly.

**PR closed** — wider-model hypothesis disproved for this budget. The slice direction is being explored separately.

---

## 2026-04-28 21:56 — PR #777: Log-Re input jitter for cross-regime generalization ✗ CLOSED

- Branch: `willowpai2e1-thorfinn/re-jitter-aug`
- Hypothesis: Gaussian jitter on the normalized log(Re) input feature (dim 13) during training improves cross-regime generalization (val_re_rand target).

| Variant     | std  | val_avg/mae_surf_p | Δ vs control | W&B run |
|-------------|------|--------------------:|-------------|---------|
| no-jitter (control) | 0.00 | **124.149** | — | [ze94qebq](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ze94qebq) |
| jitter-0.10 | 0.10 | 132.247 | +6.5% worse | [atr6fwx4](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/atr6fwx4) |
| jitter-0.05 | 0.05 | 140.081 | +12.8% worse | [etpurp6h](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/etpurp6h) |
| jitter-0.20 | 0.20 | 146.337 | +17.9% worse | [ov86n58c](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ov86n58c) |

The control wins on every val split including val_re_rand (the targeted one: 114.57 vs 117.58 for best jitter). Only val_geom_camber_cruise showed marginal jitter-0.10 benefit (−1.6%).

**Analysis and conclusions:**

All runs hit the 30-min timeout at epoch 13-14. Input augmentation slows convergence — the regularization benefit only emerges once the unaugmented model starts to overfit, which never happens before the budget is exhausted. The effect is monotone in std: larger jitter = more damage. Augmentation as a regularization strategy is incompatible with our short-budget regime in its current form.

**Note:** The no-jitter control run (ze94qebq) provides an approximate unmodified baseline at epoch 14: val_avg=124.149, test_single=128.67, test_rc=125.35, test_re_rand=114.09 (test_geom_camber_cruise=NaN, unfixed run). Awaiting PR #846 for the authoritative full-budget baseline.

Follow-up idea (flagged for later): curriculum jitter starting at epoch 30+ once overfitting begins, or per-AoA jitter only. Not worth pursuing until budget is extended.

**PR closed** as a dead end.

---
