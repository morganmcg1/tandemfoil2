# SENPAI Research Results

## 2026-04-29 01:20 — PR #881: Huber δ-scan + EMA stack ✓ MERGED (new best — massive win)

- Branch: `willowpai2e1-alphonse/huber-ema-stack`
- Hypothesis: Huber + EMA operate on orthogonal aspects of training (per-sample loss vs weight trajectory) and should stack. δ scan ∈ {0.1, 0.25, 0.5} + EMA=0.99 to confirm stacking and find the δ floor.

| Variant | best_epoch | val_avg/mae_surf_p | Δ vs Huber-alone | test_avg/mae_surf_p | W&B run |
|---------|:----------:|-------------------:|:----------------:|--------------------:|---------|
| **huber δ=0.1 + ema 0.99** | 14 | **85.23** | **−17.1%** | **76.64** | [jej4y8gt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/jej4y8gt) |
| huber δ=0.5 + ema 0.99 | 14 | 90.17 | −12.3% | 80.33 | [c55ye285](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/c55ye285) |
| huber δ=0.25 + ema 0.99 | 14 | 90.42 | −12.1% | 80.42 | [ek98s7vq](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ek98s7vq) |
| MSE + ema 0.99 (control) | 13 | 119.73 | — | 107.09 | [14coc4pt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/14coc4pt) |

All runs: EMA=0.99, no clip, no warmup. Huber-alone baseline (PR #769): val=102.86, test=94.83.

Per-split val (δ=0.1 + EMA): single=103.34, rc=96.44, cruise=64.23, re_rand=76.91

Per-split test (δ=0.1 + EMA): **single=94.31, rc=86.19, cruise=53.08, re_rand=72.97**

Stack confirmation: Huber+EMA nearly additive on log scale (Huber alone over MSE+EMA = −14.1%; EMA alone over Huber = −12.3%; combined vs MSE+EMA = −24.7% vs −27% expected if perfectly multiplicative).

δ-curve with EMA on: δ=0.5 and δ=0.25 are tied; δ=0.1 breaks away by 5.5%. Monotonicity continues past δ=0.5; δ=0.1 still descending at epoch 14 (strong evidence floor not reached).

**Analysis and conclusions:**

The PR-881 Huber+EMA combination without clip yields val=85.23 — **beating the prior 4-way stack champion (PR #775, val=96.54)** by 11.7%. This means δ=0.1 is the dominant lever, larger than the clip+warmup combination tested in PR #775. The cruise split benefit is extraordinary: test_cruise=53.08 vs 76.12 (Huber-δ=0.5-alone) = −30.3%.

Key implication: whether clip+warmup=0 still help at δ=0.1 is an open question. The 5-way stack (δ=0.1 + clip + warmup=0 + EMA) may push further into the low-70s or even high-60s range on val. Assigned to alphonse as follow-up (PR #957).

Control reproduced within 0.3% of PR #773 baseline (119.73 vs 119.35). MSE+EMA comparison is clean.

**New best: val=85.23, test=76.64. Merged. Minimum required flags: --huber_delta 0.1 --ema_decay 0.99**

---

## 2026-04-29 01:20 — PR #776: Deeper Transolver n_layers=8 ✗ CLOSED (budget-incompatible)

- Branch: `willowpai2e1-tanjiro/deeper-model`
- Hypothesis: n_layers=8 (vs default 5) provides more depth of reasoning for complex tandem-foil flow topology, particularly for cruise-split wake interaction.

| Variant | n_layers | lr | Epochs | val_avg/mae_surf_p | test_avg/mae_surf_p | peak VRAM | W&B run |
|---------|:--------:|:---:|:------:|-------------------:|--------------------:|:---------:|---------|
| layers8-lr3e4 | 8 | 3e-4 | 9 | 112.39 | 101.46 | 64.5 GB | [ig4rsoc6](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ig4rsoc6) |
| layers8-lr5e4 | 8 | 5e-4 | 9 | 113.82 | 102.81 | 64.5 GB | [51pqcdny](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/51pqcdny) |
| layers10-lr3e4 | 10 | 3e-4 | 7 | 122.84 | 110.72 | 79.5 GB | [a9dwqw4v](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/a9dwqw4v) |

All runs: Huber δ=0.5 + EMA=0.99. Budget: n_layers=8 at 3.54 min/epoch → 9 epochs in 30 min; n_layers=10 at 4.34 min/epoch → 7 epochs.

Test cruise (layers8-lr3e4): 71.30 — small improvement vs Huber-alone baseline's 76.12, but obliterated by PR #881's δ=0.1+EMA result (53.08).

**Analysis and conclusions:**

Hypothesis weakly supported but practically unrewarded: depth=8 is under-converged at 9 epochs (still descending), but the new PR #881 winner (val=85.23) at n_layers=5 already beats depth=8 by 27 val units. The cruise-split signal for depth (71 vs 76) is consistent with long-range-wake-coupling intuition but irrelevant against the new baseline.

The binding constraint is **per-epoch compute cost**. At 1.65× per-epoch slowdown for depth=8, BF16 mixed precision would need to deliver ≥1.65× throughput to make depth=8 competitive. Assigned to tanjiro as follow-up (PR #959).

Useful outcome: `--n_layers` CLI flag added to train.py; per-epoch timing profiling (2.14 min/ep at n=5, 3.54 at n=8, 4.34 at n=10) gives concrete data for budget planning.

**Closed. Budget-binding; architectural direction deferred pending BF16 throughput confirmation.**

---

## 2026-04-29 01:05 — PR #867: AdamW β₂ scan with EMA ✗ CLOSED (hypothesis rejected)

- Branch: `willowpai2e1-edward/beta2-scan`
- Hypothesis: Heavy-tailed gradient distribution (median pre-clip ~60, p95 ~268) makes AdamW's default β₂=0.999 suboptimal — stale large-gradient second moments keep per-parameter step size artificially small. Lower β₂ (0.95, 0.99) makes second moments adapt faster.

| β₂ | val_avg/mae_surf_p | Δ vs control | test_avg/mae_surf_p | Δ vs control | best_epoch | W&B run |
|----|-------------------:|:------------:|--------------------:|:------------:|:----------:|---------|
| 0.999 (ctrl) | **110.09** | — | **98.23** | — | 14 | [sniqfdpt](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/sniqfdpt) |
| 0.95 | 110.38 | +0.3% | **98.07** | −0.2% | 14 | [oj773wzh](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/oj773wzh) |
| 0.99 | 115.30 | +4.7% | 104.12 | +6.0% | 14 | [4wor9sf7](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/4wor9sf7) |
| 0.9999 | 120.38 | +9.3% | 107.64 | +9.6% | 12 | [z9baj5zl](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/z9baj5zl) |

All runs: EMA=0.99, no Huber, no clip. Best of β₂=0.999 control at val=110.09 doesn't beat new 4-way stack baseline (96.54).

Per-split val (β₂=0.999 control): single=129.31, rc=121.72, cruise=87.77, re_rand=101.57
Per-split test (β₂=0.95, test-best): single=110.07, rc=111.00, cruise=72.34, re_rand=98.87

**Diagnostic (v_mean/v_p99 at run end):** confirmed mechanical effect of β₂ — β₂=0.95 produces ~3× smaller v_mean (0.33 vs 1.03) and ~5× smaller v_p99 (1.16 vs 5.49). Mechanism works as expected; gain just doesn't materialize as val improvement.

**Analysis and conclusions:**

Hypothesis rejected. β₂=0.999 is already at the optimum for this regime. β₂=0.95 ties the control; β₂=0.99 regresses 5%; β₂=0.9999 regresses 10% (only 12 epochs, bias-correction lag). The non-monotone β₂=0.99 result (worse than both 0.95 and 0.999) is surprising but explainable: it sits in a "responsiveness vs stability" trough — fast enough to be whiplashed but not fast enough to forget quickly.

The PR's own failure-mode prediction was accurate: "if the spread is <2 val units, β₂ should be tuned *after* clipping." The control is ~2 val units better than β₂=0.95, confirming this. The diagnostic confirms clipping changes what Adam's second-moment estimator sees — β₂ tuning *with* clip is a valid future probe but low priority given the tiny expected gain. Runs predated Huber+clip merges so the absolute values are not competitive.

**Closed. Note: the --beta2 flag implementation remains on the student branch and can be cherry-picked if needed.**

---

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

## 2026-04-29 01:00 — PR #775 Round 2: warmup=0 + clip=0.5 + EMA + Huber stack ✓ MERGED (new best)

- Branch: `willowpai2e1-nezuko/warmup-grad-clip` (rebased post-Huber+EMA advisor)
- Hypothesis: Dropping warmup (warmup_epochs=0) with clip_norm=0.5 + EMA=0.99 + Huber δ=0.5 forms a 4-way stack. The original round-1 winner was warmup5-clip0.5 (no EMA); this round tested all permutations with EMA on.

| Variant | best_val_avg/mae_surf_p | Δ vs Huber baseline | test_avg/mae_surf_p | best_epoch | W&B run |
|---------|------------------------:|:-------------------:|--------------------:|:----------:|---------|
| **w0-clip0.5-ema0.99** | **96.54** | **−6.1%** | **85.33** | 14 | [h22uwyy3](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/h22uwyy3) |
| w5-clip0.25-ema0.99 | 97.85 | −4.9% | 86.98 | 14 | [xncrhud8](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/xncrhud8) |
| w5-clip0.1-ema0.99 | 98.48 | −4.3% | 87.87 | 14 | [cvbk5249](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/cvbk5249) |
| warmup5-clip0.5-ema0.99 | 99.47 | −3.3% | 89.38 | 14 | [z2gglbld](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/z2gglbld) |

Huber baseline (PR #769): val=102.86, test=94.83

Per-split val (w0-clip0.5-ema0.99):

| Split | val/mae_surf_p |
|-------|---------------:|
| single_in_dist | 112.12 |
| geom_camber_rc | 114.30 |
| geom_camber_cruise | **70.72** |
| re_rand | **89.02** |

Per-split test (w0-clip0.5-ema0.99):

| Split | test/mae_surf_p |
|-------|----------------:|
| single_in_dist | 97.91 |
| geom_camber_rc | 100.57 |
| geom_camber_cruise | **58.92** |
| re_rand | **83.91** |

Gradient norm pre-clip summary (w0 run): median=45.6, p95=202.0, p99=354.3, max=841.7 — 100% of steps clipped. Notably ~25% lower median than all warmup=5 variants.

**Analysis and conclusions:**

The four-way stack (Huber + EMA + clip + warmup=0) is confirmed: −31.5% vs unmodified default (140.95), −6.1% vs Huber-alone baseline. The critical finding is that **warmup=0 outperforms warmup=5** with all other factors equal. The warmup variants showed that clip=0.25 slightly beat clip=0.5 (97.85 vs 99.47 val), but warmup=0 with clip=0.5 beats both. Mechanistically: with clip dominant from step 1, the optimizer immediately enters a stable low-lr-effective-step regime that warmup was meant to achieve gradually. Dropping warmup recovers ~5 epochs of optimization budget with no early instability (lower pre-clip median confirms this). Cruise split benefit is largest (test 58.92 vs Huber-alone 76.12 = −22.7%), consistent with high-Re cruise samples driving the biggest gradients.

**New best: val=96.54, test=85.33. Merged. New default flags: --warmup_epochs 0 --clip_norm 0.5 --huber_delta 0.5 --ema_decay 0.99**

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

## 2026-04-29 01:00 — PR #860: OneCycle LR schedule alignment ↩ SENT BACK (rebase + full stack)

- Branch: `willowpai2e1-thorfinn/schedule-alignment`
- Hypothesis: The default CosineAnnealingLR(T_max=50) barely anneals within the 14-epoch budget. Aligning T_max=14 or switching to OneCycleLR should release 3–8% more gains.

| Variant | val_avg/mae_surf_p | Δ vs EMA baseline | test_avg/mae_surf_p | W&B run |
|---------|-------------------:|:-----------------:|--------------------:|---------|
| **onecycle-T14-ema0.99** | **112.98** | **−5.3%** | **100.67** | [qjdthms4](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/qjdthms4) |
| cosine-T14-ema0.99 | 117.44 | −1.6% | 106.21 | [or22h0fu](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/or22h0fu) |
| cosine-T50-ema0.99 (control) | 119.99 | +0.5% | 107.93 | [a58tmier](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/a58tmier) |

EMA baseline (PR #773): val=119.35, test=108.79. All runs: no Huber, no clip.

Per-split test (onecycle-T14-ema0.99): single=116.37, rc=110.38, cruise=74.15, re_rand=101.77. OneCycle wins on all 4 test splits.

**Analysis and conclusions:**

Schedule alignment hypothesis confirmed: OneCycle −5.3% val / −7.5% test over EMA-only baseline. OneCycle does more than cosine-T14 alone (−3.8% extra): the slow warmup ramps from base_lr/25=4e-5 over 4.2 epochs, avoiding early noisy gradient steps, then the faster cool-down with EMA accumulation yields better generalization. The cruise split recovered the most (74.15 vs 81.38 test for EMA baseline).

**However, runs predate PR #769 (Huber) and PR #775 (clip+warmup) merges.** OneCycle on EMA-alone baseline (val=112.98) does NOT beat the new 4-way stack baseline (val=96.54). The direction is promising — schedule alignment is an orthogonal lever that may compound with the 4-way stack. **Sent back for full-stack test.**

---

## 2026-04-29 01:00 — PR #862: Slice token scan {64,96,128,192} with EMA ↩ SENT BACK (downward scan + full stack)

- Branch: `willowpai2e1-frieren/slice-scan`
- Hypothesis: More slice tokens give PhysicsAttention richer mesh decomposition. PR #774 showed slim+2x-slices config (slice=128) won on the pre-EMA baseline. Scan ∈ {64, 96, 128, 192} with EMA to find the true optimum.

| slice_num | val_avg/mae_surf_p | best_epoch | test_avg/mae_surf_p | peak VRAM | W&B run |
|----------:|-------------------:|:----------:|--------------------:|:---------:|---------|
| **64** | **111.85** | 14 | **99.99** | 42.1 GB | [lrcnin0y](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/lrcnin0y) |
| 96 | 119.24 | 11 | 107.45 | 47.6 GB | [ciehqon3](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/ciehqon3) |
| 128 | 136.41 | 11 | 124.14 | 54.5 GB | [68hlrshj](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/68hlrshj) |
| 192 | 137.45 | 9 | 123.00 | 68.4 GB | [sn2akl99](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/sn2akl99) |

EMA baseline (PR #773, val=119.35). All runs: no Huber, no clip.

Per-split (slice64): val: single=135.03, rc=121.20, cruise=87.30, re_rand=103.87. Test: single=116.44, rc=109.75, cruise=74.30, re_rand=99.47.

**Analysis and conclusions:**

val_avg vs slice_num is **monotonically increasing (worse)** — opposite of the original PR #774 hypothesis. The pre-EMA slim+2x result was misleading (fewer epochs, different baseline). Under the same EMA + wall-clock budget, slice count above 64 both uses more memory and reaches fewer epochs before timeout. Slice attention compute grows O(slice_num × N), costing ~3 extra epochs at slice128 and ~5 at slice192 vs slice64.

The optimum is at the **lower boundary of the scan** — strongly suggesting {32, 48} are worth exploring. Lower slice_num = coarser mesh tokenization = more nodes per slice token = more global context per token. With Huber+EMA+clip already handling the training dynamics, the model may not need fine-grained slice resolution. Also: lower slice count frees VRAM (42 vs 68 GB) that could be reinvested in batch_size or n_layers.

**However, runs predate PR #769 (Huber) and PR #775 (clip+warmup) merges.** slice64+EMA at val=111.85 does NOT beat the 4-way stack baseline (96.54). **Sent back for downward scan {32, 48, 64} on full stack.**

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
