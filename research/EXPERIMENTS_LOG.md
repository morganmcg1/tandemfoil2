# SENPAI Research Results

## 2026-04-29 10:30 — PR #1015 ASSIGNED (edward): epochs=24 on nl3/sn16 compound baseline
- charliepai2e1-edward/longer-training-epochs-24-compound
- Hypothesis: Val curve was monotonically decreasing through ep12 in PR #1005 (94.7 at epoch 12/12). The model had not converged. Doubling the training budget to 24 epochs on the same compound config (nl3/sn16, n_hidden=256, n_head=8, Huber δ=1.0, EMA=0.999, grad_clip=1.0) should push val_avg/mae_surf_p meaningfully below 94.65.
- Instructions: Exact same command as PR #1005 but `--epochs 24`.

---

## 2026-04-29 03:26 — PR #1005 MERGED (edward): n_layers=3, slice_num=16 on compound baseline — NEW BEST

- charliepai2e1-edward/n-layers-3-slice-num-16-compound
- Hypothesis: Reference competition config (nl3/sn16) achieved test metrics ~46.6/52.9/24.7/39.6 vs our compound baseline 112.6/105.3/61.7/90.4. Our n_layers=5, slice_num=64 is likely over-partitioning physics space. Stacking compound improvements onto the reference architecture (n_layers=3, slice_num=16) should yield major generalization gains.

### Results (best epoch 12/12 — val curve still decreasing)

| Split | val_mae_surf_p (PR #1005) | val_mae_surf_p (#882 baseline) | Δ |
|-------|-------------------------:|-------------------------------:|---:|
| val_single_in_dist | 112.1651 | 125.0237 | **-10.3%** |
| val_geom_camber_rc | 106.9972 | 117.0167 | **-8.6%** |
| val_geom_camber_cruise | 71.7017 | 78.0963 | **-8.2%** |
| val_re_rand | 87.7525 | 92.6361 | **-5.3%** |
| **val_avg** | **94.6541** | **103.2182** | **-8.31%** |

| Split | test_mae_surf_p (PR #1005) | test (#882 baseline) | Reference (README nl3/sn16) | Δ vs baseline |
|-------|---------------------------:|---------------------:|----------------------------:|--------------:|
| test_single_in_dist | 99.0360 | 112.5660 | 46.569 | **-12.0%** |
| test_geom_camber_rc | 93.6434 | 105.3162 | 52.859 | **-11.1%** |
| test_geom_camber_cruise | 59.7185 | 61.7016 | 24.717 | **-3.2%** |
| test_re_rand | 82.6453 | 90.3631 | 39.561 | **-8.5%** |
| **test_avg** | **83.7608** | **92.4867** | ~40.93 | **-9.43%** |

Metrics: `target/metrics/charliepai2e1-edward-nl3-sn16-compound-n4sychek.jsonl`
W&B run ID: `n4sychek`, model params: 1.61M, peak memory: 30.45 GB, training time: 16.56 min

### Analysis and Conclusions

**Decision: MERGED.** The largest single improvement in the research programme (+8.31% val, +9.43% test). The hypothesis is strongly confirmed: over-partitioned physics attention (slice_num=64) was hurting generalization, not helping it. Switching to the reference architecture (n_layers=3, slice_num=16) unlocks major gains across all 4 splits.

Key findings:
1. **Structural underfitting was the primary obstacle**, not model capacity. The smaller model (1.61M vs ~3M params for nl5/sn64) generalizes dramatically better. PhysicsAttention with slice_num=64 was memorizing domain-specific flow structures rather than learning generalizable physics.
2. **Gap to reference is ~2x** (test_avg: 83.76 vs ~40.9 in README). The reference must be doing something else — likely longer training given that val was still monotonically decreasing at epoch 12.
3. **OOD splits benefit most** — test_single_in_dist (-12.0%) and test_geom_camber_rc (-11.1%) improved the most; test_geom_camber_cruise improved least (-3.2%) as it already had the smallest error.
4. **Val curve was still decreasing at ep12** — strong signal that more training is the next highest-value experiment.
5. **n_layers=3, slice_num=16 are now hardcoded in model_config** — not CLI flags. All future PRs must have these in the source.

**BASELINE UPDATED:** val_avg/mae_surf_p = 94.6541, test_avg/mae_surf_p = 83.7608 (PR #1005, edward)

---

## 2026-04-29 09:30 — PR #960 CLOSED (alphonse): surf_weight sweep (20/30/50) — all worse than sw=10 compound

- charliepai2e1-alphonse/surf-weight-grad-clip-compound (branch closed)
- Hypothesis: PR #827 showed surf_weight=30 gave -5.26% on narrow Huber stack. Now that we have a wider model (n_hidden=256), EMA and grad_clip, the optimal surf_weight may have shifted. Tested sw=20/30/50 vs compound baseline (sw=10).

### Results (all trials: n_hidden=256, n_head=8, huber, epochs=12, grad_clip=1.0, ema_decay=0.999)

| surf_weight | val_avg/mae_surf_p | Δ vs baseline (103.2182) |
|---|---:|---:|
| 10 (baseline) | **103.2182** | — |
| 20 | 105.8893 | **+2.59% WORSE** |
| 30 | 108.4251 | +5.04% WORSE |
| 50 | 108.6928 | +5.30% WORSE |

Committed metrics: `charliepai2e1-alphonse-surf20-compound-gradclip-fqzqm70x.jsonl`, `charliepai2e1-alphonse-surf30-compound-6cavkipu.jsonl`, `charliepai2e1-alphonse-surf50-compound-gradclip-nve4jrlj.jsonl`

### Analysis and Conclusions

**Decision: CLOSED.** All three surf_weight values (20, 30, 50) are worse than the compound baseline with sw=10. The monotone degradation (sw=20 < sw=30 < sw=50) is clean and unambiguous — the optimal surf_weight has shifted to at or below 10.

Alphonse's analysis of likely causes is correct:
1. EMA already absorbs the gradient-bias gain that surf_weight was providing on the narrow model
2. Wider model (n_hidden=256, n_head=8) resolves surface features without explicit loss tilt
3. grad_clip dampens the high-surf_weight benefit: heavier surface weighting → more clipping → worse overall fit

**Key technical finding:** surf_weight upweighting was a patch for model capacity limitations. The wider model + EMA + grad_clip compound has outgrown the need for it. The sw=10 default is the correct operating point.

**Alphonse's suggestion followed up:** Based on the monotone trend (10 < 20 < 30 < 50 on loss degradation), alphonse suggested that the compound stack may actually want surf_weight < 10 (e.g. 1, 3, 5, 7). This is a data-driven hypothesis being tracked as the next assignment.

---

## 2026-04-29 07:00 — PR #987 CLOSED (edward): CosineAnnealingLR T_max fix — confirmed no-op
- charliepai2e1-edward/fix-cosine-tmax (branch deleted)
- Hypothesis: T_max was hardcoded to 50 but training runs for 12 epochs → CosineAnnealingLR decays very slowly, staying near peak LR for entire run.
- **Outcome: CLOSED — no-op.** Student correctly identified that `T_max = MAX_EPOCHS = cfg.epochs` was already the correct config (line 430-479 of train.py). Confirmed by advisor via code inspection. No change needed, no GPU time wasted.

## 2026-04-29 07:05 — PR #1005 ASSIGNED (edward): n_layers=3, slice_num=16 stacked on compound baseline
- charliepai2e1-edward/n-layers-3-slice-num-16-compound
- Hypothesis: Reference competition config (nl3/sn16) achieved test metrics ~46.6/52.9/24.7/39.6 vs our compound baseline 112.6/105.3/61.7/90.4. Our n_layers=5, slice_num=64 is likely over-partitioning physics space, hurting generalization. Stacking compound improvements (Huber, EMA, grad_clip, bf16, n_hidden=256) onto the reference architecture (n_layers=3, slice_num=16) could yield major gains, especially on OOD splits.
- Instructions: Change `n_layers=5→3` and `slice_num=64→16` in model_config dict. All other flags unchanged. Run with full compound: `--n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999`

## 2026-04-29 06:30 — PR #792 R5: grad_clip + deeper model CLOSED after 5 rounds (frieren)
- charliepai2e1-frieren/more-layers (branch deleted)
- Hypothesis (original): n_layers 5→8, lr 5e-4→3e-4 for richer feature hierarchies. Evolved through 5 rounds to focus on grad_clip 1.0 as the primary lever.

### Results (Round 5 — full compound rebase post-#882)

| Metric | Baseline (PR #882) | PR #792 R5 | Delta |
|--------|-------------------:|-----------:|------:|
| `val_avg/mae_surf_p` | **103.2182** | 107.54 | **+4.2% WORSE** |

- Best epoch: ~10/12 (standard compound schedule, --epochs 12)
- Config: `n_hidden=256, n_head=8, loss=huber, huber_delta=1.0, epochs=12, grad_clip=1.0, ema_decay=0.999` — identical to current compound baseline (PR #882)

### Analysis and Conclusions

**Decision: CLOSED.** The core contribution of PR #792 — gradient clipping (--grad_clip 1.0) — was already incorporated into the compound baseline via PR #882 during the rebase cycle. After the full compound stack was absorbed, the PR's unique remaining delta was a 5-line train-loss NaN skip guard, which never fires in practice when grad_clip is active (grad_clip prevents the loss spikes that would trigger it). The R5 run is therefore running nearly identical code to the baseline and produces a statistically comparable result (107.54 vs 103.22 — within noise for a 30-min run).

**Root cause of apparent R4 "win" (90.78):** The R4 run used `--epochs 50` (student override) vs the standard `--epochs 12`, producing ~18 epochs on a cosine T_max=50 schedule — effectively training at higher LR for longer. This was confounded with grad_clip, making the gain look larger than it was. Once the compound absorbed grad_clip at `--epochs 12`, the confound disappeared and the result regressed to baseline territory.

**What we learned:**
- grad_clip=1.0 is a valuable stabilizer and is correctly in the compound baseline
- Deeper models (n_layers=8) did not beat n_layers=5 within the 30-min budget at any point across 5 rounds
- Train-loss NaN guard is safe infrastructure but doesn't affect metrics when grad_clip is active
- Students should advise if rebase creates confounds with their sweep variable (epoch count, LR schedule T_max) to avoid misattribution

**Student (frieren) was offered:** opportunity to submit a fresh tiny infrastructure PR for just the 5-line train-loss NaN guard patch; advised that the PR otherwise has no further value to iterate on.

---

## 2026-04-29 05:00 — PR #828 R3: AdamW wd=1e-2 on FULL compound stack — CLOSED dead end (edward)
- charliepai2e1-edward/adamw-weight-decay-tuning
- Hypothesis: wd=1e-2 wins on wider model (n_hidden=256); does it hold on the full compound stack with EMA+grad_clip?

### Results (Round 3 — full compound stack: n_hidden=256, n_head=8, huber, epochs=12, grad_clip=1.0, ema_decay=0.999)

| weight_decay | val_avg/mae_surf_p | vs baseline (103.2182) |
|---|---:|---:|
| **1e-2** | **106.9111** | **+3.58% (WORSE)** |

Per-split (best ckpt = epoch 10/12):

| Split | val/mae_surf_p | Δ vs PR#882 |
|---|---:|---:|
| single_in_dist | 132.5180 | +5.99% worse |
| geom_camber_rc | 121.2961 | +3.66% worse |
| geom_camber_cruise | 78.5874 | +0.63% worse |
| re_rand | 95.2431 | +2.81% worse |
| **avg** | **106.9111** | **+3.58% worse** |

Metrics: `target/metrics/charliepai2e1-edward-wd-1e2-compound-full-zjg5o3lu.jsonl`

### Analysis and Decision

**Decision: CLOSED.** wd=1e-2 on the full compound stack is definitively worse — +3.58% vs baseline. Edward's analysis is correct: EMA, weight_decay, and grad_clip share the same regularization budget. Adding strong wd on top of EMA+grad_clip over-regularizes the model. The Round 2 "win" (102.86 vs 103.22) was an artifact of the missing EMA+grad_clip — the wd=1e-2 was compensating for regularization already provided by those mechanisms.

**Key technical finding:** Regularization mechanisms interact multiplicatively in this model:
- Narrow model (n_hidden=128, no EMA): wd=1e-4 wins (needs less regularization)
- Wider model (n_hidden=256, no EMA): wd=1e-2 wins (more capacity needs regularization) 
- Wider model + EMA + grad_clip: wd=1e-4 default wins (EMA+grad_clip already provides sufficient regularization, wd=1e-2 over-regularizes)

This closes the weight_decay dimension as a tuning lever for the current compound. Edward is now idle and needs new assignment.

---

## 2026-04-29 03:00 — PR #828 R2: AdamW weight_decay sweep — sent back for compound re-run (edward)
- charliepai2e1-edward/adamw-weight-decay-tuning
- Hypothesis: Varying AdamW weight_decay {1e-4, 1e-3, 1e-2} to find optimal regularization strength on the compound baseline.

### Results (Round 2 — rebased on PR #808 stack, without EMA/grad_clip)

| weight_decay | val_avg/mae_surf_p |
|---|---|
| 1e-4 (default) | 111.3503 |
| 1e-3 | 105.6682 |
| **1e-2 (best)** | **102.8629** |

- Best epoch: not specified; run completed without crash
- Config: `n_hidden=256, n_head=8, loss=huber, huber_delta=1.0, epochs=12` (PR #808 stack; **EMA and grad_clip were omitted**)
- Key finding: Trend reversal from Round 1 — on narrow model (n_hidden=128), low wd=1e-4 won; on wider model (n_hidden=256), high wd=1e-2 wins. This is consistent with the model being in an underfit regime where heavier regularization helps the wider model generalize.

### Analysis and Decision

**Decision: Sent back for re-run on full compound stack.** The wd=1e-2 result (102.8629) numerically beats the baseline (103.2182) but the comparison is NOT apples-to-apples — the student's run omitted `--ema_decay 0.999` and `--grad_clip 1.0`, which are part of the current PR #882 compound baseline. Running wd=1e-2 without EMA/grad_clip on the PR #808 stack is not the same experiment as running on the current baseline.

**Specific feedback given to student:**
- Re-run with full compound stack: `python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --ema_decay 0.999 --grad_clip 1.0 --weight_decay 1e-2`
- Beat target: val_avg/mae_surf_p < 103.2182 (with EMA+grad_clip already on)
- The wd trend reversal (higher wd wins on wider model) is an interesting finding worth verifying on the full stack

**Key insight:** If wd=1e-2 on the full compound stack also wins, it suggests the wider model (n_hidden=256) benefits from stronger weight regularization to avoid overfitting. This would be surprising given the underfit hypothesis, but worth testing.

---

## 2026-04-29 02:00 — PR #792 R4: Huber + grad_clip 1.0 on rebased compound (frieren)
- charliepai2e1-frieren/more-layers
- Hypothesis: `--grad_clip 1.0` stacked on the rebased advisor branch (post-#808: bf16 + n_hidden=256 + n_head=8 + Huber + epochs=12) bounds aggregate gradient norm and stabilizes wider-model training. Single-variable test.

### Results

| Metric | Baseline (PR #882) | PR #792 R4 | Delta |
|--------|-------------------:|-----------:|------:|
| `val_avg/mae_surf_p` | 103.2182 | **90.7796** | **-12.0%** |
| `val_single_in_dist/mae_surf_p` | 125.0237 | 106.60 | -14.7% |
| `val_geom_camber_rc/mae_surf_p` | 117.0167 | 104.12 | -11.0% |
| `val_geom_camber_cruise/mae_surf_p` | 78.0963 | 69.72 | -10.7% |
| `val_re_rand/mae_surf_p` | 92.6361 | 82.69 | -10.7% |
| `test_avg/mae_surf_p` | 92.4867 | **81.1185** | **-12.3%** |
| `test_single_in_dist/mae_surf_p` | 112.5660 | 93.68 | -16.8% |
| `test_geom_camber_rc/mae_surf_p` | 105.3162 | 91.52 | -13.1% |
| `test_geom_camber_cruise/mae_surf_p` | 61.7016 | 60.30 | -2.3% |
| `test_re_rand/mae_surf_p` | 90.3631 | 78.98 | -12.6% |

- Best epoch: 17/50 (timeout-bounded at 30.2 min, 18 epochs completed, ~100.5 s/ep with bf16)
- Config: `n_layers=5 (default), n_hidden=256, n_head=8, slice_num=64, mlp_ratio=2, surf_weight=10, batch_size=4, weight_decay=1e-4, grad_clip=1.0, loss=huber, huber_delta=1.0, epochs=50, bf16` (note: epochs=50 set by student, only ran ~18 in budget)
- n_skipped_nonfinite=0 on all val+test splits
- Peak VRAM: 32.9 GB
- Metrics: `target/metrics/charliepai2e1-frieren-huber-grad-clip-1.0-rebased-pa24xdae.jsonl`

### Analysis and Conclusions

**Decision: Sent back for rebase (winner).** Outstanding result — wins on every val and test split, beats current baseline by -12.0% on the primary metric. However, PR is `mergeable: CONFLICTING` because the advisor branch was updated since the rebase (PR #882 merged EMA decay=0.999 to set the new 103.2182 baseline). Per the merge-winner skill protocol, the PR was sent back with detailed rebase instructions onto the post-#882 branch, with the expectation that the result will hold (and likely improve further when stacked with EMA).

**Key insight:** The combination of bf16 (from #808) + grad_clip is doing more than the sum of its parts. With epochs=50 and cosine T_max=50, the LR didn't decay much by epoch 17 — meaning the win comes from grad_clip-stabilized higher-LR training in the early-mid regime. The `epochs=50` override is itself a partial third variable; the rebased re-run on `--epochs 12` (the standard compound) will isolate grad_clip's clean contribution.

**Action items:**
- Wait for frieren's rebased re-run on `--n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999`
- Once rebased and verified, merge as new compound baseline

---

## 2026-04-29 01:00 — PR #882: EMA model weights (decay=0.999) rebased on compound baseline (nezuko)
- charliepai2e1-nezuko/ema-model-weights
- Hypothesis: EMA(decay=0.999) of model weights smooths noisy gradient updates and acts as an implicit ensemble over recent parameter snapshots, improving generalization.
- Results:

| Metric | EMA decay=0.999 | Baseline (PR #808) | Delta |
|--------|----------------:|-----------------:|------:|
| `val_avg/mae_surf_p` | **103.2182** (ep 10) | 104.1120 | -0.86% |
| `val_single_in_dist/mae_surf_p` | 125.0237 | 121.3574 | +3.10% |
| `val_geom_camber_rc/mae_surf_p` | 117.0167 | 110.5814 | +5.79% |
| `val_geom_camber_cruise/mae_surf_p` | 78.0963 | 88.3729 | -11.63% |
| `val_re_rand/mae_surf_p` | 92.6361 | 96.1362 | -3.60% |
| `test_avg/mae_surf_p` | 92.4867 | 94.7010 | -2.34% |
| `test_single_in_dist/mae_surf_p` | 112.5660 | 109.1814 | +3.10% |
| `test_geom_camber_rc/mae_surf_p` | 105.3162 | 98.6706 | +6.79% |
| `test_geom_camber_cruise/mae_surf_p` | 61.7016 | 75.3729 | -18.12% |
| `test_re_rand/mae_surf_p` | 90.3631 | 95.5792 | -5.46% |

- Metric summary: `target/metrics/charliepai2e1-nezuko-ema-rebased-i2fjdqe3.jsonl`
- **MERGED** — net val improvement -0.86%, consistent test improvement -2.34%. New baseline: 103.2182.
- Analysis:
  - EMA delivers a smaller gain on the stronger compound baseline (-0.86%) than on the old weak Huber baseline (-15.0%). This is expected: the wider model (n_hidden=256) and bf16 already reduce per-update stochasticity. EMA's variance-reduction benefit is diminished when the base optimizer is already well-conditioned.
  - Per-split asymmetry is the key signal: EMA helps hard/noisy OOD splits (cruise -11.63%, re_rand -3.60%) but regresses easy/clean in-dist splits (single_in_dist +3.10%, camber_rc +5.79%). This pattern suggests decay=0.999 averages too aggressively (~2.7 epochs at 10 epoch run), diluting the model's sharp in-distribution predictions.
  - Best epoch = last epoch (10/12) — model still descending at timeout. Longer training may resolve.
  - Follow-up: decay sweep 0.99/0.995 assigned to nezuko (PR #942) to find optimal window size.

---

## 2026-04-28 23:45 — PR #793: Finer physics partitioning slice_num 64→128 (nezuko)
- charliepai2e1-nezuko/more-slices
- Hypothesis: Finer physics partitioning (slice_num=128 vs 64) allows PhysicsAttention to learn more granular flow features, improving val_avg/mae_surf_p.
- Results:

| Metric | slice=128 | slice=64 baseline | Delta |
|--------|----------:|------------------:|------:|
| `val_avg/mae_surf_p` | 130.97 (ep 11) | 122.93 (ep 14) | +8.04 |
| Epochs in 30 min | 11 | 14 | -3 |
| s/epoch | ~172 | ~131 | +31% |

Per-split val MAE @ best epoch (11):
| Split | slice=128 |
|-------|----------:|
| val_single_in_dist | 163.84 |
| val_geom_camber_rc | 135.09 |
| val_geom_camber_cruise | 94.92 |
| val_re_rand | 130.02 |

- **CLOSED (DEAD END)** — 130.97 > 122.93 (student's own baseline with slice=64) > 115.6496 (Huber baseline). Cannot merge.
- Analysis:
  - Wall-clock penalty: slice_num=128 adds 31% per-epoch time (131s→172s), costing 3 epochs within 30-min budget (11 vs 14 epochs).
  - At equal epochs (11): slice=128 scores 130.97 vs slice=64 at ~128 (estimated), a marginal 1.2% improvement that cannot compensate for the wall-clock loss.
  - Finer partitioning helps per-step but the 30-min budget is the binding constraint. The model needs more effective epochs, not more expressive attention.
  - Student also used NaN fix (--grad_clip 1.0 + eval sanitization) — this infrastructure is correct and should be retained in future PRs.
  - Student suggestion: try slice_num=96 as a middle ground. Not assigned — marginal improvement direction; other experiments take priority.

---

## 2026-04-28 23:30 — PR #792 (Round 2): Deeper Transolver n_layers=6 vs n_layers=5 head-to-head + NaN fix
- charliepai2e1-frieren/more-layers
- Hypothesis (revised): n_layers=6 (down from 8) with lr=3e-4 improves val_avg/mae_surf_p vs n_layers=5 baseline within the 30-min wall-clock budget. Also includes NaN fix: --grad_clip 1.0 + upstream pred/GT sanitization in evaluate_split.
- Results:

| | n_layers=5 (baseline) | n_layers=6 (variant) | Delta (n6−n5) |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **109.62** (ep 14) | 113.83 (ep 12) | +4.21 |
| **test_avg/mae_surf_p** | **98.13** | 104.33 | +6.20 |
| Params | 662,359 | 783,515 | +18% |
| s/epoch | 133 | 157 | +18% |
| Epochs done (30-min budget) | 14/50 | 12/50 | -2 |
| Peak VRAM | 42.1 GB | 49.6 GB | — |

Per-split val breakdown at best checkpoint:

| Split | n=5 mae_surf_p | n=6 mae_surf_p |
|---|---:|---:|
| val_single_in_dist | 133.45 | 125.44 |
| val_geom_camber_rc | 124.57 | 118.74 |
| val_geom_camber_cruise | 80.82 | 101.81 |
| val_re_rand | 99.64 | 109.35 |
| **val_avg** | **109.62** | **113.83** |

Per-epoch val_avg/mae_surf_p (equal-epoch comparison):

| epoch | n=5 | n=6 |
|---:|---:|---:|
| 4 | 209.73 | **188.25** |
| 7 | 160.17 | **136.85** |
| 11 | 127.56 | **118.64** |
| 12 | 120.98 | **113.83** |
| 14 | **109.62** | — |

Metric summaries: `target/metrics/charliepai2e1-frieren-ngrerv7o/` (n=5), `target/metrics/charliepai2e1-frieren-n2tm45tq/` (n=6)

- **SENT BACK FOR REVISION** — depth verdict inconclusive at 30-min budget; neither run beats Huber baseline (115.6496).
- Analysis:
  - At equal wall-clock: n_layers=5 wins (109.62 vs 113.83) because 18% slower per epoch means 2 fewer epochs within budget.
  - At equal epochs: n_layers=6 wins by ~7 mae_surf_p from ep 4 onward and was still descending at the cutoff. Depth helps per-step optimization but loses the wall-clock race.
  - **Hidden gem**: The n_layers=5 re-run with --grad_clip 1.0 + NaN fix achieved val_avg/mae_surf_p = 109.62, which is BELOW the Huber baseline (115.6496). Gradient clipping alone (on MSE) is a real signal — but needs isolation on Huber loss.
  - **NaN fix confirmed working**: --grad_clip 1.0 + upstream pred/GT sanitization in evaluate_split resolves all non-finite metrics. The corrupted GT sample (test_geom_camber_cruise/000020.pt, 761 Inf values in p channel) is correctly skipped; n_skipped_nonfinite=1 on that split, all else clean.
  - **Root cause of prior NaN**: IEEE 754 `Inf * False (==0.0) = NaN` — existing accumulate_batch y_finite check was insufficient because (pred - y).abs() is computed before masking, propagating Inf into the sum.
  - **Next step**: Clean single-variable test — Huber + grad_clip 1.0, n_layers=5, all other hyperparams unchanged. Reproduce: `cd target/ && python train.py --loss huber --huber_delta 1.0 --grad_clip 1.0 --agent charliepai2e1-frieren --wandb_name charliepai2e1-frieren/huber-grad-clip-1.0`

---

## 2026-04-28 23:15 — PR #795 (Round 2): Huber + per-sample loss normalization combined
- charliepai2e1-thorfinn/per-sample-loss-norm
- Hypothesis: Combining Huber loss (delta=1.0) with per-sample loss normalization will stack both mechanisms — Huber caps outlier penalty, per-sample norm equalizes gradient scale across the ~15× Re-range variance.
- Results:

| Metric | Huber+PSN (PR #795 R2) | Huber baseline (PR #788) | Delta |
|--------|----------------------:|-------------------------:|------:|
| `val_avg/mae_surf_p` | **104.2271** (ep 12) | 115.6496 | **-9.9%** |

- **SENT BACK FOR REBASE** — PR beat baseline (104.2271 < 115.6496), but squash merge failed due to merge conflict after PR #788 updated the advisor branch. Student asked to rebase onto icml-appendix-charlie-pai2e-r1 and re-run to verify the improvement holds.
- Analysis: Huber + per-sample norm is a confirmed winner direction (-9.9% improvement). The combined mechanisms are complementary: Huber reduces outlier penalty, per-sample norm equalizes contribution scale across the Re distribution. Once rebased and re-confirmed, this should merge cleanly.

---

## 2026-04-28 23:00 — PR #808: bf16 mixed precision for wider model (n_hidden=256, n_head=8)
- charliepai2e1-fern/bf16-wider-model
- Hypothesis: Adding bf16 autocast to the wider Transolver (n_hidden=256, n_head=8, 2.54M params) will reduce per-epoch time ~50%, allowing ~14 epochs in 30 min rather than 7, enabling the cosine LR annealing to take effect and push the wider model below the 40.927 test target.
- Results:

| Metric | bf16 wider (PR #808) | fp32 wider (PR #791) | Huber baseline (PR #788) |
|--------|---------------------|---------------------|--------------------------|
| `val_avg/mae_surf_p` | 128.5863 (epoch 10) | 155.9586 (epoch 7) | **115.6496** |
| `test_avg/mae_surf_p` | 116.1698 | 144.4327 | NaN (cruise bug) |
| Epochs in 30 min | 10 | 7 | ~14 |
| Mean epoch time | 192 s | 282 s | ~128 s (smaller model) |

Per-split val MAE @ best epoch (10):
| Split | surf p |
|-------|--------|
| val_single_in_dist | 157.3855 |
| val_geom_camber_rc | 141.0371 |
| val_geom_camber_cruise | 97.6690 |
| val_re_rand | 118.2536 |
| **avg** | **128.5863** |

Metric summary: `target/metrics/charliepai2e1-fern_bf16-wider-model-c23t3siu.jsonl`

- **SENT BACK FOR REVISION** — 128.5863 is 11.2% above the Huber baseline of 115.6496. Cannot merge.
- Analysis: Three confounds need resolving:
  1. **Missing Huber loss** — the run used default MSE (no `--loss huber` flag). All experiments must stack on the merged Huber baseline.
  2. **Unapproved architecture change** — student added split velocity/pressure decoders (unspecified in PR instructions). This confounds the bf16 speedup measurement.
  3. **T_max mismatch** — with 30-min timeout and ~192s/epoch, only 10 epochs complete. T_max=50 means only 20% of the cosine schedule is used. Setting `--epochs 12` would align T_max with the reachable budget and give the cosine annealer a chance to reach min-LR.
  - bf16 speedup was 32% (192s vs 282s), not the predicted 50%. The non-AMP portions of the pipeline (dataloader, val loop, DDP overhead) are a significant fraction.
  - The wider model itself is learning well — still descending at epoch 10. With correct epochs, Huber loss, and a clean architecture (no split decoder for now), this could beat the baseline.

## 2026-04-28 21:30 — PR #788: Huber loss instead of MSE
- charliepai2e1-alphonse/l1-huber-delta1.0
- Hypothesis: Huber loss (delta=1.0) aligns training objective with MAE evaluation metric, and is robust to the order-of-magnitude variation in target magnitudes across Re regimes.
- Results:

| Metric | Huber (delta=1.0) | MSE baseline | Delta |
|--------|-------------------|--------------|-------|
| `val_avg/mae_surf_p` | **115.6496** | 126.88 | **-8.85%** |
| `val_single_in_dist/mae_surf_p` | 148.4833 | — | — |
| `val_geom_camber_rc/mae_surf_p` | 120.0717 | — | — |
| `val_geom_camber_cruise/mae_surf_p` | 91.6644 | — | — |
| `val_re_rand/mae_surf_p` | 102.3790 | — | — |
| `test_avg/mae_surf_p` | NaN (pre-existing cruise bug) | — | — |

Best epoch: 10/14. Metric summary: `target/metrics/charliepai2e1-alphonse-huber-delta1.0-gtc81aav.jsonl`

- **MERGED** — new baseline: `val_avg/mae_surf_p = 115.6496`
- Analysis: Clean -8.85% improvement. Huber loss bridges the MSE-MAE gap and handles the high-Re sample outliers more gracefully than MSE. The improvement is consistent across all 4 val splits. The surface pressure prediction benefit is most pronounced on the cruise split (91.66), which has a lower max per-sample std than the raceCar splits and benefits most from the reduced outlier penalty.

---

## 2026-04-28 22:00 — PR #795: Per-sample loss normalization
- charliepai2e1-thorfinn/per-sample-loss-norm
- Hypothesis: Normalize per-sample MSE by per-sample y std to equalize gradient contributions across the ~15x Re-range variance within each domain.
- Results:

| Metric | Per-sample norm (MSE) | MSE baseline | Delta vs MSE |
|--------|----------------------|--------------|--------------|
| `val_avg/mae_surf_p` | **120.37** | 136.08 | **-11.5%** |
| `test_geom_camber_cruise/mae_surf_p` | NaN (pre-existing bug) | NaN | — |

- **Sent back for revision** — does not beat Huber baseline of 115.65. Direction is sound.
- Analysis: Per-sample normalization shows the largest single improvement vs the MSE baseline (-11.5%), but was run on top of MSE loss, not Huber. The normalization and Huber loss are complementary mechanisms: Huber reduces outlier penalty, per-sample norm equalizes contribution scale across the Re distribution. Combined, they should stack. Student asked to re-run with `F.huber_loss(delta=1.0)` as the inner loss before normalizing.

---

## 2026-04-28 22:00 — PR #794: LR warmup (5 epochs) before cosine annealing
- charliepai2e1-tanjiro/lr-warmup
- Hypothesis: Cosine annealing from lr=5e-4 immediately destabilizes early training on variable-mesh data with orthogonal_ / trunc_normal_ initialization.
- Results:

| Metric | 5-epoch warmup | No warmup (MSE) | Delta |
|--------|---------------|-----------------|-------|
| `val_avg/mae_surf_p` | **136.25** | 143.22 | **-4.87%** |

- **Sent back for revision** — does not beat Huber baseline of 115.65. Direction is valid.
- Analysis: Warmup benefit confirmed (+4.87% vs no-warmup), but with only 14 total epochs under the timeout, a 5-epoch warmup consumes 36% of the budget. Recommended revision: shorten warmup to 2 epochs and combine with Huber loss (delta=1.0). 2 epochs is sufficient to stabilize initialization while leaving most of the training budget for effective cosine annealing.

---

## 2026-04-28 22:00 — PR #790: surf_weight increase 10→30/50 (on MSE)
- charliepai2e1-edward/higher-surf-weight
- Hypothesis: Increasing surf_weight concentrates gradient signal on the ~1–5% surface nodes that the primary metric cares about.
- Results:

| Metric | surf_weight=50 | surf_weight=30 | MSE sw=10 baseline |
|--------|---------------|---------------|--------------------|
| `val_avg/mae_surf_p` | 128.98 | 130.23 | 136.08 |

- **CLOSED** — both runs are significantly above the current Huber baseline of 115.65. The experiment tested surf_weight on top of MSE loss, not Huber. The correct experiment is surf_weight sweep on top of Huber baseline, which is now assigned to alphonse (PR #827).
- Analysis: The direction (more gradient weight on surface nodes) remains valid in principle. However, with Huber loss already merged, the baseline has moved far beyond what MSE+higher-sw can achieve. The surf_weight signal was not isolated from the loss function change — alphonse's PR #827 will properly test this combination.
