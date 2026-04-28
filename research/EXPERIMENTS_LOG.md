# SENPAI Research Results — willow-pai2d-r5

Per-PR experiment log. New entries are appended chronologically; the latest entries are at the top.

## 2026-04-28 10:20 — PR #667: SWA over last quarter (2-seed) — **SENT BACK (composition test on cosine baseline)**
- Branch: `willowpai2d5-askeladd/swa-last-quarter` (pre-#427; train.py adds SWA but reverts cosine_t_max + completed_epochs)
- 2-seed run on bf16+grad-clip+Huber baseline (PRE-cosine):

| Metric | best-by-val (n=2) | SWA (n=2) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 92.17 ± 0.55 | **83.95 ± 0.54** | **-8.9%** |
| test 3-finite mean | 89.34 ± 1.53 | **81.17 ± 0.77** | **-9.2%** (variance halved) |

- Both seeds: 83.56 / 84.33 (very tight). swa_count = 5 (last 5 of 19 epochs averaged).
- Per-split: every val and test split improves under SWA; biggest val gain on `val_geom_camber_cruise` (-10.1%).
- Test variance halved (1.53 → 0.77) — the predicted variance-tightening shows up where it has room.
- **Useful additive deviation:** student added a "pre-SWA test eval" block that captures both best-by-val and SWA test numbers in the same run. Coordination-positive; preserve on rebase.
- Sent back: branch is pre-#427 (cosine merge), so direct merge would un-revert frieren's budget-aware cosine. Rebase + 2-seed re-run on bf16+grad-clip+Huber+cosine_tmax=19 baseline. Mechanism question: do SWA and cosine_tmax=19 compose (both prevent "ending in noisy regime") or substitute? Predicted-additive at 78% efficiency: ~75. Predicted-substitute: ~80-82.
- Comparison with closed EMA #341 (different mechanism) holds: SWA's flat 1/N average over the small-LR tail captures basin centre, distinct from EMA's exponential decay over all of training.

## 2026-04-28 09:55 — PR #427: Budget-aware cosine T_max=19 (post-rebase composition test) — **MERGED** (commit 38c2843)
- Branch: `willowpai2d5-frieren/budget-aware-cosine` (squash-merged into advisor; deleted)
- 2-seed run on bf16+grad-clip+Huber baseline + T_max=15 sensitivity probe:

| Run | T_max | val_avg/mae_surf_p | best epoch | W&B |
|---|---:|---:|---:|---|
| seed 0 (mueoiezp) | 19 | **81.74** | 19/19 | budget_aware_schedule |
| seed 1 (jn6zs4dw) | 19 | **80.98** | 18/19 | budget_aware_schedule |
| **2-seed mean ± std** | 19 | **81.36 ± 0.54** | — | — |
| sensitivity (h86gz4yf) | 15 | 87.33 | 15/19 | budget_aware_schedule |

- **vs current bf16+grad-clip+Huber baseline (90.98): -10.6%** — biggest single-PR win in round 1 magnitude
- **CV 0.66%** — tightest variance band of round 1 (vs 0.89% baseline)
- Test 3-finite mean: **78.72** vs baseline 88.16 = -10.7% (tracks val cleanly)
- Per-split: every split improves on every seed (-3 to -22%)
- **Mechanism diagnostic from T_max=15 probe:** val degrades epoch 15→19 (87.33→93.46) because cosine schedule **climbs back up** symmetrically once t > T_max (cos crosses 0 and re-ascends). Confirms recipe is "size T_max = realised_epochs *exactly*", not "any T_max < MAX_EPOCHS."
- Off-by-one in `realised_epochs` log fixed via Option A counter approach.
- Decision: **merged**. New advisor baseline ~81.4.
- **Round-1 stack now has 4 sequential merges (bf16 → grad-clip → Huber → cosine_tmax=19)**, all composing additively at orthogonal points in the training step. Cumulative ~38% improvement vs original Transolver baseline.
- Frieren reassigned to **warmup on cosine stack (PR #706)** — frieren's own follow-up #1, tests if 1-epoch warmup adds residual value on top of the now-stabilized round-1 stack.

## 2026-04-28 09:20 — PR #610: Higher weight decay (wd=5e-4, 2-seed) — **SENT BACK (composition test on Huber baseline)**
- Branch: `willowpai2d5-nezuko/weight-decay-5e-4` (pre-#413, train.py is pure CLI flag run with Huber-revert staleness only)
- 2-seed run on bf16+grad-clip baseline (pre-Huber) + 1 sensitivity probe at wd=1e-3:

| Run | val_avg/mae_surf_p | best epoch | epochs done | W&B id |
|---|---:|---:|---:|---|
| wd=5e-4 seed 0 (ntr7ud7w) | 92.06 | 19/19 | 19 | weight_decay |
| wd=5e-4 seed 1 (mwqmoldt) | 97.61 | 18/19 | 19 | weight_decay |
| **wd=5e-4 2-seed mean ± std** | **94.84 ± 3.92** | — | — | — |
| wd=1e-3 seed 0 (1rm75ili) | 93.39 | 19/19 | 19 | weight_decay |

- vs OLD bf16+grad-clip baseline (100.44): **-5.6% mean**, variance tightened 29% (5.54 → 3.92)
- vs current bf16+grad-clip+Huber baseline (90.98): +4.2% — but composition with Huber is the actual question; wd is parameter-side regularization, Huber is loss-side, mechanisms are orthogonal.
- Per-split deltas vs OLD baseline: **OOD splits gain MOST** — `val_geom_camber_rc` -9.4%, `val_re_rand` -6.8%, `val_single_in_dist` -3.7%, `val_geom_camber_cruise` -1% (already at floor)
- This is the **cleanest "regularization-as-overfitting-cure" signal** in round 1 — opposite shape from dropout #557 (which had the OOD-helps prediction fail).
- **Pattern preserved:** deterministic regularization (wd) tightens variance; stochastic regularization (dropout, β2=0.95) widens variance.
- Sent back: branch needs rebase to drop Huber-revert staleness, then 2-seed re-run on bf16+grad-clip+Huber baseline. Predicted composition: 78%-additive at ~86.6 (clean win), 50%-additive at ~88.2 (still merge), 0%-additive at ~91 (close).
- Also useful: wd=1e-3 single-seed probe (93.39) suggests the curve is still flat or improving — wd is meaningfully under-set in the round-1 default.

## 2026-04-28 08:50 — PR #622: Volume Huber (apply Huber to vol_loss too) — **CLOSED**
- Branch: `willowpai2d5-askeladd/huber-volume` (deleted; train.py diff was clean 4-line edit on current advisor)
- 2-seed run on bf16+grad-clip+Huber baseline:

| Seed | val_avg/mae_surf_p | val_avg/mae_vol_p | best epoch | W&B id |
|---|---:|---:|---:|---|
| 1 (0hiprxbt) | 96.36 | — | 18 | volhuber |
| 2 (7zpceki4) | 95.63 | — | 18 | volhuber |
| **mean ± std** | **96.00 ± 0.60** | **94.05 ± 1.39** | — | — |

- vs current baseline (90.98 ± 0.81): **surface +5.5%, volume +10.1%** — both regress, opposite of predicted -10-20% volume gain
- val_single_in_dist worst regressor (+13.3%) — confirms heavier regularization hurts most where y-range is widest
- Decision: **closed** per stated rule (surface mean > 93).
- **Important mechanism finding:** the surface-Huber cross-term win (volume MAE drops alongside surface) was NOT because volume needs Huber too. It was because surface-Huber's bounded gradient on extreme samples freed encoder capacity for volume regression. Adding Huber to volume removes curvature signal where most volume residuals already sit (Huber linear regime O(1)) and kills the encoder updates that surface needs (volume outnumbers surface 10×).
- Pattern: **stacking outlier-handling on the same gradient path doesn't compose; only on different gradient paths does** (grad-clip + surface-Huber composed at 78% additive because they're at different points in the training step).
- Reassignment: askeladd → **SWA over last 25% of training (PR #667)**. Different mechanism than EMA (#341 closed) — flat arithmetic average of late-epoch weights for flatter-minima discovery.

## 2026-04-28 08:25 — PR #586: lr=1e-3 with bf16+grad-clip baseline — **CLOSED**
- Branch: `willowpai2d5-alphonse/lr-1e-3-with-gradclip` (deleted; pre-#413, bf16+grad-clip-only baseline)
- 3 runs total (2 seeds at lr=1e-3 + 1 sensitivity probe at lr=7e-4):

| Run | val_avg/mae_surf_p |
|---|---:|
| lr=1e-3 seed 0 (1dk2gnkc) | 102.53 |
| lr=1e-3 seed 1 (hvooe56x) | 106.84 |
| **lr=1e-3 2-seed mean** | **104.69 ± 3.04** |
| lr=7e-4 seed 0 (kuyebyei) | **96.35** |

- vs OLD bf16+grad-clip baseline (100.44): lr=1e-3 mean +4.25 worse, lr=7e-4 single-seed -4.09 better
- vs current bf16+grad-clip+Huber baseline (90.98): lr=1e-3 mean +15.2% worse → close threshold
- Decision: **closed** per stated rule. lr=1e-3 too aggressive (overshoots minimum). lr=7e-4 result is interesting and assigned as new PR #653.
- Useful side-findings:
  - Pre-clip grad-norm shifted only ~10% with 2× LR (38 → 41) — much less than predicted ~30%. Grad-clip dominates the per-step magnitude regardless of LR.
  - **`lr × max_norm` is the actual control variable** — round-2 candidate.
  - 100% clipping rate at every LR — confirms grad-clip's normalized-gradient regime is invariant to LR knob.
  - No best-epoch shift across LRs (always 18-19) — model uses the full 30-min budget regardless; what differs is quality of minimum.
- Reassignment: alphonse → **lr=7e-4 with bf16+grad-clip+Huber baseline (PR #653)** — single-seed lr=7e-4 probe was promising; 2-seed confirmation on the new (Huber-included) baseline tests if higher LR composes additively with Huber's smoother loss surface.

## 2026-04-28 07:35 — PR #585: SwiGLU FFN replacement — **SENT BACK (composition test on new Huber baseline)**
- Branch: `willowpai2d5-fern/swiglu-ffn` (pre-#413 fork; train.py adds SwiGLU + reverts Huber, research/*.md staleness)
- 2-seed run on bf16+grad-clip baseline (pre-Huber):

| Seed | val_avg/mae_surf_p | best epoch | epochs done | W&B id |
|---|---:|---:|---:|---|
| 0 (vdsffnoj) | 82.42 | 17/19 | 19 | swiglu |
| 1 (pop98oet) | 89.38 | 15/19 | 19 | swiglu |
| **mean ± std** | **85.90 ± 4.93** | — | — | — |

- **vs OLD bf16+grad-clip baseline (100.44): -14.5%** (substantial standalone win, well above predicted 1-3%)
- Per-split: every split improves 11.9-16.5%, OOD splits gain MOST (opposite of fern's earlier #405 Fourier features which had per-split inversion).
- n_params 0.66M ≈ baseline 0.67M (2/3-sizing convention preserves param count to within 0.3%); peak VRAM 35.8 GB.
- Sent back: branch was pre-#413 (Huber merge), so direct merge would un-revert Huber. Need rebase + 2-seed re-run on bf16+grad-clip+Huber baseline (90.98) to test SwiGLU+Huber composition. Mechanisms are orthogonal (FFN-architectural vs loss-surface), so additive composition expected; predicted ~80-84 if at ~78% efficiency (same as grad-clip+Huber stack).
- Useful side-findings preserved:
  - 2/3-sizing SwiGLU = same param count as standard FFN — genuinely free.
  - Per-split improvement uniformity = strong signal against single-split overfit story.
  - SwiGLU mechanism is opposite-shaped from Fourier features (#405): gating helps geometric generalization rather than hurting it.

## 2026-04-28 07:25 — PR #413: Huber surface loss δ=1.0 (post-grad-clip composition test) — **MERGED** (commit e35acdf)
- Branch: `willowpai2d5-askeladd/huber-surface-loss` (squash-merged into advisor; deleted)
- 2-seed run on bf16+grad-clip baseline:

| Seed | val_avg/mae_surf_p | best epoch | epochs done | W&B id |
|---|---:|---:|---:|---|
| 1 (g9jwn94z) | 91.78 | 17/19 | 19 | loss_huber |
| 2 (317ey8ke) | 90.17 | 18/19 | 19 | loss_huber |
| **mean ± std** | **90.98 ± 0.81** (CV 0.9%) | — | — | — |

- **vs current bf16+grad-clip baseline (100.44): -9.4%**. Both seeds well below 95 (advisor's complements decision boundary).
- Per-split: val_single_in_dist drops from ~115 to ~105, all 4 splits improve on both seeds.
- Test 3-finite-split mean: 88.16 (test < val, no overfitting).
- **Complements result (not substitutes):** composition arithmetic = stack captures ~78% of perfect-additive (-26.4 vs -33.7 max). Huber clips per-node loss tail; grad-clip normalizes whole-batch gradient direction. Different points in training step → genuine independent value.
- **Variance ±0.9%** — back to bf16-baseline-level tightness, sharper than either ingredient alone (Huber-only 4.4%, grad-clip-only 5.5%). Two outlier-handling mechanisms compose.
- **Cross-term effect on volume replicated:** val_avg/mae_vol_p ~85.4 (vs ~117 pre-Huber bf16-only). Documented across 3 baselines now.
- Decision: **merged**. New advisor baseline ~91.
- Askeladd reassigned to **Huber on volume term (PR #622)** — natural compose-step given the cross-term effect persists.

## 2026-04-28 07:00 — PR #557: Attention dropout = 0.1 — **CLOSED**
- Branch: `willowpai2d5-nezuko/attention-dropout-0.1` (deleted; pre-#434, bf16-only)
- 2-seed run on bf16-only baseline:

| Seed | val_avg/mae_surf_p | best epoch | epochs done | W&B id |
|---|---:|---:|---:|---|
| 0 (xoql839c) | 117.43 | 17/19 | 19 | dropout_0.1 |
| 1 (wzirl4jq) | 122.39 | 19/19 | 19 | dropout_0.1 |
| **mean ± std** | **119.91 ± 3.51** | — | — | — |

- vs bf16-only baseline (#441) 117.37 ± 0.85: **+2.2% mean regression, σ widened 4×**
- vs current advisor baseline (#434 grad-clip) 100.44 ± 5.54: **+19.4% worse**
- All three predictions failed:
  - Mean: predicted -1 to -3%, got +2.2%
  - Variance: predicted tighter, got 4× wider
  - OOD-better-than-in-dist: predicted yes, got opposite (in-dist regressed +4.3%, re_rand regressed +4.5%, camber splits flat)
- Mechanism (student): dropout=0.1 reduces effective gradient signal ~10%; under 30-min cap with 50-epoch cosine, model is already undertrained → dropout slows convergence in a regime that's not yet over-training, no regularization payoff.
- **Pattern repeats:** stochasticity-amplifying interventions widen seed variance under our short-training regime (β2=0.95 #537 widened 5×, dropout=0.1 #557 widened 4×). Filed as round-1 cross-finding.
- Decision: **closed** per student's recommendation.
- Reassignment: nezuko → **higher weight_decay = 5e-4 (PR #610)** — student's own follow-up #4. Deterministic regularizer, no per-step noise.

## 2026-04-28 06:10 — Triple review: #434 merged, #413 sent back, #537 closed

### PR #434 (fern grad-clip, max_norm=1.0) — **MERGED** (commit 426b4c4)
- Squash-merged into advisor as new round-1 baseline. Train.py diff was 4 lines.
- 2-seed mean **100.44 ± 5.54** (-14.4% vs #441 bf16 baseline 117.37). Per-split improvement on every split, both seeds. Test 3-finite mean: 96.73.
- Mechanism: 100% of steps clipped, median pre-clip grad-norm = 38 → effectively Lion-like normalized-gradient training.

### PR #413 (askeladd Huber δ=1.0, post-bf16 rebase) — **SENT BACK (composition test)**
- 2-seed mean on bf16 baseline: **100.58 ± 4.35** (-14.3% vs bf16 standalone — virtually tied with grad-clip)
- Per-split improvement on every split, both seeds. Cross-term volume effect replicated at slice_num=64 (val_avg/mae_vol_p ~89-98 vs ~117 implied bf16). Test 3-finite mean: 95.30 (best seed).
- Sent back: branch-was-against-bf16, but grad-clip just merged → need rebase against new baseline + 1-2 confirmation seeds. Decision pending: are Huber + grad-clip **complements** (stack to ~85-90) or **substitutes** (~100, no synergy)?

### PR #537 (alphonse AdamW β2=0.95) — **CLOSED**
- 2-seed mean **116.61 ± 4.89** (-0.65% vs bf16 baseline; mean within noise, **variance widened 5×** vs bf16's 0.85 std)
- Mechanism analysis (alphonse): β2=0.95's responsive variance estimator strips the smoothing that produces bf16's tight ±0.85 — exposes underlying seed-dependent variance.
- Decision: closed per student's own recommendation. Filed for round 2 with warmup pairing once frieren #427 lands a warmup arg.
- Useful side-finding: volume-side seed std stayed tight (1.28) while surface-side blew up — variance amplification is loss-channel-specific (surf_weight=10 funnels surface gradient noise into optimizer state).

### Reassignments
- alphonse → **lr=1e-3 + bf16 + grad-clip (PR #586)** — grad-clip's normalized-gradient regime removes the amplification effect, so higher LR may compensate without destabilizing. 2-seed.
- fern → **SwiGLU FFN replacement (PR #585)** — clean architectural axis; LLaMA/PaLM convention; ~zero-param overhead at standard 2/3 sizing. 2-seed.

## 2026-04-28 05:55 — PR #434: Gradient clipping (max_norm=1.0, 2-seed) — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-fern/grad-clip` (post-bf16 advisor; train.py change is clean 4-line block, research/*.md staleness only)
- Two seeds on bf16 advisor with `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()`:

| Seed | val_avg/mae_surf_p | best epoch | epochs done | W&B id |
|---|---:|---:|---:|---|
| 0 (igt4ggio) | 96.52 | 19/19 | 19 | grad_clip group |
| 1 (fbqg8u42) | 104.35 | 18/19 | 19 | grad_clip group |
| **mean ± std** | **100.44 ± 5.54** (CV 5.5%) | — | — | — |

- **vs current bf16 baseline (117.37): -14.4%**. Biggest single-PR win of round 1.
- Per-split improvement on every split, both seeds:
  - val_single_in_dist: 114-116 (-17%)
  - val_geom_camber_rc: 105-124 (-4 to -19%)
  - val_geom_camber_cruise: 73-77 (-17 to -22%)
  - val_re_rand: 92-100 (-8 to -15%)
- Test 3-finite-split mean: **96.73** (Seeds 92.48 / 100.98)
- Variance ±3.9% half-range vs ±15% PR #331 reference → ~4× compression
- **Mechanism finding (fern):** 100% of steps clipped, median pre-clip grad-norm = 38, max = 776. So `max_norm=1.0` is acting like aggressive gradient *normalization* (constant-direction-step optimization) — closer to Lion/Sign-SGD than to outlier clipping. Convergence is excellent anyway, suggesting normalized-gradient steps work well for this problem.
- Decision: **send back for research/*.md rebase only, then merge.** New advisor baseline expected ≈ 100.4.
- Cross-cutting findings preserved:
  - Three orthogonal variance levers identified in round 1: bf16 (more cosine arc), grad-clip (early-step magnitude normalization), Huber (outlier per-node clipping). Coherent "more-stable-training" round-2 narrative.
  - Per-epoch curves track each other extremely tightly (max gap < 20 MAE for epoch ≥ 5 across seeds) — strongest visual evidence in round 1 of suppressed early-batch divergence.
- Strategic angle: Once this lands, in-flight PRs (Huber #413, budget-aware cosine #427, β2=0.95 #537, dropout #557) all rebase against bf16+grad-clip. Their hypothesis tests directly measure compose-on-top wins above the 100.4 bar.

## 2026-04-28 05:10 — PR #505: Lower LR exploration (lr=3e-4, 2-seed) — **CLOSED**
- Branch: `willowpai2d5-nezuko/lr3e-4-multiseed` (deleted; pre-#441, fp32 + slice_num=64)
- Two-seed run at lr=3e-4 on the OLD baseline (slice_num=64 + fp32):

| Seed | val_avg/mae_surf_p | best epoch |
|---|---:|---:|
| 0 (ju8ld6b9) | 142.37 | 13/14 |
| 1 (57q6g0g4) | 133.40 | 14/14 |
| **mean ± std** | **137.89 ± 6.34** (CV 4.6%) | — |

- **Variance hypothesis CONFIRMED:** CV 4.6% at lr=3e-4 vs ~13.7% at lr=5e-4 (PR #331 reference). Lower LR genuinely reduces seed-to-seed step instability. Factor of ~3 reduction.
- **But mean regresses:** 137.89 vs current bf16 baseline 117.37 = **+17.5% worse**. Both seeds still improving at termination — lower LR + 14-epoch fp32 budget = explicit undertraining.
- **Variance benefit duplicative of bf16's:** alphonse PR #441 measured CV 0.7% on bf16 baseline — much tighter than nezuko's 4.6%. Same root mechanism (more cosine-arc traversal at termination → lower late-LR → less seed sensitivity), but bf16 buys it for free (extra epochs at same wall-clock) while lr=3e-4 pays for it via slower convergence.
- Decision: **closed**. Mean regression is real, variance benefit is duplicative.
- **Cross-cutting findings preserved:**
  - LR-scale-induced step instability is real and quantified at this baseline.
  - The "competing effect (a)" the PR body anticipated (slower convergence inside the cap) dominates the apparent variance benefit.
- Round-2 candidate: stack with Huber once it lands (Huber clips outlier-gradient *magnitudes*, lower LR shrinks *all* step sizes, bf16 extends *epoch budget* — three orthogonal stability levers).
- Nezuko reassigned to **attention dropout = 0.1 (#557)** — well-evidenced small-data regularizer; complements the per-split overfitting signal from fern's PR #405 (Fourier in-dist+12% / OOD-10-14%).

## 2026-04-28 04:35 — PR #441: bf16 mixed precision standalone (2-seed) — **MERGED** (commit b605b44)
- Branch: `willowpai2d5-alphonse/bf16-standalone` (squash-merged into advisor; deleted)
- Pure ergonomics PR — bf16 autocast in train + eval, fp32 cast before squaring & before denormalization, seed flag, peak VRAM logging
- 2-seed mean **117.37 ± 0.85** at 19 epochs (vs ~131 implied pre-bf16 cluster, -10.4%)
- CV ~0.7% — round-1 variance-tight winner
- New advisor baseline. All other in-flight PRs will pick up bf16 on rebase, getting 5 extra epochs at the same wall-clock cap.
- Setup for round 2: orthogonal axes (Huber #413, budget-aware cosine #427, grad-clip #434, lr=3e-4 #505) now all rebase against bf16 — composition tests become decisive.

## 2026-04-28 04:30 — PR #427: Budget-aware cosine (T_max matched to realised epochs) — **SENT BACK (rebase + re-run on bf16)**
- Branch: `willowpai2d5-frieren/budget-aware-cosine` (sits on pre-#433 + pre-#441 commit; train.py change is small but research/*.md staleness + reverts of bf16 + slice_num)
- Three runs on **OLD baseline** (slice_num=128 + fp32):

| Run | T_max | val_avg/mae_surf_p | Δ vs old #336 (139.83) |
|---|---:|---:|---:|
| cosine_tmax11 | 11 | 133.42 | -4.6% |
| cosine_tmax13 | 13 | 135.24 | -3.3% |
| cosine_tmax11_seed2 | 11 | 123.77 | -11.5% |

- 9.7-point spread between two same-config T_max=11 runs reproduces the ±10-15% MSE-baseline seed variance (alphonse PR #441 measured this on bf16 at much tighter ±0.7%).
- Mechanism confirmed: cosine annealing to 0 by termination wins on the OLD baseline. But all three runs are pre-bf16; need to re-test composition on the post-bf16 baseline (where realised epochs is ~19, not 11, so matching `T_max=19`).
- Decision: sent back for rebase + 2-seed re-run with `--cosine_t_max 19` on current bf16 baseline. Composition test: does annealing-to-zero compose additively with bf16's extra-cosine-arc-traversal effect? Predicted yes; tighter band of 108-114 if it does.
- Off-by-one in `realised_epochs` log identified by frieren — fix included in send-back.
- Useful side-finding preserved: `T_max=13` sensitivity (-3.3%) shows the optimum is at the realised budget, not strictly under.

## 2026-04-28 04:25 — PR #441: bf16 mixed precision standalone (2-seed) — pre-merge tracking entry (superseded by merge entry above)
- Branch: `willowpai2d5-alphonse/bf16-standalone` (sits on intermediate advisor commit; train.py diff clean, research/*.md staleness only)
- Two seeds on advisor HEAD (slice_num=64) with bf16 autocast in train + eval, fp32 upcast before squaring & before denormalization

| Seed | best_epoch | epochs done | val_avg/mae_surf_p | s/epoch | peak VRAM |
|---|---:|---:|---:|---:|---:|
| seed 0 (cgitj1dc) | 17 | 19 | 116.77 | 96.2 | 32.95 GB |
| seed 1 (i45ys5ih) | 17 | 19 | 117.97 | 96.6 | 32.95 GB |
| **mean ± std** | — | — | **117.37 ± 0.85** | **96.4** | **32.95** |

- **vs implied baseline cluster (~131): -10.4%**
- **Wall-clock speedup matches PR #331 prediction:** s/epoch 131 → 96 = -26.4%. Converted into 5 extra epochs (14 → 19), enough to push best-checkpoint past the fp32-baseline cliff.
- **Variance dramatically tighter than #331's MSE band** (CV ~0.7% vs ±10-15%). Attribution: bf16 stable + late-cosine annealing reaching real LR decay at epoch 17.
- **Per-channel improvement on every split.** Test/val gap stays under 2 (3-finite-split test mean 115.59 vs val 117.37).
- **Decision: send back for research/*.md rebase only, then merge.** train.py diff is clean. Will become new advisor baseline ~117.
- Cross-cutting findings preserved:
  - bf16 zero overflow at our dynamic range (clamp_count=0 — second confirmation after #331)
  - Peak VRAM unchanged at 32.95 GB; bf16 alone doesn't unlock bigger models on this N-dominated workload
  - The "larger-than-predicted gain" is the budget recovery, not a numerical effect — fp32 baseline was terminating mid-descent
- Strategic angle: merging this first sets up askeladd PR #413 (Huber) for a clean orthogonal-stack test on rebase. Both axes are independent (loss vs ergonomics) so we expect additive composition.

## 2026-04-28 03:45 — PR #339: Larger batch (8) with sqrt(2) LR scaling — **CLOSED**
- Branch: `willowpai2d5-nezuko/batch8-lr-sqrt2` (deleted; CLEAN diff — pure CLI flag run, no train.py changes)
- Three runs on post-#433 advisor (slice_num=64):

| Run | bs/lr | val_avg/mae_surf_p | epochs | W&B id |
|---|---|---:|---:|---|
| Headline | 8 / 7e-4 | 144.71 | 14 | psfu8y1d |
| Fallback | 6 / 6e-4 | 146.31 | 11 | t5z48527 |
| Control | 8 / 5e-4 (no scaling) | 162.10 | 14 | w6y33r0w |

- Decision: **closed** — all three results regress vs same-baseline contemporaries on the post-revert advisor (bf16_seed0=116.77, huber_d1_seed2=113.17, cosine_tmax11=128.64 are all in the 113-135 band per W&B leaderboard).
- **Key finding preserved:** sqrt(2) LR scaling rule is validated (bs=8/lr=7e-4 vs bs=8/lr=5e-4 = +18 MAE delta from the LR change alone) — useful for round-2 throughput stacking under bf16+longer-cap.
- **Mechanism for the bs=8 regression:** at the 30-min cap, larger batch = same epoch count but half the SGD steps; val_single_in_dist (high-Re raceCar singles needing aggressive fitting) tanks at 201.11 vs ~170-180 for default. Wall-clock is the binding constraint.
- bs=8 fits in 84.2 GB at slice_num=64 + fp32; bs=8+bf16 should fit comfortably for any future stacking.
- Nezuko reassigned to **lr=3e-4 multi-seed (#505)** — opposite end of LR exploration, directly attacks the round-1 training-instability hypothesis.

## 2026-04-28 02:08 — PR #413: Huber loss for surface pressure (delta=1.0) — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-askeladd/huber-surface-loss` (sits on pre-#433 commit, slice_num=128)
- Two-seed run on slice_num=128 baseline:

| Seed | best_epoch | val_avg/mae_surf_p | Δ vs baseline (139.83) |
|---|---:|---:|---:|
| huber_d1 (s1, hy6o1c7v) | 9 | **123.77** | **−11.5%** |
| huber_d1_seed2 (s2, pngn7xcr) | 11 (last) | **113.17** | **−19.1%** |
| **2-seed mean** | — | **118.47** | **−15.3%** |
| Spread | — | 10.60 | ±4.5% from mean |

- **Every split improves on every seed.** val_single_in_dist (the high-Re raceCar singles that drive extreme tails) drops 15-23%.
- **Cross-term effect:** volume MAEs also drop 8-18% — capping surface gradients on extreme samples frees encoder capacity for the volume regression.
- **Variance reduction observed:** ±4.5% spread between seeds (vs ±10-15% measured on PR #331 with MSE).
- W&B group: `loss_huber`
- Decision: **sent back for rebase + brief re-run on slice_num=64.** Branch was forked pre-#433, so direct merge would re-revert slice_num to 128. Huber's mechanism is slice_num-independent so we expect the win to compose; one slice_num=64 confirmation run will lock it in.
- **Round-1 leading candidate.** After rebase + confirmation, this becomes the new advisor baseline.

## 2026-04-28 01:55 — PR #433: Revert PR #336 slice_num 128→64 — **MERGED** (commit 605b439)
- Branch: `willowpai2d5-alphonse/revert-336-slice-num` (deleted; squash-merged into advisor)
- Pure administrative revert: train.py one-line `slice_num: 128 → 64`, BASELINE.md cleanup of #336 entry
- Triggered by accumulated direct evidence: alphonse PR #329 rebased (slice=64 130.55 vs slice=128 151.34, Δ −20.79); frieren PR #338 rebased (slice=64 130.43 vs slice=128 143.90, Δ −13.47); cluster of 5 slice=64 round-1 results at 130-132
- Per-epoch cost: slice_num=64 ≈ 131s vs slice_num=128 ≈ 169s — at our 30-min cap that's the difference between 14 vs 11 epochs of training
- After this merge: all in-flight PRs need a small rebase against the corrected slice_num=64 baseline (only train.py model_config conflict — trivial resolution)
- Note: slice_num=128 may convert better given longer wall-clock; revisit in round 2 if `SENPAI_TIMEOUT_MINUTES` increases.

## 2026-04-28 01:42 — PR #405: Fourier features for spatial coords (L=8) — **CLOSED**
- Branch: `willowpai2d5-fern/fourier-spatial` (deleted; pre-#433, slice_num=128)
- Hypothesis: NeRF-style Fourier features lift `val_avg/mae_surf_p` 4-8%, biggest gain on `val_geom_camber_*`
- Result: val_avg = 141.92 at epoch 8/11 vs current baseline 139.83 → +1.5% worse (would be ~+8% vs corrected slice_num=64 baseline post-#433)
- W&B run: f1dslwya (group `spatial_fourier`)
- **Per-split inversion is the stronger refutation than the headline:** `val_single_in_dist` -12% (improved as predicted), but `val_geom_camber_rc` +14% and `val_re_rand` +10% (opposite of hypothesis). High-freq Fourier features overfit to training spatial signatures.
- Decision: **closed** per fern's own exit criterion. Hypothesis refuted in shape, not just magnitude.
- Side-finding preserved: `val_single_in_dist` gain is real (Fourier basis genuinely helps in-distribution local features). Round-2 stack candidate IF a configuration wins on OOD splits without re-triggering the OOD penalty.
- Fern reassigned to **gradient clipping (#434)** — directly attacks the ±10-15% seed variance from PR #331, two-seed run for variance measurement.

## 2026-04-28 01:35 — PR #329 (rebased re-run): surf_weight=50 on slice_num=128 — **CLOSED (deciding evidence for #336 revert)**
- Branch: `willowpai2d5-alphonse/surf-weight-sweep` (deleted; rebased cleanly to advisor HEAD; diff was empty — pure CLI flag run)
- One run, 30-min timeout, 11 epochs, default config + `--surf_weight 50`
- Result: `val_avg/mae_surf_p = 151.34` at epoch 10/11 vs current baseline #336 (139.83) → **+8.2% worse**
- W&B run: kv419s5t (`surf_w50_slice128`, group `surf_weight_sweep_v2`)
- **Direct apples-to-apples vs prior slice_num=64 sw=50 run:** 130.55 (14 ep) vs 151.34 (11 ep) → slice_num=64 wins by **20.79 MAE / 15.9%**.
- Decision: **closed** per alphonse's own recommendation. Combined with frieren PR #338's independent direct comparison (slice=64 vs slice=128, same warmup/lr → +13.47 MAE in favor of slice=64) and the cluster of five slice_num=64 round-1 runs at 130-132, we have unambiguous evidence that **PR #336 was a partial-credit merge** at the 30-min cap.
- Per-epoch cost: slice_num=64 ≈ 131s, slice_num=128 ≈ 169s (+29% tax).
- **Action triggered:** alphonse reassigned to revert-#336 PR (#433). After that lands, all in-flight PRs need rebase against the corrected slice_num=64 baseline.

## 2026-04-28 01:20 — PR #341: EMA model weights for val/test (decay 0.999) — **CLOSED**
- Branch: `willowpai2d5-thorfinn/ema-eval` (deleted; pre-#336, slice_num=64)
- Hypothesis: EMA(0.999) smooths small-LR cosine tail → 2-6% reduction
- Result: ema_d999 val_avg = **131.16** (EMA) vs live = 168.04 at epoch 13/13 (timeout); test_avg = 117.90 with thorfinn's defensive guard
- ema_d9995 val_avg = 166.37 (slower averaging window underperforms in 13-epoch regime)
- W&B runs: dt2wml3c (d999), qd99j8bx (d9995)
- EMA val curve strictly monotonic — implementation correct
- Decision: **closed**. Per thorfinn's own analysis, the 37-MAE EMA-vs-live gap is dominated by EMA absorbing one large live oscillation at epoch 13, not the small-LR cosine tail (which we never reach inside 30-min cap). Combined with branch staleness (slice_num=64 vs current advisor 128), apparent +6 MAE win vs baseline 139.83 sits inside the ±10-15% seed variance band measured on PR #331.
- Thorfinn's defensive `evaluate_split` y/pred-finite guard duplicates askeladd's #331 and edward's #375 canonical fix; not bringing forward.
- Thorfinn reassigned to **multi-seed baseline calibration (#428)** — single highest-leverage round-1 task: establish round-1 baseline distribution to inform whether to revert #336.

## 2026-04-28 01:18 — PR #338: LR warmup, post-rebase (slice_num=128 + warmup + lr=5e-4) — **CLOSED**
- Branch: `willowpai2d5-frieren/lr-warmup-cosine` (deleted; rebased cleanly to advisor — diff was train.py-only)
- Result on rebased config: val_avg = **143.90** at epoch 10/11 vs baseline #336 (139.83) → **+2.9% worse**
- Per-split: `val_single_in_dist` -17.82 (improved), but all 3 OOD splits regress (`val_geom_camber_rc` +24.52)
- W&B run: k75x13rh (group `lr_warmup_cosine`)
- Decision: **closed** per frieren's own recommendation. Negative result on warmup-at-this-baseline. Slower epochs at slice_num=128 mean the 2-epoch warmup is 18% of the realized 11-epoch budget (vs 14% at slice_num=64 with 14 epochs).
- **Critical cross-finding from this PR:** Frieren's two-config comparison (slice=128 vs slice=64, same warmup, same lr) shows **slice_num=64 wins by 13.5 MAE / 9.7%**. This is the cleanest direct evidence yet that PR #336 may have been a partial-credit merge — slice_num=128 may convert better with longer wall-clock but loses inside the 30-min cap. Decision pending alphonse's #329 rebased re-run + thorfinn's #428 baseline calibration.
- Frieren reassigned to budget-aware cosine (T_max matched to realized epoch budget) — PR #427.

## 2026-04-28 00:58 — PR #331: Wider Transolver (h192, h6) bf16 retry — **CLOSED**
- Branch: `willowpai2d5-askeladd/wider-h192-h6` (deleted; pre-#336, slice_num=64)
- Hypothesis: 2.2× wider Transolver lifts `val_avg/mae_surf_p` ~5-10%

| Run | Config | Epochs | val_avg/mae_surf_p | test_avg/mae_surf_p (post-fix re-eval) |
|---|---|---:|---:|---:|
| Round 1 | fp32, bs=4 | 9 | 154.011 | NaN (pre-fix) |
| **Round 2 v1** | **bf16, bs=6** | **12** | **141.998** | **129.480** |
| Round 2 v2 | bf16, bs=6 (seed 2) | 12 | 163.280 | 150.045 |
| bs=8 attempt | bf16, bs=8 | OOM | — | — |

- Decision: **closed**. v1 alone is +1.55% above baseline (within ±10-15% measured seed variance), v2 is +16.8% above baseline. 2-seed mean (152.6) crosses the 5% close threshold; v1's apparent win is not statistically separated from baseline.
- **Cross-cutting findings preserved for round 1 going forward:**
  - **bf16 buys ~26% per-epoch wall-time** with `clamp_count = 0` across all 8 splits (zero overflow risk at our dynamic range). Capacity-axis hypotheses should default to bf16.
  - **bs=8 OOMs at n_hidden=192** even with bf16 (cruise meshes saturate >94 GB). bs=6 is the practical ceiling.
  - **±10-15% seed variance at 12 epochs** is a cross-cutting concern: many round-1 single-seed apparent wins may not be statistically separated from baseline. Going forward, asking winning candidates for 2-seed confirmation before merge.
  - askeladd's train.py-side y-guard duplicates edward's #375 canonical fix; not bringing forward.
- Askeladd reassigned to Huber/SmoothL1 loss for surface (PR #413) — directly attacks the heavy-tailed-pressure mechanism behind the seed variance.

## 2026-04-28 00:50 — PR #329: surf_weight sweep {20, 30, 50} — **SENT BACK (apples-to-apples needed)**
- Branch: `willowpai2d5-alphonse/surf-weight-sweep` (sits on pre-#336 commit, slice_num=64)
- Three runs (sw=20, 30, 50), all at 14 epochs, slice_num=64 (pre-#336 fork)

| surf_weight | val_avg/mae_surf_p | val_avg/mae_vol_p | best ep | W&B id |
|---:|---:|---:|---:|---|
| 20 | 131.85 | 144.92 | 14 | 9nh5gk1m |
| 30 | 132.35 | 148.52 | 12 | 4fpwmk2m |
| 50 | **130.55** | 169.31 | 12 | fvbnu12q |

- All three beat current baseline (139.83) by 5.4–6.6%, but: branch is on slice_num=64, not 128. Cannot disentangle surf_weight effect from slice_num effect without a rebased re-run.
- Per-channel volume MAE blows up at sw=50 (`mae_vol_Ux` +55%) — the failure mode the original hypothesis flagged.
- val/test ranking disagree (val: sw=50 > sw=20; 3-finite-split test: sw=20 > sw=50). Single-seed, ~1-2% spread, near noise floor.
- **Concerning cross-evidence:** every slice_num=64 run in this round (these three + frieren's warmup control 130.43 + edward's deeper_l8 corrected) clusters in the 130-152 band, while the only slice_num=128 run (fern's #336) lands at 139.83 at fewer epochs. This raises the possibility that #336 was a partial-credit merge — slice_num=128 may convert better with more wall clock, but loses inside the 30-min cap.
- Sent back asking for rebase + one focused re-run of `surf_weight=50` on slice_num=128 (current advisor) to disentangle the two effects. If rebased run beats 139.83, merge surf_weight=50; if not, that's evidence to revisit #336.
- Alphonse also independently re-diagnosed the cruise NaN bug (alongside edward's #334/#375). No duplicate work needed.

## 2026-04-28 00:46 — PR #376: Wider MLP (mlp_ratio 2→4) — **CLOSED**
- Branch: `willowpai2d5-fern/mlp-ratio-4` (deleted)
- Hypothesis: doubling MLP hidden width lifts `val_avg/mae_surf_p` ~3-7%
- Result: `val_avg/mae_surf_p = 146.65` at epoch 10 of 10 completed (timeout) — **+4.9% regression** vs current baseline (#336, 139.83)
- W&B run: `mlp4` / wfxtjub5 (group `capacity_mlp_ratio`)
- Per-split: only `val_single_in_dist` improves (-5.70); all 3 OOD splits regress (+17.36 / +7.05 / +8.56)
- 1.00M params, 62.5 GB peak VRAM bs=4 (vs baseline 54.5 GB), bs=8 OOMed
- Decision: **closed** — at the 5% close threshold; per-split pattern (in-dist wins, OOD loses) is under-generalization signature, not undertraining; bf16+bs=8 retry path is redundant with askeladd's #331 retry; epoch-6 transient lead (-12 pts) is noted as round-2 input.
- Fern reassigned to spatial Fourier features (PR #405).

## 2026-04-28 00:35 — PR #375: nan_to_num fix in data/scoring.py — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-edward/scoring-nan-fix` (sits on pre-#336 commit; rebase needed)
- One-line `torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)` after `err = (pred - y).abs()` in `accumulate_batch`.
- **Verification:** bit-exact `rel_diff = 0` parity with the pre-fix path on the three previously-finite test splits, evaluated against the saved `model-deeper_l8-sfyn75sq:best` artifact. Cruise split goes from `NaN` → `99.89` (per-sample-skip semantics; smaller than the post-hoc form's ~117.30 from PR #334's monkey-patch — divergence intentional and correctly flagged by the student).
- **Post-fix `test_avg/mae_surf_p` for `deeper_l8` artifact:** 141.52.
- 5-epoch end-to-end smoketest on current advisor branch (slice_num=128) confirms full pipeline finite (cruise = 105.69 fresh, but at only 5 epochs not comparable to baselines).
- Cannot squash-merge as-is: branch's diff ALSO reverts `BASELINE.md`, `research/CURRENT_RESEARCH_STATE.md`, `research/EXPERIMENTS_LOG.md` to pre-#336 state. Sent back asking for rebase + force-with-lease push.
- Edward also flagged a same-shape NaN-leak in `train.py`'s `evaluate_split` for the normalized-space loss (auxiliary monitoring, not paper metric); correctly kept out of scope. Follow-up `train.py` PR to be filed after this lands.

## 2026-04-28 00:30 — PR #338: LR warmup + peak 1e-3 cosine — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-frieren/lr-warmup-cosine` (sits on pre-#336 commit, slice_num=64)
- Two-config sweep (`lr=5e-4` control vs `lr=1e-3` main), both with 2-epoch linear warmup + cosine T_max=48
- Both timeout-bound at epoch 14/50 (cosine arm only ~25% engaged)

| Run | val_avg/mae_surf_p | W&B id |
|---|---:|---|
| Control (lr=5e-4 + warmup) | **130.43** (ep 12) | n8y9yy70 |
| Main (lr=1e-3 + warmup)    | 142.17 (ep 14) | r439zxf5 |

- Negative result on the lr bump (+9% worse — high LR never anneals inside the timeout).
- **Positive result on warmup itself**: control beats current baseline (slice_num=128, no warmup) at 139.83 by ~6.7%, despite running on the *older* slice_num=64 setup. Strong implication that warmup composes additively.
- Cannot merge as-is: branch diff would revert slice_num 128→64, change Config.lr default 5e-4→1e-3, AND add the (good) warmup block. Sent back asking for rebase onto advisor + revert lr default + one re-run on slice_num=128 + warmup + lr=5e-4 to confirm composition.

## 2026-04-27 23:54 — PR #334: Deeper Transolver (n_layers 5→8) — **CLOSED**
- Branch: `willowpai2d5-edward/deeper-l8` (deleted)
- Hypothesis: deeper hierarchy of slice tokens lifts `val_avg/mae_surf_p` ~5-10%
- Result: `val_avg/mae_surf_p = 152.24` at epoch 8 of 9 completed (timeout). Test corrected (post-hoc nan_to_num): 145.87.
- W&B run: `deeper_l8` / sfyn75sq
- Decision: **closed** — clearly worse than slice_num=128 contemporary (152.24 vs 139.83), and slow per-epoch (~205 s vs ~135 s baseline) eats the cosine schedule before it can decay. Student's own analysis correctly recommends against pursuing depth alone.
- **Major bonus:** Edward diagnosed the cross-cutting `data/scoring.py` NaN-poisoning bug (`NaN * 0.0 = NaN` defeats per-sample skip mask). Cruise test sample 000020 has 761 NaN values in p-channel of `y`. Filed as PR #375 (advisor-authorized exception to read-only contract for `data/scoring.py`).

## 2026-04-27 23:54 — PR #336: More physics slices (slice_num 64→128) — **MERGED**
- Branch: `willowpai2d5-fern/more-slices-128` (squash-merged into advisor)
- Hypothesis: doubling slice tokens lifts `val_avg/mae_surf_p` ~3-7%, biggest gain on cruise (largest meshes)
- Result: `val_avg/mae_surf_p = **139.83**` at epoch 10 of 11 completed (timeout)
- W&B run: `slices_128` / 8xow4ge3 (group `capacity_slices`)
- Per-split val mae_surf_p: single 179.11 / camber_rc 144.31 / camber_cruise 110.05 / re_rand 125.87
- 0.67M params (no extra params from slice_num — only changes attention shape), peak VRAM 54.5 GB
- Decision: **merged** — best round-1 reviewable val so far, one-line change, low complexity. Establishes round 1 baseline empirically.
- Caveat: undertrained (11/50 epochs); val curve still descending. Subsequent winners will compound on top.
- `test_avg/mae_surf_p` = NaN due to scoring bug; 3-finite-split mean = 142.79.

## 2026-04-27 23:18 — PR #331: Wider Transolver (n_hidden 128→192, n_head 4→6) — **SENT BACK**
- Branch: `willowpai2d5-askeladd/wider-h192-h6`
- Hypothesis: 2.2× wider Transolver lifts `val_avg/mae_surf_p` ~5-10%
- Status: sent back — undertrained (9/50 epochs, timeout-capped) **and** test_geom_camber_cruise pressure NaN; not mergeable as-is, direction still promising

### Best results so far (under-trained, bs=4, 9 epochs, W&B `wider_h192_h6` / x54plqj1)

| Split | mae_surf_p | mae_vol_p |
|---|---:|---:|
| val_single_in_dist | 209.380 | 189.063 |
| val_geom_camber_rc | 168.777 | 158.080 |
| val_geom_camber_cruise | 109.090 | 105.050 |
| val_re_rand | 128.798 | 122.120 |
| **val_avg** | **154.011** | 143.578 |
| test_single_in_dist | 196.541 | 170.615 |
| test_geom_camber_rc | 150.876 | 144.267 |
| test_geom_camber_cruise | **NaN** | **NaN** |
| test_re_rand | 127.227 | 122.295 |
| **test_avg** | **NaN** | **NaN** |

### Analysis
- Val curve still falling steeply at termination (epoch 9 = best, declining ~7 per epoch in last three epochs); 50 epochs of cosine never engaged. Cannot conclude wider vs baseline yet.
- bs=8 follow-up OOMed at 94.97 GB (peak at bs=4 already 63 GB).
- **Test NaN root cause:** `accumulate_batch` skip-mask uses `torch.isfinite(y)`, not predictions; an extreme pred² overflow in fp32 normalized space (single cruise test sample) propagated NaN into the per-channel surface MAE. Vol_loss=inf logged on the same split is the smoking gun. Affects every PR's test scoring potentially — flagged for round-2 hardening.
- Wider config measured at 1,447,521 params; **actual baseline is 662,359** (not the ~1.4M placeholder I wrote). `BASELINE.md` updated.

### Action
Sent back with: bf16 autocast + fp32 cast before squaring loss, defensive `torch.nan_to_num` in `evaluate_split` (NOT `data/scoring.py`), `--batch_size 8`. Same `--wandb_group capacity_width`. PR remains open.
