# SENPAI Research Results

<!-- This log is maintained by the advisor. Each entry records a reviewed experiment PR. -->

## 2026-04-29 04:00 — PR #913: n_layers depth sweep + bf16 autocast (MERGED — NEW BEST 63.0588)
- Branch: charliepai2e5-tanjiro/n-layers-depth-sweep (squash-merged into icml-appendix-charlie-pai2e-r5)
- Hypothesis: Shallower Transolver (fewer layers) may generalize better on this dataset; combined with bf16 autocast for ~1.7× throughput gain, enabling more epochs within the wall-clock budget.
- Results (Round 2 — rebased on PR #926 config, EMA=0.995, T_max=15, sw=28):

  | Config | Best epoch | val_avg/mae_surf_p | Δ vs baseline 70.3212 | Δ vs best 67.2490 |
  |--------|------------|-------------------:|----------------------:|------------------:|
  | **n_layers=3+bf16** ⭐ | **29** | **63.0588** | **−10.3%** | **−6.22%** |
  | n_layers=4+bf16 | 22 | 65.3669 | −7.03% | −2.80% |
  | (PR #926 baseline: n_layers=5, fp32) | 14 | 67.2490 | — | — |

  Per-split at n_layers=3+bf16 (epoch 29):

  | Split | surf Ux | surf Uy | surf p |
  |-------|--------:|--------:|-------:|
  | val_single_in_dist | 0.7230 | 0.3749 | 69.5556 |
  | val_geom_camber_rc | 1.4011 | 0.6057 | 77.6902 |
  | val_geom_camber_cruise | 0.4399 | 0.2571 | 41.7779 |
  | val_re_rand | 0.9299 | 0.4210 | 63.2113 |
  | **avg** | **0.8735** | **0.4147** | **63.0588** |

  Metric files: `metrics/tanjiro-nlayers3-bf16-wyals8i4.jsonl`, `metrics/tanjiro-nlayers4-bf16-sk82pcjo.jsonl`

- Analysis: **Merged as new baseline.** Two key findings:
  1. **Shallower is better**: Monotonic trend (n_layers=6 > 5 > 4 > 3) consistent across both rounds, even after rebasing onto the best config. The Transolver with n_layers=3 has 0.420M params vs 0.541M for n_layers=5 — fewer params generalize better on this ~1.5K-sample dataset.
  2. **bf16 compounds the gain**: The same architecture runs 29 epochs in the same wall-clock budget as 14 fp32 epochs (≈2× throughput). More epochs on the cosine schedule means the model sees the full annealing cycle rather than being cut off mid-curve.
  3. **camber_rc improves most**: 82.60 → 77.69 (−5.8%) on the hardest OOD split — shallower depth reduces overfitting to training distribution geometry, helping generalization.
  4. **Uy MAE also drops**: 0.4681 → 0.4147 (−11.4%) across all splits — a clean win on velocity prediction as well.
  New baseline: **63.0588**. All subsequent experiments must rebase on n_layers=3+bf16+sw=28+T_max=15+EMA=0.995.

## 2026-04-29 05:00 — PR #977: Lion LR warmup (linear warmup 1–3 epochs before cosine decay) (CLOSED — NEGATIVE)
- Branch: charliepai2e5-alphonse/lion-lr-warmup (deleted on close)
- Hypothesis: Lion's sign-based updates may benefit from a short linear warmup before cosine decay to avoid large sign-gradient steps from a cold start. Sweep warmup_epochs ∈ {1, 2, 3}.
- Results (run against OLD config: n_layers=5, fp32, no bf16; baseline 67.2490):

  | Config | Best epoch | val_avg/mae_surf_p | Δ vs baseline 67.2490 |
  |--------|------------|-------------------:|----------------------:|
  | baseline (no warmup) | 14 | 67.2490 | — |
  | warmup-1ep | ~14 | 73.9375 | +9.9% |
  | warmup-2ep | ~14 | 71.9860 | +7.0% |
  | warmup-3ep | ~14 | 68.1991 | +1.4% |

  Per-split at best (warmup-3ep): single=68.9881, camber_rc=81.2823, camber_cruise=54.1673, re_rand=68.3585

- Analysis: **Closed — hypothesis rejected.** All three warmup variants regress vs baseline. The structural flaw: warmup_epochs shrinks the effective cosine decay budget. With T_max=15 and a 3-epoch warmup, the cosine phase runs for only 12 epochs — less refinement time than baseline. The loss curves show Lion is well-behaved from epoch 1 (no cold-start overshoot), confirming there is no cold-start pathology to cure. Monotonic trend (longer warmup = less regression) is entirely explained by less cosine budget being consumed: warmup-3ep converges to a partially-decayed LR that is closer to cosine-only. Even with re-run on the current best config (n_layers=3 + bf16, target 63.0588), the structural problem would remain unchanged. If warmup is revisited: extend T_max by warmup_epochs, or use per-iteration warmup over first ~500 gradient steps (decoupled from epoch count). Alphonse reassigned to new experiment.

  **Config note**: experiment ran on old config (n_layers=5, fp32, no bf16) without rebasing on PR #913; conclusion is unaffected since all variants regress even against the stale 67.2490 baseline.

## 2026-04-29 — PR #893: Lion lr sweep (lr=1e-4/5e-4/6e-4 vs baseline 3e-4) (SENT BACK — REBASE REQUIRED)
- Branch: charliepai2e5-frieren/lion-lr-sweep
- Hypothesis: Lion's default lr=3e-4 (borrowed from ImageNet paper) may be suboptimal for this task; sign-based step requires finer sweep. Test lr=1e-4, 5e-4, 6e-4.
- Results (run against OLD config: surf_weight=20, T_max=50, no EMA):

  | LR     | Best epoch | val_avg/mae_surf_p | Δ vs old baseline 77.30 |
  |--------|------------|-------------------|--------------------------|
  | 1e-4 ⭐ | 14/50      | 73.4909           | −4.93%                   |
  | 3e-4   | 14/50      | 77.2954           | (0%, baseline repro)     |
  | 5e-4   | 13/50      | 81.7269           | +5.73%                   |
  | 6e-4   | 14/50      | 80.5594           | +4.22%                   |

  Per-split at lr=1e-4: single=81.73, camber_rc=88.78, camber_cruise=51.47, re_rand=71.98

- Analysis: **Sent back for rebase.** The lr=1e-4 finding is directionally valid and important — Lion benefits from a smaller LR than the default 3e-4 (−4.93% improvement even without current best config). However, the experiment was run against the old config (surf_weight=20, T_max=50, no EMA=0.995), predating PR #926. Best result 73.49 does not beat current baseline 67.2490. Student instructed to rerun lr=5e-5, lr=1e-4, lr=1.5e-4 on top of the full current best config (surf_weight=28, T_max=15, EMA=0.995, Lion, L1, clip=1.0). Frieren remains assigned to #893 with the rebase task.

## 2026-04-29 — PR #922: Multi-step LR schedule for Lion optimizer (milestones=[7,11], gamma=0.3) (CLOSED — NEGATIVE)
- Branch: charliepai2e5-askeladd/multi-step-lr-lion (deleted)
- Hypothesis: MultiStepLR with milestone drops at 50% and 80% of the ~14-epoch budget (epochs 7 and 11), gamma=0.3, would outperform CosineAnnealingLR T_max=15, motivated by the Lion paper showing step-LR can beat cosine on ImageNet-scale training.
- Results:

  | Config | val_avg/mae_surf_p | Δ vs baseline 71.2882 |
  |--------|-------------------:|----------------------:|
  | MultiStepLR [7,11] gamma=0.3 | 71.5764 | +0.40% (regression) |
  | **CosineAnnealingLR T_max=15 (baseline)** | **71.2882** | — |

  Per-split: single=79.12 (−0.29 vs baseline 79.41), camber_rc=87.45 (+4.27 regression), camber_cruise=50.50 (−3.68 improvement), re_rand=69.23 (−0.85 improvement)

  Metric file: `metrics/charliepai2e5-askeladd_multistep-lion-7-11-2rtaam4i.jsonl` (branch deleted on close)

- Analysis: Closed as negative result. The 50%/80% milestone heuristic from the Lion paper is calibrated for long training schedules (~90 epochs, ImageNet). At our 14-epoch budget: (a) first drop at epoch 7 is already 50% of training — too late to benefit from the elevated initial LR compared to cosine's continuous annealing; (b) post-drop plateau (lr=9e-5 at epochs 9-11) shows MAE 81→82→83 — slightly *increasing*, wasting training budget; (c) cosine gives smooth continuous decay to near-zero within the budget, which the short schedule strongly favors. The camber_rc split regression (+4.27) confirms the late-milestone schedule is harder on OOD generalization. Student suggested tighter milestones [4,9] or [5,10] or a 3-step [5,9,12] as follow-ups. Askeladd now idle, assigned a new experiment.

## 2026-04-29 — PR #817: surf_weight sweep for L1 loss (values 10/15/20/25/30) (CLOSED — BASELINE MOVED)
- Branch: charliepai2e5-alphonse/surf-weight-l1-sweep (deleted)
- Hypothesis: With L1 loss (vs MSE), the optimal surf_weight should shift because L1 no longer quadratically inflates vol-loss gradients on high-Re outliers, making the surface/volume gradient balance different. Sweep surf_weight ∈ {10, 15, 20, 25, 30}.
- Results:

  | surf_weight | best epoch | val_avg/mae_surf_p | vs old baseline 97.4483 |
  |------------:|----------:|-------------------:|------------------------:|
  | 10          | 13/14     | 101.7187           | +4.38%                  |
  | 15          | 14/14     | 102.5887           | +5.27%                  |
  | 20          | 14/14     | 97.5026            | +0.06% (baseline repro) |
  | **25**      | **14/14** | **95.5619**        | **−1.94% (winner)**     |
  | 30          | 14/14     | 107.9856           | +10.81%                 |

  All runs: 14/50 epochs (30-min timeout), peak GPU 42.1 GB H100.

  Per-split at best (sw=25): single=123.51, camber_rc=101.52, camber_cruise=70.96, re_rand=86.26.

  Metric paths: `metrics/charliepai2e5-alphonse-surf-weight-l1-sweep-sw{10/15/20/25/30}-{run_id}.jsonl` (branch deleted on close)

- Analysis: **Closed** — PR #817's best result (95.5619 at sw=25) does not beat the current baseline of **71.2882** (PR #901 was merged during this experiment's run). Directional finding is valuable: unimodal sweep confirms optimal surf_weight with L1 shifts from 20 → 25. Consistent with hypothesis (L1 deflates outlier vol gradients, optimum re-centers higher). Sweep curve peaks clearly at 25, drops sharply at 30. Student also flagged the pre-existing test_geom_camber_cruise NaN bug (present since at least PR #798). Follow-up: test sw=25 with current best config (Lion + T_max=15); PR #894 covers {5,10,30,40} and does not include 25.

## 2026-04-28 23:59 — PR #901: Cosine LR T_max budget align: T_max 50→15 to match timeout budget (MERGED — NEW BEST)
- Branch: charliepai2e5-askeladd/cosine-tmax-budget-align
- Hypothesis: `CosineAnnealingLR(T_max=50)` is misaligned with the ~14-epoch actual runtime budget under the 30-min timeout. With T_max=15, the LR fully anneals from 3e-4 → ~0 within the available training window, providing the low-LR fine-tuning phase cosine annealing is designed for.
- Results:

  | Split | surf Ux (baseline) | surf Ux (this) | surf Uy (baseline) | surf Uy (this) | surf p (baseline) | surf p (this) | Δ surf p |
  |-------|-------------------:|---------------:|-------------------:|---------------:|------------------:|--------------:|---------:|
  | val_single_in_dist     | 1.3596 | 0.7788 | 0.4770 | 0.4462 |  92.0183 |  79.4120 | −12.61 |
  | val_geom_camber_rc     | 1.6130 | 1.4460 | 0.6790 | 0.6702 |  87.7708 |  83.1787 |  −4.59 |
  | val_geom_camber_cruise | 1.0149 | 0.4725 | 0.3605 | 0.3372 |  57.9690 |  54.1816 |  −3.79 |
  | val_re_rand            | 1.2637 | 0.9296 | 0.4993 | 0.4974 |  71.4235 |  68.3805 |  −3.04 |
  | **val_avg**            | **1.3128** | **0.9077** | **0.5040** | **0.4877** | **77.2954** | **71.2882** | **−7.78%** |

- Metric summary: `research/charliepai2e5-askeladd-cosine-tmax-15-9b1s4s0x.jsonl`
- Analysis: Clear winner. A 7.78% improvement in the primary metric `val_avg/mae_surf_p` from a one-line scheduler change. The improvement is consistent across all 4 val splits and all 3 non-NaN test splits, confirming this is a fundamental training dynamics improvement rather than a split-specific artifact. The LR table shows the key insight: at epoch 14, baseline LR is still 55% of initial (1.665e-4) while the aligned schedule reaches ~1% (3.28e-6) — the model was never reaching the low-LR refinement phase. The large Ux improvement (−31%) suggests the model was also undertrained on velocity fields. The `test_geom_camber_cruise/mae_surf_p` NaN is a pre-existing data issue unrelated to this hypothesis. All WIP students now targeting the new baseline of 71.2882.
- **New baseline: val_avg/mae_surf_p = 71.2882**

## 2026-04-28 (post-resume) — PR #823: asinh pressure target transform (scale 100/500/2000) (CLOSED)
- Branch: charliepai2e5-tanjiro/asinh-pressure-target-transform
- Hypothesis: Apply asinh(x/scale)*scale to the pressure channel before normalization to compress its long tail and reduce loss explosion on extreme high-Re samples; sweep scale ∈ {100, 500, 2000}.
- Results:

  | Run | --asinh_p_scale | val_avg/mae_surf_p | Δ vs old AdamW+L1 (97.45) | Δ vs current Lion+L1 (77.30) |
  |-----|----------------:|-------------------:|--------------------------:|----------------------------:|
  | scale=100   | 100  | ~109   | +12% | +41% |
  | **scale=500** | **500** | **99.26** | **+1.9%** | **+28.4%** |
  | scale=2000  | 2000 | ~104   | +6.7% | +35% |

  Per-split at best (scale=500): single=120.39, camber_rc=125.93, camber_cruise=62.54, re_rand=88.17.

- Metric summary: `metrics/charliepai2e5-tanjiro-asinh-p-500-zu5vml2g.jsonl` (student fork branch, deleted on close)
- Analysis: Closed as negative result. Three reasons the transform fails on top of L1+Lion: (a) L1 already handles long tails via its median-seeking property — asinh doubles up on a non-issue; (b) asinh is symmetric but pressure has an asymmetric tail (large negative suctions on upper surfaces, small positive stagnation peaks), so the squashing matches neither tail; (c) the transform attenuates mid-magnitude gradients, which dominate validation MAE. Compare to other rejected loss-shaping experiments (#822 Huber, #806 FiLM) — pattern is clear: with Lion+L1 the loss-surface is already well conditioned and additional shaping subtracts.

## 2026-04-29 02:00 — PR #824: Gradient clipping: stabilize L1 training under heavy-tailed targets (max_norm sweep) (CLOSED)
- Branch: charliepai2e5-frieren/gradient-clipping-and-weight-decay
- Hypothesis: Gradient clipping by global norm (tested values 0.5, 1.0, 5.0) would stabilize L1 training under heavy-tailed target distributions and improve the AdamW+L1 baseline (97.4483).
- Results (compared against PR #799 Lion baseline of 77.2954):

  | Run | --grad_clip | val_avg/mae_surf_p | Δ vs AdamW+L1 baseline (97.45) | clip_frac |
  |-----|------------:|-------------------:|-------------------------------:|----------:|
  | Baseline (no clip) | — | **97.4483** | — | — |
  | grad-clip-5.0 | 5.0 | 101.2255 | +3.9% | 1.00 |
  | grad-clip-0.5 | 0.5 | 103.9121 | +6.6% | 1.00 |
  | grad-clip-1.0 | 1.0 | 110.0872 | +13.0% | 1.00 |

  Per-split (best run, clip=5.0):

  | Split | clip=5.0 | L1+AdamW baseline |
  |-------|---------|-----------------|
  | val_single_in_dist     | 124.30 | 126.62 |
  | val_geom_camber_rc     | 104.16 | 110.45 |
  | val_geom_camber_cruise |  83.39 |  65.88 |
  | val_re_rand            |  93.04 |  86.84 |
  | **avg**                | **101.23** | **97.45** |

  Note: All three runs far worse than current Lion+L1 baseline of 77.2954 (PR #799).

  Metric files: `metrics/charliepai2e5-frieren-grad-clip-0.5-uavkg60o.jsonl`, `metrics/charliepai2e5-frieren-grad-clip-1.0-jd39rjdv.jsonl`, `metrics/charliepai2e5-frieren-grad-clip-5.0-ggwb2ohg.jsonl` (branch deleted, files not recovered)

- Analysis: Closed. All three clip thresholds hurt performance vs AdamW+L1 baseline (and are completely irrelevant vs the 77.2954 Lion+L1 baseline). The key finding: natural gradient norm in this setup is **mean 85–115, max 180–440** due to `surf_weight=20 * L1` keeping the surface-loss gradient large and constant. All tested thresholds (0.5/1.0/5.0) are 1–2 orders of magnitude below the typical norm — binding every single batch (clip_frac=1.00), effectively acting as a ~100x LR reduction. The original hypothesis was based on standard transformer-regime assumptions (norm ~1–10); this regime is completely different due to the heavy surface-weight amplification. Crucially, the unclipped baseline shows no instability (monotonic val improvement), confirming that clipping was solving a non-problem. Student analysis was first-rate — the grad-norm trace was the decisive evidence. Important note: the Lion+L1+clip1.0 in PR #799 succeeds because Lion's sign-gradient normalization implicitly regularizes gradient magnitude — completely different mechanism from L1+AdamW+clip.

## 2026-04-29 00:15 — PR #799: Lion optimizer + L1 loss + gradient clipping (MERGED, NEW BASELINE)
- Branch: charliepai2e5-askeladd/lion-optimizer
- Hypothesis: Lion optimizer (sign-based updates, lr=3e-4, wd=1e-2) combined with L1 loss and gradient clipping (max_norm=1.0) to stabilize training and beat AdamW+L1 baseline.
- Results:

  | Split | L1 baseline (PR #798) | Lion+L1+clip | Δ |
  |-------|-----------------------:|-------------:|---:|
  | val_single_in_dist     | 126.6157 | **92.0183** | −27.3% |
  | val_geom_camber_rc     | 110.4532 | **87.7708** | −20.5% |
  | val_geom_camber_cruise |  65.8819 | **57.9690** | −12.0% |
  | val_re_rand            |  86.8424 | **71.4235** | −17.8% |
  | **avg (surf p)**       | **97.4483** | **77.2954** | **−20.68%** |

  Metric summary: `research/charliepai2e5-askeladd_lion-l1-clip-plhsfvbu.jsonl`

- Analysis: Merged as new baseline. The Lion + L1 + grad clip combination is multiplicative: L1 replaced MSE for −24.4% (PR #798), Lion replaces AdamW for an additional −20.7% on top. Gradient clipping (max_norm=1.0) was critical — eliminated mid-training val spikes present in Lion+MSE runs (117→188 spike absent here; per-epoch val is near-monotonic). Best epoch was the last reached (14/50, timeout-bound, still descending) — clear headroom for longer training. Test NaN on camber_cruise is the pre-existing `data/scoring.py` bug, not a regression; 3/4 finite test splits show a −23.7% win, larger than the val gain. Per-epoch trajectory at epoch 14: val_avg 77.30, cosine schedule still in active descent phase. The surf_weight=20 is a carry-over from MSE era — re-tuning for Lion+L1 is the highest-priority follow-up. Student's independent confirmation of the scoring.py NaN mechanism was thorough and accurate.

## 2026-04-28 23:55 — PR #822: SmoothL1 (Huber) loss: smooth gradient near convergence (beta sweep 0.1/0.3/1.0) (CLOSED)
- Branch: charliepai2e5-nezuko/smoothl1-huber-loss
- Hypothesis: SmoothL1 (Huber) loss interpolates between L1 (large errors) and L2 (small errors), giving robustness AND smooth gradients near convergence. Tested beta=0.1, 0.3, 1.0.
- Results:

  | Run | beta | val_avg/mae_surf_p | Δ vs L1 baseline (97.4483) |
  |-----|-----:|-------------------:|---------------------------:|
  | 1   | 0.1  | 106.2121 | +8.97% |
  | 2   | 0.3  | 103.0013 | +5.69% |
  | 3   | 1.0  | 115.9634 | +18.99% |
  | **L1 baseline** | — | **97.4483** | — |

  Per-split (best run, beta=0.3, epoch 13):

  | Split | beta=0.3 | L1 baseline |
  |-------|---------|-------------|
  | val_single_in_dist     | 120.3429 | 126.6157 |
  | val_geom_camber_rc     | 114.8342 | 110.4532 |
  | val_geom_camber_cruise |  80.7244 |  65.8819 |
  | val_re_rand            |  96.1035 |  86.8424 |
  | **avg (surf p)**       | **103.0013** | **97.4483** |

  Metric summaries: `research/charliepai2e5-nezuko-huber-beta-0.1-rzazrjrc.jsonl`, `research/charliepai2e5-nezuko-huber-beta-0.3-yodd8o6e.jsonl`, `research/charliepai2e5-nezuko-huber-beta-1.0-xz81qjra.jsonl`

- Analysis: Closed. No beta variant of Huber beat pure L1. The ordering (smaller beta = better) implies the gradient-smoothing contribution near zero is harmful, not helpful. Key insight: L1's constant gradient regardless of error magnitude is a *feature* for the long-tailed pressure distribution — it prevents the loss from up-weighting already-well-predicted in-distribution samples at the expense of tail/OOD residuals. The model is not yet in the late-convergence regime (only 14/50 epochs due to timeout) where Huber's L2 region would fire. Camber cruise split regresses most under Huber, confirming L1's robustness advantage on shifted distributions. Student analysis was thorough and insightful. Suggested follow-ups noted: pinball/quantile loss (more interesting than Huber for skewed distributions), longer training via throughput improvements (bf16 single at bs4), and late-stage fine-tuning with Huber after L1 warmup.

## 2026-04-28 21:00 — PR #798: L1 loss: align training objective with MAE metric (MERGED, NEW BASELINE)
- Branch: charliepai2e5-alphonse-l1-loss
- Hypothesis: Replace MSE loss with L1 to align the training objective with the MAE evaluation metric.
- Results:

  | Split | surf Ux | surf Uy | surf p |
  |-------|--------:|--------:|-------:|
  | val_single_in_dist     | — | — | 126.6157 |
  | val_geom_camber_rc     | — | — | 110.4532 |
  | val_geom_camber_cruise | — | — |  65.8819 |
  | val_re_rand            | — | — |  86.8424 |
  | **avg**                | **1.3095** | **0.5908** | **97.4483** |

- Metric summary: `metrics/charliepai2e5-alphonse-l1-loss-2dl6j00h.jsonl`
- Analysis: Major win (-24.4% from 128.83 to 97.45). L1's median-seeking property is well-matched to the MAE evaluation metric and is more robust to the long-tailed pressure distribution induced by extreme high-Re samples. New baseline. All subsequent experiments must rebase against the L1 loss recipe.

## 2026-04-28 22:30 — PR #803: Surface-node feature noise (std=0.0025, dims 0-11) regularisation (CLOSED)
- Branch: charliepai2e5-frieren/surface-feature-noise
- Hypothesis: Small Gaussian noise on surface-node geometric features acts as augmentation, improving generalization on OOD camber/Re splits (carried over from pai2d2 win).
- Results:

  | Split | surf p (this PR) | surf p (current baseline #798) | Δ |
  |-------|-----------------:|-------------------------------:|---:|
  | val_single_in_dist     | 167.79 | 126.62 | +32.5% |
  | val_geom_camber_rc     | 150.87 | 110.45 | +36.6% |
  | val_geom_camber_cruise | 120.42 |  65.88 | +82.8% |
  | val_re_rand            | 129.92 |  86.84 | +49.6% |
  | **avg**                | **142.25** | **97.45** | **+46.0%** |

- Metric summary: `runs/surface_feature_noise.metrics.jsonl` (student branch, deleted on close)
- Analysis: Closed. Regression on every val split, with cruise hit hardest — opposite direction from pai2d2. The pai2d2 win likely depended on co-occurring regularizers (EMA/DropPath) absent in pai2e-r5. Bare noise injection acts as label-mismatch under L1 loss. Student also flagged the data/scoring.py inf*0=NaN bug on test_geom_camber_cruise (pre-existing, organizer-side fix needed).

## 2026-04-28 22:30 — PR #804: Cosine LR eta_min=5e-5 + 3-epoch linear warmup (CLOSED)
- Branch: charliepai2e5-nezuko/cosine-lr-eta-min-warmup
- Hypothesis: SequentialLR with linear warmup (3 epochs, start_factor=1e-3) followed by CosineAnnealingLR (T_max=MAX_EPOCHS-3, eta_min=5e-5) provides smoother early-training and a non-zero floor at the end.
- Results:

  | Metric | This PR | Old baseline (#738) | Current baseline (#798) |
  |---|---|---|---|
  | val_avg/mae_surf_p | 128.23 | 128.83 | **97.45** |

- Analysis: Closed. Marginal beat over old PR #738 (-0.5%) but +31.6% regression vs current PR #798 baseline. With a 14-epoch wall-clock budget out of 50 nominal, a 3-epoch warmup eats 21% of useful learning time. Student compared against stale baseline (PR #738 not #798). Could be revisited with shorter (1-epoch) warmup or no warmup, on top of the L1-loss baseline.

## 2026-04-28 23:00 — PR #802: bf16 autocast + TF32 for 2x throughput via batch_size=8 (CLOSED)
- Branch: charliepai2e5-fern/bf16-tf32-precision
- Hypothesis: torch.autocast(bf16) + TF32 saves VRAM, enabling batch_size 4→8 for ~2× effective throughput within the 30-min wall-clock budget.
- Results:

  | Run | val_avg/mae_surf_p | vs baseline (#798) |
  |-----|------------------:|-------------------:|
  | Run 1 (dsyf1v9o) | 129.14 | +32.5% |
  | Run 2 rerun (eir5oqbd) | 132.72 | +36.2% |
  | **Current baseline** | **97.45** | — |

- Metric summary: `research/charliepai2e5-fern-bf16-tf32-bs8-dsyf1v9o.jsonl`, `research/charliepai2e5-fern-bf16-tf32-bs8-rerun-eir5oqbd.jsonl`
- Analysis: Closed. ~32–36% regression vs current baseline. Throughput gain was ~1.24× not 2×; Transolver's attention slicing doesn't fully vectorize under bf16 autocast. More critically, bs4→bs8 halved the gradient update count (fewer steps in fixed 30-min budget), which under L1 loss hurt more than the extra samples-per-step helped. L1 loss converges more slowly at large batch sizes due to sign-gradient variance properties. Student independently confirmed the data/scoring.py NaN bug on test_geom_camber_cruise. Note: bf16-only at bs4 (no batch change) remains worth testing as an isolated throughput idea.

## 2026-04-28 23:30 — PR #806: FiLM domain conditioning (3 regimes: single/rcTandem/cruise) (CLOSED)
- Branch: charliepai2e5-thorfinn/film-domain-conditioning
- Hypothesis: Per-domain affine shift+scale (FiLM/AdaLN-zero recipe, 768 extra params) will reduce cross-domain distribution shift, improving OOD camber/Re splits.
- Results (L1 re-run after send-back):

  | Split | surf p (this PR) | surf p (baseline #798) | Δ |
  |-------|-----------------:|----------------------:|---:|
  | val_single_in_dist     | ~128  | 126.62 | +1.1% |
  | val_geom_camber_rc     | ~110  | 110.45 | ~flat |
  | val_geom_camber_cruise | ~90   |  65.88 | +37.9% |
  | val_re_rand            | ~107  |  86.84 | +23.2% |
  | **avg**                | **106.4943** | **97.4483** | **+9.3%** |

- Metric summary: `target/metrics/charliepai2e5-thorfinn-film-domain-conditioning-l1-bztrpe2i.jsonl`
- Analysis: Closed. FiLM conditioning is counterproductive on top of L1 loss. The student's analysis was correct: L1 loss already handles cross-domain magnitude bias via its linear penalization (median-seeking property), so the affine FiLM correction adds noise without compensating anything real. The cruise split regression (+37.9%) is the largest, suggesting FiLM's domain-specific scale shift is actively harmful when training data per domain is limited. Note: FiLM showed a marginal win (+2.6%) over the MSE baseline (128.83→125.48) — consistent with FiLM compensating MSE's quadratic domain-bias amplification. With L1 the interaction disappears and only the added noise remains.

## 2026-04-28 — PR #852: Per-channel L1 loss weighting: amplify pressure in surf_loss (p_weight sweep 2/5/10) (CLOSED)
- Branch: charliepai2e5-fern/per-channel-loss-weighting (deleted on close)
- Hypothesis: Weight the pressure channel (dim 2) more heavily within `surf_loss` via per-channel multiplicative weights `[1.0, 1.0, p_weight]`, normalising by `ch_weights.mean()` to keep `surf_weight=20` approximately calibrated. Hypothesis: amplifying the pressure gradient signal inside `surf_loss` would drive the model toward better surface pressure predictions directly.
- Results:

  | Run | p_weight | best_val_avg/mae_surf_p | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | Δ vs baseline (77.30) |
  |-----|:--------:|:-----------------------:|-------------------:|-------------------:|-----------------------:|------------:|----------------------:|
  | pweight-2  | 2  | 97.811 | 119.20 | 111.46 | 69.96 | 90.63 | +26.5% |
  | pweight-5  | 5  | 106.602 | 140.60 | 119.53 | 76.67 | 89.61 | +37.9% |
  | pweight-10 | 10 | 105.874 | 132.27 | 115.39 | 78.86 | 96.98 | +36.9% |
  | **Current baseline (PR #799)** | — | **77.2954** | 92.02 | 87.77 | 57.97 | 71.42 | — |

  All runs hit ~30-min timeout at epoch 13 or 14/50.

  Metric files (on deleted branch): `metrics/charliepai2e5-fern-pweight-2-5ojx60ru.jsonl`, `metrics/charliepai2e5-fern-pweight-5-udy75eqy.jsonl`, `metrics/charliepai2e5-fern-pweight-10-hf2pxrva.jsonl`

- Analysis: Closed. All three `p_weight` values produced large regressions vs the Lion+L1+clip baseline (77.30). Best result was `p_weight=2` at 97.81 (+26.5%). The approach fails for two structural reasons: (a) the `surf_weight=20` analogy breaks down at the channel level — globally up-weighting surface loss vs volume loss exploits different gradient flow paths than intra-channel re-weighting within an already-normalised L1 sum; (b) the mean-normalisation (`/ ch_weights.mean()`) preserves total loss scale but cannot prevent the model from over-specialising to minimise the pressure channel at the cost of Ux/Uy residuals, whose correlated errors then propagate back into camber_cruise and single_in_dist splits. The OOD splits (camber_cruise, re_rand) regress most sharply, consistent with increased sensitivity to distribution shift when the channel weighting distorts the loss landscape. Note: pre-existing `test_geom_camber_cruise` NaN/Inf bug is unrelated to this PR. Student suggested promising follow-ups: per-sample pressure variability scaling (adaptive weighting), auxiliary pressure decoder (separate objective), physics-informed regularisation terms.

## 2026-04-28 22:30 — PR #805: Preprocess MLP +1 residual layer (n_layers=0->1) for richer embeddings (CLOSED)
- Branch: charliepai2e5-tanjiro/preprocess-mlp-depth
- Hypothesis: Adding a residual layer to Transolver's preprocess MLP (n_layers=0->1, res=False->True) provides richer input embeddings without OOM.
- Results:

  | Metric | This PR | Old baseline (#738) | Current baseline (#798) |
  |---|---|---|---|
  | val_avg/mae_surf_p | 138.60 | 128.83 | **97.45** |

- Analysis: Closed. ~7.6% regression vs old baseline, ~42.2% regression vs current baseline. +16,384 params, peak GPU 44.10 GB, 31.87 min. Extra preprocess depth likely amplifies gradient noise from L1 loss's non-smooth surface near boundaries. Student also independently confirmed the data/scoring.py NaN bug.
