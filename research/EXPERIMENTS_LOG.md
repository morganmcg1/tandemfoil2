# SENPAI Research Results

<!-- This log is maintained by the advisor. Each entry records a reviewed experiment PR. -->

## 2026-04-29 01:00 — PR #879: Wider hidden dim: n_hidden 128→256 for more capacity under L1 loss (CLOSED)
- Branch: charliepai2e5-thorfinn/wider-hidden-dim-256
- Hypothesis: Increase Transolver hidden dimension from 128→256 (2.6M params) to provide more model capacity for the pressure field, under L1 loss.
- Results:

  | Split | surf p (PR #879, AdamW) | surf p (baseline #799, Lion) | surf p (baseline #798, AdamW) | Δ vs Lion baseline |
  |-------|-----------------------:|-----------------------------:|-----------------------------:|-------------------:|
  | val_single_in_dist     | 159.50 | 92.02  | 126.62 | +73.3% |
  | val_geom_camber_rc     | 129.46 | 87.77  | 110.45 | +47.5% |
  | val_geom_camber_cruise |  88.81 | 57.97  |  65.88 | +53.2% |
  | val_re_rand            | 107.58 | 71.42  |  86.84 | +50.7% |
  | **avg (surf p)**       | **121.34** | **77.30** | **97.45** | **+57.0%** |

  Best checkpoint: epoch 8/9. Training rate: ~225s/epoch, 9 epochs in 33.7 min (timeout-bound).
  NaN on test_geom_camber_cruise (pre-existing data/scoring.py bug).
  Metric summary: `metrics/charliepai2e5-thorfinn-wider-hidden-dim-256-c5pmcj4x.jsonl` (on student branch before deletion).

- Analysis: Closed. Two compounding failure modes:
  1. **Wrong optimizer**: Run used AdamW (lr=5e-4, wd=1e-4, the pre-PR-799 defaults), not Lion. The current baseline's key improvement is Lion's sign-based updates (−20.7%). Testing capacity expansion without Lion is an invalid comparison.
  2. **FLOP-budget mismatch**: n_hidden=256 yields 2.6M params vs ~1.0M at 128, making each epoch ~61% slower (225s vs ~140s). Only 9 epochs completed in 30 min vs 14 for the baseline — fewer gradient steps under a fixed-wall-clock budget is a net loss for L1 convergence.
  The model's per-epoch trajectory showed continued convergence at timeout (121→124 oscillation), and the student's analysis correctly identified these causes. The experiment would need: (a) Lion optimizer, (b) n_hidden ≤ 192 to stay within ~14-epoch budget, or (c) explicit bf16 speedup to fit 256 within the wall clock. Assigned thorfinn a slice_num sweep instead as the next capacity-related experiment.

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

## 2026-04-28 22:30 — PR #805: Preprocess MLP +1 residual layer (n_layers=0->1) for richer embeddings (CLOSED)
- Branch: charliepai2e5-tanjiro/preprocess-mlp-depth
- Hypothesis: Adding a residual layer to Transolver's preprocess MLP (n_layers=0->1, res=False->True) provides richer input embeddings without OOM.
- Results:

  | Metric | This PR | Old baseline (#738) | Current baseline (#798) |
  |---|---|---|---|
  | val_avg/mae_surf_p | 138.60 | 128.83 | **97.45** |

- Analysis: Closed. ~7.6% regression vs old baseline, ~42.2% regression vs current baseline. +16,384 params, peak GPU 44.10 GB, 31.87 min. Extra preprocess depth likely amplifies gradient noise from L1 loss's non-smooth surface near boundaries. Student also independently confirmed the data/scoring.py NaN bug.
