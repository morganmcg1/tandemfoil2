# SENPAI Research State
- 2026-04-29 03:00 (icml-appendix-charlie-pai2e-r5)
- Most recent research direction from human researcher team: None received yet.

## Current Best Baseline

**PR #799** — `val_avg/mae_surf_p` = **77.2954** (epoch 14/50, timeout-bound, still descending)
Branch: `charliepai2e5-askeladd/lion-optimizer` (merged)

Configuration: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **Lion** optimizer, lr=3e-4, wd=1e-2, surf_weight=20, batch_size=4, CosineAnnealingLR T_max=MAX_EPOCHS, **L1 loss** (vol + surf_weight * surf), grad_clip=1.0, no EMA, no bf16.

Per-split surf p MAE: single=92.02, camber_rc=87.77, camber_cruise=57.97, re_rand=71.42.

Baseline improvement trajectory: 128.83 (MSE+AdamW) → 97.45 (L1+AdamW, PR #798, −24.4%) → **77.30 (L1+Lion+clip, PR #799, −20.7%)**

**All current experiments must rebase on the Lion+L1+clip recipe.**

## Completed Experiments (this round)

### Merged (wins)
| PR | Hypothesis | Result | Δ |
|----|------------|--------|---|
| #798 | L1 loss (align objective with metric) | 97.45 | −24.4% vs 128.83 |
| #799 | Lion optimizer + L1 + clip=1.0 | 77.30 | −20.7% vs 97.45 |

### Closed (dead ends)
| PR | Hypothesis | Result | Reason |
|----|------------|--------|--------|
| #803 | Surface feature noise | 142.25 | +46% regression |
| #804 | Cosine LR warmup (3-epoch) | 128.23 | +32% vs current baseline |
| #805 | Preprocess MLP depth +1 residual layer | 138.60 | +42% regression |
| #806 | FiLM domain conditioning | 106.49 | +9.3% regression |
| #802 | bf16 autocast + TF32 + batch_size=8 | 129.14 | +32.5% regression |
| #822 | SmoothL1/Huber loss (beta sweep) | 103.00 | +5.7% regression |
| #824 | Gradient clipping at 0.5/1.0/5.0 | 101.23 | +3.9% regression (wrong regime — natural grad norm 85–115, not 1–10) |
| #879 | Wider hidden dim n_hidden 256 (AdamW, wrong recipe) | 121.34 | +57% regression vs current baseline; wrong optimizer, FLOP-bound |

## Currently Running (status:wip)

| PR | Student | Hypothesis |
|----|---------|------------|
| #901 | charliepai2e5-askeladd | Cosine LR T_max budget align: T_max 50→15 (match actual epoch budget) |
| #894 | charliepai2e5-nezuko | Lion+L1 surf_weight re-tune: sweep 5/10/30/40 vs baseline 20 |
| #893 | charliepai2e5-frieren | Lion lr sweep: test lr=1e-4, 5e-4, 6e-4 vs baseline 3e-4 |
| #908 | charliepai2e5-thorfinn | slice_num sweep: 32/64/128 physics attention bottleneck |
| #857 | askeladd | Drop-path stochastic depth regularization sweep (rate=0.1/0.2) |
| #852 | charliepai2e5-fern | Per-channel L1 loss weighting: amplify pressure channel in surf_loss |
| #823 | charliepai2e5-tanjiro | asinh pressure target transform: compress long-tailed p distribution |
| #817 | charliepai2e5-alphonse | surf_weight sweep for L1 loss (values 10/15/20/25/30) |
| #801 | charliepai2e5-edward | EMA model averaging (decay=0.995) for better generalization |

## Idle Students Needing Assignment

None — all students assigned (9/9 GPU slots occupied).

## Current Research Focus

With the **Lion + L1 + clip** combination established as the new baseline (77.30), the remaining levers to investigate are:

1. **Loss surface refinements on Lion+L1 base**: surf_weight re-tuning for Lion (not AdamW), per-channel pressure amplification, focal/hard-example mining
2. **Architecture capacity**: n_hidden sweep, slice_num, additional attention layers
3. **Regularization**: stochastic depth (in-flight #857), dropout variants compatible with Lion+L1
4. **Training dynamics**: learning rate (Lion may prefer different lr than 3e-4), warmup shape, T_max re-tuning
5. **Data pipeline**: asinh/sqrt pressure transform to compress heavy tails (in-flight #823)
6. **Ensemble / test-time**: train multiple seeds, average predictions

## Potential Next Research Directions

### High priority
1. **Lion lr sweep**: Try lr=1e-4, 5e-4, 6e-4 for Lion — 3e-4 was chosen based on Lion paper recommendations but may not be optimal for this dataset. Lion is more sensitive to lr than AdamW.
2. **surf_weight re-tune for Lion+L1**: The existing surf_weight=20 was optimized for AdamW+MSE then carried through. With Lion's sign-based updates the gradient magnitude scaling is fundamentally different — a sweep of 5/10/15/20/30 on the Lion+L1 base is required.
3. **slice_num sweep (32/64/128)**: Physics-aware attention bottleneck — current 64 may not be optimal for L1+Lion regime.
4. **n_layers depth sweep (4/5/6)**: Model depth hasn't been explored; 5 is the Transolver default but pressure modeling on complex tandem foil geometry may benefit from more depth.
5. **Larger batch with gradient accumulation**: bs=1 or bs=2 with accumulation to bs=4 — smaller batches may improve Lion convergence (sign gradients are more informative with less averaging).

### Medium priority
6. **Focal L1 / hard-example mining**: Weight loss by node-wise rolling error to chase the long pressure tail — especially valuable for camber_cruise split which still has large error.
7. **Multi-step learning rate with Lion**: Try [1e-4, 3e-4, 1e-4] step schedule rather than cosine — Lion paper shows step LR often outperforms cosine.
8. **bf16 at bs=4** (no batch increase): Pure throughput gain without batch-size penalty — previously tested at bs=8 which hurt; bs=4 with bf16 would give ~1.3x more epochs within timeout.
9. **Temperature scaling on attention slices**: Learnable temperature τ per head for Transolver's slice attention.
10. **Re-conditional normalization per sample**: Pass Re number through FiLM on the volume nodes only (not surface) — FiLM on surface hurt, but Re conditioning on the flow domain may help.

### Bigger swings (architecture/representation)
11. **Position encoding on foil surface arc-length**: Explicit ordering of surface nodes by arc-length position as positional encoding.
12. **Geometry-physics bottleneck**: Separate encoder for chord/camber/gap/stagger/Re → compact geometry token injected as cross-attention key/value.
13. **Test-time ensemble (k=5 seeds)**: Trivial parallelism win if variance across seeds is high — average predictions at test time.

## Known Issues

- **`data/scoring.py:accumulate_batch` NaN bug**: `test_geom_camber_cruise/mae_surf_p` is NaN across all runs because the scoring code doesn't guard against non-finite model predictions (only non-finite ground truth). Affects test metrics, not val metrics. Three students have independently flagged this. Needs organizer fix in `data/scoring.py`.
