# SENPAI Research State
- 2026-04-28 (last check: 2026-04-28, all 8 students active, all pods READY 1/1); branch icml-appendix-charlie-pai2e-r5
- Most recent research direction from human researcher team: None received yet.

## Current Best Baseline

**PR #801** — `val_avg/mae_surf_p` = **70.3212** (epoch 14, best checkpoint, −1.36% vs previous best 71.2882)
Branch: `charliepai2e5-edward/ema-decay-0.995` (merged)

Configuration: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **Lion** optimizer, lr=3e-4, wd=1e-2, surf_weight=20, batch_size=4, **CosineAnnealingLR T_max=15**, **L1 loss** (vol + surf_weight * surf), grad_clip=1.0, **EMA decay=0.995**.

Per-split surf p MAE: single=74.06, camber_rc=87.18, camber_cruise=51.83, re_rand=68.22.

Baseline improvement trajectory: 128.83 (MSE+AdamW) → 97.45 (L1+AdamW, PR #798, −24.4%) → 77.30 (L1+Lion+clip, PR #799, −20.7%) → 71.29 (Cosine T_max=15, PR #901, −7.78%) → **70.32 (EMA decay=0.995, PR #801, −1.36%)**

**All current experiments must rebase on the Lion+L1+clip+T_max=15+EMA(0.995) recipe.**

**Weakest split: `val_geom_camber_rc` (surf p = 87.18).** This split represents the hardest OOD geometry and is the main drag on the average. Follow-on experiments that reduce camber_rc error faster than other splits will compound gains well.

## Completed Experiments (merged wins)

| PR | Hypothesis | Result | Δ |
|----|------------|--------|---|
| #798 | L1 loss (align objective with metric) | 97.45 | −24.4% vs 128.83 |
| #799 | Lion optimizer + L1 + clip=1.0 | 77.30 | −20.7% vs 97.45 |
| #901 | Cosine LR T_max budget alignment (T_max 50→15) | 71.29 | −7.78% vs 77.30 |
| #801 | EMA model averaging (decay=0.995) | 70.32 | −1.36% vs 71.29 |

## Closed (dead ends)

| PR | Hypothesis | Result | Reason |
|----|------------|--------|--------|
| #803 | Surface feature noise | 142.25 | +46% regression |
| #804 | Cosine LR warmup (3-epoch) | 128.23 | +32% vs current baseline |
| #805 | Preprocess MLP depth +1 residual layer | 138.60 | +42% regression |
| #806 | FiLM domain conditioning | 106.49 | +9.3% regression |
| #802 | bf16 autocast + TF32 + batch_size=8 | 129.14 | +32.5% regression |
| #822 | SmoothL1/Huber loss (beta sweep) | 103.00 | +5.7% regression |
| #824 | Gradient clipping at 0.5/1.0/5.0 | 101.23 | +3.9% regression |
| #823 | asinh pressure target transform | 99.26 | +28.4% vs current 71.29 |
| #817 | surf_weight sweep L1 AdamW (10/15/20/25/30) | 95.56 at sw=25 | Predated T_max=15 baseline; superseded by #926 |
| #852 | Per-channel L1 loss weighting | — | Superseded/redirected |
| #879 | Wider hidden dim n_hidden 256 (AdamW) | 121.34 | Wrong optimizer |
| #857 | Drop-path stochastic depth | — | Closed |

## Currently Running (status:wip)

| PR | Student | Hypothesis |
|----|---------|------------|
| #893 | charliepai2e5-frieren | Lion lr sweep: test lr=1e-4, 5e-4, 6e-4 vs baseline 3e-4 |
| #894 | charliepai2e5-nezuko | Lion+L1 surf_weight re-tune: sweep 5/10/30/40 vs baseline 20 (T_max=MAX_EPOCHS, dual-baseline comparison) |
| #908 | charliepai2e5-thorfinn | slice_num sweep 32/64/128: physics attention bottleneck tuning |
| #913 | charliepai2e5-tanjiro | n_layers depth sweep (4/5/6) on Lion+L1+clip |
| #918 | charliepai2e5-fern | Lion weight_decay sweep: 1e-3/5e-3/5e-2 vs baseline 1e-2 |
| #926 | charliepai2e5-alphonse | surf_weight=25 validation on Lion+T_max=15 baseline (sw sweep 20/25/28) |
| #941 | charliepai2e5-askeladd | bf16 autocast at bs=4 for throughput gain (more epochs within timeout) |
| #945 | charliepai2e5-edward | EMA higher decay (0.999/0.9985) with bias-correction warmup |

## Idle Students Needing Assignment

(none — all 8 students have active WIP PRs)

## Current Research Focus

With the **Lion + L1 + clip + T_max=15 + EMA(0.995)** combination established as the new baseline (70.32), focus areas are:

1. **LR tuning for Lion** (in-flight #893): 3e-4 may not be the global optimum with the current stack
2. **surf_weight re-tuning** (in-flight #894, #926): Previous best was sw=25 under old schedule; validate with T_max=15
3. **Architecture: slice_num and depth** (in-flight #908, #913): Physics attention bottleneck and model capacity
4. **Lion weight decay** (in-flight #918): wd=1e-2 inherited from AdamW regime; Lion may prefer different values
5. **bf16 throughput** (in-flight #941): More epochs = more data seen = lower val error
6. **EMA higher decay variants** (suggested by edward): 0.999/0.9985 with bias-correction warmup — student observed consistent per-split gains from 0.995

## Potential Next Research Directions

### High priority (freshest ideas)
1. **EMA higher decay (0.999 / 0.9985) with bias-correction warmup**: edward's data showed EMA 0.995 improved all splits except camber_rc. A higher decay (longer averaging window) may smooth out the camber_rc variance further. Bias-correction warmup avoids cold-start bias in early EMA weights.
2. **Focal L1 / hard-node weighting**: camber_rc surf_p = 87.18 is the main drag on the average. Weight loss contributions by node-by-node running error (e.g., multiply each node's L1 by a self-calibrated weight that decays as the model improves on that node). Targets the long pressure tail.
3. **Multi-sample ensemble at inference**: Train 3–5 seeds of the baseline, average predictions. Trivial variance reduction. If inter-seed variance is large this is a free win at test time.

### Medium priority
4. **Geometry-physics bottleneck token**: Compact geometry token (chord/camber/gap/stagger/Re → learnable embedding) injected via cross-attention key/value into Transolver blocks — gives the model a global geometry summary without relying on local node features to carry that signal.
5. **Re-conditional normalization on volume nodes only**: FiLM on all surface nodes hurt (#806), but pass Re number through FiLM conditioning on volume nodes only — flow domain conditioning without polluting the surface predictions.
6. **Position encoding on foil surface arc-length**: Explicit ordering of surface nodes by arc-length position as positional encoding — helps the model reason about leading-edge vs. trailing-edge pressure distribution.
7. **Separate surface/volume encoders**: Dedicated MLP towers for surface nodes vs. volume nodes before the Transolver blocks — respects the physical distinction.

### Bigger swings
8. **Hidden dim sweep: n_hidden=192 or 256 with Lion+T_max=15**: Previous n_hidden=256 test used AdamW and wrong recipe. With Lion+T_max=15+EMA the wider model may not suffer the same convergence failure.
9. **n_head sweep (4→8)**: More attention heads in Transolver could improve multi-scale feature extraction.
10. **Warmup cosine annealing (1-epoch linear warmup + cosine)**: The current T_max=15 schedule drops LR fast; a short linear warmup may stabilize early optimization with Lion.

## Known Issues

- **`data/scoring.py:accumulate_batch` NaN bug**: `test_geom_camber_cruise/mae_surf_p` is NaN across all runs because the scoring code doesn't guard against non-finite model predictions. Affects test metrics only, not val metrics. Four students have independently flagged this. Needs organizer fix in `data/scoring.py`.
