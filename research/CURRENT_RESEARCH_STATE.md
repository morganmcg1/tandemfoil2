# SENPAI Research State
- 2026-04-29 05:15 — All 8 students active; PR #977 closed (alphonse, Lion LR warmup — all variants regress, warmup shrinks cosine budget); alphonse reassigned to PR #1035 (Focal L1 hard-example mining, targeting camber_rc OOD failure). Branch icml-appendix-charlie-pai2e-r5.
- Most recent research direction from human researcher team: None received yet.

## Current Best Baseline

**PR #913** — `val_avg/mae_surf_p` = **63.0588** (epoch 29, n_layers=3+bf16)
Branch: `charliepai2e5-tanjiro/nlayers3-bf16` (merged 2026-04-29 04:00)

Configuration: n_hidden=128, **n_layers=3**, n_head=4, slice_num=64, mlp_ratio=2, **Lion** optimizer, lr=3e-4, wd=1e-2, **surf_weight=28**, batch_size=4, **CosineAnnealingLR T_max=15**, **L1 loss** (vol + surf_weight * surf), grad_clip=1.0, **EMA decay=0.995**, **bf16=True** (~29 epochs in budget vs 14 in fp32).

Per-split surf p MAE: single=69.5556, camber_rc=77.6902, camber_cruise=41.7779, re_rand=63.2113.

Baseline improvement trajectory: 128.83 (MSE+AdamW) → 97.45 (L1+AdamW, PR #798, −24.4%) → 77.30 (L1+Lion+clip, PR #799, −20.7%) → 71.29 (Cosine T_max=15 aligned, PR #901, −7.78%) → 70.32 (EMA decay=0.995, PR #801, −1.36%) → 67.25 (surf_weight=28, PR #926, −4.35%) → **63.06 (n_layers=3+bf16, PR #913, −6.22%)**

**All current experiments must rebase on: n_layers=3 + bf16 + Lion + L1 + clip + T_max=15 + EMA=0.995 + sw=28.**

## Completed Experiments (this round)

### Merged (wins)
| PR | Hypothesis | Result | Δ |
|----|------------|--------|---|
| #798 | L1 loss (align objective with metric) | 97.45 | −24.4% vs 128.83 |
| #799 | Lion optimizer + L1 + clip=1.0 | 77.30 | −20.7% vs 97.45 |
| #901 | Cosine LR T_max budget alignment (T_max 50→15) | 71.29 | −7.78% vs 77.30 |
| #801 | EMA model averaging (decay=0.995) | 70.32 | −1.36% vs 71.29 |
| #926 | surf_weight=28 (sweep 20/25/28 on Lion+T_max=15+EMA) | 67.25 | −4.35% vs 70.32 |
| **#913** | **n_layers=3 + bf16 autocast (depth reduction + precision)** | **63.06** | **−6.22% vs 67.25** |

### Closed (dead ends / superseded)
| PR | Hypothesis | Result | Reason |
|----|------------|--------|--------|
| #803 | Surface feature noise | 142.25 | +46% regression |
| #804 | Cosine LR warmup (3-epoch) | 128.23 | +32% vs current baseline |
| #805 | Preprocess MLP depth +1 residual layer | 138.60 | +42% regression |
| #806 | FiLM domain conditioning | 106.49 | +9.3% regression |
| #802 | bf16 autocast + TF32 + batch_size=8 | 129.14 | +32.5% regression |
| #822 | SmoothL1/Huber loss (beta sweep) | 103.00 | +5.7% regression |
| #824 | Gradient clipping at 0.5/1.0/5.0 | 101.23 | +3.9% regression (wrong regime — natural grad norm 85–115, not 1–10) |
| #823 | asinh pressure target transform (scale 100/500/2000) | 99.26 | +28.4% vs current 71.29 — symmetric squash mismatched to asymmetric pressure tail |
| #817 | surf_weight sweep L1 AdamW (10/15/20/25/30) | 95.56 at sw=25 | Doesn't beat 71.29 baseline (run predated PR #901); directional finding: optimum shifts to sw=25 under L1. Superseded by #926 |
| #852 | Per-channel L1 loss weighting: amplify pressure (p_weight 2/5/10) | 97.81 (p_weight=2) | +26.5% vs 71.30 baseline; channel-level reweighting distorts loss landscape, OOD splits regress hardest |
| #857 | Drop-path stochastic depth (askeladd, cross-cohort) | — | Closed stale — assigned against old 97.45 baseline with AdamW+MSE. Will be re-assigned when relevant. |
| #922 | Multi-step LR schedule for Lion (milestones=[7,11], gamma=0.3) | 71.5764 | +0.40% vs baseline — 50%/80% milestone rule calibrated for long schedules; at 14-epoch budget, first drop at epoch 7 too late; cosine handles short budget better. camber_rc regressed +4.27. |
| #941 | bf16 autocast at bs=4 for ~1.3× throughput (standalone) | 69.78 | Superseded by PR #913 which incorporated bf16 as part of n_layers=3 experiment yielding 63.06 |

## Currently Running (status:wip)

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #1014 | charliepai2e5-askeladd | T_max re-tune for bf16 budget: sweep T_max 15/20/25/29 | Freshly assigned 2026-04-29 04:30; T_max=15 calibrated for fp32 14-epoch budget but bf16 now gives ~29 epochs |
| #1013 | charliepai2e5-tanjiro | n_layers=2 + bf16 depth floor | In-flight since PR #913 merged |
| #1035 | charliepai2e5-alphonse | Focal L1: node-wise error EMA weighting for hard-example mining | Freshly assigned 2026-04-29 05:15; targets camber_rc OOD split (77.69); focal_decay sweep 0.9/0.99 |
| #893 | charliepai2e5-frieren | Lion lr rebase: rerun lr=5e-5/1e-4/1.5e-4 on current best config (sw=28, T_max=15, EMA=0.995) | Sent back 2026-04-29 02:23; needs rebase on new 63.06 baseline |
| #894 | charliepai2e5-nezuko | Lion+L1 surf_weight re-tune: sweep 5/10/30/40 vs baseline | Needs rebase on new 63.06 baseline |
| #908 | charliepai2e5-thorfinn | slice_num sweep 32/64/128: physics attention bottleneck tuning | Needs rebase on new 63.06 baseline |
| #918 | charliepai2e5-fern | Lion weight_decay sweep: 1e-3/5e-3/5e-2 vs baseline 1e-2 | Needs rebase on new 63.06 baseline |
| #945 | charliepai2e5-edward | EMA higher decay sweep (0.999/0.9985) with bias-correction warmup | Student fixing math bug in EMA init; re-run with zero-init pending |

## Idle Students Needing Assignment

None — all 8 students assigned.

## Current Research Focus

With **n_layers=3 + bf16 + Lion + L1 + clip + T_max=15 + EMA=0.995 + surf_weight=28** established as the new best (63.06, PR #913), the remaining levers being investigated are:

1. **LR schedule alignment**: T_max re-tune for bf16 ~29-epoch budget (in-flight #1014 — T_max 15/20/25/29 sweep); LR warmup (in-flight #977); LR magnitude sweep (in-flight #893, needs rebase)
2. **Architecture depth floor**: n_layers=2 + bf16 depth floor (in-flight #1013); slice_num sweep (in-flight #908, needs rebase)
3. **Regularization**: weight_decay sweep (in-flight #918, needs rebase); surf_weight re-tuning (in-flight #894, needs rebase)
4. **EMA decay tuning**: Higher decay (0.999/0.9985) with zero-init bias-correction (in-flight #945, bug-fix in progress)
5. **The camber_rc split** (77.69 surf p MAE at new baseline) is consistently the worst — geometry OOD. Focus needed here.

Key observations so far: n_layers=3 outperforms n_layers=5 — the Transolver with fewer layers may avoid overfitting with the small TandemFoilSet dataset. Combined with bf16 throughput gain, this gave −6.22% improvement. The T_max misalignment (calibrated for 14 fp32 epochs, now running 29 bf16 epochs) is an urgent follow-up.

## Potential Next Research Directions

### High priority (not yet assigned)
1. **T_max re-tune follow-up**: Results from PR #1014 will inform whether an even longer T_max (e.g., 35–40 for possible future epoch count increases) is worth exploring.
2. **Focal L1 / hard-example mining**: Weight loss by node-wise rolling error to chase the long pressure tail — camber_rc split (77.69) is the worst and likely drives the average. **IN-FLIGHT PR #1035 (alphonse)**.
3. **Geometry-physics bottleneck token**: Compact geometry token (chord/camber/gap/stagger/Re → learnable embedding) injected via cross-attention key/value into Transolver blocks.

### Medium priority
3. **Temperature scaling on attention slices**: Learnable temperature τ per head for Transolver's slice attention (already has a temperature parameter — probe initialization and learning rate).
4. **Re-conditional normalization (volume only)**: Pass Re number through FiLM on volume nodes only — FiLM on all surface nodes hurt (#806), but Re conditioning only on flow domain may help.
5. **Position encoding on foil surface arc-length**: Explicit ordering of surface nodes by arc-length position as positional encoding.
6. **Test-time ensemble (k=5 seeds)**: Trivial parallelism win if variance across seeds is high — average predictions at test time.
7. **Tighter multi-step LR milestones [4,9] or 3-step [5,9,12]**: Student suggestion from #922 — with the 50/80 rule ruled out, tighter early milestones may help. Low priority given cosine is already well-tuned.

### Bigger swings (architecture/representation)
8. **Deeper preprocess MLP**: 2-layer preprocess with skip connection (previous attempt failed at 1 residual layer with wrong architecture — a clean 3-layer MLP may work).
9. **Separate surface/volume encoders**: Dedicated MLP towers for surface nodes vs. volume nodes before the Transolver blocks — respects the physical distinction in the data.

## Known Issues

- **`data/scoring.py:accumulate_batch` NaN bug**: `test_geom_camber_cruise/mae_surf_p` is NaN across all runs because the scoring code doesn't guard against non-finite model predictions (only non-finite ground truth). Affects test metrics, not val metrics. Three students have independently flagged this. Needs organizer fix in `data/scoring.py`.
