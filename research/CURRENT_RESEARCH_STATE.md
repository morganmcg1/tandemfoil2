# SENPAI Research State
- 2026-04-29 06:00 — PR #1013 merged (tanjiro, n_layers=1+bf16 — landmark −24.3% win, 63.06→47.74). New best baseline established. All in-flight experiments must rebase. Branch icml-appendix-charlie-pai2e-r5.
- Most recent research direction from human researcher team: None received yet.

## Current Best Baseline

**PR #1013** — `val_avg/mae_surf_p` = **47.7385** (epoch 50, n_layers=1+bf16)
Branch: `charliepai2e5-tanjiro/n-layers-2-bf16-depth-floor` (merged 2026-04-29 04:42)

Configuration: n_hidden=128, **n_layers=1**, n_head=4, slice_num=64, mlp_ratio=2, **Lion** optimizer, lr=3e-4, wd=1e-2, **surf_weight=28**, batch_size=4, **CosineAnnealingLR T_max=15**, **L1 loss** (vol + surf_weight * surf), grad_clip=1.0, **EMA decay=0.995**, **bf16=True** (~50 epochs in budget — T_max=15 cycles 3× at n_layers=1 speed).

Per-split surf p MAE: single=49.6805, camber_rc=60.8209, camber_cruise=30.5543, re_rand=49.8983.

Baseline improvement trajectory: 128.83 (MSE+AdamW) → 97.45 (L1+AdamW, PR #798, −24.4%) → 77.30 (L1+Lion+clip, PR #799, −20.7%) → 71.29 (Cosine T_max=15 aligned, PR #901, −7.78%) → 70.32 (EMA decay=0.995, PR #801, −1.36%) → 67.25 (surf_weight=28, PR #926, −4.35%) → 63.06 (n_layers=3+bf16, PR #913, −6.22%) → **47.74 (n_layers=1+bf16, PR #1013, −24.3%)**

**All current experiments must rebase on: n_layers=1 + bf16 + Lion + L1 + clip + T_max=15 + EMA=0.995 + sw=28.**
**URGENT: T_max=15 now cycles 3× over ~50 epochs — misalignment. T_max=50 is the correct target for n_layers=1 budget.**
**VRAM: n_layers=1 uses only ~9 GB peak — massive headroom for width expansion (n_hidden 128→256/512).**

## Completed Experiments (this round)

### Merged (wins)
| PR | Hypothesis | Result | Δ |
|----|------------|--------|---|
| #798 | L1 loss (align objective with metric) | 97.45 | −24.4% vs 128.83 |
| #799 | Lion optimizer + L1 + clip=1.0 | 77.30 | −20.7% vs 97.45 |
| #901 | Cosine LR T_max budget alignment (T_max 50→15) | 71.29 | −7.78% vs 77.30 |
| #801 | EMA model averaging (decay=0.995) | 70.32 | −1.36% vs 71.29 |
| #926 | surf_weight=28 (sweep 20/25/28 on Lion+T_max=15+EMA) | 67.25 | −4.35% vs 70.32 |
| #913 | n_layers=3 + bf16 autocast (depth reduction + precision) | 63.06 | −6.22% vs 67.25 |
| **#1013** | **n_layers=1 + bf16 depth floor (also tested n_layers=2: 47.97)** | **47.74** | **−24.3% vs 63.06** |

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
| #977 | Lion LR warmup (all variants) | Regressed | Warmup shrinks cosine budget; closed. |

## Currently Running (status:wip)

| PR | Student | Hypothesis | Notes |
|----|---------|------------|-------|
| #1014 | charliepai2e5-askeladd | T_max re-tune for bf16 budget: sweep T_max 15/20/25/29 | **HIGHEST PRIORITY** — needs rebase; n_layers=1 runs ~50 epochs so T_max range should be extended to at least 50; T_max=50 is likely optimal |
| #1043 | charliepai2e5-frieren | Lion lr upper sweep: test lr=4e-4 and 5e-4 on current best config | Needs rebase on new 47.74 baseline (n_layers=1) |
| #1035 | charliepai2e5-alphonse | Focal L1: node-wise error EMA weighting for hard-example mining | Needs rebase on new 47.74 baseline; camber_rc is still worst split (60.82) |
| #894 | charliepai2e5-nezuko | Lion+L1 surf_weight re-tune: sweep 5/10/30/40 vs baseline | Needs rebase on new 47.74 baseline |
| #908 | charliepai2e5-thorfinn | slice_num sweep 32/64/128: physics attention bottleneck tuning | Needs rebase on new 47.74 baseline; at n_layers=1, slice_num may have more impact |
| #918 | charliepai2e5-fern | Lion weight_decay sweep: 1e-3/5e-3/5e-2 vs baseline 1e-2 | Needs rebase on new 47.74 baseline |
| #945 | charliepai2e5-edward | EMA higher decay sweep (0.999/0.9985) with bias-correction warmup | Student fixing math bug in EMA init; re-run with zero-init pending; note: shallow 1-layer model may actually benefit from *lower* EMA decay (0.99/0.98) |
| #893 | charliepai2e5-frieren | Lion lr lower sweep: 5e-5/1e-4/1.5e-4 | Needs rebase on new 47.74 baseline |

## Idle Students Needing Assignment

None — all 8 students assigned. (After all WIP PRs rebase on new 47.74 baseline, new experiments to assign:)
- **Width expansion**: n_hidden 128→256/512 at n_layers=1 (~9 GB VRAM peak, massive headroom)
- **T_max=50 re-tune** (follow-up to #1014 depending on results)
- **EMA lower decay** (0.99/0.98/0.95) for shallow 1-layer model

## Current Research Focus

**LANDMARK WIN**: n_layers=1 + bf16 achieves 47.74 (−24.3% vs 63.06). The depth-shallowing trend is monotonic and dramatic:
- n_layers=6: ~70.x → n_layers=5: 67.25 → n_layers=4: 65.37 → n_layers=3: 63.06 → n_layers=2: 47.97 → **n_layers=1: 47.74**

The jump from n=3→n=2 is the largest single gain (−24.3%). This strongly suggests TandemFoilSet is severely over-parameterized at depth — the Transolver with a single transformer layer is the right inductive bias for this small dataset.

**Immediate priorities:**

1. **T_max alignment (URGENT — #1014 askeladd)**: T_max=15 calibrated for 14 fp32 epochs; n_layers=1 runs ~50 epochs, so it cycles 3×. This is the biggest unresolved misalignment. T_max should be 50 for single-cycle alignment. Rebase #1014 and extend sweep to T_max 25/29/40/50.

2. **Width expansion (new — unassigned)**: With only 9 GB VRAM at n_layers=1 vs 96 GB available, there is enormous headroom. n_hidden 128→256 or 512 could recover expressiveness lost from depth reduction. This is the next architectural lever after exhausting depth.

3. **EMA re-tune for shallow model**: EMA decay=0.995 was tuned for a 3-layer model. With 50 epochs and a single layer, lower decay (0.99/0.98/0.95) may be better — faster adaptation to recent gradients. Also need to verify zero-init fix in #945.

4. **camber_rc still hardest split** (60.82 surf p MAE): −22% improved vs old 77.69 but still worst. Focal L1 (#1035 alphonse) specifically targets this.

5. **All in-flight experiments must rebase** on n_layers=1 baseline before their results are interpretable.

## Potential Next Research Directions

### High priority (not yet assigned)
1. **Width expansion (n_hidden 128→256/512)**: Critical next experiment. Only 9 GB VRAM at n_layers=1 — can grow hidden dim 2-4× with no memory concern. The model may need width to compensate for depth reduction.
2. **T_max=50 single-cycle alignment** (follow-up to #1014): After confirming T_max=50 is optimal, explore even longer T_max (70–100) for possible training extension.
3. **n_head scaling**: With n_hidden potentially doubling, n_head=4 may be a bottleneck. Test n_head=8 alongside width expansion.

### Medium priority
4. **Temperature scaling on attention slices**: Learnable temperature τ per head for Transolver's slice attention (already has a temperature parameter — probe initialization and learning rate). At n_layers=1, this is the only transformer block's temperature.
5. **Re-conditional normalization (volume only)**: Pass Re number through FiLM on volume nodes only — FiLM on all surface nodes hurt (#806), but Re conditioning only on flow domain may help.
6. **Position encoding on foil surface arc-length**: Explicit ordering of surface nodes by arc-length position as positional encoding.
7. **Test-time ensemble (k=5 seeds)**: Trivial parallelism win if variance across seeds is high — average predictions at test time.
8. **Geometry-physics bottleneck token**: Compact geometry token (chord/camber/gap/stagger/Re → learnable embedding) injected via cross-attention key/value into Transolver blocks. At n_layers=1, this is the single block's key/value.

### Bigger swings
9. **Separate surface/volume encoders**: Dedicated MLP towers for surface nodes vs. volume nodes before the Transolver blocks — respects the physical distinction in the data. With n_layers=1, the preprocess MLP is the dominant encoder.
10. **Deeper preprocess MLP**: 2-layer preprocess with skip connection (previous attempt failed at 1 residual layer with wrong architecture — a clean 3-layer MLP may work; at n_layers=1 the preprocess MLP may matter more).
11. **n_layers=0 (MLP-only)**: With n_layers=1 winning, test pure MLP encoder+decoder with no attention. May not generalize but worth understanding the floor.

## Known Issues

- **`data/scoring.py:accumulate_batch` NaN bug**: `test_geom_camber_cruise/mae_surf_p` is NaN across all runs because the scoring code doesn't guard against non-finite model predictions (only non-finite ground truth). Affects test metrics, not val metrics. Three students have independently flagged this. Needs organizer fix in `data/scoring.py`.
