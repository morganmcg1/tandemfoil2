# SENPAI Research State
- Last updated: 2026-04-30 00:10 (branch: icml-appendix-charlie-pai2f-r4)
- No human researcher team directives received yet.
- 8 students WIP; frieren reassigned to aux-surf-weight-calibration (PR #1344).

## Current r4 Baseline

| Metric | Value | PR |
|--------|-------|----|
| **val_avg/mae_surf_p** | **79.823** | #1271 (fern, SwiGLU FFN port, epoch 14/50) |

Per-split val breakdown (PR #1271):
- val_single_in_dist: 93.331
- val_geom_camber_rc: 92.371
- val_geom_camber_cruise: 58.837
- val_re_rand: 74.753

Test results (PR #1271):
- test_single_in_dist: 82.707
- test_geom_camber_rc: 79.046
- test_geom_camber_cruise: NaN (corrupt sample — 761 +Inf values in ground-truth pressure)
- test_re_rand: 68.332

**ARCHITECTURAL STATUS:** r4 codebase now has SwiGLU FFN (merged PR #1271). Still missing RFF positional encoding (tanjiro #1193 WIP). Combined RFF+SwiGLU expected to approach r1's 75.750.
- `Transolver.preprocess`: plain `MLP(fun_dim + space_dim, n_hidden*2, n_hidden, n_layers=0)` — **no RFF** (tanjiro #1193 covers this)
- `TransolverBlock.mlp`: SwiGLU (LLaMA-style gate/value/output, bias-free) — **merged PR #1271**
- Target for r4 WIP students: **< 79.823**
- To approach 75.750, RFF must still be ported to r4 (tanjiro's PR #1193)

## r4 Improvement Chain

1. PR #1201 (fern): CosineAnnealingLR T_max=15 + LR 1e-3 → **92.170**
2. PR #1243 (fern): n_hidden=192 + AMP bfloat16 → **88.421**
3. PR #1271 (fern): SwiGLU FFN (LLaMA-style 3-layer bias-free) → **79.823** (current r4 baseline)

The r1 branch history (75.750) is a separate track; PRs #1201, #1243, and #1271 are the only merged improvements on r4.

## Active Experiments (Round 5, r4)

| PR    | Student    | Status | Hypothesis |
|-------|------------|--------|-----------|
| #1332 | fern       | WIP    | surf_weight fine-sweep post-SwiGLU ({6,8,10,12,15,20}) — re-calibrate surface pressure weighting after SwiGLU architectural change |
| #1333 | edward     | WIP    | Divergence-free velocity penalty (lambda=0.01) — soft ∇·u≈0 physics constraint as aux loss; expected to help OOD re_rand split |
| #1193 | tanjiro    | WIP    | Random Fourier Features for multi-scale node positional encoding (n_rff=16, rff_scale=10.0) — **highest priority; patches last r4 architectural gap vs r1** |
| #1137 | nezuko     | WIP    | Scale Transolver to n_hidden=256, n_layers=8 for high-Re splits |
| #1117 | thorfinn   | WIP    | Re-conditioned output scale head for magnitude adaptation |
| #1111 | askeladd   | WIP    | Layer-wise LR decay for geometry-stable representations |
| #1110 | alphonse   | WIP    | Log-modulus transform on pressure channel loss |
| #1344 | frieren    | WIP    | Calibrated aux surface-pressure head: sweep aux_surf_weight {1,2,5} + 1/σ re-adaptive weighting on aux loss — fixes 73% gradient dominance from PR #1321 |

All students should target **val_avg/mae_surf_p < 79.823** (r4 true baseline after SwiGLU merge).

## Recently Merged (r4)

- PR #1271 (fern): SwiGLU FFN port → 79.823 (**current r4 baseline**, -9.7% vs 88.421)
- PR #1243 (fern): n_hidden=192 + AMP bfloat16 → 88.421 (-4.1% vs 92.170)
- PR #1201 (fern): CosineAnnealingLR T_max=15 + LR 1e-3 → 92.170
- PR #1187 (fern): Gradient clipping (max_norm=1.0) + LR 8e-4 → 102.080
- PR #1128 (edward): Per-sample Re-adaptive loss 1/σ → 124.727
- PR #1112 (edward): Attention dropout=0.1 → 129.531

## Recently Closed

- PR #1321 (frieren, aux surface-pressure head): CLOSED — 86.547 (+8.4% regression vs 79.823). Root cause: aux_surf_weight=20 caused aux MSE to constitute 73% of total gradient (vol=5.9%, surf=21.4%, aux=72.7%), overwhelming the main loss. No 1/σ weighting on aux loss. Fix: sweep {1.0, 2.0, 5.0} targeting ~10-15% aux gradient contribution + 1/σ weighting on aux (frieren #1344).
- PR #1312 (fern, CosineAnnealingWarmRestarts T_0=7): CLOSED — 80.255 (+0.54% worse than 79.823 baseline). Mechanism: restart spike at epoch 8 costs 3-4 recovery epochs within tight 14-epoch budget; net-negative robust across 3 seeds. SGDR warm restart direction closed for this budget.
- PR #1249 (edward, curvature-weighted surface loss — post-SwiGLU rerun): CLOSED — 80.953 (+1.42% worse than 79.823). Pre-SwiGLU result (86.867) beat old 88.421 bar, but post-SwiGLU the gated FFN (silu(W_gate)×W_value) already captures position-conditioned curvature weighting intrinsically. Curvature loss direction closed — superseded by SwiGLU architecture.
- PR #1299 (frieren, per-channel output heads): CLOSED — 81.324 (+1.88% regression vs 79.823 baseline). Root cause: targets already normalized so per-channel scale head adds no signal; +5% epoch overhead cost one training epoch. Channel-specific output heads direction closed.
- PR #1284 (frieren, slice_num=128 with AMP+n_hidden=192): CLOSED — 109.067 (+23.4% worse at epoch 11). Root cause: throughput artifact (+36% per-epoch time → 11 epochs instead of 15 in 30-min budget). Slice attention saturated at slice_num=64. VRAM actual: +13.3 GB (57.3 GB total). Slice granularity direction conclusively closed.
- PR #1114 (frieren, curriculum+sweep surf_weight): CLOSED — sw=4 (92.544) lost to sw=10 control (89.734). surf_weight=10 confirmed as default for current and future recipes until RFF+SwiGLU land.
- PR #1230 (fern, gradient accumulation bs=4 + lr=1e-3): CLOSED — 101.013 (+33.4% worse). Batch-size direction conclusively closed.
- PR #1213 (fern, batch_size=8 + linear LR 2e-3): CLOSED — 118.098 (+56% worse). Linear LR scaling overshoots small-batch regime.
- PR #1186 (edward, sw5+per-sample loss rerun): CLOSED — 87.924/90.97 vs old baseline → regression on r4 recipe.
- PR #1235 (thorfinn-r1, deeper FiLM 3-layer residual): CLOSED — slower convergence at budget, OOD splits regress.

## Key Findings

1. **SwiGLU FFN** (PR #1271, r4): 88.421→79.823 (-9.7%); 1.84M params (+25%); VRAM 51.44 GB of 96 GB; ~135s/epoch; 14 epochs in 30-min budget
2. **n_hidden=192 + AMP bfloat16** (PR #1243, r4): 92.170→88.421 (-4.1%); VRAM 44.0 GB of 96 GB; ~124s/epoch; 1.47M params
3. **AMP bfloat16 + n_hidden=160** (PR #1197, r1 track): best known absolute result 75.750; NOT r4 codebase
4. **CosineAnnealingLR T_max=15** matched to budget: 102.080→92.170; T_max mismatch was critical
5. **Gradient clipping + LR 8e-4**: 124.727→102.080 (18.2% improvement)
6. **Per-sample Re-adaptive loss** (1/σ weighting): 129.531→124.727
7. **Attention dropout=0.1**: established 129.531 at round 4 start
8. **surf_weight=10 confirmed with current recipe**: full sweep favored sw=10 after SwiGLU merge
9. **VRAM headroom**: peak 51.44 GB of 96 GB — ~44 GB unused; room for larger n_hidden or n_layers
10. **Training budget**: ~14 epochs of 50 configured with SwiGLU+AMP; convergence speed and throughput are primary levers
11. **Batch-size direction conclusively closed**: bs=8 + linear LR scaling (#1213, +56%) and bs=4 grad-accum effective bs=8 (#1230, +33%) both regress
12. **Architectural gap with r1**: r4 still missing RFF positional encoding — tanjiro's #1193 covers this; SwiGLU already merged in #1271
13. **Aux head weight calibration critical**: aux_surf_weight=20 causes 73% gradient dominance and regression; targeting ~10-15% with weight in {1,2,5} (frieren #1344)

## Key Dataset Observations

- Cruise split (58.837 val) dramatically easier than raceCar splits (~74-93): multi-domain difficulty imbalance
- Per-sample pressure std varies by order of magnitude within a split (high-Re drives extremes)
- VRAM: peaked 51.44 GB of 96 GB available — substantial headroom for scaling
- Timeout: ~30 min wall-clock → ~14 epochs with SwiGLU+AMP bfloat16; LR schedule / convergence speed is major lever
- test_geom_camber_cruise/000020.pt has 761 +Inf values in ground-truth pressure (scoring returns NaN for this split without +Inf masking)

## Current Research Focus

**Target:** TandemFoilSet CFD surrogate — predict (Ux, Uy, p) at every mesh node.
**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across 4 val splits (lower is better).
**Model:** Transolver with SwiGLU FFN and physics-aware attention over irregular meshes.
**Round 5/5. Best r4 result: PR #1271 (fern) at val_avg/mae_surf_p=79.823.**

Key strategic priorities:
1. **Port RFF positional encoding to r4** (tanjiro #1193 WIP — highest priority; directly patches remaining architectural gap)
2. **surf_weight re-calibration post-SwiGLU** (fern #1332 — sweep {6,8,10,12,15,20} to confirm or update default sw=10)
3. **Physics-informed regularization** (edward #1333 — divergence-free ∇·u≈0 penalty λ=0.01; expected OOD gain on re_rand split)
4. **Calibrated aux surface-pressure head** (frieren #1344 — aux_surf_weight sweep {1,2,5} + 1/σ weighting; fixes 73% gradient dominance from PR #1321)
5. **Architecture scaling on capacity axis** (nezuko #1137 — n_hidden=256/n_layers=8)
6. **LR scheduling improvements** (askeladd #1111 — layer-wise LR decay)
7. **Loss formulation** (alphonse #1110 — log-modulus pressure transform)
8. **Output head specialization** (thorfinn #1117 — re-conditioned output scale head)

## Potential Next Research Directions (Post Round 5)

1. **RFF + SwiGLU combined**: After #1193 lands, test RFF+SwiGLU together on r4 to see if they reproduce the r1 75.750 result
2. ~~**Larger slice_num with AMP**~~ CLOSED — PR #1284 confirms slice_num=128 is a throughput artifact; slice_num=64 is saturated for this dataset scale
3. **surf_weight re-sweep on the unified r4+RFF+SwiGLU recipe**: Once RFF lands, the optimum may shift — a tight {6, 8, 10, 12, 16} sweep would be the right shape
4. **Divergence-free penalty**: Approximate ∇·u=0 constraint penalty (lambda=0.01) — physics-informed regularization
5. **Multi-seed bracketing** for the top-3 contenders post-RFF — to firm up the magnitude estimates of small wins
6. **Test split NaN-guard fix** (evaluate_split predictions-side check for non-finite preds — flagged 4× by frieren, currently only ground-truth side is guarded)
7. **n_hidden=256 + SwiGLU** — now that SwiGLU is merged, capacity scaling combines with architectural improvement; VRAM headroom supports it
8. **Aux head with intermediate-block branching**: Rather than branching from the final block (which creates competing gradient pressure on the specialized mlp2), try branching from block 3 or 4 hidden state — gives the head more representational freedom
