# SENPAI Research State
- 2026-04-29 23:45 (icml-appendix-charlie-pai2f-r3)
- No recent research directives from the human researcher team
- Current best (merged): `val_avg/mae_surf_p = 33.1552` (PR #1258, charliepai2f3-nezuko, lr=1.5e-4 + 100ep + T_max=100 + warmup=5, best epoch=66/100 — still improving at wall-clock cutoff). test_avg/mae_surf_p = 28.1158.
- Previous best: `val_avg/mae_surf_p = 34.3851` (PR #1226, lr=3e-4 + T_max=100, 100ep, warmup=5)

## Current Research Focus and Themes

Round 3 of the charlie-pai2f series. Progress chain:
compound baseline (47.3987) → Fourier pos enc (44.4154) → extended freqs (43.9575) → FiLM conditioning (39.9450) → LR warmup + single-decay cosine (37.0739) → extended training 75ep T_max=75 (35.8406) → 100ep T_max=100 warmup=5 (34.3851) → **lr=1.5e-4 (33.1552)**

Two dominant themes have crystallized:

1. **Training horizon and schedule**: Every increase in T_max and epoch budget yielded improvement; the model has never plateaued. Best epoch always equals the last epoch reached in winning runs — wall-clock limited, not converged.
2. **Lower-magnitude Lion updates win**: Halving peak LR from 3e-4 → 1.5e-4 produced a clean −3.58% gain. The U-shape from PR #1209 lr-sweep (1e-4 too slow, 5e-4 too high) plus this win suggests the optimum lies below 2e-4. The sign-based Lion optimizer prefers smaller per-step magnitudes given EMA(0.995) and the bf16/L1 setup.

PR #1257 (T_max=200) was a clean negative: extending the LR horizon further without reducing peak LR causes oscillation. This pairs with the lr=1.5e-4 win to define the productive frontier: reduce peak rather than extend horizon when you want the LR-late-in-training to stay smaller.

The primary metric is `val_avg/mae_surf_p` (lower is better), averaged across 4 validation splits: single_in_dist, geom_camber_rc, geom_camber_cruise, re_rand. Current test_avg=28.1158.

## Consolidated Key Signals from All Experiments

- **Model NEVER converges at budget cap**: Best epoch = last epoch in every winning run. Extended training ALWAYS helps along the productive frontier.
- **T_max = epochs (single full-cycle decay) is optimal**: Confirmed up to T_max=100. T_max=200 regresses (PR #1257). T_max=150 still in flight (PR #1254).
- **Lion peak LR optimum < 2e-4**: PR #1258 lr=1.5e-4 wins. PR #1209 had a U-shape suggesting lr=2e-4 > lr=3e-4 (stale config). PR #1250 (lr=2e-4) and any further finer sweep is high priority.
- **Warmup critical for Lion**: 5-epoch linear warmup (start_factor=1/30) is essential for stable initialization.
- **FiLM global conditioning (STRONG, MERGED)**: Re/AoA/NACA regime conditioning via scale+shift per TransolverBlock.
- **Fourier pos enc (x,z) optimal at freqs=(1,2,4,8,16,32,64)**: Adding freq=128 regresses. Do not change.
- **SAF Fourier encoding NEGATIVE**: PR #1210. Do not revisit.
- **dsdf Fourier encoding NEGATIVE**: PR #1169. Do not revisit.
- **n_hidden width scaling NEGATIVE on early configs**: 128 < 192 < 256 under n_layers=1 without warmup+full schedule. n_hidden=192 + full schedule completed; result pending review.
- **Per-channel pressure weighting NEGATIVE**: W_p in {2,3,5} did not beat baseline. Do not revisit.
- **surf_weight=28 assumed optimal but being re-verified**: PR #1286 re-sweeping {14, 21, 28, 35, 42} to confirm optimality on current full config.
- **slice_num=64 assumed optimal but being re-verified**: PR #1285 sweeping {32, 64, 96, 128} on full current config.
- **n_layers=2 at iso-param failed (compute-budget)**: PR #1252 with full params; PR #1277 testing iso-compute swap (n_hidden=96).

## Active WIP Experiments

NOTE: All experiments below must beat current best baseline (**33.1552**, PR #1258) to be mergeable.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1269 | alphonse (other track) | mlp_ratio=4 + ema_decay=0.999 on previous best — wider MLP + slower EMA window | Running |
| #1272 | charliepai2f3-alphonse | SWA over last 10 epochs | Running |
| #1174 | tanjiro | Extended Fourier freqs on (x,z): sweep L in {5,6,7,8} octaves | Running |
| #1250 | frieren | lr=2e-4 on previous best config (T_max=100 + warmup=5 + 100ep) — pairs with PR #1258 lr=1.5e-4 win to triangulate optimum | Running |
| #1280 | nezuko | Lion lr=1e-4 on full current-best config (finer LR sweep below 1.5e-4) | Running |
| #1282 | fern | Lion weight_decay sweep {5e-3, 1e-2, 2e-2, 4e-2} at lr=1.5e-4 | Running |
| #1285 | askeladd | slice_num sweep {32, 64, 96, 128}: Transolver mesh partition granularity | Running |
| #1286 | edward | surf_weight re-sweep {14, 21, 28, 35, 42}: stale hyperparameter after FiLM+Fourier | Running |
| #1277 | thorfinn | Depth-vs-width iso-compute swap: n_layers=2 + n_hidden=96 (~252K params) | Running |

## Potential Next Research Directions

### Top Priority (directly motivated by PR #1258 win)

1. **Even-finer Lion LR around 1.5e-4** (e.g. lr=1.0e-4, lr=1.25e-4): The PR #1258 win is monotonic from PR #1209 lr=1e-4 (was bad on stale config without warmup). Rerunning lr=1.0e-4 on the FULL current best config (T_max=100 + warmup=5 + 100ep) may close the gap further or reveal a sharp optimum near 1.25–1.5e-4.

2. **Lower-LR + extended horizon combo**: T_max=150 with lr=1.5e-4 — keep low peak AND extend horizon. T_max=200 alone failed (PR #1257) because LR remained too high; pairing with 1.5e-4 might unlock both effects.

3. **Lion + slower EMA (ema_decay=0.999)**: Already in flight as PR #1269 but on stale config. Rerun on lr=1.5e-4 + T_max=100 + warmup=5 baseline.

4. **Weight decay sensitivity for Lion at lr=1.5e-4**: weight_decay=1e-2 was tuned for lr=3e-4. Halving the LR effectively halves the WD term in the Lion sign-update; sweep wd ∈ {5e-3, 1e-2, 2e-2, 4e-2}.

### Medium Priority

5. **Gradient accumulation + larger effective batch** at lr=1.5e-4: Effective batch=8 or 16 may smooth Lion updates further given lower per-step magnitude.

6. **Lookahead-Lion or RAdam-Lion hybrids**: Adaptive variants of Lion that may stabilize the wall-clock-limited regime.

7. **Test-time evaluation from EMA-averaged checkpoint**: Already doing EMA(0.995) — but explicit Polyak average over last K epochs (PR #1272 SWA test) is complementary.

8. **Per-split normalization** for OOD geom_camber_rc (worst split — 47.20 val mae_surf_p vs avg 33.16).

9. **Relative positional encoding within slices**: Arc-length based RPE inside PhysicsAttention slices.

### Longer-term (if current track plateaus)

10. **GNN backbone** to replace Transolver with explicit mesh connectivity
11. **Ensemble of diverse checkpoints** across seeds/hyperparameter variants
12. **Physics-informed regularization**: Continuity-equation soft constraints
13. **Curriculum learning**: Start on single-foil easy cases
14. **Data augmentation**: Geometric perturbations for OOD geom_camber gen
