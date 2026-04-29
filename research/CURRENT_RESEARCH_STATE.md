# SENPAI Research State
- 2026-04-29 20:45 (icml-appendix-charlie-pai2f-r3)
- No recent research directives from the human researcher team
- Current best (merged): `val_avg/mae_surf_p = 34.3851` (PR #1226, charliepai2f3-frieren, 100ep + T_max=100 + warmup=5, best epoch=66/100 — NOT converged, wall-clock timeout)
- Previous best: `val_avg/mae_surf_p = 35.8406` (PR #1208, T_max=75, 75ep, no warmup)

## Current Research Focus and Themes

Round 3 of the charlie-pai2f series. Progress chain:
compound baseline (47.3987) → Fourier pos enc (44.4154) → extended freqs (43.9575) → FiLM conditioning (39.9450) → LR warmup + single-decay cosine (37.0739) → extended training 75ep T_max=75 (35.8406) → extended training 100ep T_max=100 warmup=5 (34.3851)

The dominant theme is **training horizon and schedule**: every increase in T_max and epoch budget has yielded improvement, and the model has never plateaued — best epoch always equals the last epoch reached. The model is consistently training-limited by wall-clock timeout, not by convergence. The natural next step is to push beyond 100ep — testing T_max=150 or T_max=200 — though we are approaching physical limits on the 30-min wall-clock budget.

Key mechanism update: The model was cut at ep66/100 under T_max=100. This means the full 100-epoch decay horizon was never realized. T_max=150 or 200 would keep LR even higher at ep66 cutoff, potentially yielding further improvement.

Additionally, PR #1209 (Lion LR sweep) identified lr=2e-4 as marginally better than lr=3e-4 on an equivalent (stale) config — this signal needs validation on the current best config (100ep + T_max=100 + warmup=5).

The primary metric is `val_avg/mae_surf_p` (lower is better), averaged across 4 validation splits: single_in_dist, geom_camber_rc, geom_camber_cruise, re_rand. Current test_avg=29.0050.

## Consolidated Key Signals from All Experiments

- **Model NEVER converges at budget cap**: Best epoch = last epoch in every run. Extended training ALWAYS helps. Top priority: push horizon further.
- **T_max = epochs (single full-cycle decay) is optimal**: LR slow decay over the full training horizon >> multi-cycle cosine. Do not use restarts.
- **Warmup critical for Lion**: 5-epoch linear warmup (start_factor=1/30) is essential for stable initialization.
- **FiLM global conditioning (STRONG, MERGED)**: PR #1104 −9.13% improvement. Re/AoA/NACA regime conditioning via scale+shift per TransolverBlock.
- **Fourier pos enc (x,z) optimal at freqs=(1,2,4,8,16,32,64)**: PR #1106, PR #1148. Adding freq=128 regresses. Do not change this.
- **lr=2e-4 vs 3e-4 (weak signal, stale config)**: PR #1209 — lr=2e-4 beat 3e-4 by 1.33% on old T_max=50 config. Needs revalidation on current best config (T_max=100 + warmup=5 + 100ep).
- **SAF Fourier encoding is NEGATIVE**: PR #1210 — saf (arc-length, dims 2-3) Fourier encoding regresses +12.4%. Do not revisit.
- **dsdf Fourier encoding is NEGATIVE**: PR #1169 — shape descriptor dims 4-11 Fourier encoding +11-13% regression. Do not revisit.
- **n_hidden width scaling is NEGATIVE on early configs**: 128 < 192 < 256 under n_layers=1 without warmup+full schedule. n_hidden=192 + warmup+T_max=100 is still untested.
- **Per-channel pressure weighting is NEGATIVE**: W_p in {2,3,5} did not beat baseline. Do not revisit.
- **surf_weight=28 confirmed optimal**: PRs #1173, #1141.
- **Batch size sweep {8,16,32}**: Being tested in PR #1234 (alphonse).
- **Extended Fourier freqs octave sweep**: Being tested in PR #1174 (tanjiro).

## Active WIP Experiments

NOTE: All experiments below must beat current best baseline (**34.3851**, PR #1226) to be mergeable.

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1234 | alphonse | Batch size sweep {8,16,32} with sqrt-scaled LR on FiLM+Fourier+warmup baseline | Running |
| #1174 | tanjiro | Extended Fourier freqs on (x,z): sweep L in {5,6,7,8} octaves | Running |
| #1250 | frieren | lr=2e-4 on current best config (T_max=100 + warmup=5 + 100ep) — revalidation of PR #1209 signal | Assigned |
| #1252 | thorfinn | n_layers=2 on full FiLM+Fourier+warmup+T_max=100 config — first test of depth scaling under current schedule | Assigned |
| #1254 | edward | T_max=150 to keep LR ~49% of peak at ep66 cutoff vs ~35% under T_max=100 | Assigned |
| #1255 | askeladd | n_hidden=192 width scaling on full FiLM+Fourier+warmup+T_max=100 config — first test with proper schedule | Assigned |
| #1257 | fern | T_max=200 extreme slow decay to keep LR ~54% of peak at ep66 cutoff | Assigned |
| #1258 | nezuko | lr=1.5e-4 finer LR sweep point to bracket optimal LR with frieren's lr=2e-4 | Assigned |

## Potential Next Research Directions

### Top Priority (directly motivated by current results)

1. **Extended training T_max=150 or T_max=200 + 100ep**: Model cut at ep66/100 under T_max=100 with LR still ~0.35× peak. T_max=150+ would keep LR higher at cutoff, extending the productive learning region. Single most promising next direction.

2. **lr=2e-4 on current best config**: PR #1209 found lr=2e-4 > lr=3e-4 on old config. Must retest on T_max=100 + warmup=5 + 100ep pipeline. This is a low-risk, potentially high-reward experiment.

3. **n_layers=2 with current best config**: Depth scaling may benefit from the full warmup+T_max=100 schedule. If depth helps at all, it would show here.

4. **n_hidden=192 with current best config**: Width scaling on old configs showed regression, but the full schedule (T_max=100 + warmup=5) may change this. Low risk if we include warmup.

5. **Finer LR sweep at current config**: {1.5e-4, 2e-4, 2.5e-4} on the T_max=100 + warmup=5 config, once the lr=2e-4 signal is validated.

### Medium Priority (architecture / regularization)

6. **Gradient accumulation + larger effective batch**: Current batch_size=4; accumulating to effective 16–32 may smooth Lion updates and improve OOD generalization. Requires train.py code change to support --grad_accum_steps.

7. **SWA over last N epochs**: Averaging weights across the end of training to stabilize predictions, especially helpful if LR drops sharply near cutoff.

8. **Relative positional encoding within slices**: Arc-length based RPE inside PhysicsAttention slices to explicitly model proximity along the airfoil surface.

9. **Per-split normalization**: Separate normalization statistics per regime to reduce distributional mismatch across splits, especially geom_camber_rc (consistently the hardest split).

### Longer-term (if current track plateaus)

10. **Graph neural network backbone**: Replace Transolver with a GNN that explicitly models mesh connectivity
11. **Ensemble of diverse checkpoints**: Average predictions from models with different seeds or hyperparameter variants
12. **Physics-informed regularization**: Continuity-equation soft constraints or pressure gradient smoothness
13. **Curriculum learning**: Start on single-foil easy cases before introducing tandem/camber regimes
14. **Data augmentation**: Geometric perturbations to improve OOD generalization on geom_camber splits
