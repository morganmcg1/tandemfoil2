# SENPAI Research State
- 2026-04-29 16:30 (icml-appendix-charlie-pai2f-r3)
- No recent research directives from the human researcher team

## Current Best Baseline

**val_avg/mae_surf_p = 39.9450** (PR #1104, merged 2026-04-29)
Configuration: Lion + L1 + EMA(0.995) + bf16 + n_layers=1 + surf_weight=28 + cosine T_max=15 + clip_grad=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + 50 epochs + Fourier pos enc on (x,z) freqs=(1,2,4,8,16,32,64) + **FiLM global conditioning** (Re/AoA/NACA scale+shift per TransolverBlock, DiT/AdaLN-Zero init)

Note: FiLM conditioning injects 11-dim global physics vector (log Re, AoA1, NACA1(3d), AoA2, NACA2(3d), gap, stagger) as learned scale+shift applied to each TransolverBlock. Fourier pos enc expands (x,z) from 2-dim to 30-dim; total input dim 52. Adding freq=128 regresses sharply (Nyquist aliasing near mesh resolution scale).

Per-split val:
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 38.3034 |
| val_geom_camber_rc | 56.1374 |
| val_geom_camber_cruise | 22.9918 |
| val_re_rand | 42.3473 |
| **val_avg** | **39.9450** |

Per-split test:
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 32.0588 |
| test_geom_camber_rc | 49.6238 |
| test_geom_camber_cruise | 18.9013 |
| test_re_rand | 33.7205 |
| **test_avg** | **33.5761** |

Key observation: `val_geom_camber_rc` (56.14) remains the dominant error split — the hardest OOD split. FiLM helped all splits but rc only minimally (−3.5%), suggesting rc needs more targeted mechanisms (domain-explicit conditioning, arc-length RPE, or OOD geometry augmentation). Best epoch 49/50 — model not converged; extended training is the primary next lever.

## Current Research Focus

Build on the FiLM + Fourier combined baseline (PR #1104). Active experiments target:
1. **Training dynamics** — extended training (75ep + T_max=75, PR #1208), LR sweep ({1e-4,2e-4,3e-4,5e-4}, PR #1209), and warmup+cosine schedules to push past epoch 50 convergence limit.
2. **Input representation** — Fourier on dsdf dims 4-11 (#1169), extended frequency sweep L={5,6,7,8} (#1174).
3. **Depth** — n_layers in {2,3} on Fourier baseline (#1170) — CLOSED DECISIVE NEGATIVE. If any other signal suggests depth, must re-test WITH FiLM.
4. **surf_weight tuning** — CLOSED: sw=28 confirmed optimal; sw sweep {28,32,40,50} exhausted (PR #1173).

NOTE: PRs #1168/#1175/#1169/#1174/#1170 run WITHOUT FiLM on Fourier-only baseline. They must beat 39.9450 to be mergeable. If they show strong directional signals (e.g., depth helps, different octave count helps), the idea should be re-tested WITH FiLM.

**Primary metric**: `val_avg/mae_surf_p` (lower is better)

## Active Experiments (WIP)

| PR    | Student   | Hypothesis                                                                          | Status |
|-------|-----------|-------------------------------------------------------------------------------------|--------|
| #1208 | frieren   | Extended training 75ep + T_max=75 on FiLM+Fourier baseline                         | Assigned (NEW) |
| #1209 | nezuko    | Lion LR sweep {1e-4,2e-4,3e-4,5e-4} on FiLM+Fourier baseline                      | Assigned (NEW) |
| #1168 | askeladd  | Extended training 75 epochs + T_max=10 on Fourier baseline                          | Running (NOTE: no FiLM — result may not beat 39.9450) |
| #1175 | thorfinn  | LR warmup (5 epoch linear) + cosine decay on Lion, on Fourier baseline              | Running (NOTE: no FiLM — result may not beat 39.9450) |
| #1169 | edward    | Fourier encoding on dsdf dims 4-11: multi-scale shape descriptor expansion           | Running (NOTE: no FiLM — result may not beat 39.9450) |
| #1174 | tanjiro   | Extended Fourier freqs sweep L in {5,6,7,8} octaves on (x,z)                       | Running (NOTE: no FiLM — result may not beat 39.9450) |
| #1170 | fern      | Depth sweep: n_layers in {2,3} on Fourier pos enc baseline                          | Running (NOTE: no FiLM — result may not beat 39.9450) |
| #1167 | alphonse  | FiLM global conditioning + Fourier pos enc (REBASE PENDING)                         | pre-rebase val_avg=40.6661 — WORSE than new baseline 39.9450; must rebase onto post-#1104 tip and re-run |

## Closed This Round (Negatives / Superseded)

- PR #1173 (nezuko, surf_weight sweep {28,32,40,50}) — DEAD END: best (sw=32, val_avg=43.2052) designed against PR #1106 baseline; current best is PR #1104 (39.9450). sw=28 confirmed optimal; direction exhausted.
- PR #1196 (frieren, T_max=50 on Fourier baseline) — MERGED but does NOT beat FiLM baseline. val_avg=42.4863 (beats old Fourier baseline 43.9575, but not FiLM baseline 39.9450). T_max=50 improvement should be adopted in FiLM branch experiments.
- PR #1181 (frieren, wider hidden + FiLM combined) — DECISIVE NEGATIVE: val_avg=63.93 (+45% regression). Catastrophic train-loss spike at epoch 31. Rules out wider hidden direction.
- PR #1170 (fern, depth sweep n_layers {2,3}) — DECISIVE NEGATIVE: depth hurts; n_layers=3 overfits at epoch 22, n_layers=2 timed out projected above baseline. Depth expansion closed.
- PR #1108 (tanjiro, n_hidden width sweep) — DECISIVE NEGATIVE: width monotonically hurts (128 < 192 < 256), do not revisit
- PR #1105 (fern, per-channel pressure weighting W_p in {2,3,5}) — NEGATIVE: up-weighting doesn't help on aggregate
- PR #1103 (askeladd, slice_num sweep {32,64,128}) — CLOSED: ran on old compound baseline without --fourier_pos_enc
- PR #1109 (thorfinn, BL feature log(Re×|saf|+ε)) — CLOSED NEGATIVE: ran without --fourier_pos_enc; BL feature showed zero gain

## Current Research Themes

1. **FiLM global conditioning** — confirmed −9.13% gain (PR #1104 merged). Strongest lever found so far. Physics-scale/shift conditioning on Re/AoA/NACA regime vector. Key insight: DiT/AdaLN-Zero init critical for stable training.
2. **Fourier positional / geometric representation** — proven wins: PR #1106 (+6.29%), PR #1148 (+1.03%). Current frontier: freqs=(1,2,4,8,16,32,64). Follow-ups: Fourier on dsdf dims 4-11 (#1169), extended freq sweep L={5,6,7,8} (#1174).
3. **Training dynamics** — best epoch 49/50 (not converged). Extended training (#1208 on FiLM baseline), LR sweep (#1209 on FiLM baseline), LR warmup+cosine (#1175 on Fourier baseline).
4. **surf_weight tuning** — CLOSED: sw=28 confirmed optimal; direction exhausted.
5. **Depth scaling** — CLOSED DECISIVE NEGATIVE: n_layers=1 is the sweet spot.

## Potential Next Research Directions (after current round)

### Short-term (immediately actionable — next idle students)
1. **n_layers=2 + FiLM**: Depth sweep (PR #1170) runs on Fourier-only baseline; if depth helps directionally, combine with FiLM for additive gain. Strong candidate for next assignment.
2. **Best octave count + FiLM compound**: Best octave count from tanjiro's sweep + FiLM — likely additive if independent mechanisms.
3. **SWA (Stochastic Weight Averaging)**: Average model weights across last 10–20 epochs instead of EMA — especially compelling now that best epoch is at the cap.
4. **FiLM + extended training + optimal LR combined**: Once LR sweep (nezuko #1209) and extended training (frieren #1208) both return results, combine the best LR with extended training for a potential compound gain.
5. **alphonse rebase**: alphonse's FiLM+Fourier approach (pre-rebase val_avg=40.6661) is now worse than the merged baseline. If alphonse makes architectural changes and hits below 39.9450, it should be re-merged.

### In-flight experiments to watch
- PRs #1168/#1175/#1169/#1174/#1170 all run WITHOUT FiLM on the Fourier baseline — they must beat 39.9450 to be mergeable. Likely can't beat the FiLM baseline. If any show a strong directional signal (e.g., different octave count helps, LR warmup helps), the idea should be re-tested WITH FiLM.
- Key question: do the in-flight Fourier-only experiments (octave sweep, LR warmup, etc.) produce additive gains when combined with FiLM?

### Medium-term (if current round plateaus)
6. **Relative positional encoding within slices**: Arc-length based RPE inside PhysicsAttention slices to explicitly model proximity along the airfoil surface
7. **Domain-explicit conditioning**: Inject split label (single/raceCar-tandem/cruise-tandem) as one-hot embedding into FiLM conditioning vector, targeting `val_geom_camber_rc` (56.14) — the hardest OOD split
8. **Per-split normalization**: Separate normalization statistics per regime (single_in_dist, geom_camber, re_rand) to reduce distributional mismatch
9. **Data augmentation**: Geometric perturbations (slight chord scaling, small rotation jitter) to improve OOD generalization to geom_camber splits
10. **Learnable random Fourier features (RFF)**: Replace fixed octave grid with a trainable Gaussian projection matrix (Tancik et al. 2020), letting the network learn its own spatial frequencies

### Longer-term (if plateau persists)
11. **Graph neural network backbone**: Replace Transolver with a GNN that explicitly models mesh connectivity
12. **Ensemble of diverse checkpoints**: Average predictions from models with different random seeds or hyperparameter variants
13. **Physics-informed regularization**: Add continuity-equation soft constraints or pressure gradient smoothness terms
14. **Curriculum learning**: Start on single-foil easy cases before introducing tandem/camber regimes
15. **Attention temperature tuning**: Make PhysicsAttention temperature a learnable parameter per head with bf16-safe clipping
