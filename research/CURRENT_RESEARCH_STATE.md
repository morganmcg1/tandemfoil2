# SENPAI Research State
- 2026-04-29 12:30 (round: icml-appendix-charlie-pai2f-r3)
- Most recent research direction from human researcher team: None (no GitHub Issues)

## Current Best Baseline

**val_avg/mae_surf_p = 44.4154** (PR #1106, merged 2026-04-29)
Configuration: Lion + L1 + EMA(0.995) + bf16 + n_layers=1 + surf_weight=28 + cosine T_max=15 + clip_grad=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + 50 epochs + Fourier pos enc on (x,z) freqs=(1,2,4,8,16)

Note: Fourier pos enc expands spatial (x,z) from 2-dim to 22-dim; total input dim 44.

Per-split val:
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 45.6222 |
| val_geom_camber_rc | 58.5071 |
| val_geom_camber_cruise | 26.7073 |
| val_re_rand | 46.8250 |
| **val_avg** | **44.4154** |

Key observation: `val_geom_camber_rc` (58.51) is the dominant error split — the hardest OOD split and largest contributor to the average. Experiments targeting this split will have disproportionate impact.

## Current Research Focus

Build on Fourier positional encoding (PR #1106 win, −6.29%). The Fourier encoding expanded the input representation and gave the model richer geometric information — the natural next family of experiments is to extend/refine this representation (more frequencies, Fourier on other dims, learnable frequencies) and to combine it with orthogonal improvements (physics conditioning via FiLM, architecture depth).

Best checkpoint was still at epoch 50 (last epoch, not yet converged) → training duration remains an active lever. FiLM conditioning previously achieved −11.2% on an earlier baseline — if it still works on the new Fourier baseline, the combination could be transformative.

**Primary metric**: `val_avg/mae_surf_p` (lower is better)

## Active Experiments (8/8 students running)

| PR   | Student   | Hypothesis                                                            | Status |
|------|-----------|-----------------------------------------------------------------------|--------|
| #1127 | alphonse  | Extended training: 75 epochs + cosine T_max=10                       | wip |
| #1104 | edward    | FiLM global conditioning (Re/AoA/NACA via scale+shift on hidden)     | wip |
| #1141 | fern      | surf_weight sweep {28, 32, 40, 50} on Fourier pos enc baseline       | wip |
| #1140 | frieren   | Lion LR sweep {1e-4, 3e-4, 1e-3} on Fourier pos enc baseline        | wip |
| #1107 | nezuko    | EMA decay sweep {0.99, 0.995, 0.999} on compound baseline            | wip |
| #1139 | tanjiro   | Depth sweep: n_layers in {2, 3} on Fourier pos enc baseline          | wip |
| #1155 | thorfinn  | LR warmup (5–10 epochs linear) + cosine decay (T_max=40–45)         | wip |
| #1148 | askeladd  | Extended Fourier freqs on (x,z): freqs up to 32/64/128              | wip |

## Closed This Round (Negatives)
- PR #1108 (tanjiro, n_hidden width sweep) — DECISIVE NEGATIVE: width monotonically hurts (128 < 192 < 256), do not revisit
- PR #1105 (fern, per-channel pressure weighting W_p in {2,3,5}) — NEGATIVE: up-weighting doesn't help on aggregate
- PR #1103 (askeladd, slice_num sweep {32,64,128}) — CLOSED: ran on old compound baseline without --fourier_pos_enc; best 47.2312 far below current baseline 44.4154; slice_num=64 confirmed Pareto-optimal
- PR #1109 (thorfinn, BL feature log(Re×|saf|+ε)) — CLOSED NEGATIVE: ran without --fourier_pos_enc; BL feature showed zero gain vs own control (~46.5 vs ~46.2); re_rand split showed no benefit; direction closed

## Current Research Themes

1. **Fourier positional / geometric representation** — proven +6.29% win (PR #1106). Follow-ups: extended freq spectrum (askeladd #1148), Fourier on dsdf dims (4–11), learnable Gaussian random Fourier features (Tancik et al. 2020).
2. **Physics conditioning** — FiLM conditioning (edward #1104) was the strongest pre-advance signal at −11.2%; now running on Fourier baseline. If orthogonal to Fourier encoding, could compound to a very large improvement.
3. **Training dynamics** — extended training (alphonse #1127), EMA decay (nezuko #1107), LR sweep (frieren #1140). Training consistently not converged by epoch 50.
4. **Architecture capacity** — depth sweep n_layers {2,3} (tanjiro #1139). With wider input (44-dim) from Fourier, more depth may pay off.
5. **Loss weighting** — surf_weight sweep (fern #1141). Heavier surface weighting may pull rc/cruise splits together.
6. **Learning rate schedule** — warmup + cosine (thorfinn #1155). Lion momentum stabilisation via linear LR warmup followed by cosine decay; testing 5-epoch and 10-epoch warmup variants.

## Potential Next Research Directions (after current round)

1. **Stack FiLM + Fourier** — if both independently improve the baseline, combine them. Physics conditioning + geometric encoding are likely orthogonal. High expected value.
2. **Fourier on dsdf channels (dims 4–11)** — expand the 8-dim wall-distance descriptor with sinusoidal encoding. dsdf captures local geometry and is likely frequency-rich.
3. **Learnable random Fourier features** — replace fixed octave grid {1,2,4,8,16} with a trainable Gaussian projection matrix, letting the network find its own spatial frequencies.
4. **Domain-explicit conditioning** — inject domain label (single/raceCar-tandem/cruise-tandem) as one-hot or embedding, targeting `val_geom_camber_rc` (58.51) specifically.
5. **Longer training** — if alphonse shows 75 epochs still descending, push to 100 epochs with a very long cosine cycle.
6. **Width scaling with Fourier** — n_hidden=192/256 on the Fourier baseline; wider input may warrant wider hidden states.
7. **Separate pressure decoder head** — specialized decoder for pressure channel while sharing velocity encoder.
8. **CosineAnnealingWarmRestarts with T_mult=2** — broader loss landscape exploration with lengthening restart cycles.
9. **Curriculum learning** — order samples by split difficulty (rc > in-dist > re-rand > cruise) to improve OOD generalization.
10. **Graph-based k-NN local attention** — boundary layer capture via explicit local mesh connectivity.
11. **Physics-informed auxiliary loss** — soft incompressibility constraint as an auxiliary term.
