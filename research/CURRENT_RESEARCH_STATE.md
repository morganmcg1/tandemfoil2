# SENPAI Research State
- 2026-04-29 (updated: round icml-appendix-charlie-pai2f-r3)
- Most recent research direction from human researcher team: None (no GitHub Issues)
- Pending winner (rebase): PR #1167 (FiLM+Fourier combined) pre-rebase result `val_avg = 40.6661` (−7.5%); merge blocked pending alphonse rebase onto post-#1148 tip

## Current Best Baseline

**val_avg/mae_surf_p = 43.9575** (PR #1148, merged 2026-04-29)
Configuration: Lion + L1 + EMA(0.995) + bf16 + n_layers=1 + surf_weight=28 + cosine T_max=15 + clip_grad=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + batch_size=4 + 50 epochs + Fourier pos enc on (x,z) freqs=(1,2,4,8,16,32,64)

Note: Fourier pos enc expands (x,z) from 2-dim to 30-dim; total input dim 52. Adding freq=128 regresses sharply (Nyquist aliasing near mesh resolution scale).

Per-split val:
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 44.6169 |
| val_geom_camber_rc | 57.7367 |
| val_geom_camber_cruise | 26.7301 |
| val_re_rand | 46.7462 |
| **val_avg** | **43.9575** |

Key observation: `val_geom_camber_rc` (57.74) is the dominant error split — the hardest OOD split and largest contributor to the average. Experiments targeting this split will have disproportionate impact.

## Current Research Focus

Build on the extended Fourier frequency baseline (PR #1148, freqs=(1,2,4,8,16,32,64)). The two most promising orthogonal directions are:
1. **FiLM global conditioning** — edward's PR #1104 achieved −11.2% on an earlier baseline; if confirmed on the current Fourier baseline, the combination could be transformative.
2. **Training duration** — best checkpoint at epoch 48/50 (still descending), so longer training remains an active lever.

**Primary metric**: `val_avg/mae_surf_p` (lower is better)

## Active Experiments (8/8 students running)

| PR    | Student   | Hypothesis                                                                          | Status |
|-------|-----------|-------------------------------------------------------------------------------------|--------|
| #1104 | edward    | FiLM global conditioning: rebase on freqs=(1,2,4,8,16,32,64) baseline (rebase pending) | wip |
| #1167 | alphonse  | FiLM global conditioning + Fourier pos enc on current best baseline                 | REBASE PENDING — sent back 2026-04-29; pre-rebase val_avg=40.6661 (−7.5%); strong winner pending rebase |
| #1168 | askeladd  | Extended training 75 epochs + T_max=10 on Fourier pos enc baseline                  | wip |
| #1169 | edward    | Fourier encoding on dsdf dims 4-11: multi-scale shape descriptor expansion           | wip |
| #1170 | fern      | Depth sweep: n_layers in {2,3} on Fourier pos enc baseline                          | wip |
| #1173 | nezuko    | surf_weight sweep {28, 32, 40, 50} on Fourier pos enc baseline                      | wip |
| #1174 | tanjiro   | Extended Fourier freqs on (x,z): sweep L in {5,6,7,8} octaves                       | wip |
| #1175 | thorfinn  | LR warmup + cosine decay for Lion: stabilise early updates                          | wip |
| #1181 | frieren   | Wider hidden (n_hidden=256) + FiLM global conditioning on Fourier pos enc baseline  | wip |

Note: edward has 2 active PRs (#1104 rebase + #1169 new); all other students have 1 each.

## Closed This Round (Negatives / Dead Ends)

- PR #1108 (tanjiro, n_hidden width sweep) — DECISIVE NEGATIVE: width monotonically hurts (128 < 192 < 256), do not revisit
- PR #1105 (fern, per-channel pressure weighting W_p in {2,3,5}) — NEGATIVE: up-weighting doesn't help on aggregate
- PR #1103 (askeladd, slice_num sweep {32,64,128}) — CLOSED: ran on old compound baseline without --fourier_pos_enc
- PR #1109 (thorfinn, BL feature log(Re×|saf|+ε)) — CLOSED NEGATIVE: ran without --fourier_pos_enc; BL feature showed zero gain

## Current Research Themes

1. **Fourier positional / geometric representation** — proven wins: PR #1106 (+6.29%), PR #1148 (+1.03%). Current frontier: freqs=(1,2,4,8,16,32,64). Follow-ups: Fourier on dsdf dims 4-11 (#1169), extended freq sweep L={5,6,7,8} (#1174).
2. **Physics conditioning via FiLM** — strongest pre-advance signal at −11.2% (edward #1104, awaiting rebase); if orthogonal to Fourier encoding, combination could yield very large improvement. Multiple students now testing FiLM variants (#1104, #1167, #1181).
3. **Training dynamics** — extended training (#1168), LR warmup/cosine (#1175). Training still descending at epoch 50; longer training expected to improve.
4. **Architecture capacity** — depth sweep n_layers {2,3} (#1170), wider hidden + FiLM (#1181).
5. **Loss weighting** — surf_weight sweep {28,32,40,50} (#1173). `val_geom_camber_rc` is the dominant error; heavier surface weighting may close the gap.

## Potential Next Research Directions (after current round)

1. **Stack confirmed FiLM + Fourier + best training schedule** — compound the orthogonal wins once individual components are confirmed.
2. **Learnable random Fourier features** — replace fixed octave grid with a trainable Gaussian projection matrix (Tancik et al. 2020), letting the network find its own spatial frequencies.
3. **Domain-explicit conditioning** — inject split label (single/raceCar-tandem/cruise-tandem) as one-hot embedding, specifically targeting `val_geom_camber_rc` (57.74).
4. **CosineAnnealingWarmRestarts with T_mult=2** — broader loss landscape exploration with lengthening restart cycles.
5. **Curriculum learning** — order samples by split difficulty (rc > in-dist > re-rand > cruise) to improve OOD generalization.
6. **Physics-informed auxiliary loss** — soft incompressibility constraint as an auxiliary loss term.
7. **Separate pressure decoder head** — specialized decoder for pressure channel while sharing velocity encoder.
8. **Graph-based k-NN local attention** — boundary layer capture via explicit local mesh connectivity.
9. **Fourier on velocity channels** — apply sinusoidal encoding to velocity-field input dims in addition to spatial dims.
10. **Ensemble / test-time augmentation** — lightweight ensemble of multiple checkpoints or TTA over symmetric geometry augmentations.
