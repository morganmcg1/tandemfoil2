# SENPAI Research State
- 2026-04-29 (updated)
- No directives from human researcher team
- Branch: icml-appendix-charlie-pai2e-r1

## Current Best (val_avg/mae_surf_p)

**94.6541** — PR #1005 (edward): n_layers=3, slice_num=16 stacked on compound baseline, epoch 12/12

Reproduce:
```bash
cd target/ && python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999
```
*(Note: `n_layers=3` and `slice_num=16` are hardcoded in the `model_config` dict in `train.py` — not CLI flags.)*

## Merged Winner Chain (cumulative stacking)

| PR | Description | val_avg/mae_surf_p | Delta |
|----|-------------|-------------------|-------|
| Baseline (MSE) | Default train.py | 126.88 | — |
| #788 | Huber loss (delta=1.0) | 115.6496 | -8.85% |
| #827 | + surf_weight=30 | 109.5716 | -5.26% |
| #808 | + bf16 + n_hidden=256 + n_head=8 + epochs=12 | 104.1120 | -4.97% |
| #882 | + EMA(decay=0.999) | 103.2182 | -0.86% |
| #1005 | + n_layers=3, slice_num=16 (reference arch) | **94.6541** | **-8.31%** |

## Active Experiments (all WIP — 8 students, 0 idle)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #1015 | edward | epochs=24 on compound baseline (nl3/sn16); val still falling at ep12 | Running | Val curve was monotonically decreasing through ep12; doubling budget should give major gain |
| #998 | frieren | slice_num 64→128 on compound baseline (wider PhysicsAttn) | Running | Testing if more slices improves spatial resolution of physics attention |
| #1011 | alphonse | surf_weight sub-10 sweep (1/3/5/7) on compound baseline | Running | PR #960 showed monotone degradation: sw=10 < sw=20 < sw=30 < sw=50. Optimum may be below 10. |
| #942 | nezuko | EMA decay sweep: 0.99/0.995 vs 0.999 on compound | Running | Tighter decay may fix in-dist/OOD asymmetry |
| #904 | fern | Huber delta sweep (0.25/0.5/1.0/2.0) on compound | Running | Finer Huber delta tuning on full compound stack |
| #795 | thorfinn | Per-sample loss normalization + Huber (rebase) | Awaiting rebase | R3 result: 93.3991 (-9.5% vs #808). Must rebase onto post-#1005 baseline + re-run |
| #794 | tanjiro | LR warmup (2 epochs) + Huber (rebase) | Awaiting rebase/revision | Warmup benefit confirmed (-4.87%); revise to 2-epoch warmup + full compound stack |
| #789 | askeladd | Gradient clipping (max_norm=1.0) (rebase) | Awaiting rebase | v3 result: 109.6719 vs OLD Huber baseline. Must re-run on full compound |

## Key Technical Insights

1. **Compound baseline is mandatory.** All PRs must stack on: `--n_hidden 256 --n_head 8 --loss huber --huber_delta 1.0 --epochs 12 --grad_clip 1.0 --ema_decay 0.999` with `n_layers=3, slice_num=16` hardcoded in model_config.
2. **n_layers=3, slice_num=16 is the biggest win yet.** PR #1005 gave -8.31% val and -9.43% test vs the compound baseline. Over-partitioned physics attention (slice_num=64) was hurting generalization. The reference architecture generalizes dramatically better, especially OOD.
3. **EMA merged.** decay=0.999 gives -0.86% val overall. The per-split asymmetry (helps OOD, hurts in-dist) points toward a tighter decay (0.99/0.995) being potentially better.
4. **Model is still converging at epoch 12 on nl3/sn16.** Val curve was monotonically decreasing through ep12 (94.7 at the final epoch). Longer training (24 epochs) is the highest-confidence next win.
5. **Gap to reference is ~2x.** README reference (nl3/sn16) reports test_avg ~40.9; current best is 83.76. Reference likely used longer training, possibly different data normalization or LR schedule horizon. Closing this gap is the primary research focus.
6. **Per-sample norm (thorfinn) shows strong signal.** R3 result: 93.3991 on post-#808 stack — outstanding potential winner. Must rebase onto post-#1005 compound and re-run.
7. **AdamW wd=1e-2 CLOSED — over-regularizes on full compound.** PR #828 R3: 106.9111 vs 103.2182 (+3.58% WORSE). Default wd (1e-4) is appropriate for the current stack.
8. **surf_weight upweighting is dead on compound stack.** PR #960 showed clean monotone: sw=10 < sw=20 < sw=30 < sw=50. The question is whether sw < 10 helps (PR #1011 in flight).
9. **T_max cosine fix is confirmed no-op.** PR #987 verified T_max was already correct.
10. **Surface pressure dominates the gap to reference.** Velocity errors (Ux ~1.3, Uy ~0.6) are already reasonable; `mae_surf_p` drives the ranking metric.

## Priority Queue for Next Hypotheses (when students become idle)

**Immediate high-priority (closing gap to reference ~40.9 test):**
1. ~~**epochs=24 on nl3/sn16 compound**~~ — In flight (PR #1015, edward). Val was monotonically decreasing at ep12.
2. **epochs=36 on nl3/sn16 compound** — Follow-up to #1015 if val is still decreasing at ep24.
3. **n_layers=3, slice_num=32** — Midpoint between sl=16 and sl=64. May have a Goldilocks optimum.
4. **Per-sample norm compound (thorfinn rebase)** — R3: 93.4 on old stack; rebase on post-#1005 compound. Could stack with major gain.

**Architecture tuning on nl3/sn16 baseline:**
5. **n_hidden=192 or 128 with nl3/sn16** — With 3 layers, hidden=256 may now be over-parameterized. Smaller hidden could reduce overfitting further.
6. **FiLM conditioning** — Inject Re and AoA as global conditioning on slice tokens (physics-informed).
7. **Separate pressure decoder** — Surface pressure (p) may benefit from its own decoder head vs shared decoder.

**Optimization / regularization:**
8. **EMA decay 0.99 or 0.995** — In flight (PR #942, nezuko). Tighter decay may help with per-split asymmetry.
9. **Huber delta tuning on nl3/sn16** — In flight (PR #904, fern). Optimal delta may shift with new architecture.
10. **AdamW beta2=0.95** — Faster momentum decay for noisy gradients.

**Closing the 2x reference gap (if training alone doesn't close it):**
11. **Reference LR schedule investigation** — What LR does the README reference use? Try lr=1e-3 or lr=2e-4 on nl3/sn16.
12. **surf_weight=1 or surf_weight=0** — Is any surface upweighting needed on nl3/sn16? May be cleaner without it.

**Bold bets (if plateau):**
13. **Physics residual loss** — Add divergence-free penalty on velocity field (Ux, Uy).
14. **Multi-resolution inputs** — Downsample mesh for early layers, full resolution for surface prediction.

## Research Theme: Closing the 2× Reference Gap

The model architecture discovery (n_layers=3, slice_num=16) was the biggest single jump (+8.3% val, +9.4% test). We now sit at val=94.65, test=83.76. The reference competition result is test_avg ~40.9 — still 2× better than us.

Key drivers of the remaining gap (most to least likely):
1. **Training duration** — val curve was still decreasing at ep12. Longer training (24/36 epochs) is the #1 hypothesis.
2. **LR schedule tuning** — The reference config may use different peak LR or cosine horizon. Our default lr=5e-4 was not tuned for nl3/sn16.
3. **Per-sample normalization** — thorfinn's per-sample loss norm showed R3 result of 93.4 on the OLD stack; stacking on nl3/sn16 compound could be a multiplicative win.
4. **Architecture capacity** — n_hidden=256 with 3 layers may still be over-/under-parameterized; sweep needed.
5. **Loss formulation** — Huber delta=1.0 was tuned on nl5/sn64; optimal delta may shift on the new architecture.

The reference config test best of 40.927 vs current test best 83.76 suggests approximately 2× headroom. Longer training alone could close a substantial fraction of this gap given the monotone decreasing val curve.

## Round 1 & 2 — Merged / Closed / Final Dispositions

**MERGED:**
- **PR #788** (alphonse, Huber loss): MERGED — val 115.6496
- **PR #827** (alphonse, surf_weight=30): MERGED — val 109.5716
- **PR #808** (fern, bf16+n_hidden=256+n_head=8+epochs=12): MERGED — val 104.1120
- **PR #882** (nezuko, EMA decay=0.999): MERGED — val 103.2182
- **PR #1005** (edward, n_layers=3, slice_num=16): MERGED — **val 94.6541 (current best), test 83.7608**

**CLOSED:**
- **PR #790** (edward, surf_weight on MSE): 128.98, above Huber baseline. Re-assigned as #827.
- **PR #791** (fern, wider model fp32): 155.96 on MSE. bf16 follow-up became #808.
- **PR #792** (frieren, n_layers=8 → grad_clip focus): 5 rounds, R5 no improvement over compound. grad_clip already in compound.
- **PR #793** (nezuko, slice_num=128): 130.97 > 115.6496 baseline; wall-clock penalty dominates.
- **PR #828** (edward, AdamW wd=1e-2 on full compound): 106.9111 vs 103.2182 (+3.58% WORSE). Over-regularizes.
- **PR #960** (alphonse, surf_weight 20/30/50 sweep): All worse than sw=10. monotone degradation — sw upweighting dead on compound.
- **PR #987** (edward, LR cosine T_max fix): No-op — T_max was already correct.
- **PR #792 R5** (frieren, full compound rebase): 107.54 vs 103.22; grad_clip already in compound; no unique delta remaining.
