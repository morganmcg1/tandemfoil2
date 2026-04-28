# SENPAI Research State

- **Last updated**: 2026-04-28 20:30 UTC
- **Branch**: `icml-appendix-charlie-pai2e-r3`
- **Most recent direction from human researcher team**: none on file. Default contract from `target/program.md` stands — drive `val_avg/mae_surf_p` (and the matching `test_avg/mae_surf_p` from best val checkpoint) down on the equal-weight 4-split mean.

## Current focus

Round 1 in flight. All 8 student PRs assigned and running (PRs #830–#837). Goal: cast a **wide, diverse net** of round-1 hypotheses to find where the headroom on top of the default Transolver actually lives, then concentrate later rounds on the most productive axes.

The default config is small (n_hidden=128, 5 layers, 4 heads, slice=64, ~1–2M params, batch=4, lr=5e-4, surf_weight=10). Each GPU has 96 GB VRAM, so capacity scaling has cheap headroom; epoch and timeout caps cap wall-clock instead.

The hardest splits are the camber-holdouts (`val_geom_camber_rc` M=6–8, `val_geom_camber_cruise` M=2–4) — front-foil geometry never seen in training.

### Round 1 assignments

| PR  | Student   | Hypothesis                                                   |
|-----|-----------|--------------------------------------------------------------|
| #830 | alphonse  | Larger model: n_hidden=256, n_layers=7, n_head=8            |
| #831 | askeladd  | Higher surf_weight: 10→50                                   |
| #832 | edward    | More attention slices: slice_num=64→128                     |
| #833 | fern      | Warmup+cosine LR: 5-epoch linear warmup, peak lr=1e-3       |
| #834 | frieren   | Dropout regularization: attn_drop=proj_drop=0.1 in attention+FFN |
| #835 | nezuko    | MAE (L1) loss instead of MSE for Re-outlier robustness      |
| #836 | tanjiro   | Wider FFN: mlp_ratio=2→4                                    |
| #837 | thorfinn  | Pressure channel weighting: channel_weights=[1,1,5] in MSE  |

## Themes for round 2+ (breadth before depth)

1. **Loss reformulation** — Huber/L1 vs MSE; per-channel weighting that better targets pressure; log-space pressure target; density-weighted loss for large-y outliers at high-Re.
2. **Capacity scaling** — wider / deeper Transolver, more attention slices; 96 GB unlocks 4–10× the param count.
3. **Surface-aware modeling** — Fourier surface features, surface-only auxiliary head, surface-biased attention, more weight on the surface MAE that we're actually evaluated on.
4. **Coordinate / feature embeddings** — Fourier features over (x, z), log-Re, camber; FiLM conditioning on geometry tokens.
5. **Geometry conditioning** — cross-attention to compact foil-geometry tokens; better handling of camber holdouts.
6. **Optimization recipe** — warmup + cosine, OneCycle, gradient clipping, AdamW betas, EMA / SWA, larger effective batch via gradient accumulation.
7. **Augmentation / sampling** — reflection symmetry around chord, Re jitter, oversample tandem, multi-batch by mesh size.
8. **Regularization** — stochastic depth, attention dropout, MixOut, layer scale.

## Potential next directions (round 3+)
- Once the productive axes are known, stack them — capacity × loss × schedule.
- Explore physics-informed losses (divergence-free penalty on (Ux, Uy) for incompressible flow; pressure-Poisson residual).
- Multigrid / multi-resolution training (coarse-to-fine).
- Equivariant features (geometric symmetry), neural cellular-automaton-style iterative refinement.
- Two-stage train: pre-train on rc-single, finetune on tandem.
- Meta-learning / per-domain heads or routers.
- Larger pre-trained backbones (ViT-style backbones over slice tokens).

## Open risks / things to watch
- 50-epoch budget is small for very large models — capacity scaling must come with throughput improvements (mixed precision, larger batch, fewer dataloader stalls) or it won't converge.
- Padding mask correctness — any custom loss must respect `mask` (no padding contributing to surface MAE).
- Test metric discipline — `test_avg/mae_surf_p` from **best val checkpoint**, not last epoch.
