# SENPAI Research State

- **Updated:** 2026-04-28 19:10 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none yet — no GitHub issues)_

## Current research focus

Round 1 of a fresh research track. Goal is to beat the unmodified Transolver baseline (`n_hidden=128, n_layers=5, n_head=4, slice_num=64`, AdamW lr=5e-4, MSE, surf_weight=10, batch_size=4) on `val_avg/mae_surf_p` and the corresponding `test_avg/mae_surf_p` across the four validation/test tracks (`single_in_dist`, `geom_camber_rc`, `geom_camber_cruise`, `re_rand`).

Round 1 deploys 8 hypotheses spanning four diverse strategy categories so that even if any single category is unproductive, we learn quickly:

1. **Loss reformulation** (alphonse — Huber; edward — learnable uncertainty per channel)
2. **Architecture / capacity** (frieren — width=256, slices=128; tanjiro — depth=8/10; askeladd — surface-aware slice routing)
3. **Optimization / training stability** (nezuko — LR warmup + grad clip; fern — EMA weights)
4. **Data augmentation** (thorfinn — log(Re) jitter)

Surface-pressure MAE is the ranking metric, so each hypothesis is scored against effect on `val_avg/mae_surf_p` and follow-up paper-facing `test_avg/mae_surf_p`. Per-split disagreements are flagged as information (which split a hypothesis helps tells us about its mechanism).

## Potential next research directions (Round 2+)

Pending Round 1 outcomes, candidate themes for Round 2:

- **Compounding wins.** If Huber+EMA+grad-clip all win independently, stack them. Architecture and optimization wins are usually orthogonal.
- **Pressure-channel-only training tail.** A second-stage fine-tune that drops the Ux/Uy loss entirely — only the metric-of-record is optimized.
- **Better surface conditioning.** Cross-attention between surface and volume nodes; surface-pinned skip connections; explicit boundary-layer inductive biases.
- **Stronger geometry generalization.** Random foil reflection (cruise-style symmetric flow), camber/thickness perturbation augmentation, mesh subsampling for regularization.
- **Architectural alternatives.** Swap PhysicsAttention for Galerkin attention, or try a hybrid Transolver + GNN message-passing module on surface nodes.
- **Multi-task curriculum.** Train Ux/Uy first, freeze, fine-tune p — or domain-specific heads (raceCar single vs. tandem vs. cruise).
- **Loss shape on tails.** If Huber wins at delta=2.0, try log-cosh or asymmetric losses tuned to pressure's positive/negative skew.
- **Mixed precision + larger effective batch.** AMP + gradient accumulation to free wall-clock for higher epoch counts.
- **Test-time scoring with TTA.** Mesh perturbation TTA averaged at inference.

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling lives in `train.py`.
- One hypothesis per PR. Compound only after each isolated win is verified.
- Prefer common-recipe changes that survive across all four splits over hacks that only improve one.
