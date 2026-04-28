# SENPAI Research State

- **Last update:** 2026-04-28 06:55 (advisor branch `icml-appendix-charlie-pai2d-r2`)
- **Most recent human-team direction:** N/A — no open human-tagged issues at this time.
- **Current baseline (directly measured): `val_avg/mae_surf_p = 66.195`, `test_avg/mae_surf_p = 58.063`** (PR #575, EMA decay_target=0.995, epoch 14). All 4 test splits finite.

## Merged compound stack (current advisor branch)

1. PR #282 — Huber loss (δ=1.0). val_avg = 105.999.
2. PR #361 — NaN-safe `evaluate_split`. First finite test_avg = 97.957.
3. PR #363 — EMA(decay=0.999). val_avg = 101.350.
4. PR #391 — LLaMA-style SwiGLU FFN. val_avg = 88.227.
5. PR #426 — EMA decay 0.999 → 0.99. val_avg = 83.223.
6. PR #455 — Stochastic depth (DropPath 0→0.1). val_avg = 80.480.
7. PR #463 — Huber δ=1.0 → 0.25. val_avg = 72.414.
8. PR #480 — AdamW β₂=0.95. Standalone val_avg = 77.951.
9. PR #479 — Bias-corrected EMA (warmup_steps=10). Standalone val_avg = 81.251.
10. PR #520 — PhysicsAttention temperature init=1.0. val_avg = 71.6985.
11. PR #527 — AdamW wd=3e-5. val_avg = 70.814.
12. PR #518 — EMA warmup_steps=50. val_avg = 71.428.
13. PR #525 — Cosine 1-ep warmup + T_max=13. **val_avg = 67.306** (biggest single-PR delta since δ=0.25).
14. PR #526 — Feature noise std=0.005. val_avg = 71.359 (standalone pre-#525).
15. PR #548 — PhysicsAttention temperature init=1.5. Standalone: 70.617.
16. PR #563 — Feature noise std=0.0025. val_avg = 66.841. test_avg = 58.488.
17. PR #574 — PhysicsAttention temperature init=2.0. val_avg = 66.847. test_avg = 58.112.
18. **PR #575 — EMA decay_target 0.99 → 0.995. val_avg = 66.195. test_avg = 58.063. CURRENT BASELINE.**

## Active experiments (WIP)

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #510 | alphonse | torch-compile-baseline | torch.compile(model) speed-up | WIP — long-running |
| #581 | edward | slice-num-96 | slice_num 64 → 96 (untested axis) | WIP |
| #582 | askeladd | grad-clip-10 | Gradient clipping max_norm=10 | WIP |
| #562 | fern | cosine-tmax-12-warmup-2 | 3-ep warmup + T_max=11 (budget-aligned, softer start) | WIP (sent back, rebase) |
| #554 | tanjiro | weight-decay-1e-5 | wd=0 on new stack (close wd direction) | WIP (sent back, rebase + wd=0) |
| #595 | frieren | feature-noise-zero-vs-schedule | std=0.0 (close noise direction) | WIP |
| #600 | nezuko | ema-decay-target-0999 | EMA decay_target 0.995 → 0.999 (bracket UP direction) | WIP (just assigned) |
| #601 | thorfinn | huber-delta-0p1 | Huber δ=0.25 → 0.1 (push toward pseudo-L1) | WIP (just assigned) |

## Current research focus

**Hyperparameter closure + profile extension on multiple active axes.** The merged stack now includes 18 improvements; we are bracketing the remaining open directions:

1. **EMA decay profile** (nezuko #600, 0.999): profile 0.95→75.655, 0.99→67.306, 0.995→66.195 is still descending — 0.999 brackets the upper end.
2. **Huber δ profile** (thorfinn #601, 0.1): profile monotone toward L1 with non-diminishing returns. δ=0.1 is pseudo-L1 for ~95% of training gradients.
3. **Feature noise std** (frieren #595, 0.0): close direction to zero; optimum may be in (0, 0.0025].
4. **Weight decay** (tanjiro #554, 0.0): wd profile flattening at 1e-5; wd=0 closes the question.
5. **LR schedule shape** (fern #562, 3-ep warmup + T_max=11): warmup/cosine tradeoff bracket.
6. **Slice count** (edward #581, 96): first test of this architectural axis on the merged stack.
7. **Gradient clipping** (askeladd #582, max_norm=10): safety test for exploding gradient prevention.
8. **torch.compile throughput** (alphonse #510): speed multiplier enabling more epochs per budget.

## Most promising potential next research directions

1. **Huber δ=0.05 / pure L1** — If δ=0.1 wins, push further. The δ profile has non-diminishing returns toward L1; the question is where the floor is. Pure L1 (`F.l1_loss`) is the degenerate case.

2. **EMA decay_target=0.9999** — If 0.999 wins, bracket further up. Each step has given consistent improvement; only empirical test can find the saturation point.

3. **Decaying feature noise schedule** — Rather than a constant std, decay noise linearly from 0.0025 → 0.0 aligned with the cosine LR schedule. Highest noise early (when LR is high), zero noise in the fine-tuning tail. Could combine the regularization benefit of noise with the cleaner-signal benefit of zero noise at low LR.

4. **surf_weight sweep** — Current surf_weight=10. Not swept since early rounds (PR #286, MSE baseline). On the current huber+EMA+SwiGLU stack, the optimal surface/volume tradeoff may have shifted. Try surf_weight=5, 15, 20.

5. **Cosine eta_min > 0** — Currently cosine decays to LR=0. Setting eta_min=5e-6 or eta_min=1e-5 prevents full LR collapse. CosineAnnealingLR accepts `eta_min` parameter directly.

6. **Larger batch size (batch=8)** — Current batch_size=4. GPU has 96GB with ~46GB in use. Batch=8 reduces gradient noise and may compound with reduced feature noise. Requires verifying no OOM on the largest meshes (~242K nodes).

7. **FiLM-style Re conditioning** — Embed `log(Re)` as a 1D scalar via a learned FiLM affine transform applied per-block (γ, β = Linear(log_Re, 2*n_hidden)). Directly targets the `val_re_rand` split. Re is already in input features but as a shared scalar — FiLM makes it an architectural primitive that modulates per-block.

8. **Slice temperature init=2.5/3.0** — Profile: 1.0→71.699, 1.5→70.617, 2.0→66.847 (accelerating improvement). Optimum not bracketed from above; 2.5 is the natural next probe.

9. **Per-block learnable temperature** — PhysicsAttention temperature currently shared across blocks (single init). Making it per-block (5 independent scalars initialized at 2.0) could give different blocks different sharpness profiles. Low complexity.

## Disconfirmed directions (do not retry)

Per-channel surface loss weighting, per-channel output heads, depth-8, balanced capacity scale-up, max_norm=1.0 grad clipping under MSE, Fourier Re embedding standalone, activation choice (GELU vs SiLU), per-node Gaussian feature noise (semantics-unaware), SwiGLU preprocess MLP at input projection, huber δ=2.0, SwiGLU output head (no residual buffer).

## Constraints / guardrails

- Branch: `icml-appendix-charlie-pai2d-r2`
- Local JSONL metric logging only. No W&B / wandb / Weave.
- Do not override `SENPAI_TIMEOUT_MINUTES` or `--epochs`.
- Read-only: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, `data/generate_manifest.py`, `data/split_manifest.json`.
- Experiment edits live in `train.py` only.
