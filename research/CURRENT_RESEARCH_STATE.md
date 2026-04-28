# SENPAI Research State

- **Last update:** 2026-04-28 06:20 (advisor branch `icml-appendix-charlie-pai2d-r2`)
- **Most recent human-team direction:** N/A — no open human-tagged issues at this time.
- **Current baseline (directly measured): `val_avg/mae_surf_p = 66.841`, `test_avg/mae_surf_p = 58.488`** (PR #563, feature-noise-0025, epoch 14). All 4 test splits finite.

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
16. **PR #563 — Feature noise std=0.0025. val_avg = 66.841. test_avg = 58.488. NEW BASELINE.**

## Active experiments (WIP)

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #510 | alphonse | torch-compile-baseline | torch.compile(model) speed-up | WIP — long-running |
| #574 | thorfinn | slice-temp-2p0 | PhysicsAttention temp init 1.5 → 2.0 | WIP |
| #575 | nezuko | ema-decay-target-0995 | EMA decay_target 0.99 → 0.995 (UP) | WIP |
| #581 | edward | slice-num-96 | slice_num 64 → 96 (untested axis) | WIP |
| #582 | askeladd | grad-clip-10 | Gradient clipping max_norm=10 | WIP |
| #562 | fern | cosine-warmup-3ep | 3-ep warmup + T_max=11 (budget-aligned, softer start) | WIP (sent back) |
| #554 | tanjiro | weight-decay-zero | wd=0 on new stack (close wd direction) | WIP (sent back, rebase + wd=0) |
| #595 | frieren | feature-noise-zero-vs-schedule | std=0.0 (then 0.001 if needed) | WIP (just assigned) |

## Current research focus

**Fine-grain hyperparameter closure on the merged stack.** Three major sweep directions are being closed simultaneously:
1. **Feature noise std**: frieren closing the direction (0.0 → done, or interior min at ~0.001)
2. **Weight decay**: tanjiro closing (wd=0 test on new stack)
3. **LR schedule shape**: fern sweeping warmup length (3-ep warmup + T_max=11)

**Orthogonal architectural axes in flight:**
- Slice count (edward #581: slice_num=96 — first test of this axis post-compound-stack)
- Slice temperature (thorfinn #574: 2.0 — temperature profile still ascending)
- EMA decay UP direction (nezuko #575: 0.995 — negative regulation hypothesis)
- Gradient clipping (askeladd #582: max_norm=10 — safe range under huber-δ=0.25)
- torch.compile throughput (alphonse #510 — speed multiplier)

## Most promising potential next research directions

1. **Huber δ further reduction** — The δ profile has been monotone L1-approaching; δ=0.125 or δ=0.05 (near-L1) may give another increment. The mechanism is clear (heavy-tailed pressure error distribution). This is the most mechanistically motivated next sweep.

2. **Decaying feature noise schedule** — Rather than a constant std, decay noise linearly from 0.005 → 0.0 aligned with the cosine LR schedule (most noise early when LR is high; zero noise in the fine-tuning tail). Could combine benefits of both noise levels. One-liner: multiply std by `(1 - epoch/T_max)`.

3. **surf_weight sweep** — Current surf_weight=10. This has not been swept since the early rounds (pr #286 tried surf_weight=25 on the MSE baseline and failed). On the current huber+EMA+SwiGLU stack, the optimal surface/volume tradeoff may have shifted. Try surf_weight=5, 15, 20 to find the current optimum.

4. **Cosine eta_min > 0** — Currently cosine decays to LR=0. Setting eta_min=5e-6 or eta_min=1e-5 prevents full LR collapse and may keep the model in a more active update regime at ep14. CosineAnnealingLR accepts `eta_min` parameter.

5. **Larger batch size** — Current batch_size=4. Batch=8 with gradient accumulation or batch=8 directly (GPU has 96GB). Larger effective batch reduces gradient noise and may compound with the reduced feature noise. Note: alphonse's torch.compile may enable higher throughput.

6. **FiLM-style Re conditioning** — Embed `log(Re)` as a 1D scalar via a learned FiLM affine transform applied per-block (γ, β = Linear(log_Re, 2*n_hidden)). This would let the model explicitly modulate features by Reynolds number, directly targeting the `val_re_rand` split. The Re is already in the input features but as a shared global — FiLM makes it an architectural primitive.

7. **Per-block learnable temperature** — PhysicsAttention temperature is currently a single init value converging to ~0.95 post-training. Making it a per-block learnable scalar (initialized to 1.5) could give different blocks different softmax sharpness. Low complexity; might compound with the temperature init gains already captured.

8. **Reduced huber delta δ=0.1** — Profile so far: δ=2 → worse, δ=1 → 83.2, δ=0.5 → 87.3 (pre-EMA), δ=0.25 → 72.4 (merged). The non-diminishing returns from δ=1→0.5→0.25 suggest trying δ=0.1 is warranted. Essentially pseudo-L1 in the loss.

## Disconfirmed directions (do not retry)

Per-channel surface loss weighting, per-channel output heads, depth-8, balanced capacity scale-up, max_norm=1.0 grad clipping under MSE, Fourier Re embedding standalone, activation choice (GELU vs SiLU), per-node Gaussian feature noise (semantics-unaware), SwiGLU preprocess MLP at input projection, huber δ=2.0, SwiGLU output head (no residual buffer).

## Constraints / guardrails

- Branch: `icml-appendix-charlie-pai2d-r2`
- Local JSONL metric logging only. No W&B / wandb / Weave.
- Do not override `SENPAI_TIMEOUT_MINUTES` or `--epochs`.
- Read-only: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, `data/generate_manifest.py`, `data/split_manifest.json`.
- Experiment edits live in `train.py` only.
