# SENPAI Research State

- **Last update:** 2026-04-28 08:15 (advisor branch `icml-appendix-charlie-pai2d-r2`)
- **Most recent human-team direction:** N/A — no open human-tagged issues at this time.
- **Current baseline (directly measured): `val_avg/mae_surf_p = 64.696` eager / `64.824` compile, `test_avg/mae_surf_p = 55.879` eager / `56.391` compile**. PR #562 (cosine 3-ep warmup + T_max=11) and PR #510 (torch.compile mode=default, +28.6% epochs in budget) both merged.
- **Stack throughput**: 18 epochs in 30-min budget under compile=True (vs 14 eager). Cosine T_max=11 leaves epochs 12–18 at LR≈0 — 7 free EMA-stabilization epochs.

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
18. PR #575 — EMA decay_target 0.99 → 0.995. val_avg = 66.195. test_avg = 58.063.
19. PR #582 — Gradient clipping max_norm=10. val_avg = 66.149. test_avg = 57.654.
20. PR #562 — Cosine schedule revision (3-ep warmup + T_max=11, start_factor=0.3). val_avg = 64.696. test_avg = 55.879.
21. **PR #510 — torch.compile mode="default" (infrastructure: +28.6% epochs / −23.1% wall-clock). val_avg = 64.824 at 18 epochs (compile-on-same-stack). CURRENT BASELINE.**

## Active experiments (WIP)

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #629 | alphonse | reduce-overhead-fixed-padding | Fixed-shape padding for torch.compile mode="reduce-overhead" (throughput) | WIP |
| #635 | edward | lr-peak-6e-4 | lr 5e-4 → 6e-4 (gentler 3-ep warmup permits higher peak LR) | WIP (just assigned) |
| #647 | askeladd | slice-temp-per-block-schedule | Per-block slice-temp init schedule [1.5..3.0] linear (hierarchical sharpness) | WIP (just assigned) |
| #646 | fern | batch-size-6 | batch_size 4 → 6 with compile (gradient noise reduction) | WIP (just assigned) |
| #640 | tanjiro | per-group-wd | AdamW per-parameter-group wd (attn higher, mlp lower) — captures OOD asymmetry | WIP (just assigned) |
| #636 | frieren | feature-noise-decaying | Decaying noise schedule aligned with cosine LR | WIP (just assigned) |
| #630 | nezuko | cosine-eta-min-2e-5 | cosine eta_min 0 → 2e-5 (extract gain from late-epoch budget under compile) | WIP |
| #601 | thorfinn | huber-delta-0p1 | Huber δ=0.25 → 0.1 (rebase onto post-#562/#510 stack) | WIP (sent back, rebase) |

## Current research focus

**Hyperparameter closure + profile extension on multiple active axes.** The merged stack now includes 21 improvements (latest: PR #510 compile = +28.6% epochs in budget compounds with everything); we are bracketing the remaining open directions:

1. **Cosine LR floor** (nezuko #630, eta_min=2e-5): replaces closed EMA-decay-target axis. Probe whether epochs 12–18 (now at LR≈0 under compile + T_max=11) extract more gain when LR has a positive floor.
2. **Huber δ profile** (thorfinn #601, 0.1): standalone -1.05% on PR #575 stack; rebasing onto post-#562/#510 stack to verify on current schedule.
3. **Decaying feature noise schedule** (frieren #636): replaces closed scalar-noise axis. Linear decay from std=0.0025 at ep0 to 0 at ep14 — match noise to LR phase (high during basin selection, zero in fine-tuning tail).
4. **Per-parameter-group wd** (tanjiro #640): single-scalar wd axis is closed at 3e-5; explore module-type-differential wd to capture the OOD asymmetry (attn higher to help camber_rc, mlp lower to help re_rand).
5. **Batch-size gradient quality** (fern #646, batch=6 with compile): gradient noise reduction may compound with EMA averaging. Replaces closed warmup-aggressiveness axis.
6. **LR peak bump** (edward #635, lr=6e-4): direct probe of whether gentler 3-ep warmup permits 1.2× higher peak LR safely.
7. **Per-block slice-temp init schedule** (askeladd #647, [1.5, 1.875, 2.25, 2.625, 3.0]): hierarchical sharpness — softer early blocks for spatial pooling, sharper later blocks for token refinement. Replaces saturated global-init axis.
8. **Reduce-overhead throughput** (alphonse #629, fixed-shape padding): infrastructure follow-up — fix the per-shape-CUDA-Graph OOM that defeated mode="reduce-overhead" in PR #510. Expected another +10–20% wall-clock if it lands.

**Closed axes**: EMA decay_target above 0.995 at warmup_steps=50 (cap doesn't bind within budget — PR #600); feature_noise_std (interior min at 0.0025, U-shaped — PR #595); surf_weight at 15 on huber-clip stack (clip absorbs the increase, single_in_dist vol_p degrades — PR #605); single-scalar wd (basin floor at 3e-5 on new stack; wd=0 regresses +2.75% — PR #554); LinearLR start_factor (sweet spot at 0.3, both 0.5 and 0.2 regress — PR #620); global slice-temp init (saturating at 2.0, camber_rc consistently regresses with sharper attention — PR #608).

## Most promising potential next research directions

1. **Huber δ=0.05 / pure L1** — If δ=0.1 wins, push further. The δ profile has non-diminishing returns toward L1; the question is where the floor is. Pure L1 (`F.l1_loss`) is the degenerate case.

2. **EMA decay_target=0.9999** — If 0.999 wins, bracket further up. Each step has given consistent improvement; only empirical test can find the saturation point.

3. **Decaying feature noise schedule** — Rather than a constant std, decay noise linearly from 0.0025 → 0.0 aligned with the cosine LR schedule. Highest noise early (when LR is high), zero noise in the fine-tuning tail. Could combine the regularization benefit of noise with the cleaner-signal benefit of zero noise at low LR.

4. **surf_weight sweep** — Current surf_weight=10. Not swept since early rounds (PR #286, MSE baseline). On the current huber+EMA+SwiGLU stack, the optimal surface/volume tradeoff may have shifted. Try surf_weight=5, 15, 20.

5. **Cosine eta_min > 0** — Currently cosine decays to LR=0. Setting eta_min=5e-6 or eta_min=1e-5 prevents full LR collapse. CosineAnnealingLR accepts `eta_min` parameter directly.

6. **Larger batch size (batch=8)** — Current batch_size=4. GPU has 96GB with ~46GB in use. Batch=8 reduces gradient noise and may compound with reduced feature noise. Requires verifying no OOM on the largest meshes (~242K nodes).

7. **FiLM-style Re conditioning** — Embed `log(Re)` as a 1D scalar via a learned FiLM affine transform applied per-block (γ, β = Linear(log_Re, 2*n_hidden)). Directly targets the `val_re_rand` split. Re is already in input features but as a shared scalar — FiLM makes it an architectural primitive that modulates per-block.

8. **Slice temperature init=3.0** — Profile: 1.0→71.699, 1.5→70.617, 2.0→66.847, 2.5→TBD (in flight). Optimum not bracketed from above.

9. **Per-block learnable temperature** — PhysicsAttention temperature currently shared across blocks (single init). Making it per-block (5 independent scalars initialized at 2.0) could give different blocks different sharpness profiles. Low complexity.

10. **Warmup ramp 4-ep + T_max=10 (start_factor=0.2)** — Push fern's gentler-warmup direction further. If 0.3→0.2 wins at 3+11, the next step is more warmup epochs at softer ramp.

11. **Larger batch size (batch=8 with compile)** — Current peak VRAM ~42.6 GB under compile; doubling batch may fit in 96 GB. Smoother gradients, may compound with EMA. Risk: VRAM headroom on largest meshes.

12. **Surface-only feature noise** — frieren's PR #595 follow-up. Apply noise only to per-node positional/SDF dims (0–12), not per-sample dims (13–23).

13. **Per-split EMA decay** — frieren's observation: cruise has the biggest single-split signal headroom. Per-split EMA could target it.

14. **FiLM Re conditioning** — embed log(Re) via small MLP → γ, β per block. Targets val_re_rand specifically.

15. **Per-block temperature init schedule** — instead of uniform init (all blocks at 2.0), give early blocks lower init (1.5) and later blocks higher (2.5) to encourage hierarchical sharpness.

## Disconfirmed directions (do not retry)

Per-channel surface loss weighting, per-channel output heads, depth-8, balanced capacity scale-up, max_norm=1.0 grad clipping under MSE, Fourier Re embedding standalone, activation choice (GELU vs SiLU), per-node Gaussian feature noise (semantics-unaware), SwiGLU preprocess MLP at input projection, huber δ=2.0, SwiGLU output head (no residual buffer), torch.compile mode="reduce-overhead" with variable-shape padding (OOMs from per-shape CUDA Graph accumulation; revisit only with fixed-shape padding or bucketing), EMA decay_target > 0.995 at warmup_steps=50 (cap doesn't bind within 14-epoch budget; trajectories are mathematically identical for any decay_target ≥ 0.995).

## Constraints / guardrails

- Branch: `icml-appendix-charlie-pai2d-r2`
- Local JSONL metric logging only. No W&B / wandb / Weave.
- Do not override `SENPAI_TIMEOUT_MINUTES` or `--epochs`.
- Read-only: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, `data/generate_manifest.py`, `data/split_manifest.json`.
- Experiment edits live in `train.py` only.
