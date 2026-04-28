# SENPAI Research State

- **Last update:** 2026-04-28 10:45 (advisor branch `icml-appendix-charlie-pai2d-r2`)
- **Most recent human-team direction:** N/A — no open human-tagged issues at this time.
- **Current baseline (directly measured, all standalone)**: `val_avg/mae_surf_p` = **`58.742`** (PR #698 eta_min=5e-5, BEST — direct compound on PR #630) / `59.907` (PR #630 eta_min=2e-5) / `60.5905` (PR #661 TF32 infra) / `61.245` (PR #696 surface-only noise) / `61.872` (PR #647 per-block temp) / `62.747` (PR #640 per-group wd) / `62.879` (PR #601 δ=0.1) / `63.131` (PR #635 lr=6e-4) / `63.222` (PR #636 decaying noise). Test_avg = 50.789 / 52.656 / 53.404 / 53.605 / 54.555 / 54.512 / 54.561 / 55.026 / 54.900.
- **Stack throughput**: 17-18 epochs in 30-min budget under compile=True. Cosine T_max=11 → eta_min=0 at ep15, then cosine cycles back from ep16+.

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
21. PR #510 — torch.compile mode="default" (infrastructure: +28.6% epochs / −23.1% wall-clock). val_avg = 64.824 at 18 epochs.
22. PR #635 — lr peak bump 5e-4 → 6e-4. val_avg = 63.131. test_avg = 55.026.
23. PR #636 — Decaying feature-noise schedule (linear decay 0.0025→0 over 14 ep). val_avg = 63.222. test_avg = 54.900.
24. PR #640 — Per-parameter-group weight decay (attn=1e-4, mlp=1e-5, other=3e-5). val_avg = 62.747. test_avg = 54.512.
25. PR #601 — Huber δ=0.25 → 0.10 (rebased on post-#562/#510 stack). val_avg = 62.879. test_avg = 54.561.
26. PR #647 — Per-block slice-temp init schedule [1.5, 1.875, 2.25, 2.625, 3.0]. val_avg = 61.872. test_avg = 54.555.
27. PR #630 — Cosine eta_min 0 → 2e-5 (with periodic-rebound mechanism). val_avg = 59.907. test_avg = 52.656.
28. PR #696 — Surface-only feature noise (dims 0-12 only, skip per-sample globals). val_avg = 61.245. test_avg = 53.605.
29. PR #661 — TF32 matmul precision (infrastructure: +16.7% epochs in budget). val_avg = 60.5905 standalone (on PR #510 baseline). 21 epochs / 86.8 s/epoch.
30. **PR #698 — Cosine eta_min 2e-5 → 5e-5 (descent-phase mechanism). val_avg = 58.742. test_avg = 50.789. Direct compound on PR #630. CURRENT BASELINE.**

## Active experiments (WIP)

| PR | Student | Slug | Lever | Status |
|----|---------|------|-------|--------|
| #722 | alphonse | bf16-autocast | BF16 autocast forward pass (next throughput rung past TF32) | WIP (just assigned) |
| #697 | edward | cosine-tmax-14 | Cosine T_max 11 → 14 (sent back: rebase onto post-#630 stack, T_max × eta_min interaction) | WIP (sent back, rebase) |
| #708 | askeladd | slice-temp-per-block-cruise-anchor | Cruise-friendly anchor [1.0, 1.75, 2.5, 3.25, 4.0] (range=3.0, mean=2.5) | WIP (just assigned) |
| #646 | fern | batch-size-6 | batch_size 4 → 6 with compile (rebase onto post-#647 stack) | WIP (sent back, rebase) |
| #673 | tanjiro | per-group-wd-extreme | wd_attn 1e-4→3e-4, wd_mlp 1e-5→3e-6 (sent back: rebase onto post-#630 stack) | WIP (sent back, rebase) |
| #719 | frieren | feature-noise-surface-only-005 | Surface-only base_std 0.0025 → 0.005 (push magnitude under clean per-node arm) | WIP (just assigned) |
| #726 | nezuko | cosine-eta-min-1e-4 | cosine eta_min 5e-5 → 1e-4 (push profile, bracket from above) | WIP (just assigned) |
| #709 | thorfinn | huber-per-channel-delta | Per-channel δ (δ_p=0.05, δ_Ux=δ_Uy=0.10) | WIP (just assigned) |

## Current research focus

**Hyperparameter closure + profile extension on multiple active axes.** The merged stack now includes 21 improvements (latest: PR #510 compile = +28.6% epochs in budget compounds with everything); we are bracketing the remaining open directions:

1. **Cosine eta_min push** (nezuko #726, 1e-4): PR #698 just merged with -1.94% gain (val=58.742). Profile (0/2e-5/5e-5) is monotone-improving with diminishing returns. Mechanism shifted from rebound (PR #630) to descent phase (PR #698). First soft signal of overshoot at eta_min=5e-5 (ep18 slightly worse than ep17). Pushing 1e-4 brackets from above.
2. **Per-channel huber δ** (thorfinn #709): scalar δ axis saturated at 0.10 (PR #674); the `p` channel has lowest lin fraction (37% at thr=0.05). Apply δ_p=0.05 + δ_Ux=δ_Uy=0.10 to focus outlier downweighting where it matters most.
3. **Surface-only noise magnitude push** (frieren #719, base_std=0.005): PR #696 just merged with -1.01% gain. With per-sample destabilizing arm removed, the magnitude ceiling that capped PR #669 (90% clip rate at full 0.005) may have lifted under clean per-node arm.
4. **Per-parameter-group wd** (tanjiro #640): single-scalar wd axis is closed at 3e-5; explore module-type-differential wd to capture the OOD asymmetry (attn higher to help camber_rc, mlp lower to help re_rand).
5. **Batch-size gradient quality** (fern #646, batch=6 with compile): gradient noise reduction may compound with EMA averaging. Replaces closed warmup-aggressiveness axis.
6. **Cosine T_max alignment** (edward #697, T_max=14): align cosine with realized 18-epoch compile budget. Currently cosine ends at ep14 then wraps; T_max=14 gives clean monotone descent through ep17.
7. **Per-block cruise-friendly anchor** (askeladd #708, [1.0, 1.75, 2.5, 3.25, 4.0]): combine block-0=1.0 (cruise benefit) with mean=2.5 (higher than #647's 2.25) and range=3.0 (wider). Disambiguates variance-vs-mean per PR #682's confound finding.
8. **BF16 autocast forward pass** (alphonse #722): TF32 just merged with -15.1% per-iter; matmul fraction is only ~50-60%. BF16 autocast accelerates ALL BF16-safe ops (linear, attention, GELU), not just matmul. Expected 30-50% additional throughput.

**Closed axes**: EMA decay_target above 0.995 at warmup_steps=50 (cap doesn't bind within budget — PR #600); feature_noise_std (interior min at 0.0025, U-shaped — PR #595); surf_weight at 15 on huber-clip stack (clip absorbs the increase, single_in_dist vol_p degrades — PR #605); single-scalar wd (basin floor at 3e-5 on new stack; wd=0 regresses +2.75% — PR #554); LinearLR start_factor (sweet spot at 0.3, both 0.5 and 0.2 regress — PR #620); global slice-temp init (saturating at 2.0, camber_rc consistently regresses with sharper attention — PR #608); torch.compile reduce-overhead with naive fixed-shape padding (compute-bound at max-mesh shape, throughput regressed −22% — PR #629; bucketed batching is the right next probe); noise schedule magnitude above 0.0025 (base_std=0.005 destabilizes basin selection at peak LR via 90% clip rate — PR #669); lr peak above 6e-4 (clip not the bottleneck; high-LR phase doesn't extract more under merged regime — PR #668); scalar Huber δ below 0.10 (profile saturated, -0.19% step at 0.10→0.05 vs -3.00% at 0.25→0.10; converged residual distribution is invariant to δ — PR #674); per-block schedule range without higher mean (variance with lower mean costs val_avg; mean drop is load-bearing cost — PR #682).

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
