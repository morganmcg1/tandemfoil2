# SENPAI Research State

- **Last update:** 2026-04-28 04:15 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Current baseline (merged): conservative target `val_avg/mae_surf_p < 72.414`** (fern's pre-DropPath standalone measurement). Stack now includes 3 orthogonal compound levers: nezuko's β₂=0.95 (#480), askeladd's bias-corrected EMA (#479), and the prior huber/SwiGLU/DropPath/EMA stack. Combined-stack actual val_avg unmeasured.
  - PR #282 — Huber loss (δ=1.0). val_avg = 105.999.
  - PR #361 — NaN-safe `evaluate_split` workaround. First finite test_avg = 97.957.
  - PR #363 — EMA(decay 0.999). val_avg = 101.350.
  - PR #391 — LLaMA-style SwiGLU FFN. val_avg = 88.227. Param-matched.
  - PR #426 — EMA decay 0.999 → 0.99. val_avg = 83.223.
  - PR #455 — Stochastic depth (DropPath 0→0.1). val_avg = 80.480. Param-identical.
  - PR #463 — Huber δ=1.0 → 0.25. **val_avg = 72.414 (−13.0% vs EMA(0.99)+SwiGLU; −10.0% vs DropPath baseline). test_avg = 63.082.** All 4 val splits improved 10–18%. Cruise canary gained MOST (−17.88%). Largest single-PR delta of the programme.
  - PR #480 — AdamW betas (0.9, 0.999) → (0.9, 0.95). Standalone on EMA(0.99)+SwiGLU pre-DropPath: val_avg = 77.951 (−6.34%). Orthogonal to δ=0.25.
  - PR #479 — Bias-corrected EMA (decay_target=0.99, warmup_steps=10). Standalone on EMA(0.99)+SwiGLU pre-DropPath: val_avg = 81.251 (−2.37%). Cold-start gain transferred AND fast-tracking preserved. Strict superset of EMA(0.99). MERGED as compound.

## Current research focus

Compound improvements on the round-1 huber baseline. Recover the paper-facing test metric. Test orthogonal levers (capacity, slice count, optimizer recipe, surface weighting, regularization, EMA, channel weighting) so round-3 can stack winners.

## Outcomes to date (28 reviewed)

Sorted by val_avg ascending (best first). Δ column references the current 72.414 conservative target. Full per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`.

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | Δ | Decision |
|------|----|---------|------|--------------------------:|--:|----------|
| 1 | #463 | fern | huber-delta-025 | **72.414** | — | MERGED — biggest single-PR delta |
| 2 | #480 | nezuko | adamw-betas-095 | 77.951 (on EMA(0.99)+SwiGLU baseline) | +7.6% standalone | MERGED — orthogonal compound (β₂=0.95) |
| 3 | #455 | thorfinn | stochastic-depth-01 | 80.480 | +11.1% | MERGED (DropPath intermediate) |
| 4 | #460 | frieren | per-sample-feature-noise | 81.437 | +12.5% | CLOSED (diagnosis confirmed; doesn't beat current) |
| 5 | #426 | askeladd | ema-decay-099 | 83.223 | +14.9% | MERGED (intermediate) |
| 6 | #456 | edward | layerscale-1e4 | 83.544 | +15.4% | CLOSED |
| 7 | #488 | alphonse | rmsnorm-manual | 84.149 | +16.2% | CLOSED (eager-mode wall-clock issue; needs torch.compile) |
| 8 | #454 | askeladd | ema-bias-correction (0.999) | 84.645 | +16.9% | CLOSED |
| 9 | #439 | fern | huber-delta-05 | 87.265 | +20.5% | CLOSED |
| 10 | #440 | tanjiro | silu-everywhere | 88.128 | +21.7% | CLOSED (null result) |
| 11 | #391 | thorfinn | swiglu-mlp | 88.227 | +21.8% | MERGED (SwiGLU baseline) |
| 12 | #459 | tanjiro | swiglu-preprocess | 88.299 | +21.9% | CLOSED (SwiGLU at input prunes signal) |
| 13 | #425 | frieren | input-noise-001 | 89.984 | +24.2% | CLOSED (per-node noise broke per-sample globals) |
| 14 | #424 | thorfinn | swiglu-head | 90.298 | +24.7% | CLOSED (no residual buffer for the head) |
| 15 | #450 | alphonse | rmsnorm-everywhere (nn.RMSNorm) | 91.342 | +26.1% | CLOSED (ATen wall-clock penalty) |
| 16 | #363 | thorfinn | ema-eval | 101.350 | +39.9% | MERGED (intermediate) |
| 17 | #370 | askeladd | cosine-tmax-14 | 102.359 | +41.4% | CLOSED (T_max ↔ EMA non-additive) |
| 18 | #418 | edward | re-fourier-4 | 102.916 | +42.1% | CLOSED (Fourier-Re partial recovery) |
| 19 | #412 | tanjiro | per-channel-heads | 105.580 | +45.8% | CLOSED (capacity-in-head falsified) |
| 20 | #282 | edward | huber-loss | 105.999 | +46.4% | MERGED (huber baseline) |
| 20b | #361 | edward | nan-safe-eval | 108.103 (rerun) | n/a — RNG | MERGED (metric-pipeline fix) |
| 21 | #411 | fern | huber-delta-2 | 107.609 | +48.6% | CLOSED (δ profile peak at 2 falsified) |
| 22 | #362 | tanjiro | surf-channel-on-huber | 107.920 | +49.0% | CLOSED (channel-weight dead) |
| 23 | #286 | frieren | surf-weight-25 | 108.222 | +49.4% | CLOSED |
| 24 | #392 | frieren | mlp-ratio-4 | 108.558 | +49.9% | CLOSED |
| 25 | #386 | edward | re-fourier-8 | 109.131 | +50.7% | CLOSED (high-freq aliasing) |
| 26 | #377 | fern | warmup-cosine-1e3-no-clip | 116.352 | +60.7% | CLOSED |
| 27 | #284 | fern | warmup-cosine-1e3 | 123.135 | +70.0% | CLOSED (clip masked recipe) |
| 28 | #371 | nezuko | grad-accum-2 | 123.997 | +71.2% | CLOSED (halved step count under fixed wall-clock) |
| 29 | #291 | nezuko | dropout-0p1 | 128.896 | +78.0% | CLOSED |
| 30 | #295 | tanjiro | pressure-channel-weight | 130.916 | +80.8% | CLOSED |
| 31 | #279 | alphonse | capacity-medium | 142.446 | +96.7% | CLOSED (compute-infeasible) |
| 32 | #281 | askeladd | slice-128 | 154.594 | +113.5% | CLOSED |
| 33 | #297 | thorfinn | depth-8 | 168.836 | +133.2% | CLOSED |

Per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`. Per-experiment JSONL summaries in `research/student_metrics/` (note: nezuko, askeladd & fern did not commit their training metrics files; their PR-comment numbers are recorded as JSONL summaries instead).




## Round-6 in flight (5 students)

Built on the merged baseline. Conservative target val_avg < 72.414.

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #493 | fern | huber-delta-01 | Push δ profile further: 0.25 → 0.1 (saturation test) | −1% to −5% |
| #494 | tanjiro | weight-decay-3e-4 | AdamW weight_decay 1e-4 → 3e-4 (orthogonal regularization knob) | −0.5% to −1.5% |
| #495 | frieren | feature-noise-002 | Sweep semantics-aware feature noise std 0.01 → 0.02 | −0.5% to −2% |
| #510 | alphonse | torch-compile-baseline | Wrap model in `torch.compile(mode='reduce-overhead')` — speeds up baseline, opens door to RMSNorm fusion | −1% to −3% |
| #511 | nezuko | adamw-beta2-090 | AdamW betas (0.9, 0.95) → (0.9, 0.90) — push β₂ profile further | −0.5% to −1.5% (could regress) |

## Round-7 just-assigned (3 students)

Built on the merged baseline. Conservative target val_avg < 72.414.

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #518 | askeladd | bias-corrected-ema-warmup-50 | EMA warmup_steps 10 → 50 (longer cold-start ramp; her own follow-up #1 from #479) | −0.5% to −1.5% |
| #519 | edward | n-head-8 | Multi-head attention 4 → 8 (head_dim 32 → 16; param-matched parallel attention paths) | −0.5% to −1.5% |
| #520 | thorfinn | slice-temp-1p0 | PhysicsAttention temperature init 0.5 → 1.0 (default-sharpness softmax start; learnable param-matched) | −0.3% to −1.5% |

## Disconfirmed directions (do not retry on this branch)

- **Per-channel surface loss weighting toward `p`** — falsified across PR #295 (`[1,1,2.5]`, +23.5%) and PR #362 (`[0.5,0.5,2.5]`, +1.81%). Mechanism works (Ux/Uy degraded relatively more than `p`) but absolute `mae_surf_p` got worse in both. Move on.
- **Per-channel output heads** — falsified by PR #412 (+19.7% vs SwiGLU baseline; canary `val_geom_camber_rc` regressed most). Combined with the mlp_ratio=4 failure (PR #392), capacity in the head/FFN is not the bottleneck on this problem. Architectural form (SwiGLU) matters more than capacity.
- **Pure depth scale at default budget** — PR #297 (`n_layers=8`) compute-infeasible at 30-min budget (9/50 epochs). Revisit only if the timeout changes or per-epoch throughput improves.
- **Balanced capacity scale-up** — PR #279 (`n_hidden=192, n_layers=6, n_head=6`) compute-infeasible at 30-min budget (8/50 epochs at 240 s/epoch). Same shape as #297. Combined with the SwiGLU+param-matched win (#391), the lesson is that **architectural quality matters more than raw capacity** at this budget.
- **`max_norm=1.0` grad clipping under MSE on this problem** — PR #284 showed it clips 100% of batches with pre-clip mean 30–200, masking any other lever it's combined with. Always pair clipping decisions with the loss's actual gradient scale.
- **Huber `δ=2.0`** — PR #411 (+21.97% vs SwiGLU). At the high-error early-training regime we're stuck in, δ=2's quadratic region for moderate errors underweights the bulk. δ=1 is the sweet spot above; PR #439 testing whether δ<1 helps further.
- **SwiGLU output head (`mlp2`)** — PR #424 (+2.35% vs SwiGLU FFN). Head has no residual buffer (unlike per-block FFN which sits inside `+fx`), so SwiGLU's gating non-linearity acts unbuffered and the 3× param count amplifies non-generalizing directions. Direction not dead at residual-SwiGLU-head, but that's a future fix.
- **Fourier embedding of `log(Re)` standalone** — PRs #386 (bands=8, +23.7% vs SwiGLU) and #418 (bands=4, +16.6% vs SwiGLU). Real signal on `val_single_in_dist` (smooth low-freq Re-trend), but doesn't beat the SwiGLU lever. FiLM-style Re conditioning is a queued alternative.
- **Activation choice (GELU vs SiLU)** — PR #440 (null result, −0.11% on val within noise; +1.06% on test). Below the noise floor at this scale (0.67M params, 1499 train samples). Don't sweep activation again unless model size doubles.
- **Per-node Gaussian feature noise (uniform across all 24 dims)** — PR #425 (+1.76% vs SwiGLU; +8.1% vs current). Falsified because dims 13–23 are per-sample-constant globals (Re, AoA, NACA, gap, stagger) — per-node noise destroyed (geometry, flow conditions) → field map. **Semantics-aware version PR #460 confirmed the diagnosis (−2.15% vs EMA(0.99)+SwiGLU).**
- **SwiGLU preprocess MLP (LLaMA-everywhere at the input projection)** — PR #459 (+6.1% vs EMA(0.99)+SwiGLU; +9.7% vs DropPath baseline). Per-block SwiGLU gates AFTER attention has mixed the residual stream (prunes redundancy in already-mixed representations); input-projection SwiGLU gates BEFORE mixing, so it prunes raw physical input channels (Re, AoA, NACA, etc.) that the network needs later. **Gating at the input prunes signal, not redundancy.** Trajectory matched baseline through ep5 then diverged as fine-detail learning kicked in. Don't extend SwiGLU to the input projection.

## Test-metric NaN (cross-PR issue)

All round-1 runs reported `test_avg/mae_surf_p = NaN`. Root cause:
1. `test_geom_camber_cruise` sample 20 has 761 non-finite values in `y[p]` volume nodes.
2. `data/scoring.py:accumulate_batch` is documented to skip samples with non-finite `y`, but computes `err = (pred - y).abs()` *before* applying the mask. IEEE 754 then propagates `Inf * 0 = NaN` to the per-channel sum.
3. `data/scoring.py` is read-only per `program.md`, so the workaround lives in `train.py:evaluate_split` (filter samples with non-finite `y` before calling `accumulate_batch`). PR #361 is the carrier.

After PR #361 lands, `test_avg/mae_surf_p` becomes a recoverable paper-facing metric for all subsequent rounds.

## Potential next research directions (round 3+)

Pending the still-WIP round-1 PRs and the round-2 results, the most promising compound directions:

1. **Stack winners.** If capacity scaling (alphonse) + warmup-cosine (fern) + surf_weight (frieren) all individually beat huber, combine them as a stacked PR.
2. **Sweep δ in Huber** ({0.5, 1.0, 2.0}) — δ=1.0 in normalized space already saturates linear, so δ=2.0 (closer to MSE near optimum) may be a Pareto improvement.
3. **Time-aware cosine T_max** — set T_max to actual-epochs-fitting-in-budget rather than `cfg.epochs` so LR fully decays. With ~14 epochs reachable at default model size, T_max=14 (or `min(cfg.epochs, expected_epochs_in_budget)`).
4. **Per-Re or per-domain feature embedding.** Three physical domains have very different y ranges; sinusoidal embedding of `log Re` (or a learned domain bias) could help re_rand and cruise-camber splits.
5. **Per-channel output heads.** Currently the last layer projects `[Ux, Uy, p]` jointly. Splitting into per-channel heads (or a separate surface-only branch fed by surface-aware features) would let the model specialize for the headline `p` channel.
6. **Geometry-aware features.** Add a per-node distance-to-surface field and a normal-vector encoding to help surface pressure prediction.
7. **Larger batch + grad accumulation.** Bigger effective batch + EMA may further reduce the late-training validation noise observed in round 1.
8. **Activation/norm variants.** SwiGLU, RMSNorm, pre-norm + LayerScale — small architectural tweaks frequently helpful for transformer-style models.

## Constraints / guardrails (this replicate)

- Branch: `icml-appendix-charlie-pai2d-r2` (PRs target it, branches off it, merges squash into it).
- Local JSONL metric logging only. **No W&B / wandb / Weave anywhere.**
- Do not override `SENPAI_TIMEOUT_MINUTES` or `--epochs` in any experiment.
- Read-only: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, `data/generate_manifest.py`, `data/split_manifest.json`. Experiment edits live in `train.py` (and `pyproject.toml` if a new package is genuinely needed).
- Isolated replicate: do not reference / compare against / inspect prior launches or sibling advisor branches.
