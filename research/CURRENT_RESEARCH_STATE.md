# SENPAI Research State

- **Last update:** 2026-04-28 02:40 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Current baseline (merged): `val_avg/mae_surf_p = 83.223`, `test_avg/mae_surf_p = 73.904`** (PR #426 EMA(0.99) on SwiGLU baseline).
  - PR #282 — Huber loss (δ=1.0). val_avg = 105.999.
  - PR #361 — NaN-safe `evaluate_split` workaround. First finite test_avg = 97.957.
  - PR #363 — EMA(decay 0.999). val_avg = 101.350 (−4.39% vs huber).
  - PR #391 — LLaMA-style SwiGLU FFN inside `TransolverBlock`. val_avg = 88.227 (−12.95% vs EMA). Param-matched.
  - PR #426 — EMA decay 0.999 → 0.99 (shorter half-life). **val_avg = 83.223 (−5.67% vs SwiGLU), test_avg = 73.904 (−5.66%).** All 4 val + test splits improved uniformly. Mechanism: EMA bias correction during the under-trained cold start.

## Current research focus

Compound improvements on the round-1 huber baseline. Recover the paper-facing test metric. Test orthogonal levers (capacity, slice count, optimizer recipe, surface weighting, regularization, EMA, channel weighting) so round-3 can stack winners.

## Outcomes to date (18 reviewed)

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | Δ vs 83.223 (current) | Decision |
|------|----|---------|------|--------------------------:|----------------------:|----------|
| 1 | #426 | askeladd | ema-decay-099 | **83.223** | (current baseline, MERGED) | MERGED |
| 2 | #391 | thorfinn | swiglu-mlp | 88.227 | +6.0% | MERGED (SwiGLU baseline) |
| 3 | #363 | thorfinn | ema-eval | 101.350 | +21.8% | MERGED (intermediate) |
| 4 | #282 | edward | huber-loss | 105.999 | +27.4% | MERGED (huber baseline) |
| 4b | #361 | edward | nan-safe-eval | 108.103 (rerun) | n/a — RNG noise | MERGED (metric-pipeline fix) |
| 4c | #439 | fern | huber-delta-05 | 87.265 | +4.9% | CLOSED (-1.1% vs SwiGLU baseline 88.227, but +4.9% vs current; δ profile shows diminishing returns) |
| 5 | #440 | tanjiro | silu-everywhere | 88.128 | +5.9% | CLOSED (null result; activation choice below noise floor) |
| 5b | #424 | thorfinn | swiglu-head | 90.298 | +8.5% | CLOSED (head SwiGLU lacks residual buffer; OOD splits regress) |
| 5c | #425 | frieren | input-noise-001 | 89.984 | +8.1% | CLOSED (per-node noise broke per-sample feature consistency on dims 13–23) |
| 6 | #370 | askeladd | cosine-tmax-14 | 102.359 | +23.0% | CLOSED (T_max ↔ EMA non-additive) |
| 7 | #412 | tanjiro | per-channel-heads | 105.580 | +26.9% | CLOSED (capacity-in-head falsified) |
| 7b | #411 | fern | huber-delta-2 | 107.609 | +29.3% | CLOSED (δ=1 sweet spot) |
| 8 | #362 | tanjiro | surf-channel-on-huber | 107.920 | +29.7% | CLOSED (channel-weight dead direction) |
| 9 | #286 | frieren | surf-weight-25 | 108.222 | +30.0% | CLOSED |
| 10 | #392 | frieren | mlp-ratio-4 | 108.558 | +30.4% | CLOSED |
| 11 | #386 | edward | re-fourier-8 | 109.131 | +31.1% | CLOSED (high-freq aliasing) |
| 12 | #418 | edward | re-fourier-4 | 102.916 | +23.7% | CLOSED (Fourier-Re partial recovery; doesn't beat SwiGLU) |
| 13 | #377 | fern | warmup-cosine-1e3-no-clip | 116.352 | +39.8% | CLOSED |
| 14 | #284 | fern | warmup-cosine-1e3 | 123.135 | +47.9% | CLOSED |
| 15 | #291 | nezuko | dropout-0p1 | 128.896 | +54.9% | CLOSED |
| 16 | #295 | tanjiro | pressure-channel-weight | 130.916 | +57.3% | CLOSED |
| 17 | #279 | alphonse | capacity-medium | 142.446 | +71.2% | CLOSED (compute-infeasible) |
| 18 | #281 | askeladd | slice-128 | 154.594 | +85.8% | CLOSED |
| 19 | #297 | thorfinn | depth-8 | 168.836 | +102.9% | CLOSED |

Per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`. Per-experiment JSONL summaries in `research/student_metrics/` (note: nezuko, askeladd & fern did not commit their training metrics files; their PR-comment numbers are recorded as JSONL summaries instead).


## In flight from earlier rounds (1 student)

| PR | Student | Slug | Lever | Predicted (vs base at submission) |
|----|---------|------|-------|------------------------------------|
| #371 | nezuko | grad-accum-2 | gradient accumulation 2 (effective batch 8) with √2 lr scaling — branched on huber pre-EMA, will be ranked against the EMA(0.99)+SwiGLU baseline (83.223) when it returns | −1% to −4% |

## Round-4 in flight (1 student)

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #450 | alphonse | rmsnorm-everywhere | replace `nn.LayerNorm` with `RMSNorm` in all 3 norm sites in `TransolverBlock` (branched on SwiGLU pre-EMA(0.99); will be ranked against current 83.223) | −0.5% to −1.5% |

## Round-5 in flight (6 students)

Built on the merged EMA(0.99)+SwiGLU baseline (83.223).

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #454 | askeladd | ema-bias-correction | Adam-style EMA bias correction: `decay_t = min(0.999, (1+t)/(10+t))` — keeps decay=0.999 asymptotic but ramps from cold start | −1% to −3% |
| #455 | thorfinn | stochastic-depth-01 | Stochastic depth (DropPath) with linear schedule 0 → 0.1 across 5 blocks | −1% to −3% |
| #456 | edward | layerscale-1e4 | CaiT-style LayerScale: per-branch scalar gates initialized to 1e-4 | −1% to −2% |
| #459 | tanjiro | swiglu-preprocess | replace preprocess MLP with `SwiGLU_MLP` (LLaMA-everywhere completion); per-token gating at the input projection | −0.5% to −2% |
| #460 | frieren | per-sample-feature-noise | semantics-aware noise: per-node on dims 0–12 (positions, saf, dsdf, is_surface) + per-sample broadcast on dims 13–23 (per-sample globals: Re, AoA, NACA, gap, stagger). Direct correction of PR #425's failure mode | −1% to −3% |
| #463 | fern | huber-delta-025 | huber `δ → 0.25` on EMA(0.99)+SwiGLU baseline. Tests whether the monotone δ profile (2→1→0.5: 107.6 → 88.2 → 87.3) saturates or keeps improving toward L1, and whether δ=0.25 stacks with EMA(0.99) | −0.5% to −1.5% (could regress if profile saturated) |

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
- **Per-node Gaussian feature noise (uniform across all 24 dims)** — PR #425 (+1.76% vs SwiGLU; +8.1% vs current). Falsified because dims 13–23 are per-sample-constant globals (Re, AoA, NACA, gap, stagger) — per-node noise destroyed (geometry, flow conditions) → field map. **Semantics-aware version queued as PR #460.**

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
