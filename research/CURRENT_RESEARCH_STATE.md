# SENPAI Research State

- **Last update:** 2026-04-28 01:25 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Current baseline (merged): `val_avg/mae_surf_p = 88.227`, `test_avg/mae_surf_p = 78.338`** (PR #391 SwiGLU FFN).
  - PR #282 — Huber loss (δ=1.0). val_avg = 105.999.
  - PR #361 — NaN-safe `evaluate_split` workaround. First finite test_avg = 97.957.
  - PR #363 — EMA(decay 0.999) of weights for val/test + checkpoint. val_avg = 101.350 (−4.39% vs huber).
  - PR #391 — LLaMA-style SwiGLU FFN inside `TransolverBlock`. **val_avg = 88.227 (−12.95% vs EMA), test_avg = 78.338 (−20.03% vs PR #361 finite measurement).** Param-matched (+1.3%).

## Current research focus

Compound improvements on the round-1 huber baseline. Recover the paper-facing test metric. Test orthogonal levers (capacity, slice count, optimizer recipe, surface weighting, regularization, EMA, channel weighting) so round-3 can stack winners.

## Outcomes to date (9 reviewed)

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | Δ vs 88.227 (current) | Decision |
|------|----|---------|------|--------------------------:|----------------------:|----------|
| 1 | #391 | thorfinn | swiglu-mlp | **88.227** | (current baseline, MERGED) | MERGED |
| 2 | #363 | thorfinn | ema-eval | 101.350 | +14.9% | MERGED (intermediate baseline) |
| 3 | #282 | edward | huber-loss | 105.999 | +20.1% | MERGED (huber baseline) |
| 3b | #361 | edward | nan-safe-eval | 108.103 (rerun) | n/a — RNG noise on same recipe | MERGED (metric-pipeline fix; first finite test_avg) |
| 4 | #370 | askeladd | cosine-tmax-14 | 102.359 | +16.0% | CLOSED (T_max ↔ EMA non-additive) |
| 5 | #362 | tanjiro | surf-channel-on-huber | 107.920 | +22.3% | CLOSED (channel-weight on huber dead direction) |
| 6 | #286 | frieren | surf-weight-25 | 108.222 | +22.7% | CLOSED |
| 7 | #392 | frieren | mlp-ratio-4 | 108.558 | +23.0% | CLOSED (per-split contradicts capacity-bottleneck) |
| 8 | #386 | edward | re-fourier-8 | 109.131 | +23.7% | CLOSED (high-freq aliasing; salvageable at narrower band — see #418) |
| 9 | #377 | fern | warmup-cosine-1e3-no-clip | 116.352 | +31.9% | CLOSED (T_max/budget mismatch, lr too hot) |
| 10 | #284 | fern | warmup-cosine-1e3 | 123.135 | +39.6% | CLOSED (clip masked recipe) |
| 11 | #291 | nezuko | dropout-0p1 | 128.896 | +46.1% | CLOSED |
| 12 | #295 | tanjiro | pressure-channel-weight | 130.916 | +48.4% | CLOSED |
| 13 | #281 | askeladd | slice-128 | 154.594 | +75.2% | CLOSED |
| 14 | #297 | thorfinn | depth-8 | 168.836 | +91.4% | CLOSED |

Per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`. Per-experiment JSONL summaries in `research/student_metrics/` (note: nezuko, askeladd & fern did not commit their training metrics files; their PR-comment numbers are recorded as JSONL summaries instead).


## In flight from earlier rounds (4 students)

These were branched **before** the SwiGLU merge — they will be ranked against the new SwiGLU baseline (88.227) when they return. A result < 88.227 wins; a result that beat its prior baseline but >88.227 means the lever helps on the older stack but hasn't compounded with SwiGLU.

| PR | Student | Slug | Lever | Predicted (vs base at submission) |
|----|---------|------|-------|------------------------------------|
| #279 | alphonse | capacity-medium | n_hidden 128→192, n_layers 5→6, n_head 4→6 — branched pre-huber | −5% to −12% |
| #371 | nezuko | grad-accum-2 | gradient accumulation 2 (effective batch 8) with √2 lr scaling — branched on huber pre-EMA | −1% to −4% |
| #411 | fern | huber-delta-2 | huber `δ=1.0 → 2.0` (smoother near optimum) — branched on EMA pre-SwiGLU | −1% to −3% |
| #412 | tanjiro | per-channel-heads | replace shared output `mlp2` with three per-channel heads — branched on EMA pre-SwiGLU | −2% to −4% |
| #418 | edward | re-fourier-4 | narrowed Fourier embedding of `log(Re)` (4 bands, max 2^3=8 rad/log_re_unit) — direct test of aliasing diagnosis from #386 | −1% to −4% |

## Round-4 just-assigned (3 students)

Built on the merged SwiGLU baseline (88.227). All three are single-axis tests on top of the new stack.

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #424 | thorfinn | swiglu-head | SwiGLU output head (`mlp2`) on top of merged SwiGLU FFN — aligns head's expressive form with the rest of the model | −0.5% to −2% |
| #425 | frieren | input-noise-001 | Input feature noise augmentation (Gaussian noise std=0.01 on fun-features only, train-time only) — targets the *generalization* bottleneck frieren correctly diagnosed (val_geom_camber_rc, val_re_rand) rather than capacity | −1% to −3% |
| #426 | askeladd | ema-decay-099 | EMA decay 0.999 → 0.99 (half-life ~0.18 epochs vs ~1.85 epochs) — student's follow-up #1 from PR #370; tests EMA decay sensitivity on the SwiGLU baseline | −0.5% to −2% (could regress) |

## Disconfirmed directions (do not retry on this branch)

- **Per-channel surface loss weighting toward `p`** — falsified across PR #295 (`[1,1,2.5]`, +23.5%) and PR #362 (`[0.5,0.5,2.5]`, +1.81%). Mechanism works (Ux/Uy degraded relatively more than `p`) but absolute `mae_surf_p` got worse in both. Move on.
- **Pure depth scale at default budget** — PR #297 (`n_layers=8`) compute-infeasible at 30-min budget (9/50 epochs). Revisit only if the timeout changes or per-epoch throughput improves.
- **`max_norm=1.0` grad clipping under MSE on this problem** — PR #284 showed it clips 100% of batches with pre-clip mean 30–200, masking any other lever it's combined with. Always pair clipping decisions with the loss's actual gradient scale.

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
