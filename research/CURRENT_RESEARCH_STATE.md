# SENPAI Research State

- **Last update:** 2026-04-28 01:00 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Current baseline (merged): `val_avg/mae_surf_p = 101.350`** (PR #363 EMA-eval).
  - PR #282 — Huber loss (δ=1.0) on normalized targets. `val_avg = 105.999`.
  - PR #361 — NaN-safe eval workaround in `evaluate_split`. `test_avg = 97.957` (first finite measurement).
  - PR #363 — EMA copy of weights (decay 0.999) for val/test eval. `val_avg = 101.350` (−4.39%); test_avg pending finite re-measurement post-merge (3-split mean was 100.03 with cruise NaN since #361 hadn't landed for that run).

## Current research focus

Compound improvements on the round-1 huber baseline. Recover the paper-facing test metric. Test orthogonal levers (capacity, slice count, optimizer recipe, surface weighting, regularization, EMA, channel weighting) so round-3 can stack winners.

## Round-1 outcomes (6 reviewed)

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | Δ vs 105.999 | Decision |
|------|----|---------|------|--------------------------:|-------------:|----------|
| 1 | #363 | thorfinn | ema-eval | **101.350** | −4.39% (current baseline) | **MERGED** |
| 2 | #282 | edward | huber-loss | 105.999 | (huber baseline) | **MERGED** |
| 2b | #361 | edward | nan-safe-eval | 108.103 (rerun) | +1.99% RNG noise | **MERGED** (metric-pipeline fix; first finite `test_avg = 97.957`) |
| 3 | #362 | tanjiro | surf-channel-on-huber [0.5,0.5,2.5] | 107.920 | +1.81% (vs 105.999) | CLOSED (channel-weight on huber, dead direction) |
| 4 | #286 | frieren | surf-weight-25 | 108.222 | +2.10% (vs 105.999) | CLOSED |
| 5 | #377 | fern | warmup-cosine-1e3-no-clip | 116.352 | +9.8% (vs 105.999) | CLOSED (T_max/budget mismatch, lr too hot) |
| 6 | #284 | fern | warmup-cosine-1e3 | 123.135 | +16.2% | CLOSED (clip masked recipe) |
| 7 | #291 | nezuko | dropout-0p1 | 128.896 | +21.6% | CLOSED |
| 8 | #295 | tanjiro | pressure-channel-weight | 130.916 | +23.5% | CLOSED |
| 9 | #281 | askeladd | slice-128 | 154.594 | +45.8% | CLOSED |
| 10 | #297 | thorfinn | depth-8 | 168.836 | +59.3% | CLOSED |

Per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`. Per-experiment JSONL summaries in `research/student_metrics/` (note: nezuko, askeladd & fern did not commit their training metrics files; their PR-comment numbers are recorded as JSONL summaries instead).

## Round-1 still WIP (1 student)

| PR | Student | Slug | Lever |
|----|---------|------|-------|
| #279 | alphonse | capacity-medium | n_hidden 128→192, n_layers 5→6, n_head 4→6 (branched off pre-huber; will be ranked against the EMA baseline 101.350) |

## Round-2 in flight (3 students)

Branched off huber baseline (PR #282 + #361) **before** the EMA merge (#363). When they return, their absolute val_avg compares against the new EMA baseline (101.350); a result < 101.350 wins; a result < 105.999 but > 101.350 helps on huber-no-EMA but doesn't beat huber+EMA, and would need a follow-up to claim a compound win.

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #370 | askeladd | cosine-tmax-14 | align cosine `T_max` with actual budget; LR fully decays during 30-min run | −3% to −8% |
| #371 | nezuko | grad-accum-2 | gradient accumulation 2 (effective batch 8) with √2 lr scaling | −1% to −4% |
| #386 | edward | re-fourier-8 | Fourier embedding of `log(Re)` (8 bands → 16 dims) concatenated to input features inside the model | −2% to −5% (with disproportionate help on `val_re_rand` and `val_single_in_dist`) |

## Round-3 in flight (4 students)

Built on the merged EMA+huber+NaN-safe baseline (101.350).

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #391 | thorfinn | swiglu-mlp | LLaMA-style SwiGLU FFN inside `TransolverBlock` (replaces 2-linear GELU MLP); per-block gating | −1% to −3% |
| #392 | frieren | mlp-ratio-4 | Standard transformer FFN expansion: `mlp_ratio` 2→4 (MLP hidden 256→512); pure capacity in the per-token mixing layer | −2% to −5% |
| #411 | fern | huber-delta-2 | Switch huber loss `δ=1.0 → 2.0` (smoother near optimum, MSE-like for typical errors, linear for outliers) | −1% to −3% |
| #412 | tanjiro | per-channel-heads | Replace shared 3-channel output head with three separate per-channel heads (orthogonal to channel-loss-weighting, which is now disconfirmed) | −2% to −4% |

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
