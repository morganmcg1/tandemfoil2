# SENPAI Research State

- **Last update:** 2026-04-27 23:55 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Round-1 baseline (merged):** PR #282 — Huber loss (δ=1.0) on normalized targets. `val_avg/mae_surf_p = 105.999` at epoch 14. `test_avg/mae_surf_p = NaN` (scoring/data issue, fix in flight as PR #361).

## Current research focus

Compound improvements on the round-1 huber baseline. Recover the paper-facing test metric. Test orthogonal levers (capacity, slice count, optimizer recipe, surface weighting, regularization, EMA, channel weighting) so round-3 can stack winners.

## Round-1 outcomes (5 reviewed)

| Rank | PR | Student | Slug | best `val_avg/mae_surf_p` | Δ vs 105.999 | Decision |
|------|----|---------|------|--------------------------:|-------------:|----------|
| 1 | #282 | edward | huber-loss | **105.999** | (baseline) | **MERGED** |
| 2 | #291 | nezuko | dropout-0p1 | 128.896 | +21.6% | CLOSED |
| 3 | #295 | tanjiro | pressure-channel-weight | 130.916 | +23.5% | CLOSED |
| 4 | #281 | askeladd | slice-128 | 154.594 | +45.8% | CLOSED |
| 5 | #297 | thorfinn | depth-8 | 168.836 | +59.3% | CLOSED |

Per-experiment numbers in `research/EXPERIMENT_METRICS.jsonl`. Per-experiment JSONL summaries in `research/student_metrics/` (note: nezuko & askeladd did not commit their training metrics file; their PR-comment numbers are recorded as JSONL summaries instead).

## Round-1 still WIP (3 students)

These were branched off the **pre-huber** advisor and test isolated levers without huber. Their results will be evaluated against the new huber baseline (105.999) when they come back for review.

| PR | Student | Slug | Lever |
|----|---------|------|-------|
| #279 | alphonse | capacity-medium | n_hidden 128→192, n_layers 5→6, n_head 4→6 |
| #284 | fern | warmup-cosine-1e3 | linear-warmup 3ep → cosine, peak lr 1e-3, betas (0.9,0.95), grad clip 1.0 |
| #286 | frieren | surf-weight-25 | surf_weight 10→25 |

## Round-2 in flight (5 students)

All built on the merged huber baseline (PR #282).

| PR | Student | Slug | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|----|---------|------|-------|-------------------------------------|
| #361 | edward | nan-safe-eval | filter non-finite-`y` samples in `evaluate_split` before `accumulate_batch` | 0 (recovers `test_avg`, not `val_avg`) |
| #362 | tanjiro | surf-channel-on-huber | per-channel surf loss weights `[0.5, 0.5, 2.5]` on top of huber | −3% to −10% |
| #363 | thorfinn | ema-eval | EMA copy of weights (decay 0.999) for val/test evaluation | −2% to −5% |
| #370 | askeladd | cosine-tmax-14 | align cosine `T_max` with actual budget; LR fully decays during 30-min run | −3% to −8% |
| #371 | nezuko | grad-accum-2 | gradient accumulation 2 (effective batch 8) with √2 lr scaling | −1% to −4% |

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
