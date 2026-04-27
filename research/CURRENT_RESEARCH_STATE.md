# SENPAI Research State

- **Last update:** 2026-04-27 (advisor branch `icml-appendix-charlie-pai2d-r2`, fresh isolated replicate)
- **Most recent human-team direction:** N/A — no team issues consulted (isolated replicate; only entrypoint-surfaced PRs in scope).
- **Current research focus:**
  - Establish a strong round-1 baseline for `val_avg/mae_surf_p` / `test_avg/mae_surf_p` on the TandemFoilSet `splits_v2` (4 val × 100, 4 test × 200) layout, using a Transolver surrogate.
  - Probe orthogonal levers in parallel across 8 students so round 2 can compound the winners.

## Round 1 hypothesis families (8 students, all WIP)

| # | Student | Slug | PR | Lever | Predicted Δ on `val_avg/mae_surf_p` |
|---|---------|------|----|-------|-------------------------------------|
| 1 | alphonse | capacity-medium | #279 | n_hidden 128→192, n_layers 5→6, n_head 4→6 | −5% to −12% |
| 2 | askeladd | slice-128 | #281 | slice_num 64→128 | −3% to −8% |
| 3 | edward   | huber-loss | #282 | MSE → Huber(δ=1.0) on normalized targets | −2% to −6% |
| 4 | fern     | warmup-cosine-1e3 | #284 | linear-warmup 3ep → cosine, peak lr 1e-3, betas (0.9,0.95), grad clip 1.0 | −3% to −10% |
| 5 | frieren  | surf-weight-25 | #286 | surf_weight 10→25 | −3% to −10% |
| 6 | nezuko   | dropout-0p1 | #291 | dropout 0→0.1 in attention + MLP | −2% to −6% |
| 7 | tanjiro  | pressure-channel-weight | #295 | per-channel surface loss weights [1.0, 1.0, 2.5] (Ux, Uy, p) | −5% to −12% |
| 8 | thorfinn | depth-8 | #297 | n_layers 5→8 (no other change) | −3% to −8% |

Each PR is required to commit `models/<exp>/metrics.jsonl` and `metrics.yaml` and report best `val_avg/mae_surf_p`, `test_avg/mae_surf_p`, and per-split surface MAE (Ux/Uy/p where applicable).

## Hypothesis themes covered

- **Capacity scaling** — width+depth+heads (alphonse), slice tokens (askeladd), depth alone (thorfinn). These probe whether the default `1M-param Transolver` is undersized for ~1500 train samples × 74K-242K nodes.
- **Loss reformulation** — robust loss (edward: Huber), surface emphasis (frieren: surf_weight), channel-targeted emphasis (tanjiro: per-channel weights on `p`). All target the headline metric directly.
- **Optimizer recipe** — fern: warmup + higher peak LR + transformer-style betas + grad clip.
- **Regularization** — nezuko: dropout for OOD geometry generalization.

## Potential next research directions (round 2)

Pending round-1 results, the most promising compound directions:

1. **Stack the orthogonal winners.** If capacity scaling + surf_weight + warmup-cosine all help individually, combine them. Each compounded gain is what we want over a published baseline.
2. **Per-domain or per-Re calibration.** Train sees 3 unequal physical domains (single-foil 599, raceCar tandem 457, cruise tandem 443); per-domain output scaling/heads, or a Re-aware embedding (sinusoidal of `log Re`), could help cruise-camber and re_rand splits.
3. **Geometry-aware features.** Currently `is_surface` is the only surface signal. Adding a per-node distance-to-surface field (or signed distance via `dsdf` re-derived) and a normal vector encoding could help surface pressure.
4. **Slice initialization / routing.** The `in_project_slice` orthogonal init is fine, but slice "softmax temperature" `temperature=0.5` is fixed. A learned per-block schedule, or initialization tied to mesh-region heuristics, may give better partitioning.
5. **Larger effective batch + EMA.** EMA of weights for evaluation often helps transformer-style architectures; combined with grad accumulation it stabilizes the metric trajectory.
6. **Loss/metric matching.** If MSE-Huber doesn't beat MSE, try L1 on normalized space directly — closest to the eval metric.
7. **Output head architecture.** Currently all 3 channels share the final MLP head. Per-channel heads (or a separate surface-only branch fed by surface-aware features) would let the model specialize for `p` near the foil.
8. **Activation / norm variants.** SwiGLU MLPs, RMSNorm, or pre-norm LayerScale (γ=1e-4) — small architectural tweaks that have helped transformer-class models.

## Constraints / guardrails (this replicate)

- Branch: `icml-appendix-charlie-pai2d-r2` (PRs target it, branches off it, merges squash into it).
- Local JSONL metric logging only. **No W&B / wandb / Weave anywhere.**
- Do not override `SENPAI_TIMEOUT_MINUTES` or `--epochs` in any experiment.
- Read-only: `data/loader.py`, `data/scoring.py`, `data/prepare_splits.py`, `data/generate_manifest.py`, `data/split_manifest.json`. All experiment edits live in `train.py` (and `pyproject.toml` if a new package is genuinely needed).
