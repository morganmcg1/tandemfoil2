# SENPAI Research State

- **Date:** 2026-04-27 23:25
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet (this is a fresh launch)
- **Cross-cutting concern surfaced by #331:** `data/scoring.py` skips on non-finite *ground truth* but not non-finite *predictions*; a single overflowing pred can NaN-poison `test_avg/mae_surf_p`. `data/` is read-only by program contract, so the defensive guard lives in `train.py` (`torch.nan_to_num` before `accumulate_batch`). Worth checking whether other in-flight PRs hit the same tail.

## Current research focus

Round 1 — establish baseline calibration and probe orthogonal levers on the default Transolver:

1. **Loss reweighting** — push capacity toward the surface-pressure metric that ranks runs.
2. **Capacity scaling** — width, depth, slice tokenization.
3. **Training dynamics** — learning rate schedule with warmup, batch size.
4. **Eval improvement** — EMA weight averaging.

Each axis is tested independently in this round so we can attribute gains cleanly. Winners on independent axes are then candidates for compounding in round 2.

## Round 1 assignments

| PR | Student | Hypothesis | Axis |
|----|---------|------------|------|
| #329 | alphonse  | Surface-loss weight sweep (`surf_weight ∈ {20, 30, 50}`)        | Loss  |
| #331 | askeladd  | Wider Transolver (`n_hidden 128→192`, `n_head 4→6`)             | Capacity |
| #334 | edward    | Deeper Transolver (`n_layers 5→8`)                              | Capacity |
| #336 | fern      | More physics slices (`slice_num 64→128`)                        | Capacity |
| #338 | frieren   | LR warmup + peak `1e-3` then cosine to 0                        | Schedule |
| #339 | nezuko    | Larger batch (`batch_size 4→8`) with √2 LR scale                | Dynamics |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | Loss  |
| #341 | thorfinn  | EMA model weights for val/test (decay 0.999)                    | Eval  |

## Potential next research directions (round 2+)

Once round 1 results are in, the most likely directions are:

- **Compound the orthogonal winners** — e.g. wider + EMA + tuned surf_weight stacked.
- **Loss reformulation** — Huber/SmoothL1 instead of MSE for surface (robustness to high-Re extreme pressures).
- **Modern transformer ergonomics** — SwiGLU activation, stochastic depth, RMSNorm.
- **Spatial inductive bias** — Fourier features on `(x, z)` coordinates fed into the preprocess MLP.
- **Mixed precision** — bf16 to free VRAM/time for wider/deeper models inside the 30-min cap.
- **Multi-scale slice tokenization** — different `slice_num` per layer to capture both global and local physics.
- **Per-domain calibration heads** — separate output projections for racecar-single / tandem / cruise picked by routing on the geometry features.
- **Boundary-layer-aware sampling** — over-sample high-Re extremes during training since they dominate the metric.

## Notes

- `BASELINE.md` reference numbers are TBD until first round runs land.
- `EXPERIMENTS_LOG.md` tracks per-PR results once they arrive.
