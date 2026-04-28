# SENPAI Research State

- **Date:** 2026-04-28 00:48
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet
- **Empirical baseline (round 1):** `val_avg/mae_surf_p = 139.83` from PR #336 (slice_num=128). All future runs compound on top of this.
- **Cross-cutting bug being fixed:** `data/scoring.py:accumulate_batch` propagates `NaN` through the per-sample-skip mask (`NaN * 0.0 = NaN`). Diagnosed by edward on PR #334; root cause is 761 NaN values in the `p` channel of `test_geom_camber_cruise/000020.pt`'s ground truth `y`. Fix in flight as PR #375 (edward) — advisor-authorized exception to the read-only contract on `data/`.

## Current research focus

Round 1 in progress. Strategy:

1. Independent axes tested first (one hypothesis per PR) to attribute gains cleanly.
2. Winners merge sequentially, best-first, each becoming the new baseline.
3. Round 2 compounds the orthogonal winners.

## In-flight PRs (status as of 2026-04-27 23:55)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #329 | alphonse  | Surface-loss weight sweep (`surf_weight ∈ {20, 30, 50}`)        | wip |
| #331 | askeladd  | Wider Transolver (`n_hidden 128→192`, `n_head 4→6`) + bf16/bs8  | wip (sent back; retry with bf16, bs=8, defensive nan_to_num) |
| #338 | frieren   | LR warmup + peak `1e-3` then cosine to 0                        | wip (sent back; control proved warmup wins at lr=5e-4 → 130.43, but branch needs rebase + Config.lr revert + one re-run on slice_num=128 to confirm composition) |
| #339 | nezuko    | Larger batch (`batch_size 4→8`) with √2 LR scale                | wip |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | wip |
| #341 | thorfinn  | EMA model weights for val/test (decay 0.999)                    | wip |
| #375 | edward    | Bugfix: nan_to_num in `data/scoring.py`                         | wip (sent back; fix is bit-exact correct, but branch needs rebase before squash-merge to drop reverts to BASELINE.md / research/*.md) |
| #405 | fern      | Spatial Fourier features (NeRF-style, L=8 octaves)              | wip (new; replaces #376) |

## Closed / merged

| PR | Student | Outcome |
|----|---------|---------|
| #334 | edward | Deeper (n_layers 5→8) — **closed**, clear regression vs slice_num=128 |
| #336 | fern   | More slices (slice_num 64→128) — **merged**, val_avg=139.83 (round 1 baseline) |
| #376 | fern   | Wider MLP (mlp_ratio 2→4) — **closed**, +4.9% regression and OOD splits all worse |

## Potential next research directions (round 2+)

After round 1 fully resolves, the strongest candidates are:

- **Compound orthogonal round-1 winners** — e.g. slice_num=128 (already merged) + winning loss tweak (alphonse/tanjiro) + winning schedule (frieren/nezuko) + EMA (thorfinn).
- **Wider MLP × wider hidden** — if both #376 and the askeladd retry win, stack them.
- **Modern transformer ergonomics** — SwiGLU, stochastic depth, RMSNorm.
- **Spatial inductive bias** — Fourier features on `(x, z)` fed into the preprocess MLP.
- **Mixed precision** — bf16 to free VRAM/time for wider/deeper models inside the 30-min cap (askeladd is testing this).
- **Multi-scale slice tokenization** — different `slice_num` per layer (global + local physics).
- **Per-domain calibration heads** — output routing on geometry features.
- **Boundary-layer-aware sampling** — over-sample high-Re extremes (which dominate the metric).
- **Longer wall-clock budget** — once a 30-min run shows we're cutting cosine schedules in half, the question of whether `n_layers=8` was truly worse vs just under-trained becomes worth re-asking.
