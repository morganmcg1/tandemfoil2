# SENPAI Research State

- **Date:** 2026-04-28 02:00
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet
- **Empirical baseline (round 1):** PR #336 (slice_num=128) was reverted on commit `605b439` via PR #433. Current advisor HEAD is the **default Transolver with slice_num=64**. Cluster evidence from round 1 puts `val_avg/mae_surf_p` for this baseline in the **130-132 band** (single-seed); thorfinn's PR #428 multi-seed calibration is establishing the precise distribution.
- **Cross-cutting bug being fixed:** `data/scoring.py:accumulate_batch` propagates `NaN` through the per-sample-skip mask (`NaN * 0.0 = NaN`, plus `0 * inf = NaN` per alphonse's independent diagnosis). Root cause is 761 non-finite values in the `p` channel of `test_geom_camber_cruise/000020.pt`'s ground truth `y`. Fix in flight as PR #375 (edward) — advisor-authorized exception to the read-only contract on `data/`.
- **#336 reverted (commit 605b439):** Direct apples-to-apples evidence from alphonse PR #329 rebased and frieren PR #338 rebased confirmed slice_num=128 was a partial-credit merge inside the 30-min cap. All round-1 in-flight PRs forked off pre-revert advisor will need a small rebase before they can merge (only `train.py` model_config will conflict — trivial resolution: keep advisor's slice_num=64).
- **Seed variance (NEW from #331 close):** measured at **±10-15% on `val_avg/mae_surf_p` at 12 epochs** (askeladd's v1=141.998 vs v2=163.280 same config). Many round-1 apparent wins on single seeds are inside this noise band. Going forward, ask winning candidates for a 2-seed confirmation before merge.
- **bf16 calibration (NEW from #331 close):** bf16 buys ~26% per-epoch wall-time with zero clamp events (no model-output overflow at our dynamic range). Capacity-axis hypotheses should default to bf16. bs=8 still OOMs at `n_hidden=192` even with bf16; bs=6 is the practical ceiling.

## Current research focus

Round 1 in progress. Strategy:

1. Independent axes tested first (one hypothesis per PR) to attribute gains cleanly.
2. Winners merge sequentially, best-first, each becoming the new baseline.
3. Round 2 compounds the orthogonal winners.

## In-flight PRs (status as of 2026-04-27 23:55)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #441 | alphonse  | bf16 mixed precision standalone (2-seed for variance)           | wip (new; bf16 buys ~26% wall-clock back per #331 — every other PR's hypothesis test becomes more decisive) |
| #413 | askeladd  | Huber loss for surface pressure (delta=1.0)                     | wip (new; replaces closed #331 — Huber attacks the heavy-tailed-pressure mechanism behind round-1 seed variance) |
| #427 | frieren   | Budget-aware cosine (T_max=11 matched to realized epochs)       | wip (new; replaces closed #338) |
| #339 | nezuko    | Larger batch (`batch_size 4→8`) with √2 LR scale                | wip |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | wip |
| #428 | thorfinn  | Multi-seed baseline calibration (3 seeds of default config)     | wip (new; replaces closed #341) |
| #375 | edward    | Bugfix: nan_to_num in `data/scoring.py`                         | wip (sent back; fix is bit-exact correct, but branch needs rebase before squash-merge to drop reverts to BASELINE.md / research/*.md) |
| #434 | fern      | Gradient clipping (max_norm=1.0), 2-seed for variance measurement | wip (new; replaces closed #405) |

## Closed / merged

| PR | Student | Outcome |
|----|---------|---------|
| #334 | edward | Deeper (n_layers 5→8) — **closed**, clear regression vs slice_num=128 |
| #336 | fern   | More slices (slice_num 64→128) — **merged then reverted**, val_avg=139.83 was a partial-credit single-seed result; reverted via PR #433 (commit 605b439) |
| #433 | alphonse | Revert #336 — **merged**, restores slice_num=64 as round-1 baseline |
| #376 | fern   | Wider MLP (mlp_ratio 2→4) — **closed**, +4.9% regression and OOD splits all worse |
| #331 | askeladd | Wider (n_hidden 192, n_head 6) — **closed** after bf16+bs6 retry; v1=141.998 vs v2=163.280 reveals ±10-15% seed variance, no clean win |
| #338 | frieren | LR warmup post-rebase (slice_num=128) — **closed**, +2.9% regression; slice_num=64+warmup vs slice_num=128+warmup direct comparison shows slice_num=64 wins by 9.7% |
| #341 | thorfinn | EMA(0.999) on slice_num=64 — **closed**, apparent win is single-oscillation absorption + slice_num confound; not statistically separated from baseline |
| #329 | alphonse | surf_weight sweep — **closed** after rebased re-run; sw=50 on slice_num=128 hit 151.34 (+8.2% vs baseline). Direct evidence triggered #336 revert (PR #433). |
| #405 | fern | Fourier features L=8 — **closed**, +1.5% regression but bigger refutation is per-split inversion (in-dist wins, OOD splits regress — opposite of hypothesis). |

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
