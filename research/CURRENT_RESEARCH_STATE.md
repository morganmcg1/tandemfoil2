# SENPAI Research State

- **Date:** 2026-04-28 05:55
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet
- **Empirical baseline (round 1):** PR #441 (bf16 standalone) merged on commit `b605b44`. 2-seed mean: **117.37 ± 0.85** at 19 epochs reached, CV ~0.7%. Advisor HEAD is now: slice_num=64, bf16 autocast, fp32 cast before squaring, seed flag, peak VRAM logging. Multi-seed distribution being further calibrated by thorfinn's PR #428.
- **Cross-cutting bug being fixed:** `data/scoring.py:accumulate_batch` propagates `NaN` through the per-sample-skip mask (`NaN * 0.0 = NaN`, plus `0 * inf = NaN` per alphonse's independent diagnosis). Root cause is 761 non-finite values in the `p` channel of `test_geom_camber_cruise/000020.pt`'s ground truth `y`. Fix in flight as PR #375 (edward) — advisor-authorized exception to the read-only contract on `data/`.
- **#336 reverted (commit 605b439):** Direct apples-to-apples evidence from alphonse PR #329 rebased and frieren PR #338 rebased confirmed slice_num=128 was a partial-credit merge inside the 30-min cap. All round-1 in-flight PRs forked off pre-revert advisor will need a small rebase before they can merge (only `train.py` model_config will conflict — trivial resolution: keep advisor's slice_num=64).
- **Seed variance (NEW from #331 close):** measured at **±10-15% on `val_avg/mae_surf_p` at 12 epochs** (askeladd's v1=141.998 vs v2=163.280 same config). Many round-1 apparent wins on single seeds are inside this noise band. Going forward, ask winning candidates for a 2-seed confirmation before merge.
- **bf16 calibration (NEW from #331 close):** bf16 buys ~26% per-epoch wall-time with zero clamp events (no model-output overflow at our dynamic range). Capacity-axis hypotheses should default to bf16. bs=8 still OOMs at `n_hidden=192` even with bf16; bs=6 is the practical ceiling.

## Current research focus

Round 1 in progress. Strategy:

1. Independent axes tested first (one hypothesis per PR) to attribute gains cleanly.
2. Winners merge sequentially, best-first, each becoming the new baseline.
3. Round 2 compounds the orthogonal winners.

## In-flight PRs (status as of 2026-04-28 04:45)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #537 | alphonse  | AdamW β2=0.95 (transformer-recipe convention, 2-seed)          | wip (new; replaces merged #441 — orthogonal optimizer-internal axis, expected modest 1-4% gain) |
| #413 | askeladd  | Huber loss for surface pressure (δ=1.0)                          | wip (sent back; pre-rebase 2-seed mean 118.47 = -15.3% vs OLD baseline; rebasing onto bf16 advisor + 1 confirmation run, then immediate merge) |
| #427 | frieren   | Budget-aware cosine (T_max matched to realised epochs)          | wip (sent back; 3 OLD-baseline runs confirmed mechanism (-3 to -11% vs #336), need rebase + 2-seed re-run with --cosine_t_max=19 on bf16 baseline) |
| #557 | nezuko    | Attention dropout = 0.1 (small-data regularization, 2-seed)     | wip (new; replaces closed #505 — orthogonal regularization axis, expected differential OOD-helpful effect) |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | wip |
| #428 | thorfinn  | Multi-seed baseline calibration (3 seeds of default config)     | wip |
| #375 | edward    | Bugfix: nan_to_num in `data/scoring.py`                         | wip (sent back; bit-exact correct fix, awaiting rebase before merge) |
| #434 | fern      | Gradient clipping (max_norm=1.0), 2-seed for variance measurement | wip (sent back; 2-seed mean 100.44 ± 5.54 = -14.4% vs bf16 baseline; train.py clean, awaiting research/*.md rebase, then immediate merge) |

## Closed / merged

| PR | Student | Outcome |
|----|---------|---------|
| #334 | edward | Deeper (n_layers 5→8) — **closed**, clear regression vs slice_num=128 |
| #336 | fern   | More slices (slice_num 64→128) — **merged then reverted**, val_avg=139.83 was a partial-credit single-seed result; reverted via PR #433 (commit 605b439) |
| #433 | alphonse | Revert #336 — **merged**, restores slice_num=64 as round-1 baseline |
| #339 | nezuko | bs=8 + sqrt(2) LR scaling — **closed**, bs=8/lr=7e-4 = 144.71 (+9-20% vs round-1 same-baseline cluster). Wall-clock is binding constraint at our cap. sqrt(2) LR rule itself is validated and preserved for round 2. |
| #441 | alphonse | bf16 mixed precision standalone — **merged** (commit b605b44). 2-seed mean 117.37 ± 0.85, CV ~0.7%. New advisor baseline. |
| #505 | nezuko | lr=3e-4 multi-seed — **closed**, mean 137.89 (+17.5% vs bf16 baseline) but CV 4.6% confirmed lower-LR variance reduction; mechanism duplicative of bf16's extra-epochs effect. Round-2 stack candidate. |
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
