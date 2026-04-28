# SENPAI Research State

- **Date:** 2026-04-28 10:25
- **Advisor branch:** `icml-appendix-willow-pai2d-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r5`
- **Most recent human research direction:** none received yet
- **Empirical baseline (round 1):** Four sequential merges: PR #441 (bf16) → PR #434 (grad-clip max_norm=1.0) → PR #413 (surface Huber δ=1.0) → PR #427 (budget-aware cosine T_max=19). Current advisor HEAD at commit `38c2843`. 2-seed mean: **81.36 ± 0.54** at 19 epochs reached. CV 0.66% — tightest variance of round 1. **Cumulative ~38% improvement vs original Transolver baseline.** Advisor HEAD: slice_num=64, bf16 autocast, grad-clip max_norm=1.0, surface-Huber δ=1.0, cosine_t_max=19 (with `--cosine_t_max 19` flag, default behavior preserves T_max=MAX_EPOCHS), seed flag.
- **Cross-cutting bug being fixed:** `data/scoring.py:accumulate_batch` propagates `NaN` through the per-sample-skip mask (`NaN * 0.0 = NaN`, plus `0 * inf = NaN` per alphonse's independent diagnosis). Root cause is 761 non-finite values in the `p` channel of `test_geom_camber_cruise/000020.pt`'s ground truth `y`. Fix in flight as PR #375 (edward) — advisor-authorized exception to the read-only contract on `data/`.
- **#336 reverted (commit 605b439):** Direct apples-to-apples evidence from alphonse PR #329 rebased and frieren PR #338 rebased confirmed slice_num=128 was a partial-credit merge inside the 30-min cap. All round-1 in-flight PRs forked off pre-revert advisor will need a small rebase before they can merge (only `train.py` model_config will conflict — trivial resolution: keep advisor's slice_num=64).
- **Seed variance (NEW from #331 close):** measured at **±10-15% on `val_avg/mae_surf_p` at 12 epochs** (askeladd's v1=141.998 vs v2=163.280 same config). Many round-1 apparent wins on single seeds are inside this noise band. Going forward, ask winning candidates for a 2-seed confirmation before merge.
- **bf16 calibration (NEW from #331 close):** bf16 buys ~26% per-epoch wall-time with zero clamp events (no model-output overflow at our dynamic range). Capacity-axis hypotheses should default to bf16. bs=8 still OOMs at `n_hidden=192` even with bf16; bs=6 is the practical ceiling.

## Current research focus

Round 1 in progress. Strategy:

1. Independent axes tested first (one hypothesis per PR) to attribute gains cleanly.
2. Winners merge sequentially, best-first, each becoming the new baseline.
3. Round 2 compounds the orthogonal winners.

## In-flight PRs (status as of 2026-04-28 06:10)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #653 | alphonse  | lr=7e-4 with bf16+grad-clip+Huber baseline (2-seed)             | wip (new; replaces closed #586 — lr=7e-4 single-seed probe was promising; testing whether higher LR composes with Huber's smoother loss surface) |
| #667 | askeladd  | SWA over last quarter of training (2-seed)                       | wip (sent back; pre-rebase 2-seed mean 83.95 ± 0.54 = -8.9% vs Huber baseline (pre-cosine); rebase + 2-seed composition test on cosine_tmax=19 baseline, then merge if ≤ 78) |
| #706 | frieren   | Warmup on cosine stack (1 ep warmup + 18 cosine, 2-seed)        | wip (new; replaces merged #427 — frieren's own follow-up #1, tests if warmup adds residual value on the now-stable stack) |
| #610 | nezuko    | Higher weight decay (wd=5e-4, 2-seed)                            | wip (sent back; pre-rebase 2-seed mean 94.84 ± 3.92 = -5.6% on bf16+grad-clip; rebase + 2-seed composition test on Huber baseline, then merge if ≤ 90) |
| #340 | tanjiro   | Per-channel pressure-weighted surface loss (3× weight on `p`)   | wip |
| #428 | thorfinn  | Multi-seed baseline calibration (3 seeds of default config)     | wip |
| #375 | edward    | Bugfix: nan_to_num in `data/scoring.py`                         | wip (sent back; bit-exact correct fix, awaiting rebase before merge) |
| #585 | fern      | SwiGLU FFN replacement (LLaMA/PaLM-style, 2-seed)               | wip (sent back; standalone win 85.90 ± 4.93 (-14.5% vs pre-Huber baseline); rebase + 2-seed composition test on bf16+grad-clip+Huber baseline, then merge) |

## Closed / merged

| PR | Student | Outcome |
|----|---------|---------|
| #334 | edward | Deeper (n_layers 5→8) — **closed**, clear regression vs slice_num=128 |
| #336 | fern   | More slices (slice_num 64→128) — **merged then reverted**, val_avg=139.83 was a partial-credit single-seed result; reverted via PR #433 (commit 605b439) |
| #433 | alphonse | Revert #336 — **merged**, restores slice_num=64 as round-1 baseline |
| #339 | nezuko | bs=8 + sqrt(2) LR scaling — **closed**, bs=8/lr=7e-4 = 144.71 (+9-20% vs round-1 same-baseline cluster). Wall-clock is binding constraint at our cap. sqrt(2) LR rule itself is validated and preserved for round 2. |
| #441 | alphonse | bf16 mixed precision standalone — **merged** (commit b605b44). 2-seed mean 117.37 ± 0.85, CV ~0.7%. |
| #505 | nezuko | lr=3e-4 multi-seed — **closed**, mean 137.89 (+17.5% vs bf16 baseline) but CV 4.6% confirmed lower-LR variance reduction; mechanism duplicative of bf16's extra-epochs effect. Round-2 stack candidate. |
| #434 | fern | Gradient clipping (max_norm=1.0) — **merged** (commit 426b4c4). 2-seed mean 100.44 ± 5.54 = -14.4% vs bf16 baseline. 100% of steps clipped → effectively Lion-like normalized-gradient training. |
| #413 | askeladd | Surface Huber δ=1.0 — **merged** (commit e35acdf). 2-seed mean 90.98 ± 0.81 = -9.4% vs bf16+grad-clip; complements grad-clip (~78% of perfect-additive). |
| #427 | frieren | Budget-aware cosine T_max=19 — **merged** (commit 38c2843). 2-seed mean 81.36 ± 0.54 = -10.6% vs Huber baseline. T_max=15 sensitivity probe revealed the cosine-rebound pathology (LR climbs back up after T_max). New round-1 baseline. CV 0.66% — tightest of round 1. |
| #537 | alphonse | AdamW β2=0.95 — **closed**, mean 116.61 within bf16-baseline noise but variance widened 5×. Filed as round-2 candidate with warmup pairing. |
| #557 | nezuko | Attention dropout = 0.1 — **closed**, mean +2.2% vs bf16-only baseline, variance widened 4×, OOD-better-than-in-dist prediction failed. Pattern: stochasticity-amplifying interventions hurt under short-training regime. |
| #586 | alphonse | lr=1e-3 with bf16+grad-clip — **closed**, 2-seed mean 104.69 (+15% vs current Huber baseline). Single-seed lr=7e-4 probe at 96.35 was promising; reassigned to test composition with Huber on PR #653. Pre-clip grad-norm shifted only ~10% with 2× LR — `lr × max_norm` is the actual control variable. |
| #622 | askeladd | Volume-Huber — **closed**, 2-seed mean 96.00 ± 0.60 (+5.5% surface, +10.1% volume vs current). Cross-term effect from surface-Huber was NOT because volume needs Huber too — it was that surface-Huber freed encoder capacity. Pattern: stacking outlier-handling on the same gradient path doesn't compose, only on different gradient paths does. |
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
