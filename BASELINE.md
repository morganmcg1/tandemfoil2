# Baseline — `icml-appendix-charlie-pai2d-r1`

## Current best
- **`val_avg/mae_surf_p` = 43.165** (EMA(0.995), epoch 20/50, timeout-cut)
- **`test_avg/mae_surf_p` = 36.746**
- Set by **PR #675** (`charliepai2d1-askeladd/ema-decay-0p995`), merged 2026-04-28 09:15 UTC.
- Beats prior baseline (#394) by **−1.17 % val / −0.47 % test**. Both val and test improve consistently — first non-divergent improvement after 6+ rebased-PR loop turns. Per-split val: rc gains −4.33 % (largest), re_rand −1.38 %, single +2.22 %, cruise +0.49 %. Per-split test: single −4.42 % (largest), re_rand +0.27 %, rc +1.59 %, cruise +1.25 %.
- **Mechanism**: EMA basin shifted with budget. Under 14-ep eager budget, EMA(0.99) was strictly optimal (#417 merged; #445/#474 closed at 0.95/0.97 with 6-9 % regressions). Under 20-ep compile budget, EMA(0.995) (half-life ~138 batches = 8.1 % of run) better matches the longer cosine-descent path than EMA(0.99) (4.1 % of run). Polyak-Ruppert "half-life ~5-15 % of run" heuristic applies — basin shifts slightly slower as budget grows.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- **MLP block**: SwiGLU `(W_g(x) ⊙ silu(W_v(x))) W_o` at `swiglu_inner=168` (matched param count).
- **Optimizer: Lion** (sign-of-momentum) `lr=2.5e-4`, `weight_decay=3e-4`, `betas=(0.9, 0.999)`. Basin upper edge in [2.5e-4, 3.3e-4] (under β2=0.99); β2=0.999 (longer-history buffer) wins decisively over default 0.99.
- **Loss**: `MSE_vol + 10.0 * SmoothL1_surf(β=0.5)` in normalized space.
- Schedule: cosine annealing over `epochs` (T_max=50). At ep20 cosine ≈ 0.85 of peak (lr ~2.13e-4).
- Batch: `batch_size=4`, balanced-domain weighted sampling
- **TF32 matmul precision**: `torch.set_float32_matmul_precision('high')`.
- **`torch.compile(model, ema_model)`** with `mode="default"`, `dynamic=True`. **20 epochs in 30-min budget** (vs 14 eager). −28.4 % steady-state per-epoch.
- **EMA(0.995)** shadow weights drive validation, best-checkpoint selection, and final test eval (slower decay matched to 20-ep budget; PR #675). Compile preserved by storing `_orig_mod` references for save/load.
- **`evaluate_split` NaN-safe pre-pass**: drops samples with non-finite ground truth.
- **Gradient clipping at `max_norm=0.5`** between `loss.backward()` and `optimizer.step()` — under Lion this only smooths the momentum buffer; kept for lineage.

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Per-split breakdown (PR #675 best-EMA-epoch checkpoint, ep20/50, full live config)

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| single_in_dist | 46.500 | 40.058 |
| geom_camber_rc | 56.991 | 49.690 |
| geom_camber_cruise | 26.468 | 22.002 |
| re_rand | 42.703 | 35.234 |
| **avg** | **43.165** | **36.746** |

## Reproduce (current baseline)
```
cd target/ && python train.py --agent <name> --experiment_name <name>/<slug>
```
(Trains with TF32 + `torch.compile` + Lion(lr=2.5e-4, β2=0.999) + SwiGLU(168) + EMA(0.99) + NaN-safe pre-pass + grad-clip(0.5) + SmoothL1(β=0.5)/MSE-vol per current `train.py`. Hits `SENPAI_TIMEOUT_MINUTES=30` at ~ep20 of 50 configured. Steady-state wall clock ~94 s/epoch.)

## History
- 2026-04-27 23:42 — **PR #356** (tanjiro/ema-eval): EMA(0.999) + NaN-safe pre-pass. val=132.276 / test=118.041.
- 2026-04-28 00:43 — **PR #374** (tanjiro/grad-clip-1p0): `clip_grad_norm_(1.0)`. val=113.157 (−14.45 %).
- 2026-04-28 01:29 — **PR #402** (tanjiro/grad-clip-0p5): `max_norm=1.0 → 0.5`. val=110.822 (−2.07 %).
- 2026-04-28 01:41 — **PR #408** (fern/higher-lr-1e3): `lr=5e-4 → 1e-3`. val=107.957 (−2.59 %).
- 2026-04-28 01:54 — **PR #417** (askeladd/ema-decay-0p99): `ema_decay=0.999 → 0.99`. val=98.581 (−8.69 %).
- 2026-04-28 02:48 — **PR #398** (nezuko/swiglu-mlp-matched): GELU MLP → SwiGLU at matched params. val=89.349 (−9.36 %).
- 2026-04-28 03:46 — **PR #430** (tanjiro/lion-optimizer): AdamW → Lion. val=67.737 (−24.19 %). Biggest single-PR delta.
- 2026-04-28 04:33 — **PR #352** (edward/smoothl1-surface): SmoothL1(β=1.0)/MSE-vol. val=64.158 (−5.28 %).
- 2026-04-28 05:17 — **PR #491** (fern/tf32-matmul-precision): TF32 fp32-matmul. val=63.218 (−1.47 %). Throughput multiplier (14 epochs vs 12).
- 2026-04-28 05:27 — **PR #535** (edward/smoothl1-beta-0p5): SmoothL1 β=1.0 → 0.5. val=61.508 (−2.70 %).
- 2026-04-28 06:12 — **PR #536** (tanjiro/lion-lr-2p5e-4): Lion `lr=1.7e-4 → 2.5e-4`. val=**60.478** (−1.67 %) / test=**52.676** (+0.65 %, within noise). All val splits improve uniformly; `single_in_dist` largest gain (−11.95 % val, −14.17 % test). Lion basin upper edge confirmed in [2.5e-4, 3.3e-4].
- 2026-04-28 06:23 — **PR #571** (frieren/lion-beta2-0p999): Lion `β2=0.99 → 0.999` (longer-history buffer for sign-update direction). val=**52.116** (−13.83 %) / test=**45.413** (−13.79 %). All four splits gain ≥10 %; `single_in_dist` largest (−16.02 % val, −13.74 % test). Establishes β1-vs-β2 mechanism distinction: β1 trades responsiveness for direction smoothness (lose case in #545); β2 trades a few warm-up batches for persistent direction smoothness while retaining full responsiveness (clean win here). Largest single-PR delta after #430 (Lion adoption).
- 2026-04-28 07:23 — **PR #394** (thorfinn/torch-compile-throughput): `torch.compile(model, ema_model)` mode="default" + `dynamic=True`. val=**43.677** (−16.19 %) / test=**36.920** (−18.70 %). **−28.4 % steady-state per-epoch** (94 s vs 131 s eager) → 20 epochs in 30-min budget (vs 14). All four splits gain (val Δ −8.69 % to −22.75 %; test Δ −13.80 % to −24.92 %); `geom_camber_cruise` largest test gainer (−24.92 %), `single_in_dist` largest val gainer (−19.71 %). Mechanism: same per-step behavior, more steps under cosine descent (ep14 EMA 51.55 ≈ #571 ep14 52.12; entire metric Δ comes from extra ep15–20). Throughput multiplier orthogonal to all axes; permanent floor for round-2.
- 2026-04-28 09:15 — **PR #675** (askeladd/ema-decay-0p995): EMA decay `0.99 → 0.995` (slower averaging matched to 20-ep budget). val=**43.165** (−1.17 %) / test=**36.746** (−0.47 %). Per-split val: rc gains −4.33 % (largest), re_rand −1.38 %, single +2.22 %, cruise +0.49 %. Per-split test: single −4.42 % (largest), re_rand +0.27 %, rc +1.59 %, cruise +1.25 %. **Mechanism**: EMA basin shifted with budget — Polyak-Ruppert "half-life ~5-15 % of run" heuristic applies. Under 14-ep budget, EMA(0.99) at 5.8 % of run was optimum; under 20-ep budget, EMA(0.995) at 8.1 % of run wins by 1.2 %. First non-divergent improvement after 6+ rebased-PR loop turns of wash/regress. Mechanism: longer half-life smooths more late-stage iterate noise (ep15 raw spike 63.7 → 49.2 EMA is the cleanest single-epoch demonstration); best-epoch stays at ep20 (no shadow lag).
