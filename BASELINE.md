# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #467 (Huber β=0.5 + Fourier + EMA + clip + bf16 + compile) is the current best.

> **Round-1 budget caveat (revised after #401).** `SENPAI_TIMEOUT_MINUTES=30` is still binding, but with `torch.compile(mode=reduce-overhead, dynamic=True)` on top of bf16, per-epoch wall-clock dropped from 141 s → 55 s. **Round 1 is now a ~33-epoch ranking exercise** — the cosine schedule actually enters its decay tail and EMA has time to do its job. The bottleneck has shifted from "compute-bound" to "architecture and effective EMA horizon". Future architectural-scale PRs (wider, deeper) that previously couldn't fit the budget should be revisited.

## Current best (PR #467, askeladd, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **57.50** (EMA-evaluated) | 34 / 50 (timeout-capped, still descending) |
| `test_avg/mae_surf_p` | **50.51** (EMA-evaluated) | best ckpt = epoch 34 |
| Per-epoch wall-clock | 54.1 s (median) | matches #289/#368 |
| Total epochs in budget | 34 | (1 more than #368, +cudagraph_skip applied) |
| Peak GPU memory | 23.8 GB | unchanged |

> **IMPORTANT — reproduce requires `--huber_beta 0.5`:** the train.py Config default is still `huber_beta=1.0` (the value at merge-base). Reproducing the merged baseline requires explicit `--huber_beta 0.5`. Askeladd's next PR (β finer sweep) will flip the Config default to 0.5.

### Per-split val (epoch 34, EMA weights, β=0.5)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     |  64.77 |
| val_geom_camber_rc     |  70.21 |
| val_geom_camber_cruise |  38.46 |
| val_re_rand            |  56.57 |

### Per-split test (best EMA checkpoint, post-fix scoring, β=0.5)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     |  57.26 |
| test_geom_camber_rc     |  63.95 |
| test_geom_camber_cruise |  32.27 |
| test_re_rand            |  48.58 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50 --huber_beta 0.5` (Fourier + Huber β=0.5 + EMA + clip + bf16 + compile + cudagraph_skip all stacked).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50). At 33 epochs we reach 66% of decay. |
| Batch size | 4 |
| Surf weight | 10.0 (published default) |
| Epochs (configured / completed) | 50 / ~33 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| **Input encoding** | **8-frequency Fourier features** on (x, z) position (from #368). fun_dim 22 → 54, +8K params on preprocess. |
| **Loss** | **SmoothL1 / Huber β=0.5** in normalized space (from #467; default still β=1.0 in Config — pass `--huber_beta 0.5` to reproduce) |
| **EMA** | decay=0.995; eval + test use EMA weights |
| **Grad clip** | max_norm=10.0 |
| **bf16 autocast** | wraps `model({"x":x_in})["preds"]` in train + eval (from #372) |
| **torch.compile** | `mode="reduce-overhead", dynamic=True` (from #401) |
| **cudagraph_skip** | `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` (from #467, eliminates per-shape CUDAGraph private-pool flakiness; throughput-neutral) |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-beta0.5-20260428-032300/metrics.jsonl`

> **Known compile flakiness:** 2 of 4 launches at this stack crashed before completion, both with CUDAGraph private-pool blowup at variable mesh sizes. Setting `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` would eliminate this failure mode at ~10-15% throughput cost. Queued as a small infrastructure PR.

> **Round-2 implication.** The compile-driven epoch-budget recovery (#372 + #401) is the throughput foundation; Huber (#289) compounds with the EMA + clip stack at ~5%. Cumulative −53% from the published-baseline-equivalent. Future levers to try: cosine T_max retune (avoids schedule mismatch on depth-increasing PRs), per-channel pressure ramp (fern's #453 in flight), additive surface decoder (thorfinn's #436), Fourier features (edward's #368 rebasing).

## Compoundable wins still on the table

PR #287 (surf_weight=25) was merged independently before #308 landed; the artifact files are in `models/model-surf-weight-25-20260427-225335/`. **The two changes are orthogonal** — combining surf_weight=25 with EMA+clip is a likely round-2 candidate.

## Update history

| PR | val_avg/mae_surf_p | Notes |
|---|---|---|
| #287 (merged) | 126.67 | surf_weight 10→25, 14/50 epochs, timeout-capped. |
| #308 (merged) | 106.40 | EMA(0.999) + grad clip 1.0, 13/50 epochs, EMA-evaluated. -16.2% vs #287. |
| #372 (merged, infrastructure) | 108.93 (no EMA) | bf16 autocast (1.36× speedup, 19/50 epochs). Treated as infra; baseline anchor stayed at 106.40. |
| #381 (merged) | 98.85 | EMA(0.995) + grad clip 10.0, 13/50 epochs, EMA-evaluated. -7.1% vs #308. EMA crosses online at epoch 2. |
| #401 (merged) | 66.89 | torch.compile(reduce-overhead, dynamic) + bf16 + EMA + clip. 33/50 epochs in budget. -37.1% vs #308, -32.3% vs #381. Throughput-budget recovery is dominant mechanism. |
| #289 (merged) | 63.33 | SmoothL1/Huber β=1.0 loss replacing MSE. 32/50 epochs in budget. -5.31% vs #401. Per-split mechanism preserved. |
| #368 (merged) | 62.94 | 8-freq Fourier positional encoding on (x, z) input. 33/50 epochs. -0.62% vs #289 on val, -1.30% on test_avg. Mechanism: Fourier accelerates convergence in warm-LR phase. |
| #467 (merged) | **57.50** | **Huber β=0.5** (from default β=1.0) + cudagraph_skip robustness flag. 34/50 epochs. **-8.65% val, -7.71% test vs #368** — strongest single-knob win since compile. Strict monotone β order (0.5 < 1.0 < 2.0) on every val + test split + every channel. Cruise gains most (-12% val) — sharper-near-zero helps low-amplitude splits. **Note**: Config default still β=1.0; reproduce requires `--huber_beta 0.5`. |
