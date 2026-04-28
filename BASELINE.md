# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #539 (Huber β=0.3 + Config default flip to 0.5) is the current best.

> **Round-1 budget caveat (revised after #401).** `SENPAI_TIMEOUT_MINUTES=30` is still binding, but with `torch.compile(mode=reduce-overhead, dynamic=True)` on top of bf16, per-epoch wall-clock dropped from 141 s → 55 s. **Round 1 is now a ~33-epoch ranking exercise** — the cosine schedule actually enters its decay tail and EMA has time to do its job. The bottleneck has shifted from "compute-bound" to "architecture and effective EMA horizon". Future architectural-scale PRs (wider, deeper) that previously couldn't fit the budget should be revisited.

## Current best (PR #539, askeladd, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **55.43** (EMA-evaluated) | 34 / 50 (timeout-capped) |
| `test_avg/mae_surf_p` | **47.98** (EMA-evaluated) | best ckpt = epoch 34 |
| Per-epoch wall-clock | 53-55 s (median) | matches #484 |
| Total epochs in budget | 34 | |
| Peak GPU memory | 24.2 GB | unchanged |

> **Reproduce status — there are TWO meaningful configurations:**
> 1. `python train.py --epochs 50` (no flags) reproduces β=0.5 + FiLM (Config default) ≈ 57.37 val (matches #484, since #539 only flipped default to 0.5 not 0.3).
> 2. `python train.py --epochs 50 --huber_beta 0.3 --film` reproduces the **best-known** config = 55.43 val (this PR's headline number, but askeladd's run was without FiLM; combined β=0.3+FiLM expected to compound a bit further).

### Per-split val (epoch 34, EMA weights, β=0.3, no FiLM in askeladd's measurement)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     |  57.83 |
| val_geom_camber_rc     |  70.08 |
| val_geom_camber_cruise |  36.67 |
| val_re_rand            |  57.16 |

### Per-split test (best EMA checkpoint, post-fix scoring, β=0.3)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     |  51.08 |
| test_geom_camber_rc     |  62.30 |
| test_geom_camber_cruise |  31.21 |
| test_re_rand            |  47.31 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50 --huber_beta 0.5 --film` (Fourier + Huber β=0.5 + surface-conditional FiLM + EMA + clip + bf16 + compile + cudagraph_skip all stacked).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50). At 33 epochs we reach 66% of decay. |
| Batch size | 4 |
| Surf weight | 10.0 (published default) |
| Epochs (configured / completed) | 50 / ~33 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| **Input encoding** | **8-frequency Fourier features** on (x, z) position (from #368). fun_dim 22 → 54, +8K params on preprocess. |
| **Loss** | **SmoothL1 / Huber β=0.3** in normalized space (from #539; Config default flipped to 0.5 by #539, so `python train.py` reproduces β=0.5; **best-known is β=0.3 — pass `--huber_beta 0.3` to reproduce**) |
| **EMA** | decay=0.995; eval + test use EMA weights |
| **Grad clip** | max_norm=10.0 |
| **bf16 autocast** | wraps `model({"x":x_in})["preds"]` in train + eval (from #372) |
| **torch.compile** | `mode="reduce-overhead", dynamic=True` (from #401) |
| **cudagraph_skip** | `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` (from #467, eliminates per-shape CUDAGraph private-pool flakiness; throughput-neutral) |
| **Surface-conditional FiLM** | learned `(γ_surf, β_surf)` vs `(γ_vol, β_vol)` modulating the LayerNorm output before `mlp2` in last TransolverBlock (from #484). Identity init (γ=1, β=0). +512 params (4×n_hidden). Domain-conditional decoder with no parallel pathway. |
| **`--cosine_epochs` flag** | plumbed at default 50 (from #466), available for explicit override |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-beta0.3-20260428-052101/metrics.jsonl`

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
| #467 (merged) | 57.50 | Huber β=0.5 + cudagraph_skip robustness flag. -8.65% val, -7.71% test vs #368 — strongest single-knob win since compile. |
| #466 (merged, infrastructure) | 64.20 (reproduce-of-#289) | --cosine_epochs flag plumbed at default 50 (no behavior change; available for explicit override). cudagraph_skip auto-deduped. |
| #484 (merged) | 57.37 | Surface-conditional FiLM. +512 params. -0.23% val / -3.06% test vs #467. Paired -3.05% val, all 8 splits gain. |
| #539 (merged) | **55.43** | **Huber β finer sweep — β=0.3 wins** by -6.2% val / -4.9% test (paired). All 4 val + 4 test splits gain at β=0.3. Per-channel mechanism: cruise pressure -11.6% (heavy-tailed residuals benefit most from L1-leaning shape). 5-pt monotone β grid {0.3, 0.5, 0.7, 1.0, 2.0} — no interior optimum. **Config default flipped from 1.0 to 0.5** (verified). vs current merged baseline #484: -3.4% val, -2.0% test. **Note**: askeladd's run was pre-#484 (no FiLM); β=0.3+FiLM combined expected to compound to ~54-55 range. Reproduce best: `--huber_beta 0.3 --film`. |
