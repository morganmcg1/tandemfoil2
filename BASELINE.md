# Baseline — icml-appendix-charlie-pai2d-r4

**Status:** Round 1 in flight. PR #549 (linear warmup=3 epochs on top of merged stack) is the current best.

> **Round-1 budget caveat (revised after #401).** `SENPAI_TIMEOUT_MINUTES=30` is still binding, but with `torch.compile(mode=reduce-overhead, dynamic=True)` on top of bf16, per-epoch wall-clock dropped from 141 s → 55 s. **Round 1 is now a ~33-epoch ranking exercise** — the cosine schedule actually enters its decay tail and EMA has time to do its job. The bottleneck has shifted from "compute-bound" to "architecture and effective EMA horizon". Future architectural-scale PRs (wider, deeper) that previously couldn't fit the budget should be revisited.

## Current best (PR #549, alphonse, merged 2026-04-28)

| Metric | Value | Epoch |
|---|---|---|
| `val_avg/mae_surf_p`  | **54.12** (EMA-evaluated) | 34 / 50 (timeout-capped) |
| `test_avg/mae_surf_p` | **47.54** (EMA-evaluated) | best ckpt = epoch 34 |
| Per-epoch wall-clock | 53-55 s (median) | matches #539 |
| Total epochs in budget | 34 | |
| Peak GPU memory | 24.2 GB | unchanged |

> **Reproduce status — three meaningful configurations now:**
> 1. `python train.py --epochs 50` (no flags) reproduces the Config default: β=0.5, no FiLM, no warmup ≈ 58.34 val (alphonse's paired warmup0 ref).
> 2. `python train.py --epochs 50 --huber_beta 0.3 --film` reproduces best β=0.3 + FiLM ≈ 55.43 val (#539's headline).
> 3. `python train.py --epochs 50 --huber_beta 0.5 --warmup_epochs 3` reproduces this PR's best ≈ 54.12 val (no FiLM in alphonse's measurement).
> **Predicted combined config** `--huber_beta 0.3 --film --warmup_epochs 3`: ~52-53 val (untested). Future PRs should include all three flags to test the full compound.

### Per-split val (epoch 34, EMA weights, warmup3 + β=0.5, no FiLM in alphonse's measurement)
| Split | mae_surf_p |
|---|---|
| val_single_in_dist     |  58.49 |
| val_geom_camber_rc     |  65.31 |
| val_geom_camber_cruise |  36.39 |
| val_re_rand            |  56.27 |

### Per-split test (best EMA checkpoint, post-fix scoring, warmup3 + β=0.5)
| Split | mae_surf_p |
|---|---|
| test_single_in_dist     |  52.37 |
| test_geom_camber_rc     |  60.09 |
| test_geom_camber_cruise |  31.05 |
| test_re_rand            |  46.65 |

## Configuration of the current best

Reproduce: `cd target && python train.py --epochs 50 --huber_beta 0.5 --warmup_epochs 3` (Fourier + Huber β=0.5 + linear warmup=3 + EMA + clip + bf16 + compile + cudagraph_skip — but no FiLM since alphonse's branch was pre-#484). Untested config that should beat this number: `--huber_beta 0.3 --film --warmup_epochs 3` (combining all three winning levers).

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs (50). At 33 epochs we reach 66% of decay. |
| Batch size | 4 |
| Surf weight | 10.0 (published default) |
| Epochs (configured / completed) | 50 / ~33 (capped by `SENPAI_TIMEOUT_MINUTES=30`) |
| Model | Transolver: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| **Input encoding** | **8-frequency Fourier features** on (x, z) position (from #368). fun_dim 22 → 54, +8K params on preprocess. |
| **Loss** | SmoothL1 / Huber β=0.5 in this PR's measurement (β=0.3 is independently best from #539 but wasn't combined here; combined config untested) |
| **Schedule** | **3-epoch linear warmup** (start_factor=1e-3, end_factor=1.0) → cosine over remaining epochs (from #549). Cosine endpoint preserved by `T_max = cosine_epochs - warmup_epochs`. |
| **EMA** | decay=0.995; eval + test use EMA weights |
| **Grad clip** | max_norm=10.0 |
| **bf16 autocast** | wraps `model({"x":x_in})["preds"]` in train + eval (from #372) |
| **torch.compile** | `mode="reduce-overhead", dynamic=True` (from #401) |
| **cudagraph_skip** | `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` (from #467, eliminates per-shape CUDAGraph private-pool flakiness; throughput-neutral) |
| **Surface-conditional FiLM** | learned `(γ_surf, β_surf)` vs `(γ_vol, β_vol)` modulating the LayerNorm output before `mlp2` in last TransolverBlock (from #484). Identity init (γ=1, β=0). +512 params (4×n_hidden). Domain-conditional decoder with no parallel pathway. |
| **`--cosine_epochs` flag** | plumbed at default 50 (from #466), available for explicit override |
| Eval | MAE in physical space, primary metric `val_avg/mae_surf_p` |

JSONL: `models/model-charliepai2d4-alphonse-warmup3-20260428-060819/metrics.jsonl`

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
| #539 (merged) | 55.43 | Huber β=0.3 wins by -6.2% paired val. 5-pt monotone β grid {0.3-2.0} — no interior optimum. Config default flipped 1.0→0.5. |
| #549 (merged) | **54.12** | **Linear warmup=3 epochs** (start_factor=1e-3 → 1.0, then cosine to existing endpoint). Mechanism: epoch-1 grad norms 2-2.5× smaller (max 35-43 vs 178), AdamW m/v init properly. Crossover at ep 20: warmup arms start behind, pull ahead in cosine tail. Sweet spot at warmup=3 (warmup=2 too short, warmup=5 over-spends). Paired -7.23% vs warmup=0; vs #539: -2.4% val, -0.9% test. **Note**: alphonse's run was pre-#484 (no FiLM) and β=0.5 (not 0.3); combined `warmup3 + β=0.3 + FiLM` predicted ~52-53 range, untested. |
