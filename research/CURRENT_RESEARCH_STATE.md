# SENPAI Research State

- **Updated:** 2026-04-29 04:30 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #860 R3 (OneCycle T=16 + slice=32 + 4-way δ=0.5+clip+w0+EMA, FP32) — MERGED:**  
`val_avg/mae_surf_p = 75.94`, `test_avg/mae_surf_p = 65.86`  
**−46.1% val / −48.7% test vs unmodified default (140.95/128.32).**

**Milestone chain:**
- Unmodified default: val=140.95 (PR #846)
- EMA alone: val=119.35 (PR #773)
- Huber δ=0.5 alone: val=102.86 (PR #769)
- 4-way stack (δ=0.5+EMA+clip+w0): val=96.54 (PR #775)
- Huber δ=0.1 + EMA (slice=64): val=85.23 (PR #881)
- slice=32 + 4-way stack (δ=0.5): val=82.64 (PR #862)
- BF16 + δ=0.1 + EMA (slice=64): val=79.82 (PR #959)
- **OneCycle T=16 + slice=32 + 4-way (δ=0.5, FP32): val=75.94 (PR #860) ← current best**

**Five confirmed independent wins:**
1. **δ=0.1 (PR #881):** δ=0.1 + EMA → val=85.23
2. **slice=32 (PR #862):** slice=32 → val=82.64
3. **BF16 (PR #959):** BF16 + more epochs → val=79.82
4. **OneCycle T=16 (PR #860):** aligned OneCycle schedule → val=75.94 (NEW BEST)
5. **clip+warmup=0 at δ=0.1 (PR #957, not merged):** −4.7% val at slice=64

**Two parallel winning paths (not yet fully combined):**
- **δ=0.5 path:** slice=32 + 4-way + OneCycle T=16 (FP32) → val=75.94
- **δ=0.1 path:** BF16 + δ=0.1 + EMA + slice=64 → val=79.82

The grand combination (BF16 + slice=32 + δ=0.1 + clip + w0 + EMA + OneCycle T=20) is being tested by alphonse #957 R3.

## Current research focus

**Phase: convergence of all wins — the grand combination probe**

Six students now working the BF16 + OneCycle + slice=32 + δ-floor space simultaneously.

Key open questions in priority order:

1. **Grand combination: BF16+slice32+δ=0.1+clip+w0+EMA+OneCycle T=20?** (alphonse #957 R3 — highest priority, predicted val ~70)
2. **Does BF16 stack on top of PR #860's OneCycle win?** (thorfinn #1053 — δ=0.5 path + BF16, predicted val ~71-73)
3. **What is BF16+slice32+δ=0.1 combination without OneCycle?** (tanjiro #1025)
4. **Optimal EMA decay at BF16+slice32+δ=0.1?** (fern #1027 — {0.99, 0.995, 0.999})
5. **Does slice < 32 improve at δ=0.1?** (frieren #1004 — slice scan {8,16,24,32,64})
6. **Does EMA warmup_epochs matter for OneCycle+BF16?** (nezuko #1054 — {3,5,8} epochs)
7. **Does asymmetric δ_p=0.05 push below uniform δ floor?** (edward #951 R3)
8. **Surface-aware architecture routing?** (askeladd #770 — long-running)

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #957 | alphonse | BF16+slice32+δ=0.1+clip+w0+EMA+OneCycle T=20 grand combination | Status:WIP R3 |
| #1053 | thorfinn | BF16 + OneCycle T=20 on slice=32+δ=0.5+4-way | Status:WIP (NEW) |
| #1054 | nezuko | EMA warmup scan {3,5,8} on BF16+OneCycle+slice32+4-way | Status:WIP (NEW) |
| #1025 | tanjiro | BF16 + slice=32 + δ=0.1 + EMA | Status:WIP |
| #1027 | fern | EMA decay scan {0.99,0.995,0.999} on BF16+slice32+δ=0.1 | Status:WIP |
| #1004 | frieren | Slice scan {8,16,24,32,64} on δ=0.1 + EMA | Status:WIP |
| #951 | edward | Asymmetric δ_p ∈ {0.1,0.05} + δ_vel=0.5 at slice=32 | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |

## Settled knobs (no further tuning needed)

- **surf_weight=10**: PR #859 — SW=10 optimal at δ=0.1 (SW=20 regresses +4.55%).
- **clip_norm=0.5 for slice=32**: PR #944 — clip=0.25 transfers only to slice=64 (slice-dependent).
- **δ_floor=0.1**: PR #957 — δ=0.05/0.025 both regress; δ=0.1 is the floor.
- **n_layers=5**: PR #776 — n_layers=8 incompatible with 30-min budget.
- **AdamW β₂=0.999**: PR #867 — default optimal.

## Key OneCycle insight (from PR #860 R3)

**`--onecycle_total_epochs` must match the actual epoch budget for each stack.** At BF16+slice=32, expect ~20-22 epochs in 30 min. Must measure and set T accordingly. Using T from a different slice/precision combination will truncate training early or leave the LR at near-zero before timeout.

EMA alignment pattern: OneCycle peak at ~30%×T ≈ epoch 6 aligns with `ema_warmup_epochs=5` — EMA accumulates entirely post-peak. This is the sweet spot; tuning `ema_warmup_epochs` around this alignment is nezuko #1054's hypothesis.

## Potential next research directions

**After alphonse #957 R3 (grand combination):**
- If val drops below 70: close in on the limit, consider wider exploration (new models, architectures)
- If asymmetric δ (edward) compounds: combine δ_p=0.05+δ_vel=0.5 with OneCycle+BF16
- Peak LR scan for OneCycle: {7.5e-4, 1.5e-3, 2e-3} — still using the 1e-3 default from R1

**Architecture (still unexplored):**
- Surface-aware routing (askeladd #770)
- Galerkin attention swap
- MAE loss (δ→0 pure): if δ=0.025 stacked, pursue pure MAE

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- With BF16 + slice=32 at n_layers=5: ~20-22 epochs/run expected. All models still descending at timeout.
- 96 GB VRAM per GPU, batch_size=4 default.
- One hypothesis per PR. Compound only after isolated wins verified.
- **Minimum flags for ALL new experiments:**
  `--use_bf16 --slice_num 32 --huber_delta 0.1 --ema_decay 0.99 --warmup_epochs 0 --clip_norm 0.5`
  `--scheduler onecycle --peak_lr 1e-3 --pct_start 0.3 --onecycle_total_epochs 20`
  (T=20 is estimate; measure actual epochs and adjust)
- CLI flags: `--warmup_epochs`, `--clip_norm`, `--huber_delta`, `--ema_decay`, `--slice_num`, `--n_layers`, `--surf_weight`, `--use_bf16`, `--scheduler`, `--peak_lr`, `--pct_start`, `--onecycle_total_epochs`, `--ema_warmup_epochs`, `--huber_delta_p`
