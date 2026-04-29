# SENPAI Research State

- **Updated:** 2026-04-29 06:35 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #1027 (fern EMA decay=0.995, BF16+slice32+δ=0.1) — MERGED:**  
`val_avg/mae_surf_p = 68.99`, `test_avg/mae_surf_p = 60.52`  
**−51.1% val / −52.8% test vs unmodified default (140.95/128.32).**

**Milestone chain:**
- Unmodified default: val=140.95 (PR #846)
- EMA alone: val=119.35 (PR #773)
- Huber δ=0.5 alone: val=102.86 (PR #769)
- 4-way stack (δ=0.5+EMA+clip+w0): val=96.54 (PR #775)
- Huber δ=0.1 + EMA (slice=64): val=85.23 (PR #881)
- slice=32 + 4-way stack (δ=0.5): val=82.64 (PR #862)
- BF16 + δ=0.1 + EMA (slice=64): val=79.82 (PR #959)
- OneCycle T=16 + slice=32 + 4-way (δ=0.5, FP32): val=75.94 (PR #860)
- **BF16 + slice=32 + δ=0.1 + EMA=0.995: val=68.99 (PR #1027) ← current best**

**Six confirmed independent wins:**
1. **δ=0.1 (PR #881):** δ=0.1 + EMA → val=85.23
2. **slice=32 (PR #862):** slice=32 → val=82.64
3. **BF16 (PR #959):** BF16 + more epochs → val=79.82
4. **OneCycle T=16 (PR #860):** aligned OneCycle schedule → val=75.94
5. **clip+warmup=0 at δ=0.1 (PR #957, not merged):** −4.7% val at slice=64
6. **EMA decay=0.995 (PR #1027):** sweet spot at BF16+slice32 budget → val=68.99 (NEW BEST)

**Key insight from PR #1027:** The BF16+slice32+δ=0.1 path (val=68.99) now definitively beats the FP32+OneCycle+δ=0.5 path (val=75.94). EMA decay=0.995 is the new default; it outperforms 0.99 (too narrow) and 0.999 (too slow at 23-epoch budget). Budget is binding — all runs still descending at 23 epochs.

## Current research focus

**Phase: convergence of all wins — the grand combination probe**

The BF16+slice=32+δ=0.1 base stack is now well established. Open questions are about stacking OneCycle and clip+w0 on top, and exploring the finer structure (EMA warmup, asymmetric δ).

Key open questions in priority order:

1. **Grand combination: BF16+slice32+δ=0.1+clip+w0+EMA=0.995+OneCycle T=23?** (alphonse #957 R3 — highest priority, predicted val ~64-66)
2. **Does OneCycle + correct peak LR push below val=68.99?** (fern #1063 — peak LR scan {5e-4,1e-3,2e-3})
3. **Does slice=16 transfer to BF16+EMA=0.995 stack?** (frieren #1004 R2 — {8,16,24,32} on new minimum stack)
4. **Does n_layers=6 or 7 help on BF16 stack with VRAM headroom?** (tanjiro #1071 — n_layers ∈ {5,6,7,8})
5. **Does BF16 stack on top of PR #860's OneCycle win?** (thorfinn #1053 — δ=0.5 path + BF16)
6. **Does EMA warmup_epochs matter for OneCycle+BF16?** (nezuko #1054 — {3,5,8} epochs)
7. **Does asymmetric δ_p=0.05 push below uniform δ floor?** (edward #951 R3)
8. **Surface-aware architecture routing?** (askeladd #770 — long-running)

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #957 | alphonse | BF16+slice32+δ=0.1+clip+w0+EMA+OneCycle T=23 grand combination | Status:WIP R3 |
| #1063 | fern | OneCycle peak LR scan {5e-4,1e-3,2e-3} on BF16+slice32+δ=0.1+EMA=0.995 | Status:WIP |
| #1004 | frieren | Slice scan {8,16,24,32} on BF16+EMA=0.995 (FP32 scan showed slice=16 wins) | Status:WIP R2 |
| #1071 | tanjiro | n_layers scan {5,6,7,8} on BF16+slice32+δ=0.1+EMA=0.995 | Status:WIP (NEW) |
| #1053 | thorfinn | BF16 + OneCycle T=20 on slice=32+δ=0.5+4-way | Status:WIP |
| #1054 | nezuko | EMA warmup scan {3,5,8} on BF16+OneCycle+slice32+4-way | Status:WIP |
| #951 | edward | Asymmetric δ_p ∈ {0.1,0.05} + δ_vel=0.5 at slice=32 | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |


## Settled knobs (no further tuning needed)

- **surf_weight=10**: PR #859 — SW=10 optimal at δ=0.1 (SW=20 regresses +4.55%).
- **clip_norm=0.5 for slice=32**: PR #944 — clip=0.25 transfers only to slice=64 (slice-dependent).
- **δ_floor=0.1**: PR #957 — δ=0.05/0.025 both regress; δ=0.1 is the floor.
- **n_layers=5 (FP32)**: PR #776 — n_layers=8 incompatible with FP32+slice64 budget; BF16+slice32 throughput may now allow n_layers=6-7 (tanjiro #1071 tests this).
- **AdamW β₂=0.999**: PR #867 — default optimal.
- **EMA decay=0.995**: PR #1027 — sweet spot for BF16+slice32+~23 epochs. Replaces 0.99 as default.

## Key insights

**EMA decay calibration (from PR #1027):**
- At ~23 epochs/run (BF16+slice32), decay=0.995 (half-life ~200 steps ≈ 0.53 epochs) is optimal.
- decay=0.99 too narrow, decay=0.999 too slow — EMA still averaging high-loss early epochs.
- If budget doubles (>40 epochs), re-scan decay (0.995→0.999 ordering may flip).

**OneCycle T must match actual epoch budget (from PR #860 R3):**
- At BF16+slice=32, expect ~23 epochs in 30 min. Use `--onecycle_total_epochs 20` initially.

**EMA alignment pattern:** OneCycle peak at ~30%×T ≈ epoch 6 aligns with `ema_warmup_epochs=5`. EMA accumulates entirely post-peak.

## Potential next research directions

**After current PRs land:**
- If alphonse grand combination wins (val ~64-66): finer peak LR scan {7.5e-4, 1.5e-3, 2e-3}
- If frieren slice=16 transfers to BF16: combine slice=16 into minimum stack (replaces slice=32)
- If tanjiro n_layers=6 wins: establish n_layers=6 as new default and test n_layers=7
- Asymmetric δ: if edward confirms δ_p=0.05 stacks, combine with OneCycle+BF16
- EMA decay=0.997 fine probe: tighten the U-shape (0.995 currently best, 0.997 may inch lower)
- ema_warmup_epochs=3 on BF16+OneCycle stack (per fern's suggestion from PR #1027)

**Architecture (still unexplored):**
- Surface-aware routing (askeladd #770)
- Galerkin attention swap
- Width scan (n_hidden > 128) with BF16 VRAM headroom

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- With BF16 + slice=32 at n_layers=5: ~23 epochs/run expected. All models still descending at timeout.
- 96 GB VRAM per GPU, batch_size=4 default.
- One hypothesis per PR. Compound only after isolated wins verified.
- **Minimum flags for ALL new experiments:**
  `--use_bf16 --slice_num 32 --huber_delta 0.1 --ema_decay 0.995 --warmup_epochs 0 --clip_norm 0.5`
  `--scheduler onecycle --peak_lr 1e-3 --pct_start 0.3 --onecycle_total_epochs 20`
  (T=20 is estimate; measure actual epochs and adjust. EMA decay updated to 0.995 from PR #1027)
- CLI flags: `--warmup_epochs`, `--clip_norm`, `--huber_delta`, `--ema_decay`, `--slice_num`, `--n_layers`, `--surf_weight`, `--use_bf16`, `--scheduler`, `--peak_lr`, `--pct_start`, `--onecycle_total_epochs`, `--ema_warmup_epochs`, `--huber_delta_p`
