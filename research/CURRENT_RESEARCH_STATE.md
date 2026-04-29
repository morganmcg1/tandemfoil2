# SENPAI Research State

- **Updated:** 2026-04-29 03:55 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #959 (BF16 + δ=0.1 + EMA, slice=64, 18 epochs) — MERGED:**  
`val_avg/mae_surf_p = 79.82`, `test_avg/mae_surf_p = 70.00`  
**−43.4% val / −45.4% test vs unmodified default (140.95/128.32).**

**Milestone chain:**
- Unmodified default: val=140.95 (PR #846)
- EMA alone: val=119.35 (PR #773)
- Huber δ=0.5 alone: val=102.86 (PR #769)
- 4-way stack (δ=0.5+EMA+clip+w0): val=96.54 (PR #775)
- Huber δ=0.1 + EMA (slice=64): val=85.23 (PR #881)
- slice=32 + 4-way stack (δ=0.5): val=82.64 (PR #862)
- **BF16 + δ=0.1 + EMA (slice=64): val=79.82 (PR #959) ← current best**

**Four confirmed independent wins (all need combining):**
1. **δ=0.1 (PR #881):** δ=0.1 + EMA at slice=64 → val=85.23
2. **slice=32 (PR #862):** slice=32 + 4-way stack → val=82.64
3. **BF16 (PR #959):** BF16 + δ=0.1 + EMA at slice=64 → val=79.82 (NEW BEST) — 1.353× throughput, 18 epochs vs 14
4. **OneCycle (PR #860 not merged):** −2.8% vs cosine on same stack, slice=32 test pending

## Current research focus

**Phase: BF16 throughput as new baseline + δ×slice×schedule convergence**

BF16 has changed the game — all experiments should now use `--use_bf16`. Critical insight: BF16 gives +4 epochs at n_layers=5. Combined with slice=32 (lower VRAM → potentially more epochs), and δ=0.1, the predicted next milestone is **val ~74–76**.

Key open questions in priority order:

1. **Does BF16 + slice=32 + δ=0.1 combine for a new record?** (tanjiro #1025 — predicted val ~75.8)
2. **Does δ < 0.1 improve further with BF16?** (alphonse #957 — tests δ=0.05/0.025 on EMA stack, no BF16 yet)
3. **Does slice < 32 improve at δ=0.1?** (frieren #1004 — slice {16,24,32,64} on δ=0.1 + EMA, no BF16)
4. **Does OneCycle stack with slice=32?** (thorfinn #860 round 3 — sent back, predicted val ~80)
5. **Does clip=0.25 transfer to slice=32?** (nezuko #944 round 3 — sent back)
6. **Does asymmetric δ_p < δ_vel push below uniform δ=0.1 floor?** (edward #951 round 3 — sent back)

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #1025 | tanjiro | BF16 + slice=32 + δ=0.1 + EMA — full triple-win combination | Status:WIP (NEW) |
| #957 | alphonse | δ scan {0.1,0.05,0.025} × {clip on/off} on EMA stack | Status:WIP |
| #1004 | frieren | Slice scan {8,16,24,32,64} on δ=0.1 + EMA | Status:WIP |
| #944 | nezuko | clip=0.25 vs clip=0.5 head-to-head at slice=32 | Status:WIP |
| #951 | edward | Asymmetric δ_p ∈ {0.1,0.05} + δ_vel=0.5 at slice=32 | Status:WIP |
| #859 | fern | Surface weight scan on full stack | Status:WIP |
| #860 | thorfinn | OneCycle + slice=32 on 4-way stack | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |

**Note on outdated PRs:** #957, #1004, #860, #944, #951, #859 are running on stacks without `--use_bf16`. Their results will be valid signals on FP32 stacks but all will need re-testing with BF16 when they return, as BF16 is now the required minimum.

## Key pending questions

1. **Does BF16 + slice=32 + δ=0.1 combine for val ~75?** (tanjiro #1025 — TOP PRIORITY)
2. **Does slice < 32 improve further at δ=0.1?** (frieren #1004)
3. **Does δ < 0.1 improve further?** (alphonse #957, tests δ=0.05 and 0.025)
4. **Does clip+warmup=0 stack with δ=0.1?** (alphonse #957)
5. **Does OneCycle stack with slice=32?** (thorfinn #860 round 3, predicted val ~80 on FP32)
6. **Does clip=0.25 transfer to slice=32?** (nezuko #944 round 3)
7. **Does asymmetric δ_p=0.05 push below uniform δ floor?** (edward #951 round 3)
8. **Does surf_weight tuning help on full stack?** (fern #859 rebase)

## Potential next research directions

**After tanjiro #1025 returns (BF16 + slice=32 + δ=0.1):**
- If confirmed win: this becomes new minimum stack for all experiments
- Combine with OneCycle (if thorfinn confirms) → potential val ~72-74
- Combine with clip=0.25 (if nezuko confirms) → potential ~1-2% further

**Once alphonse #957 returns (δ × clip at δ=0.1):**
- If clip+δ=0.1 stacks: add --clip_norm to minimum flags
- If δ=0.05 wins: push to δ=0.025 or MAE (δ→0) with BF16 and slice=32
- EMA decay scan at δ=0.1 + BF16 (EMA 0.99 was tuned at FP32 / δ=0.5)

**BF16 throughput opens new directions:**
- n_layers=6 or 7 intermediate depth (between n=5 and n=8 — currently n=8 is still too slow even with BF16)
- batch_size=8 with BF16 (33 GB peak at batch=4; batch=8 is ~48 GB → fits comfortably)
- Mesh node subsampling: synthetic throughput gain complementary to BF16

**Unexplored with strong motivation:**
- MAE loss (δ→0 limit): if δ=0.025 still wins, MAE may be the asymptotic optimum
- Cross-attention surface↔volume: physics inductive bias for boundary conditions
- Galerkin attention swap (askeladd's architecture lane)

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- With BF16 at n_layers=5: ~18-20 epochs/run. All models still descending at timeout.
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling in `train.py`.
- One hypothesis per PR. Compound only after isolated wins verified.
- **Minimum flags for ALL new experiments: `--use_bf16 --huber_delta 0.1 --ema_decay 0.99 --slice_num 32`**
- clip+warmup interaction at δ=0.1 is still being investigated — do NOT mandate `--clip_norm 0.5 --warmup_epochs 0` until alphonse #957 confirms
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
- CLI flags: `--warmup_epochs`, `--clip_norm`, `--huber_delta`, `--ema_decay`, `--slice_num`, `--n_layers`, `--surf_weight`, `--use_bf16`, `--scheduler`, `--peak_lr`, `--pct_start`, `--huber_delta_p`
