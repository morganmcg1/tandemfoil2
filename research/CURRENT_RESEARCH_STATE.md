# SENPAI Research State

- **Updated:** 2026-04-29 02:55 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #862 Round 2 (slice=32 + Huber δ=0.5 + EMA + clip + warmup=0) — MERGED:**  
`val_avg/mae_surf_p = 82.64`, `test_avg/mae_surf_p = 73.02`  
**−41.4% val / −43.1% test vs unmodified default (140.95/128.32).**

Beats PR #881 (val=85.23, δ=0.1 + EMA at slice=64) by 3.0% val / 4.7% test.

**Milestone chain:**
- Unmodified default: val=140.95 (PR #846)
- EMA alone: val=119.35 (PR #773)
- Huber δ=0.5 alone: val=102.86 (PR #769)
- 4-way stack (δ=0.5+EMA+clip+w0): val=96.54 (PR #775)
- Huber δ=0.1 + EMA (slice=64): val=85.23 (PR #881)
- **slice=32 + 4-way stack (δ=0.5): val=82.64 (PR #862) ← current best**

**Three confirmed independent winning directions** (one new from PR #860):
1. **Lower δ (PR #881):** δ=0.1 + EMA at slice=64 → val=85.23
2. **Lower slice_num (PR #862):** slice=32 + 4-way stack → val=82.64
3. **OneCycle scheduling (PR #860 not merged):** OneCycle peak_lr=1e-3 + 4-way stack at slice=64 → val=83.76 (−2.8% vs cosine on same stack). Sent back to thorfinn to test on slice=32 stack (predicted val ~80).

## Current research focus

**Phase: triple convergence — δ-floor × slice-floor × OneCycle**

Three independent wins now confirmed on their respective stacks:
- **δ-floor:** δ=0.1 wins over δ=0.5 at slice=64; δ=0.05/0.025 being tested (alphonse #957)
- **slice-floor:** slice=32 wins monotonically; slice=16 being tested on δ=0.1 (frieren #1004)
- **schedule:** OneCycle with peak_lr=1e-3 adds clean −2.8% val on 4-way stack; testing OneCycle + slice=32 (thorfinn #860 round 3)
- **Combinations:** slice=32 + δ=0.1 predicted val ~76–79; slice=32 + OneCycle predicted ~80; slice=32 + δ=0.1 + OneCycle predicted potentially low 70s

Four key open questions:

1. **Does slice=32 + δ=0.1 combine for a new record?** (frieren #1004, includes slice=32+δ=0.1 run)
2. **Does δ < 0.1 improve further, and does clip compound?** (alphonse #957 — 2D scan)
3. **Does OneCycle stack with slice=32?** (thorfinn #860 round 3 — sent back)
4. **Can BF16 give us more epochs per 30-min budget?** (tanjiro #959) All models still descending at final epoch — throughput is binding.

**Secondary:** surf_weight (fern #859), per-channel Huber δ (edward #951), and clip fine-scan (nezuko #944) remain in flight on outdated baselines. Results will be informative even so; if any beat val=82.64, merge and re-evaluate.

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #957 | alphonse | δ scan {0.1,0.05,0.025} × {clip on/off} — 5-way stack + δ-floor | Status:WIP |
| #959 | tanjiro | BF16 mixed precision for throughput (more epochs in budget) | Status:WIP |
| #944 | nezuko | Clip norm fine-scan {0.1,0.25,0.5,1.0} on δ=0.5+EMA+w0 stack | Status:WIP (outdated δ) |
| #951 | edward | Per-channel Huber δ (δ_p ∈ {0.1,0.25,0.5,1.0} vs δ_vel=0.5) on full stack | Status:WIP |
| #859 | fern | Surface weight scan sw ∈ {10,15,20,30} on full stack | Status:WIP (rebase) |
| #860 | thorfinn | OneCycle schedule on full 4-way stack | Status:WIP (rebase) |
| #1004 | frieren | Slice scan {16,24,32,64} on δ=0.1 + EMA — find slice-floor at new loss | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |

**Note on outdated PRs:** #944, #859 are running or rebasing with δ=0.5 as base. Their results will be valid signals at δ=0.5 but need re-testing at δ=0.1 if they win. The current baseline to beat is val=82.64.

**Round 3 in flight (sent back):**
- #860 thorfinn: OneCycle confirmed −2.8% val on slice=64 4-way stack (val=83.76). Sent back to test slice=32 + OneCycle. Predicted val ~80.

## Key pending questions

1. **Does slice=32 + δ=0.1 combine for a new record?** (frieren #1004 — top priority, predicted val ~76–79)
2. **Does slice < 32 improve further at δ=0.1?** (frieren #1004, tests slice=16)
3. **Does clip+warmup=0 stack with δ=0.1?** (alphonse #957)
4. **Does δ < 0.1 improve further?** (alphonse #957, tests δ=0.05 and 0.025)
5. **Does BF16 give ≥1.5× throughput?** (tanjiro #959 — unlocks depth=8 and ~18-20 epochs/run)
6. **Does OneCycle stack with slice=32?** (thorfinn #860 round 3 — sent back, predicted val ~80)
7. **Does per-channel Huber δ_p < δ_vel help?** (edward #951)
8. **Does surf_weight tuning help on full stack?** (fern #859 rebase)

## Potential next research directions

**Immediate (once alphonse #957 returns):**
- If clip+δ=0.1 stacks: update all in-flight PRs to use δ=0.1 + clip
- If δ=0.05 wins: push to δ=0.025 or MAE (δ→0)
- EMA decay scan at δ=0.1: PR #773 fixed 0.99 without considering smaller δ

**Once thorfinn #860 round 3 returns (OneCycle + slice=32):**
- If OneCycle wins: scan peak_lr ∈ {7.5e-4, 1.5e-3, 2e-3} on slice=32 stack
- Combine OneCycle + δ=0.1 + slice=32 (3-way orthogonal stack)
- pct_start scan once peak_lr is locked

**After BF16 throughput confirmed:**
- Depth=8 retry with BF16 (tanjiro's own follow-up)
- Longer cosine schedule if we get 20+ epochs: T_max tuning becomes less critical
- Consider n_layers=6/7 as intermediate options

**Unexplored with strong motivation:**
- MAE loss (δ→0 limit): if δ=0.025 still wins, MAE may be the asymptotic optimum
- Cross-attention surface↔volume: physics inductive bias for boundary conditions
- Mesh node subsampling: synthetic throughput gain without code complexity

**Architecture (deferred):**
- Surface-aware routing (askeladd #770 still in flight)
- Galerkin attention swap

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap (~14 epochs for n_layers=5).
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling in `train.py`.
- One hypothesis per PR. Compound only after isolated wins verified.
- - **Minimum flags for new experiments: `--huber_delta 0.1 --ema_decay 0.99 --slice_num 32`** (three confirmed wins)
- clip+warmup interaction at δ=0.1 is still being investigated — do NOT mandate `--clip_norm 0.5 --warmup_epochs 0` until alphonse #957 confirms
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
- CLI flags available: `--warmup_epochs`, `--clip_norm`, `--huber_delta`, `--ema_decay`, `--slice_num`, `--n_layers`, `--surf_weight`, `--beta2`, `--use_bf16`
