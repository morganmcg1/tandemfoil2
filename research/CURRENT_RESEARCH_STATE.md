# SENPAI Research State

- **Updated:** 2026-04-29 01:25 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #881 (Huber δ=0.1 + EMA=0.99, no clip, no warmup) — MERGED:**  
`val_avg/mae_surf_p = 85.23`, `test_avg/mae_surf_p = 76.64`  
**−39.5% val / −40.3% test vs unmodified default (140.95/128.32).**

This beats the prior 4-way stack (PR #775, val=96.54) by 11.7%. δ=0.1 is the dominant lever.

Minimum required flags: `--huber_delta 0.1 --ema_decay 0.99`  
Whether clip+warmup=0 help at δ=0.1 is actively being tested (alphonse PR #957).

**Milestone chain:**
- Unmodified default: val=140.95 (PR #846)
- EMA alone: val=119.35 (PR #773)
- Huber δ=0.5 alone: val=102.86 (PR #769)
- 4-way stack (δ=0.5+EMA+clip+w0): val=96.54 (PR #775)
- **Huber δ=0.1 + EMA: val=85.23 (PR #881) ← current best**

## Current research focus

**Phase: δ-floor + 5-way stack confirmation**

The δ=0.1 finding is the biggest win so far. Three key open questions:

1. **Does δ=0.1 still improve with clip+warmup=0 on top?** (alphonse #957) The 4-way stack logic still applies, but the interaction at δ=0.1 is untested.
2. **Is there more headroom below δ=0.1?** (alphonse #957, tests δ=0.05 and δ=0.025) The curve is monotone, still descending.
3. **Can BF16 mixed precision give us more epochs per 30-min budget?** (tanjiro #959) All models are still descending at epoch 14; throughput is the binding constraint.

**Secondary focus:** validating remaining in-flight directions (OneCycle scheduling, slice scan, surf_weight scan, per-channel Huber δ) on the new δ=0.1 default. These PRs are in rebase or running on outdated δ=0.5 assumptions and will compare against val=85.23.

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #957 | alphonse | δ scan {0.1,0.05,0.025} × {clip on/off} — 5-way stack + δ-floor | Status:WIP |
| #959 | tanjiro | BF16 mixed precision for throughput (more epochs in budget) | Status:WIP |
| #944 | nezuko | Clip norm fine-scan {0.1,0.25,0.5,1.0} on δ=0.5+EMA+w0 stack | Status:WIP (outdated δ) |
| #951 | edward | Per-channel Huber δ (δ_p ∈ {0.1,0.25,0.5,1.0} vs δ_vel=0.5) on full stack | Status:WIP |
| #859 | fern | Surface weight scan sw ∈ {10,15,20,30} on full stack | Status:WIP (rebase) |
| #860 | thorfinn | OneCycle schedule on full 4-way stack | Status:WIP (rebase) |
| #862 | frieren | Slice scan downward {32,48,64} on full 4-way stack | Status:WIP (rebase) |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |

**Note on outdated PRs:** #944, #859, #860, #862 are running or rebasing with δ=0.5 as base. Their results will be valid signals at δ=0.5 but need re-testing at δ=0.1 if they win. The new baseline to beat is val=85.23.

## Key pending questions

1. **Does clip+warmup=0 stack with δ=0.1?** (alphonse #957 — top priority)
2. **Does δ < 0.1 improve further?** (alphonse #957)
3. **Does BF16 give ≥1.5× throughput?** (tanjiro #959 — if yes, unlocks depth=8 and ~20 epochs/run)
4. **Does OneCycle help on the full stack?** (thorfinn #860 rebase)
5. **Does lower slice_num {32,48} beat 64 on full stack?** (frieren #862 rebase)
6. **Does per-channel Huber δ_p < δ_vel help?** (edward #951)
7. **Does clip norm tuning matter at the new δ?** (nezuko #944 — partially answered by alphonse #957)

## Potential next research directions

**Immediate (once alphonse #957 returns):**
- If clip+δ=0.1 stacks: update all in-flight PRs to use δ=0.1 + clip
- If δ=0.05 wins: push to δ=0.025 or MAE (δ→0)
- EMA decay scan at δ=0.1: PR #773 fixed 0.99 without considering smaller δ

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
- **Minimum flags for all future runs: `--huber_delta 0.1 --ema_decay 0.99`**
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
- CLI flags available: `--warmup_epochs`, `--clip_norm`, `--huber_delta`, `--ema_decay`, `--slice_num`, `--n_layers`, `--surf_weight`, `--beta2`
