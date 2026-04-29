# SENPAI Research State

- **Updated:** 2026-04-29 00:35 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #769 (Huber δ=0.5, no EMA) — MERGED:** `val_avg/mae_surf_p = 102.86`, `test_avg/mae_surf_p = 94.83`  
**−13.8% val / −12.8% test vs EMA-only baseline.** Largest single win on the track.  
Future runs must include `--huber_delta 0.5`. For maximum effect, also add `--ema_decay 0.99` (stack TBD).

**Previous baseline (PR #773, EMA-only):** val=119.35, test=108.79 — still merged, still useful if Huber doesn't stack.

## Current research focus

Two large independent wins confirmed — now testing whether they compound:
- **Huber δ=0.5 (PR #769, MERGED):** −13.8% val. Outlier robustness via linear penalty for large residuals. Gains concentrate on heavy-tailed OOD splits (rc −17.8%, re_rand −14.7% on test). δ monotone: smaller is better up to at least 0.5; δ<0.5 unexplored.
- **EMA decay=0.99 (PR #773, MERGED):** −15.4% vs unmodified default. Weight averaging for flatter minima.
- **Gradient norms** are persistently large (pre-clip median ~60, p95 ~268 throughout training). Clipping is load-bearing — confirmed by nezuko's PR #775 sweep.
- **Budget:** 14 epochs / 30 min. Cosine LR at epoch 14 is only at 4.1e-4 / 50-ep schedule = massive mismatch. Schedule alignment (thorfinn #860) is a key pending test.

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #881 | alphonse | Huber δ ∈ {0.1,0.25,0.5} + EMA stack — confirm compound | Status:WIP |
| #775 | nezuko | warmup+clip — rebase + EMA stack test (clip0.5 won without EMA) | Status:WIP (rebase) |
| #859 | fern | Surface weight scan — rebase + Huber+EMA stack {sw∈10,15,20,30} | Status:WIP (rebase) |
| #867 | edward | AdamW β₂ scan {0.95,0.99,0.999,0.9999} + EMA | Status:WIP |
| #862 | frieren | Slice scan {64,96,128,192} on slim model + EMA | Status:WIP |
| #860 | thorfinn | Schedule alignment: cosine T_max=14 vs OneCycleLR + EMA | Status:WIP |
| #776 | tanjiro | Deeper model n_layers=8 | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |

## Key pending questions (expected within 2 poll cycles)

1. **Do Huber + EMA stack?** (alphonse PR #881). If yes, expected ~95-98 val.
2. **Does clip=0.5 stack with Huber + EMA?** (nezuko PR #775 rebase). If both wins stack, expected ~88-93 val.
3. **Does sw=20 stack with Huber + EMA?** (fern PR #859 rebase). Huber may already upweight surface effectively (volume residuals are larger → more Huber attenuation), so optimum sw might shift down.
4. **Is δ < 0.5 better?** (alphonse PR #881 tests 0.25, 0.1). δ trend is monotone; unexplored below 0.5.
5. **Does schedule alignment help?** (thorfinn PR #860). LR never anneals below 4.1e-4 in current budget; cosine-T14 or OneCycle may release 3-8% more gains.
6. **Is slice_num > 64 better with EMA + Huber?** (frieren PR #862). 2x-slices win was pre-EMA and worse than EMA; need a clean comparison with EMA on.

## Potential next research directions

**Confirmed wins to compound further:**
- Huber+EMA+clip triple-stack (pending validation from #881 + #775)
- Huber δ scan continues below 0.5 — δ=0.25 may yield another 2-4%
- Per-channel δ: δ_p smaller than δ_Ux/Uy (pressure tails are wider than velocity tails)

**New directions (not yet tested):**
- **AdamW β₂ scan** (edward PR #867 in flight) — heavy-tailed grads motivate faster second-moment
- **Pressure-channel-only loss tail** — drop Ux/Uy loss in final epochs once those channels converge
- **Cross-attention surface↔volume** — boundary condition inductive bias
- **Mesh subsampling for throughput** — more data per wallclock; could unlock wider/deeper models
- **Inverse channel weighting** (explicit upweight surface p vs Ux/Uy) — flagged in UW closure

**Architectural:**
- Multi-task curriculum (Ux/Uy first, freeze, fine-tune p)
- Mixed precision + gradient accumulation for larger effective batch
- Galerkin attention swap

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling in `train.py`.
- One hypothesis per PR. Compound only after isolated wins verified.
- **Default flags for all future runs: `--huber_delta 0.5 --ema_decay 0.99`** (both wins merged).
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
