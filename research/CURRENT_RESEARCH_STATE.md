# SENPAI Research State

- **Updated:** 2026-04-28 22:15 UTC
- **Track:** `icml-appendix-willow-pai2e-r1` (TandemFoilSet ICML appendix, Willow PAI2E Round 1)
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1`
- **Most recent direction from human researcher team:** _(none — no open ADVISOR issues)_

## Current best

**PR #773 (EMA decay=0.99) — MERGED:** `val_avg/mae_surf_p = 119.35`, `test_avg/mae_surf_p = 108.79`  
EMA is part of the advisor baseline. All future student runs should use `--ema_decay 0.99`.

**Pending win (not yet baseline):** PR #775 (nezuko warmup5-clip0.5) hit val_avg=115.01 / test_avg=101.64 WITHOUT EMA, beating the EMA baseline by 3.6%. Sent back for rebase + EMA stack test before merge.

## Current research focus

Round 1 results have settled. Key findings:
- **EMA (decay=0.99)** is confirmed as an independent win (+6% over live model). New baseline.
- **Gradient clipping (max_norm=0.5)** is confirmed as an independent win (+3.6% over EMA baseline, WITHOUT EMA). Acts as continuous regularizer, not just early stabilization. Dominant over warmup.
- **Wider model (n_hidden=256)** is budget-incompatible. Only reaches 6-7 epochs before timeout.
- **Slice doubling (slice_num=128)** on slim model is a promising direction — 19.8% test gain vs wider model in PR #774 sweep, nearly free in params.
- **Input jitter** on log(Re) uniformly hurts under 14-epoch budget. Closed.
- **Gradient norm profile**: pre-clip norms median ~60, p95 ~268 throughout training. TandemFoilSet has inherently large gradients; clipping is load-bearing all the way through.

## Active PRs (WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #775 | nezuko | warmup+clip — sent back for rebase + EMA stack test | Status:WIP (rebase in progress) |
| #862 | frieren | Slice scan {64,96,128,192} on slim model + EMA decay=0.99 | Newly assigned |
| #859 | fern | Surface weight scan {10,20,50,100} + EMA | Status:WIP |
| #860 | thorfinn | Schedule alignment: cosine T_max=14 vs OneCycleLR + EMA | Status:WIP |
| #846 | edward | Unmodified baseline run (clean reference) | Status:WIP |
| #776 | tanjiro | Deeper model n_layers=8 | Status:WIP |
| #770 | askeladd | Surface-aware slice routing in PhysicsAttention | Status:WIP |
| #769 | alphonse | Huber loss for outlier-robust pressure regression | Status:WIP |

## Potential next research directions (Round 2+)

**High priority — stack confirmed wins:**
- **EMA + clipping + warmup stack** (pending PR #775 rebase result). If EMA+clip stack, new baseline ~110-113.
- **EMA + clipping + slice-192** (pending PR #862). Slice direction may compound with clipping.
- **EMA + clipping + Huber** (pending PR #769). Huber dampens large-gradient outliers; may interact positively or negatively with clipping.
- **Clip norm scan** ∈ {0.1, 0.25, 0.5} once clip+EMA baseline confirmed — nezuko's own suggestion, pre-clip p95=268 suggests 0.5 may not be the optimum.

**Medium priority — new directions:**
- **OneCycleLR vs cosine-T14** (thorfinn PR #860 in flight). Budget mismatch is the most underexplored inefficiency.
- **AdamW beta2 scan** (0.95, 0.99, 0.999) — with large gradient variance, β₂ controls the effective memory of the second moment. β₂=0.99 may be more robust than default 0.999.
- **Pressure-channel only loss tail.** Once the model converges on Ux/Uy, drop vol loss entirely and optimize only surface p for the final few epochs.
- **Cross-attention surface→volume conditioning.** Surface nodes can act as boundary condition anchors; explicit cross-attention adds inductive bias.
- **Mesh subsampling for throughput.** Random node subsampling in early epochs to see more data per wallclock. Could unlock wider models.

**Longer-term:**
- Multi-task curriculum (Ux/Uy first, freeze, fine-tune p)
- Mixed precision + gradient accumulation for larger effective batch
- Galerkin attention swap for PhysicsAttention
- Test-time augmentation (mesh perturbation TTA)

## Standing constraints

- 30 min wall-clock per run (`SENPAI_TIMEOUT_MINUTES`), 50-epoch cap.
- 96 GB VRAM per GPU, batch_size=4 default; meshes up to 242K nodes.
- No edits to `data/`. All augmentation/sampling in `train.py`.
- One hypothesis per PR. Compound only after isolated wins verified.
- All future runs: `--ema_decay 0.99` is the default.
- NaN guard (commit 49c55ed) is in advisor — all new branches get it for free.
