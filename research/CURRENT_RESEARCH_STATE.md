# SENPAI Research State

- 2026-04-29 (round 4 mid-cycle; 3 PRs reviewed: #966 closed dead end, #997 accidentally merged empty, nezuko idle reassigned; fern #1009 and nezuko #1010 newly assigned)
- No recent research direction from human researcher team (no open GitHub issues found)

## Current Focus

**Track: icml-appendix-charlie-pai2e-r2**

### Working Baseline

| Metric | Value | PR | Notes |
|--------|-------|----|-------|
| `val_avg/mae_surf_p` | **94.7833** | #931 | Per-sample Re-weighted loss; epoch 14/50; ckpt_avg K=3; T_max=15, max_norm=5.0 |

#### Per-split breakdown (PR #931, ckpt_avg epochs 12-13-14):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| `val_single_in_dist` | 104.91 | 96.86 |
| `val_geom_camber_rc` | 105.49 | 93.28 |
| `val_geom_camber_cruise` | 77.32 | 64.54 |
| `val_re_rand` | 91.41 | 86.18 |
| **avg** | **94.7833** | **85.22** |

### Baseline Configuration History

| Date | PR | val_avg/mae_surf_p | Key Change |
|------|----|--------------------|------------|
| 2026-04-28 | #931 | **94.7833** | + per-sample Re-weighted loss (w_i=1/log_Re) |
| 2026-04-29 | #911 | 97.5181 | + T_max=15 + max_norm=5.0 compound fix |
| 2026-04-28 | #899 | 104.6986 | + checkpoint averaging K=3 |
| 2026-04-28 | #778 | 104.7457 | + clip_grad_norm=1.0 (first big win, -24%) |
| 2026-04-28 | #764 | 137.0013 | First measured number (n_hidden=256, epoch 9, undertrained) |

### Active WIP Experiments

| PR | Student | Hypothesis | Target / Motivation |
|----|---------|------------|---------------------|
| #978 | charliepai2e2-askeladd | T_max=18 on compound baseline | Follow-up to T_max=20 (95.84 on compound stack); T_max=18 is midpoint between winning 20 and baseline 15; ckpt_avg K=3 anti-correlation mechanism |
| #981 | charliepai2e2-frieren | Re × domain stratified sampler | Joint Re + domain batch diversity; compounds with per-sample Re-weighting (batch-min formula more meaningful when batch is Re-diverse) |
| #985 | charliepai2e2-thorfinn | AdamW: exclude bias/LayerNorm from weight decay | Proper parameter group decoupling; WD sweet spot between 1e-5 baseline and 1e-2 test |
| #992 | charliepai2e2-tanjiro | Re-weighting alpha sweep: alpha=1.25 and alpha=1.5 | Fine-tune exponent; alpha=1 wins, alpha=2 dead end (-6.86%); unexplored 1.25–1.5 space |
| #996 | charliepai2e2-alphonse | Curriculum by Re-regime: 3-stage epoch curriculum | Stages: bottom 33% Re (epochs 1–5), bottom 67% (6–10), full dataset (11–14); Re-weighting alpha=1 throughout |
| #1001 | charliepai2e2-edward | n_head=2 / head_dim=64 (wider heads) | n_head=8 (PR #974) regressed +15.2%; opposite direction: fewer, wider heads; more context per head |
| #1009 | charliepai2e2-fern | Relative MAE on surface pressure channel | Node-level normalization: `\|pred-true\|/(|true|+eps)`; orthogonal to Re-weighting (sample-level) and Huber (clips residuals); eps=0.1 |
| #1010 | charliepai2e2-nezuko | n_layers=8, T_max=12 (budget-aligned depth test) | PR #766 (no clipping, 160.24) incomparable; compound stack tames explosion; ~40% slower epochs → ~12 in budget; T_max=12 for full cosine execution |

8 students have active WIP assignments.

### Closed / Rejected Experiments (this round)

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #997 | charliepai2e2-edward: Huber loss on pressure channel (delta=1.0) | ACCIDENTAL MERGE — branch contained only assignment commit; no experiment ran; baseline unchanged |
| #974 | charliepai2e2-nezuko: n_head=8 / head_dim=16 | CLOSED: val_avg=109.18 (+15.2%); head_dim=16 too narrow; 36% slower (only 11/14 epochs). Negative: n_head=2 / head_dim=64 now being tested in #1001 |
| #967 | charliepai2e2-alphonse: gradient accumulation N=4 | CLOSED: Dead end; high-Re samples still dominate within each micro-batch; no new signal |
| #966 | charliepai2e2-fern: n_hidden=256 + T_max=12 | CLOSED: val_avg=110.04 (+16.1%); ~85% epoch slowdown → only 9 epochs vs 14 baseline; throughput deficit can't be overcome within 30-min budget |
| #965 | Relative MAE surf-p loss (P_EPS=1.0) | CLOSED: +87.9% regression; gradient explosion from near-zero pressure points; P_EPS=1.0 too small for normalized quantities — PR #1009 retries with eps=0.1 on actual pressure channel only |
| #964 | charliepai2e2-alphonse: Re-weighting alpha=2 | CLOSED: val_avg=101.64 (+6.86%); 20× weight ratio concentrates gradient on single low-Re sample per batch; alpha=1 near optimum |
| #930 | eta_min=1e-5 cosine floor | REJECTED: val_avg=100.70 (+3.18); model benefits from near-frozen final epoch at 1e-6 |
| #921 | charliepai2e2-askeladd: T_max sweep {20,25,30} | CLOSED: T_max=20→99.75 was regression on OLD baseline; now retested as #978 on compound stack |

### Research Themes Currently Active

1. **Re-weighting and loss reformulation** — building on PR #931's per-sample Re-weighting win:
   - Alpha sweep: PR #992 (alpha=1.25 and 1.5) — bridge gap between winning alpha=1 and dead-end alpha=2
   - Relative MAE loss: PR #1009 — node-level pressure normalization (retries the direction of PR #965 with proper eps=0.1)
   - These are orthogonal: Re-weighting acts at sample level, relative MAE acts at node level within a sample

2. **Training dynamics: curriculum and sampling** — when and how samples are presented:
   - Curriculum by Re: PR #996 — structured low→high Re exposure over epochs
   - Stratified sampling: PR #981 — batch diversity for more stable Re-weighting

3. **Architecture and capacity** — within the 30-min budget constraint:
   - Depth: PR #1010 (n_layers=8, budget-aligned T_max=12) — first test with compound stack
   - Attention heads: PR #1001 (n_head=2 / head_dim=64) — orthogonal to closed n_head=8 failure
   - n_hidden=256 is a **dead end** at this time budget (only 9 epochs possible)

4. **LR schedule fine-tuning** — PR #978 (T_max=18) — systematic exploration of schedule after T_max=20 showed +1.05 improvement on compound stack

5. **Regularization tuning** — PR #985 (AdamW bias/LayerNorm decoupling + WD sweet spot)

## Key Insights

1. **Gradient explosion from high-Re samples was the dominant failure mode** — pre-clip norms 40–900× above threshold. Fixed by clipping (max_norm=1.0 → 5.0).
2. **LR schedule mismatch was structural** — T_max=50 with ~14 epoch budget means LR stays at 84% of initial throughout. Aligned to T_max=15 in PR #911.
3. **Per-sample Re-weighting is additive to clipping** — 1/(log_re−min+1) weighting reduces grad norms by 2.6× without suppressing gradient direction. w_i range (0.17–0.39) gives a 4.5–4.9× ratio — meaningful but not extreme.
4. **Checkpoint averaging (ckpt_avg K=3) is free improvement** — zero training cost. Standard on all experiments.
5. **Wall-clock budget is the binding constraint for capacity** — n_hidden=256 gets only 9 epochs (vs 14 for baseline); 4× params, 74 GB peak memory. Capacity wins only when epoch count is matched.
6. **OOD split asymmetry is systematic**: Any improvement must be net-positive across all 4 splits (single_in_dist, geom_camber_rc, geom_camber_cruise, re_rand).
7. **Grad norm monitoring is a proxy for Re-regime health** — watch mean pre-clip norms as a training diagnostic.

## High-Priority Backlog (not yet assigned)

1. **Multi-task auxiliary loss: add Re prediction head** — small MLP predicts log(Re) from encoder; forces encoder to disentangle Re from geometry; may improve OOD generalization.
2. **Input feature dropout (geometry dropout)** — zero foil-2 geometry features with p=0.2; may improve val_geom_camber splits.
3. **slice_num ablation** — try slice_num∈{32, 96, 128}; physics partitioning directly affects mesh node grouping for tandem foil.
4. **Architecture: U-Net style skip connections** — residual connections between Transolver layers 1↔5, 2↔4 to preserve low-frequency flow features.
5. **SGDR warm restarts** — CosineAnnealingWarmRestarts T_0=5, T_mult=2 would give restart at epochs 5, 15 within budget.
