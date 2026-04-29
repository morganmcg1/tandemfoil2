# SENPAI Research State

- 2026-04-29 (updated)
- No recent research direction from human researcher team (no open GitHub issues found)
- Current research focus and themes:

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

### In-Flight Experiments (WIP — as of 2026-04-29)

| PR | Student | Hypothesis | Target / Motivation |
|----|---------|------------|---------------------|
| #981 | charliepai2e2-frieren | Re × domain stratified sampler | Joint Re + domain batch diversity; compounds with per-sample Re-weighting (batch-min formula more meaningful when batch is Re-diverse) |
| #978 | charliepai2e2-askeladd | T_max=20 on compound baseline | Re-test optimal LR schedule on PR #931 stack (old baseline T_max=20 was optimum; now valid on compound) |
| #974 | charliepai2e2-nezuko | n_head=8 (4→8 attention heads) | More attention specialization for OOD splits |
| #968 | charliepai2e2-thorfinn | AdamW weight_decay=1e-3 | Sweet spot between 1e-4 and 1e-2 for OOD/in-dist balance |
| #967 | charliepai2e2-tanjiro | Gradient accumulation N=4 (effective batch=16) | Smoother gradients, better convergence in 14-epoch budget |
| #966 | charliepai2e2-fern | n_hidden=256 + T_max=12 (budget-aligned) | Capacity increase on full compound stack |
| #997 | charliepai2e2-edward | Huber loss on pressure channel (delta=1.0) | Node-level gradient clamping for pressure; orthogonal to per-sample Re-weighting; L2 near zero, L1 for large residuals |
| #996 | charliepai2e2-alphonse | Curriculum by Re-regime: low→high Re staged training | Staged training: train on low-Re first, add high-Re samples progressively |
| #992 | charliepai2e2-tanjiro | Re-weighting alpha sweep: test alpha=1.25 and alpha=1.5 | Fine-tune the Re-weighting strength |
| #985 | charliepai2e2-thorfinn | AdamW: exclude bias/LayerNorm from weight decay | Proper parameter group decoupling |
| #981 | charliepai2e2-frieren | Re × domain stratified sampler on compound stack | Joint Re + domain batch diversity |
| #978 | charliepai2e2-askeladd | T_max=20 on compound baseline | Re-test optimal LR schedule on PR #931 stack |
| #974 | charliepai2e2-nezuko | n_head=8 (4→8 attention heads) | More attention specialization for OOD splits |
| #966 | charliepai2e2-fern | n_hidden=256 + T_max=12 (budget-aligned) | Capacity increase on full compound stack |

8 students have active WIP assignments.

### Closed / Rejected Experiments (this round)

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #965 | Relative MAE surf-p loss (auto Re-regime normalization) | CLOSED: Dead end — +87.9% regression; gradient explosion from near-zero pressure points with P_EPS=1.0; relative MAE unworkable on quantities that cross zero |
| #930 | eta_min=1e-5 cosine floor | REJECTED: val_avg 100.70 (+3.18 vs baseline); model benefits from near-frozen final epoch at 1e-6 |
| #875 | charliepai2e2-frieren: schedule-to-budget T_max=14 | CLOSED (superseded by #911) |
| #921 | charliepai2e2-askeladd: T_max sweep {20,25,30} | CLOSED (stale; #911 established T_max=15 as baseline) |
| #906 | fern: SGDR warm restarts T_0=5, T_mult=2 | CLOSED (stale; schedule experiments concluded with #911) |
| #907 | thorfinn: Adaptive Gradient Clipping (AGC) | CLOSED (stale; max_norm=5.0 already merged in #911) |

### Research Themes Currently Active

1. **Re-weighting and loss reformulation** — building on PR #931's per-sample Re-weighting win:
   - Steeper alpha=2 version (#964)
   - Relative MAE loss for auto normalization (#965)
   - Re × domain stratified sampler to make batch-min Re-weighting more meaningful (#981)
   - These are orthogonal regularization and sampling axes.

2. **Architecture capacity** — n_hidden=256 (#966) and n_head=8 (#974) with the full compound stack.

3. **Gradient optimization quality** — gradient accumulation (#967) for smoother updates with effective batch size 16.

4. **Regularization tuning** — weight_decay sweep (#968); WD=1e-3 as sweet spot hypothesis.

5. **Batch sampling strategy** — Re-stratified sampling (#981) addresses the training dynamics axis: Re-diverse batches compound with the per-sample Re-weighting already in the compound baseline.

## Potential Next Research Directions

1. **EMA model weights** — checkpoint EMA-averaged model for evaluation; PR #856 was queued but may have been superseded. Particularly useful at the undertrained 14-epoch regime.

2. **Label smoothing of surface mask** — soft surface weighting based on signed distance instead of hard 0/1.

3. **Physics-informed regularizers** — divergence-free velocity constraint (continuity equation).

4. **Multi-scale attention** — combine coarse background zone with dense foil-zone features.

5. **Output head architecture** — separate per-channel output head (deeper, with its own normalization) for pressure channel.

6. **Batch size effects** — what if batch_size=8 with accumulation=2 (same effective BS=16 but different normalization) behaves differently than batch_size=4 + accum=4?

7. **Cosine schedule with eta_min sweep** — orthogonal axis to T_max: what is the optimal final LR given that we're in a 14-epoch regime? (Note: eta_min=1e-5 was tested/rejected in PR #930 — model needs near-zero LR at epoch 14 for fine-tuning effect.)

8. **Fixed-floor Re-weighting normalization** — the current `re_weight = 1/(log_re - log_re_min_batch + 1)` uses batch minimum (non-deterministic). Replace with global training-set minimum for deterministic, reproducible weighting.
