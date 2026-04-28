# SENPAI Research State

- **Date:** 2026-04-28 00:10
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **94.54** | PR #296 (fern, L1 + warmup + budget-matched cosine) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 91.85 | PR #296 |

Per-split val on the new baseline: `val_single_in_dist=114.30`, `val_geom_camber_rc=105.46`, `val_geom_camber_cruise=70.45`, `val_re_rand=87.96`.

Two orthogonal axes now stacked: **L1 loss** in normalized space (PR #293) × **linear warmup → cosine** with peak `lr=1e-3` and `--epochs 14` budget-matched (PR #296). Everything else at original `train.py` defaults: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `weight_decay=1e-4`, `surf_weight=10.0`, `batch_size=4`.

## Current research focus

The first two round-1 wins were both *cheap-per-epoch, training-dynamics* changes. The capacity-heavy round-1 hypotheses (`n_hidden=192`, `n_layers=8`, `slice_num=128 + n_head=8`) all closed with the same diagnosis: per-epoch wall time too high to compete with cheaper changes inside the 30-min `SENPAI_TIMEOUT_MINUTES` cap.

Round 2 is now a stacking exercise: each open hypothesis attaches to the L1+warmup baseline and tests an orthogonal axis. The remaining unmeasured axes are:
- **Loss formulation refinements** (Huber, per-channel weighting)
- **Regularization** (drop-path, weight decay, EMA)
- **Feature augmentation** (Fourier positional)
- **Checkpoint / weight averaging** (top-K best-val averaging, EMA)

These compose well — the binding constraint is that each one needs to be measured individually before stacking the winners.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR. Independent diagnoses from edward (#293), alphonse (#278), nezuko (#301), askeladd (#290), frieren (#299), tanjiro (#303), fern (#296): test sample 20 has 761 non-finite values in the volume `p` channel of GT; `data/scoring.accumulate_batch` computes `(pred_orig - y).abs()` before masking, so `NaN * 0 = NaN` propagates. `data/scoring.py` is read-only per program constraints. Rank PRs by the **3-clean-split test mean** alongside `val_avg/mae_surf_p`.

## Open PRs

### Sent back to rebase onto L1+warmup baseline (status:wip)

| PR | Axis | Student | Hypothesis | Why send-back |
|----|------|---------|------------|---------------|
| #278 | Loss | alphonse | `surf_p_weight=5` (pressure-channel up-weight in surface loss) | Originally measured on MSE; rebase onto L1+warmup |
| #301 | Loss | nezuko | `surf_weight=30` (surface emphasis) | Originally measured on MSE; rebase onto L1+warmup |
| #303 | Weights | tanjiro | EMA weights (decay 0.999) | Originally measured on MSE; rebase onto L1+warmup. EMA-vs-live diagnostic confirmed predicted 4.66% delta. |

### Round 2 / 3 (status:wip, on top of L1 — pre-PR #296 merge)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber (smooth_l1, beta=1.0) — quadratic-near-zero on top of L1 |
| #365 | Feature | thorfinn | Fourier positional features (8 freqs on normalized `x, z`) |
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging (top-3, post-hoc) |

These are all on the L1 (pre-warmup) baseline. When they land, expect either a clean win (if their axis stacks straightforwardly with warmup, e.g. drop-path, ckpt-avg, EMA) or a send-back-for-rebase if the schedule interaction matters (e.g. Huber with warmup may differ from Huber without).

### Round 3 (status:wip, on top of L1+warmup baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #385 | Regularization | fern | `weight_decay` 1e-4 → 5e-4 |

## Round-1+2 ranking (val_avg/mae_surf_p, lower is better)

| Rank | PR | Student | Stack | val_avg | Verdict |
|---:|----|---------|-------|---------:|---------|
| 1 | #296 | fern | L1 + warmup + budget | **94.54** | **Merged (new baseline)** |
| 2 | #293 | edward | L1 only | 101.87 | Merged (previous baseline) |
| 3 | #303 | tanjiro | EMA on MSE | 127.65 | Sent back (rebase to L1+warmup) |
| 4 | #296 (run 1) | fern | warmup-1e3 on MSE | 137.32 | Superseded by rerun |
| 5 | #299 | frieren | n_layers=8 (best of 2) | 139.29 | Closed (budget penalty) |
| 6 | #301 | nezuko | surf_weight=30 on MSE | 141.56 | Sent back |
| 7 | #290 | askeladd | wider 192 | 152.24 | Closed (budget penalty) |
| 8 | #278 | alphonse | surf_p_weight=5 on MSE | 156.16 | Sent back |
| 9 | #305 | thorfinn | slices+heads 2x | 160.68 | Closed (budget + dim_head=16 instability) |

## Potential next research directions

When the round-2/3 PRs land:
- **Stack winners.** L1 + warmup × {Huber? drop-path? Fourier? ckpt-avg? EMA? wd=5e-4?} — all designed to compose orthogonally.
- **Schedule refinements once full stack is known.** `--epochs 13` (tighter), `peak lr=2e-3` (more aggressive), `eta_min > 0` cosine end (don't anneal all the way to zero).
- **Mesh-aware augmentation.** Random node-loss subsampling (model is permutation-invariant, so this is "free" regularization).
- **Domain conditioning.** Explicit token / FiLM on (raceCar single | raceCar tandem | cruise tandem).
- **Output residual from a free-stream estimate** for `Ux, Uy` — reduces dynamic range to learn from scratch.
- **Per-channel volume weighting** — currently `vol_loss` treats Ux/Uy/p equally; pressure dominates the eval metric.
- **Mixed precision (bf16 autocast)** — speedup → more epochs in budget. Risk: precision loss on small residuals at convergence; needs careful op-level control.
- **Gradient clipping.** Cheap robustness.
- **Best-val cosine annealing with warm restarts** (SGDR) within the budget.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
