# SENPAI Research State

- **Date:** 2026-04-28 01:05
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **87.86** | PR #365 (thorfinn, L1 + warmup + Fourier) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 84.22 | PR #365 |

Per-split val on the new baseline: `val_single_in_dist=104.53`, `val_geom_camber_rc=104.44`, `val_geom_camber_cruise=62.81`, `val_re_rand=79.64`.

Three orthogonal axes now stacked: **L1 loss** (PR #293) × **linear warmup → cosine** with peak `lr=1e-3` and budget-matched `--epochs 14` (PR #296) × **8-band Fourier features on normalized (x, z)** (PR #365). All other knobs at originals: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `weight_decay=1e-4`, `surf_weight=10.0`, `batch_size=4`, `fun_dim=54` (post-Fourier).

## Current research focus

The first three round-1/2 wins were all *cheap-per-epoch* changes — confirming the binding constraint is wall-clock, not VRAM, and approaches that fit more useful work into 14 epochs win. Round 3 is now testing the next set of orthogonal axes on top of L1+warmup+Fourier:
- **Regularization variants** (drop-path, weight decay, gradient clip, EMA)
- **Loss refinements** (Huber)
- **Checkpoint averaging**
- **Surface emphasis** (surf_weight tweak)
- **Feature-augmentation extensions** (Fourier on dsdf)

The pattern holds — these compose orthogonally with the merged stack and individual improvements should be incremental but stackable.

## Notable directional finding

`val_geom_camber_rc` improved **least** (−1.0%) under Fourier features while every other split improved 8–11%. Position Fourier was *predicted* to help most on this split (high-frequency near-leading-edge variation on raceCar tandem M=6–8 holdout); empirically it helped least. Most likely explanation: this split's residual is dominated by **geometry-extrapolation** in camber space (M=6–8 not in train), not spectral bias of the MLP. Worth a future deep-dive — possibly a domain-conditioning or test-time augmentation hypothesis to specifically attack this mode.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR. Diagnosed independently by edward (#293), alphonse (#278), nezuko (#301), askeladd (#290), frieren (#299), tanjiro (#303), thorfinn (#365): test sample 20 has 761 non-finite values in volume `p` channel of GT; `data/scoring.accumulate_batch` computes `(pred_orig - y).abs()` before masking, so `NaN * 0 = NaN` propagates. `data/scoring.py` is read-only per program constraints. Rank by **3-clean-split test mean** alongside `val_avg/mae_surf_p`.

## Open PRs

### Sent back to rebase onto current advisor (status:wip)

These were sent back when the baseline was L1+warmup; they should now rebase onto the new L1+warmup+Fourier baseline.

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #301 | Loss | nezuko | `surf_weight=30` (surface emphasis) |
| #303 | Weights | tanjiro | EMA weights (decay 0.999) |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging top-3 + val-on-averaged eval |

### Round 3 (status:wip, on top of L1+warmup baseline; pre-Fourier merge)

These were assigned when the baseline was L1+warmup; when their results land they'll be evaluated against the new L1+warmup+Fourier baseline. Likely will need rebase if their hypothesis is plausibly orthogonal.

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #385 | Regularization | fern | `weight_decay` 1e-4 → 5e-4 |
| #387 | Stability | alphonse | Gradient clipping `max_norm=1.0` |

### Round 2 carry-over (status:wip, on top of L1 only — pre-warmup, pre-Fourier)

These were assigned when the baseline was L1 only. Will likely need rebase when results land.

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber (smooth_l1, beta=1.0) |
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |

### Round 4 (status:wip, on top of L1+warmup+Fourier baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #414 | Feature | thorfinn | Fourier features on dsdf channels (4 freqs, dims 2–11) |

## Round-1+2+3 ranking (val_avg/mae_surf_p)

| Rank | PR | Student | Stack | val_avg | Verdict |
|---:|----|---------|-------|---------:|---------|
| 1 | #365 | thorfinn | L1 + warmup + Fourier | **87.86** | **Merged (current baseline)** |
| 2 | #296 | fern | L1 + warmup + budget | 94.54 | Merged (previous baseline) |
| 3 | #293 | edward | L1 only | 101.87 | Merged (older baseline) |
| 4 | #380 | frieren | L1 + ckpt-avg (val of single best) | 104.43 | Sent back (rebase + val-on-averaged) |
| 5 | #278 | alphonse | L1 + surf_p_weight=5 | 108.63 | Closed (falsified at 5×) |
| 6 | #303 | tanjiro | EMA on MSE | 127.65 | Sent back (rebase) |
| 7 | #296 (run 1) | fern | warmup-1e3 on MSE | 137.32 | Superseded by rerun |
| 8 | #299 | frieren | n_layers=8 | 139.29 | Closed (budget penalty) |
| 9 | #301 | nezuko | surf_weight=30 on MSE | 141.56 | Sent back |
| 10 | #290 | askeladd | wider 192 | 152.24 | Closed (budget penalty) |
| 11 | #278 | alphonse | surf_p_weight=5 on MSE | 156.16 | Sent back, then re-ran post-L1, then closed |
| 12 | #305 | thorfinn | slices+heads 2x | 160.68 | Closed (budget + dim_head=16 instability) |

## Potential next research directions

When round-3+ PRs land:
- **Stack winners.** The current baseline already has 3 orthogonal axes stacked; adding Huber, drop-path, EMA, ckpt-avg, gradient clip, weight decay all compose plausibly orthogonally.
- **`val_geom_camber_rc` deep-dive.** Domain-conditioning, test-time augmentation, or per-sample Re-conditioning to specifically attack this geometry-extrapolation-dominated split.
- **Schedule refinements once full stack lands.** `--epochs 13` (tighter), `peak lr=2e-3` (more aggressive), `eta_min > 0` cosine end.
- **Mesh-aware augmentation.** Random node-loss subsampling.
- **Output residual from a free-stream estimate** for `Ux, Uy`.
- **Trainable Fourier projection** (Tancik 2020) — random Gaussian projection in place of dyadic frequencies, optionally learnable.
- **Mixed precision (bf16 autocast)** — enables more epochs in same budget.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
