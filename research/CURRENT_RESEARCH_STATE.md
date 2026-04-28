# SENPAI Research State

- **Date:** 2026-04-28 03:35
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **73.91** | PR #464 (alphonse, full stack with grad-clip 0.5) — merged |
| `test_avg/mae_surf_p` (3-split mean) | 70.37 | PR #464 |

Per-split val on the new baseline: `val_single_in_dist=81.66`, `val_geom_camber_rc=87.95`, `val_geom_camber_cruise=54.47`, `val_re_rand=71.55`.

**Five orthogonal axes stacked, with the clip refinement:** L1 loss × linear warmup → cosine, peak `lr=1e-3`, `--epochs 14` × 8-band Fourier on positions × `surf_weight=30` × **`grad_clip_norm=0.5`**.

## Current research focus

Diminishing returns on regularization-style hypotheses are now well-established:
- **Saturated** (worked on simpler stacks, neutral or hurt on full stack): EMA (#303), wd=5e-4 (#385).
- **Marginal** (cost roughly equals benefit): dsdf-Fourier (#414), Huber-beta-1.0 (#364, but beta=0.5 still pending).
- **Falsified outright**: surf_p_extra=3.0 (#444), surf_p_weight=5 (#278) — pressure-channel-emphasis-on-shared-backbone.

The path forward is now diversified across:
1. **Attack the binding wall-clock constraint directly** — bf16 mixed precision (alphonse #496) for more epochs.
2. **Different regularization mechanisms** — mesh node loss subsampling (fern #497, data augmentation rather than weight-magnitude regularization).
3. **Refined hypothesis tests** — Huber beta=0.5 (edward #364), trainable random Fourier (thorfinn #470), attention dropout (nezuko #471), drop-path (askeladd #369), ckpt-avg (frieren #380), higher LR (tanjiro #473).

The val curve still descends strict-monotone at the 14-epoch budget cap across every winning PR, strongly suggesting we are budget-limited, not capacity-limited or regularization-limited. bf16 directly attacks this.

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for **every** PR. Diagnosed independently 9+ times; `data/scoring.py` is read-only. Rank by 3-clean-split test mean.

## Open PRs

### Round 6 (status:wip, on top of full current stack)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #496 | Throughput | alphonse | bf16 mixed precision (autocast) + epochs 14→24 |
| #497 | Data aug | fern | Mesh node random loss subsampling (keep 85%) |

### Round 5 (status:wip, on top of full current stack)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #470 | Feature | thorfinn | Trainable random Fourier projection (Tancik 2020) |
| #471 | Regularization | nezuko | Attention dropout 0.05 in PhysicsAttention |
| #473 | Schedule | tanjiro | Higher peak LR (1e-3 → 2e-3) protected by grad-clip |

### Sent back to rebase onto current advisor (status:wip)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #364 | Loss | edward | Huber smooth_l1 **beta=0.5** (refined from beta=1.0) |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging top-3 + val-on-averaged |

### Round 2 carry-over (status:wip, on stale baseline)

| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #369 | Regularization | askeladd | Drop-path 0.1 on attention + MLP residuals |

## Round-1+...+5 ranking (val_avg/mae_surf_p)

| Rank | PR | Student | Stack | val_avg | Verdict |
|---:|----|---------|-------|---------:|---------|
| 1 | #464 | alphonse | full + grad-clip-0.5 | **73.91** | **Merged (current baseline)** |
| 2 | #387 | alphonse | full + grad-clip-1.0 | 74.44 | Merged (previous baseline) |
| 3 | #385 (rerun) | fern | full + wd=5e-4 | 75.94 | Closed (saturated) |
| 4 | #414 (rerun) | thorfinn | sw=30 baseline + dsdf-Fourier | 76.17 | Closed (marginal) |
| 5 | #301 | nezuko | L1+warmup+Fourier+sw=30 | 76.68 | Merged |
| 6 | #444 | nezuko | sw=30 + surf_p_extra=3 | 77.92 | Closed (falsified) |
| 7 | #303 (rerun) | tanjiro | full + EMA | 82.44 | Closed (regime-saturated) |
| 8 | #364 | edward | L1+warmup+Fourier + Huber-1.0 | 85.58 | Sent back, refined to beta=0.5 |
| 9 | #365 | thorfinn | L1+warmup+Fourier | 87.86 | Merged |
| 10 | #296 | fern | L1+warmup+budget | 94.54 | Merged |
| 11 | #293 | edward | L1 only | 101.87 | Merged |

## Notable directional findings (running list)

1. **Regularization saturation on the full stack:** wd=5e-4, EMA both worked on simpler stacks but saturated/hurt on the full stack. Pattern: each new regularizer absorbs some "training noise" budget previous regularizers were after.

2. **Pressure-channel-emphasis loss tweaks fail on shared-backbone models:** two independent falsifications (alphonse's surf_p_weight=5, nezuko's surf_p_extra=3.0). Even when individual channel weights stay at 1×, raising total surface loss magnitude pulls shared backbone capacity onto the surface-p subspace. Future work would need separate decoder heads.

3. **Clipping is per-step magnitude bound (PR #387 / #464 telemetry):** pre-clip ‖∇‖ identical at clip=1.0 vs clip=0.5 (peak ~270, end ~60). Optimizer sees identical gradients; only update magnitudes differ.

4. **`val_geom_camber_rc` Fourier anomaly persists:** improved least under Fourier (−1.0% vs −8.5% to −10.8% on others), regressed slightly under tighter clipping. Most likely explanation: residual is geometry-extrapolation-dominated. Worth a future targeted hypothesis (domain conditioning, per-Re conditioning, test-time augmentation).

5. **Volume regression with surf_weight=30 (PR #301):** `val_avg/mae_vol_p` regressed +13.2%. Volume isn't ranked.

6. **Budget-limited, not capacity-limited:** every winning PR ends with val curve still descending. bf16 PR #496 attacks this directly.

## Potential next research directions

When the round-5/6 PRs land:
- **Stack new winners** if any of the round-5 PRs (Huber-beta-0.5, attn-dropout, drop-path, ckpt-avg, higher LR, trainable Fourier) helps.
- **`val_geom_camber_rc` deep-dive** — domain conditioning, per-Re conditioning, test-time augmentation. The persistent failure mode that current axes don't touch.
- **Output residual from a free-stream estimate** for `Ux, Uy`.
- **Per-channel volume weighting** — addresses the volume regression flag from PR #301.
- **Capacity bumps revisited** — if bf16 succeeds, deeper/wider models become tractable in the budget. Round-1's deeper-8-layers and wider-192 would benefit from re-evaluation.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
- For PRs that are CLI-flag-only changes (no train.py diff), the Config default is updated on the advisor branch in a follow-up commit at merge time so future PRs reproduce the new baseline without explicit flags.
- Per-epoch grad-norm telemetry (`train/grad_norm_avg`) is in the merged train.py — every PR's metrics.jsonl includes it.
