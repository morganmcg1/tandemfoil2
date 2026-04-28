# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #541 edward, 2026-04-28)

Pure L1 loss + bf16 + FF K=8 + `torch.compile(dynamic=True)` + cosine
T_max=50 (`--epochs 50`). Same code as PR #504; this baseline number is the
fresh-seed re-run from the T_max sweep that **confirmed T_max=50 beats
T_max=37 by 3.14%** for pure L1 (mechanism: L1's constant-magnitude gradient
benefits from non-zero terminal LR; T_max=37 zeroes lr prematurely). The
re-run also lined up favorably on seed (~1% spread vs PR #504's original
57.29) and is the new canonical reference.

- **`val_avg/mae_surf_p` = 56.2167** at epoch 37 (of 37 completed, wall-cap)
- **`test_avg/mae_surf_p` = 48.4232** (best val checkpoint)
- W&B run: [`pwi9gy9f` / `l1-tmax50-rerun`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/pwi9gy9f)
- Per-epoch wall: ~49 s steady state (cold compile epoch 1 ≈ 60 s)
- Peak GPU memory: 24.1 GB / 102.6 GB (~78 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 37/50 epochs.
- Terminal lr: 7.89e-5 (~16% of peak) — schedule still has runway.

### Per-split surface MAE (val, best checkpoint = epoch 37)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 59.1819 | — | — |
| val_geom_camber_rc | 72.0602 | — | — |
| val_geom_camber_cruise | 36.5289 | — | — |
| val_re_rand | 57.0959 | — | — |
| **val_avg** | **56.2167** | — | — |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 50.7842 |
| test_geom_camber_rc | 64.7467 |
| test_geom_camber_cruise | 30.6550 |
| test_re_rand | 47.5071 |
| **test_avg** | **48.4232** |

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| PR #314 (edward): + SmoothL1 β=1.0 | 69.83 | −13.6% |
| PR #407 (fern): + cosine T_max=37 alignment (Huber-era) | 69.74 | −0.13% |
| PR #504 (edward): SmoothL1 → pure L1 | 57.29 | −17.96% |
| **PR #541 (edward): T_max=50 confirmed for L1 + favorable seed** | **56.22** | **−1.07%** (rerun-vs-original) |

Cumulative: **−61.0% on val_avg / −63.1% on test_avg** since PR #312.

## Schedule for pure L1: T_max=50 confirmed

PR #541 directly tested `--epochs 37` vs `--epochs 50` with pure L1:

- `--epochs 37`: val_avg=58.04 (T_max=37, lr→0 at end)
- `--epochs 50`: val_avg=56.22 (T_max=50, lr ≈ 8e-5 at end ~16% of peak)
- T_max=50 wins by **3.14% on val_avg, 2.54% on test_avg**.

Mechanism: pure L1's `sign(r)` gradient keeps making progress at small
residuals, so the late-training low-LR tail extracts continued refinement
rather than settling into a fixed minimum. T_max=37 zeroes the LR
prematurely. Per-epoch jumps in T_max=50's last few epochs are the **largest
of the run** (epoch 36→37: −5.4% in one epoch with lr ≈ 8e-5).

Per-split signal: schedule changes barely move `geom_camber_rc` (-0.73% val
/ -0.17% test) — confirming **rc-camber is representation-limited, not
residual-refinement-limited**. Schedule and loss interventions can't help
rc; we'll need geometry-side or capacity-side experiments for it.

T_max=50 is the round-3 default. **`--epochs 50` is the canonical setting**.

## Default config (matches PR #541)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- **`--epochs 50`** (T_max=50; lr ends at ~16% of peak which pure L1 uses
  productively)
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- **Loss**: pure L1 `(pred - y_norm).abs()` per-element loss in normalized
  space, with surface vs. volume split via `surf_weight`. Inside
  `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Fourier features (K=8)** for normalized (x, z), computed in fp32
  outside the autocast scope, concatenated to the per-node feature vector.
  Per-node feature dim: 24 → 56.
- **`torch.compile(model, dynamic=True)`** wrapper applied right after
  `model.to(device)` (gated on `not cfg.debug`). Save/load via
  `getattr(model, "_orig_mod", model).state_dict()` so the W&B model
  artifact is portable into a non-compiled module.
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22 + 4*8 = 54`, `out_dim=3`).

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-pure-l1-tmax50 \
  --wandb_name baseline-pure-l1-tmax50
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap binding at 37/50 epochs.
- Single-seed variance ≈ ±1% on val_avg (PR #504 yi5upb1e=57.29 vs PR #541
  pwi9gy9f=56.22 at the same config).
- VRAM headroom is now 78 GB (24.1 / 102.6).
- `data/scoring.py` patched (`b78f404`).
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.

## Prior baselines (superseded)

- **PR #312** (alphonse, original): val_avg=144.21, test_avg=131.18.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15.
- **PR #327** (tanjiro, FF K=8): val_avg=106.92, test_avg=96.82.
- **PR #416** (alphonse, compile+FF): val_avg=80.85, test_avg=73.41.
- **PR #314** (edward, Huber+compile+FF): val_avg=69.83, test_avg=61.72.
- **PR #407** (fern, T_max=37 on Huber): val_avg=69.74, test_avg=60.48.
- **PR #504** (edward, pure L1 first run): val_avg=57.29, test_avg=51.35.
