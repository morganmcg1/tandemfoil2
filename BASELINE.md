# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #407, fern, 2026-04-28)

Cosine T_max alignment (`--epochs 37` instead of 50) on top of the Huber +
bf16 + FF + `torch.compile` stack from PR #314 v3. CLI-only change; no
code modifications. Cosine reaches lr=0 at epoch 37, six consecutive late
epochs (31-36) each set a new best — schedule mechanism worked exactly as
predicted.

- **`val_avg/mae_surf_p` = 69.7385** at epoch 36 (of 37 completed)
- **`test_avg/mae_surf_p` = 60.4829** (best val checkpoint)
- W&B run: [`7xvc5hfl` / `tmax37-on-huber-compile-ff`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/7xvc5hfl)
- Per-epoch wall: ~49 s steady state (same as PR #314 — schedule shape
  doesn't change compute)
- Peak GPU memory: 24.1 GB / 102.6 GB (~78 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 37/37 epochs (full
  schedule completed for the first time on this branch).

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 73.8820 | 0.9139 | 0.4760 |
| val_geom_camber_rc | 83.7924 | 1.4330 | 0.6757 |
| val_geom_camber_cruise | 52.5412 | 0.6111 | 0.4038 |
| val_re_rand | 68.7384 | 1.0232 | 0.5355 |
| **val_avg** | **69.7385** | 1.0003 | 0.5228 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 63.4753 | 0.9386 | 0.4560 |
| test_geom_camber_rc | 71.5669 | 1.3637 | 0.6191 |
| test_geom_camber_cruise | 45.5087 | 0.5877 | 0.3614 |
| test_re_rand | 61.3805 | 0.9134 | 0.4960 |
| **test_avg** | **60.4829** | 0.9508 | 0.4831 |

### Schedule sanity

LR at best epoch (36, T_cur=35): 3.60e-6 (0.72% of peak). LR at final
epoch (37): 9.01e-7. Schedule completes the cosine tail cleanly.

## Stack composition (cumulative wins)

| Round component | val_avg | Δ vs prior |
|---|---|---|
| PR #312 (original): default Transolver | 144.21 | — |
| PR #359 (alphonse): + bf16 autocast | 121.85 | −15.5% |
| PR #327 (tanjiro): + FF K=8 | 106.92 | −12.2% |
| PR #416 (alphonse): + `torch.compile(dynamic=True)` | 80.85 | −24.4% |
| PR #314 (edward): + SmoothL1 β=1.0 | 69.83 | −13.6% |
| **PR #407 (fern): + cosine T_max=37 alignment** | **69.74** | **−0.13% val / −2.0% test** |

Cumulative: **−51.7% on val_avg / −53.9% on test_avg** since PR #312.

## Default config (matches PR #407)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- **`--epochs 37`** (matches the achievable budget on the current stack —
  cosine T_max=epochs, decays to lr=0 by epoch 37).
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- **Loss**: SmoothL1 / Huber (β=1.0) per-element loss in normalized space,
  with surface vs. volume split via `surf_weight`. Inside
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

> **Note on `--epochs 37`:** The Config dataclass default is still
> `epochs: int = 50`. New experiments should explicitly pass `--epochs 37`
> on the command line to match the current canonical schedule. If a future
> merge changes per-epoch wall time meaningfully (e.g. another compile-
> style win, or a model-size change), re-evaluate the achievable epoch
> budget and update `--epochs N` in the reproduce command and in new
> assignments accordingly.

## Reproduce

```bash
cd target && python train.py \
  --epochs 37 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-tmax37-huber-compile-ff \
  --wandb_name baseline-tmax37-huber-compile-ff
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap binding at 37 epochs — schedule now fully aligned.
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
