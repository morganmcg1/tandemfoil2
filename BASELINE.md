# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #416, alphonse, 2026-04-28)

`torch.compile(model, dynamic=True)` stacked on top of bf16 + FF K=8.
Single one-time recompile at train→eval flip; no shape recompiles across
the 74K-242K node range; FF concat does not trip dynamic-shape
specialization. 2.0× per-epoch speedup unlocks 37/50 epochs in the 30-min
cap, letting cosine actually decay.

- **`val_avg/mae_surf_p` = 80.8506** at epoch 37 (of 37 completed)
- **`test_avg/mae_surf_p` = 73.4107** (best val checkpoint)
- W&B run: [`ewq3guz2` / `compile-bf16-ff-bsz4`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/ewq3guz2)
- Per-epoch wall: ~49 s steady state (cold compile 27.85 s + 84 s for epoch 1)
- Peak GPU memory: 24.1 GB / 96 GB (~70 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 37/50 epochs (was 19/50 pre-compile).

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 84.2016 | 1.1797 | 0.5843 |
| val_geom_camber_rc | 93.3940 | 1.7614 | 0.7951 |
| val_geom_camber_cruise | 65.9494 | 0.8740 | 0.4909 |
| val_re_rand | 79.8573 | 1.3063 | 0.6400 |
| **val_avg** | **80.8506** | 1.2804 | 0.6276 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 76.5973 | 1.1318 | 0.5759 |
| test_geom_camber_rc | 83.4492 | 1.7213 | 0.7486 |
| test_geom_camber_cruise | 56.7227 | 0.8223 | 0.4433 |
| test_re_rand | 76.8738 | 1.2020 | 0.6088 |
| **test_avg** | **73.4107** | 1.2193 | 0.5942 |

### Updated per-split rc-camber understanding

The OOD `geom_camber_rc` split — which FF on its own only relieved by
−3.3% — gets to −25.8% with compile + FF. So the rc-camber gap on the FF
baseline was *schedule-truncation-bound*, not a fundamental representation
bottleneck. **Camber-aware feature embedding is no longer queued** as the
rc-camber-targeted experiment for round 2; throughput unlock alone closed
the gap. Capacity scale-up may now matter more than camber feature work.

## Default config (matches PR #416)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- Loss: per-channel-equal MSE in normalized space, with surface vs. volume
  split via `surf_weight`. **Forward + loss inside `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.**
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
  --wandb_group baseline-compile-ff-bf16 \
  --wandb_name baseline-compile-ff-bf16
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- 30-min wall-clock cap is **still binding** at 37/50 epochs. Cosine T_max
  alignment is the obvious next throughput-side lever (currently in flight
  as fern's PR #407, but with `--epochs 20` matched to the *pre-compile*
  19-epoch achievable budget — likely needs adjustment to match the new
  37-epoch achievable budget).
- VRAM headroom is now ~70 GB. Capacity scale-up (PR #393 territory)
  becomes feasible again.
- `data/scoring.py` patched (`b78f404`).
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` — does not affect MAE rankings.

## Prior baselines (superseded)

- **PR #312** (alphonse, original): val_avg=144.21, test_avg=131.18.
  Default Transolver, no bf16/FF/compile.
- **PR #359** (alphonse, bf16): val_avg=121.85, test_avg=111.15. bf16
  autocast on forward + loss. Superseded by PR #327 on 2026-04-28.
- **PR #327** (tanjiro, FF K=8): val_avg=106.92, test_avg=96.82. Sinusoidal
  Fourier features for (x, z), K=8. Superseded by PR #416 on 2026-04-28.

Cumulative improvement: **−44.0% on val_avg** (144.21 → 80.85), **−44.0%
on test_avg** (131.18 → 73.41) since the original PR #312 reference.
