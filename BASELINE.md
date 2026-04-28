# Baseline — willow-pai2d-r1

Canonical reference for `icml-appendix-willow-pai2d-r1`. Lower
`val_avg/mae_surf_p` is better; round-1 ranking is by best validation
checkpoint, with `test_avg/mae_surf_p` reported as the paper-facing number.

## Current best (PR #359, alphonse, 2026-04-28)

bf16 autocast on forward + loss in `train.py`. Same model, same optimizer,
same schedule; just unlocks the 30-min wall-clock cap.

- **`val_avg/mae_surf_p` = 121.8478** at epoch 16 (of 19 completed)
- **`test_avg/mae_surf_p` = 111.1495** (best val checkpoint)
- W&B run: [`ot9decu8` / `bf16-bsz4`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r1/runs/ot9decu8)
- Per-epoch wall: 96–99 s (vs ~131 s pre-bf16, **−26%**)
- Peak GPU memory: 32.9 GB / 96 GB (~63 GB headroom)
- Wall: 30-min `SENPAI_TIMEOUT_MINUTES` binding at 19/50 epochs.

### Per-split surface MAE (val, best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| val_single_in_dist | 141.2387 | 1.5993 | 0.8116 |
| val_geom_camber_rc | 130.2818 | 2.7214 | 1.0283 |
| val_geom_camber_cruise |  99.8260 | 1.2243 | 0.6061 |
| val_re_rand | 116.0448 | 1.9411 | 0.8464 |
| **val_avg** | **121.8478** | 1.8715 | 0.8231 |

### Per-split surface MAE (test, best val checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 123.7300 | 1.5057 | 0.7837 |
| test_geom_camber_rc | 121.5380 | 2.6987 | 0.9655 |
| test_geom_camber_cruise |  85.6478 | 1.1834 | 0.5640 |
| test_re_rand | 113.6821 | 1.7717 | 0.8290 |
| **test_avg** | **111.1495** | 1.7898 | 0.7855 |

## Default config (matches PR #359)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- AdamW + CosineAnnealingLR(T_max=epochs), no warmup
- Loss: per-channel-equal MSE in normalized space, with surface vs. volume
  split via `surf_weight`. **Forward + loss inside `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.**
- Model: Transolver (`n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`,
  `mlp_ratio=2`, `space_dim=2`, `fun_dim=22`, `out_dim=3`)

## Reproduce

```bash
cd target && python train.py \
  --epochs 50 --batch_size 4 --lr 5e-4 \
  --surf_weight 10.0 --weight_decay 1e-4 \
  --agent baseline \
  --wandb_group baseline-bf16-r1 \
  --wandb_name baseline-bf16-r1
```

## Notes

- Primary ranking metric: `val_avg/mae_surf_p`. Lower is better.
- Test-time number reported alongside: `test_avg/mae_surf_p`.
- Both are equal-weight means across the four val/test tracks.
- 30-min wall-clock cap (`SENPAI_TIMEOUT_MINUTES=30`) is **still binding** at
  this configuration: 19/50 epochs ran. The cosine LR schedule was specified
  for 50 epochs, so lr at the best epoch (16) was still ~78% of peak (~3.9e-4).
  Cosine T_max alignment is the obvious next throughput-side lever.
- VRAM headroom is now ~63 GB — capacity scale-up experiments that previously
  couldn't converge under 30 min may now be feasible.
- `data/scoring.py` patched in commit `b78f404` to filter non-finite-y samples;
  `test_avg/mae_surf_p` is finite for all sibling round-1 PRs.
- Cosmetic: `train.py::evaluate_split`'s normalised-loss accumulator still
  prints NaN for `test_geom_camber_cruise` (the one Inf-pressure sample),
  because that path doesn't use `data/scoring.py`'s filter. Surface/volume MAE
  rankings are correct.

## Prior baseline (PR #312, superseded)

Default Transolver, no bf16. val_avg = 144.2118, test_avg = 131.1823 at
14/50 epochs. Superseded by PR #359 on 2026-04-28.
