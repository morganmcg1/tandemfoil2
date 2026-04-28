# Baseline — TandemFoilSet (willow-pai2d-r5)

**Status:** Round 1 in flight. Sequential merges: PR #441 bf16 (commit `b605b44`) → PR #434 grad-clip max_norm=1.0 (commit `426b4c4`) → PR #413 surface-Huber δ=1.0 (commit `e35acdf`). New round-1 baseline: **2-seed mean = 90.98 ± 0.81** at 19 epochs (CV ~0.9% — best variance band of round 1, back to bf16-baseline-level tightness). Composition arithmetic: stack captures ~78% of perfect-additive (-26.4 vs -33.7 max), confirming Huber and grad-clip are **complements** attacking different parts of the optimization step. Several round-2 stack candidates in flight.

**Quirk note:** `max_norm=1.0` clips 100% of training steps at this regime (median pre-clip grad-norm ≈ 38). Effectively normalized-gradient training (Lion-like). Future students changing the optimizer or LR should be aware that the gradient-magnitude amplification effect is removed; lr values that would overshoot under unclipped MSE are safe here.

## Reference configuration (current `train.py` HEAD)

The baseline is the default Transolver in `train.py` at HEAD of `icml-appendix-willow-pai2d-r5`:

- **Model:** Transolver, `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2` (~0.67M params)
- **Optimizer:** AdamW `lr=5e-4`, `weight_decay=1e-4`
- **Schedule:** CosineAnnealingLR with `T_max=epochs`
- **Batch size:** 4
- **Loss:** MSE volume + Huber(δ=1.0) surface, in normalized space; `loss = vol_loss_mse + surf_weight * surf_loss_huber`, `surf_weight=10` (PR #413)
- **Mixed precision:** bf16 autocast in train + eval, fp32 cast before squaring loss + before denormalization (PR #441)
- **Gradient clipping:** `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` (PR #434)
- **Training:** `epochs=50`, capped by `SENPAI_TIMEOUT_MINUTES=30` wall-clock
- **Sampling:** `WeightedRandomSampler` over balanced domain weights

## Reproduce command

```bash
cd /workspace/senpai/target
python train.py --epochs 50
```

## Primary metric

**`val_avg/mae_surf_p`** — equal-weight mean of surface pressure MAE across the four validation splits:
- `val_single_in_dist/mae_surf_p`
- `val_geom_camber_rc/mae_surf_p`
- `val_geom_camber_cruise/mae_surf_p`
- `val_re_rand/mae_surf_p`

Lower is better. The matching test metric `test_avg/mae_surf_p` is computed at the end of every run from the best validation checkpoint.

## Best results

_(round 1 in flight; baseline distribution being established by thorfinn's PR #428)_

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| **#413** | **90.98 ± 0.81** (n=2) | 88.16 (3-finite-split) * | + surface Huber δ=1.0; -9.4% vs #434; complements grad-clip (~78% of perfect-additive); CV ~0.9% — best-yet variance |
| #434 | 100.44 ± 5.54 (n=2) | 96.73 (3-finite-split) * | + gradient clipping max_norm=1.0; -14.4% vs #441; 100% steps clipped → Lion-like normalized-gradient regime |
| #441 | 117.37 ± 0.85 (n=2) | 115.59 (3-finite-split) * | bf16 mixed precision standalone; 19 epochs reached vs ~14 fp32; CV ~0.7% |

\* `test_avg/mae_surf_p` 4-split mean is still NaN on cruise pending PR #375 (data/scoring.py fix). Per-channel test surf MAEs: single 122.76, geom_camber_rc 118.27, re_rand 105.73 (3-finite mean). Once #375 lands, can re-evaluate the saved bf16 artifacts (`model-bf16_seed0-cgitj1dc`, `model-bf16_seed1-i45ys5ih`) for canonical 4-split numbers.

### Reverted

- **PR #336** (slice_num 64→128, val_avg=139.83 single seed) was reverted on commit `605b439` after direct apples-to-apples evidence (PRs #329 and #338) showed slice_num=128 loses by 10-20 MAE inside the 30-min wall-clock cap. slice_num=128 may convert better with longer wall-clock; revisit in round 2 if `SENPAI_TIMEOUT_MINUTES` increases.
