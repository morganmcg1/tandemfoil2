# Baseline

The current research baseline on `icml-appendix-willow-pai2d-r2`.

The primary ranking metric is `val_avg/mae_surf_p` — equal-weight mean surface
pressure MAE across the four validation tracks. Lower is better.

## 2026-04-27 23:50 — PR #328: Round 1 axis: physics-token count — slice_num 64 → 128

Doubled the Transolver `slice_num` from 64 to 128 (the only change). All
other knobs are baseline defaults.

```
lr=5e-4  weight_decay=1e-4  batch_size=4  surf_weight=10.0  epochs=50
n_hidden=128  n_layers=5  n_head=4  slice_num=128  mlp_ratio=2
```

- **val_avg/mae_surf_p:** **133.55** (best epoch 11 of 50; run timed out at
  31.6 min wall clock — undertrained, val curve still descending)
- **val_avg/mae_surf_Ux:** 2.162
- **val_avg/mae_surf_Uy:** 0.896

Per-split (best-val checkpoint):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 164.27 | 1.822 | 0.892 |
| val_geom_camber_rc | 139.35 | 3.023 | 1.156 |
| val_geom_camber_cruise | 104.92 | 1.521 | 0.635 |
| val_re_rand | 125.66 | 2.284 | 0.900 |

Test-side metrics: `test_avg/mae_surf_p` is `NaN` because of a known
scoring bug on `test_geom_camber_cruise` (sample 20 GT has 761 NaN values
in the pressure channel; `accumulate_batch` masks them but `0.0 * NaN =
NaN` defeats the mask). Tracked as a dedicated bug-fix PR. Other test
splits are clean: `test_single_in_dist/mae_surf_p=140.58`,
`test_geom_camber_rc/mae_surf_p=131.24`, `test_re_rand/mae_surf_p=122.71`.

- **W&B run:** `s1p2qs7l` —
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/s1p2qs7l
- **Reproduce:** `cd target/ && python train.py --wandb_name "willow-r2-fern-slice-128" --agent willowpai2d2-fern`

## Validation tracks

- `val_single_in_dist` — single-foil sanity
- `val_geom_camber_rc` — held-out raceCar tandem front-foil camber (M=6-8)
- `val_geom_camber_cruise` — held-out cruise tandem front-foil camber (M=2-4)
- `val_re_rand` — stratified Re holdout across all tandem domains

The paper-facing test counterpart is `test_avg/mae_surf_p`, computed at the
end of every run from the best validation checkpoint. Currently NaN until
the cruise scoring bug fix lands.
