# Baseline

The current research baseline on `icml-appendix-willow-pai2d-r2`.

The primary ranking metric is `val_avg/mae_surf_p` — equal-weight mean surface
pressure MAE across the four validation tracks. Lower is better.

## 2026-04-28 01:30 — PR #330: Round 1 axis: loss formulation — MSE → Huber (β=1)

Replaced per-element MSE with Huber (Smooth L1, β=1) loss in both the training
loop and `evaluate_split`, applied via
`F.smooth_l1_loss(pred, y_norm, reduction="none", beta=1.0)`. Stacks on top of
the merged slice_num=128 architecture from PR #328.

```
lr=5e-4  weight_decay=1e-4  batch_size=4  surf_weight=10.0  epochs=50
n_hidden=128  n_layers=5  n_head=4  slice_num=128  mlp_ratio=2
loss=Huber(beta=1.0) on normalized residuals  (was MSE)
```

- **val_avg/mae_surf_p:** **115.61** (best epoch 11 of 50; run timed out at
  31.5 min wall clock — undertrained, val curve still descending)
- **val_avg/mae_surf_Ux:** 1.81
- **val_avg/mae_surf_Uy:** 0.75

Per-split (best-val checkpoint):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 137.21 | 1.56 | 0.74 |
| val_geom_camber_rc | 118.60 | 2.44 | 0.97 |
| val_geom_camber_cruise | 98.74 | 1.28 | 0.55 |
| val_re_rand | 107.89 | 2.00 | 0.74 |

**Δ vs prior baseline (PR #328 slice_num=128, MSE @ 133.55): −13.4 %.** Outside
the ±10 % single-seed noise floor (per thorfinn #337 replicate). Strongest
per-split delta on `val_re_rand` (107.89 vs 125.66 = −14.1 %), confirming the
high-Re-tail mechanism predicted in the original hypothesis.

Test-side: `test_avg/mae_surf_p` is `NaN` due to the
`test_geom_camber_cruise/.gt/000020.pt` non-finite-pressure scoring bug.
Bug-fix PR #367 in flight (3 students independently confirmed the root cause
and equivalent patch). Other test splits are clean:
`test_single_in_dist/mae_surf_p=125.75`, `test_geom_camber_rc/mae_surf_p=108.59`,
`test_re_rand/mae_surf_p=105.10`.

- **W&B run:** `uip4q05z` —
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/uip4q05z
- **Reproduce:** `cd target/ && python train.py --wandb_name "willow-r2-frieren-huber-b1-on-slice128" --agent willowpai2d2-frieren`

## Prior baseline — PR #328 (slice_num=128, MSE)

Doubled the Transolver `slice_num` from 64 to 128. All other knobs at original
defaults.

- val_avg/mae_surf_p = 133.55 (best epoch 11/50, run `s1p2qs7l`)
- Set the architectural baseline that frieren's Huber rebased onto.

## Validation tracks

- `val_single_in_dist` — single-foil sanity
- `val_geom_camber_rc` — held-out raceCar tandem front-foil camber (M=6-8)
- `val_geom_camber_cruise` — held-out cruise tandem front-foil camber (M=2-4)
- `val_re_rand` — stratified Re holdout across all tandem domains

The paper-facing test counterpart is `test_avg/mae_surf_p`, computed at the
end of every run from the best validation checkpoint. Currently NaN until
the cruise scoring bug fix (PR #367) lands.
