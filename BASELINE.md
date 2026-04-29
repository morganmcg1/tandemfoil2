# Baseline — icml-appendix-willow-pai2e-r2

Active branch: `icml-appendix-willow-pai2e-r2`.

## Current best (this branch)

- **PR**: #1048 — "Architecture width: n_hidden=128 -> 192 (relative_mae + warmup baseline)" (merged 2026-04-29)
- **Config**: `n_layers=3, slice_num=16, n_head=1, n_hidden=192, mlp_ratio=2` + `loss_type=relative_mae` (default) + `lr=2e-3` + `batch_size=16` + `compile=True` + `warmup_epochs=5`
- **val_avg/mae_surf_p** (best checkpoint, epoch 49): **50.61** (W&B `ovkjhjyo`, default seed — no `PYTHONHASHSEED` env var)
- **test_avg/mae_surf_p** (best checkpoint): **44.15** (finite across all 4 splits)
- **Wall clock**: 30.6 min / 50 epochs (right at the 30 min harness cap)
- **Peak VRAM**: 73.6 GB on a 96 GB GPU

> ⚠️ **Seed-swap caveat still applies**: Default seed (no `PYTHONHASHSEED`) is the better seed under warmup. Do NOT pin `PYTHONHASHSEED=42` when reproducing.
>
> ⚠️ **Wall-clock budget**: 30.6 min is at the harness cap. Width-only re-runs on this baseline have effectively zero remaining budget for additional per-step compute. Increases to depth, mlp_ratio, slice_num, or `--epochs 60+` will need throughput offsets (compile-tuning, faster kernels, or grad accum at bs=8 if it helps) or an explicit timeout relaxation.

### 2-seed variance corridor (PR #1048)

| Seed | W&B | best val_avg | test_avg | best epoch |
|------|-----|-------------:|---------:|-----------:|
| default | `ovkjhjyo` | **50.61** | **44.15** | 49 |
| 42 | `v9ipztd9` | 56.24 | 48.33 | 48 |
| **mean** | | **53.42** | **46.24** | |
| **spread (max−min)** | | **5.63** | **4.18** | |

> The seed corridor halved (12.58 → 5.63 val; 9.65 → 4.18 test) compared to the prior n_hidden=128 baseline. The wider model has a friendlier optimization landscape — fewer bad basins for unlucky seeds.
>
> **Single-seed screening convention** (preserved from PR #1008): With 2-seed val-spread now 5.63, future hypothesis PRs may use a single default-seed run for screening. A 2nd seed is required only for the final candidate before merge. PRs claiming improvements smaller than ~3 val pts on default seed must run ≥ 2 seeds.
>
> **Settled negative** (preserved): ε ≠ 1e-6 (`rel_mae_eps`) does NOT compose with warmup. PR #940 round 2 ran ε=1e-3 + warmup at 2 seeds and regressed (best-seed val=61.62). May need re-test under n_hidden=192 if loss-side levers re-open, but currently low priority.
>
> **Settled negative** (preserved): warmup_epochs=10 hurts (val=64.77 at n_hidden=128). warmup_epochs=3 within seed noise of 5; 5 remains default.

### Per-split test metrics (default seed `ovkjhjyo`, best checkpoint)

| Split | test mae_surf_p (n_hidden=128) | test mae_surf_p (n_hidden=192) | Δ |
|-------|---:|---:|---:|
| `test_single_in_dist`       | 67.22 | **58.21** | −9.01 (−13.4%) |
| `test_geom_camber_rc`       | 60.38 | **57.94** | −2.44 (−4.0%)  |
| `test_geom_camber_cruise`   | 23.79 | **21.65** | −2.14 (−9.0%)  |
| `test_re_rand`              | 41.20 | **38.79** | −2.41 (−5.9%)  |
| **test_avg/mae_surf_p**     | **48.15** | **44.15** | **−4.00 (−8.3%)** |

All four splits improve. Largest gain on `test_single_in_dist` (high-variance raceCar-single-dominant track).

**Reference target gap**: prior-round PR #32 hit `test_avg/mae_surf_p = 40.93`. Current best 44.15 is now **3.22 pts** off the target (was 7.22 pts before this merge).

### Reproduce (new defaults after PR #1048)

```bash
cd target && python train.py \
    --epochs 50 \
    --wandb_group nh192-relmae-warmup \
    --wandb_name nh192-default \
    --agent willowpai2e2-edward
```

`model_config` in `train.py` is now: `n_layers=3, n_head=1, slice_num=16, n_hidden=192, mlp_ratio=2`. CLI defaults after PR #1048: `batch_size=16`, `lr=2e-3`, `compile=True`, `loss_type="relative_mae"`, `warmup_epochs=5`. Schedule = `LinearLR(0.05→1.0)` for the first 5 epochs, then `CosineAnnealingLR(T_max=45, eta_min=1e-6)` for the remaining 45. Best epoch is still 48–49/50 — the schedule has not saturated and is signal-positive at the very end of training.

### Paired seed (seed42, W&B `v9ipztd9`)

val=56.24 / test=48.33 (best ep 48). Per-split test: single=75.42, rc=57.85, cruise=21.83, re_rand=38.21. Same config, slightly worse local minimum — but the seed-corridor is now ~5–6 pts wide, half the prior width.

---

## 2026-04-29 — PR #1048: n_hidden=128 → 192 (architecture width under AMP throughput) (merged)

- **val_avg/mae_surf_p:** 50.61 (best epoch 49, default seed `ovkjhjyo`)
- **test_avg/mae_surf_p:** 44.15 (finite across all 4 splits)
- **Per-split test:** single=58.21, rc=57.94, cruise=21.65, re_rand=38.79
- **Paired seed (seed42, `v9ipztd9`):** val=56.24, test=48.33 — slightly worse local min
- **Variance:** 2-seed val-spread 5.63 (was 12.58 at n_hidden=128), test-spread 4.18 (was 9.65) — corridor halved
- **Delta vs previous best (PR #971 default seed):** −4.09 (−7.5%) val_avg / −4.00 (−8.3%) test_avg
- **Reference target gap:** test 44.15 vs target 40.93 → **3.22 pts** (was 7.22)
- **Wall clock:** 30.6 min / 50 epochs (right at harness cap; peak VRAM 73.6 GB / 96 GB)
- **Status:** Merged. New `model_config["n_hidden"]=192` (was 128); ~2.2× param count (1.24M vs 0.56M); all other defaults inherited from PR #971.

---

## 2026-04-29 — PR #971: LR warmup (5-epoch linear) + flip loss_type default to relative_mae (merged)

- **val_avg/mae_surf_p:** 54.70 (best epoch 49, default seed `1xfcb5h5`)
- **test_avg/mae_surf_p:** 48.15 (finite across all 4 splits)
- **Per-split test:** single=67.22, rc=60.38, cruise=23.79, re_rand=41.20
- **Paired seed (seed42, `9a9di1dz`):** val=67.28, test=57.80 — seed-swap vs round-3
- **Variance:** 2-seed spread 12.58 (was 27.07), mean 60.99 (was 69.43) → narrowed and improved
- **Delta vs previous best (PR #821 seed42):** −1.20 (−2.1%) val_avg / −1.49 (−3.0%) test_avg
- **Wall clock:** 22.4 min / 50 epochs
- **Status:** Merged. New CLI defaults: `loss_type="relative_mae"`, `warmup_epochs=5`. Schedule = `SequentialLR(LinearLR(start=0.05) for 5ep, then CosineAnnealingLR(T_max=45, eta_min=1e-6))`.

---

## 2026-04-29 — PR #821: AMP/bf16 + bs=16 + lr=2e-3 + torch.compile + NaN-safe eval (merged)

- **val_avg/mae_surf_p:** 55.90 (epoch 50, seed42 `66c4gac6` — still descending)
- **test_avg/mae_surf_p:** 49.64 (finite across all 4 splits)
- **Per-split test:** single=63.94, rc=62.62, cruise=26.87, re_rand=45.11
- **Alternate seed (default, `1d8nkjir`):** val=82.97, test=72.01 — 27-pt spread; LR warmup needed
- **Delta vs previous best (PR #840 seed42):** −13.5% val_avg / −10.9% test_avg
- **Wall clock:** 22.5 min / 50 epochs (AMP + bs=16 + compile ≈ 1.5–1.8× speedup over fp32/bs=4)
- **Status:** Merged. New CLI defaults: `lr=2e-3`, `batch_size=16`, `compile=True`

---

## 2026-04-28 18:00 — PR #840: per-sample relative MAE (merged)

- **val_avg/mae_surf_p:** 64.16 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 55.73 (finite across all 4 splits)
- **Per-split val:** single=77.07, rc=84.10, cruise=36.86, re_rand=58.58
- **Per-split test:** single=71.33, rc=70.62, cruise=30.92, re_rand=50.04
- **Delta vs previous best (PR #783):** −11.77 (−15.5%) on val_avg/mae_surf_p
- **W&B run:** t5p9xzxx (rebased re-run)
- **Status:** Merged into advisor branch

---

## 2026-04-28 16:00 — PR #840: per-sample relative MAE (new best — pending merge)

- **val_avg/mae_surf_p:** 64.73 (epoch 32, timed out at 32/50 — still improving)
- **test_avg/mae_surf_p:** 56.92 (finite across all 4 splits)
- **Per-split val:** single=80.41, rc=78.51, cruise=40.13, re_rand=60.73
- **Per-split test:** single=77.25, rc=67.74, cruise=32.35, re_rand=50.35
- **Delta vs previous best (PR #783):** −11.20 (−14.7%) on val_avg/mae_surf_p
- **W&B run:** nz8eev8e
- **Status:** Winner declared; sent back for rebase (merge conflict on advisor branch)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (prev best)

---

## 2026-04-28 14:00 — PR #783: Huber loss δ=1.0 (new best)

- **Surface MAE (val_avg):** 75.93 (epoch 32, timed out at 32/50 — still improving)
- **Per-split val:** single=85.84, rc=91.20, cruise=54.68, re_rand=71.99
- **Per-split test (finite):** single=79.35, rc=82.61, re_rand=64.29; cruise=NaN (scoring bug)
- **Delta vs previous best:** −20.87 (−21.6%) on val_avg/mae_surf_p
- **W&B run:** 2y1lj209
- **Reproduce:** see above — add `--huber_delta 1.0` to the compound anchor command

---

## 2026-04-28 12:00 — PR #779: Round 1 anchor

- **Surface MAE (val_avg):** 96.80
- **W&B run:** ez3f10h3
- **Reproduce:** see above

---

## Reference context (from `target/README.md` leaderboard)

A previous senpai-vs-kagent investigation against this same dataset/Transolver
baseline found that a compounded reduction of model size dominated the
leaderboard. Use these as targets, not as merged baselines on this branch:

- Reference baseline (default config, similar to our `train.py`): `test_avg/mae_surf_p ≈ 80–82`
- Reference compound winner (PR #32 in that older repo): `test_avg/mae_surf_p = 40.927`
  - Configuration: `n_layers=3, slice_num=16, n_head=1, n_hidden=128, mlp_ratio=2`
  - Compound was the combination of three independent reductions (depth, slice
    count, single-head attention) on top of the default optimizer/loss.

## Default training command

```bash
cd target && python train.py --epochs 50 --wandb_name <descriptive-name>
```

Architecture parameters (`n_hidden`, `n_layers`, `n_head`, `slice_num`,
`mlp_ratio`) are not CLI flags — students must edit `model_config` in
`target/train.py` to change them. Optimizer and loss parameters (`lr`,
`weight_decay`, `batch_size`, `surf_weight`, `epochs`) are CLI flags via
`Config`.

## Per-split structure (4 val + 4 test tracks)

The primary metric `val_avg/mae_surf_p` is the **equal-weight mean of surface
pressure MAE across the four validation splits**. The same average across the
four held-out test splits is `test_avg/mae_surf_p`. Lower is better. Best
checkpoint is selected on `val_avg/mae_surf_p` and that checkpoint is used for
the end-of-run test eval. See `target/program.md` for the full split design.

| Track | Tests |
|-------|-------|
| `val_single_in_dist` / `test_single_in_dist` | Sanity (single-foil random holdout) |
| `val_geom_camber_rc` / `test_geom_camber_rc` | RaceCar tandem, unseen front-foil camber M=6–8 |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | Cruise tandem, unseen front-foil camber M=2–4 |
| `val_re_rand` / `test_re_rand` | Stratified Re holdout across all tandem domains |
