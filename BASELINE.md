# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **99.2257** | `m46h5g4s` | #754 |
| `test_avg/mae_surf_p` | **92.6101** ✓ unblocked | `2hcmefh9` | #797 |
| Best epoch | 12 / 50 (timeout @ ep 14) | | |
| Wall time | 30.77 min | | |

Note: `val_avg` and `test_avg` are reported from different runs of the
same merged code path. `train.py` does not seed torch/numpy/sampler, so
val_avg drifts ~5–10% across runs without any code change. The 99.23
val number is from `m46h5g4s` (PR #754); the 92.61 test number is from
`2hcmefh9` (PR #797 rebased — same code, different seed init). Tracking
seeding as a separate priority for the next PR.

## Per-split val (epoch 12, run `m46h5g4s`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|-------------|-------------|
| `val_single_in_dist` | 116.675 | 1.358 | 0.657 |
| `val_geom_camber_rc` | 113.935 | 2.820 | 0.934 |
| `val_geom_camber_cruise` | 75.015 | 1.129 | 0.487 |
| `val_re_rand` | 91.279 | 1.850 | 0.693 |
| **val_avg** | **99.226** | **1.789** | **0.693** |

## Per-split test (post-#797 unblock, run `2hcmefh9`, epoch 14)

| Split | mae_surf_p | nonfinite_pred | nonfinite_gt |
|-------|-----------|----------------|--------------|
| `test_single_in_dist` | 117.771 | 0 | 0 |
| `test_geom_camber_rc` | 99.491 | 0 | 0 |
| `test_geom_camber_cruise` | **65.287** | 0 | **1** (filtered) |
| `test_re_rand` | 87.891 | 0 | 0 |
| **test_avg/mae_surf_p (all 4 splits)** | **92.610** | 0 | 1 |

## Configuration (post-#797)

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| **`channel_weights`** | **[1.0, 1.0, 3.0]** for [Ux, Uy, p] |
| Loss | L1 (absolute error) on normalized space, vol + surf_weight × surf |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |
| **NaN guards** | **active in `evaluate_split` (#797)** — drops cruise sample 000020 from accumulator |
| Seed | **NONE** (val_avg drifts ~5-10% across runs; seed PR is next priority) |

## Delta vs prior baseline (PR #752 L1)

| Metric | L1-only (#752) | L1 + ch=[1,1,3] (#754) | Δ |
|---|---|---|---|
| `val_avg/mae_surf_p` | 101.93 | **99.23** | **−2.65%** |
| 3-split test mean | 100.83 | 99.34 | −1.48% |
| `val_avg/mae_surf_Ux` | 1.429 | 1.789 | +25.2% |
| `val_avg/mae_surf_Uy` | 0.611 | 0.693 | +13.4% |

The 3× pressure weight stacks with L1: net pressure improvement, with
acceptable velocity-channel regression (we don't rank on velocity).
Biggest val gain on `val_single_in_dist` (133.25 → 116.68, −12.4%) — the
heaviest-tail split where pressure outliers dominate.

## Reproduce

```bash
cd target/
python train.py --agent willowpai2e4-fern \
  --wandb_name "willowpai2e4-fern/p-channel-3x-on-L1"
```

## Open issues

- **Run-to-run val variance:** `train.py` does not set torch/numpy/sampler
  seeds. Across two runs of the merged code (m46h5g4s vs 2hcmefh9), val_avg
  drifted 99.23 → 105.22, almost entirely on `val_single_in_dist`. The
  diagnostic counts confirm the NaN-guard #797 was no-op on val (so the
  drift is not caused by the merge), but it does mean baseline comparisons
  are noisy. **Adding a seed is the next priority** — askeladd assigned.
- **Cruise-test `-Inf` GT (resolved by #797 workaround):** Confirmed by
  askeladd: `test_geom_camber_cruise/000020.pt` has **761 `-Inf` values**
  in the `p` channel (all in volume nodes; surface MAE not directly
  affected). The merged guard filters this sample from
  `accumulate_batch`. The dataset itself is read-only. One residual
  display-only NaN in `cruise/loss` (cosmetic) traced to `nan_to_num`
  default args overflowing through `channel_weights[2]=3` — fix is a
  one-line argument addition (`nan=0.0, posinf=0.0, neginf=0.0`); will
  ride along with the next `evaluate_split`-touching PR.
