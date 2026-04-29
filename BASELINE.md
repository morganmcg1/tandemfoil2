# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Best metrics

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **81.8075** | `2akpdg9t` | #914 |
| `test_avg/mae_surf_p` | **73.04** (4-split, finite) | `2akpdg9t` | #914 |
| 3-split test mean (excl. cruise) | **81.28** | `2akpdg9t` | #914 |
| Best epoch | 12 / 13 (timeout) | | |
| Wall time | 30.6 min | | |
| Params | 661,735 (−4,720 vs prior; SwiGLU 2/3 trick) | | |

**`test_avg/mae_surf_p` is now finite (73.04)** — NaN guards + SwiGLU run
on the merged branch confirms cruise sample 000020 is correctly filtered
by #797.

## Per-split val (epoch 12, run `2akpdg9t`)

| Split | mae_surf_p |
|-------|-----------|
| `val_single_in_dist` | 97.53 |
| `val_geom_camber_rc` | 94.17 |
| `val_geom_camber_cruise` | **59.18** |
| `val_re_rand` | 76.36 |
| **val_avg** | **81.8075** |

## Per-split test (epoch 12, run `2akpdg9t`)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|-------|-----------|------------|------------|----------|
| `test_single_in_dist` | 86.73 | 1.354 | 0.500 | 88.08 |
| `test_geom_camber_rc` | 85.54 | 1.640 | 0.679 | 83.16 |
| `test_geom_camber_cruise` | 48.32 | 0.919 | 0.356 | 44.53 |
| `test_re_rand` | 71.57 | 1.192 | 0.525 | 65.67 |
| **test_avg** | **73.04** | **1.276** | **0.515** | **70.36** |

## Configuration (post-#914)

| Knob | Value |
|------|-------|
| Model | Transolver + **SwiGLU MLP** |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| **`use_swiglu`** | **True** — SiLU-gated FFN, n_hidden_swiglu=168 (int(256×2/3) → multiple of 8) |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| Schedule | CosineAnnealingLR, T_max=epochs |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `channel_weights` | [1.0, 1.0, 3.0] for [Ux, Uy, p] |
| Loss | L1 (absolute error) on normalized space, vol + surf_weight × surf |
| `fourier_bands` | 4 — `[sin/cos(π·2^k·x), sin/cos(π·2^k·z)]` for k=0..3 prepended to input |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |
| NaN guards | active in `evaluate_split` (#797) — drops cruise sample 000020 |
| Seed | NONE (val_avg drifts ~5-10% across runs; seed PR #863 in flight) |
| Params | **661,735** (−4,720 vs prior; SwiGLU 2/3 dim trick saves ~792 params/block) |

## Delta history

| Metric | L1-only (#752) | +ch=[1,1,3] (#754) | +Fourier PE K=4 (#820) | +SwiGLU MLP (#914) |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.93 | 99.23 | 89.71 | **81.81** |
| `test_avg/mae_surf_p` | 100.83† | 99.34† | NaN† | **73.04** |
| Δ vs prior | — | −2.65% | −9.59% | **−8.81%** |
| Cumul. Δ vs L1-only | — | −2.65% | −12.0% | **−19.7%** |

†3-split test mean (excl. cruise) for pre-#914 runs; #914 is first with
finite `test_avg` across all 4 splits.

SwiGLU's per-feature gating is load-bearing on CFD inputs because feature
scales are heterogeneous (pressure dominates Ux/Uy by 10–100×) — the gate
learns to allocate capacity to pressure features, which is exactly what
ch=[1,1,3] pays for in the loss. Also: at mlp_ratio=2 the FFN is bottlenecked,
so gating's information-routing effect matters more than in wide NLP FFNs.

Key observation: converged at epoch 12 vs baseline epoch 14 — SwiGLU reaches a
better minimum *faster*, suggesting the inductive bias matches the CFD task
well. Timeout cut the run at epoch 13 — there may be additional headroom.

## Reproduce

```bash
cd target/
python train.py --use_swiglu \
  --agent willowpai2e4-tanjiro \
  --wandb_name "willowpai2e4-tanjiro/swiglu-mlp"
```

(All other hyperparameters are now defaults in `Config`: lr=5e-4, surf_weight=10,
channel_weights=[1,1,3], batch_size=4, fourier_bands=4, CosineAnnealingLR.)

## Open issues

- **Run-to-run val variance:** Seed PR #863 (askeladd) is in flight — will
  eliminate the ~5-10% init/sampler drift once merged.
- **Best epoch cliff at 12/13:** The best epoch was epoch 12; epoch 13 slightly
  bounced (85.75). With more epochs / longer LR schedule, SwiGLU may have
  additional headroom — unexplored within current 30-min budget.
- **Cruise-test `-Inf` GT (workaround active):** `test_geom_camber_cruise/000020.pt`
  has 761 `-Inf` values in the `p` channel. The per-sample `y_finite` guard
  in `evaluate_split` (#797) filters this sample. Dataset is read-only.

---

## Prior baseline (PR #820, Fourier PE K=4)

| Metric | Value | Run |
|--------|-------|-----|
| `val_avg/mae_surf_p` | 89.7141 | `w9xbc0wl` |
| 3-split test mean (excl. cruise) | 88.16 | `w9xbc0wl` |
| Best epoch | 14 / 50 (timeout) | |
