# Current Baseline — `icml-appendix-willow-pai2e-r4`

## Current best (post-#963 T_max=13 schedule fix)

**⚠️ All future PRs must pass `--t_max 13` and beat val_avg = 64.91.**

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | **64.9148** | `j8yi780z` | #963 |
| `test_avg/mae_surf_p` | **57.2466** (4-split, finite) | `j8yi780z` | #963 |
| Best epoch | 13 / 13 (timeout — plateau forming, Δ=−0.70 at ep13) | | |
| Wall time | 30.6 min | | |
| Params | 661,735 (0 change — scheduler-only) | | |

**Key insight (#963):** The −20.66% val gain was free — zero params, zero
wall-clock cost. Every prior gain (Fourier PE, channel weights, SwiGLU) was
real but was measured under T_max=50 under-training. The cosine annealing tail
(LR→0, epochs 35–50) was unreachable under the 30-min budget; effective LR
was stuck at ~85% of peak. With T_max=13, the full decay fits the budget.

## Per-split val — current best (epoch 13, run `j8yi780z`, PR #963)

| Split | mae_surf_p | Δ vs prior best |
|-------|-----------|----------------|
| `val_single_in_dist` | **71.86** | −26.3% |
| `val_geom_camber_rc` | **76.45** | −18.8% |
| `val_geom_camber_cruise` | **46.54** | −21.4% |
| `val_re_rand` | **64.81** | −15.1% |
| **val_avg** | **64.91** | **−20.7%** |

## Per-split test — current best (epoch 13, run `j8yi780z`, PR #963)

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | **64.75** |
| `test_geom_camber_rc` | **69.21** |
| `test_geom_camber_cruise` | **39.29** |
| `test_re_rand` | **55.74** |
| **test_avg** | **57.25** |

## Seeded canonical history

| Baseline state | val_avg @ `--seed 0` | test_avg @ `--seed 0` | Run | PR | T_max |
|---|---:|---:|---|---|---:|
| post-#914 (Fourier + SwiGLU) | 85.14 | (n/a) | `j1r5y758` | #863 | 50 (under-trained) |
| **post-#963 (T_max=13)** | **65.8478** | **57.2459** | `zicvysyj` | #1000 | 13 |

Note: `zicvysyj` (val=65.85, test=57.2459) is the seeded canonical at the
30-min budget. Test_avg reproduces unseeded `j8yi780z` (test=57.2466) to 4
decimal places — the two runs sample the same post-#963 population. Use this
as the comparison reference for borderline-ablation PRs (<2% absolute claims).
For ≥2% claims, the unseeded 64.91 still serves as the headline target since
seed noise (σ≈3.74 val from #972) is small relative to the claim size.

### A/B determinism proof — bit-exact (0.0000 abs diff, T_max=50 era)

| Tree | Run A | Run B | val_avg @ seed 0 | |Δ| |
|---|---|---|---:|---:|
| pre-#820 (no Fourier, no SwiGLU) | `sk040lf3` | `gkqoo0v4` | 100.80 | 0.0000 |
| post-#820 (Fourier only) | `0zx4mdxs` | `u58qtuye` (4 ep) | 97.92 | 0.0000 |
| post-#914 (Fourier + SwiGLU, T_max=50) | `j1r5y758` | (no A/B) | 85.14 | (carry) |

Bit-exactness holds for non-conv, non-AMP, single-GPU runs.

**Borderline-ablation contract:** PRs claiming <2% absolute val_avg
improvement must run `--seed {0, 1, 2}` and report mean ± std. ≥2% effects
can stay single-seed for now.

## Prior best metrics — unseeded (pre-#963)

| Metric | Value | Run | PR |
|--------|-------|-----|----|
| `val_avg/mae_surf_p` | 81.8075 | `2akpdg9t` | #914 |
| `test_avg/mae_surf_p` | 73.04 | `2akpdg9t` | #914 |
| 3-split test mean (excl. cruise) | 81.28 | `2akpdg9t` | #914 |

## Configuration (post-#963)

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
| Schedule | CosineAnnealingLR, **`--t_max 13`** (budget-matched; `t_max=0` default uses epochs=50) |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `channel_weights` | [1.0, 1.0, 3.0] for [Ux, Uy, p] |
| Loss | L1 (absolute error) on normalized space, vol + surf_weight × surf |
| `fourier_bands` | 4 — `[sin/cos(π·2^k·x), sin/cos(π·2^k·z)]` for k=0..3 prepended to input |
| Sampler | `WeightedRandomSampler` over balanced domain groups |
| `epochs` | 50 (capped) |
| Timeout | 30 min |
| NaN guards | active in `evaluate_split` (#797) — drops cruise sample 000020 |
| **Seed** | **`--seed 0` default (#863)** — bit-perfect determinism on single-GPU; pass `generator=g` to sampler+DataLoader |
| Params | **661,735** (−4,720 vs prior; SwiGLU 2/3 dim trick saves ~792 params/block) |

**⚠️ `--t_max 13` is NOT the default** (default is 0 = T_max=epochs=50). All
future experiments must pass `--t_max 13` explicitly or they will compare
against a different schedule regime.

## Delta history

| Metric | L1-only (#752) | +ch=[1,1,3] (#754) | +Fourier PE K=4 (#820) | +SwiGLU MLP (#914) | +T_max=13 (#963) |
|---|---:|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 101.93 | 99.23 | 89.71 | 81.81 | **64.91** |
| `test_avg/mae_surf_p` | 100.83† | 99.34† | NaN† | 73.04 | **57.25** |
| Δ vs prior | — | −2.65% | −9.59% | −8.81% | **−20.66%** |
| Cumul. Δ vs L1-only | — | −2.65% | −12.0% | −19.7% | **−36.3%** |

†3-split test mean (excl. cruise) for pre-#914 runs; #914 is first with
finite `test_avg` across all 4 splits.

All gains prior to #963 were measured under T_max=50 (under-trained). They
remain valid relative improvements — each added real capacity — but their
*absolute magnitudes* were understated. #963 retroactively clarifies that
the prior improvements stacked correctly but the optimizer was never given
a proper convergence tail to exploit them. The −20.66% from #963 is not
a model improvement; it is collecting on the debt accumulated by a mismatched
LR schedule across every prior run.

## Reproduce (current best — post-#963)

```bash
cd target/
python train.py --t_max 13 --use_swiglu --fourier_bands 4 \
  --agent <your-agent-name> \
  --wandb_name "<your-run-name>"
```

(W&B run: `j8yi780z`. All other hyperparameters are defaults in `Config`.)

## Open issues

- **T_max seeded canonical pending:** Current best (64.91) is unseeded. A
  seeded T_max sweep at `--seed 0` with `--t_max {10, 12, 13, 16}` (frieren,
  in-flight) will produce the post-#963 seeded canonical.
- **Prior closed PRs may be worth retesting at T_max=13:** n_layers=6 (#939)
  was closed as "under-trained" — exactly the regime fixed by #963. High-
  priority round-3 retest candidate.
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
