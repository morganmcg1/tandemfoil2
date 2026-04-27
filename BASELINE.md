# TandemFoilSet Baseline — `icml-appendix-willow-r3`

**Advisor track:** `icml-appendix-willow-r3`
**W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-r3`

---

## Current best (round 0 — vanilla Transolver)

**No experiments have been run on this track yet.** The opening baseline is the
vanilla `train.py` config in this repo (Transolver, MSE loss, no AMP, no
Fourier PE, slice_num=64, n_layers=5, n_head=4, surf_weight=10).

A clean baseline anchor run (PR #1, alphonse) is being launched in round 1 to
establish the actual `val_avg/mae_surf_p` and `test_avg/mae_surf_p` numbers
under the current config; this file will be updated immediately when that run
reports results.

### Default Config (vanilla, this repo's `train.py`)

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs | 50 (capped by `SENPAI_TIMEOUT_MINUTES=30` wall clock) |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| loss | MSE (normalized space) |
| AMP | off |
| Fourier PE | none |
| FFN | standard MLP (Linear → GELU → Linear) |

### Reproduce (vanilla baseline)

```
cd target && python train.py --epochs 50 --agent <name> --wandb_name <name>
```

---

## Prior-art priors (from sibling track `kagent_v_students`)

A separate research lineage on `kagent_v_students` accumulated 8 stacked
compounding wins ending at:

- **val_avg/mae_surf_p ≈ 49.4 (2-seed mean)**, **test_avg/mae_surf_p ≈ 42.5**
- Recipe: `L1 + sw=1 + AMP + grad_accum=4 + Fourier PE σ=0.7 m=160 + SwiGLU + slice_num=8 + n_layers=3`
- 8 compounding components, each merged after a beat-baseline PR.

These results were obtained on the same TandemFoilSet data and the same
Transolver model contract, but on a different repo branch with the wins applied
in `train.py`. **The first job on this `icml-appendix-willow-r3` track is to
re-introduce the strongest of those wins, one orthogonal axis per PR**, and
re-validate that they stack here too. The numbers above set a strong target:
any final result much worse than ~50 val / ~42 test is leaving signal on the
table.

This file will be rewritten the first time a PR on this track posts a clean
`val_avg/mae_surf_p` and `test_avg/mae_surf_p` from the trainer.
