# ML Intern TandemFoilSet-Balanced — pai2 24h replicate `mlintern-pai2-24h-v3-r1`

**Status**: in progress (this file is rewritten as more runs complete)
**Pod start**: 2026-04-30 06:21 UTC
**Pod kill deadline**: 2026-05-01 06:21 UTC
**Branch**: `mlintern-pai2-24h-v3-r1`
**W&B project**: `wandb-applied-ai-team/senpai-v1-ml-intern`
**W&B group**: `mlintern-pai2-24h-v3-r1`

## TL;DR

The default Transolver baseline has `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
(0.66M params) and is the smallest reasonable architecture for this dataset.
Simply scaling to `n_hidden=256, n_layers=8, n_head=8, slice_num=64`
(3.94M params) with `batch_size=2`, gradient checkpointing, and `surf_weight=30`
gives a clear win over the baseline within just a few epochs — `val_avg/mae_surf_p`
drops from ~500 to ~150 in 6-10 epochs.

Best configuration after the explore phase (still running):
- `n_hidden=256, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2`
- `batch_size=4`, `grad_checkpoint=True`
- `surf_weight=30`, `lr=5e-4`, `weight_decay=1e-4` (defaults)
- ~3.94M params, ~8 min/epoch on a single Blackwell B6000 (with 6 GPUs co-loaded)

## GPU usage strategy

8 × NVIDIA RTX PRO 6000 Blackwell (96GB each). Visible GPU budget: 8.

I used parallel one-GPU jobs pinned with `CUDA_VISIBLE_DEVICES=<n>`. Sweet spot
for this codebase appears to be 4-6 parallel jobs — going to 8 caused per-job
slowdown >2x because of CPU/PCIe/I-O contention (4 train workers + 16 val
workers per job × 8 jobs = >150 sub-processes). 4-6 parallel adds only 10-30%
per-epoch overhead vs running alone.

## Code changes

Single-file change in `train.py`:

1. `Config` dataclass extended with:
   - Architecture flags: `n_hidden`, `n_layers`, `n_head`, `slice_num`,
     `mlp_ratio`, `dropout`.
   - Training tweaks: `grad_clip`, `warmup_epochs`, `quiet`, `grad_checkpoint`,
     `grad_accum`.
   - Transolver++ flags: `ada_temp`, `rep_slice` (arXiv:2502.02414 extensions).
2. `PhysicsAttention` made optional Transolver++:
   - `ada_temp`: per-point temperature `τ_i = τ₀ + Linear(x_i)`, zero-init delta
     so default behaviour matches baseline.
   - `rep_slice`: Gumbel-Softmax noise added to slice logits during training
     only (no-op at eval).
3. `Transolver.forward` optionally wraps each block in
   `torch.utils.checkpoint.checkpoint`.
4. Cosine LR schedule optionally preceded by a `LinearLR` warmup.
5. Optional gradient accumulation and gradient clipping.
6. tqdm progress bar suppressed when `--quiet true` so logs are grep-friendly.

All flags are off by default — running `python ./train.py` with no flags
reproduces the original baseline behaviour exactly.

## Headline results (in-progress, val only — test eval will be added)

| Run name | Model | Notes | Best val_avg/mae_surf_p | Epoch |
|---|---|---|---:|---:|
| baseline (single-epoch debug) | 128/5/4/64 | smoke | ~487 | 3 |
| `w3b-scaled-sw30-pure` | 256/8/8/64 b=2 gc | sw=30 | **147.76** | 6 |
| `w3b-scaled-sw30-tpp` | 256/8/8/64 b=2 gc + ada+rep | sw=30 | 163.92 | 5 |
| `w3d-pure-b4` (180min cap) | 256/8/8/64 b=4 gc | sw=30 | **145.38** | 9 |
| `w3d-tpp-sw50` (180min cap) | 256/8/8/64 b=2 gc + ada+rep | sw=50 | 165.95 | 7 |
| `w3c-tpp-slice128` (180min cap) | 256/8/8/128 b=2 gc + ada+rep | sw=30 | 194.70 | 6 |
| `long-pure-sw30-b2` (8h cap) | 256/8/8/64 b=2 gc | sw=30 | 155.35 | 6 |
| `long-pure-sw30-b4` (7h cap) | 256/8/8/64 b=4 gc | sw=30 | 230 (running) | 2 |
| `long-bigger-320h10l` (8h cap) | 320/10/8/64 b=2 gc | sw=30 | 197.67 (running) | 4 |
| `long-deeper-256h12l` (8h cap) | 256/12/8/64 b=2 gc | sw=30 KILLED ep3 | 236.12 | 1 |
| `long-pure-lr3e4` (7h cap) | 256/8/8/64 b=2 gc | sw=30 lr=3e-4 | 223.87 (running) | 2 |
| `long-pure-warmup3` (7h cap) | 256/8/8/64 b=2 gc | sw=30 warmup=3 | 241.44 (running) | 2 |

## What's working / not working so far

Working:
- **Scaling from 0.66M → 3.94M params** is the single biggest gain.
- **`surf_weight=30`** improves slightly over the default `surf_weight=10` —
  the metric only counts surface MAE so up-weighting helps, but going beyond
  ~50 makes training noisier.
- **`batch_size=4` with gradient checkpointing** is roughly tied with
  `batch_size=2` per epoch wall-clock and gives slightly smoother curves.
- **`grad_checkpoint=True`** is necessary at 256/8/b=2 — without it, OOM at
  the largest mesh sizes.

Not working / mixed:
- **Transolver++ (`ada_temp`+`rep_slice`)**: positive at epoch 1 but adds
  noise during longer training (both pure sw=30 and tpp sw=30 reach
  ~150 by epoch 6, but tpp has more variance epoch-to-epoch).
- **Wider model (320/10)**: training is unstable in early epochs (val goes
  295→217→244 instead of monotonic decay). May still catch up but not yet.
- **Deeper model (256/12)**: clear overfitting — train loss falls but val
  loss rises monotonically. Killed at ep3.
- **`lr=1e-3`**: too high, harms convergence.
- **`warmup_epochs=3`**: just shifts the loss curve right; no obvious benefit.

## W&B run/group names

- Group: `mlintern-pai2-24h-v3-r1`
- All run names start with `mlintern-pai2-24h-v3-r1/...` so they group cleanly.

## Next plan

1. Let current 6 long runs continue until their caps.
2. When 180min caps end (~11:15 UTC), launch 3 fresh long runs that extend
   the current best (256/8/8/64 b=4 sw=30): vary `weight_decay` and
   `dropout` in tight ranges to test if regularization helps the b=4 run
   stay below 145.
3. After 8h long runs end (~16:30 UTC), launch the final 1-2 longest runs
   with the cumulative best config and run them for as long as possible.
4. Reserve ~1 hour at the end for `--skip_test=false` evaluation on the
   selected best-checkpoint and pushing the W&B model artifact.
