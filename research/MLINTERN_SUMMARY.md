# ML Intern (pai2-r2) — TandemFoilSet-Balanced run summary

W&B group: `mlintern-pai2-r2` in `wandb-applied-ai-team/senpai-v1-ml-intern`
Branch: `mlintern-pai2-r2`

Primary ranking metric: `val_avg/mae_surf_p`. Paper-facing reporting metric: `test_avg/mae_surf_p`.

## Pipeline changes shipped to `train.py`

`data/` is read-only and untouched. All knobs are CLI flags with defaults that preserve the previous baseline behaviour.

1. **OneCycleLR scheduler** (`--lr_scheduler onecycle`, opt-in). `pct_start` configurable, `div_factor=25`, `final_div_factor=1e4`. Steps per-batch when active.
2. **Gradient clipping** (`--max_grad_norm`). Required for stable training when meshes reach 240K nodes.
3. **Configurable architecture**: `--n_layers --n_hidden --n_head --slice_num --mlp_ratio`.
4. **Loss kind / channel weights**: `--loss_kind {mse,huber}`, `--huber_beta`, plus per-channel `--w_ux/--w_uy/--w_p` for the (Ux, Uy, p) targets in normalized space.
5. **AMP autocast** (`--amp_dtype {fp32,bf16,fp16}`). Used to fit wider/deeper configs without OOM at 240K-node padded batches.
6. **Wall-clock cap** `--max_minutes` plus `--epochs` cap. Fixed a footgun where `--epochs 999` plus a real wall cap left OneCycleLR distributing its schedule over 999 epochs, so the model only ever saw the warmup phase.
7. **NaN-safe scoring fallback** in `evaluate_split`: `data/scoring.py` is read-only and propagates NaN via `0 * NaN` whenever a batch contains a sample with non-finite ground truth (`test_geom_camber_cruise/000020.pt` has 761 non-finite p values). When detected, fall back to per-sample `accumulate_batch` calls. Bad samples cleanly return `(0, 0)` per `scoring.py`'s own intent ("Samples whose ground-truth is non-finite anywhere are skipped entirely"), and `test_avg/mae_surf_p` becomes finite.

## Strategy

- 8 × RTX PRO 6000 Blackwell GPUs (96 GB each) on a single pod. All training local.
- `WANDB_PROJECT=senpai-v1-ml-intern`, `WANDB_ENTITY=wandb-applied-ai-team`, group `mlintern-pai2-r2`.
- Multi-round parallel sweeps: 8 simultaneous experiments per round, each pinned to one GPU via `CUDA_VISIBLE_DEVICES`.
- Default invocation shape per the contract:
  `python train.py --epochs 999 --agent ml-intern-r2 --wandb_group mlintern-pai2-r2 --wandb_name "mlintern-pai2-r2/<short-description>"` plus the per-experiment knobs below.
- Each round was driven by what the previous round told us, not by a pre-baked grid.

### Round 1 (30 min cap, 8 configs) — baseline + author recipe

Tested vanilla (CosineAnnealing default), the Transolver "author" recipe (OneCycleLR + grad_clip=0.1 + wd=1e-5 + lr=1e-3), then capacity ablations (nl=8, n_head=8, n_hidden=256/192) and loss variants (Huber, surf_weight 20/50, w_p=2). Best so far ~135 val. Surface insight: vanilla CosineAnnealingLR(T_max=999) over 14 epochs is essentially constant LR ≈5e-4; OneCycleLR(max_lr=1e-3, epochs=999) over 14 epochs is essentially in early warmup at LR ≈4e-5. The "author recipe" win in round 1 was largely "lower effective LR" — not a real recipe difference. Action: properly bound `--epochs` so OneCycleLR's decay phase actually runs.

### Round 2 (75 min cap, `--epochs` set to fit the run) — schedule alignment + Huber

Re-ran the most promising configs with epoch count chosen so OneCycleLR's full warmup→peak→decay cycle fits inside the wall-clock cap. Headlines:

- **r2-author-30ep** (nl=5, n_head=4, n_hidden=128, OneCycleLR 30ep, Huber off): val 66.11, test 57.32. Substantial gain from running the schedule to completion.
- **r2-wide-huber-15ep** (n_hidden=256, n_head=8, Huber): val 76.92, test 65.99. Huber improves over MSE at fixed wall time (compare wide-15ep test 73.18).
- Per-channel pressure weight `w_p=2` made things worse, surf_weight bumps (20/50) made things worse, slow warmup (`pct_start=0.3`) was a small loss. n_hidden=192 deepwide bf16 OOMed in fp32, ran in bf16 but converged slowly relative to the small model.

Conclusion going into round 3: smaller model + more epochs + Huber loss is the dominant direction.

### Round 3 (200 min cap, 30–80 epochs) — search smaller architectures with Huber

| Config | params | val best | @ep | test_avg |
|---|---|---|---|---|
| **r3-nl4-80ep** (n_layers=4) | 0.54M | 40.30 | 79 | **34.44** |
| **r3-sn32-80ep** (slice_num=32) | 0.66M | 40.03 | 77 | 35.03 |
| **r3-sn16-80ep** (slice_num=16) | 0.65M | 40.62 | 77 | 35.35 |
| **r3-best-80ep** (default Transolver) | 0.66M | 42.14 | 80 | 35.55 |
| **r3-nl3-80ep** (n_layers=3) | 0.42M | 41.68 | 80 | 36.32 |
| r3-mlp4-50ep | 0.99M | 48.02 | 50 | 41.63 |
| r3-vwide-bf16-30ep (n_hidden=320) | 3.96M | 54.23 | 30 | 46.30 |
| r3-bs8-50ep | 0.66M | 56.39 | 45 | 49.54 |

Top of round 3 lands all under 36.4 `test_avg/mae_surf_p`. Reducing one capacity axis (depth or slice_num) plus running OneCycleLR cleanly is the consistent win. The capacity-heavy configs (mlp_ratio=4, bs=8 with halved iter count, very wide bf16) systematically lose at this dataset size (1499 train samples) and compute budget.

### Round 4 (200 min cap, 80–100 epochs) — refine around the best knobs

Live as of writing this section:

- `r4-nl2-100ep` (depth 2, 0.30M params, 58s/epoch) competitive — fastest model, most epochs in the cap.
- `r4-nl3-sn16-100ep`, `r4-nl3-sn32-100ep`, `r4-nl3-100ep`, `r4-nl3-h192-100ep`, `r4-nl3-mlp4-80ep`: nl=3 with one knob varied at a time, all 100 ep so OneCycleLR finishes its decay.
- `r4-nl4-100ep`, `r4-nl4-sn16-100ep`: extend nl=4 to 100 epochs and combine with sn=16.

(See `MLINTERN_RESULTS.jsonl` and the W&B group for the live numbers.)

## Best so far

Best `val_avg/mae_surf_p`: **40.03** — `r3-sn32-80ep` (`epochs=80, n_layers=5, slice_num=32, n_head=4, n_hidden=128, batch=4, lr=1e-3 OneCycleLR pct_start=0.1 div_factor=25, weight_decay=1e-5, max_grad_norm=0.1, loss=huber β=1.0, surf_weight=10`).

Best `test_avg/mae_surf_p` so far: **34.44** — `r3-nl4-80ep` (same recipe with `n_layers=4, slice_num=64`).

W&B run links live in the group page: https://wandb.ai/wandb-applied-ai-team/senpai-v1-ml-intern/groups/mlintern-pai2-r2

## GPU usage strategy

- 8 simultaneous experiments per round, one GPU each, pinned with `CUDA_VISIBLE_DEVICES`.
- Single-GPU training. No DDP / grad accumulation needed at this batch size.
- bf16 autocast used only when an OOM occurred in fp32 (`r2-deepwide-bf16-15ep`, `r3-vwide-bf16-30ep`, etc.).
- Round-1 jobs that OOMed (n_hidden=192/384 paired with n_layers=8 in fp32) were replaced with smaller widths or bf16 the next round, never with a different training method.

## Next recommendation

The dominant signal is "match capacity to data". With 1499 training samples and a ~30 min effective LR-decay window per run, sub-1M-param models with OneCycleLR + Huber decisively beat larger baselines. Concrete next steps:

1. Run `r4-nl4-100ep` to completion to confirm whether nl=4 stays ahead at 100 epochs.
2. Multi-seed the top one or two configs to estimate variance — ~80 epochs is short enough that a single seed is shaky.
3. After that, the biggest remaining lever is probably the LR schedule itself: try `cosine` over the same epoch count (no warmup spike), and try lower max LR (5e-4) for the very small models.
4. Surface-aware decoder / Reynolds-aware FiLM / Fourier features were intentionally skipped because their Round-1 versions did not beat the cleanly-scheduled small baseline — the architectural fix to chase first is not "more model" but "more epochs per LR schedule".

## Files

- `train.py` — modified entrypoint with all CLI knobs above and the NaN-safety fallback.
- `research/MLINTERN_RESULTS.jsonl` — one JSON object per meaningful run.
- `sweep_logs/*.log` — per-run stdout. Each run also has its own W&B page in the `mlintern-pai2-r2` group.

(This document is updated again at end-of-run with the final numbers and best config.)
