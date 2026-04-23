# TandemFoilSet Baseline

**Advisor track:** `kagent_v_students`
**Research tag:** `kagent-v-students-20260423-2055`
**W&B project:** `wandb-applied-ai-team/senpai-kagent-v-students`

---

## Current best

**None yet.** This is the bootstrap round. Every first-round PR includes a vanilla baseline run (default Transolver + default hyperparameters, `--wandb_name <student>/<exp>-baseline`) to anchor the `val_avg/mae_surf_p` for this research track.

Reproduce command:

```bash
cd target && python train.py \
    --agent <student> \
    --wandb_name "<student>/<experiment>-baseline"
```

Default config:

| Param | Value |
|-------|-------|
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| epochs | 50 |
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| loss | MSE (vol + surf_weight × surf) in normalized space |

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean across four validation splits. Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — same quantity, computed from the best-val checkpoint on the four held-out test splits at the end of each run.

Per-split diagnostics are logged for every run — see `program.md`.

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here, the advisor:

1. Squash-merges the winning PR into `kagent_v_students`.
2. Updates this file with the new metric, PR number, and W&B run link.
3. Commits the update on the advisor branch.
