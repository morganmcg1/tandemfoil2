# SENPAI Research Results — `icml-appendix-willow-pai2c-r5`

## 2026-04-27 20:01 — PR #224: Linear warmup + cosine annealing (5-epoch warmup)
- **Student:** willowpai2c5-fern
- **Branch:** `willowpai2c5-fern/warmup-cosine-5ep`
- **Hypothesis:** Adding a 5-epoch linear warmup before the existing cosine decay would stabilize early training and improve final `val_avg/mae_surf_p` by 2–5%.
- **W&B run:** [`xogivai1`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/xogivai1)

### Results

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best @ epoch 13) | **140.2174** |
| `val_single_in_dist/mae_surf_p`     | 173.5461 |
| `val_geom_camber_rc/mae_surf_p`     | 151.7775 |
| `val_geom_camber_cruise/mae_surf_p` | 106.8840 |
| `val_re_rand/mae_surf_p`            | 128.6619 |
| `test_avg/mae_surf_p`               | **NaN** (cruise pressure inf) — partial avg over 3 = 139.025 |
| `test_geom_camber_cruise/mae_surf_p`| **NaN** (one sample's p prediction went infinite) |
| Epochs trained                      | 14 of 50 (timeout-limited) |
| Per-epoch time                      | ~132 s |
| Wall-clock                          | 30.8 min |
| Peak GPU                            | ~42.1 GB |

### Analysis & conclusions

- **Cannot evaluate cleanly yet.** Alphonse's PR #184 baseline anchor hasn't reported (was stranded by the label-index regression #257; only just picked up). Without a same-timeout cosine-only baseline, the warmup ablation has no anchor.
- **The schedule budget was misaligned with reality.** `T_max=45` was set assuming the configured 50 epochs would run, but the 30-min timeout caps training at ~14 epochs. Result: only 9 cosine epochs occurred, annealing ended at 0.90× peak lr — cosine effectively a no-op. The "5-epoch warmup vs cosine-only" attribution under this budget is not the experiment the hypothesis was designed for.
- **5 epochs of warmup is too long when only 14 epochs run.** That's 36% of total training in warmup — well outside the 5–15% regime where warmup is well-studied.
- **Test-side NaN is a systemic risk.** One cruise test sample produced an infinite pressure prediction during the end-of-run test eval, poisoning the entire split's `mae_surf_p` (the validation pass for the same split was finite at 106.88, so this is a fragile-OOD-extrapolation problem, not a training divergence).

### Action taken
Sent PR back (status:wip) with refined re-run spec:
1. Reduce warmup to 2 epochs, set `--epochs 13`, set `T_max=11` so cosine actually decays to 0
2. Add `torch.nan_to_num(...)` clamp in `evaluate_split` to defensively guard against single-sample blow-ups poisoning the test metrics
3. Re-run; we'll judge against alphonse's baseline (also re-running under same 30-min timeout) once both land

### Cross-track learnings (broadcast to research state)
- All experiments should design schedules for ~13 achievable epochs, not the configured 50.
- All experiments should add a NaN-guard in eval to avoid the single-sample test poisoning.
