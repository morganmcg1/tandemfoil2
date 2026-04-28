# SENPAI Research Results — willow-pai2e-r3

## 2026-04-28 19:55 — PR #743: Per-channel surface loss: 3× weight on pressure
- **Branch:** `willowpai2e3-alphonse/channel-weighted-surface-loss`
- **Hypothesis:** Boost pressure channel by 3× inside surface loss to align training signal with the `mae_surf_p` ranking metric. `y_std_p ≈ 679`, ~30× larger than `y_std_Ux` and ~70× larger than `y_std_Uy`; uniform-weighted MSE under-emphasizes the metric channel.
- **Run:** W&B `zaqz12qi` (entity `wandb-applied-ai-team`, project `senpai-charlie-wilson-willow-e-r3`)
- **Budget consumed:** 14/50 epochs (hit `SENPAI_TIMEOUT_MINUTES=30` cap; ≈131 s/epoch)

### Results

| Split | val (best ckpt @ epoch 14) | test (best ckpt @ epoch 14) |
|---|---|---|
| `*_single_in_dist` | 196.59 | 166.63 |
| `*_geom_camber_rc` | 156.43 | 141.34 |
| `*_geom_camber_cruise` | **107.40** | **null** |
| `*_re_rand` | 124.01 | 122.96 |
| **avg** | **146.10** | **null (cruise NaN propagates)** |

### Analysis & decision: SEND BACK
- Val side is informative: `val_avg/mae_surf_p = 146.10`, with `cruise = 107.40` the best of the four val splits — consistent with the hypothesis that p-channel boost helps where p dominates surface dynamics.
- **Test side blocks merge.** `test_geom_camber_cruise/mae_surf_p = null` (single non-finite prediction polluting the global accumulator in `accumulate_batch` — `data/scoring.py` only skips on non-finite ground truth, not non-finite preds). Three of four test splits finite. Per CLAUDE.md, NaN/missing on the paper-facing metric blocks adoption.
- Budget reality check: at default `SENPAI_TIMEOUT_MINUTES=30` and current model size, only ~14 epochs fit. **All round-1 PRs are timeout-limited to ~14 epochs**, not 50. Future hypothesis design should account for this — recommend setting `--epochs 14` explicitly so cosine annealing reaches end-of-curve LR rather than mid-curve.
- Sent back with feedback to:
  1. Add a NaN-guard / clamp in `evaluate_split` (`pred = torch.nan_to_num(pred, ...).clamp(-20, 20)` before denormalization) — defends MAE numerics for all future students once merged.
  2. Try softer per-channel weights `[1.0, 0.5, 2.0]` instead of `[1.0, 1.0, 3.0]` — 2× boost on p (closer to variance ratio after surface gating absorbs most of it) plus 0.5× on Uy (over-represented and not in ranking metric).
  3. Set `--epochs 14` explicitly to plan for the timeout.
- Once v2 lands with finite `test_avg/mae_surf_p`, this becomes the founding baseline for the branch.

### Cross-cutting findings (apply to ALL round-1 students)

- **Timeout is the binding constraint, not epoch count.** Plan for 14 epochs, not 50.
- **NaN test poisoning is a systemic risk.** Likely to recur on other round-1 PRs as they submit. The `nan_to_num + clamp` defensive fix is the same one-line fix everywhere; will roll into baseline once first PR with the fix merges.
- **Cruise OOD camber (M=2-4) is the most extrapolation-prone test split.** Whatever sample is causing the NaN pred there will probably keep biting. Worth investigating which sample(s) trigger it as a follow-up.
