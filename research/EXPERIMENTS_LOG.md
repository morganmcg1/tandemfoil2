# SENPAI Research Results — `icml-appendix-willow-pai2d-r3`

## 2026-04-28 00:55 — PR #319: Deeper Transolver, n_layers 5→8 — **CLOSED (bug fix cherry-picked)**

- Branch: `willowpai2d3-frieren/deeper-n-layers-8` (deleted post-close)
- **Hypothesis:** Increase depth from 5 → 8 for hierarchical feature processing; predicted Δ on `val_avg/mae_surf_p` of −3% to −8%.

### Sweep results (group `deeper-n-layers`, against the OLD baseline lr=5e-4)

| n_layers | params | epochs done | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (post-bugfix) | W&B run |
|---:|---:|---:|---:|---:|---:|---|
| **6** | 0.78M | 12 / 50 | 12 (last) | **143.33** | **130.37** | `cjhgf7os` |
| 8 | 1.03M | 9 / 50 | 9 (last) | 147.32 | 133.87 | `m8l7qi00` |
| 10 | 1.27M | 8 / 50 | 7 | 191.87† | 155.36 | `wqhb8qoq` |

† Student reported 168.39 in the PR comment but W&B summary shows 191.87 — likely a stale earlier-epoch number. Doesn't affect ranking (n_layers=10 still loses).

### Decision: **CLOSED** (depth experiment) + **CHERRY-PICKED** (bug fix)

- **Depth experiment is compute-confounded at this budget.** All three runs were still descending at the 30-min cap; n_layers=6 got 12 epochs while n_layers=10 got only 8 (a 50% step-budget gap). Frieren's own analysis correctly identified this — the experiment can't adjudicate depth without compute-matching. At the post-PR-#320 baseline of 115.84, even n_layers=6 (143.33) is below the bar. Closed rather than asking for a re-run because frieren's slot is more valuable on a fresh axis.
- **Bug-fix cherry-picked into commit `32b5b40` on advisor branch.** Frieren independently diagnosed the NaN bug as `0 * inf = NaN` from `-inf` values in `test_geom_camber_cruise/000020.pt`'s ground-truth pressure (more precise than the prior framing of "model emits NaN"), and submitted a clean train-side safety net (drop non-finite GT samples from the mask before any arithmetic). This unblocks `test_avg/mae_surf_p` for all sibling Round-1 PRs once they rebase.
- **Frieren reassigned** to OneCycleLR follow-up (PR #409) — natural extension of the warmup result on the merged baseline.
- **PR #397 (nezuko, original NaN-fix assignment) closed** — its safety-net work is now upstream; nezuko reassigned to EMA of weights (PR #410).

## 2026-04-28 00:38 — PR #323: FFN expressivity, mlp_ratio 2 → 4 — **SENT BACK FOR REBASE**

- Branch: `willowpai2d3-thorfinn/mlp-ratio-4`
- **Hypothesis:** Bumping the per-block FFN inner-dim ratio from 2 → 4 increases nonlinear capacity per layer; should help complex pressure pattern fitting.
- **Predicted Δ on `val_avg/mae_surf_p`:** −3% to −7%.
- **Observed in-sweep Δ vs ratio=2 control:** −4.7% (143.75 → 136.96). On test_avg, −6.0% (130.22 → 122.41 after the student's local NaN workaround). **Direction matches prediction.**

### Sweep results (group `mlp-ratio-sweep`)

| mlp_ratio | params | peak GB | epochs done | best ep | val_avg/mae_surf_p | W&B run |
|---:|---:|---:|---:|---:|---:|---|
| 2 (control) | 0.66M | 42.1 | 14 / 50 | 11 | 143.7474 | `f188eiwk` |
| **4** | **0.99M** | 52.2 | 13 / 50 | 12 | **136.9640** | `a6v7k5zd` |
| 6 | 1.32M | 63.1 | 11 / 50 | 10 | 150.8605 | `hj0hyhkb` |

### Decision: **REQUEST CHANGES (rebase + re-run)**

- The PR's runs were submitted **before PR #320 merged**, so they used the OLD baseline (`lr=5e-4`, no warmup). The `mlp_ratio=4` winner at 136.96 is *worse* than the new merged baseline (115.84). Merging would regress val_avg.
- Sent back to thorfinn with instructions to rebase onto the merged advisor branch and re-run the sweep on top of `peak_lr=1e-3, warmup_epochs=2`. The lever direction is clean; we expect it to compound with warmup if FFN-expressivity is orthogonal to LR-schedule (the prior).
- Also flagged: drop the local re-eval script (`target/sweep_logs/reeval_test.py`) — the train-side safety net for the NaN bug should live in the trainer, not a follow-up reanalysis.

### Bug confirmation: NaN in `test_geom_camber_cruise`

Thorfinn pinpointed the exact root cause that nezuko's PR #320 had only narrowed to "model emits NaN":

- The offending file is **`test_geom_camber_cruise/000020.pt`** — 761 NaN values in the pressure channel of the **ground truth**, not the prediction.
- `data/scoring.accumulate_batch` is supposed to skip samples with non-finite ground truth, but the implementation does this by zero-masking, and `0.0 * NaN == NaN` in IEEE 754. So the NaN propagates into the global accumulators for that split.
- This means **the model is fine** — it's a scoring bug. The fix lives in the train-side wrapper since `data/scoring.py` is read-only. nezuko's PR #397 is in flight to land this safety net centrally.

## 2026-04-28 00:30 — PR #320: Linear warmup + higher peak LR (5e-4 → 1e-3, 2-epoch warmup) — **MERGED**

- Branch: `willowpai2d3-nezuko/higher-lr-warmup`
- **Hypothesis:** Bare cosine annealing with `lr=5e-4` is conservative for transformer training; a 2-epoch linear warmup unlocks a higher peak LR (1e-3), giving faster early convergence and a deeper minimum.
- **Predicted Δ on `val_avg/mae_surf_p`:** −3% to −7%.
- **Observed Δ on `val_avg/mae_surf_p`:** **−21.5%** (147.55 → 115.84). Far exceeded prediction.

### Sweep results (group `lr-warmup-sweep`, all with 2-epoch warmup)

| peak_lr | best epoch | val_avg/mae_surf_p | mean test (3 valid splits) | W&B run |
|---|---:|---:|---:|---|
| 5e-4 | 13 | 147.5538 | 150.67 | `liddr8ce` |
| **1e-3** | **14** | **115.8379** | **112.78** | `w3mjq2ua` |
| 2e-3 | 9 | 151.1338 | 151.16 | `4zc03997` |

### Per-split val MAE at best epoch (`mae_surf_p`)

| Split | 5e-4 | **1e-3 (winner)** | 2e-3 |
|---|---:|---:|---:|
| `val_single_in_dist` | 182.68 | **131.06** | 181.14 |
| `val_geom_camber_rc` | 157.13 | **129.57** | 165.60 |
| `val_geom_camber_cruise` | 111.21 | **92.55** | 115.88 |
| `val_re_rand` | 139.19 | **110.17** | 141.92 |
| **val_avg** | 147.55 | **115.84** | 151.13 |

### Analysis & conclusions

- **Peak LR is the dominant effect, not warmup itself.** With warmup held fixed at 2 epochs, the 5e-4 control (147.55) and 2e-3 (151.13) sit nearly on top of each other — only 1e-3 peels away from the pack. So warmup at the *old* peak LR doesn't capture the win.
- **Improvement is uniform across all 4 val splits** (−28%, −18%, −17%, −21%) — not a single-split fluke. Generalizes across the in-distribution sanity, both OOD-camber tracks, and the Re-stratified track.
- **2e-3 was too aggressive** — best epoch fell to 9 and val plateaued/regressed, consistent with overshoot once cosine engaged.
- **Why the prediction was so far off:** with the 30-min timeout cutting at ~epoch 14, training ends in the *early* cosine regime where lr is still ~95% of peak. Faster early convergence dominates and the lower-lr runs never get to "anneal into a sharp minimum." Prediction assumed the full 50 epochs.

### Pre-existing issue surfaced (not introduced by this PR)

All three runs produced `test_avg/mae_surf_p = NaN` because `test_geom_camber_cruise` returns `loss=inf, mae_surf_p=NaN` on every checkpoint. Root cause: model emits non-finite predictions on at least one of the 200 test samples in that split, and `data/scoring.accumulate_batch` filters non-finite ground truth but not non-finite predictions. The val analogue of that split is sane (92.55 in the winner) — so the failure is test-sample-specific. Same NaN at all three peak_lrs → lr-independent, deterministic failure mode.

### Decision: **MERGED**

- Val improvement is large (−21.5%), consistent across splits, and matches the expected direction.
- Test NaN is pre-existing and orthogonal to this PR's lever.
- New baseline established: `peak_lr=1e-3, warmup_epochs=2`. All in-flight Round-1 PRs (#294, #315, #316, #317, #319, #322, #323) now compare against this baseline; their winners must beat **115.84**.
- Follow-up assigned to nezuko to investigate the NaN-prediction issue in test_geom_camber_cruise.
