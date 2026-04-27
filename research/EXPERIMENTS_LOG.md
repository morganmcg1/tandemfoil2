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

## 2026-04-27 20:21 — PR #227: Smooth-L1 (Huber β=1.0) surface loss → MERGED
- **Student:** willowpai2c5-nezuko
- **Branch:** `willowpai2c5-nezuko/huber-surface-loss`
- **Hypothesis:** Replace MSE with Smooth-L1 (β=1.0) on surface loss only; MSE on volume unchanged. Heavy-tail pressure residuals should benefit from the linear gradient regime outside β.
- **W&B run:** [`6bylngu8`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/6bylngu8)

### Results

| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best @ epoch 14) | **112.1574** |
| `val_single_in_dist/mae_surf_p`     | 147.6473 |
| `val_geom_camber_rc/mae_surf_p`     | 112.0473 |
| `val_geom_camber_cruise/mae_surf_p` |  89.5626 |
| `val_re_rand/mae_surf_p`            |  99.3723 |
| `test_avg/mae_surf_p`               | **NaN** (cruise poisoned); partial avg 3 splits = 107.79 |
| Epochs trained                      | 14 of 50 (timeout-limited, ~133 s/epoch) |
| Peak GPU                            | ~42.1 GB |

### Analysis
- **MERGED as new track baseline.** First officially landed result with better-than-average confidence.
- Beats fern's warmup-cosine (140.22) by **20%**, but comparison is confounded (nezuko ran with no warmup, fern ran with 5-epoch warmup). Clean attribution vs MSE-no-warmup awaits alphonse's PR #184.
- Best epoch is the last — model still improving at timeout. Both this and future runs are under the T_max-50 vs 14-epoch reality mismatch flagged from fern's run.
- NaN on cruise test split (one sample, pressure blow-up). Same as fern — NaN-guard now a standard recommendation for all future PRs.

### Action
- **Merged.** BASELINE.md updated to `val_avg/mae_surf_p = 112.1574`.
- **Nezuko reassigned** to PR #259: pure L1 on surface (loss-form sweep, β=∞ limit) to isolate tail-damping vs smooth-near-zero mechanism.
- **Askeladd**: fresh PR #260 created (closed stuck PR #221 which failed label indexing for 2+ hours).

## 2026-04-27 20:33 — PR #228: Larger batch + sqrt LR (bs 4→8, lr 7.07e-4) → CLOSED
- **Student:** willowpai2c5-tanjiro
- **Branch:** `willowpai2c5-tanjiro/larger-batch-8-sqrt-lr` (deleted)
- **Hypothesis:** bs=8 + sqrt-LR scaling improves gradient quality and throughput.
- **W&B run:** [`33ltg5ro`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/33ltg5ro)

### Results
| Metric | Value |
|---|---|
| `val_avg/mae_surf_p` (best @ epoch 9) | **149.5436** (vs baseline 112.16 = 33% worse) |
| Per-epoch wall time | ~130 s (essentially identical to bs=4's ~133 s) |
| Peak VRAM | 84.2 GB (bs=4 was ~95 GB on alphonse — bs=8 actually slightly less because batched mesh-padding more efficient) |
| `test_avg/mae_surf_p` | NaN (cruise — pre-existing) |

### Analysis
- **Hypothesis falsified by direct throughput data.** bs=8 took 130 s/epoch vs bs=4's 133 s — Transolver's attention is mesh-size-dominated (slice tokens × max-mesh-N), NOT batch-dominated. Batch-scaling buys no wall-clock budget under the 30-min cap.
- 33% regression past 5% close threshold; closed as clear dead end.
- Tanjiro's analysis was the basis of the close — she correctly identified the throughput-vs-gradient-noise tradeoff. Excellent negative-result reporting.

### Action
- **Closed.** Tanjiro reassigned to **PR #263 (bf16 autocast)** — proper way to buy throughput here, predicted 1.5–2× speedup on attention-heavy Transolver, doubles effective epoch budget.

## 2026-04-27 21:00 — PR #225: Higher surface weight surf_weight=25 (MSE surface) → CLOSED
- **Student:** willowpai2c5-frieren
- **Branch:** `willowpai2c5-frieren/surf-weight-25` (deleted)
- **Hypothesis:** Increasing `surf_weight` from 10→25 would push optimization toward `val_avg/mae_surf_p` (the primary metric), since surface nodes are <5% of total nodes and may be underweighted even at sw=10.
- **W&B run:** [`tzk7dwzb`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/tzk7dwzb)

### Results (best epoch = 14 of 50, timeout-limited)

| Split | `mae_surf_p` | Δ vs baseline (112.16) |
|---|---|---|
| `val_single_in_dist`     | 151.5108 | +3.86 worse |
| `val_geom_camber_rc`     | 136.2449 | +24.20 worse |
| `val_geom_camber_cruise` |  95.8309 | +6.27 worse  |
| `val_re_rand`            | 108.9428 | +9.57 worse  |
| **`val_avg/mae_surf_p`** | **123.124** | **+10.97 (+9.8%) — WORSE** |
| `test_avg/mae_surf_p`    | NaN (cruise inf) | — |

### Analysis & conclusions

- **Dead end for MSE+sw=25.** All four val splits got worse vs baseline (Huber+sw=10).
- **Important confound:** PR #225 was branched before Huber surface loss (PR #227) was merged, so this compares MSE+sw=25 vs Huber+sw=10 — two things changed simultaneously.
- **Student's diagnosis was correct:** sw=25 gives surface nodes ~500× per-node gradient weight. Excessive surface-signal dominance degrades volume feature quality, which also scaffolds surface predictions. The OOD geometry split (val_geom_camber_rc) was hit hardest (+21.6%) — unseen-camber generalization depends on broad volume coherence.
- **Cleanly falsified: MSE+sw=25 < Huber+sw=10** on val_avg/mae_surf_p.
- **Open question:** What is the optimal surf_weight on top of the Huber baseline? Likely somewhere between 10 and 25. Frieren assigned PR #265 to test Huber+sw=15.

### Action
- **Closed.** Frieren reassigned to **PR #265 (Huber+surf_weight=15)** — cleans up the confound, directly tests surf_weight=15 on top of the merged Huber baseline.

## 2026-04-27 20:37 — PR #184: Baseline anchor (default Transolver, MSE) → CLOSED
- **Student:** willowpai2c5-alphonse
- **Branch:** `willowpai2c5-alphonse/baseline-anchor-default-config` (deleted)
- **Hypothesis:** Establish reference metric for the track using the unmodified default Transolver config.
- **W&B run:** [`c9g7dxst`](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-r5/runs/c9g7dxst)

### Results (best epoch = 13)
| Metric | Value |
|---|---|
| **`val_avg/mae_surf_p`** | **134.0906** |
| `val_single_in_dist/mae_surf_p` | 155.2003 |
| `val_geom_camber_rc/mae_surf_p` | 146.9435 |
| `val_geom_camber_cruise/mae_surf_p` | 110.5492 |
| `val_re_rand/mae_surf_p` | 123.6694 |
| `test_avg/mae_surf_p` | NaN (cruise inf — same pattern) |
| Peak GPU | ~95 GB / 96 GB (very tight) |
| Epochs | 14 of 50 (30-min timeout) |

### Analysis
- **Now provides clean attribution for the merged Huber win:** PR #227 (Huber surface) at 112.16 vs alphonse's MSE 134.09 = **clean 16.4% improvement, no confound** — same arch/schedule/budget, only loss form differs.
- **Warmup actually HURT under timeout:** PR #224 v1 (5-epoch warmup, MSE preserved) at 140.22 vs alphonse 134.09 = **4.5% regression**. The 5-epoch warmup ate too much of the 14-epoch budget. fern's v2 with 2-epoch warmup + `--epochs 13` will test whether shorter warmup + aligned schedule rescues it.
- VRAM at 95/96 GB on default config means any architectural growth (more hidden, layers, slices) needs compensating cuts elsewhere (e.g. `batch_size`).
- Cruise test NaN: same single-sample pressure-blow-up pattern as fern, nezuko, tanjiro. Confirmed pre-existing scoring fragility, not config-specific.

### Action
- **Closed as completed reference run.** Logged for the team; not merging because superseded by PR #227 (and its job was to anchor, not to win).
- **Alphonse reassigned to PR #264 (EMA weight averaging, decay=0.999)** — directly addresses the late-epoch bounce-back she observed (134.09 @ ep13 → 174.38 @ ep14, 30% one-step degradation). EMA smooths exactly that kind of noise.

### Cross-track learnings reinforced
- **Schedule must be aligned to ~13 achievable epochs** (T_max=epochs_actual, not configured 50) — confirmed from 3 independent runs (alphonse, fern, nezuko, tanjiro) that 14 epochs is the wall-clock ceiling at fp32.
- **NaN-guard in evaluate_split is mandatory** — every PR seeing the same cruise blow-up. `torch.nan_to_num(pred_orig, nan=0.0, posinf=2e4, neginf=-2e4)` after denormalization.
- **Warmup is bad under tight timeout** unless schedule fully aligned (fern v2 will confirm).
- **Batch-scaling is a dead end** for throughput gains on Transolver under our setup (mesh-size-dominated attention).
