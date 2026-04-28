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
- **NaN test poisoning is a real `data/scoring.py` bug, not a model issue.** Identified by askeladd in PR #748 (since closed): `accumulate_batch` does `err * mask` where `NaN * 0 = NaN` in IEEE-754. `test_geom_camber_cruise/sample 20` has 761 NaN values in the **ground truth** `p` channel. This poisons `mae_surf_p` for that split on every run regardless of model. Fix: `torch.where(mask, err, 0)`. Same pattern in `evaluate_split` for `vol_loss`/`surf_loss` produces `Infinity`. Assigned to askeladd as PR #807. Once that lands, all future runs will produce clean `test_avg` numbers.
- **Cruise OOD camber (M=2-4)** is otherwise the most extrapolation-prone test split — already the hardest extrapolation track regardless of the NaN bug.

## 2026-04-28 20:02 — PR #750: Linear warmup + cosine LR schedule (lr=1e-3, wd=5e-4)
- **Branch:** `willowpai2e3-edward/lr-warmup-cosine`
- **Hypothesis:** 500-step linear warmup + per-step cosine to 0, lr=1e-3, wd=5e-4 — should buy 3–8% over plain cosine.
- **Run:** W&B `thnnvgaw`, 14/50 epochs (timeout), best ckpt @ epoch 12, peak 83.6 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 162.00 | 144.47 |
| `*_geom_camber_rc` | 148.21 | 136.08 |
| `*_geom_camber_cruise` | 102.53 | null (scoring bug) |
| `*_re_rand` | 130.84 | 127.33 |
| **avg** | **135.89** | **null** |

### Decision: SEND BACK
- 1.7% better than student's quoted baseline-mean reference (5 runs, range [124.6, 146.1]) — within noise; the hypothesized 3–8% gain isn't demonstrated.
- Student's diagnosis is exactly right: cosine schedule with `T_max = 50 × 375 ≈ 18.7K steps` but only ~5.2K steps fit in 30 min — never reaches the low-LR fine-tuning regime.
- Sent back: set `--epochs 14` explicitly so cosine actually anneals end-to-end; raise peak LR to `2e-3` (warmup makes higher peaks safe — that's where the standard transformer warmup gain lives).
- (Branch hygiene: edward referenced 5 baseline runs t0xgo0zv/6zc9kq6x/6lj642bf/7qi7tbcy/zaqz12qi as a noise band; only zaqz12qi (alphonse v1) is from this advisor branch, the others are out-of-scope. The variance argument stands; the specific run-IDs do not.)

## 2026-04-28 20:04 — PR #748: Transolver 2x capacity scale-up
- **Branch:** `willowpai2e3-askeladd/transolver-2x-capacity`
- **Hypothesis:** n_hidden=192, n_layers=8, n_head=8, slice_num=128, mlp_ratio=4 (3.42M params) — predicted 5–15% gain.
- **Run:** W&B `p486z24b`, **4/50 epochs only** (timeout), best ckpt @ epoch 3, peak 82.5 GB.
- **val_avg/mae_surf_p = 203.16** (raw); test_avg null (scoring bug); **test_avg corrected = 191.71** (offline re-eval with `torch.where`).

### Decision: CLOSE
- val_avg = 203 vs. round-1 baseline-range ~140 — clear regression at 4/50 epochs.
- Approach not broken — it's that 50-epoch cosine schedule with only 4 epochs done means LR is still ~98% of peak. Model never reached convergence regime where 2× capacity is supposed to help.
- **Critical bug discovery embedded in PR comments**: pinpointed the `data/scoring.py` NaN-mask bug, validated the fix offline. Spawned PR #807 to land the fix as their next assignment.
- For round 2 capacity: budget-matched schedule (`--epochs 4` so cosine completes), or smaller capacity boost (n_layers=6 to fit ~8 epochs) — defer until scoring fix merges.

## 2026-04-28 20:15 — PR #756: Fourier features for log(Re) input encoding
- **Branch:** `willowpai2e3-frieren/fourier-re-encoding`
- **Hypothesis:** Replace scalar `log(Re)` (dim 13) with 16 sin/cos features at 8 freqs `[1, 2, 4, 8, 16, 32, 64, 128]` for richer cross-Re generalization. Predicted 3-7% gain.
- **Run:** W&B `t0xgo0zv`, 14/50 epochs (timeout), best ckpt @ epoch 14, peak 42.3 GB.

| Split | val | test |
|---|---|---|
| `*_single_in_dist` | 202.03 | 169.41 |
| `*_geom_camber_rc` | 139.03 | 122.76 |
| `*_geom_camber_cruise` | 102.50 | null (scoring bug) |
| `*_re_rand` | 121.42 | 123.83 |
| **avg** | **141.25** | **null** |

### Decision: SEND BACK
- val_avg=141.25 sits inside the round-1 noise band (other v1 runs: alphonse 146.10, edward 135.89). Predicted 3-7% gain not demonstrable at single-seed.
- val_re_rand=121.42 (the strongest per-split) is suggestive — Fourier-of-log(Re) may help cross-Re generalization. Direction worth iterating on, not abandoning.
- val curve still falling steeply at the cutoff (epochs 11-14: 152→160→160→141) — model under-converged.
- Sent back: (a) concatenate Fourier features instead of replacing dim 13 (preserves smooth scalar path); (b) drop top frequencies, use `[1, 2, 4, 8, 16, 32]` (high freqs cycle below the data's Re resolution); (c) `--epochs 14` explicit so cosine completes.
- Don't redirect yet — if v2 still lands in the 135-146 band, then Fourier-of-Re isn't a strong enough lever at this budget and we'll switch frieren to something else.

## 2026-04-28 20:08 — PR #807 (assigned): Bug fix — NaN-safe masked accumulation
- **Branch:** `willowpai2e3-askeladd/scoring-nan-mask-fix`
- **Type:** Infrastructure bug fix (not a hypothesis experiment)
- **Scope:** `data/scoring.py::accumulate_batch` (the read-only marker explicitly allows bug-fix PRs per CLAUDE.md cherry-pick guidance) and the matching `err * mask` pattern in `train.py::evaluate_split` and the training loop.
- **Verification asks:** (1) `--debug` smoke run produces identical numbers on clean splits, (2) re-evaluate saved checkpoints `zaqz12qi` (alphonse v1) and `p486z24b` (askeladd transolver-2x) against the fixed scorer to retrieve corrected `test_avg/mae_surf_p` values without retraining.
