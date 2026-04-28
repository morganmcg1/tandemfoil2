# SENPAI Research Results

## 2026-04-28 22:30 — PR #787: Fourier feature PE on (x,z) — compound base [CLOSED]
- Branch: `willowpai2e2-thorfinn/compound-fourier-pe` (closed, branch deleted)
- Hypothesis: Gaussian random Fourier features (Tancik 2020), m=8, sigma=1.0 on (x,z), concatenated to input before preprocess MLP. Should help slice attention partition geometry by surface-aware frequencies.
- W&B run: `ph75483c` (project `senpai-charlie-wilson-willow-e-r2`).

| metric | value |
|---|---:|
| best `val_avg/mae_surf_p` | **100.12** (epoch 27, baseline 40.93 → +59 / +144%) |
| `test_avg/mae_surf_p` | 89.97 |
| test_geom_camber_cruise/mae_surf_p | 69.28 |
| test_geom_camber_rc/mae_surf_p | 101.01 |
| test_re_rand/mae_surf_p | 92.95 |
| test_single_in_dist/mae_surf_p | 96.63 |
| wall clock | 30.9 min |

- Outcome: **Closed** — decisive negative result. Confirmed across the uncompressed baseline (earlier exploration) and the compressed compound nl3/sn16/nh1 anchor. Likely mechanism: slice-token attention already learns coordinate-aware partitions, so a fixed random-frequency basis at the input crowds channels with redundant info and biases the slice partitioner toward Fourier-aligned cuts that don't align with airfoil structure (LE/TE, suction/pressure surfaces, wake direction). Coordinate encoding for Transolver should respect the slice abstraction, not bypass it — future work in this lane should target *inside-slicer* position info or structural priors (signed distance, surface-normal projection) rather than generic Fourier.

## 2026-04-28 22:30 — PR #782 (round 2): GeGLU param-matched (h=168) on compound base [CLOSED]
- Branch: `willowpai2e2-edward/compound-geglu` (closed, branch deleted)
- Hypothesis: Round-2 retest of GeGLU controlling for FFN parameter count. hidden_inner=168 (mlp_ratio=1.3125), 0.986× param ratio vs GELU mlp_ratio=2 baseline. Clean activation-only A/B.
- W&B run: `7hyra9fj` (project `senpai-charlie-wilson-willow-e-r2`); round-1 confounded run was `v9ruqc0v`.

| metric | value |
|---|---:|
| best `val_avg/mae_surf_p` | **94.41** (epoch 21, baseline 40.93 → +53.5 / +131%) |
| test_avg/mae_surf_p (W&B) | NaN (cruise-split Inf bug) |
| offline avg of 3 finite test splits | ~92.47 |
| test_geom_camber_rc/mae_surf_p | 100.66 |
| test_re_rand/mae_surf_p | 84.36 |
| test_single_in_dist/mae_surf_p | 92.40 |
| wall clock | 30.5 min |

- Outcome: **Closed** — decisive negative result with a clean param-match. Gating in the FFN does not help Transolver at this scale (H=128, 3 layers, sn=16). Mechanism hypothesis: slice-attention already provides token-level adaptive routing, so an FFN-internal multiplicative gate is redundant and harder to optimize at low capacity. LLaMA/PaLM benefits from GeGLU/SwiGLU at much larger H — the inductive prior doesn't transfer down. If gating is to be revisited it should go at the slicer (gated slice aggregation), not the FFN.
- Tooling note: cruise-NaN bug now contaminates two consecutive runs' W&B test_avg. Per-sample finite-mask in test-eval scoring has been requested for the next student cleanup commit.

## 2026-04-28 19:30 — PR #782: GeGLU activation on compound base
- Branch: `willowpai2e2-edward/compound-geglu`
- Hypothesis: Replace GELU with GeGLU MLP at mlp_ratio=4 on the compound base (n_layers=3, slice_num=16, n_head=1, n_hidden=128) for richer FFN expressivity.
- W&B run: `v9ruqc0v` (project `senpai-charlie-wilson-willow-e-r2`).

| metric | value |
|---|---|
| best `val_avg/mae_surf_p` | 109.8891 (epoch 12 / 20) |
| `test/test_single_in_dist/mae_surf_p` | 110.8508 |
| `test/test_geom_camber_rc/mae_surf_p` | 105.7677 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN (cruise-split Inf bug) |
| `test/test_re_rand/mae_surf_p` | 102.0296 |
| `test_avg/mae_surf_p` (W&B) | NaN |
| offline-clean test_avg over 3 finite splits | ~106.22 |
| wall clock | 30.1 min (timeout) |

- Outcome: **Send back**. Result is clearly worse than even the default Transolver baseline (~80–82 reference). Critically the experiment is **confounded**: GeGLU's two parallel projections at mlp_ratio=4 mean the FFN has ~3x the params of the GELU mlp_ratio=2 baseline, so we cannot attribute the regression to the activation gate. Asked edward to re-run with mlp_ratio≈8/3 so FFN param count matches the baseline — that gives a clean activation-only A/B.
- Per-batch wall clock at this size (~90s/epoch) means even param-matched the run will stay ≤ 30 min easily.

## 2026-04-28 19:30 — PR #784: OneCycleLR scheduler swap
- Branch: `willowpai2e2-frieren/compound-onecycle`
- Hypothesis: Replace CosineAnnealingLR with OneCycleLR (warmup pct=0.05, div_factor=25, final_div_factor=1e4, per-batch stepping) for better late-epoch fine-tuning. Peak LR unchanged at 5e-4.
- W&B run: `icmk9yw4` (project `senpai-charlie-wilson-willow-e-r2`).

| metric | value |
|---|---|
| best `val_avg/mae_surf_p` | 91.7241 (epoch 32 / 50, last completed before timeout) |
| `test/test_single_in_dist/mae_surf_p` | 88.8587 |
| `test/test_geom_camber_rc/mae_surf_p` | 89.7523 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN (cruise-split Inf bug) |
| `test/test_re_rand/mae_surf_p` | 83.2516 |
| `test_avg/mae_surf_p` (W&B) | NaN |
| offline-clean test_avg over 3 finite splits | 81.62 |
| wall clock | 30.1 min (timeout) |
| LR at timeout | ≈1.6e-4 (vs scheduled floor 5e-8) |

- Outcome: **Send back**. The schedule never completed — only 32/50 epochs (64%) ran, so OneCycleLR's late-stage fine-anneal phase never engaged. Hypothesis is not testable as configured under the 30-min wall clock. Asked frieren to re-run with `--epochs 28` so the full schedule fits within the budget. If it works, we'll have a clean OneCycleLR-vs-cosine A/B once alphonse's anchor (#779) lands.
- Cleanly implemented per-batch step() so the bug story is not the schedule mechanics, only the budget.

## 2026-04-28 20:50 — PR #781: slice_num=8 on compound base
- Branch: `willowpai2e2-askeladd/compound-sn8` (closed, branch deleted)
- Hypothesis: Push slice-token compression one step further than the prior round's compound winner. `slice_num=8` on top of `n_layers=3, n_head=1, n_hidden=128, mlp_ratio=2`. Two seeds in `--wandb_group compound-sn-floor`.
- W&B runs: `4yjg44xu` (compound-sn8), `0587z7fk` (compound-sn8-seed42).

| metric | compound-sn8 | compound-sn8-seed42 |
|---|---:|---:|
| best `val_avg/mae_surf_p` | 92.7044 (ep 32 / 50) | 92.4994 (ep 33 / 50) |
| `test/test_single_in_dist/mae_surf_p` | 90.31 | 99.12 |
| `test/test_geom_camber_rc/mae_surf_p` | 98.65 | 92.90 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN (cruise-split Inf bug); offline 58.26 | NaN; offline 57.45 |
| `test/test_re_rand/mae_surf_p` | 86.63 | 82.74 |
| `test_avg/mae_surf_p` (W&B) | NaN | NaN |
| offline-clean test_avg over 4 splits | 83.46 | 83.05 |
| wall clock | 30.75 min (timeout) | 30.71 min (timeout) |
| peak GPU mem | 21.4 GB | 21.4 GB |

- Outcome: **Closed**. Both seeds tightly clustered (val 92.50 ± 0.10, test 83.3 ± 0.2), but both runs hit the 30-min wall clock at epoch 33/50 with val still monotonically descending. The sn=8 vs sn=16 ablation is **unresolved**, not negative — same pattern as PR #784 (compound + OneCycleLR also stopped at ~ep 32/50). Re-running sn=8 alone under the same fp32/batch=4 throughput would just reproduce the same undertrained number. Closing and pivoting askeladd to a tooling PR (AMP/bf16 + larger batch + NaN-safe eval) so future round-2 runs can complete a full 50-epoch schedule.
- Bug report (askeladd): root-caused the `test_avg/mae_surf_p` NaN to `accumulate_batch` doing `err * mask` after `err` already inherited NaN from non-finite `y` (sample `test_geom_camber_cruise/000020.pt` has 761 `-Inf` values in the `p` channel, not `+Inf` as previously assumed). Promoted the fix to the next askeladd assignment.

## Tooling note (both PRs)
`test_avg/mae_surf_p` logs as **NaN on every run on this branch**. Diagnosed via the offline re-evals in #782/#784: `test_geom_camber_cruise/000020.pt` contains 761 +Inf values in the `p` channel. In `data/scoring.py::accumulate_batch` the per-channel error is computed as `err = |pred − y|` *before* the validity mask is applied, then multiplied by `surf_mask.unsqueeze(-1)` — `Inf × 0 = NaN` in IEEE 754, which propagates through the subsequent `.sum`. The existing `y_finite` per-sample skip in `accumulate_batch` correctly intends to discard any sample with non-finite GT, but the multiplicative mask formulation defeats that intent. `data/scoring.py` is documented as read-only, so the right place to fix this is in `train.py::evaluate_split` before `accumulate_batch` is called — e.g. zeroing `pred_orig` and `y` at non-finite positions, or replacing the multiplicative mask with `torch.where(mask, err, 0)`. Until this lands, students should compute a clean test_avg over the three finite splits and report it in their PR comment alongside the W&B value. Worth picking up as a small dedicated PR if a student frees up.
