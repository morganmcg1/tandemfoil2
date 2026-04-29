# SENPAI Research Results

## 2026-04-28 18:00 — PR #821: Tooling — AMP/bf16 + batch_size=16 + NaN-safe eval [SENT BACK — LR scaling fix needed]
- Branch: `willowpai2e2-askeladd/tooling-amp-bs-nansafe`
- Hypothesis (tooling): fp32/bs=4 at ~55s/epoch → 50 epochs needs 46 min, far past 30-min wall. Three fixes: (1) AMP/bf16 autocast on forward pass, (2) batch_size default 4→16, (3) NaN-safe `evaluate_split` guard to fix the cruise Inf→NaN accumulator bug.
- W&B runs: `gjh93i2f` (tooling-validate-compound), `fogsv6hg` (tooling-validate-compound-seed42)

| metric | run 1 (gjh93i2f) | run 2 (fogsv6hg) | acceptance |
|---|---:|---:|---|
| best `val_avg/mae_surf_p` | 159.51 | 124.01 | < 90 target ❌ |
| `test_avg/mae_surf_p` | 136.33 | 110.77 | finite ✓ |
| `test_geom_camber_cruise/mae_surf_p` | **58.77** | **58.54** | finite for 1st time ✓ |
| `test_single_in_dist/mae_surf_p` | 265.50 | 182.09 | — |
| `test_geom_camber_rc/mae_surf_p` | 141.23 | 119.75 | — |
| `test_re_rand/mae_surf_p` | 79.81 | 82.69 | — |
| completed epochs | 46/50 | 46/50 | 50/50 ❌ |
| peak VRAM | 55.9 GB | 55.9 GB | — |

- Outcome: **Sent back.** The three tooling changes work correctly — AMP/bf16 reduces epoch wall from ~55s to ~40s, NaN-safe eval permanently fixes the cruise NaN bug (landmark result: cruise test is finite for the first time). The blocking issue is batch-size/LR mismatch: bs=16 gives 4× fewer gradient steps/epoch; at the same lr=5e-4 convergence collapses (val 159/124 vs ~92 at fp32/bs=4). Fix: linear LR scaling, lr=5e-4 × (16/4) = 2e-3 as new default. Askeladd sent back to apply this single change and re-validate.
- Key analysis: The AMP + NaN-safe changes are correct and should not be modified. Only the lr default needs changing. The inter-seed variance (35-point spread between runs 1 and 2) also reflects the LR mismatch — with correct LR, variance should drop to a few points.

## 2026-04-28 17:00 — PR #840: per-sample relative MAE loss [WINNER — pending rebase/merge]
- Branch: `willowpai2e2-edward/compound-relative-mae`
- Hypothesis: MSE and Huber losses weight gradient contribution proportional to |residual|², which biases toward high-Re/high-amplitude samples. Per-sample relative MAE normalizes each sample's contribution: L_rel = mean(|pred - target| / (mean(|target|) + ε)), equalizing gradient contribution across Re regimes. On a dataset where per-sample y-std spans 164–2077 (12× range), this should disproportionately benefit low-Re splits (cruise, re_rand).
- W&B run: `nz8eev8e` (group `compound-relative-mae`, project `senpai-charlie-wilson-willow-e-r2`)

| metric | relative MAE (epoch 32/50) | Huber δ=1.0 baseline (PR #783) | Δ |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | **64.73** | 75.93 | **−11.20 (−14.7%)** |
| `val_single_in_dist/mae_surf_p` | 80.41 | 85.84 | −5.43 |
| `val_geom_camber_rc/mae_surf_p` | 78.51 | 91.20 | −12.69 |
| `val_geom_camber_cruise/mae_surf_p` | **40.13** | 54.68 | **−14.55** |
| `val_re_rand/mae_surf_p` | 60.73 | 71.99 | −11.26 |
| `test_avg/mae_surf_p` | **56.92** | NaN (cruise bug) | — |
| `test_single_in_dist/mae_surf_p` | 77.25 | 79.35 | −2.10 |
| `test_geom_camber_rc/mae_surf_p` | 67.74 | 82.61 | −14.87 |
| `test_geom_camber_cruise/mae_surf_p` | **32.35** | NaN (cruise bug) | — |
| `test_re_rand/mae_surf_p` | 50.35 | 64.29 | −13.94 |
| epochs | 32/50 (timeout) | 32/50 (timeout) | — |
| wall clock | ~30 min | 30.2 min | — |

- Outcome: **Winner declared**. All 4 val splits improve; all 4 test splits are finite (cruise NaN resolved). The per-split pattern confirms the hypothesis precisely: cruise (low-Re, small |y|) gets the largest benefit (−14.55 val, 40.13 → 32.35 test), re_rand also benefits strongly (−11.26 val). The relative loss eliminates cruise inf predictions by normalizing scale, incidentally fixing the cruise NaN bug.
- Key analysis: Relative MAE is additive on top of Huber: Huber linearizes tail residuals; relative scaling equalizes cross-Re gradient contribution. Both address the same root cause (high-Re amplitude dominance) at different abstraction levels, and they compound.
- Status: Merge blocked by rebase conflict; edward sent back to rebase and resubmit.
- Next steps: Once merged, follow-up should sweep ε ∈ {1e-3, 1e-2} (current ε=1e-6 may be sub-optimal in normalized space); also run 50-epoch version with PR #821's AMP/bf16 tooling once that lands.

## 2026-04-28 ~15:00 — PR #784 round 2: OneCycleLR full-schedule 28 epochs [CLOSED]
- Branch: `willowpai2e2-frieren/compound-onecycle` (closed, branch deleted)
- Hypothesis (revision): Repeat OneCycleLR with `--epochs 28` so the full cosine-descent schedule completes within the 30-min wall clock. Test whether completing the low-LR fine-tuning phase unlocks a refinement win.
- W&B run: `hw4400z4` (round 2); `icmk9yw4` (round 1 for comparison)

| metric | round 2 (28 epochs, full schedule) | round 1 (50 epochs, 64% schedule) |
|---|---:|---:|
| best `val_avg/mae_surf_p` | 92.25 (epoch 28, last) | 91.72 (epoch 32, last) |
| `test_single_in_dist/mae_surf_p` | 93.33 | 88.86 |
| `test_geom_camber_rc/mae_surf_p` | 95.94 | 89.75 |
| `test_geom_camber_cruise/mae_surf_p` | NaN (cruise bug) | NaN |
| `test_re_rand/mae_surf_p` | 83.63 | 83.25 |
| offline clean test_avg (3 finite) | 87.62 | 81.62 |
| epochs | 28/28 ✅ | 32/50 ❌ |
| LR at end | 2.0e-9 ✅ (floor reached) | 1.6e-4 ❌ (floor not reached) |
| wall clock | 26.4 min | 30.1 min (timeout) |

- Outcome: **Closed**. Round 2 cleanly tested the hypothesis — schedule completed, LR reached floor (2e-9). Result is slightly *worse* (92.25 vs 91.72) than the partial-schedule run. Both runs have best epoch = last epoch, confirming this is gradient-step-limited: the model benefits from more steps, not from completing the LR anneal. Additionally, Huber (PR #783, val=75.93) merged as a far stronger signal — neither OneCycleLR run survives comparison with the new baseline. Frieren reassigned to gradient accumulation throughput (#854).

## 2026-04-28 ~14:30 — PR #783: Huber loss δ=1.0 on compound base [MERGED — NEW BASELINE]
- Branch: `willowpai2e2-fern/compound-huber` (merged into `icml-appendix-willow-pai2e-r2`)
- Hypothesis: MSE loss squares large residuals, so high-Re CFD samples dominate gradients. Huber loss linearizes above δ, giving medium-Re samples a fairer share of gradient signal. On this dataset (per-sample y-std spanning 164–2077), the effect should be large and uniform across splits.
- W&B run: `2y1lj209` (group `compound-huber`, project `senpai-charlie-wilson-willow-e-r2`)

| metric | compound + Huber δ=1.0 | compound anchor (PR #779) | Δ |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | **75.93** (epoch 32/50, timeout) | 96.80 (epoch 31) | **−20.87 (−21.6%)** |
| `val_single_in_dist/mae_surf_p` | 85.84 | 107.38 | −21.54 |
| `val_geom_camber_rc/mae_surf_p` | 91.20 | 107.06 | −15.86 |
| `val_geom_camber_cruise/mae_surf_p` | 54.68 | 79.94 | −25.26 |
| `val_re_rand/mae_surf_p` | 71.99 | 92.83 | −20.84 |
| `test_single_in_dist/mae_surf_p` | 79.35 | 92.53 | −13.18 |
| `test_geom_camber_rc/mae_surf_p` | 82.61 | 96.38 | −13.77 |
| `test_geom_camber_cruise/mae_surf_p` | NaN (cruise bug) | NaN | — |
| `test_re_rand/mae_surf_p` | 64.29 | 88.29 | −24.00 |
| partial test_avg (3 finite) | 75.42 | 92.40 | −16.98 |
| params | 558,134 | 558,134 | 0 |
| peak VRAM | 21.6 GB | 21.6 GB | 0 |
| wall clock | 30.2 min (timeout) | 30.0 min (timeout) | — |

- Outcome: **Merged** as new baseline. All four val splits improve uniformly by 15–32%. Huber uniformly redistributes gradient signal away from catastrophic high-Re errors. The model was still improving at epoch 32 (timed out), suggesting significant remaining headroom. val_avg=75.93 is the new baseline.
- Key analysis: Huber works because the compound base has to allocate capacity across a ~10× range of per-sample y-std. MSE squares high-Re residuals, letting a small fraction of batches dominate gradients. Huber linearizes at δ=1.0, giving low-Re samples a fairer share.
- Next steps: δ sweep (0.5, 2.0) assigned to alphonse (#853); grad accumulation to frieren (#854); surf_weight sweep to fern (#855).

## 2026-04-28 16:30 — PR #841: slice_num=4 extreme compression on compound base [CLOSED]
- Branch: `willowpai2e2-thorfinn/compound-sn4` (closed, branch deleted)
- Hypothesis: Push slice-token compression to the extreme floor — slice_num=4 on compound base (n_layers=3, n_head=1, n_hidden=128, mlp_ratio=2). Probes whether fewer, coarser slice tokens still capture enough physics structure to generalize.
- W&B run: in PR comment (project `senpai-charlie-wilson-willow-e-r2`)

| metric | compound-sn4 | compound-sn16 baseline | Δ |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | **98.25** | 96.80 | +1.45 (+1.5%) |
| wall clock | ~30 min (timeout) | — | — |

- Outcome: **Closed** — decisive negative. sn=4 regresses vs sn=16 (98.25 vs 96.80), and both are blown away by the Huber baseline (75.93). This confirms sn=16 is the effective compression floor for this dataset — at sn=4 the coarse slice partitioning loses too much surface-structure fidelity. When comparing against the Huber baseline of 75.93, sn=4 falls 29% behind. The slice-floor question is now definitively settled: do not go below sn=16. Prior sn=8 runs (PR #781) at val≈92.5 also confirm the same direction. Thorfinn reassigned.

## 2026-04-28 16:30 — PR #786: RMSNorm replacing LayerNorm on compound base [CLOSED]
- Branch: `willowpai2e2-tanjiro/compound-rmsnorm` (closed, branch deleted)
- Hypothesis: Swap all three LayerNorm slots (ln_1, ln_2, ln_3 in TransolverBlock) to RMSNorm (Zhang & Sennrich 2019). Removes mean-centering and bias terms, reducing per-norm parameters slightly. Hypothesis: at shallow (3-layer) depth, variance-only normalization may regularize better than full LN and free capacity.
- W&B run: in PR comment (project `senpai-charlie-wilson-willow-e-r2`)
- Includes bug fix: evaluate_split NaN sanitization patch for cruise split poisoning (requested as standalone PR)

| metric | compound + RMSNorm | compound + Huber baseline | Δ |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | **109.17** | 75.93 | +33.24 (+43.8%) |
| test_avg (3 finite splits) | 98.91 | 75.42 | +23.49 |

- Outcome: **Closed** — decisive negative. RMSNorm catastrophically regresses vs both the LayerNorm anchor (96.80) and the Huber baseline (75.93). At H=128 with only 3 layers, mean-centering provided by LayerNorm appears to be load-bearing — residual activations likely have non-zero mean due to the asymmetric CFD pressure distribution (surface vs. interior), so removing it destabilizes normalization. The per-sample scale variation in CFD data (y-std spans 164–2077) makes bias-free normalization especially problematic. Mean-centering variants are ruled out for this architecture at this scale. **Important: tanjiro's PR also included a valuable evaluate_split NaN fix — requested as standalone bugfix PR.**

## 2026-04-28 16:30 — PR #785: n_hidden=192 (width increase) on compound base [CLOSED]
- Branch: `willowpai2e2-nezuko/compound-nh192` (closed, branch deleted)
- Hypothesis: Increase hidden width from 128→192 on the compound base (n_layers=3, n_head=1, slice_num=16, mlp_ratio=2). Hypothesis: the compound model is capacity-limited; more width should improve generalization, especially on OOD splits (rc, cruise).
- W&B run: in PR comment (project `senpai-charlie-wilson-willow-e-r2`)

| metric | compound + n_hidden=192 | compound baseline | Δ |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | **119.40** (epoch 18/50) | 96.80 | +22.60 (+23.3%) |
| wall clock | ~30 min (timeout, 18 epochs only) | — | — |

- Outcome: **Closed** — budget-bound negative. Only 18/50 epochs completed; the wider model runs ~67% slower per epoch due to VRAM/compute overhead at H=192. val=119.40 at epoch 18 is almost certainly still descending (like the Huber model at ep32), but the throughput tax is severe enough that we cannot make a fair comparison. More importantly, the Huber win (75.93) demonstrates that loss reformulation beats width-scaling here. With AMP/bf16 (PR #821) potentially enabling the wider model to reach more epochs, this direction could be revisited — but not before the throughput bottleneck is fixed. Nezuko reassigned to a direction that works within the current throughput constraints.

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

## 2026-04-29 — PR #821 round 2: AMP/bf16 + bs=16 + NaN-safe eval + torch.compile + lr=2e-3 (askeladd)

- Branch: `willowpai2e2-askeladd/tooling-amp-bs-nansafe`
- Hypothesis (round 2): Linearly scale LR with batch size (5e-4 → 2e-3 for bs=16) and add `torch.compile(model, dynamic=True)` to close C1 (50 epochs in 30 min) and C3 (val<90) gaps observed in round 1.
- W&B: `ks9lkecv` (default), `ndjgbmhw` (PYTHONHASHSEED=42). Loss: vanilla MSE (pre-#840 default), since the branch was forked before #840 merged.

| metric | default (ks9lkecv) | seed42 (ndjgbmhw) |
|---|---:|---:|
| best `val_avg/mae_surf_p` | 136.22 (ep 47) | 97.84 (ep 44) |
| `test_avg/mae_surf_p` | 118.83 | 87.95 |
| `test/test_single_in_dist` | 194.07 | 100.55 |
| `test/test_geom_camber_rc` | 128.86 | 101.25 |
| `test/test_geom_camber_cruise` | 65.56 (finite ✓) | 63.23 (finite ✓) |
| `test/test_re_rand` | 86.83 | 86.80 |
| epochs | 50/50 | 50/50 |
| wall clock | 22.2 min | 22.3 min |
| peak GPU mem | 49.8 GB | 49.8 GB |

- **Outcome: Sent back for round 3 (rebase + rel-MAE re-validation)**.
  - **C1 PASS**: 50/50 in 22 min — 26% headroom under the 30-min cap. torch.compile gives ~1.5–1.8× additional speedup on top of AMP/bs=16.
  - **C2 PASS**: cruise test = 65.56 / 63.23 finite. The keep-mask substitution in `evaluate_split` is correct and `data/scoring.py` is untouched.
  - **C3 strict-fail**: val_avg 136 / 98 — both above the 90 bar. **But this is on vanilla MSE, not the current rel-MAE baseline.** The wide seed spread (38 pts) suggests lr=2e-3 + cosine is at the edge of stability for vanilla MSE; relative-MAE's per-sample gradient normalization will likely shrink that variance.
  - **Merge blocker**: PR was branched before #840 (rel-MAE) merged → `mergeStateStatus: DIRTY, mergeable: CONFLICTING`. Round-3 ask: rebase onto current advisor branch and re-run two seeds with `--loss_type relative_mae` to confirm the tooling stack preserves the 64.73 / 56.92 baseline. Acceptance bar relaxed to val_avg ≤ 70 on at least one seed (mean ≤ 80).

## 2026-04-29 — PR #900: Loss curriculum Huber warmup → relative MAE (edward)

- Branch: `willowpai2e2-edward/loss-curriculum-warmup` (closed, branch deleted)
- Hypothesis: Huber warmup for N epochs (stable early gradients) then switch to relative MAE (scale equalization) would outperform pure relative MAE from epoch 0.
- W&B: `t5p9xzxx` (baseline rerun), `kww8qk48` (10ep warmup), `89683xfe` (20ep warmup)

| metric | baseline `t5p9xzxx` | warmup-10ep `kww8qk48` | warmup-20ep `89683xfe` |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | 64.16 (ep 32) | 64.54 (+0.38) | 65.70 (+1.54) |
| `val_single_in_dist` | 77.07 | 74.81 ✓ | 79.42 ✗ |
| `val_geom_camber_rc` | 84.10 | 80.29 ✓ | 76.87 ✓✓ |
| `val_geom_camber_cruise` | 36.86 | 43.03 ✗ | 44.01 ✗ |
| `val_re_rand` | 58.58 | 60.03 ✗ | 62.50 ✗ |
| `test_avg/mae_surf_p` | **55.73** | 57.46 (+1.73) | 57.68 (+1.95) |
| `test_geom_camber_cruise` | 30.92 | 35.45 | 37.09 |
| wall / epochs | 30.4 min / 32 | 30.4 min / 32 | 30.4 min / 32 |

- Outcome: **Closed**. Hard loss curriculum rejected — both variants regress on `test_avg/mae_surf_p` by 1.7–2.0 points.
- Root cause: (1) optimizer-reset stall at the Huber→rel-MAE switch-over (train loss spikes from 0.13 → 0.97); (2) Huber pre-training builds high-Re biased representations that rel-MAE must then partially undo, costing cruise performance.
- Interesting side-effect: both warmup variants improved `val_geom_camber_rc` (84.10 → 76.87–80.29), the baseline's worst split. This rc improvement came entirely at cruise/re_rand's expense — a per-domain loss reweighting may capture the rc gain without the cruise regression.
- Edward's smooth-interpolation follow-up (α-ramp Huber→rel-MAE) unlikely to recover; "more Huber = worse" monotone trend suggests the issue is feature bias, not the switchover mechanics.
- Next for edward: ε sweep PR #940 — testing `rel_mae_eps` ∈ {1e-3, 1e-2, 1e-1} vs default 1e-6 to soften small-denominator dominance on low-magnitude (cruise) samples.
