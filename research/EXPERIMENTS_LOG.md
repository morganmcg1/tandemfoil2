# SENPAI Research Results

## 2026-04-29 06:30 — PR #1048: Architecture width n_hidden=128 → 192 (relative_mae + warmup baseline) [WINNER — MERGED]
- Branch: `willowpai2e2-edward/nh192-relmae-warmup`
- Hypothesis: The compound base width (n_hidden=128) was chosen under fp32/bs=4 throughput constraints in the prior round. Under AMP/bs=16/compile (PR #821) peak VRAM is 49.8 GB on a 96 GB GPU — there is now headroom for a wider model. Increased capacity should improve final-epoch performance because PR #971 still hit its best at epoch 49/50, suggesting compute-budget-limited descent.
- W&B runs (group `nh192-relmae-warmup`): `ovkjhjyo` (default seed), `v9ipztd9` (seed42)

| | n_hidden=128 (PR #971 baseline) | n_hidden=192 | Δ |
|---|---:|---:|---:|
| Default-seed val | 54.70 (`1xfcb5h5`) | **50.61** (`ovkjhjyo`) | **−4.09 (−7.5%)** |
| seed42 val | 67.28 (`9a9di1dz`) | **56.24** (`v9ipztd9`) | **−11.04 (−16.4%)** |
| 2-seed mean val | 60.99 | **53.42** | **−7.57 (−12.4%)** |
| 2-seed spread val | 12.58 | **5.63** | **−6.95** (corridor halved) |
| Best-seed test_avg | 48.15 | **44.15** | **−4.00 (−8.3%)** |
| Wall (min / 50 ep) | 22.4 | 30.6 | +8.2 (within 30 min cap) |
| Peak VRAM (GB) | 49.8 | 73.6 | +23.8 (well within 96 GB) |
| Best epoch | 49 / 50 | 49 / 50 (default) ; 48 / 50 (seed42) | still descending |
| Param count | 0.56M | 1.24M | ~2.2× |

Per-split test (default seed, ep 49):

| Split | n_hidden=128 | n_hidden=192 | Δ |
|---|---:|---:|---:|
| `test_single_in_dist`     | 67.22 | **58.21** | −9.01 (−13.4%) |
| `test_geom_camber_rc`     | 60.38 | **57.94** | −2.44 (−4.0%)  |
| `test_geom_camber_cruise` | 23.79 | **21.65** | −2.14 (−9.0%)  |
| `test_re_rand`            | 41.20 | **38.79** | −2.41 (−5.9%)  |
| **test_avg/mae_surf_p**   | **48.15** | **44.15** | **−4.00 (−8.3%)** |

- Outcome: **MERGED.** All four test splits improve. Largest gain on `test_single_in_dist` — the high-variance raceCar-single track. Reference target gap (PR #32 prior round, test=40.93): now **3.22 pts** (was 7.22).
- Two further qualitative observations from edward:
  1. **Best epoch is still 48–49/50 even after the width jump.** The 1.24M-param model has not saturated by 50 epochs of cosine decay → the schedule, not the capacity, is the next bottleneck.
  2. **Throughput cost is exactly as predicted** (36–37 s/epoch vs ~27 s for n_hidden=128 → 30.5–30.6 min total). The harness allowed all 50 epochs to finish; if the timeout were tighter the last epoch would have been lost.
- Variance result (also load-bearing): the seed-corridor halved (12.58 → 5.63), confirming the wider model has a friendlier optimization landscape — fewer bad basins for unlucky seeds. This is a result *in addition to* the headline metric improvement.
- Verification: W&B metrics match PR claims to 4 decimal places on both seeds; both runs completed all 50 epochs; config confirmed `model_config.n_hidden=192`, all other defaults inherited from PR #971.

## 2026-04-29 04:00 — PR #1008: 3rd-seed verification + warmup-length sweep [CLOSED — informational, no code change]
- Branch: `willowpai2e2-askeladd/seed3-warmup-sweep`
- Hypothesis: (a) 3rd seed at warmup=5 (PYTHONHASHSEED=7) confirms the 12.58 val-spread is representative; (b) warmup_epochs ∈ {3, 5, 10} bracket at default seed identifies whether 5 is optimal.
- W&B runs (group `lr-warmup-3rd-seed-sweep`): `qznce19f` (warmup5-seed7), `re4i40ft` (warmup3-default), `ciabcxxu` (warmup10-default)

| run | best epoch | val_avg | test_avg |
|---|---:|---:|---:|
| warmup=5, seed=7 (`qznce19f`) | 50 | 57.52 | 50.19 |
| warmup=3, default (`re4i40ft`) | 48 | 55.80 | 48.75 |
| warmup=5, default (`1xfcb5h5` baseline) | 49 | 54.70 | 48.15 |
| warmup=10, default (`ciabcxxu`) | 43 | 64.77 | 55.86 |

3-seed corridor at warmup=5: spread (max−min) = 12.58 val, 9.65 test. Mean = 59.83 val, 52.05 test (was 60.99/52.98 with 2 seeds — barely moved).

- Outcome: **CLOSED.** No code changes to merge (warmup_epochs flag was already auto-bound by `simple_parsing.parse(Config)`). Two findings promoted to "settled" status in BASELINE.md:
  1. **3-seed corridor confirmed at val-spread 12.58 / test-spread 9.65.** Single-seed screening convention adopted: future small-delta PRs may use 1 seed for screening; final candidate before merge requires 2 seeds; claims of < 6 val pt improvement require 2 seeds.
  2. **warmup_epochs=5 is the right default.** warmup=3 (+1.10 val) is within seed noise; warmup=10 (+10.07 val) is decisively worse with best epoch sliding 49 → 43 (under-decay). U-shape with soft minimum at 5.
- Key analysis: warmup=10's `test_single_in_dist` blew up to 89.35 (vs 67.22 at warmup=5) — under-decay shows up most in the in-distribution split, consistent with the model still oscillating around a high-detail minimum.

## 2026-04-29 04:00 — PR #940 round 2: ε=1e-3 + warmup (rebased onto current tooling) [CLOSED — settled negative]
- Branch: `willowpai2e2-edward/rel-mae-eps-sweep`
- Hypothesis: ε=1e-3 (winner from PR #940 round 1, pre-warmup) composes with the new tooling stack (warmup + AMP + bs=16 + compile + lr=2e-3) to push below val=54.70.
- W&B runs (group `compound-relmae-eps-1e3-rebased`): `d0v5ezta` (eps1e3-default), `jpjldfml` (eps1e3-seed42)

| | ε=1e-6 + warmup (current best) | ε=1e-3 + warmup | Δ |
|---|---:|---:|---:|
| Default-seed val | 54.70 (`1xfcb5h5`) | 69.45 (`d0v5ezta`) | **+14.75** |
| seed42 val | 67.28 (`9a9di1dz`) | 61.62 (`jpjldfml`) | −5.67 |
| 2-seed mean val | 60.99 | 65.53 | **+4.54** |
| 2-seed spread val | 12.58 | 7.83 | −4.75 |
| Best-seed test | 48.15 | 53.93 | **+5.78** |

- Outcome: **CLOSED — settled negative.** Best-seed val=61.62 is +6.92 above current best (54.70). 2-seed mean regresses by +4.54. Spread does narrow (12.58 → 7.83) but at the cost of moving both seeds to worse plateaus rather than rescuing the worse seed.
- Key analysis (from edward's diagnosis, confirmed): ε=1e-3 was protecting against early-training instability under fp32/bs=4/no-warmup. Once warmup itself protects against early instability, ε=1e-3 just becomes a regularization term that biases toward absolute MAE, erasing useful signal that the larger, smoother bf16/bs=16 step can otherwise exploit. The `single_in_dist` channel inversion is decisive evidence: this was the channel that benefited *most* from ε=1e-3 in round 1 (round 1 single 77 → 68 val); under warmup it blows up by +33 pts (default seed) — sign reversal.
- Settled fact: **ε ≠ 1e-6 does not compose with warmup**. Future relative-MAE PRs should not re-test this lever unless the warmup default itself changes.

## 2026-04-29 03:30 — PR #940 round 1: Relative MAE ε sweep on PR #840 baseline [SENT BACK — rebase onto new tooling]
- Branch: `willowpai2e2-edward/compound-relmae-eps-sweep`
- Hypothesis: ε=1e-6 in `relative_mae_loss = mean(|pred - target| / (mean(|target|) + ε))` lets near-zero-target samples (cruise) hijack the gradient via huge `1/scale` weights, starving rc and single splits. Increasing ε softens this without abandoning relative weighting, redistributing optimizer budget toward under-weighted splits.
- W&B runs (group `compound-relmae-eps-sweep`): `vevw6ksl` (ε=1e-3), `sl0udyi7` (ε=1e-2), `e272627k` (ε=1e-1)

| variant | best epoch | val_avg | test_avg | best per-split val (rc, single, cruise, re_rand) |
|---|---:|---:|---:|---|
| baseline ε=1e-6 (PR #840 `t5p9xzxx`) | 32 | 64.16 | 55.73 | 84.10, 77.07, 36.86, 58.58 |
| **ε=1e-3** (`vevw6ksl`) | 32 | **58.74** | **52.66** | 75.52, 68.63, 35.03, 55.77 |
| ε=1e-2 (`sl0udyi7`) | 30 | 61.68 | 55.41 | 75.29, 77.05, 38.57, 55.80 |
| ε=1e-1 (`e272627k`) | 31 | 60.52 | 53.17 | 73.32, 74.86, 36.61, 57.29 |

- Outcome: **Sent back for rebase.** Hypothesis validated — ε=1e-3 strictly dominates ε=1e-6 on the same tooling stack (−5.42 val, −3.07 test). The non-monotone curve (ε=1e-3 < ε=1e-1 < ε=1e-2) is interesting and suggests the rc/single-vs-cruise tradeoff only kicks in at ε ≥ 1e-2. **But** runs were branched pre-tooling: fp32/bs=4/lr=5e-4, timed out at epoch 30–32 (still descending). Compared to current PR #971 baseline (val=54.70/test=48.15 with AMP/bs=16/lr=2e-3/compile/warmup), ε=1e-3's val=58.74 is +4.04 — not because ε=1e-3 is wrong but because old tooling never got 50 epochs. Send-back ask: rebase, run ε=1e-3 only at 2 seeds with new defaults (warmup=5, relative_mae default), and flip `Config.rel_mae_eps` default 1e-6 → 1e-3 in the rebased branch. Decision criteria: best-seed val ≤ 54.70 OR test ≤ 48.15 → merge.
- Key analysis: cruise *did not pay the predicted cost*. At ε=1e-3, cruise actually improves slightly (val 36.86 → 35.03, test 30.92 → 29.37), suggesting cruise was being hurt by gradient noise from the 1/scale spike on small-magnitude samples rather than benefiting from over-weighting. ε=1e-3 wins on 6 of 8 per-split metrics.

## 2026-04-29 03:00 — PR #971: LR warmup (5ep linear) + flip loss_type default to relative_mae [WINNER — MERGED]
- Branch: `willowpai2e2-askeladd/lr-warmup-default-relmae`
- Hypothesis: PR #821 round-3 showed a 27-point val spread between seeds (default=82.97 vs seed42=55.90) at lr=2e-3 cosine-only on bs=16. The likely cause is early-epoch overshooting at the linearly-scaled lr — some seeds overshoot into a worse basin at epoch 1 and never recover. A 5-epoch linear LR warmup (0.05 × lr → lr) provides a ramp-up buffer that should narrow the seed variance and improve the typical-seed outcome. Secondary change: flip the `loss_type` default from `"mse"` to `"relative_mae"` for branch reproducibility.
- W&B runs: `1xfcb5h5` (lr-warmup-5ep-default), `9a9di1dz` (lr-warmup-5ep-seed42), both `senpai-charlie-wilson-willow-e-r2`

| metric | round-3 no warmup | round-4 warmup=5ep | Δ |
|---|---:|---:|---:|
| default-seed `val_avg` | 82.97 (`1d8nkjir`) | **54.70** (`1xfcb5h5`) | **−28.27** |
| seed42 `val_avg`       | 55.90 (`66c4gac6`) | 67.28 (`9a9di1dz`)   | +11.38 |
| 2-seed mean            | 69.43 | **60.99** | **−8.44** |
| 2-seed spread          | 27.07 | **12.58** | **−14.49** |
| best test_avg (best-seed) | 49.64 | **48.15** | **−1.49** |

| split (best-seed test) | round-3 seed42 | round-4 default | Δ |
|---|---:|---:|---:|
| `test_single_in_dist`     | 63.94 | 67.22 | +3.28 |
| `test_geom_camber_rc`     | 62.62 | 60.38 | −2.24 |
| `test_geom_camber_cruise` | 26.87 | 23.79 | −3.08 |
| `test_re_rand`            | 45.11 | 41.20 | −3.91 |
| `test_avg`                | 49.64 | **48.15** | **−1.49** |

- Outcome: **MERGED.** Best-seed beats baseline on val (−1.20, −2.1%) and test (−1.49, −3.0%). Mean across seeds improved 8.4 pts and spread narrowed by half (27 → 13). 3 of 4 test splits improve (rc, cruise, re_rand); single regresses ~3 pts. Per-split test wins on the OOD camber and Re-stratified splits are the most encouraging — those are the harder generalization tracks.
- Key analysis: warmup didn't just shift the unlucky seed up — it reshaped the loss-landscape exploration trajectory enough that basin assignment per seed effectively re-randomized. The default seed (round-3 "bad basin", val=82.97) is now the better seed (val=54.70), while seed42 (round-3 "good basin", val=55.90) is now worse (val=67.28). **This means seed-pinned reproduction is no longer reliable** — future PRs must run ≥ 2 seeds and report the spread. The 12.58-point residual variance is small enough to make A/B claims feasible but still material — a 3rd seed is the next priority before letting other PRs benchmark against this number.
- New defaults landed: `loss_type="relative_mae"`, `warmup_epochs=5`. Schedule = `SequentialLR(LinearLR(start_factor=0.05) for 5ep, then CosineAnnealingLR(T_max=45, eta_min=1e-6))`. LR sanity-checked: ramps 4.80e-4 → 2.00e-3 over epochs 0–4, then cosine-decays cleanly.

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

## 2026-04-29 — PR #821 round 3: tooling stack MERGED (askeladd)

- Branch: `willowpai2e2-askeladd/tooling-amp-bs-nansafe` (merged, squash)
- Hypothesis (tooling): AMP/bf16 + bs=16 + lr=2e-3 + torch.compile + NaN-safe eval + compound model_config default, rebased onto PR #840 (rel-MAE). Validated with relative_mae loss.
- W&B: `1d8nkjir` (default seed), `66c4gac6` (PYTHONHASHSEED=42)

| metric | default `1d8nkjir` | seed42 `66c4gac6` | baseline #840 (`t5p9xzxx`) |
|---|---:|---:|---:|
| best `val_avg/mae_surf_p` | 82.97 (ep 42) | **55.90 (ep 50, still descending)** | 64.16 (ep 32, timed out) |
| `test_avg/mae_surf_p` | 72.01 | **49.64** | 55.73 |
| `test_geom_camber_cruise` | 29.51 | 26.87 | 30.92 |
| `test_single_in_dist` | 128.18 | 63.94 | 71.33 |
| `test_geom_camber_rc` | 80.37 | 62.62 | 70.62 |
| `test_re_rand` | 49.98 | 45.11 | 50.04 |
| epochs | 50/50 | 50/50 | 32/50 (timeout) |
| wall (min) | 22.5 | 22.5 | 30.4 |
| peak GPU mem (GB) | 49.8 | 49.8 | 21.4 |

- Outcome: **MERGED — NEW BEST**. Seed42 (55.90/49.64) beats prior baseline (64.16/55.73) by 13%/11%. C1/C2/C3 all pass. Seed variance (82.97 vs 55.90 = 27-pt spread) is real; LR warmup assigned as follow-up to askeladd (#971).
- Post-merge branch now has: AMP/bf16, bs=16, lr=2e-3, torch.compile, NaN-safe eval, compound model_config, relative_mae loss.
- New CLI defaults: `batch_size=16`, `lr=2e-3`, `compile=True`. Note: `loss_type` default still "mse" — will flip in PR #971.
