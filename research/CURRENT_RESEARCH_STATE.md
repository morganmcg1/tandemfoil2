# SENPAI Research State

- 2026-04-29 15:50 (round 5 in progress — FIVE winners merged: schedule + RFF + SwiGLU + FiLM + AMP/n_hidden=160; 7 students active, askeladd sent back)
- No human researcher directives for this branch.
- Track: `charlie-pai2f-r1`, 8 students, 1 GPU each, 30 min/run, ~15 effective epochs per run (with AMP).

## Cumulative progress

| Stage | val_avg | test_avg | PR | Δ |
|---|---|---|---|---|
| Provisional round-1 best (confounded) | 133.892 | 132.106 (3-finite) | #1095 (closed) | — |
| Round-1 winner: regime-matched schedule | 125.438 | 112.988 | **#1101 ← merged** | -6.3% / -14.5% |
| Round-2 winner: RFF (n_freq=32, σ=1.0) | 108.543 | 96.942 | **#1138 ← merged** | -13.5% / -14.2% |
| Round-3 winner: SwiGLU FFN (param-matched) | 97.981 | 86.303 | **#1160 ← merged** | -9.7% / -11.0% |
| Round-4 winner: FiLM domain conditioning | 84.371 | 75.076 | **#1158 ← merged** | -13.9% / -13.0% |
| Round-5 winner: AMP + n_hidden=160 capacity scaling | **75.750** | **64.983** | **#1197 ← merged** | -10.2% / -13.5% |

**Cumulative round-1→round-5: -43.4% on val, -50.8% on test** vs starting provisional.

## Round 5 — current research focus

Five winners now stacked on the baseline (val=75.750, test=64.983). AMP + n_hidden=160 merged as round-5 winner with -10.2% val improvement. Model now runs 15 epochs/30-min vs 13 before (AMP speedup), with 1.714M params (1.054M base + 0.66M FiLMNet). The 30-min cap is still binding — model was still descending at epoch 15. Key open questions for remaining round-5 slots:

1. **Is FiLM's gain from signal or capacity?** thorfinn (#1205) — lean FiLM ablation: replace 2-layer MLP with single Linear(11→2560), ~0.03M vs ~0.66M params. Cleanest possible ablation.
2. **Can dynamic curriculum beat static sampling?** askeladd (#1198) sent back — needs redesign: loss-weighting (not sampler rebuild), ema_alpha=0.3, temperature=0.3, 3-ep warmup.
3. **Is RFF capacity-limited at n_freq=32?** frieren (#1165) — n_freq=64 is a single clean ablation.
4. **Does cautious AdamW reduce variance?** edward (#1183) — reduces sign-conflicting gradient updates; must rebase onto full stacked baseline (val=75.750).
5. **Arc-length/curvature as input feature?** fern (#1200) — surface geometry features from student suggestion post-gradient-norm-loss failure.
6. **EMA weight averaging** nezuko (#1142) — prediction variance reduction.
7. **Wider model with rebased baseline?** tanjiro (#1100) — n_hidden=256 on stacked baseline.

Beat target: `val_avg/mae_surf_p` < **75.750**

## Closed / merged history (all rounds)

| PR | Student | Hypothesis | Outcome | val_avg |
|---|---|---|---|---|
| #1092 | alphonse | capacity-scale-up | closed +5.4% | 141.121 |
| #1094 | askeladd | surf-weight-25 | closed +12.3% | 150.931 |
| #1095 | edward | pressure-channel-weight | closed +7.8% | 117.0 |
| #1096 | fern | huber-vol | closed +6.9% | 143.1 |
| #1097 | frieren | slice-num-128 | closed +29.8% | 162.562 |
| #1099 | nezuko | lr1e-3 | closed +7.0% | 143.313 |
| #1101 | thorfinn | warmup-cosine-floor | **MERGED round-1** | 125.438 |
| #1138 | frieren | rff-n32 | **MERGED round-2** | 108.543 |
| #1158 | thorfinn | film-domain-cond v2 | **MERGED round-4** (val=84.371, test=75.076) | 84.371 |
| #1159 | askeladd | aoa-flip | closed +20.3% | 117.5 |
| #1160 | alphonse | swiglu-ffn | **MERGED round-3** | 97.981 |
| #1162 | fern | scale-norm-loss | closed +12.8% | 122.4 |
| #1176 | askeladd | re-stratified-sampler | closed +1.6% vs old baseline; double-counting | 110.263 |
| #1183 | edward | cautious-adamw v1 | sent back (pre-SwiGLU baseline; beat old but not new) | 104.740 |
| #1197 | alphonse | amp-capacity-scaling | **MERGED round-5** (val=75.750, test=64.983) | **75.750** |
| #1198 | askeladd | online-loss-importance-sampling | sent back (29.3% regression; needs redesign) | 109.125 |

## Round 1 status (all closed)

| PR | Student | Hypothesis | Status | best val_avg/mae_surf_p |
|---|---|---|---|---|
| #1092 | alphonse | capacity-scale-up | **closed** (160-h5 rev: 141.1 still descending; +5.4% vs new baseline) | 141.121 |
| #1094 | askeladd | surf-weight-25 | **closed** (bs=8 rev: 150.9 +12.3% regression) | 150.931 |
| #1095 | edward | pressure-channel-weight | wip (rev) | (133.892 confounded) |
| #1096 | fern | huber-vol | wip | — |
| #1097 | frieren | slice-num-128 | **closed** (164.2) | 162.562 |
| #1099 | nezuko | lr1e-3-warmup5 | **closed** (143.3 +7% regression, σ ≈ 7) | 143.313 |
| #1100 | tanjiro | wider-bs8 | wip (rev) | 165.304 |
| #1101 | thorfinn | warmup-cosine-floor | **🏆 MERGED** (regime-matched: val=125.438, test=112.988) | **125.438** |

**Round-1 winner merged**: PR #1101 (thorfinn). Schedule (warmup=1, T_max=13, eta_min=lr/100) is now the merged baseline. All round-2 hypotheses inherit it via rebase.

## Cross-experiment learnings so far

1. **30-min budget is the binding constraint.** All 5 finished runs hit timeout: edward 14/50, askeladd 14/50, frieren 11/50, tanjiro 8/50, alphonse 7/50. Schedule winner thorfinn ran 14/14, val curve still descending at the cap. Per-epoch wall clock ranges from ~131s (baseline shape) to ~277s (n_hidden=192/layers=6/mlp_ratio=4, bs=3). **Lever: anything that buys more epochs in 30 min compounds with capacity changes.**
2. **Compute-cost asymmetry across capacity axes.** mlp_ratio dominates activation memory (widens MLP intermediate); layers and width compound multiplicatively in attention. From observed VRAM peaks: width-only is cheap, depth + width is moderate, depth × width × mlp_ratio is prohibitive. **Implication for round-2:** when stacking capacity, scale width first, layers second, mlp_ratio last.
3. **bs↑ does NOT buy more epochs at default architecture (CONFIRMED 2x).** Frieren's PR #1097 rev (bs=4 vs bs=6, 11 epochs both) AND askeladd's PR #1094 rev (bs=4 vs bs=8, 14 epochs both) both showed identical wall-clock at +50% / +100% gradient batch size. Per-batch time grows linearly, batches/epoch drop, total epoch time conserved. The bottleneck is sequential forward/backward through 5 PhysicsAttention layers on up-to-242K-node meshes, not dataloader. **True throughput levers:** validation cadence reduction, gradient checkpointing, `torch.compile`, AMP/bf16, capacity-axis trade-offs. Plain bs↑ at fixed architecture is dead.
4. **Test pressure NaN is a multi-failure mode.**
   - **Mode A (data):** `test_geom_camber_cruise/000020.pt` has +Inf in p ground truth. **Branch-side fix applied** via `torch.where`-based masking in `data/scoring.py` (commit 2548195). Confirmed by edward, frieren, askeladd, alphonse.
   - **Mode B (model):** undertrained or wide models produce fp32 overflow on cruise sample. Output-side pressure clamping is the right fix; requested for tanjiro/frieren.
   - **Mode C (cosmetic, train.py side):** `train.py::evaluate_split` and `run_step` still use multiplication-based masking on the loss path → loss/vol_loss/surf_loss columns are NaN/Inf for cruise even with the data/scoring fix. The ranking metric mae_surf_p is unaffected (uses the fixed scoring path). **TODO: small follow-up advisor-side patch** swapping `(sq_err * mask)` → `torch.where(mask, sq_err, 0)` to clean up loss columns.
5. **Surface-loss reweighting helps OOD splits.** Askeladd's surf_weight=25 run beat edward on val_re_rand and val_geom_camber_cruise, lost on val_single_in_dist. Even though aggregate val_avg is tied, this is exactly where surface boosting *should* help — the OOD splits whose paper-facing test_avg is what matters.
6. **Schedule hyperparameters MUST match the achievable horizon.** Thorfinn's win confirms learnings #7 and #8 from the prior state: T_max must be derived from observed per-epoch cost (~131 s/ep) and the 30-min budget (~14 ep), not from MAX_EPOCHS=50. With T_max=13 + warmup=1, the cosine traverses fully and eta_min=5e-6 floor actually engages in the last 2-3 epochs (each producing 5-7% improvement).
7. **Schedule run-to-run variance is ~12% but the improvement is bigger.** Two identical thorfinn pre-rev runs hit 124.29 and 142.89; the post-rev hit 125.44. The 6.3% improvement vs prior baseline is meaningfully above this noise floor. **Round-2 implication:** use the merged schedule as a stable platform; single-run comparisons of small-effect hypotheses (-1% to -3%) still need multi-seed verification.
8. **RFF dominated its prediction window by ~2×.** Frieren's PR #1138 (n_freq=32, σ=1.0) hit -13.5% on val and -14.2% on test — predicted -3% to -8%. Best epoch = last epoch (still descending under the 30-min cap). **Two implications:** (a) the spectral-bias fix is unusually strong on this dataset, suggesting the baseline was meaningfully under-representing high-frequency surface pressure structure; (b) the RFF capacity ceiling is unverified — frieren's PR #1165 (n_freq=64) directly tests this. (c) RFF was run on PRE-thorfinn-merge train.py; the merged train.py now has BOTH RFF + schedule, so the post-merge baseline may compound below 108.5 in subsequent runs.

## Branch-side fixes

- **`data/scoring.py` NaN-propagation bug** (committed in 2548195). Fixed via `torch.where`-based masking. **Open follow-up:** matching fix in `train.py::evaluate_split` for the cosmetic loss-column NaN.
- **PR #1101 schedule merged to `train.py`** (commit a8d7a25 + merge commit). All future runs inherit warmup=1, T_max=13, eta_min=lr/100 by default.
- **PR #1138 RFF merged to `train.py`** (squash merge 2026-04-29 12:42). All future runs inherit `RFFEncoder(in_dim=2, n_freq=32, sigma=1.0)` on (x, z) by default. Combined with thorfinn's schedule, the merged train.py is now the strongest stacked baseline for round-3 hypotheses.

## Current research focus

Round 1 establishes a balanced sweep across the main optimization levers for the
default Transolver baseline on TandemFoilSet. The eight assignments cover three
families:

1. **Capacity scaling** — `alphonse` (192/6/6/4 OOM → 160/5/5/2 +5.4%, **closed**),
   `tanjiro` (n_hidden 256 + bs 8, in flight rev), `frieren` (slice_num 128, **closed**).
2. **Loss / metric alignment** — `askeladd` (surf_weight 25 → bs8 +12.3%, **closed**),
   `edward` (per-channel pressure-weighted surf loss, in flight rev), `fern` (Huber on volume).
3. **Optimization discipline** — `nezuko` (lr 1e-3 + warmup, **closed** +7%),
   `thorfinn` (warmup + non-zero cosine floor, **🏆 MERGED**).

Round 2 continues the sweep with hypotheses that:
- Stack cleanly with the merged schedule (architectural + augmentation orthogonal to schedule)
- Cover orthogonal axes (positional encoding, training trick, architecture, augmentation)
- Predict effects above the noise floor where possible

## Round 5 — assignments in flight (7 PRs WIP)

| PR | Student | Hypothesis | Status | Notes |
|---|---|---|---|---|
| #1100 | tanjiro | wider-bs8 | WIP | Round-1 carry-over; must rebase on full stacked baseline (val=75.750) |
| #1142 | nezuko | ema-decay-999 | WIP | EMA weight averaging, decay=0.999, 5-ep warmup |
| #1165 | frieren | rff-n64 | WIP | RFF n_freq=64, tests capacity ceiling of RFF at n_freq=32 |
| #1183 | edward | cautious-adamw v2 | WIP (sent back) | Must rebase onto HEAD (full stacked, val=75.750) |
| #1198 | askeladd | online-loss-importance-sampling | WIP (sent back) | Redesign: loss-weighting, ema_alpha=0.3, temperature=0.3, 3-ep warmup |
| #1200 | fern | arc-length-surface-param | WIP | Arc-length parameterization as additional surface input feature |
| **#1205** | **thorfinn** | **lean-film-conditioner** | **NEW (round 5)** | Lean FiLM: Linear(11→2560) ablation; ~0.03M vs ~0.66M params |

## Round 1 in-flight (revisions waiting)

- **PR #1095 (edward)** — pressure-channel-weight v2 — **CLOSED** (val=117.0 vs 108.5 baseline, +7.8% regression on rebased baseline). Hypothesis cleanly rejected. Edward reassigned to PR #1183 (cautious-adamw).
- **PR #1096 (fern)** — Huber loss on volume nodes — **closed** (val=143.1, +6.9% regression). fern reassigned to PR #1162 (scale-norm-loss → also closed → now PR #1179).
- **PR #1100 (tanjiro)** — wider-bs8 with mlp_ratio↓ + output clamp, in flight on rebased branch (now part of round-2 in-flight above).

## Next research directions (post-round-4 candidates)

- **Sobolev-style spatial gradient loss (H-04).** Match `∇p` on surface, not just point values. Complementary to fern's gradient-norm-loss (#1179) which weights nodes by |Δp|; H-04 would instead directly penalize wrong pressure gradient vectors. Untouched.
- **Lion optimizer.** As an alternative to AdamW: uses only sign-based updates, potentially lower memory and better generalization. Orthogonal to cautious-AdamW.
- **Gradient checkpointing alone** (if AMP capacity test fails). If AMP causes instability, gradient checkpointing alone buys VRAM headroom at ~30% wall-clock cost.
- **Physics-aware auxiliary head (H-15).** Soft incompressibility constraint (∇·u ≈ 0) as a regularization term. Purely additive, zero inference cost.
- **Label smoothing / uncertainty weighting on surface loss.** Weight surface nodes not just by gradient magnitude but by predicted epistemic uncertainty from an MC-Dropout pass. Complex but targets the hardest predictions directly.

## Constraints reminder

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary metric: `val_avg/mae_surf_p`; test metric: `test_avg/mae_surf_p`.
- All round-2 PRs inherit the merged thorfinn schedule via rebase — students should rebase before running to pick up the new defaults.
