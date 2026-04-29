# SENPAI Research State

- 2026-04-29 (round 2 in progress — TWO winners merged: schedule + RFF; PRs #1159, #1162 closed; PR #1160 sent back for rebase+rerun; fern assigned PR #1179 gradient-norm-loss; askeladd PR #1176 label fixed — all 8 students active)
- No human researcher directives yet for this branch.
- Track: `charlie-pai2f-r1`, 8 students, 1 GPU each, 30 min/run, max 50 epochs effective (~14 actually achievable per run).

## Cumulative progress

| Stage | val_avg | test_avg | PR | Δ |
|---|---|---|---|---|
| Provisional round-1 best (confounded) | 133.892 | 132.106 (3-finite) | #1095 (sent back) | — |
| Round-1 winner: regime-matched schedule | 125.438 | 112.988 | **#1101 ← merged** | -6.3% / -14.5% |
| Round-2 winner: RFF (n_freq=32, σ=1.0) | **108.543** | **96.942** | **#1138 ← merged** | -13.5% / -14.2% |

**Cumulative round-1→round-2: -19.0% on val, -26.6% on test** vs starting provisional.

## Round 1 status (closing)

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

## Round 2 — assignments in flight (6 PRs)

- **PR #1142 (nezuko, ema-decay-999)** — H-06 EMA weight averaging at `decay=0.999` with 5-epoch warmup. Direct intervention against the σ ≈ 7 run-to-run variance. Zero throughput cost, low effect (-1% to -3%) but stacks for free with every future winner.
- **PR #1158 (thorfinn, film-domain-cond)** — H-10 FiLM domain conditioning over global per-sample features (Re, AoA, NACA, gap, stagger). Targets the 51% per-split val spread. ~0.05M extra params, near-zero throughput cost, identity-init for safe start. Expected -2% to -5%.
- **PR #1159 (askeladd, aoa-flip-aug)** — H-12 AoA sign-flip augmentation. **CLOSED** (+20.3% regression). Root cause: NACA camber M sign not extended when flipping AoA → geometry/label contradiction for all cambered foils. askeladd reassigned to new experiment.
- **PR #1160 (alphonse, swiglu-ffn)** — H-11 SwiGLU FFN replacing GELU MLP in TransolverBlock, param-matched. **SENT BACK** — ran without RFF, beat old baseline by -12.4% but is +1.2% worse than current RFF baseline. Must rebase onto current branch and rerun. SwiGLU+RFF untested and potentially powerful.
- **PR #1162 (fern, scale-norm-loss)** — H-03 Per-sample scale-normalized loss. **CLOSED** — val=122.470 vs baseline 108.543 (+12.8% worse). Loss redistribution helped cruise/re_rand but hurt the higher-error splits (single_in_dist, geom_camber_rc) that dominate the average. Per-sample std is a poor proxy for difficulty. fern **reassigned to PR #1179 (gradient-norm-loss)**.
- **PR #1179 (fern, gradient-norm-loss)** — **NEW (assigned 2026-04-29 14:30)** — Spatial gradient-magnitude weighted surface loss. Weights each surface node's loss by |Δp_i| (discrete first-order pressure gradient along node sequence). Directly targets leading-edge/suction-peak/TE-wake regions that dominate mae_surf_p. Addresses the failure mode of #1162 (global std ≠ local difficulty). Expected -2% to -5%.
- **PR #1165 (frieren, rff-64)** — RFF n_freq sweep follow-up to merged #1138. Tests if RFF is capacity-limited at n_freq=32. Single-variable ablation: same RFF, same sigma, just doubled frequency components. Best-epoch=last on the merged baseline → model still hungry; +0.03M params, zero throughput cost. Expected -1% to -3% if capacity-limited; flat otherwise.
- **PR #1176 (askeladd, re-stratified-sampler)** — H-13 Re-stratified sampling. Replace domain-balanced sampler with one that multiplies domain weights by `log(1 + per_sample_y_std_p)` (normalized to unit mean). Upweights hard high-Re samples within each domain, focusing the limited 14-epoch budget on gradient-rich samples. Zero throughput cost. Orthogonal to RFF and schedule. Expected -2% to -5%.

## Round 1 in-flight (revisions waiting)

- **PR #1095 (edward)** — pressure-channel-weight, corrected formula (`/ ch_w.mean()`). Round-1 cleanup.
- **PR #1096 (fern)** — Huber loss on volume nodes — **closed** (val=143.1, +6.9% regression). fern reassigned to PR #1162 (scale-norm-loss).
- **PR #1100 (tanjiro)** — wider-bs8 with mlp_ratio↓ + output clamp, in flight.

## Next research directions (post-round-2 candidates)

- **Stack winners.** Whichever of FiLM, AoA-flip, SwiGLU, RFF, EMA win get
  combined into a single recipe and re-tested. Schedule (already merged) is
  the platform.
- **AMP / gradient checkpointing for capacity stacking.** alphonse's 160/5/5/2
  + thorfinn's schedule is a clean stacking hypothesis but needs throughput
  gain to fit ≥18 epochs. AMP halves activation memory; gradient checkpointing
  trades VRAM for ~30% wall-clock cost.
- **Sobolev-style loss (gradient matching, H-04).** Per-sample scale-aware
  losses (divide errors by sample y_std), pressure-only auxiliary head with a
  stronger weight. Untouched loss family.
- **Optimizer swap (H-08 Cautious AdamW, Lion).** As alternatives to AdamW,
  especially if capacity-scaling wins because larger models often respond
  better to alternative optimizers.
- **Re-stratified sampling (H-13).** Per-sample weighting by `log(per_sample_y_std)`
  to upweight high-Re samples (harder distribution).
- **Physics-aware (H-14, H-15).** Soft incompressibility constraint
  (∇·u = 0); multi-task auxiliary head for surface Cp.
- **Sobolev-style spatial gradient loss (H-04).** Match `∇p` and `∇u` on
  surface, not just point values — directly penalizes wrong pressure
  gradients.

## Constraints reminder

- No new packages outside `pyproject.toml` (add in same PR if needed).
- `data/` is read-only in normal experiment PRs.
- Don't override `SENPAI_TIMEOUT_MINUTES` or `SENPAI_MAX_EPOCHS`.
- Primary metric: `val_avg/mae_surf_p`; test metric: `test_avg/mae_surf_p`.
- All round-2 PRs inherit the merged thorfinn schedule via rebase — students should rebase before running to pick up the new defaults.
