# SENPAI Research State
- 2026-04-29 19:45
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: FiLM Re conditioning merged (PR #1264, val=74.36). Re conditioning with a shared generator MLP proved effective across all 4 splits (+7.7% improvement). Now exploring FiLM variants, schedule tuning, augmentation, and domain re-weighting.

## Current Baseline

**val_avg/mae_surf_p = 74.36** (PR #1264, charliepai2f2-nezuko, SharedFiLMGenerator post-norm Re conditioning + BF16 AMP + OneCycleLR(max_lr=1.2e-3, pct_start=0.3) + grad_clip=1.0 + DropPath 0→0.1 + surf_weight=25 + wd=1e-4 + NaN guard + WeightedRandomSampler, epoch 18/19)

Per-split breakdown (PR #1264 — current best):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| single_in_dist     | 82.09 | 71.62 |
| geom_camber_rc     | 85.31 | 79.90 |
| geom_camber_cruise | 55.67 | 46.22 |
| re_rand            | 74.38 | 65.89 |
| **avg**            | **74.36** | **65.91** |

Stack: SharedFiLMGenerator(1→128→1280) + BF16 AMP + OneCycleLR(max_lr=1.2e-3, pct_start=0.3) + grad_clip=1.0 + DropPath 0→0.1 + surf_weight=25 + wd=1e-4 + NaN guard + WeightedRandomSampler

Note: Best checkpoint was epoch 18/18 (final) — model converged within budget. OneCycleLR annealed to lr=1.66e-05.

## Key Insights

**1. Systematic Timeout/LR Mismatch (critical, fixed)**
All experiments use budget-aware LR scheduling. With BF16: ~98–100 s/epoch, fitting 19 epochs in 30 min. OneCycleLR is sized for 15 epochs (ONECYCLE_PER_EPOCH_SEC_ESTIMATE=125s) but 19 actual epochs run — the schedule anneals faster than intended.

**2. lr=1e-3 + grad_clip=1.0 is transformative (MERGED PR #1098)**
val_avg/mae_surf_p=100.41 — largest single gain (-17.6%). All subsequent experiments build on this.

**3. Throughput > capacity under 30-min budget**
Any change slowing epoch time >20% is fatal. BF16 AMP gave +27% throughput (135→98s/epoch), enabling 19 epochs vs 14. This is the most impactful systems-level finding.

**4. OneCycleLR beats CosineAnnealingLR (MERGED PR #1211)**
BF16 + OneCycleLR combined: 89.00 → 80.53 (-10.2%). Best checkpoint = final epoch 19, confirming the training curve had not yet converged — more epochs or a better-calibrated schedule would likely help.

**5. surf_weight=25 confirmed optimal; surf_weight=50 regresses (PR #1182)**
Volume context is essential for OOD surface prediction.

**6. DropPath 0→0.1 optimal; 0→0.2 under-fits in 14-19 epoch regime (PR #1156)**

**7. Weight decay wd=1e-4 confirmed optimal; wd=1e-3 over-regularizes (PR #1178)**

**8. Key per-split gap**: raceCar splits (single_in_dist=91.33, geom_camber_rc=91.01) vs cruise (60.42). The ~31 point gap between raceCar and cruise is the dominant improvement opportunity.

## Active Experiments

| PR | Student | Hypothesis | Key Config | Status |
|----|---------|-----------|------------|--------|
| #1090 | frieren | Per-field output heads (Ux, Uy, p separate MLP decoders) | 3 separate heads on last block | Running (stale baseline — assess on absolute val) |
| #1259 | askeladd | OneCycleLR max_lr 1.2e-3→1.5e-3: higher peak | ONECYCLE_MAX_LR=1.5e-3 | Running |
| #1261 | alphonse | Re noise augmentation: corrected per-sample implementation | RE_NOISE_STD=0.05, per-sample scalar | Running |
| #1262 | edward | OneCycleLR 20-epoch budget: lower per-epoch estimate to 94s | ONECYCLE_PER_EPOCH_SEC_ESTIMATE=94.0 | Running |
| #1263 | fern | Domain re-weighting: boost raceCar samples 2x | RACECAR_BOOST=2.0 via heuristic AoA<-0.01 | Running |
| #1265 | tanjiro | OneCycleLR pct_start 0.3→0.2: shorter warmup | ONECYCLE_PCT_START=0.2 | Running |
| #1266 | thorfinn | Geometry augmentation: AoA+NACA per-sample noise | GEOM_AUG_AOA_STD=0.03, GEOM_AUG_NACA_STD=0.05 | Running |

## Merged Winners (Chronological)

1. **PR #1088** (edward, surf_weight 10→25): val_avg/mae_surf_p = 127.67
2. **PR #1091** (nezuko, DropPath 0→0.1 + budget-aware CosineAnnealingLR): **121.89** (-4.5%)
3. **PR #1098** (tanjiro, lr=1e-3 + grad_clip=1.0): **100.41** (-17.6%)
4. **PR #1184** (askeladd, BF16 AMP): **89.00** (-11.4%)
5. **PR #1195** (nezuko, OneCycleLR on BF16): **~interim** (merged; combined with BF16)
6. **PR #1211** (nezuko, BF16 + OneCycleLR combined): **80.53** (-10.2%)
7. **PR #1264** (nezuko, FiLM Re conditioning SharedFiLMGenerator): **74.36** (-7.7%) — **current baseline**

## Closed / Dead Ends

- PR #1087 (askeladd): slice_num=128 — epoch starvation
- PR #1086 (alphonse): n_hidden=192/256 — 204s/epoch, epoch starvation
- PR #1129 (askeladd): per-sample instance-normalized loss — 3x regression (366.76)
- PR #1089 (fern): n_layers=8 — 207s/epoch, epoch starvation
- PR #1161 (fern): n_layers=6 — 158s/epoch, 27% regression
- PR #1102 (thorfinn): mlp_ratio=4 — epoch slowdown, regression
- PR #1156 (nezuko): DropPath 0→0.2 — underfitting in budget regime
- PR #1178 (tanjiro): wd=1e-3 — over-regularization, regression
- PR #1144 (askeladd): BF16 AMP (stale baseline) — validated technique, merged as #1184
- PR #1185 (nezuko): SGDR warm restarts — restart spikes incompatible with budget
- PR #1182 (fern): surf_weight=50 — vol_p degradation, regression
- PR #1143 (alphonse): combined best config — superseded
- PR #1126 (edward): T_max=14 — superseded
- PR #1166 (edward): LR warmup + cosine — superseded
- PR #1199 (tanjiro): FiLM Re conditioning (full, unfair test) — ran without BF16, only 13 epochs; lightweight variant re-tested as PR #1264
- PR #1203 (fern): checkpoint averaging (last K=3) — awaiting results; not yet closed
- PR #1204 (alphonse): Re noise augmentation (buggy per-node) — implementation bug fixed in PR #1261
- PR #1206 (edward): mlp_ratio=3 — superseded by PR #1262
- PR #1207 (askeladd): batch_size=8 — superseded by PR #1259
- PR #1152 (thorfinn): CosineAnnealingLR eta_min=1e-5 (stale baseline) — superseded

## Potential Next Research Directions

After current WIPs resolve:
1. **Ensemble / checkpoint averaging**: Average last 3 best checkpoints (PR #1203 in flight)
2. **max_lr=1.0e-3**: Lower peak test (if 1.5e-3 works, compare both sides of 1.2e-3)
3. **Adaptive loss weighting (Kendall & Gal)**: Learn log-variance per output channel — balances Ux, Uy, p gradients dynamically
4. **Fourier positional features for mesh coords**: Random Fourier features for node x,y positions; improves spatial awareness in irregular mesh
5. **Cross-attention surface↔volume**: Dedicated attention stream for surface-to-volume interaction; addresses raceCar surface gap
6. **n_hidden=160, n_head=5**: Intermediate width avoiding epoch starvation seen at 192
7. **Higher surf_weight on raceCar only**: Conditional surf_weight boost (2x for raceCar samples) — more targeted than global reweighting
8. **Spectral/frequency features**: FFT-based features on boundary curves may capture camber shape more effectively
