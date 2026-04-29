# SENPAI Research State
- 2026-04-29 16:30
- No recent research directions from human researcher team (no GitHub issues found)
- Current research focus: BF16 AMP merged (PR #1184, val=89.00, -11.4%). Now exploring additional throughput/capacity improvements: batch_size=8 (VRAM freed by BF16), mlp_ratio=3 (FFN capacity, negligible overhead), and architectural extensions (FiLM Re conditioning, per-field heads, checkpoint averaging, Re noise augmentation, OneCycleLR).

## Current Baseline

**val_avg/mae_surf_p = 89.00** (PR #1184, charliepai2f2-askeladd, BF16 AMP + lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware CosineAnnealingLR + surf_weight=25, epoch 19/50)

Per-split breakdown (PR #1184 — current best):

| Split | val mae_surf_p | test mae_surf_p |
|-------|----------------|-----------------|
| single_in_dist | 105.41 | 89.40 |
| geom_camber_rc | 101.20 | 89.89 |
| geom_camber_cruise | 65.37 | 54.72 |
| re_rand | 84.04 | 76.37 |
| **avg** | **89.00** | **77.59** |

Stack: BF16 AMP (`torch.autocast` + `dtype=torch.bfloat16`) + lr=1e-3 + grad_clip=1.0 + DropPath 0→0.1 + budget-aware CosineAnnealingLR + surf_weight=25 + NaN guard + WeightedRandomSampler

Note: NaN guard (`torch.nan_to_num` + sample-level finite filter) now standard in all new experiments. test_geom_camber_cruise corrupted GT sample `000020.pt` is fully mitigated by per-sample NaN guard in train.py. Peak GPU memory: 32.95 GB (down from ~42 GB with BF16).

## Key Insights

**1. Systematic Timeout/LR Mismatch (critical, fixed)**
All experiments use `CosineAnnealingLR(T_max=50)` but the 30-min timeout means only ~14-19 epochs complete (14 without BF16, 19 with BF16). Budget-aware dynamic estimate (T_max estimated from warm-up timing) ensures cosine anneals fully within the available epochs.

**2. lr=1e-3 + grad_clip=1.0 is confirmed as transformative (MERGED PR #1098)**
val_avg/mae_surf_p=100.41 with lr=1e-3 + grad_clip=1.0. This is a -17.6% improvement — the largest single gain in this programme. All subsequent experiments build on this as the base config.

**3. Throughput > capacity under 30-min budget**
slice_num=128 (PR #1087 closed): +60% epoch time, only 9 epochs, regression. n_layers=8 (PR #1089 closed): +58% epoch time, only 9 epochs, regression. n_hidden=192 (PR #1086 iter 2 closed): +55% epoch time, only 9 epochs, regression. n_layers=6 (PR #1161 closed): +18% epoch time, only 12 epochs, 27% regression. General rule: any change that slows epoch time by >20% is likely fatal in the 30-min budget. Throughput gains (BF16, batch=8) are positive; capacity expansions without speed recovery are negative.

**4. BF16 AMP delivers ~27% throughput gain and VRAM headroom (MERGED PR #1184)**
135s/epoch → 98.5s/epoch, enabling 19 epochs vs 14. val=89.00 (-11.4% vs PR #1098). Peak VRAM 42GB → 32.95GB — now batch_size=8 is feasible on 96GB H100 (PR #1207 in flight). No GradScaler needed for BF16.

**5. Per-sample instance normalization failed catastrophically (PR #1129, closed)**
Per-channel std normalization down-weights pressure (largest std channel), causing 3x worse results (val=366.76). Physical unit scales must be anchored globally.

**6. DropPath 0→0.1 is optimal for the epoch-limited regime (0→0.2 confirmed negative, PR #1156 closed)**
PR #1091 merged: DropPath 0→0.1 improved 127.67 → 121.89 (-4.5%). PR #1156: DropPath 0→0.2 gave val=126.17 — 25.8% worse. Root cause: 14-19 epoch budget causes underfitting with higher drop rates.

**7. Weight decay wd=1e-4 confirmed optimal; wd=1e-3 over-regularizes (PR #1178, closed)**
Increasing wd to 1e-3 gave val=100.90 — +0.49% regression. OOD splits hurt most. DropPath 0→0.1 already provides sufficient implicit regularization.

**8. surf_weight=25 confirmed optimal; surf_weight=50 causes regression (PR #1182, closed)**
surf_weight=50 gave val=101.86 (+1.4% regression vs 100.41). Volume context is essential for OOD surface prediction.

**9. mlp_ratio=4 is a dead end (PR #1102, closed)**
mlp_ratio=4 gave val=136.16 — 6% regression; wider feedforward slows epoch time, fewer epochs in budget. mlp_ratio=3 (moderate increase, negligible overhead) has NOT been tested — assigned to edward PR #1206.

## Active Experiments

| PR | Student | Hypothesis | Key Config | Status |
|----|---------|-----------|------------|--------|
| #1090 | frieren | Per-field output heads: separate MLP decoder for Ux, Uy, p | n_hidden=128, 3 separate heads | Running |
| #1152 | thorfinn | CosineAnnealingLR eta_min=1e-5: non-zero LR floor for final epochs | eta_min=1e-5 (was 0), T_max=14 | Running (stale baseline 127.67) |
| #1195 | nezuko | OneCycleLR superconvergence: replace cosine anneal within budget | OneCycleLR(max_lr=1e-3, total_steps=19*steps_per_epoch) + full current stack | Running |
| #1199 | tanjiro | Re-conditioning: FiLM conditioning with log(Re) | log(Re) → FiLM inject into each layer; targets re_rand split | Running |
| #1203 | fern | Checkpoint averaging: average last K=3 best checkpoints | deque(maxlen=3) of top-3 val checkpoints; element-wise param avg post-training | Running |
| #1204 | alphonse | Re noise augmentation: Gaussian noise on log(Re) input (dim 13) | RE_NOISE_STD=0.05 normalized units; targets re_rand OOD split | Running |
| #1206 | edward | mlp_ratio=3: wider FFN (256→384), negligible timing overhead | mlp_ratio=2→3 in model_config dict | Assigned (NEW) |
| #1207 | askeladd | batch_size=8: smoother gradients with BF16 VRAM headroom | batch_size=4→8; no LR scaling; keep lr=1e-3 | Assigned (NEW) |

Note: PR #1152 (thorfinn) was started before PRs #1091 and #1098 merged and uses stale baseline 127.67. When it completes, results will likely beat its stated baseline but not current 89.00. Will evaluate when results arrive.

## Merged Winners (Chronological)

1. **PR #1088** (edward, surf_weight 10→25): val_avg/mae_surf_p = 127.67
2. **PR #1091** (nezuko, DropPath 0→0.1 + budget-aware CosineAnnealingLR): val_avg/mae_surf_p = **121.89** (-4.5%)
3. **PR #1098** (tanjiro, lr=1e-3 + grad_clip=1.0): val_avg/mae_surf_p = **100.41** (-17.6%)
4. **PR #1184** (askeladd, BF16 AMP on current stack): val_avg/mae_surf_p = **89.00** (-11.4%) — **current baseline**

## Closed / Dead Ends

- PR #1087 (askeladd): slice_num=128 — 4% regression; slower per-epoch kills epoch budget
- PR #1086 Iters 1+2 (alphonse): width expansion n_hidden=192/256 — 204s/epoch too slow; epoch starvation dominates
- PR #1129 (askeladd): per-sample instance-normalized loss — 3x regression (val=366.76); per-channel std down-weights pressure
- PR #1089 (fern): n_layers=8 — 207s/epoch, only 9 epochs, regression; also used stale surf_weight=10
- PR #1161 (fern): n_layers=6 — 158s/epoch, only 12 epochs, 27% regression (127.56 vs 100.41); throughput-over-capacity dominates
- PR #1102 (thorfinn): mlp_ratio 2→4 — val=136.16, 6% regression; wider feedforward slows epoch time, fewer epochs in budget
- PR #1156 (nezuko): DropPath 0→0.2 — val=126.17, 25.8% regression vs 100.41; 14-epoch budget causes underfitting at higher drop_path rates
- PR #1178 (tanjiro): weight decay wd=1e-3 — val=100.90 (+0.49% regression); OOD splits hurt most; wd=1e-4 confirmed optimal
- PR #1144 (askeladd): BF16 AMP (stale baseline) — val=122.39; technique validated (26% speedup), retested and MERGED as PR #1184
- PR #1185 (nezuko): SGDR warm restarts (T_0=5, T_mult=1) — val=108.69; restart spikes incompatible with 14-epoch budget
- PR #1182 (fern): surf_weight 25→50 — val=101.86 (+1.4% regression); vol_p 9.3% worse; surf_weight=25 confirmed optimal
- PR #1143 (alphonse): combined best config — CLOSED as superseded; val=98.68 within variance; validates reproducibility
- PR #1126 (edward): T_max=14 — superseded by PR #1166 (LR warmup + cosine)
- PR #1166 (edward): LR warmup + cosine (LinearLR 1e-6→1e-3 then CosineAnnealingLR) — superseded; edward now assigned PR #1206 (mlp_ratio=3)

## Potential Next Research Directions

Highest priority after current WIPs resolve:
1. **WeightedRandomSampler tuning**: Currently uniform-weighted; class-imbalanced OOD splits may benefit from over-sampling harder splits (re_rand, geom_camber_rc)
2. **Fourier features for mesh positions**: Random Fourier features for irregular mesh node coordinates; improves spatial awareness at low model cost
3. **Cross-attention surface↔volume**: Dedicated attention stream for surface-to-volume interaction; addresses biggest remaining gap (surface prediction lags volume in some splits)
4. **AoA / geometry augmentation**: Perturb AoA and NACA parameters within physical bounds during training — improves generalization across geometry OOD splits
5. **Adaptive loss weighting (uncertainty-weighted)**: Learn log-variance per output channel as in Kendall & Gal (2018); balances velocity and pressure gradients dynamically
6. **DropPath fine-grained sweep (0.05, 0.075)**: Lower rates may yield marginal gains over 0.1
7. **n_hidden=160, n_head=5**: Intermediate width that may avoid the epoch-starvation problem seen at n_hidden=192 while still adding capacity
