# SENPAI Research Results — icml-appendix-charlie-pai2c-r2

## 2026-04-27 19:52 — PR #216: Wider-shallower Transolver: n_hidden=192, n_layers=4, n_head=6
- **Branch**: `charliepai2c2-tanjiro/wider-shallower-arch`
- **Hypothesis**: For irregular-mesh CFD surrogate problems, *wider* per-node feature dim beats *deeper* network depth — surface pressure MAE depends on local per-node features, not long-range mesh attention. Predicted 5–10% reduction vs. seed baseline.
- **Result**: ESTABLISHED EMPIRICAL BASELINE for the branch. `val_avg/mae_surf_p = 130.0568` at epoch 11/50 (training timed out at 30 min wall clock; was still improving ~5%/epoch).

| Metric | val_single_in_dist | val_geom_camber_rc | val_geom_camber_cruise | val_re_rand | **val_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 161.146 | 142.415 | 98.282 | 118.383 | **130.057** |
| `mae_surf_Ux` | 1.946 | 2.640 | 1.608 | 2.113 | 2.077 |
| `mae_surf_Uy` | 0.855 | 1.129 | 0.637 | 0.882 | 0.876 |

| Metric | test_single_in_dist | test_geom_camber_rc | test_geom_camber_cruise | test_re_rand | **test_avg** |
|---|---:|---:|---:|---:|---:|
| `mae_surf_p` | 144.256 | 127.587 | NaN ⚠️ | 115.976 | NaN ⚠️ |
| `mae_surf_Ux` | 1.860 | 2.601 | 1.529 | 1.999 | 1.997 |

- **Param count**: 1.18M (~1.79× the seed config's analytical ~0.66M)
- **Peak VRAM**: 51.79 GB | **Wall-clock**: 30.3 min | **Best epoch**: 11/50
- **Metrics summary**: `models/model-wider-shallower-arch-20260427-191514/metrics.yaml`
- **JSONL (centralized)**: `research/EXPERIMENT_METRICS.jsonl`
- **Reproduce**: `cd target && python train.py --epochs 50 --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --experiment_name wider-shallower-arch`

### Analysis

Decision: **MERGED** as the empirical baseline-establisher. With no prior numbers on this advisor branch, this becomes the floor for Round 1 to beat. Two structural concerns are recorded for the round:

1. **Undertrained.** `T_max=50` cosine schedule + 30-min wall-clock cap = 11 epochs at ~165 s/epoch, with the LR never entering its decay tail. The `val_avg/mae_surf_p` curve was still dropping ~5%/epoch when the timer fired (epoch 10 → 11: 135.98 → 130.06). PR #208 (`lr-warmup-cosine-floor`) partially addresses this with the 1e-5 floor; a budget-matched `T_max` follow-up is warranted.
2. **Test-metric NaN bug.** Tanjiro identified that `inf * 0 → NaN` in `data/scoring.py::accumulate_batch` silently NaN-contaminates `test_avg/mae_surf_p` on `test_geom_camber_cruise` (single sample 020 has 761 Inf pressure values). `data/scoring.py` is read-only per `program.md` — flagging to human team. **Until patched, rank on `val_avg/mae_surf_p` only.** Mean over the 3 finite test splits is `mae_surf_p ≈ 129.27`.

### Per-split observations

The cruise tandem geometry-holdout split is the easiest (`mae_surf_p = 98.28`); single-foil in-distribution is hardest (`mae_surf_p = 161.15`). This is counterintuitive — the in-distribution sanity-check split should be easiest. Possible explanations:
- Single-foil samples have larger pressure dynamic range (per program.md: y range up to ±29K vs cruise ±7.6K), so even a small relative error compounds.
- The balanced sampler equally weights three domains (single, raceCar tandem, cruise tandem), so single-foil gets ~33% of training samples but is the highest-variance domain.

This per-split structure should inform future hypotheses — domain-id-conditioning, per-Re-bin scaling, or single-foil-specific decoder heads may pay off.
