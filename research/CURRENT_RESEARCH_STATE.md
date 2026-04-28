# SENPAI Research State

- **Date:** 2026-04-28 11:25
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **53.7986** | PR #627 (edward, preprocess-depth-1 on Lion) — merged 2026-04-28 |
| `test_avg/mae_surf_p` (3-split mean) | 52.165 | PR #627 |

Per-split val: `val_single_in_dist=54.3136`, `val_geom_camber_rc=70.8552`, `val_geom_camber_cruise=35.2098`, `val_re_rand=54.8159`.

Full reference config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, **`preprocess_layers=1`**, `fun_dim=54`, `lr=3e-4` (peak, linear warmup 5 ep), `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, `grad_clip_norm=0.5`, `amp_bf16=True`, L1 loss, `optimizer_name="lion"`, SequentialLR(LinearLR warmup x 5 ep, CosineAnnealingLR T_max=epochs-5), `--epochs 24`, 8-band Fourier features on normalized (x, z).

## Round-7 status (on Lion + preprocess-depth-1 baseline)

### WIP
| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #703 | Architecture | alphonse | preprocess_layers 1 to 2 (deeper input MLP, follow-up to PR #627 win) |
| #704 | Architecture | tanjiro | slice_num 64 to 128 (double physics attention slices, target val_geom_camber_rc) |
| #705 | Architecture | edward | n_head 4 to 8 (double attention heads, attention diversity probe) |
| #680 | Optimizer LR | thorfinn | **CLOSED** — Lion + lr=1e-4 regression (+9.7%), lr=3e-4 confirmed near-optimal |
| #710 | Schedule | thorfinn | cosine eta_min 0 → 5e-5 (tail floor probe from PR #680 diagnostic) |
| **#679 CLOSED** | Capacity x optim x schedule | fern | n_layers=6 + Lion + budget-matched cosine — **CLOSED**: +13.9% regression; budget cap kills depth (+19% wall, 3 fewer epochs) |
| *new* | Output boundary | fern | **output MLP depth +1 residual layer** (mirror of preprocess-depth-1 win at output boundary) — assigning |
| #593 | Data aug | nezuko | Re jittering (sigma=0.05 on log(Re)) at training time |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging (top-3) on Lion base |
| #369 | Regularization | askeladd | Drop-path 0.1 on Lion base |

All 8 students assigned; fern reassigned after PR #679 close (n_layers=6 budget-cap failure).

## Key findings to date

1. **Lion optimizer = dominant finding.** Sign-based update composes naturally with L1 loss + grad-clip-0.5 + bf16. -23.3% val/test improvement vs AdamW. Triple-quantized chain (L1+sign+clip) confirmed; clip=0.5 optimal; Lookahead saturates on Lion.

2. **Preprocess MLP depth-1 = confirmed stacking improvement.** PR #627 (edward): preprocess_layers=1 delivers -4.27% val, -2.18% test on the Lion baseline. New baseline val=53.7986.

3. **Persistent OOD laggard: val_geom_camber_rc=70.86.** This split has resisted all improvements (depth-1 moved it only -0.29%). New hypotheses slice_num=128 and n_head=8 target this specifically.

4. **Optimizer LR axis confirmed on Lion:** clip=1.0 neutral, Lookahead hurts, lr=1e-4 regresses (+9.7%). lr=3e-4 confirmed near-optimal. Key insight from lr=1e-4 run: cosine tail floor (~6-7e-5) is the high-leverage window. Follow-up: eta_min floor lift or lr=5e-4 upward probe.

5. **Lookahead finding generalizes:** active weight averaging during training hurts — structural test_single_in_dist regression confirmed on both AdamW and Lion. Future SWA/Polyak ideas likely hit same cost.

## Current research focus

Primary bottleneck: **val_geom_camber_rc at 70.86** — far above other splits (val_single_in_dist=54.31, val_re_rand=54.82, val_geom_camber_cruise=35.21). Two hypotheses directly targeting this:
- slice_num=128: more physics attention partitions for unseen geometry regimes
- n_head=8: more parallel attention subspaces for geometry-domain specialization

Secondary: preprocess_layers=2 (depth stacking), n_layers=6 (pending fern #679), LR probe (thorfinn #680), Re jittering (nezuko #593).

## Potential next research directions (post-current-round)

- **Physics-informed loss**: divergence penalty on velocity (Ux, Uy) or pressure Laplacian. Completely untested. May specifically help val_geom_camber_rc.
- **Per-domain loss weighting**: upweight rc-tandem samples during training to directly target the rc OOD gap.
- **Fourier feature band count**: 8 bands currently. 16 bands = higher frequency position encoding; 4 bands = cheaper. Untested on Lion stack.
- **Weight decay tuning on Lion**: wd=1e-4 (AdamW default). Lion's decoupled WD acts on momentum differently. wd=1e-5 or wd=5e-4 untested.
- **Learning rate schedule variants**: SGDR (cosine restarts), OneCycleLR. May help escape local minima.
- **mlp_ratio for preprocess MLP**: wider hidden dim in residual layers (currently 256 = n_hidden*2). Testing 512 = n_hidden*4 in preprocess layer specifically.
- **Output MLP depth**: mirror the preprocess-depth win at the output boundary. **→ ASSIGNING to fern.**

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for every PR. `data/scoring.py` is read-only. Rank by 3-clean-split test mean.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers -- local JSONL only.
- Per-epoch grad-norm telemetry (`train/grad_norm_avg`) is in the merged train.py.
- val_geom_camber_rc consistently requires special attention in per-split analysis.
