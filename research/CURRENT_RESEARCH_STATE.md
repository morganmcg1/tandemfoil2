# SENPAI Research State

- **Date:** 2026-04-28
- **Advisor branch:** `icml-appendix-willow-pai2e-r3`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r3`
- **Most recent human researcher direction:** *(none — issue queue empty)*

## Current research focus

This is **round 1** of the willow-pai2e-r3 advisor branch. There is no committed `BASELINE.md` yet — the comparison target is the default `train.py` Transolver:
- 5 layers, hidden_dim=128, n_head=4, slice_num=64, mlp_ratio=2 (~1M params)
- AdamW lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0
- 50 epochs, plain cosine annealing, MSE loss in normalized space

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the 4 validation splits. Test mirror: `test_avg/mae_surf_p`. Lower is better.

**Key dataset insight (driving round-1 hypothesis selection):** Pressure has `y_std=679`, ~30× larger than `Ux` (`y_std=22`) and ~70× larger than `Uy` (`y_std=10`). The current uniform-weighted MSE under-emphasizes the channel that drives the metric. Several round-1 experiments target this directly (channel-weighted loss, L1 surface loss).

## Round-1 hypothesis matrix (8 students, all assigned)

Diverse first-principles set, designed to span loss / optimization / architecture / input-engineering / regularization angles so we surface multiple orthogonal levers at once.

| Student | PR | Hypothesis | Predicted Δ | Angle |
|---------|----|-----------|-------------|-------|
| thorfinn | [#762](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/762) | boundary-layer-features (`log(Re·\|saf\|)`) | -10 to -25% | Input physics |
| askeladd | [#748](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/748) | transolver-2x-capacity (h=192, L=8, slice=128) | -5 to -15% | Architecture scale |
| tanjiro | [#761](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/761) | l1-surface-mae-loss | -5 to -12% | Loss alignment |
| alphonse | [#743](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/743) | channel-weighted-surface-loss (3× p) | -5 to -10% | Loss weighting |
| thorfinn (FiLM moved → boundary-layer) | — | — | — | Replaced |
| nezuko | [#759](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/759) | ema-model-weights (decay=0.999) | -3 to -8% | Training stability |
| edward | [#750](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/750) | lr-warmup-cosine (lr=1e-3, wd=5e-4, 500-step warmup) | -3 to -8% | Optimizer |
| fern | [#751](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/751) | dropout-stochastic-depth (0.1 / linear-scaled) | -3 to -8% | Regularization |
| frieren | [#756](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/756) | fourier-re-encoding (16 sin/cos features) | -3 to -7% | Input encoding |

## Potential next research directions (round 2+)

Pulled from `RESEARCH_IDEAS_2026-04-28_FIRST.md` — the researcher-agent's top ideas not yet assigned:

1. **Huber-surf-loss (delta=1.0)** — robust regression for heavy-tailed pressure; complements L1 if L1 wins.
2. **Low-rank slice attention (LRSA)** — replace S×S slice-token self-attention with rank-16 factored attention; reportedly +17% on PDE benchmarks.
3. **RevIN output normalization** — reversible per-sample amplitude normalization on `y` before loss; targets the 10× intra-split y_std variation across Re.
4. **Re-stratified oversampling** — within-domain oversample top Re-quintile samples; addresses gradient under-coverage of high-Re extremes.
5. **Focal-surface-loss** — top-20%-error-node up-weighting; concentrates gradient on stagnation point / leading edge / suction peak.
6. **Stack winners** — round-2 should bundle round-1 winners (channel-weighted + L1 + EMA + warmup are largely orthogonal).
7. **FiLM conditioning** — global flow-condition modulation per Transolver block (deferred from round 1 in favor of boundary-layer features, both target similar conditioning gap).
8. **Mixed-precision (bf16/fp16)** — opens headroom for capacity scaling under timeout cap.

## Compute state

- All 8 student pods (`willowpai2e3-*`) are running, polling for assignments.
- Per-run timeout: 30 minutes wall clock (default `SENPAI_TIMEOUT_MINUTES`).
- Per-run epoch cap: 50 epochs (default) — **but timeout binds first**: PR #743 hit timeout at exactly 14 epochs (~131 s/epoch). Plan for ~14 epochs at current model size; future hypotheses should set `--epochs 14` so cosine annealing reaches the end of its curve.

## Cross-cutting findings from round 1

- **NaN test poisoning** — `accumulate_batch` in `data/scoring.py` (read-only) only skips on non-finite ground truth, not non-finite preds. A single bad predicted `p` value on `test_geom_camber_cruise` poisoned `test_avg/mae_surf_p` on PR #743. Defensive fix in `train.py`'s `evaluate_split`: `pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)` before denormalization. Will land in baseline once first PR with the fix merges. Likely to recur on other round-1 submissions.
- **Cruise OOD camber (M=2-4)** is the most extrapolation-prone test split — appears to be where NaN preds materialize, likely a 1-2-sample edge case.
