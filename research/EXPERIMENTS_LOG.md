# SENPAI Research Results — icml-appendix-charlie-pai2e-r3

## 2026-04-28 22:30 — Round 1 review summary (PRs #830-#837)

Round 1 dispatched 8 hypotheses across the default Transolver baseline. Two PRs (#832 edward, #834 frieren) remained WIP at review time. Of the 6 reviewed, **only PR #835 (nezuko, MAE/L1 loss) beat the placeholder baseline**, and is now the merged baseline.

| PR  | Student   | Hypothesis                                  | val_avg/mae_surf_p | Δ vs new baseline (104.058) | Decision      |
|-----|-----------|---------------------------------------------|--------------------|-----------------------------|---------------|
| 835 | nezuko    | MAE/L1 loss (replace MSE)                   | **104.058**        | merged (new baseline)       | **MERGED**    |
| 831 | askeladd  | surf_weight 10 → 50                         | 122.96             | +18.2%                      | sent back     |
| 836 | tanjiro   | mlp_ratio 2 → 4                             | 126.996            | +22.0%                      | closed        |
| 833 | fern      | 5-epoch warmup + lr=1e-3                    | 145.25             | +39.6%                      | closed        |
| 837 | thorfinn  | pressure channel weight [1,1,5]             | 144.08             | +38.5%                      | closed        |
| 830 | alphonse  | larger model n_hidden=256/n_layers=7        | 183.06             | +76.0%                      | closed        |
| 834 | frieren   | dropout=0.1 (attn + FFN)                    | 139.08             | +33.7%                      | closed        |

### PR #835 — Nezuko: MAE/L1 loss (MERGED, new baseline)
- Branch: `charliepai2e3-nezuko/mae-loss`
- Hypothesis: MSE over-penalizes high-Re pressure outliers; switching to MAE makes training robust and improves surface pressure metric.
- Results: `val_avg/mae_surf_p=104.058`, `test_avg/mae_surf_p=92.608` (NaN-sample-skipped workaround).
  - `target/runs/mae-loss-metrics/metrics.jsonl`
- Conclusion: Confirmed — MAE is superior to MSE for this surrogate. MAE is now the baseline loss. As a bonus, nezuko also implemented the `data/scoring.py` NaN-poisoning workaround (skip non-finite GT samples) directly in `train.py`, fixing the `test_geom_camber_cruise` evaluation for all subsequent experiments on this branch.

### PR #831 — Askeladd: surf_weight=50
- Hypothesis: Surface metric drives the score; up-weighting surface loss should focus the model.
- Result: 122.96 (still improving at 14/50 epoch cutoff). The MAE loss change (#835) addresses the same pressure-tail concern more directly. **Sent back** for an MAE-based variation.

### PR #836 — Tanjiro: mlp_ratio=4
- Result: 126.996 val, 117.46 test. Doubled FFN params (0.66M→0.99M), epochs slowed to ~149s. No improvement. **Closed.**

### PR #833 — Fern: 5-epoch warmup + lr=1e-3
- Result: 145.25. 5 of 14 epochs spent in warmup (36% of budget), wasting effective training time within the 30min cap. **Closed.**

### PR #837 — Thorfinn: pressure channel weight [1,1,5]
- Result: 144.08. Effectively double-weights pressure on top of `surf_weight=10`, over-emphasizing pressure at the expense of velocity learning. **Closed.**

### PR #834 — Frieren: dropout=0.1
- Result: 139.08 val, 125.29 test (with NaN-sample workaround). 13/50 epochs.
- Frieren independently identified and worked around the scoring.py NaN-poisoning bug — same fix already merged via #835. Heavy regularization on an under-fit 14-epoch model adds noise without payoff. **Closed.**

### PR #830 — Alphonse: larger model (256/7/8)
- Result: 183.06. OOM at batch_size=4 (~94GB) forced fallback to bs=2, only 6/50 epochs at ~5.7min each. Larger model cannot be trained within budget without bf16/AMP. **Closed.**

## 2026-04-28 23:45 — PR #891: Smooth-L1/Huber loss (delta=1.0)

- Branch: `charliepai2e3-nezuko/smoothl1-loss`
- Hypothesis: Smooth-L1 (Huber) loss with δ=1.0 should combine the tail-robustness of MAE with smoother L2 gradients near zero, potentially outperforming both MSE and pure MAE.

| Split                   | val MAE (Smooth-L1) | val MAE (baseline #889) | Δ       |
|-------------------------|---------------------|-------------------------|---------|
| val_single_in_dist      | 128.55              | 118.130                 | +8.8%   |
| val_geom_camber_rc      | 120.12              | 100.284                 | +19.8%  |
| val_geom_camber_cruise  | 81.47               | 71.079                  | +14.6%  |
| val_re_rand             | 88.40               | 88.053                  | +0.4%   |
| **val_avg**             | **109.065**         | **94.387**              | **+15.5%** |
| test_avg (NaN-skip)     | 100.091             | 92.232                  | +8.5%   |

- Epochs: 14/50, wall-clock: 30.7 min, peak VRAM: 42.14 GB.
- Metrics path: `target/runs/smoothl1-d1.0-metrics/metrics.jsonl`

**Analysis:** Clear negative result. Root cause: in normalized-output space (std≈1), the vast majority of prediction errors fall below δ=1.0, meaning the loss operates in the quadratic (MSE-like) regime for almost all samples. Only genuine outlier predictions trigger the linear (MAE-like) tail. This effectively reintroduces MSE's original problem — high-Re pressure outliers contribute disproportionate quadratic gradient signal rather than the capped linear signal that made pure MAE better. Every split regressed relative to the pure-MAE baseline (#889, val_avg=94.387). The geometry-camber splits were the hardest hit (+14–20%), consistent with the hypothesis that these splits contain the most high-Re distribution-shift outliers.

**Decision: CLOSED.** Smooth-L1 with δ=1.0 is a regression. A much smaller δ (e.g. 0.1 or 0.3 in normalized space) would push more errors into the linear regime and might recover the MAE benefit — but that is a separate hypothesis. Pure MAE remains superior.

## 2026-04-28 23:30 — Round 2 partial review (PRs #831, #832)

Two carry-over PRs from round 1 returned; neither beat the merged MAE baseline (104.058). Both closed; both students re-assigned to fresh round-2 work.

| PR  | Student   | Hypothesis                                      | val_avg/mae_surf_p | Δ vs baseline (104.058) | Decision |
|-----|-----------|-------------------------------------------------|--------------------|-------------------------|----------|
| 832 | edward    | slice_num 64 → 128 (+ bf16 AMP, NaN-safe eval)  | 130.63             | +25.5%                  | closed   |
| 831 | askeladd  | surf_weight=50 (carry-over, MSE baseline)       | 122.96             | +18.2%                  | closed   |

### PR #832 — Edward: slice_num=128 with bf16 AMP
- Branch: `charliepai2e3-edward/more-slices` (closed)
- Round 1 (#832 original): `val_avg/mae_surf_p=140.81`, `test_avg=NaN` (NaN poisoning on test_geom_camber_cruise pressure). Sent back for NaN-safe eval + bf16 AMP.
- Round 2: `val_avg/mae_surf_p=130.63`, `test_avg/mae_surf_p=115.56` (all 4 splits clean, n_dropped=0). 13 epochs vs 11, 143s/ep vs 170s, 48.08 GB vs 54.51 GB peak VRAM.
  - `target/research/metrics/charliepai2e3-edward-slice_num_128_amp_v2_flp3r7yr.jsonl`
- **Closed** — does not beat MAE baseline. Branched off MSE; the fair comparison would be slice_num=128 on MAE.
- **Carry-over artifacts:**
  1. NaN-safe `evaluate_split` in `train.py` — per-sample `isfinite(pred_orig)` guard + `torch.nan_to_num` cleanup. Robust to both NaN-GT (already merged via #835) and NaN-pred. Worth re-introducing on top of MAE baseline.
  2. bf16 AMP integration validated — `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` on forward+loss, optimizer in fp32. ~1.2× throughput, lower VRAM, no instability.

### PR #831 — Askeladd: surf_weight=50 (carry-over)
- Branch: `charliepai2e3-askeladd/higher-surf-weight` (closed)
- Result: 122.96 val. Originally sent back from round 1 for an MAE-based variation, but askeladd and advisor agreed the MAE/L1 loss change in #835 already addresses the high-Re pressure outlier concern that motivated higher surf_weight. **Closed**, no re-run needed.

### Cross-cutting findings from Round 1
1. **MAE loss is a clean win** — every future experiment must build on it. The fix is in `train.py`.
2. **The 30-min/14-epoch budget is the binding constraint.** Schedules tuned for 50 epochs (cosine T_max=50, 5-epoch warmup) are mistuned. Future experiments must adjust schedules to the realized ~14-epoch budget.
3. **scoring.py NaN bug** is now worked around in `train.py` (skip non-finite GT). All future students inherit this fix automatically by branching from the advisor branch.
4. **Capacity scaling needs AMP** — the default model already uses ~22GB per sample. Any architecture larger than (128, 5, 4) needs bf16 mixed precision to fit at bs=4 in the budget.
5. **Camber-holdout split is hardest** (val_geom_camber_rc=116.84 vs val_geom_camber_cruise=76.93 in MAE baseline) — the rc-camber unseen-front-foil generalization gap is the largest source of error and the highest-leverage place to focus.
