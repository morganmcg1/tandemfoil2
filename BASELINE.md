# TandemFoilSet-Balanced Baseline (charlie-r5)

**Advisor track:** `icml-appendix-charlie-r5`
**Research tag:** `charlie-r5`
**W&B:** disabled (local JSONL metric logging only — see `models/<experiment>/metrics.jsonl`)

---

## Current best

**No experiment has been merged on this track yet. The current "baseline" is the vanilla Transolver shipped in `target/train.py` on `icml-appendix-charlie-r5`.**

Round 1 establishes the vanilla baseline metric and the first set of independent deltas in parallel. The very first PR to land that beats the vanilla baseline will replace this entry and become the merge anchor for this track.

### Vanilla configuration (out of `target/train.py` HEAD)

| Param | Value | Notes |
|-------|-------|-------|
| lr | 5e-4 | |
| weight_decay | 1e-4 | |
| batch_size | 4 | |
| grad_accum | 1 (no accumulation) | not yet wired |
| amp | False | not yet wired |
| surf_weight | 10.0 | code default |
| epochs | 50 | |
| n_hidden | 128 | |
| n_layers | 5 | code default |
| n_head | 4 | not yet a CLI flag |
| slice_num | 64 | not yet a CLI flag |
| mlp_ratio | 2 | not yet a CLI flag |
| optimizer | AdamW | |
| scheduler | CosineAnnealingLR(T_max=epochs) | |
| loss | MSE (vol + sw·surf, single squared-error pass) | not yet a CLI flag |
| FFN | vanilla 2-layer GELU | not yet a CLI flag |
| Fourier features | none | not yet wired |

CLI flags currently exposed by `simple_parsing`: `lr`, `weight_decay`, `batch_size`, `surf_weight`, `epochs`, `splits_dir`, `experiment_name`, `agent`, `debug`, `skip_test`. **Anything else has to be added by the student in their PR.**

### Reproduce vanilla baseline

```bash
cd target && python train.py \
    --agent <student> \
    --experiment_name <experiment-slug> \
    --epochs 50 \
    --batch_size 4
```

Local artifacts land in `models/<experiment>/`:
- `checkpoint.pt` — best-val-checkpoint state dict
- `config.yaml` — full config snapshot
- `metrics.jsonl` — per-epoch metrics + final test eval (commit this with the PR)

---

## Primary metric

- **Validation (checkpoint selection):** `val_avg/mae_surf_p` — equal-weight mean of `mae_surf_p` across the four val splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`, `val_re_rand`). Lower is better.
- **Test (paper-facing):** `test_avg/mae_surf_p` — best-val checkpoint evaluated on the four test splits, end-of-run.

Surface pressure accuracy is what matters. The trainer also logs surface/volume MAE per channel and per-split losses every epoch.

---

## Update protocol

When a PR's best `val_avg/mae_surf_p` is lower than the current entry here:

1. Squash-merge the winning PR into `icml-appendix-charlie-r5`.
2. Update this file with the new metric, PR number, run hash, run config.
3. Commit on the advisor branch.

For sub-5% claims, require a 2-seed mean and report std. As soon as we have 2-seed anchors on this track, this baseline section will get a noise band so future merges have a calibrated threshold.

---

## Carry-over knowledge from prior charlie tracks

`icml-appendix-charlie-r5` ships a clean, vanilla Transolver. Prior advisor tracks established a number of compounding wins on essentially the same training contract; the relevant ones to re-test on this track are listed below. None are assumed to transfer — each must be earned with a PR on `icml-appendix-charlie-r5`. Round-1 hypotheses are spread across these axes in parallel.

| Axis | Prior outcome | Round-1 owner |
|------|---------------|---------------|
| `loss_type` (L1 vs MSE) | L1 was a strong, early single-flag win | charlie5-alphonse |
| `surf_weight` (pressure-loss weighting) | sw=1 won over sw=10 | charlie5-frieren |
| AMP + grad_accum (throughput) | AMP + grad_accum=4 paid double — more samples per epoch + faster epochs | charlie5-askeladd |
| SwiGLU FFN | SwiGLU replaced GELU FFN | charlie5-edward |
| Fourier positional features (fixed) | σ-tuned fixed Fourier features won decisively | charlie5-fern |
| `slice_num` reduction | Smaller slice_num won repeatedly (compute-reduction theme) | charlie5-nezuko |
| `n_layers` reduction | nl=3 won over nl=5 | charlie5-tanjiro |
| `n_head` | nh trend monotonic across recipes — fewer heads + wider dim_head won | charlie5-thorfinn |
