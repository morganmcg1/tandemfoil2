# SENPAI Research Results — willow-pai2d-r5

Per-PR experiment log. New entries are appended chronologically; the latest entries are at the top.

## 2026-04-28 00:50 — PR #329: surf_weight sweep {20, 30, 50} — **SENT BACK (apples-to-apples needed)**
- Branch: `willowpai2d5-alphonse/surf-weight-sweep` (sits on pre-#336 commit, slice_num=64)
- Three runs (sw=20, 30, 50), all at 14 epochs, slice_num=64 (pre-#336 fork)

| surf_weight | val_avg/mae_surf_p | val_avg/mae_vol_p | best ep | W&B id |
|---:|---:|---:|---:|---|
| 20 | 131.85 | 144.92 | 14 | 9nh5gk1m |
| 30 | 132.35 | 148.52 | 12 | 4fpwmk2m |
| 50 | **130.55** | 169.31 | 12 | fvbnu12q |

- All three beat current baseline (139.83) by 5.4–6.6%, but: branch is on slice_num=64, not 128. Cannot disentangle surf_weight effect from slice_num effect without a rebased re-run.
- Per-channel volume MAE blows up at sw=50 (`mae_vol_Ux` +55%) — the failure mode the original hypothesis flagged.
- val/test ranking disagree (val: sw=50 > sw=20; 3-finite-split test: sw=20 > sw=50). Single-seed, ~1-2% spread, near noise floor.
- **Concerning cross-evidence:** every slice_num=64 run in this round (these three + frieren's warmup control 130.43 + edward's deeper_l8 corrected) clusters in the 130-152 band, while the only slice_num=128 run (fern's #336) lands at 139.83 at fewer epochs. This raises the possibility that #336 was a partial-credit merge — slice_num=128 may convert better with more wall clock, but loses inside the 30-min cap.
- Sent back asking for rebase + one focused re-run of `surf_weight=50` on slice_num=128 (current advisor) to disentangle the two effects. If rebased run beats 139.83, merge surf_weight=50; if not, that's evidence to revisit #336.
- Alphonse also independently re-diagnosed the cruise NaN bug (alongside edward's #334/#375). No duplicate work needed.

## 2026-04-28 00:46 — PR #376: Wider MLP (mlp_ratio 2→4) — **CLOSED**
- Branch: `willowpai2d5-fern/mlp-ratio-4` (deleted)
- Hypothesis: doubling MLP hidden width lifts `val_avg/mae_surf_p` ~3-7%
- Result: `val_avg/mae_surf_p = 146.65` at epoch 10 of 10 completed (timeout) — **+4.9% regression** vs current baseline (#336, 139.83)
- W&B run: `mlp4` / wfxtjub5 (group `capacity_mlp_ratio`)
- Per-split: only `val_single_in_dist` improves (-5.70); all 3 OOD splits regress (+17.36 / +7.05 / +8.56)
- 1.00M params, 62.5 GB peak VRAM bs=4 (vs baseline 54.5 GB), bs=8 OOMed
- Decision: **closed** — at the 5% close threshold; per-split pattern (in-dist wins, OOD loses) is under-generalization signature, not undertraining; bf16+bs=8 retry path is redundant with askeladd's #331 retry; epoch-6 transient lead (-12 pts) is noted as round-2 input.
- Fern reassigned to spatial Fourier features (PR #405).

## 2026-04-28 00:35 — PR #375: nan_to_num fix in data/scoring.py — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-edward/scoring-nan-fix` (sits on pre-#336 commit; rebase needed)
- One-line `torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)` after `err = (pred - y).abs()` in `accumulate_batch`.
- **Verification:** bit-exact `rel_diff = 0` parity with the pre-fix path on the three previously-finite test splits, evaluated against the saved `model-deeper_l8-sfyn75sq:best` artifact. Cruise split goes from `NaN` → `99.89` (per-sample-skip semantics; smaller than the post-hoc form's ~117.30 from PR #334's monkey-patch — divergence intentional and correctly flagged by the student).
- **Post-fix `test_avg/mae_surf_p` for `deeper_l8` artifact:** 141.52.
- 5-epoch end-to-end smoketest on current advisor branch (slice_num=128) confirms full pipeline finite (cruise = 105.69 fresh, but at only 5 epochs not comparable to baselines).
- Cannot squash-merge as-is: branch's diff ALSO reverts `BASELINE.md`, `research/CURRENT_RESEARCH_STATE.md`, `research/EXPERIMENTS_LOG.md` to pre-#336 state. Sent back asking for rebase + force-with-lease push.
- Edward also flagged a same-shape NaN-leak in `train.py`'s `evaluate_split` for the normalized-space loss (auxiliary monitoring, not paper metric); correctly kept out of scope. Follow-up `train.py` PR to be filed after this lands.

## 2026-04-28 00:30 — PR #338: LR warmup + peak 1e-3 cosine — **SENT BACK (intent to merge)**
- Branch: `willowpai2d5-frieren/lr-warmup-cosine` (sits on pre-#336 commit, slice_num=64)
- Two-config sweep (`lr=5e-4` control vs `lr=1e-3` main), both with 2-epoch linear warmup + cosine T_max=48
- Both timeout-bound at epoch 14/50 (cosine arm only ~25% engaged)

| Run | val_avg/mae_surf_p | W&B id |
|---|---:|---|
| Control (lr=5e-4 + warmup) | **130.43** (ep 12) | n8y9yy70 |
| Main (lr=1e-3 + warmup)    | 142.17 (ep 14) | r439zxf5 |

- Negative result on the lr bump (+9% worse — high LR never anneals inside the timeout).
- **Positive result on warmup itself**: control beats current baseline (slice_num=128, no warmup) at 139.83 by ~6.7%, despite running on the *older* slice_num=64 setup. Strong implication that warmup composes additively.
- Cannot merge as-is: branch diff would revert slice_num 128→64, change Config.lr default 5e-4→1e-3, AND add the (good) warmup block. Sent back asking for rebase onto advisor + revert lr default + one re-run on slice_num=128 + warmup + lr=5e-4 to confirm composition.

## 2026-04-27 23:54 — PR #334: Deeper Transolver (n_layers 5→8) — **CLOSED**
- Branch: `willowpai2d5-edward/deeper-l8` (deleted)
- Hypothesis: deeper hierarchy of slice tokens lifts `val_avg/mae_surf_p` ~5-10%
- Result: `val_avg/mae_surf_p = 152.24` at epoch 8 of 9 completed (timeout). Test corrected (post-hoc nan_to_num): 145.87.
- W&B run: `deeper_l8` / sfyn75sq
- Decision: **closed** — clearly worse than slice_num=128 contemporary (152.24 vs 139.83), and slow per-epoch (~205 s vs ~135 s baseline) eats the cosine schedule before it can decay. Student's own analysis correctly recommends against pursuing depth alone.
- **Major bonus:** Edward diagnosed the cross-cutting `data/scoring.py` NaN-poisoning bug (`NaN * 0.0 = NaN` defeats per-sample skip mask). Cruise test sample 000020 has 761 NaN values in p-channel of `y`. Filed as PR #375 (advisor-authorized exception to read-only contract for `data/scoring.py`).

## 2026-04-27 23:54 — PR #336: More physics slices (slice_num 64→128) — **MERGED**
- Branch: `willowpai2d5-fern/more-slices-128` (squash-merged into advisor)
- Hypothesis: doubling slice tokens lifts `val_avg/mae_surf_p` ~3-7%, biggest gain on cruise (largest meshes)
- Result: `val_avg/mae_surf_p = **139.83**` at epoch 10 of 11 completed (timeout)
- W&B run: `slices_128` / 8xow4ge3 (group `capacity_slices`)
- Per-split val mae_surf_p: single 179.11 / camber_rc 144.31 / camber_cruise 110.05 / re_rand 125.87
- 0.67M params (no extra params from slice_num — only changes attention shape), peak VRAM 54.5 GB
- Decision: **merged** — best round-1 reviewable val so far, one-line change, low complexity. Establishes round 1 baseline empirically.
- Caveat: undertrained (11/50 epochs); val curve still descending. Subsequent winners will compound on top.
- `test_avg/mae_surf_p` = NaN due to scoring bug; 3-finite-split mean = 142.79.

## 2026-04-27 23:18 — PR #331: Wider Transolver (n_hidden 128→192, n_head 4→6) — **SENT BACK**
- Branch: `willowpai2d5-askeladd/wider-h192-h6`
- Hypothesis: 2.2× wider Transolver lifts `val_avg/mae_surf_p` ~5-10%
- Status: sent back — undertrained (9/50 epochs, timeout-capped) **and** test_geom_camber_cruise pressure NaN; not mergeable as-is, direction still promising

### Best results so far (under-trained, bs=4, 9 epochs, W&B `wider_h192_h6` / x54plqj1)

| Split | mae_surf_p | mae_vol_p |
|---|---:|---:|
| val_single_in_dist | 209.380 | 189.063 |
| val_geom_camber_rc | 168.777 | 158.080 |
| val_geom_camber_cruise | 109.090 | 105.050 |
| val_re_rand | 128.798 | 122.120 |
| **val_avg** | **154.011** | 143.578 |
| test_single_in_dist | 196.541 | 170.615 |
| test_geom_camber_rc | 150.876 | 144.267 |
| test_geom_camber_cruise | **NaN** | **NaN** |
| test_re_rand | 127.227 | 122.295 |
| **test_avg** | **NaN** | **NaN** |

### Analysis
- Val curve still falling steeply at termination (epoch 9 = best, declining ~7 per epoch in last three epochs); 50 epochs of cosine never engaged. Cannot conclude wider vs baseline yet.
- bs=8 follow-up OOMed at 94.97 GB (peak at bs=4 already 63 GB).
- **Test NaN root cause:** `accumulate_batch` skip-mask uses `torch.isfinite(y)`, not predictions; an extreme pred² overflow in fp32 normalized space (single cruise test sample) propagated NaN into the per-channel surface MAE. Vol_loss=inf logged on the same split is the smoking gun. Affects every PR's test scoring potentially — flagged for round-2 hardening.
- Wider config measured at 1,447,521 params; **actual baseline is 662,359** (not the ~1.4M placeholder I wrote). `BASELINE.md` updated.

### Action
Sent back with: bf16 autocast + fp32 cast before squaring loss, defensive `torch.nan_to_num` in `evaluate_split` (NOT `data/scoring.py`), `--batch_size 8`. Same `--wandb_group capacity_width`. PR remains open.
