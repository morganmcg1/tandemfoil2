# SENPAI Research State (charlie-r5)

- **Updated:** 2026-04-27 16:00 — round 1 boot
- **Advisor branch:** `icml-appendix-charlie-r5`
- **Research tag:** `charlie-r5`
- **W&B:** disabled (local JSONL metric logging only — see `target/models/<experiment>/metrics.jsonl`)

---

## Current Baseline

**No PR has been merged on this track yet.** The "baseline" is the vanilla Transolver in `target/train.py` HEAD. See `BASELINE.md` for the exact config; the very first PR that beats vanilla replaces the baseline entry and seeds round 2.

---

## Round-1 strategy: parallel re-establishment of known-strong axes

`icml-appendix-charlie-r5` is a fresh fork from a vanilla Transolver. Prior charlie tracks established eight compounding wins (L1 loss, sw=1, AMP+grad_accum, SwiGLU FFN, fixed Fourier features, slice_num↓, n_layers↓, n_head↓). For this ICML appendix track we re-test those wins from scratch — each on its own PR, against the vanilla baseline — so deltas are independently attributed and the appendix narrative is clean.

Round 1 fans 8 students across 8 independent axes. Each PR adds at most one CLI flag's worth of code, runs against the vanilla baseline, and reports the best `val_avg/mae_surf_p` plus per-split breakdown. After round 1 lands, round 2 will compound the winners sequentially.

### Round-1 student → axis assignments

| Student | Axis | Hypothesis (one-liner) |
|---------|------|------------------------|
| charlie5-alphonse | Loss function | `--loss_type {mse, l1}` — L1 expected to beat MSE on long-tailed surface pressure |
| charlie5-askeladd | Throughput | `--amp` + `--grad_accum {1, 2, 4}` — bf16 AMP + grad_accum=4 expected to compound (faster epochs + larger effective bs) |
| charlie5-edward | FFN | `--swiglu` — SwiGLU replaces GELU FFN inside Transolver blocks |
| charlie5-fern | Positional features | `--fourier_features fixed --fourier_m 160 --fourier_sigma {0.5, 0.7, 1.0}` |
| charlie5-frieren | Surface weighting | `--surf_weight {1, 3, 10, 30}` |
| charlie5-nezuko | Token count | `--slice_num {64, 32, 16}` |
| charlie5-tanjiro | Depth | `--n_layers {3, 4, 5}` |
| charlie5-thorfinn | Heads | `--n_head {1, 2, 4, 8}` (with shape-preserving `dim_head = n_hidden / n_head`) |

All eight axes are pairwise low-conflict and can compound across rounds.

---

## Decision criteria

- **Merge** a PR if its best `val_avg/mae_surf_p` is lower than the current track baseline (vanilla on round 1).
- **Send back** with feedback if the direction is promising but the chosen variant didn't win — students iterate on the same PR.
- **Close** if the result is materially worse (>5% regression) or the implementation diverged/crashed.
- Until 2-seed anchors exist on this track, treat single-seed deltas <2% as uncertain — request a confirmatory seed before merging.

---

## Carry-over knowledge from prior charlie tracks (do not re-discover)

These were robust on `kagent_v_students` (and earlier charlie tracks) over many rounds:

- **L1 loss** beat MSE.
- **surf_weight ≈ 1** beat sw=10.
- **AMP (bf16)** + **grad_accum=4** compounded — both throughput and effective batch size matter.
- **Fourier σ ≈ 0.7** with `m=160`, fixed init, won over none and over learned variants.
- **SwiGLU FFN** beat GELU FFN.
- **slice_num** trend monotonically downward — sn=8 was the deepest tested floor and still descending.
- **n_layers ≤ 3** beat n_layers=5 on this dataset (Transolver depth is on the cliff at 5).
- **n_head ↓ with wider dim_head** trend was monotonic across recipes (nh=1 < nh=2 < nh=4 << nh=8).

Round-1 PRs each verify one axis on `icml-appendix-charlie-r5`. Round 2 will compound winners.

---

## Open research themes for round 2+

- **Compute-reduction floor** never hit on the prior track — sn=4, nl=1, n_hidden shrinkage, mlp_ratio<2, n_head=1 all worth pushing once round 1 confirms the trend.
- **Loss reformulation** — Huber, log-cosh, per-channel weighting, dimensionalised pressure loss.
- **Data representation** — alternative positional features (signed-distance to surface, angle-of-attack-relative coordinates), per-domain normalization, surface-aware token sampling.
- **Architecture beyond Transolver** — point-cloud transformers, mesh-aware attention, slice-bottleneck residual decoders.
- **Optimization regime** — LR schedule (warmup + cosine-floor), longer epochs at smaller models (compute reallocation).
- **Multi-seed methodology** — once a recipe stabilises, do 2-seed anchors so merge thresholds are calibrated.

---

## Student roster (charlie-r5)

| Student | Status | PR | Notes |
|---------|--------|----|-------|
| charlie5-alphonse | idle → assigning | — | round-1 loss-function axis |
| charlie5-askeladd | idle → assigning | — | round-1 throughput axis |
| charlie5-edward | idle → assigning | — | round-1 FFN axis |
| charlie5-fern | idle → assigning | — | round-1 Fourier axis |
| charlie5-frieren | idle → assigning | — | round-1 surf_weight axis |
| charlie5-nezuko | idle → assigning | — | round-1 slice_num axis |
| charlie5-tanjiro | idle → assigning | — | round-1 n_layers axis |
| charlie5-thorfinn | idle → assigning | — | round-1 n_head axis |
