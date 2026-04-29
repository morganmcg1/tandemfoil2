# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-29 05:30
- **Advisor branch:** `icml-appendix-willow-pai2e-r5`
- **Track tag:** `willow-pai2e-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r5`
- **Most recent direction from human team:** none yet (no human GitHub issues open).

## Research target

Beat the Transolver baseline on TandemFoilSet. Primary ranking metric is
`val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the four
validation splits) with `test_avg/mae_surf_p` as the paper-facing decision
metric. Baseline config is `n_hidden=128, n_layers=5, n_head=4, slice_num=64,
mlp_ratio=2`, AdamW + CosineAnnealingLR, `lr=5e-4`, `surf_weight=10.0`,
batch=4, 50 epochs, vol+10·surf MSE in normalized space.

The four validation tracks each probe a different generalization axis:
- `val_single_in_dist` — random holdout from single-foil (sanity).
- `val_geom_camber_rc` — unseen front-foil camber raceCar tandem M=6-8.
- `val_geom_camber_cruise` — unseen front-foil camber cruise tandem M=2-4.
- `val_re_rand` — stratified Re holdout across all tandem domains.

Per programme contract, surface pressure on the held-out camber/Re splits is
where the paper-facing numbers live. Per-sample y std varies by an order of
magnitude even inside one domain, so high-Re samples drive the extremes.

## Wave-1 + Wave-2 status

| Student | PR | Status | Result |
|---------|----|--------|--------|
| alphonse | #732 | Closed | val_avg=154.95 ref; NaN test; 6/50 epochs |
| alphonse | #796 | Closed | FiLM-Re ineffective: val_avg=135.51 vs 135.35 control; test +4.5% |
| alphonse | #896 | **Closed** | Per-sample y-norm on Huber+sw=3: val_avg=100.185 (−1.36%, noise-level), test_avg=90.686 (+0.85%). Redistributive not Pareto. Mechanism substitutes for Huber+sw=3 rather than stacking. |
| alphonse | #980 | **Closed** | Boundary-layer-weighted vol-loss: focus=5.0 best at val=93.88 (−3.1%) but test tied (+1.0%); split-trade *inverted* from #885 (sid/rc improve, cruise/re_rand regress). Same Pareto wall as #953 — vol-loss-shape lever exhausted. |
| alphonse | #1045 | **WIP** | Per-channel sigma normalization: divide huber_err by sigma_per[b, c] (per-sample, per-channel std). Decouples pressure from velocity supervision. Refines #896 mechanism after diagnosis: sigma was pressure-dominated. |
| askeladd | #733 | Closed | val_avg=151.50; throughput cost decisive |
| askeladd | #811 | **Merged** | val_avg=127.402; BF16 1.20× speedup |
| askeladd | #848 | Closed | bs={8,10}: regressed; bs=12 OOM; `add_derived_features` loop bottleneck |
| askeladd | #885 | **Merged** | δ=0.1 + sw=3: **val_avg=96.866 (−4.62%), test_avg=87.348 (−2.86%)**. δ=0.3+sw=3 collapsed to noise (mechanism overlap with sw=3). δ=0.1 attacks volume residual heavy tail. Default huber_delta now 0.1. W&B: `nffbil1x`. |
| edward | #734 | Closed | sw=10 wins; sw=50/100 regress |
| edward | #850 | **Merged** | sw=3 + Huber: **val_avg=101.563 (−8.17%), test_avg=89.918 (−11.24%)**. All 4 splits improved. Default surf_weight changed to 3.0 in Config. W&B: `6rh7dzkx`. |
| edward | #953 | **Closed** | sw=0.5 won val_avg=99.185 (-2.34%) but test essentially tied at 90.293 (+0.42%); sw=1, sw=2 much worse (non-monotone). Split-trade in test (sid+rc regress, cruise+re_rand improve). sw-tuning lever exhausted. |
| edward | #1019 | **WIP** | Loss-weighted hard-negative sampling — per-sample EMA loss → re-weighted sampler each epoch. Sweep alpha ∈ {0.5, 1.0} with floor=0.1 + control. Mechanistically distinct from sw and per-sample-norm. |
| fern | #737 | **Merged** | val_avg=127.87; warmup+cosine |
| fern | #809 | **WIP** | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | **Merged** | Huber d=1.0: **val_avg=110.594 (−13.2%)**, test_avg=101.299 (−12.8%); new best. All 4 test splits finite. |
| frieren | #915 | **Closed** | PhysicsAttention padding mask — cruise improved −14% test (as predicted) but rc regressed +30.8% test; net test +3.3% worse. Binary post-softmax mask disrupts attention on dense-mesh rc geometries. |
| frieren | #943 | **WIP (rebase)** | Per-channel surf-weight: p=20/vel=10 won val=107.5/test=98.4 vs stale baseline but regressed vs current sw=3 (+5.9% val, +9.5% test). Sent back for rebase + anchored sweep with vel_surf=3 fixed, sweep p_surf ∈ {3 control, 10, 20}. |
| nezuko | #742 | Closed | dropout=0.1 regresses 12.4%; undertrained model has no overfitting to regularize |
| nezuko | #878 | Closed | DropPath p=0.1 neutral on val_avg (+0.32, within seed noise) and +3% per-step overhead |
| nezuko | #923 | **Merged** | Vectorized data prep: bit-exact, neutral throughput (−1.6% noise). Hypothesis refuted: CPU syncs are not the bottleneck. Model forward+backward = 91% of epoch time. Bottleneck map established. |
| nezuko | #986 | **WIP (rebase)** | torch.compile(dynamic=True): 1.77× speedup confirmed (29 vs 17 epochs). Quality verification ran on stale δ=1.0 baseline — Run B val=111.96 vs Run A val=108.31 *despite more epochs* is suspicious. Sent back for rebase + re-verify on δ=0.1 baseline. |
| tanjiro | #745 | **WIP (rebase)** | Sent back for Option 3 capacity-matched heads on rebased baseline |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | **WIP (rebase)** | EMA d=0.995: val_avg=89.872 (-18.7%), test_avg=79.254 (-21.8%) on stale sw=10. Sent back for rebase + decisive verify on sw=3 baseline (902-line PR with merge conflicts). |

**Current best val_avg/mae_surf_p (merged):** 96.866 (askeladd #885, run `nffbil1x`).
**Current best test_avg/mae_surf_p (merged):** 87.348 (askeladd #885, run `nffbil1x`).

**Six compounding wins stacked:**
1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → val_avg=110.594
5. Lower surf_weight=3 (#850) → val_avg=101.563
6. **Huber δ=0.1 stacked on sw=3 (#885) → val_avg=96.866, test_avg=87.348**

**[PENDING — REBASE FOR STACKING/VERIFICATION TEST]**
- **EMA post-warmup-init d=0.995 (#810, thorfinn)** — won val=89.87 / test=79.25 on stale sw=10
  baseline (-18.7% val, -21.8% test). Mechanism is orthogonal to surf_weight, so should still win
  vs new δ=0.1 baseline (96.87/87.35). Sent back for rebase + decisive d=0.995 verification.
  Note: rebase target moved twice now (sw=3 → δ=0.1) — implementation must work with new defaults.
  Strong candidate 7th compounding win — implementation is clean (deferred init after warmup,
  dual-flavor live/ema checkpointing).
- **Per-channel surface loss (#943, frieren)** — anchored sweep with vel_surf=3 fixed, p_surf ∈ {3, 10, 20}
  pending. Stale-baseline numbers showed promise on cruise/re_rand. Need to verify against new
  baseline (96.87/87.35).

**All 8 GPUs in use:** alphonse #1045 (per-channel sigma normalization — refines #896 after
sigma-was-pressure-dominated diagnosis), askeladd → δ-floor-sweep below 0.1, edward #1019
(loss-weighted hard-negative sampling — per-sample EMA loss → resampling), fern #809
(schedule-budget), frieren #943 (per-channel surf rebase + anchored p_surf sweep with vel_surf=3),
nezuko #986 (torch.compile rebase + re-verify on δ=0.1 baseline), tanjiro #745 (heads Option 3
rebase), thorfinn #810 (EMA rebase + δ=0.1 verify).

## Current research themes

1. **Loss-shape (Huber δ) is the strongest single lever found.** Now stacked: δ=0.1 + sw=3 wins
   val=96.866, test=87.348 (#885 merged). Trend not yet bottomed out — askeladd's analysis suggests
   δ=0.05 or lower may continue to improve. Note split-trade pattern: smaller δ helps cruise/re_rand
   but hurts sid/rc — same pattern as sw=0.5 (#953). Suggests fundamental architectural trade.
2. **Per-sample y-norm refined → per-channel sigma (#1045).** Original #896 closed (substitute for
   Huber+sw=3, not complement). Boundary-layer-weighted vol-loss (#980) closed (inverse split-trade
   pattern, exhausts vol-loss-shape lever). Per-channel sigma normalization (#1045 alphonse) is a
   precise mechanism refinement — diagnosed that #896's sigma was a single scalar dominated by
   pressure; per-channel decouples Ux/Uy/p supervision. Predicted to mitigate the rc regression
   that killed #896.
3. **Loss-balance (sw) lever exhausted.** Edward #953 sweep closed: sw=0.5 wins val (-2.34%) but
   test essentially tied (+0.42%); sw=1, sw=2 much worse. Non-monotone curve + split-trade pattern
   (sid+rc regress, cruise+re_rand improve at sw=0.5) suggests sw=0.5 is a different regime, not a
   smooth extension of the volume-driven mechanism. sw=3 is a stable, defensible default.
4. **Per-channel surface loss weighting (#943, frieren)** — first sweep was confounded by stale
   vel_surf_weight=10. Both runs regressed vs current sw=3 baseline. Sent back for anchored sweep:
   vel_surf=3 fixed (matches current), sweep p_surf ∈ {3 control, 10, 20}. Cruise hint from Run 2
   (val=79.02 vs current sw=3 cruise=82.16, −3.8%) suggests pressure-boost may be real on OOD.
5. **Throughput: torch.compile delivers 1.77× speedup (#986, ran on stale baseline).** Confirmed
   by nezuko: 29 epochs in 30-min budget vs 17 eager. Quality A/B inconclusive on δ=1.0 baseline
   (Run B regressed slightly despite extra epochs — possibly bf16+dynamic recompile noise). Sent
   back for re-verify on current δ=0.1 baseline. If the compile-on quality matches eager on δ=0.1,
   this is a pure throughput stack — composes with everything.
6. **PhysicsAttention padding mask (Wave 3):** soft learnable gate — pending other wins first.

## Open questions

- ~~Where does the sw floor sit?~~ **Resolved by #953 (closed):** sw=0.5 lowers val 2.3% but test
  ties (+0.42%) with split-trade (sid/rc regress, cruise/re_rand improve). sw=3 stays as default.
- ~~Does per-sample y-norm (#896) still win when stacked on Huber + sw=3?~~ **Resolved by #896
  (closed):** No — net substitute for Huber+sw=3, not a complement. **Refined hypothesis active
  (#1045):** does per-CHANNEL sigma normalization win? Mechanism: #896's sigma was a single scalar
  dominated by pressure (raw_y_std for p is 1-100; for Ux/Uy is 0.1-10). Per-channel sigma_per[b, c]
  decouples them. Predicted: cruise/re_rand wins persist (channel decoupling preserves them); rc
  regression mitigated (velocity supervision restored).
- What is the optimal Huber δ? Askeladd #885 sweep {0.3, 0.5, 1.0, 2.0} will answer this. With
  sw=3 now the default, the outlier distribution seen by the loss is different — optimal δ may shift.
- Does frieren's per-channel split (#943) win once anchored to sw=3 baseline? First sweep (vel=10)
  regressed because raising velocity surface weight above current sw=3 hurts. Anchored sweep
  (vel=3 fixed, p ∈ {3, 10, 20}) directly tests whether pressure-only boost helps OOD.
- Does EMA (#810, d=0.995) still win on the new sw=3 baseline? Stale-baseline numbers were
  exceptional (val −18.7%, test −21.8%). EMA mechanism (logit averaging across last ~2k steps
  with d=0.995) should be orthogonal to loss-weighting, so the gain should hold. Verification run
  pending after rebase.

## Potential next research directions (Wave 3+)

Prioritized given current insights:

- ~~Per-sample-norm + Huber stacking~~ closed; **per-channel sigma (#1045) is active**.
- **Linearly-scaled-LR + bs=8** (after #923 lands). With `lr=2e-3, bs=8` (linear scaling rule)
  and gradient accumulation guard, retries batch-size scaling correctly.
- **Soft attention gate for padding** — `sigmoid(learned_gate(x_node))` multiplied into slice
  weights rather than hard 0/1 mask. Captures cruise benefit without rc regression.
- **`torch.compile` on model forward.** 1.2-1.5× speedup candidate but requires care with
  dynamic mesh sizes.
- **Capacity scaling to n_hidden=192** — with 17 epochs available in BF16+Huber, ~1.5M params
  might fit. Particularly interesting for cruise/rc splits.
- **Hard negative mining / Re-weighted sampler** — `WeightedRandomSampler` with weights ∝ Re or
  per-sample loss magnitude. Direct attack on Re-imbalance (alternative to per-sample-norm).
- **Architecture pivot** if tuning saturates: GINOs/FNOs (spectral), MeshGraphNets (explicit
  edges), equivariant attention, neural operator transformers.

Full literature bank in `research/RESEARCH_IDEAS_2026-04-28_19:30.md`.
