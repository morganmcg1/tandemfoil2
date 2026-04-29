# SENPAI Research State — willow-pai2e-r5

- **Last updated:** 2026-04-29 03:05
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
| alphonse | #980 | **WIP** | Boundary-layer-weighted volume loss — distance-decaying weight on volume nodes using x_aug[:,:,24] (dist_to_surface). Sweep boundary_focus ∈ {0, 1.0, 5.0}. Distinct mechanism. |
| askeladd | #733 | Closed | val_avg=151.50; throughput cost decisive |
| askeladd | #811 | **Merged** | val_avg=127.402; BF16 1.20× speedup |
| askeladd | #848 | Closed | bs={8,10}: regressed; bs=12 OOM; `add_derived_features` loop bottleneck |
| askeladd | #885 | **WIP (rebase)** | δ sweep done on sw=10: **δ=0.3 wins** (val=97.96, test=87.78 — beats current best 101.56/89.92 alone). PR conflicts with merged Huber #739 + sw=3 #850. Sent back for rebase + decisive δ=0.3+sw=3 stacking run + δ=0.1 trend test. |
| edward | #734 | Closed | sw=10 wins; sw=50/100 regress |
| edward | #850 | **Merged** | sw=3 + Huber: **val_avg=101.563 (−8.17%), test_avg=89.918 (−11.24%)**. All 4 splits improved. Default surf_weight changed to 3.0 in Config. W&B: `6rh7dzkx`. |
| edward | #953 | **WIP** | Sweep surf_weight ∈ {0.5, 1.0, 2.0} — find floor below sw=3. Val still descending at timeout in #850; lower sw may continue trend. |
| fern | #737 | **Merged** | val_avg=127.87; warmup+cosine |
| fern | #809 | **WIP** | Schedule sized to budget (epochs=14, warmup=2) |
| frieren | #739 | **Merged** | Huber d=1.0: **val_avg=110.594 (−13.2%)**, test_avg=101.299 (−12.8%); new best. All 4 test splits finite. |
| frieren | #915 | **Closed** | PhysicsAttention padding mask — cruise improved −14% test (as predicted) but rc regressed +30.8% test; net test +3.3% worse. Binary post-softmax mask disrupts attention on dense-mesh rc geometries. |
| frieren | #943 | **WIP** | Per-channel surface loss weights: p_surf_weight ∈ {3, 20} vs vel_surf_weight=10 |
| nezuko | #742 | Closed | dropout=0.1 regresses 12.4%; undertrained model has no overfitting to regularize |
| nezuko | #878 | Closed | DropPath p=0.1 neutral on val_avg (+0.32, within seed noise) and +3% per-step overhead |
| nezuko | #923 | **Merged** | Vectorized data prep: bit-exact, neutral throughput (−1.6% noise). Hypothesis refuted: CPU syncs are not the bottleneck. Model forward+backward = 91% of epoch time. Bottleneck map established. |
| nezuko | #986 | **WIP** | torch.compile(dynamic=True) on model forward — targets 1.2-1.5× speedup on the actual bottleneck (model FLOPs). A/B with control; mode="reduce-overhead". |
| tanjiro | #745 | **WIP (rebase)** | Sent back for Option 3 capacity-matched heads on rebased baseline |
| thorfinn | #763 | **Merged** | val_avg=141.42; features + NaN-safe eval |
| thorfinn | #810 | **WIP (rebase)** | EMA post-warmup-init + decay sweep on BF16 baseline |

**Current best val_avg/mae_surf_p (merged):** 101.563 (edward #850, run `6rh7dzkx`).
**Current best test_avg/mae_surf_p (merged):** 89.918 (edward #850, run `6rh7dzkx`).

**Five compounding wins stacked:**
1. Distance features + NaN-safe eval (#763) → val_avg=141.42
2. Warmup+cosine LR (#737) → val_avg=127.87
3. BF16 mixed precision (#811) → val_avg=127.40
4. Huber loss δ=1.0 (#739) → val_avg=110.594
5. Lower surf_weight=3 (#850) → **val_avg=101.563, test_avg=89.918**

**[PENDING — REBASE FOR STACKING TEST]**
- **Huber δ=0.3 (#885, askeladd)** — δ=0.3 alone won val=97.96 / test=87.78 (beats current
  best 101.56/89.92 in absolute terms!), but on stale sw=10. Sent back for rebase + δ=0.3+sw=3
  decisive stacking run. If δ=0.3+sw=3 wins → merge as 6th compounding win.

**All 8 GPUs in use:** alphonse #980 (boundary-layer vol-loss sweep), askeladd #885 (δ=0.3
rebase + stacking test), edward #953 (sw-below-3 sweep), fern #809 (schedule-budget), frieren
#943 (per-channel-surf-weight), nezuko #986 (torch.compile A/B), tanjiro #745 (heads Option 3
rebase), thorfinn #810 (EMA rebase).

## Current research themes

1. **Loss-shape (Huber δ) is the strongest single lever found.** δ=0.3 alone (val=97.96) beats
   current baseline (101.56) on stale sw=10. Askeladd's rebase + δ=0.3+sw=3 stacking run (#885)
   is the highest-priority in-flight experiment. Potential 6th compounding win.
2. **Per-sample y-norm (alphonse) is exhausted as a standalone lever.** It's a substitute for
   Huber+sw=3 rather than a complement. Closed #896. Assigned #980 (boundary-layer vol-loss) —
   distinct mechanism using dist_to_surface feature to weight near-wall volume nodes more.
3. **Loss-balance (sw) floor still unknown.** Edward #953 sweeps sw ∈ {0.5, 1.0, 2.0}.
   The val curve was still descending at epoch 17 with sw=3 — lower sw may continue the trend.
4. **Per-channel surface loss weighting (#943, frieren)** — tests whether splitting surface loss
   into channel-specific terms (p vs velocity) improves over uniform sw=3. Run 1 (p=3, vel=10)
   effectively raises velocity supervision above baseline; Run 2 (p=20, vel=10) boosts pressure.
5. **Throughput: model FLOPs = actual bottleneck.** #923 (merged) established the cost map:
   model forward+backward = 91% of epoch time, data prep = 9%. torch.compile is the correct lever
   (#986, nezuko). Expected: 19-22 effective epochs in budget (vs 17 today) if 1.2-1.5× speedup lands.
6. **PhysicsAttention padding mask (Wave 3):** soft learnable gate — pending other wins first.

## Open questions

- Where does the sw floor sit? sw=3 is still descending at timeout; sw=1 or sw=2 may be better.
  sw=0.5 (near-pure volume) may degrade — watching for a reversal in edward's sweep (#953).
- Does per-sample y-norm (#896) still win when stacked on top of Huber + sw=3? The MSE+sw=10 run
  won by −4.7% val / −7.4% test, but the new baseline is already much lower. With sw=3 already
  exploiting volume signal, the per-sample-norm mechanism may partially overlap.
- What is the optimal Huber δ? Askeladd #885 sweep {0.3, 0.5, 1.0, 2.0} will answer this. With
  sw=3 now the default, the outlier distribution seen by the loss is different — optimal δ may shift.
- Does frieren's per-channel split (#943) reveal that velocity surface supervision (now at w=3)
  was the hidden bottleneck? Run 1 (p=3, vel=10) effectively raises velocity weight above sw=3
  baseline — if it wins, velocity surface weight was too low.
- Does EMA (#810) still win on the new sw=3 baseline?

## Potential next research directions (Wave 3+)

Prioritized given current insights:

- **Per-sample-norm + Huber confirmed stacking** → then per-channel sigma (normalize each of Ux,
  Uy, p separately, so pressure doesn't dominate sigma_per scalar).
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
