# SENPAI Research Results — `icml-appendix-willow-pai2d-r3`

## 2026-04-28 10:00 — PR #577: Surface-only auxiliary pressure head — **CLOSED (degenerate by mechanism check)**

- Branch: `willowpai2d3-tanjiro/surface-only-aux-head` (deleted post-close)
- **Hypothesis:** A small dedicated MLP on the backbone's per-node features predicts surface pressure with its own loss term, decoupling capacity allocation from loss balancing. Predicted Δ: −5 to −15%.

### Results (group `surface-aux-head`, on the post-EMA baseline — predates #294 and #409)

| Run | aux | weight | depth | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B |
|---|---|---:|---:|---:|---:|---|
| `aux-off-ctrl` (paired) | off | — | — | **118.57** | 106.61 | `a42f7a10` |
| `aux-off-ctrl` (2nd seed) | off | — | — | 119.59 | 108.52 | `r6s0l4i5` |
| `aux-w0.5-d2` (best) | on | 0.5 | 2 | 118.03 | 106.00 | `iihvpg9s` |
| `aux-w1.0-d2` | on | 1.0 | 2 | 119.91 | 107.66 | `umwdew6e` |
| `aux-w2.0-d2` | on | 2.0 | 2 | 120.30 | 109.00 | `xg0ff99u` |
| `aux-w1.0-d1` | on | 1.0 | 1 | 123.10 | 108.48 | `0qp9qmvh` |

### Decision: **CLOSED** (degenerate; mechanism check failed)

Within-sweep best delta = −0.55 MAE val, **inside the inter-control spread of 1.02 MAE** between two no-aux seeds. Indistinguishable from luck.

**Critical mechanism check FAILED:** `mae_aux_only ≈ mae_backbone_only` across every variant and every split (aux is consistently +0.5 to +1 MAE *worse*). The aux head reads from the same `ln_3(fx)` features the backbone's `mlp2` already projects from — **no information advantage**. The aux head is a parallel degenerate copy of the backbone's own pressure prediction.

**Sub-trend confirms degeneracy:** lower aux loss weight monotonically better on val (0.5 < 1.0 < 2.0 at depth=2) — exactly what you'd expect if the aux head adds no signal but its loss term *displaces* backbone gradient capacity.

The right pivot per tanjiro's contingency is **separate features** (concatenate post-attention features from earlier blocks before the aux head) — but this is significant architectural surgery, better suited to edward's lane. Tanjiro reassigned to **arc-length surface smoothness regularization** (PR #707) — a fresh mechanism (spatial smoothness, physics-motivated) distinct from the residual-reweighting territory we've now established as bounded.

**On the `ema_decay` bug-fix bundled in this PR:** rejected the change. Tanjiro's historical reasoning is correct (PR #410's winner `22a7k787` ran at 0.99), but the post-#294 and post-#409 winners both ran at 0.999 — that's the current operational baseline. Flipping the default to 0.99 would diverge from current operational reality. Better to leave at 0.999 and document.

## 2026-04-28 09:45 — PR #482: Multi-seed baseline + deterministic seeding — **SENT BACK FOR REBASE (infrastructure ready; multi-seed needs current baseline)**

- Branch: `willowpai2d3-thorfinn/multi-seed-baseline`
- **Hypothesis:** Add deterministic seeding to `train.py` and characterize the merged-baseline variance with 5 seeds. Predicted Δ on val_avg = 0% (infrastructure PR).

### Determinism: rock-solid

Two separate seed-0 runs produced **bit-identical val_avg/mae_surf_p across 8 epochs** (different GPUs, hours apart). Four-line block (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`) placed right after `cfg = sp.parse(Config)`. `cudnn.deterministic` deliberately not enabled (perf cost).

### Multi-seed study (5 seeds on the post-EMA baseline — pre-#294, pre-#409)

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| **best_val_avg/mae_surf_p** | **136.8210** | **8.9868** | 128.2880 | 148.8332 |
| test_avg/mae_surf_p | 123.40 | 7.91 | 115.07 | 135.14 |

**Per-split variance (val):** `val_single_in_dist` 30%, `val_geom_camber_rc` 25%, `val_geom_camber_cruise` 17%, `val_re_rand` 14% (relative std). Aggregated val_avg has ~6.6% relative std because splits aren't perfectly correlated.

**Reference points vs the multi-seed distribution:**

| reference | val_avg | z-score |
|---|---:|---:|
| PR #320's `w3mjq2ua` (recorded baseline) | 115.84 | **−2.33 σ** |
| Thorfinn's r2 mlp_ratio control `xklvptvl` | 140.70 | +0.43 σ |

**PR #320's 115.84 confirmed as a lucky outlier** (z=−2.33, ~1% probability under a Gaussian assumption). Thorfinn's r2 control was completely typical at z=+0.43.

### Decision: **REQUEST CHANGES (rebase + redo multi-seed at current merged baseline)**

The seeding infrastructure is the deliverable — small, focused, well-tested. **But** the 5-seed study characterizes an *obsolete* baseline (post-EMA-only, mean 136.82) rather than the current operational baseline (post-#409 OneCycle + EMA + L1, single-seed 87.74). For the multi-seed mean ± std to be useful as the merge bar, it needs to be at the current config.

Additionally, the PR has merge conflicts (DIRTY) since PR #294 and #409 both modified Config.

Sent back for rebase + 5-seed redo on the current merged baseline. Smoke test first (two `--seed 0 --debug` runs) to confirm seeding survives the rebase, then 5-seed production sweep. Total ~2.5h GPU.

The historical 5-seed numbers on the post-EMA baseline remain useful as a reference for past merge-decision recalibration — kept in the report rather than discarded.

## 2026-04-28 09:30 — PR #609: Focal-L1 (per-node residual reweighting on top of L1) — **CLOSED (clean negative result + reusable team insight)**

- Branch: `willowpai2d3-alphonse/focal-l1-surface` (deleted post-close)
- **Hypothesis:** Per-node focal reweighting (γ>0) on top of L1 should up-weight hard residuals and improve fit on the worst nodes. Predicted Δ: −3 to −8% (borderline at noise floor).

### Sweep results (group `focal-l1-surface`, on the post-#294 baseline)

| γ | val_avg/mae_surf_p | Δ vs ctrl | test_avg/mae_surf_p | W&B run |
|---|---:|---:|---:|---|
| **0 (ctrl)** | **98.19** | — | **87.23** | `9l47599l` |
| 0.5 | 102.71 | **+4.5** | 90.84 | `eoiyqw9p` |
| 1.0 | 114.37 | **+16.2** | 102.60 | `pyd6uqt6` |
| 2.0 | 212.08 | **+113.9** | 196.81 | `84n0uv2y` |

### Decision: **CLOSED + reusable team insight**

Within-sweep monotonic NEGATIVE result. The γ=0 control reproduced the merged baseline (98.19 vs 94.89, within seed noise) and the focal sweep monotonically degraded both val and test. Mechanism is the cleanest possible: **L1 was specifically chosen to NOT amplify large residuals; focal does the opposite. They directly cancel.**

The focal_w distribution at γ=2.0 (max=973× the mean, p99/p50=54.4×) confirms the lever is doing what the math says — dumping all gradient into a handful of extreme residuals. No NaN despite the huge weights because L1's gradient is `sign(r)` (bounded magnitude regardless of weight) — the lever doesn't destabilize, just allocates capacity wrong.

### **Reusable team insight (research-direction guideline)**

This generalizes tanjiro's per-sample inverse-std finding (PR #508) to the per-node level:

> **Residual-magnitude reweighting on the surface objective is the wrong axis once the loss is already the metric.**

Both PR #508 (per-sample) and PR #609 (per-node) confirmed mechanism but failed to compound — both because reweighting by residual magnitude *fights* the metric we're optimizing for (MAE-aligned L1 already gives equal weight to all residuals by design). Future loss-formulation experiments should attack different mechanisms (smoothness penalties, derived-physics aux losses, robust-loss bounding) rather than residual-magnitude reweighting.

**Per-split prediction directional check held at γ=0.5** (`val_geom_camber_rc` was the least-hurt split, +2.88 — exactly as predicted *if* hard-node focus were helping), but the magnitude is wrong-sign.

Alphonse reassigned to **PR #691: saturated L1** — the *opposite-sign* lever, bounding pathological residuals' contribution rather than amplifying them. Predicted small effect (−1 to −4%).

## 2026-04-28 09:20 — PR #618: UNet-style skip from preprocess into last block — **SENT BACK (negative result + clean diagnostic, iterate to ReZero-gated)**

- Branch: `willowpai2d3-edward/unet-skip-preprocess`
- **Hypothesis:** Skip connection from post-preprocess features into last block's residual stream addresses information bottleneck through deep slice-attention. Predicted Δ: −3 to −10%.

### Sweep results (group `unet-skip-preprocess`, on the post-#294 baseline — predates OneCycle merge)

| run | unet_skip | dropout | val_avg | test_avg | Δ val vs ctrl | skip-to-deep ratio | W&B |
|---|---|---|---:|---:|---:|---:|---|
| `unet-off-ctrl` | False | – | **94.36** | **83.71** | – | – | `y4d9aysm` |
| `unet-skip-d0` | True | 0.0 | 96.04 | 85.32 | **+1.68** | 0.499 | `mx3q0xhe` |
| `unet-skip-d0.1` | True | 0.1 | 98.04 | 87.58 | **+3.68** | 0.817 | `qqi08dix` |

### Decision: **REQUEST CHANGES (iterate to ReZero-gated skip)**

Negative result on simple concat+linear UNet skip. Edward's diagnostic is rigorous: the skip path is *actively used* (`skip_to_deep_ratio` ~0.5–0.8) but the random-init `skip_fuse: Linear(2H, H)` perturbs the well-trained deep path at step 0. The optimizer can't unlearn this perturbation in 14 epochs. Per-split deltas hit **`val_geom_camber_rc` worst** (the predicted-best-helped split) — the cleanest possible refutation of "lever doesn't help" interpretation; instead it says "the implementation initialization is wrong."

The fix is edward's own follow-up #1 (and the researcher-agent's pre-implementation flag): **ReZero-gated skip** — `fx = fx + α·skip_proj(fx_skip)` with `α ∈ R` learnable, init **α=0**. At step 0 the model is identical to the control; the optimizer can lift α as it discovers value. Eliminates the init perturbation while preserving the lever.

Also: edward flagged a second real bug — `train.py` line 451 has `huber_delta: float = 1.0`, but the merged baseline uses `huber_delta=0` (pure L1). All recent PRs had to pass `--huber_delta 0` explicitly. **Bundling the one-line default fix in the next iteration.**

Edward stays on this lever for one focused iteration: control + ReZero-gated skip (drop the `dropout=0.1` variant given the monotonic-bad result). Two runs, decision tree spelled out — either α lifts and val_avg improves (ship), or α stays near zero or doesn't help (close and park, pivot to deeper output head MLP).

## 2026-04-28 08:53 — PR #409 (round 3): OneCycleLR rebased onto post-EMA+L1 baseline — **MERGED**

- Branch: `willowpai2d3-frieren/onecycle-lr`
- **Hypothesis (round 3):** OneCycle 2e-3 + EMA + L1 should win cleanly over the warmup-cosine + EMA + L1 control on the post-#294 baseline.

### Results (group `onecycle-lr-r3`, on the post-EMA+L1 baseline)

| Schedule | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | EMA−live | W&B run |
|---|---:|---:|---:|---:|---|
| **onecycle 2e-3 + EMA + L1** | **14** | **87.7443** | **78.2370** | **+5.69 (EMA WORSE)** | `4fas292o` |
| warmup-cosine + EMA + L1 (control) | 14 | 95.2555 | 84.8760 | −7.26 (EMA better) | `of3vm0qy` |

### Decision: **MERGED**

**Within-sweep delta = −7.51 MAE on val_avg, −6.64 MAE on test_avg.** OneCycle wins on all 4 test splits. **vs prior merged baseline (PR #294, 94.89/83.94):** −7.15 MAE val, −5.70 MAE test. Below the ~25 MAE seed-noise floor in absolute terms but consistent with R2's pre-EMA −33 MAE delta — the schedule effect shrunk because OneCycle's cool-down and EMA+L1 partially overlap (both contribute to a smoother converged state). Still net-positive.

**Critical mechanistic finding: EMA-vs-live sign-flip across schedules.**

- OneCycle: live (82.06) BEATS EMA-averaged (87.74) by +5.69 MAE
- warmup-cosine: EMA (95.26) beats live (102.51) by −7.26 MAE

A 12.95-point swing in EMA's relative value driven purely by LR schedule. Mechanism: EMA averages out training noise — useful when the schedule leaves the model still noisy at end-of-training (warmup-cosine flat at 85% peak), counterproductive when the schedule converges (OneCycle cool-down to 1.5% peak). The within-run live-vs-EMA diagnostic is seed-independent, making this a robust mechanistic claim.

**Implication:** EMA may now be sub-optimal in baseline. If `use_ema=False` with OneCycle gives val_avg ≈ 82 (per the live diagnostic), that's another −5.7 MAE on top of the new 87.74. **Frieren reassigned to PR #671 (`use_ema=False` confirmation under OneCycle).**

**New baseline:** OneCycle 2e-3 with `pct_start=0.1, total_epochs=15` is the default schedule. Warmup-cosine path remains available behind `--schedule warmup-cosine`.

## 2026-04-28 07:05 — PR #420 (round 2): Random Fourier features — **CLOSED (substitutes with EMA, doesn't compound)**

- Branch: `willowpai2d3-edward/fourier-features-coords` (deleted post-close)
- **Hypothesis (round 2):** Single-σ Fourier (σ=0.5) + EMA + multi-scale escalation should give −15 to −25% on val_avg, clearly clearing noise floor.

### Sweep results (group `fourier-features-coords-r2`, on the post-EMA baseline — predates PR #294 L1 merge)

| Variant | best_ep | val_avg/mae_surf_p | test_avg/mae_surf_p | params | W&B |
|---|---:|---:|---:|---:|---|
| **Run 1: Fourier σ=0.5 + EMA** | 13 | **116.66** | **105.52** | 0.69M | `lhoidw09` |
| Run 2: multi-scale + concat_raw + EMA | 14 | 128.56 | 117.87 | 0.76M | `9kuqs8m5` |

### Decision: **CLOSED + park**

Edward executed the two-run focused experiment cleanly. **Run 1 was direction-positive on every val and test split** (largest gain on `geom_camber_cruise`, ~−8%) but didn't clear the 110 bar; **Run 2 actively regressed** (single_in_dist +27 MAE while modestly helping camber_cruise).

Edward's key mechanistic insight: **EMA and Fourier features substitute, not compound.** Both regularize the same spectral-fit problem (EMA averages weight oscillations across steps; Fourier replaces the linear coord projection that was the spectral bottleneck). Within-experiment effects: EMA alone −21.71 MAE, Fourier σ=0.5 alone −18.3 MAE, both stacked only −4.78 MAE. That's the substitution signature.

**Run 2 also exposed a stability problem with multi-scale + concat_raw**: input dim blew up ~17× (24 → 386 spatial dims), the live training path was unstable (epoch 7→8 jump of +30 MAE), and EMA-vs-live diagnostic showed −30.88 MAE (EMA was rescuing live during instability). At the 30-min wall-clock budget the wider input layer didn't get enough optimization steps to stabilize.

**The bar moved further**: PR #294 (pure L1) merged at 06:47 UTC (concurrent with edward's runs), taking the merged baseline to 94.89/83.94. Run 1's 116.66 is now 21.77 MAE above the bar — well outside any reasonable noise envelope. Confirmed Fourier doesn't survive the L1 merge either.

**Bonus team-record contribution: edward flagged a real `ema_decay` doc/code mismatch.** BASELINE.md claimed PR #410 set `ema_decay=0.99` as the default, but `train.py` actually defaults to 0.999. PR #410's winner ran at 0.99 but the merged code committed 0.999, and all subsequent runs (alphonse's L1 winner, tanjiro's experiments) used 0.999. Practically the gap is small (1.76 MAE), but BASELINE.md should match the actual code default — fixed in commit `3033b5f` to reflect 0.999.

Edward reassigned to **PR #618 (UNet-style skip connection from preprocess into last block)** — a structurally different lever attacking information path length within the transformer rather than the input representation.

## 2026-04-28 06:47 — PR #294 (round 2): Pure L1 surface loss — **MERGED (strongest result yet)**

- Branch: `willowpai2d3-alphonse/huber-loss-surf-p`
- **Hypothesis (round 2):** Push the Huber δ all the way to 0 (pure L1 = MAE-in-normalized-space) on top of the merged warmup+EMA baseline. Predicted Δ: −3 to −8% on val_avg.

### Sweep results (group `huber-loss-surf-p-r2`, on the post-EMA baseline)

| huber_delta | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---:|:--:|---:|---:|---|
| **0 (pure L1)** | 14 | **94.89** | **83.94** | `1zpw3ts2` |
| 0.5 | 14 | 100.70 | 90.97 | `b8pvs5c9` |
| 1.0 | 14 | 103.65 | 93.22 | `lpb2xfkp` |
| 2.0 | 14 | 110.61 | 98.18 | `y7hajox7` |

### Decision: **MERGED**

**Within-sweep delta = −15.72 MAE on val_avg, −14.24 on test_avg** across the monotonic 4-point sweep. **vs the merged baseline (PR #410): −26.55 MAE val, −24.72 MAE test** — both comfortably above the ~25 MAE seed-noise floor. Clean monotonic trend (smaller δ → better), confirming the L1-aligns-with-MAE-metric story.

**Compounding analysis** (alphonse's key contribution): R1's OLD-config-Huber-δ=0.5 (106.36) → R2's NEW-config-pure-L1 (94.89) breaks down cleanly: −5.66 from baseline upgrade (NEW+Huber-0.5 lands at 100.70), −5.81 from δ shift (0.5 → 0 in R2). Loss-shape lever (Huber/L1 in loss landscape) and optimizer/EMA levers (warmup + weight averaging) are orthogonal mechanisms that stack additively — exactly the most-exciting outcome.

**Per-split breakdown** for the winner: biggest absolute gains on high-residual splits (`val_single_in_dist` 148.90 → 115.49, `val_geom_camber_rc` 130.69 → 107.52). All four splits improve cleanly. Mechanism check via `train/surf_huber_outlier_frac` confirms: at δ=0 every surface element is in the linear regime by construction (1.0); at δ=2.0 it stays near 0.

**New baseline:** `huber_delta=0` (pure L1) is the surface-loss default; volume stays MSE.

**Alphonse reassigned to PR #609** — focal-L1 (per-node residual reweighting on top of L1).

## 2026-04-28 05:50 — PR #508: Per-sample inverse-std weighting on surface loss — **CLOSED (mechanism real, magnitude bounded by dataset structure)**

- Branch: `willowpai2d3-tanjiro/per-sample-inverse-std-weighting` (deleted post-close)
- **Hypothesis:** Per-sample inverse-std-weighted surface loss to normalize gradient signal across samples. Predicted Δ: −5 to −15%.

### Sweep results (group `inv-std-weighting`, on the post-EMA baseline)

| Run | inv_std_weight | inv_std_eps | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---|---|---:|---:|---:|---|
| ctrl-no-invstd | False | — | 129.62 | 115.26 | `ohpxjdsv` |
| **invstd-eps1.0** | True | 1.0 | **125.52** | **113.56** | `d2hzjvca` |
| invstd-eps0.1 | True | 0.1 | 130.05 | 117.32 | `vas31tia` |

### Decision: **CLOSED** (mechanism confirmed; magnitude limited by dataset structure)

Within-sweep Δ = 4.53 MAE on val_avg, **well below the ~15-25 MAE noise floor**. Per-split confirms: cruise (lowest std) wins big (−12.5 MAE val, −8.9 MAE test); raceCar-rc (high std) regresses (+8.3 MAE val, +4.5 MAE test). **Mechanism is real but bounded.**

Two important diagnostic findings from tanjiro:

1. **The predicted 100× weight ratio failed (observed ~2×).** Computing std in `y_norm` space absorbs the per-sample variance into the global normalization (sd ≈ 1 by construction). My PR instructions had the math backwards; tanjiro caught it.

2. **The eps trend (1.0 → 0.1 = ratio 1.77× → 2.23×) shows bigger reweighting makes the aggregate WORSE.** Cruise gains scale with the ratio, but raceCar regression also scales. So even fixing the y_norm space issue (which would push ratio toward predicted 100×) wouldn't help — raceCar would regress catastrophically. The lever is fundamentally bounded by the dataset's natural domain structure.

Closed rather than rerun because the upper bound is established. Tanjiro reassigned to **PR #577 (surface-only auxiliary prediction head)** — a structurally different lever that decouples capacity allocation from loss balancing rather than rebalancing existing capacity.

## 2026-04-28 05:00 — PR #420: Random Fourier features for spatial coords — **SENT BACK (rebase + escalate to multi-scale)**

- Branch: `willowpai2d3-edward/fourier-features-coords`
- **Hypothesis:** Sinusoidal encoding of (x, z) coords addresses transformer spectral bias for sharp leading-edge pressure peaks. Predicted Δ: −3 to −10%.

### Sweep results (group `fourier-features-coords`, runs PRE-DATE the EMA merge — measured against pre-EMA controls)

| Variant | best_ep | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---|---:|---:|---:|---|
| no-fourier control r1 | 10 | 148.42 | 136.33 | `09ir2ii6` |
| no-fourier control r2 | 14 | 135.15 | 121.29 | `8wkfykai` |
| **fourier σ=0.5** | 14 | **123.48** | **107.38** | `d2dv9ppp` |
| fourier σ=1.0 | 10 | 128.78 | 115.83 | `kziehbre` |
| fourier σ=2.0 | 14 | 134.29 | 126.06 | `imzk20e7` |
| fourier σ=5.0 | 14 | 140.24 | 126.55 | `m0dbquea` |

### Decision: **REQUEST CHANGES (rebase + escalate to multi-scale Fourier)**

**The σ ordering is monotone** (5.0 > 2.0 > 1.0 > 0.5 — lower σ better, matching the spectral-bias prior). **Geom-camber splits show the largest improvement** (~−18% vs ~−15% on the in-distribution and Re splits), weakly confirming the geometry-extrapolation prior. **But the within-sweep effect at σ=0.5 (−12 to −25 MAE depending on which control we compare against) is borderline at the ~25 MAE seed-noise floor.** Edward's two control runs themselves differ by 13 MAE — a clean within-experiment measurement of the variance issue.

Two issues compound: (1) edward's branch was never rebased onto the post-EMA baseline (some runs were launched after PR #410 merged but still don't include EMA defaults), so the comparison is against an obsolete baseline; (2) single-σ at the noise floor is fragile.

Sent back with a focused two-run rebased experiment: (a) σ=0.5 with EMA on (sanity-check the lever stacks with EMA), and (b) **multi-scale Fourier** with `σ ∈ {0.25, 0.5, 1.0}` concatenated + raw coords concatenated — the strongest variant of the lever. Predicted multi-scale + EMA Δ: −15 to −25% on val_avg, which would clearly clear the noise floor.

If the two runs land at val_avg < 110, this becomes the next merge after frieren's OneCycle PR #409. If only the single-σ confirms but multi-scale doesn't add value, we still get a merge candidate. If neither beats 110, we close and park.

## 2026-04-28 04:50 — PR #409: OneCycleLR — **SENT BACK FOR REBASE (next merge candidate)**

- Branch: `willowpai2d3-frieren/onecycle-lr`
- **Hypothesis:** OneCycleLR's aggressive cool-down phase (LR ↓ 1–4 orders of magnitude in last 30%) finds a sharper minimum than `LinearLR + CosineAnnealingLR` does at our 30-min wall-clock budget (where cosine never anneals past ~85% peak).

### Sweep results (group `onecycle-lr`, runs PRE-DATE the EMA merge — measured against pre-EMA control)

| Schedule | peak_lr | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p | LR at exit | W&B run |
|---|---:|---:|---:|---:|---:|---|
| warmup-cosine (control) | 1e-3 | 13 | 140.42 | 128.85 | 8.5e-4 (85% peak) | `j9b9g70e` |
| onecycle | 1e-3 | 14 | 109.75 | 97.81 | 1e-5 (1.0% peak) | `gddb79de` |
| **onecycle** | **2e-3** | **14** | **107.20** | **95.92** | 3e-5 (1.5% peak) | **`m9jvp59q`** |
| onecycle | 3e-3 | 13 | 118.07 | 106.94 | 1.6e-4 (5.3% peak) | `itvy80t3` |

### Decision: **REQUEST CHANGES (rebase + single confirming run, this is the merge candidate)**

**Within-sweep delta = −33.22 MAE on val_avg, −32.93 on test_avg** — well above the seed-noise floor of ~25 MAE we surfaced via PR #323. Mechanism is confirmed: control LR stays flat at 85% peak (cosine never engages at 30-min cap), while OneCycle 2e-3 drops to 1.5% peak — exactly the sharper-minimum-annealing the hypothesis targeted. Test gain (-32.93 MAE) is even bigger than val gain (-33.22 MAE), indicating the cool-down is helping generalization, not just memorization.

Frieren also caught a subtle calibration bug: OneCycle's cool-down is fraction-of-`total_steps`-based, so with `epochs=50` but wall-clock at ~14 epochs, the cool-down barely fired in their initial run. Fix was a separate `onecycle_total_epochs=15` config field that aligns the schedule with reachable epochs while keeping `--epochs 50` for the wall-clock budget. Clean engineering.

### Why send back instead of merge

1. **Merge conflicts**: frieren's branch and the upstream EMA merge (PR #410) both modified `Config`; can't auto-resolve.
2. **Runs predate EMA**: the 107.20 number was measured WITHOUT EMA. I asked for one more confirming run on the post-EMA baseline so we know whether OneCycle + EMA compound. The within-sweep evidence is so strong (-33 MAE > 25 MAE noise) that the rebased re-run is essentially a sanity check, not a re-validation.

After the rebased confirming run lands, this becomes the merge that drops the baseline to ~95-100 val_avg.

## 2026-04-28 03:55 — PR #322 round 2: rebased channel-weighted-loss — **CLOSED (noise-bound at this budget)**

- Branch: `willowpai2d3-tanjiro/channel-weighted-loss` (deleted post-close)
- **Hypothesis (round 2):** confirm `surf_p_weight=3.0` lever stacks with the merged warmup baseline.

### Sweep results (group `channel-weighted-loss-r2`, on the post-warmup baseline)

| surf_p_weight | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p (4-split, NaN-clean) | W&B run |
|---:|:--:|---:|---:|:--|
| 1.0 (control) | 13 | 148.34 | 135.57 | `qhcn6jmu` |
| 2.0 | 12 | 147.20 | 137.67 | `8r4stm2j` |
| 3.0 (r1 winner) | 11 | 155.06 | 142.14 | `bb0us7f3` |
| **5.0** (r2 winner) | 13 | **136.51** | **122.82** | `x7vwby6d` |

### r1 vs r2 comparison (same code, different runs of same config)

| surf_p_weight | r1 best val | r2 best val | Δ |
|---:|---:|---:|---:|
| 1.0 (control) | 138.87 | 148.34 | +9.47 |
| 2.0 | 139.33 | 147.20 | +7.87 |
| **3.0** *(r1 winner)* | **126.18** | 155.06 | **+28.88** |
| **5.0** *(r2 winner)* | 142.29 | **136.51** | −5.78 |

### Decision: **CLOSED** (noise-bound; revisit post-multi-seed)

The r1 winner regressed by ~29 MAE in r2 on a single setting, and the r1 worst-case became the r2 winner — a clear seed-variance signature. **The lever effect (~10 MAE within-sweep) is below the noise floor at this 30-min budget (~15–30 MAE across seeds).** Tanjiro's framing nailed the diagnosis: "I cannot conclude either way from these data."

What's reassuring: **direction and magnitude are consistent across rounds** (−9.13% in r1, −7.97% in r2, both within-sweep). That's the signature of a real-but-noise-bound effect, not a falsified lever. So channel-weighting goes in the "park for now, revisit after multi-seed infrastructure lands" bucket — not a closed dead end.

The y_finite filter from frieren's cherry-pick is confirmed working: `test_geom_camber_cruise/mae_surf_p` is finite (85–99) across all 4 runs.

**Tanjiro reassigned to PR #508 (per-sample inverse-std weighting on surface loss)** — a higher-effect-size lever in the same loss-formulation lane. Predicted Δ: −5% to −15%, large enough to clear the seed-noise floor from a single sweep.

## 2026-04-28 03:37 — PR #410: EMA of weights at eval time, decay ∈ {0.99, 0.999, 0.9995} — **MERGED**

- Branch: `willowpai2d3-nezuko/ema-of-weights`
- **Hypothesis:** Exponential moving average of model parameters during training, evaluated at val/test time. Implicit regularization toward flatter minima + variance reduction. Predicted Δ: −1 to −5%.

### Sweep results (group `ema-of-weights`, on the merged baseline)

| Run | use_ema | ema_decay | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---|:---:|:---:|---:|---:|---:|---|
| no-ema-control | False | – | 11 | 143.14 | 130.74 | `1xyo3uci` |
| **ema-d0.99** | **True** | **0.99** | **12** | **121.44** | **108.66** | `22a7k787` |
| ema-d0.999 | True | 0.999 | 14 (last) | 123.20 | 110.57 | `d9spggkd` |
| ema-d0.9995 | True | 0.9995 | 14 (last) | 153.21 | 139.56 | `9s9ah0s9` |

### Decision: **MERGED** despite absolute number above the 115.84 single-seed bar

**Within-sweep evidence is overwhelming:**

1. **EMA d=0.99 vs no-EMA control (same seed env):** −21.71 MAE on val_avg, −22.08 on test_avg.
2. **End-of-training live-vs-EMA diagnostic (same run, apples-to-apples, seed-independent):** EMA gives −30.74 MAE over live weights at d=0.99, −36.84 at d=0.999, +8.05 at d=0.9995 (too slow for our 14-epoch regime).
3. **Per-epoch trajectories:** EMA-d0.99 monotone after epoch 6; EMA-d0.999 strictly monotone every epoch; no-EMA control jitters by ±20 MAE epoch-to-epoch. Mechanistic confirmation of the variance-reduction hypothesis.

The absolute val_avg=121.44 sits *above* PR #320's recorded 115.84, but post-variance-investigation we know that bar was a single favorable seed; nezuko's same-config no-EMA control on a different seed produced 143.14. The lever effect is robust; the absolute number depends on seed.

**EMA is now in baseline:** `use_ema=True, ema_decay=0.99, ema_warmup_steps=100` defaults in train.py. EMA also reduces seed variance — averaging weights damps out single-seed noise — so the post-merge multi-seed std should be smaller than the pre-EMA distribution.

**Test_geom_camber_cruise is now finite** across all four runs (79–110), confirming frieren's y_finite filter works.

**Follow-up assigned:** nezuko reassigned to PR #502 (AdamW betas + weight_decay sweep on the EMA+warmup baseline).

## 2026-04-28 02:55 — PR #323 (round 2): mlp_ratio rebased onto merged baseline — **CLOSED + critical variance finding**

- Branch: `willowpai2d3-thorfinn/mlp-ratio-4` (deleted post-close)
- **Hypothesis (round 2):** mlp_ratio=4 wins from r1 should compound with the warmup baseline. Thorfinn rebased onto merged advisor and re-ran with `peak_lr=1e-3, warmup_epochs=2`.

### Sweep results (group `mlp-ratio-sweep-r2`, against the NEW merged baseline)

| mlp_ratio | epochs done | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---:|---:|---:|---:|---:|---|
| **2 (control)** | **14** | **10** | **140.7015** | **126.4203** | `xklvptvl` |
| 4 | 13 | 13 (last) | 145.3649 | 132.6583 | `eaprgj0f` |
| 6 | 11 | 11 (last) | 154.4035 | 138.5408 | `vkvbnm52` |

### Decision: **CLOSED** + research-infrastructure pivot

**Within-sweep outcome:** mlp_ratio=2 wins on the rebased baseline. The r1 win at ratio=4 was an artifact of the OLD-LR schedule. Larger FFNs are training-budget-starved at the 30-min wall-clock cap — both ratio=4 and ratio=6 hit their best val at the very last completed epoch (still descending). Stacking with warmup did not happen; the levers actively interfere at this budget.

### **Critical variance finding (research-blocking)**

Thorfinn flagged that their r2 ratio=2 control landed at `val_avg=140.70`, while PR #320's same-config run `w3mjq2ua` produced `val_avg=115.84`. I verified: configs are byte-identical (`peak_lr=1e-3, warmup_epochs=2, weight_decay=1e-4, batch_size=4, surf_weight=10.0, model_config, n_params=662359, identical lr-ramp shape, runtime=1863s`). The only differences are bookkeeping (`agent` name, run name, group name). And `train.py` has **no seed control whatsoever** — `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed` all absent.

**Implications:**

- The 25-point (~21%) gap is pure seed variance.
- **PR #320's 115.84 was a single favorable seed**, not a robust baseline. The true baseline mean is likely ~130-145 with very wide variance.
- All in-flight rebase decisions (#294, #317, #322) are operating against a single-seed bar that may be unrepresentative.
- The "−21.5% from PR #320 merge" headline figure is now uncertain; the lever may have delivered closer to −5% with the rest being seed luck.

### Pivot to research infrastructure

Thorfinn reassigned to PR #482 (multi-seed baseline + deterministic seeding). 5 seeds of the merged config to establish `mean ± std`, plus add `torch.manual_seed` etc. to `train.py` so future runs are reproducible. ~2.5h of GPU; high operational value, unblocks confident decisions for every in-flight rebase PR.

## 2026-04-28 01:18 — PR #294: Huber surface loss, δ ∈ {0.5, 1.0, 2.0} — **SENT BACK FOR REBASE (strongest Round-1 candidate)**

- Branch: `willowpai2d3-alphonse/huber-loss-surf-p`
- **Hypothesis:** MSE loss penalizes outliers quadratically while the metric is MAE; Huber loss aligns the surface-loss objective with the eval metric. Predicted Δ: −3 to −8%.

### Sweep results (group `huber-loss-surf-p`, against the OLD baseline lr=5e-4)

| huber_delta | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|---:|---:|---:|---:|---|
| **0.5** | **12** | **106.3626** | **94.8919** | `dhuubteb` |
| 1.0 | 14 | 113.7845 | 102.9870 | `3z9jv6as` |
| 2.0 | 14 | 116.7367 | 103.2956 | `ag67f5k6` |

### Decision: **REQUEST CHANGES (rebase + re-run + extend sweep to δ=0)**

This is the **strongest Round-1 result so far** — `huber_delta=0.5` already beats the new merged baseline (115.84) **at OLD LR**, hitting 106.36 (−8.2% vs. new baseline, −28% vs. old default). Monotonic trend (smaller δ → better) is consistent on val and test. The instrumented `train/surf_huber_outlier_frac` (30% at δ=0.5 vs 5% at δ=2.0) provides clean mechanism evidence — most of the surface batch sits in the linear (L1) regime at δ=0.5, very little at δ=2.0.

Alphonse independently diagnosed the same inf-GT bug as frieren and thorfinn, with the bonus catch that the inf values look like fp16-overflow data-prep artifacts (`-65504` is the fp16 floor). Their y_finite implementation is similar in spirit to frieren's upstream version but uses subset-extraction instead of mask-zero-out — both correct.

Sent back for rebase because:

1. **Merge conflict** in `evaluate_split` (their y_finite block vs frieren's already-upstream version).
2. **Interaction with warmup unknown** — the 106.36 number was produced at `lr=5e-4`. The most likely outcome is Huber and warmup compound (orthogonal mechanisms — loss landscape vs optimizer schedule), but there's a real possibility Huber's smaller gradient magnitudes prefer the lower LR. We need to measure.

I also asked alphonse to extend the sweep to δ=0 (pure L1) — the monotonic trend implies the elbow is below δ=0.5, and degenerate L1 is the cleanest end of the range.

If the rebased run confirms compounding, this is the next merge candidate — likely producing a baseline below 100.

## 2026-04-28 01:05 — PR #316: More physics-attention slices, slice_num 64→128 — **CLOSED (negative result)**

- Branch: `willowpai2d3-edward/more-slice-tokens-128` (deleted post-close)
- **Hypothesis:** 64 slice tokens at ~3.7K nodes/slice is too coarse for sharp leading-edge gradients; doubling to 128 should help. Predicted Δ on `val_avg/mae_surf_p`: −5 to −12%.

### Sweep results (group `slice-num-sweep`, against the OLD baseline lr=5e-4)

| slice_num | epochs done | best ep | val_avg/mae_surf_p | test 3-split mean | W&B run |
|---:|---:|---:|---:|---:|---|
| **64 (baseline winner)** | **14** | **13** | **129.90** | **128.71** | `h9gzkjqq` |
| 96 | 12 | 11 | 135.01 | 135.28 | `l6ed3ins` |
| 128 | 11 | 10 | 149.98 | 149.40 | `s5uebsen` |
| 192 | 9 | 9 | 142.20 | 139.00 | `1p66k7d2` |

### Decision: **CLOSED** (confirmed negative result)

Edward added `slice_num=64` to the sweep on their own initiative (good call — the baseline becomes a control rather than just a reference number) and discovered the lever runs in the *opposite* direction at this budget: **slice_num=64 wins** by ~10% over 128. Three-line story behind the negative result:

1. **Compute–accuracy tradeoff at fixed wall-clock.** Slice attention is `O(B·H·N·slice_num + B·H·slice_num²)`; per-epoch time scales clearly with slices (131s → 213s, 1.6× from 64 → 192). At the 30-min cap, 64 gets 14 epochs vs. 192's 9.
2. **Even at matched epoch 9, slice_num=192 only beats 64 by 1.4** (142.20 vs 143.60) — within single-seed noise. If the hypothesis were strongly true, we'd see a clean gap at every epoch, not just convergence parity at low epochs.
3. **Aligns with the original Transolver paper's ablation,** which reports peak performance around 32–64 slices on most problems with diminishing/negative returns above.

Closed rather than rerun on the merged baseline because the negative result is well-supported and the mechanism (compute-budget bound + slice-softmax dilution) doesn't change with the LR baseline. Edward reassigned to **Fourier features for spatial coordinates** (PR #420) — architecture lane, well-known operator-learning trick.

## 2026-04-28 01:05 — PR #322: Per-channel surface loss, surf_p_weight ∈ {1, 2, 3, 5} — **SENT BACK FOR REBASE**

- Branch: `willowpai2d3-tanjiro/channel-weighted-loss`
- **Hypothesis:** Upweighting the pressure channel in the surface loss biases gradients toward the ranking metric. Predicted Δ: −3 to −6%.

### Sweep results (group `channel-weighted-loss`, against OLD baseline)

| surf_p_weight | best epoch | val_avg/mae_surf_p | test 3-split mean | W&B run |
|---:|---:|---:|---:|---|
| 1.0 (control) | 12 | 138.87 | 140.32 | `qomfj1kn` |
| 2.0 | 14 | 139.33 | 138.89 | `t0lgcgus` |
| **3.0** | 14 | **126.18** | **128.57** | `wkx4lwo5` |
| 5.0 | 13 | 142.29 | 140.37 | `4la7fez5` |

### Decision: **REQUEST CHANGES (rebase + re-run)**

In-sweep direction is clean — `surf_p_weight=3.0` wins by **−9.13%** over control, **exceeding the predicted −3 to −6% band**. Curve shows a single sharp optimum at 3.0 with non-monotonic behavior on both sides; 5.0 overweighting tanks both surface and volume MAE. Tanjiro's "Ux/Uy tax is real but small relative to pressure gain" framing is exactly the analysis I want.

But the absolute winner at 126.18 is below the new merged baseline of 115.84, so merging would regress val_avg. Sent back for rebase + re-run on top of `peak_lr=1e-3, warmup_epochs=2`. The 1.0 re-run will act as the control for confirming the rebase is clean; the 3.0 re-run tests stacking with the warmup gain.

## 2026-04-28 01:05 — PR #317: Surface-vs-volume balance, surf_weight ∈ {5, 20, 40, 80} — **SENT BACK FOR REBASE**

- Branch: `willowpai2d3-fern/surface-weight-sweep`
- **Hypothesis:** Sweeping `surf_weight` upward (default 10) directly increases gradient flow toward surface predictions. Predicted Δ: −3 to −8%.

### Sweep results (group `surface-weight-sweep`, against OLD baseline)

| surf_weight | best epoch | val_avg/mae_surf_p | test 3-split mean | W&B run |
|---:|---:|---:|---:|---|
| 5 | 8 | 143.93 | 142.63 | `vzvez6w9` |
| **20** | 13 | **129.41** | **129.99** | `tos0mpbx` |
| 40 | 14 | 129.93 | 131.37 | `baq7trjw` |
| 80 | 12 | 144.66 | 142.14 | `cir9lsik` |

### Decision: **REQUEST CHANGES (rebase + re-run)**

In-sweep U-shape is clean: `surf_weight=20` wins by **−10.1%** over the sw=5 control, **exceeding the predicted −3 to −8% band**. Volume MAE table monotonically degrades from sw=5 → sw=80, confirming the model is genuinely reallocating capacity surface ← volume; sw=80 overshoots and degrades both surface and volume.

Fern also independently diagnosed the `test_geom_camber_cruise` NaN bug — pinpointing it to `accumulate_batch`'s `0 * inf = NaN` issue — same root cause frieren and thorfinn arrived at. The fix is now upstream from frieren's cherry-pick (commit `32b5b40`).

But the absolute winner at 129.41 is below the new merged baseline of 115.84, so merging would regress val_avg. Sent back for rebase + re-run on top of `peak_lr=1e-3, warmup_epochs=2`.

## 2026-04-28 00:55 — PR #319: Deeper Transolver, n_layers 5→8 — **CLOSED (bug fix cherry-picked)**

- Branch: `willowpai2d3-frieren/deeper-n-layers-8` (deleted post-close)
- **Hypothesis:** Increase depth from 5 → 8 for hierarchical feature processing; predicted Δ on `val_avg/mae_surf_p` of −3% to −8%.

### Sweep results (group `deeper-n-layers`, against the OLD baseline lr=5e-4)

| n_layers | params | epochs done | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (post-bugfix) | W&B run |
|---:|---:|---:|---:|---:|---:|---|
| **6** | 0.78M | 12 / 50 | 12 (last) | **143.33** | **130.37** | `cjhgf7os` |
| 8 | 1.03M | 9 / 50 | 9 (last) | 147.32 | 133.87 | `m8l7qi00` |
| 10 | 1.27M | 8 / 50 | 7 | 191.87† | 155.36 | `wqhb8qoq` |

† Student reported 168.39 in the PR comment but W&B summary shows 191.87 — likely a stale earlier-epoch number. Doesn't affect ranking (n_layers=10 still loses).

### Decision: **CLOSED** (depth experiment) + **CHERRY-PICKED** (bug fix)

- **Depth experiment is compute-confounded at this budget.** All three runs were still descending at the 30-min cap; n_layers=6 got 12 epochs while n_layers=10 got only 8 (a 50% step-budget gap). Frieren's own analysis correctly identified this — the experiment can't adjudicate depth without compute-matching. At the post-PR-#320 baseline of 115.84, even n_layers=6 (143.33) is below the bar. Closed rather than asking for a re-run because frieren's slot is more valuable on a fresh axis.
- **Bug-fix cherry-picked into commit `32b5b40` on advisor branch.** Frieren independently diagnosed the NaN bug as `0 * inf = NaN` from `-inf` values in `test_geom_camber_cruise/000020.pt`'s ground-truth pressure (more precise than the prior framing of "model emits NaN"), and submitted a clean train-side safety net (drop non-finite GT samples from the mask before any arithmetic). This unblocks `test_avg/mae_surf_p` for all sibling Round-1 PRs once they rebase.
- **Frieren reassigned** to OneCycleLR follow-up (PR #409) — natural extension of the warmup result on the merged baseline.
- **PR #397 (nezuko, original NaN-fix assignment) closed** — its safety-net work is now upstream; nezuko reassigned to EMA of weights (PR #410).

## 2026-04-28 00:38 — PR #323: FFN expressivity, mlp_ratio 2 → 4 — **SENT BACK FOR REBASE**

- Branch: `willowpai2d3-thorfinn/mlp-ratio-4`
- **Hypothesis:** Bumping the per-block FFN inner-dim ratio from 2 → 4 increases nonlinear capacity per layer; should help complex pressure pattern fitting.
- **Predicted Δ on `val_avg/mae_surf_p`:** −3% to −7%.
- **Observed in-sweep Δ vs ratio=2 control:** −4.7% (143.75 → 136.96). On test_avg, −6.0% (130.22 → 122.41 after the student's local NaN workaround). **Direction matches prediction.**

### Sweep results (group `mlp-ratio-sweep`)

| mlp_ratio | params | peak GB | epochs done | best ep | val_avg/mae_surf_p | W&B run |
|---:|---:|---:|---:|---:|---:|---|
| 2 (control) | 0.66M | 42.1 | 14 / 50 | 11 | 143.7474 | `f188eiwk` |
| **4** | **0.99M** | 52.2 | 13 / 50 | 12 | **136.9640** | `a6v7k5zd` |
| 6 | 1.32M | 63.1 | 11 / 50 | 10 | 150.8605 | `hj0hyhkb` |

### Decision: **REQUEST CHANGES (rebase + re-run)**

- The PR's runs were submitted **before PR #320 merged**, so they used the OLD baseline (`lr=5e-4`, no warmup). The `mlp_ratio=4` winner at 136.96 is *worse* than the new merged baseline (115.84). Merging would regress val_avg.
- Sent back to thorfinn with instructions to rebase onto the merged advisor branch and re-run the sweep on top of `peak_lr=1e-3, warmup_epochs=2`. The lever direction is clean; we expect it to compound with warmup if FFN-expressivity is orthogonal to LR-schedule (the prior).
- Also flagged: drop the local re-eval script (`target/sweep_logs/reeval_test.py`) — the train-side safety net for the NaN bug should live in the trainer, not a follow-up reanalysis.

### Bug confirmation: NaN in `test_geom_camber_cruise`

Thorfinn pinpointed the exact root cause that nezuko's PR #320 had only narrowed to "model emits NaN":

- The offending file is **`test_geom_camber_cruise/000020.pt`** — 761 NaN values in the pressure channel of the **ground truth**, not the prediction.
- `data/scoring.accumulate_batch` is supposed to skip samples with non-finite ground truth, but the implementation does this by zero-masking, and `0.0 * NaN == NaN` in IEEE 754. So the NaN propagates into the global accumulators for that split.
- This means **the model is fine** — it's a scoring bug. The fix lives in the train-side wrapper since `data/scoring.py` is read-only. nezuko's PR #397 is in flight to land this safety net centrally.

## 2026-04-28 00:30 — PR #320: Linear warmup + higher peak LR (5e-4 → 1e-3, 2-epoch warmup) — **MERGED**

- Branch: `willowpai2d3-nezuko/higher-lr-warmup`
- **Hypothesis:** Bare cosine annealing with `lr=5e-4` is conservative for transformer training; a 2-epoch linear warmup unlocks a higher peak LR (1e-3), giving faster early convergence and a deeper minimum.
- **Predicted Δ on `val_avg/mae_surf_p`:** −3% to −7%.
- **Observed Δ on `val_avg/mae_surf_p`:** **−21.5%** (147.55 → 115.84). Far exceeded prediction.

### Sweep results (group `lr-warmup-sweep`, all with 2-epoch warmup)

| peak_lr | best epoch | val_avg/mae_surf_p | mean test (3 valid splits) | W&B run |
|---|---:|---:|---:|---|
| 5e-4 | 13 | 147.5538 | 150.67 | `liddr8ce` |
| **1e-3** | **14** | **115.8379** | **112.78** | `w3mjq2ua` |
| 2e-3 | 9 | 151.1338 | 151.16 | `4zc03997` |

### Per-split val MAE at best epoch (`mae_surf_p`)

| Split | 5e-4 | **1e-3 (winner)** | 2e-3 |
|---|---:|---:|---:|
| `val_single_in_dist` | 182.68 | **131.06** | 181.14 |
| `val_geom_camber_rc` | 157.13 | **129.57** | 165.60 |
| `val_geom_camber_cruise` | 111.21 | **92.55** | 115.88 |
| `val_re_rand` | 139.19 | **110.17** | 141.92 |
| **val_avg** | 147.55 | **115.84** | 151.13 |

### Analysis & conclusions

- **Peak LR is the dominant effect, not warmup itself.** With warmup held fixed at 2 epochs, the 5e-4 control (147.55) and 2e-3 (151.13) sit nearly on top of each other — only 1e-3 peels away from the pack. So warmup at the *old* peak LR doesn't capture the win.
- **Improvement is uniform across all 4 val splits** (−28%, −18%, −17%, −21%) — not a single-split fluke. Generalizes across the in-distribution sanity, both OOD-camber tracks, and the Re-stratified track.
- **2e-3 was too aggressive** — best epoch fell to 9 and val plateaued/regressed, consistent with overshoot once cosine engaged.
- **Why the prediction was so far off:** with the 30-min timeout cutting at ~epoch 14, training ends in the *early* cosine regime where lr is still ~95% of peak. Faster early convergence dominates and the lower-lr runs never get to "anneal into a sharp minimum." Prediction assumed the full 50 epochs.

### Pre-existing issue surfaced (not introduced by this PR)

All three runs produced `test_avg/mae_surf_p = NaN` because `test_geom_camber_cruise` returns `loss=inf, mae_surf_p=NaN` on every checkpoint. Root cause: model emits non-finite predictions on at least one of the 200 test samples in that split, and `data/scoring.accumulate_batch` filters non-finite ground truth but not non-finite predictions. The val analogue of that split is sane (92.55 in the winner) — so the failure is test-sample-specific. Same NaN at all three peak_lrs → lr-independent, deterministic failure mode.

### Decision: **MERGED**

- Val improvement is large (−21.5%), consistent across splits, and matches the expected direction.
- Test NaN is pre-existing and orthogonal to this PR's lever.
- New baseline established: `peak_lr=1e-3, warmup_epochs=2`. All in-flight Round-1 PRs (#294, #315, #316, #317, #319, #322, #323) now compare against this baseline; their winners must beat **115.84**.
- Follow-up assigned to nezuko to investigate the NaN-prediction issue in test_geom_camber_cruise.
