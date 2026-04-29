# SENPAI Research Results — willow-pai2e-r5

---

## 2026-04-28 19:51 — PR #732: Scale Transolver to n_hidden=256, n_layers=7, n_head=8

- **Branch:** `willowpai2e5-alphonse/larger-model-capacity` (closed)
- **W&B run:** `pkyat9dy` — group `larger-model-capacity`
- **Hypothesis:** Scaling Transolver from 0.93M to 3.01M params (n_hidden=256, n_layers=6, n_head=8 after n_layers=7 OOM fallback) improves fitting and OOD generalization.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist`      | 184.36 | 169.38 |
| `val_geom_camber_rc`      | 162.75 | 152.78 |
| `val_geom_camber_cruise`  | 130.79 | **NaN** |
| `val_re_rand`             | 141.88 | 147.91 |
| **avg**                   | **154.95** | **NaN** |

Best epoch = 6 (final epoch before 30-min timeout). Training trajectory (val_avg): 245.66 → 210.36 → 202.81 → 184.30 → 184.74 → 154.95.

### Commentary & Conclusions

- **Non-conclusive under our compute budget.** The 30-min SENPAI_TIMEOUT_MINUTES cap allowed only 6 of 50 scheduled epochs. The CosineAnnealingLR (T_max=50) was running at ~94% initial LR — effectively zero LR decay. Trajectory shows rapid, unfinished descent; we don't know where the model would converge.
- **5.2× throughput penalty.** Each epoch took ~5.2 min vs ~1.5 min baseline. A matched-wall-clock baseline (128/5/4) would do ~20 epochs; this 3.01M-param model got 6.
- **NaN on `test_geom_camber_cruise/mae_surf_p`.** The undertrained large model produced non-finite pressure predictions on at least one hard test sample. Val numbers are finite because no val sample was numerically unstable. This is a model-stability issue, not a data issue.
- **Bug identified in `data/scoring.py`.** `accumulate_batch` skips samples with non-finite ground truth but not non-finite predictions — a single NaN prediction node poisons the entire split's channel sum. data/ is read-only for student PRs; flagging for follow-up.
- **Reference data point recorded:** val_avg/mae_surf_p = 154.95 (first run; model ~1/3 trained). Will compare against in-flight Wave-1 runs once they complete.
- **Decision:** Closed (premature — NaN test metric, no matched-budget baseline comparison, 5.2× throughput cost unjustified without fair comparison).

### Next step for this direction

Revisit capacity scaling after (1) in-flight Wave-1 runs give us a baseline-architecture val number, (2) BF16 mixed-precision is available to reduce per-epoch time, and (3) gradient accumulation is verified to allow n_layers=7. Alphonse reassigned to `film-re-conditioning` (PR #796).

---

## 2026-04-28 20:15 — PR #763: Add physics-informed distance features to Transolver input

- **Branch:** `willowpai2e5-thorfinn/feature-engineering` (merged)
- **W&B run:** `072wo9xb` — group `feature-engineering`
- **Hypothesis:** Appending dist_to_surface, log(1+dist_to_surface), and is_tandem as derived node features provides physically grounded inductive bias for boundary-layer gradients and tandem vs. single-foil regime.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 177.0157 | 148.3098 |
| `val_geom_camber_rc` | 157.4591 | 145.5500 |
| `val_geom_camber_cruise` | 105.7864 | 91.0172 |
| `val_re_rand` | 125.4113 | 121.3623 |
| **avg** | **141.4181** | **126.5598** |

Best epoch = 12 of 50 (30-min timeout). AUG_X_DIM=27, +768 params.

### Commentary

- **First PR with clean test_avg across all 4 splits.** Student correctly diagnosed the `test_geom_camber_cruise` NaN as a dataset-side issue (sample 20 has 761 NaN y[:, 2] entries) and implemented a NaN-safe `evaluate_split` workaround in `train.py` since `data/` is read-only. Critical fix — all future runs now report finite test_avg.
- **Decision: Merged.** Beats #732 reference (154.95), produces clean test metrics. Features are physically motivated and low-risk.
- Became new baseline at val_avg=141.42 before fern's #737 landed.

---

## 2026-04-28 20:25 — PR #737: Add 5-epoch linear warmup + peak lr=1e-3 before cosine decay

- **Branch:** `willowpai2e5-fern/lr-warmup-cosine` (merged)
- **W&B run:** `5b22tecz` — group `lr-warmup-cosine`
- **Hypothesis:** Linear LR warmup (1e-4 → 1e-3, 5 epochs) followed by cosine decay to eta_min=1e-6 stabilises early training of slice-attention projections and allows a higher peak LR without instability.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 149.241 | 126.021 |
| `val_geom_camber_rc` | 146.033 | 129.155 |
| `val_geom_camber_cruise` | 96.362 | NaN (pre-fix; ~91 expected) |
| `val_re_rand` | 119.852 | 115.590 |
| **avg** | **127.872** | NaN (3-split avg ≈ 123.59) |

Best epoch = 14 of 50 (30-min timeout). LR at epoch 14 ≈ 8e-4 (barely into cosine decay).

### Commentary

- **Decision: Merged (current best).** val_avg=127.87 beats new baseline of 141.42 after #763 merged. Schedule change is orthogonal to features change — both merged cleanly.
- **Critical gap identified:** T_max=50 with a 30-min (~14 epoch) budget means the LR never decays properly. Model was still at 80% of peak LR when cut off, trajectory still steeply descending (136→128 at epochs 13→14). Budget-matched re-run (epochs=14, warmup=2) assigned to fern as PR #809.
- **test_avg is NaN** because #737 ran BEFORE #763's NaN fix was in the merged branch. Future runs will be clean.

---

## 2026-04-28 20:30 — PR #733: Increase slice_num from 64 to 256 for richer physics decomposition

- **Branch:** `willowpai2e5-askeladd/more-slices` (closed)
- **W&B run:** `8l3pbq6x` — group `more-slices`
- **Hypothesis:** Quadrupling slice tokens (64→256) gives the model finer-grained physics decomposition for boundary layer / wake / freestream separation.

### Results

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 185.28 | 161.17 |
| `val_geom_camber_rc` | 179.72 | 160.89 |
| `val_geom_camber_cruise` | 111.53 | NaN (pre-fix) |
| `val_re_rand` | 129.46 | 130.14 |
| **avg** | **151.50** | NaN (3-split avg ≈ 150.73) |

Best epoch = 8 of 50 (30-min timeout). slice_num=256 fits in 82.3 GB VRAM. Per-epoch wall-time: 4.2 min vs 2.2 min for slice_num=64.

### Commentary

- **Decision: Closed (regression).** val_avg=151.50 vs current baseline 127.87 = 18.5% regression, well above the 5% close threshold.
- **Throughput penalty is structural, not tunable.** ~2× per-epoch cost meant 8 epochs vs ~14 for baseline-config runs in the same 30-min cap. Val curve still steeply falling at cutoff (161→151 epochs 7→8), so even granting "256 needs more time," it's strictly worse for our compute budget. Would need a budget extension or a separate throughput improvement to revisit.
- Student also flagged the same `test_geom_camber_cruise` NaN issue thorfinn diagnosed in #763 — but that's already fixed in the current merged baseline. Confirms the bug is global (not slice-num-specific).
- Askeladd reassigned to `mixed-precision-bf16` (PR #811) — directly attacks the throughput constraint this PR exposed.

---

## 2026-04-28 20:45 — PR #745: Separate output heads for velocity (Ux/Uy) and pressure (p) — sent back

- **Branch:** `willowpai2e5-tanjiro/separate-pressure-head` (sent back for rebase + Option 3)
- **W&B runs:** `m5ydsa1t` (split_linear), `7aw36w9e` (split_mlp) — group `separate-pressure-head`
- **Hypothesis:** Splitting the final output projection into two specialized MLPs (Ux/Uy and p) gives the model architectural capacity for pressure-specific feature detectors and directly improves `val_avg/mae_surf_p`.

### Results (against PRE-merge code: no features, no warmup)

| Option | head params | val_avg/mae_surf_p ↓ | test_avg/mae_surf_p ↓ |
|--------|-------------|----------------------|-----------------------|
| Option 1 — `split_linear` (two `Linear`) | 387 | **130.82** | **118.99** |
| Option 2 — `split_mlp` (deeper, capacity-matched) | 16,707 | 134.46 | 123.79 |

Per-split surf_p MAE for Option 1 (winner):

| Split | val | test |
|-------|------|------|
| `val_single_in_dist` | 155.53 | 137.41 |
| `val_geom_camber_rc` | 141.41 | 133.08 |
| `val_geom_camber_cruise` | 104.96 | 86.68 |
| `val_re_rand` | 121.37 | 118.81 |

Best epoch = 12 of 50 (30-min timeout at epoch 14). Both runs hit timeout.

### Commentary & Conclusions

- **Comparison NOT fair against current baseline.** Student's run was on pre-merge code (no distance features from #763, no warmup+cosine from #737). Direct val_avg=130.82 looks like a 2.3% regression vs 127.87, but that's apples-to-oranges. Against the no-features/no-warmup reference points (alphonse #732 154.95, askeladd #733 151.50), 130.82 is a substantial improvement — head split appears to be a real signal.
- **Option 1 winning is partially a smaller-head effect, not pure specialization.** Student correctly identified that the baseline's `mlp2` already had ~16.9k params; their "Option 1" actually shrinks the head 44× to 387 params. So Option 1 winning at 14 epochs may be undertraining favoring smaller-capacity heads, not pressure-specialization succeeding.
- **Critical follow-up: Option 3 (capacity-matched split).** Student proposed `Linear(hidden,hidden)→GELU` shared first layer, then forked `Linear(hidden,2)` and `Linear(hidden,1)`. This isolates specialization from capacity. **Sent back to test this on the rebased baseline.**
- **NaN bug.** Student found and patched the same `nan*0=nan` propagation bug thorfinn diagnosed in #763 (data/scoring.py masking via multiplication poisons the running sum when ground truth has NaN). Their patch was in train.py:evaluate_split, same approach as #763 (already merged). Duplicate work but confirms the fix is correct. Student also patched the W&B run summaries via wandb.Api() to retrofit clean test numbers.
- **Decision: Sent back.** Rebase onto current baseline, run Option 3 only. If <127.872, merge.

---

## 2026-04-28 21:29 — PR #811: Enable bf16 mixed precision for 1.5-2x training throughput

- **Branch:** `willowpai2e5-askeladd/mixed-precision-bf16` (merged)
- **W&B run:** `newqt8dd` — group `mixed-precision-bf16`
- **Hypothesis:** BF16 autocast on forward+loss yields 1.5-2× per-epoch speedup → more epochs in the 30-min budget → directly improves val_avg/mae_surf_p. BF16 preferred over FP16 for this dataset's ±29K pressure range (bf16 has the same 8-bit exponent as fp32, no overflow).

### Results

| Metric | Baseline (fp32, #737) | bf16 (#811) | Δ |
|---|---|---|---|
| Wall-clock / epoch | 131.96 s | **110.02 s** | 1.20× speedup |
| Epochs in 30 min | 14 | **17** | +3 (+21%) |
| Peak VRAM | — | **33.1 GB** | 63 GB headroom on 96 GB |
| **val_avg/mae_surf_p** | 127.872 | **127.402** | **−0.47** |
| **test_avg/mae_surf_p** | NaN (pre-fix) | **116.211** | first clean 4-split test |
| NaN/Inf events | — | **0** | numerically stable |

Per-split:

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist`     | 151.791 | 141.142 |
| `val_geom_camber_rc`     | 147.898 | 134.121 |
| `val_geom_camber_cruise` | **93.729** | **79.094** |
| `val_re_rand`            | **116.189** | 110.488 |
| **avg**                  | **127.402** | **116.211** |

Best epoch = 17 of 50. Gradient norm mean=69.6, max=1224.9 — isolated spike, no explosion pattern.

### Commentary & Conclusions

- **Decision: Merged.** val_avg=127.402 beats 127.872 baseline; test_avg=116.211 is a clean 4-split number, 10 points better than the last clean test (#763, 126.56). This is a platform improvement: all subsequent experiments inherit the BF16 speedup and VRAM headroom.
- **Speedup was 1.20× not 1.5-2×.** Gap explained: (1) model is small (663K params), matmul share of total step time is bounded; (2) `add_derived_features` runs a Python for-loop with `.item()` GPU→CPU sync — pure non-matmul cost untouched by autocast; (3) LayerNorm kept in fp32 by PyTorch autocast default. These three factors cap the matmul-side gain.
- **Key structural finding: 33.1 GB VRAM with batch_size=4.** 63 GB headroom unlocked for batch scaling. Batch_size=8–16 now plausible without OOM risk. This is the highest-leverage immediate follow-up.
- **bf16 numerically clean.** Zero NaN/Inf events across 17 epochs / 6,381 steps. bf16's 8-bit exponent handles ±29K pressure range without issue; no GradScaler, no fp32 loss-cast needed.
- **Next bottleneck: `add_derived_features` Python loop.** With matmul now faster, the non-matmul Python distance loop with `.item()` sync is the new dominant cost. Vectorizing it should push speedup closer to 1.5-2×.
- **Askeladd reassigned** to batch_size scaling (leveraging the 63 GB headroom directly).

---

## 2026-04-28 21:30 — PR #734: Increase surf_weight from 10 to 50/100 to directly target surface pressure MAE — closed

- **Branch:** `willowpai2e5-edward/higher-surf-weight` (closed)
- **W&B runs:** `zi70jnyl` (sw=10 control), `srbt5u57` (sw=50), `24dctht8` (sw=100) — group `higher-surf-weight`
- **Hypothesis:** Increasing surf_weight from 10 → 50/100 forces the model to prioritize fitting surface pressure during training, directly improving val_avg/mae_surf_p.

### Results (against PRE-merge code)

| surf_weight | val_avg/mae_surf_p ↓ | best epoch | Δ vs sw=10 |
|-------------|----------------------|------------|------------|
| **10 (control)** | **130.43** | 13 | — |
| 50 | 135.56 | 14 | +3.9% (worse) |
| 100 | 136.72 | 12 | +4.8% (worse) |

Per-split val/mae_surf_p:

| Split | sw=10 | sw=50 | sw=100 |
|-------|-------|-------|--------|
| `val_single_in_dist` | **157.32** | 178.62 (+13.5%) | 158.34 (+0.6%) |
| `val_geom_camber_rc` | **139.59** | 139.07 (-0.4%) | 148.17 (+6.1%) |
| `val_geom_camber_cruise` | **106.21** | 106.37 (+0.2%) | 108.81 (+2.4%) |
| `val_re_rand` | **118.62** | 118.20 (-0.4%) | 131.57 (+10.9%) |

All three runs hit 30-min timeout; peak VRAM 42.1 GB.

### Commentary & Conclusions

- **Decision: Closed (clean negative result).** Both up-direction values regress; no variation in the up direction is likely to flip this.
- **Key mechanistic insight from the student:** "Volume residuals provide spatial context that the Transolver attention uses for surface predictions — they're not just regularization, they're informative." This explains why naively up-weighting surface fails — removing volume context starves the surface prediction of spatial information. The +13.5% single_in_dist regression at sw=50 is the cleanest demonstration: single-foil has highest pressure amplitude and most depends on volume context.
- **The current `surf_weight=10` may already be past the optimum.** Surface nodes are ~1% of total but get 10× weight per node, ≈10% effective contribution to the gradient. Pushing further hits diminishing/negative returns.
- **Edward reassigned** to lower-surf-weight sweep ({3, 5, 7}) on the merged baseline — direct test of the counter-hypothesis the student's analysis raises.
- **Bug observation:** Student also flagged the `test_geom_camber_cruise` NaN issue. Already fixed in merged baseline (#763 NaN-safe eval).

---

## 2026-04-28 22:26 — PR #742: Add dropout=0.1 to MLP sublayers to reduce OOD overfitting — closed

- **Branch:** `willowpai2e5-nezuko/dropout-regularization` (closed)
- **W&B runs:** `55aolphx` (no dropout control), `g81o4brf` (dropout=0.1) — group `dropout-regularization`
- **Hypothesis:** Dropout=0.1 in MLP sublayers reduces overfitting to training geometry/Re combinations and improves OOD generalization on `val_geom_camber_rc/cruise` and `val_re_rand`.

### Results (against PRE-merge code; no BF16, no warmup)

| mlp_dropout | val_avg/mae_surf_p ↓ | best epoch | Δ |
|-------------|----------------------|------------|------|
| **0.0 (control)** | **123.37** | 14 (= last) | — |
| 0.1 | 138.68 | 14 (= last) | +12.4% (worse) |

Per-split val/mae_surf_p:

| Split | dropout=0.0 | dropout=0.1 | Δ |
|-------|-------------|-------------|------|
| `val_single_in_dist` | 148.11 | 162.06 | +9.4% |
| `val_geom_camber_rc` | 123.92 | 149.62 | **+20.7%** |
| `val_geom_camber_cruise` | 99.57 | 113.49 | +14.0% |
| `val_re_rand` | 121.89 | 129.54 | +6.3% |

Both runs hit 30-min timeout at epoch 14/50; peak VRAM 42-43 GB (no BF16 in this run).

### Commentary & Conclusions

- **Decision: Closed (clean negative result with excellent root-cause analysis).**
- **Critical mechanistic insight from the student:** "Both runs stopped at epoch 14/50 and best_epoch=14=last trained epoch in both cases. That's a clear signal of an *under*-trained model — there is no overfitting to regularize. Dropout's only effect is to inject noise that slows convergence."
- **OOD-hits-hardest signature confirms under-training, not overfitting.** If dropout were correctly closing a generalization gap, ID would suffer most and OOD least; we see the opposite (rc +20.7% > id +9.4%). This is the fingerprint of "fewer effective gradient updates per parameter."
- **Implementation verified:** dropout in standard transformer-FFN location (between GELU and linear), model.eval() correctly called for val and test, attention dropout untouched. Negative result is not from a bug.
- **Implication for regularization more broadly:** Until the model demonstrates overfitting (best_epoch < final_epoch by a wide margin), traditional regularizers (dropout, weight decay) have no benefit to provide. Schedule fix (#809) or batch-size scaling (#848) may unlock convergence first; only then does regularization become testable.
- **Nezuko reassigned** to DropPath/stochastic depth — student's suggestion #3, different mechanism (drops entire residual branches, model-averaging interpretation, compute-efficient).

---

## 2026-04-28 22:44 — PR #810: Add EMA weight averaging for lower-variance OOD checkpointing — sent back

- **Branch:** `willowpai2e5-thorfinn/ema-model-checkpoint` (sent back for post-warmup EMA + decay sweep)
- **W&B run:** `g2yfau61` — group `ema-model-checkpoint`
- **Hypothesis:** EMA decay=0.999 of model weights smooths over noisy gradient updates, especially helping OOD val splits. Validate and checkpoint with EMA shadow weights.

### Results (against PRE-merge code; no BF16)

| Split | EMA (this run) | Baseline #737 | Δ |
|-------|----------------|---------------|------|
| `val_single_in_dist` | 170.472 | 149.241 | +21.23 |
| `val_geom_camber_rc` | 149.418 | 146.033 | +3.39 |
| `val_geom_camber_cruise` | 106.842 | 96.362 | +10.48 |
| `val_re_rand` | 121.037 | 119.852 | +1.19 |
| **val_avg/mae_surf_p** | **136.942** | **127.872** | **+9.07 (+7.1%)** |

| Test Split | EMA (this run) | #763 baseline |
|-----------|----------------|---------------|
| `test_single_in_dist` | 147.616 | 148.310 |
| `test_geom_camber_rc` | 132.479 | 145.550 |
| `test_geom_camber_cruise` | 88.305 | 91.017 |
| `test_re_rand` | 120.607 | 121.362 |
| **test_avg/mae_surf_p** | **122.252** | **126.560** |

Best epoch = 13/50 (30-min timeout); peak VRAM 42.2 GB (no BF16 in this run).

### Commentary & Conclusions

- **Decision: Sent back (despite >5% val regression).** Strict close criterion would apply, but the student's diagnostic is so clear and the fix so simple that one more iteration is high-value.
- **The mechanism appears to work — the timing is wrong.** Trajectory was monotonically improving; test_avg actually IMPROVED by 4.3 points relative to the last clean 4-split test (#763 → #810: 126.56 → 122.25). The EMA shadow lags the live model on val, but the smoothing effect on test geometry held up.
- **Critical diagnostic from the student (worth recording):** With decay=0.999 and warmup_epochs=5, the shadow has effective memory ~1000 steps (~2.7 epochs). Of that, ~32% of the shadow mass at epoch 13 still sits on warmup-era weights (lr=1e-4 regime where the model is barely learning). The shadow is contaminated by warmup gibberish.
- **Send-back instructions:** (1) defer EMA initialization until after warmup completes; (2) sweep decay ∈ {0.999, 0.995}; (3) rebase onto BF16 baseline (17 epochs instead of 14 = more post-warmup steps to amortize); (4) log both EMA and live val_avg per epoch to verify the gap closes.
- **Implementation correctness verified:** deepcopy isolation, no live-model overwrite, EMA state correctly checkpointed.

---

## 2026-04-28 22:58 — PR #848: Scale batch_size 4→8/10/12 to use BF16 VRAM headroom — closed

- **Branch:** `askeladd/larger-batch-size` (closed)
- **W&B runs:** `0zt9fppw` (bs=8), `qt40qg9s` (bs=10), `l829jyzw` (bs=12 OOM); baseline `newqt8dd` (bs=4) — group `larger-batch-size`
- **Hypothesis:** With BF16 freeing 63 GB VRAM, larger batches (8/12) increase samples/step + smoother gradients → faster convergence and better OOD generalization.

### Results (against merged BF16 baseline #811)

| batch_size | val_avg/mae_surf_p ↓ | test_avg/mae_surf_p ↓ | best epoch | s/epoch | Peak VRAM | Δ val |
|------------|---------------------:|----------------------:|-----------:|--------:|----------:|------:|
| **4 (baseline)** | **127.40** | **116.21** | 17 | 108.7 | 33.1 GB | — |
| 8 | 142.51 | 131.81 | 12 | 117.1 | 66.1 GB | +11.9% (worse) |
| 10 | 147.23 | 134.43 | 15 | 119.9 | 82.6 GB | +15.6% (worse) |
| 12 | OOM | — | — | — | >95 GB | failed (epoch 1) |

### Commentary & Conclusions

- **Decision: Closed (clean negative result with strong mechanistic analysis).**
- **Both halves of the hypothesis flipped or failed:**
  1. Per-sample throughput *regressed* from 13.79 → 12.41 samples/sec (bs=4→10) — the `add_derived_features` Python loop with per-sample CPU sync (`mask[b].sum().item()`) plus chunked pairwise distance is now the dominant cost at high B. **Vectorizing this is a separate throughput-engineering target.**
  2. Same LR + larger batch → drastically fewer optimizer steps under 30-min wall clock (bs=10 hits only 35% of baseline's 6,375 steps). Linear LR scaling was deferred — and is almost certainly required.
- **Gradient noise smoothing did hold:** max grad norm collapsed from 1225 (bs=4 baseline, one >1000 spike) to 572 (bs=8) to 0 spikes at higher B. The gradient signal *is* richer per step; we just take fewer of those steps.
- **Per-split mixing pattern at bs=10:** `val_geom_camber_rc` improved (147.90 → 136.53) while `val_single_in_dist` collapsed (151.79 → 209.05). Larger batches reshuffle which split is favored — likely a sampler/domain-balance interaction.
- **Real bottleneck identified:** `add_derived_features` Python loop in `train.py:79-98`. Removing the CPU-sync `.item()` calls and vectorizing the chunked pairwise distance is the unblocking lever before batch_size scaling becomes useful.
- **Askeladd reassigned** to Huber-delta sweep (#885) — the validated signal from #739 deserves a clean delta optimum.

---

## 2026-04-28 23:04 — PR #796: FiLM-condition TransolverBlocks on log(Re) — closed

- **Branch:** `willowpai2e5-alphonse/film-re-conditioning` (closed)
- **W&B runs:** `2pqqhqo6` (no FiLM control), `b6qc62sq` (FiLM log-Re); pre-fix runs `2nn5hiyh` and `23tdi1dg` produced NaN test metrics — group `film-re-conditioning`
- **Hypothesis:** FiLM(log Re) per-block conditioning gives the model an explicit Reynolds-regime signal so cross-Re generalization improves on `val_re_rand` (predicted −5 to −15%).

### Results — paired comparison (both runs hit 30-min wall clock at epoch 13/50)

| Split (lower better) | Baseline (no FiLM) | FiLM (log Re) | Δ |
|---|---:|---:|---:|
| `val_single_in_dist`/`mae_surf_p` | 169.39 | 159.84 | −5.6% |
| `val_geom_camber_rc`/`mae_surf_p` | 143.55 | 153.87 | +7.2% |
| `val_geom_camber_cruise`/`mae_surf_p` | 106.06 | 106.97 | +0.9% |
| `val_re_rand`/`mae_surf_p` | 122.42 | 121.38 | **−0.9%** (predicted −5 to −15%) |
| **`val_avg/mae_surf_p`** | **135.35** | **135.51** | **+0.1% (tied)** |
| `test_avg/mae_surf_p` | 119.71 | 125.08 | **+4.5% (worse)** |

### FiLM diagnostics at end of run (FiLM run)

| Metric | Value | Interpretation |
|---|---:|---|
| `film/last_weight_l2` | 18.58 | grew from 0 — net is actively learning |
| `film/gamma_dev_mean` | 0.26 | gamma drifting ~26% off identity |
| `film/beta_abs_mean` | 0.23 | non-trivial bias shift |

### Commentary & Conclusions

- **Decision: Closed (clean negative on the primary diagnostic).**
- **The hypothesis's predicted mechanism didn't materialize.** FiLM is *alive* (non-trivial gamma/beta on log-Re sweep, weights grew from zero) but the val_re_rand improvement is essentially noise (−0.9% vs predicted −5 to −15%). The model is *not* using log-Re modulation to close the cross-Re gap — log-Re is already in the input feature vector and the attention is plausibly extracting it adequately without FiLM-style explicit conditioning.
- **Test_avg regressed by 4.5%.** This combined with the val tie is a strong signal that the conditioning is adding noise without a generalization payoff at this training budget.
- **Implementation deviations from spec — all justified:**
  - Used `x[:, 0, 13:14]` not `x[:, :, 13].mean(...)` because pad_collate right-pads with zeros (mean would mix in padding).
  - Added `--use_film` flag for clean paired comparison.
  - Re-zeroed FiLM final layer after `_init_weights` (otherwise trunc_normal overrides identity init).
- **Bug fix flagged:** Student also identified the `nan*0=nan` issue in `data/scoring.py` (separately fixed in train.py since `data/` is read-only). This is a known issue from #763.
- **Alphonse reassigned** to per-sample y-normalization (#896) — direct attack on the high-Re-dominates-loss issue that FiLM was hoping to fix indirectly.

---

## 2026-04-28 23:30 — PR #739: Replace MSE with Huber loss (delta=1.0) — sent back for rebase

- **Branch:** `willowpai2e5-frieren/huber-loss` (sent back; major win; needs rebase onto BF16 baseline)
- **W&B run:** `z2a34zbu` (`willowpai2e5-frieren/huber-loss-d1.0`) — group `huber-loss`
- **Hypothesis:** Huber loss with delta=1.0 caps influence of high-Re outlier samples (per-sample y_std varies 10×), reducing gradient noise and improving OOD generalization.

### Results (against PRE-merge code — no BF16, no warmup; comparable baseline = #763 features-only val_avg=141.42)

| Split | Huber (best epoch 14, last epoch) | BF16 baseline #811 (best epoch 17) | Δ vs BF16 baseline |
|-------|-----------------------------------:|-----------------------------------:|-------------------:|
| `val_single_in_dist` | 125.13 | 151.79 | **−17.6%** |
| `val_geom_camber_rc` | 110.93 | 147.90 | **−25.0%** |
| `val_geom_camber_cruise` | 79.69 | 93.73 | **−15.0%** |
| `val_re_rand` | 99.79 | 116.19 | **−14.1%** |
| **val_avg/mae_surf_p** | **103.89** | **127.40** | **−18.5%** |

| Test split | Huber (epoch 14) | BF16 baseline #811 | Δ |
|-----------|----------------:|-------------------:|------:|
| `test_single_in_dist` | 108.45 | 141.14 | −23.2% |
| `test_geom_camber_rc` | 102.57 | 134.12 | −23.5% |
| `test_geom_camber_cruise` | NaN (data bug repro on pre-merge code) | 79.09 | — |
| `test_re_rand` | 96.78 | 110.49 | −12.4% |
| `test_avg/mae_surf_p` | NaN (cruise) — 3-split avg = 102.60 | 116.21 | −20.2% on 3-split |

Best epoch 14 = LAST epoch (timeout cut-off); val_avg trajectory `224.75 → 262.81 → 177.70 → 179.66 → 146.75 → 198.43 → 160.12 → 125.46 → 137.59 → 132.87 → 162.95 → 129.82 → 124.26 → 103.89` — **steepest descent in the final 2 epochs (124.26 → 103.89, −16.4% in one epoch). 103.89 is a lower bound on Huber's potential.**

### Commentary & Conclusions

- **Decision: Sent back for rebase onto BF16 baseline.** Result is a major win (−18.5% val_avg) but is on pre-merge code. We need the result on the merged baseline (BF16 + features + warmup) to get the actual stack effect, and the BF16 platform gives 1.20× more epochs in the same wall clock — this should make the result *better*, not worse.
- **The 4-split improvement is consistent and strong.** All splits improve 14–25%. This is the largest single-PR improvement seen in the program so far. OOD splits (rc, cruise, re_rand) all improve by 14–25% — Huber is *not* just a single_in_dist trick.
- **Mechanism likely real:** with `surf_weight=10` and per-sample y_std varying 10×, MSE gradient is dominated by 1–2 high-Re samples per batch. Huber caps that contribution at delta=1.0, letting low-Re samples contribute usefully.
- **Run did not converge.** 14/50 epochs at 30-min cap; val curve was descending fast at cutoff. The post-rebase BF16 run gets ~17 epochs in same wall clock — 3 more steep-descent epochs available.
- **NaN test_geom_camber_cruise:** this is the known IEEE-754 padded-batch issue. Already fixed in BF16 baseline via NaN-safe eval (#763). The rebase will inherit the fix and produce a clean 4-split test_avg.
- **Validated independent investigation by student:** ran `batch_size=1` inference on cruise to confirm the NaN is a padding-related issue in `PhysicsAttention` (no node mask in attention), not a Huber problem.
- **Implication for #885 (askeladd Huber-delta-sweep):** delta=1.0 on pre-merge code already wins by 18.5%; sweep delta ∈ {0.3, 0.5, 1.0, 2.0} on BF16 baseline becomes the natural follow-up (already in flight as #885).
- **Send-back instructions to frieren:** (1) rebase onto current `icml-appendix-willow-pai2e-r5` (BF16 + features + warmup); (2) re-run with same `delta=1.0`; (3) confirm the −18.5% holds on the merged baseline.

---

## 2026-04-28 23:40 — PR #739 (rebased): Huber loss δ=1.0 on BF16 baseline — **MERGED**

- **Branch:** `willowpai2e5-frieren/huber-loss` (merged)
- **W&B run:** `l95azbnv` (`willowpai2e5-frieren/huber-loss-d1.0-rebased`) — group `huber-loss`
- **Hypothesis:** Same as pre-rebase #739 above; now on merged BF16 + features + warmup baseline.

### Results (rebased onto BF16 baseline #811; best epoch 16 of 17 completed)

| Split | val/mae_surf_p | test/mae_surf_p |
|-------|----------------|-----------------|
| `val_single_in_dist` | 130.87 | 124.544 |
| `val_geom_camber_rc` | 115.14 | 99.385 |
| `val_geom_camber_cruise` | 92.61 | 80.195 |
| `val_re_rand` | 103.76 | 101.070 |
| **avg** | **110.594** | **101.299** |

vs BF16 baseline #811 (val_avg=127.40, test_avg=116.21):
- val_avg: **−13.2%**; test_avg: **−12.8%**
- Per-split improvement: rc −22.1% / sid −13.8% / re_rand −10.7% / cruise −1.2% (val)
- All 4 test splits finite (NaN-safe eval inherited from baseline)
- Peak VRAM 33.1 GB (unchanged), 30.0 min wall clock

### Commentary & Conclusions

- **Decision: Merged. New baseline.** Four compounding wins: distance features (#763) + warmup+cosine (#737) + BF16 (#811) + Huber δ=1.0 (#739). New floor: val_avg=110.594, test_avg=101.299.
- **Mechanism validated.** Huber caps gradient contribution of high-Re outlier samples (per-sample y_std varies 10×, surf_weight=10 amplifies the imbalance). All 4 splits improve, OOD splits (rc −22.1%, re_rand −10.7%) benefit most.
- **Gain is slightly attenuated vs pre-rebase (−13.2% vs −18.5%).** This is consistent with run-to-run variance (~8% across 3 Huber runs). The direction and approximate magnitude robustly replicate. Also, the rebased run started from a harder baseline (127.40 vs 141.42 before warmup).
- **Still timeout-limited at epoch 16/17.** Val curve descending at cutoff (epoch 15→16: 125 → 110). Convergence floor not reached.
- **`val_geom_camber_cruise` stagnated.** Only −1.2% val (vs −15% pre-rebase on older baseline). cruise test slightly worse (+1.4%). Frieren diagnosed the likely cause: `PhysicsAttention` distributes softmax mass over padded positions — cruise has the most geometric diversity (most variable mesh sizes) → worst padding ratio per batch. **Assigned frieren to fix this (#915).**
- **Askeladd #885 now the right follow-up.** Sweep δ ∈ {0.3, 0.5, 1.0, 2.0} on top of this merged baseline.

---

## 2026-04-29 00:05 — PR #878: DropPath/stochastic depth on residual branches — closed

- **Branch:** `willowpai2e5-nezuko/drop-path` (closed)
- **W&B runs:** `zrpxz35j` (control p=0.0), `vixyda0y` (p=0.1) — group `drop-path`
- **Hypothesis:** DropPath at p_max=0.1 (linear schedule across 5 layers) provides implicit ensembling regularization, frees per-step compute (skipped residual branches), and shifts best_epoch later → better generalization on undertrained model.

### Results (against BF16 baseline `newqt8dd`, before Huber merge)

| drop_path | val_avg/mae_surf_p (best) | test_avg/mae_surf_p | best_epoch | s/epoch | epochs in 30 min |
|-----------|--------------------------:|--------------------:|-----------:|--------:|-----------------:|
| **0.0 (control)** | **131.89** | **121.64** | 16 | 110.0 | **17** |
| 0.1 | 132.22 | 122.19 | 12 | 113.2 | 16 |

(both runs worse than current Huber baseline 110.594, but the comparison is internal to this PR)

### Commentary & Conclusions

- **Decision: Closed. Clean negative with excellent mechanistic analysis.** All three pillars of the hypothesis falsified.
- **No regularization benefit:** val_avg delta (+0.32) is well within seed noise (~4–5 units). Per-split redistribution exists (rc −10.9, cruise +14.6) but net-cancels. Implicit-ensembling claim does not survive at one-seed precision.
- **Wall-clock argument falsified:** per-epoch time *increased* by +3% (113.2 vs 110.0 s/epoch). The per-sample mask construction and division cost across 10 residual branches per step exceeds the autograd savings on n_layers=5 × n_hidden=128. DropPath wall-clock benefits show up at ViT scale (12+ layers, 384+ hidden), not here.
- **best_epoch moved earlier (16 → 12):** opposite direction from the predicted later-best-epoch signature. Adding noise to an undertrained model finds the optimum sooner and then plateaus — the opposite of "regularization extending useful training horizon".
- **Joint with #742 (per-activation dropout):** two negative results in a row with the same mechanistic root cause — the model is undertrained, not overfit. Standard transformer regularizers do not pay rent in this regime.
- **Student's diagnosis is correct:** the dominant constraint is wall-clock, not regularization. Their #1 follow-up suggestion (`torch.compile`) and the broader throughput direction are exactly right.
- **Nezuko reassigned** to vectorize `add_derived_features` (#923) — the per-sample CPU-sync Python loop identified by askeladd in #848 as the throughput bottleneck. This is an exact, contained, deterministic optimization that should free 5–15% wall-clock.

---

## 2026-04-29 00:14 — PR #850: Lower surf_weight sweep {3, 5, 7} — sent back for sw=3 + Huber stack

- **Branch:** `willowpai2e5-edward/lower-surf-weight` (sent back; result is internally clean but compared against pre-Huber baseline)
- **W&B runs:** `2sv6lptb` (sw=3), `ge7sjn6i` (sw=5), `rnhf5mmx` (sw=7) — group `lower-surf-weight`. **All three runs used `huber_delta=None` (MSE; pre-#739 merge code).**
- **Hypothesis:** Lowering surf_weight (counter to refuted #734 going up) lets the volume residuals contribute more spatial context, which the Transolver attention propagates back to surface predictions.

### Results (against BF16 baseline #811 val_avg=127.40, BEFORE Huber merged)

| sw | val_avg/mae_surf_p | test_avg/mae_surf_p | best epoch |
|----|--------------------:|--------------------:|-----------:|
| 10 (baseline) | 127.402 | 116.211 | 17 |
| **3** | **124.053 (-2.6%)** | **112.563 (-3.1%)** | 13 |
| 5 | 125.837 (-1.2%) | 115.176 (-0.9%) | 17 (still descending) |
| 7 | 142.777 (+12.0%) | 133.367 (+14.7%) | 17 (plateau) |

### Per-channel diagnostic (val avg)

| sw | surf_p ↓ | surf_Ux | surf_Uy | vol_p ↓ |
|----|---------:|--------:|--------:|--------:|
| 3 | **124.05** | 2.41 | 0.89 | **111.31** |
| 7 | 142.78 | **2.00** | **0.85** | 136.10 |

### Commentary & Conclusions

- **Decision: Sent back for re-run on Huber baseline.** The mechanism is real and partially complementary to Huber — needs a single decisive sw=3 + Huber run.
- **Mechanism validated within sweep:**
  1. Lower sw → substantial improvement in vol_p (sw=3: 111.31 vs sw=7: 136.10, -22%). Volume signal is informative for surface prediction; weakening it hurts surface predictions too.
  2. Counter-intuitive trade: surf_p improves at sw=3 but surf_Ux/Uy degrade. Pressure has fat-tailed magnitudes (high-Re outliers); strong surface emphasis amplifies gradient noise on `p` more than on velocity.
- **Stale baseline issue:** all three runs predate the Huber merge (sw=3 created at 22:25, Huber merged at 23:40). Current best is now 110.594, so sw=3's 124.05 looks worse than baseline. But Huber and sw-lowering attack different mechanisms: Huber caps high-Re gradient contribution, low-sw boosts volume informativeness. They MAY stack.
- **Send-back instructions:** Single re-run with `--surf_weight 3.0` on rebased branch (Huber δ=1.0 is now default). If beats 110.594 → merge as new baseline. If lands 105-115 → marginal. If >115 → Huber already captured this lever.
- **Future PR (per-channel surface weights):** student suggested `surf_weight_p=3, surf_weight_uv=10` to keep velocity accuracy while gaining pressure improvement — clever, assigned to frieren as #943.

---

## 2026-04-29 01:55 — PR #850: Lower surf_weight 10→3 on Huber+BF16 stack — **MERGED** (new best)

- **Branch:** `willowpai2e5-edward/lower-surf-weight` (merged)
- **W&B run:** `6rh7dzkx` — group `lower-surf-weight-huber-stack`
- **Hypothesis:** Lower `surf_weight` from 10 to 3 on the Huber+BF16 baseline. The mechanism: lower surface weight forces more gradient signal through volume residuals; Transolver cross-token attention propagates that volume information back to surface predictions, exploiting the global pressure-Poisson relationship.

### Results vs Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | Huber sw=10 | sw=3 + Huber | Δ val | test sw=10 | test sw=3 | Δ test |
|-------|------------:|-------------:|------:|-----------:|----------:|-------:|
| `single_in_dist`    | 130.87 | **120.51** | −7.92% | 124.544 | **102.846** | −17.4% |
| `geom_camber_rc`    | 115.14 | **107.95** | −6.24% | 99.385  | **94.352**  | −5.1%  |
| `geom_camber_cruise`| 92.61  | **82.16**  | −11.3% | 80.195  | **70.128**  | −12.5% |
| `re_rand`           | 103.76 | **95.64**  | −7.83% | 101.070 | **92.346**  | −8.6%  |
| **avg**             | **110.594** | **101.563** | **−8.17%** | **101.299** | **89.918** | **−11.24%** |

Best epoch = 17/17 (val still descending at 30-min timeout cutoff).

### Commentary & Conclusions

- **Decisive win — all 4 splits improved on both val and test.** Test improvements larger than val (e.g., sid −17.4% test vs −7.9% val), consistent with better generalization rather than noise.
- **Mechanisms stack orthogonally:** Huber caps gradient *magnitude* on high-Re outliers; lower sw rebalances surface vs volume *weight* in the loss. These are independent levers → compounding gains as predicted.
- **val curve was still descending at epoch 17** (trajectory: 230.5 → 165.0 → 147.8 → 128.2 → 124.9 → 143.2 → 110.7 → 101.56). Suggests further improvement possible with more budget or even lower sw.
- **PR diff was initially empty** (CLI-only run; student didn't update default). Sent back for one-line Config change (`surf_weight: float = 3.0`). Merged cleanly on resubmit.
- **Fifth compounding win stacked:** distance-features → warmup+cosine → BF16 → Huber → sw=3.
- **Follow-up: #953 (edward)** — sweep sw ∈ {0.5, 1.0, 2.0} to find floor below sw=3.

---

## 2026-04-29 02:25 — PR #885: Sweep Huber loss delta {0.3, 0.5, 1.0, 2.0} — sent back for rebase + stacking test

- **Branch:** `askeladd/huber-delta-sweep` (sent back; conflicts with merged Huber #739 and stale sw=10)
- **W&B runs:** `3yiixbyg` (δ=0.3), `vr6g2rxa` (δ=0.5), `295hulp0` (δ=1.0), `ki36m2z6` (δ=2.0) — group `huber-delta-sweep`
- **Hypothesis:** Sweep δ ∈ {0.3, 0.5, 1.0, 2.0} on the BF16 baseline to find the optimal Huber transition threshold. Smaller δ should help the heavy-tailed normalized residual distribution (high-Re outliers).

### Results vs pre-Huber MSE baseline (val_avg=127.402, test_avg=116.211, sw=10) — askeladd's sweep

| delta | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p |
|------:|:---------:|--------------------:|--------------------:|
| 2.0 | 17 | 113.804 | 102.353 |
| 1.0 | 17 | 115.386 | 104.471 |
| 0.5 | 16 | 107.271 | 97.437 |
| **0.3** | **16** | **97.963 (−23.1%)** | **87.785 (−24.5%)** |

### Compared to current merged baseline (#850, sw=3 + Huber δ=1.0: val_avg=101.563, test_avg=89.918)

- **δ=0.3 alone (sw=10) BEATS current best (sw=3 + δ=1.0)** by −3.5% val / −2.4% test in absolute terms.
- The δ lever appears stronger than the sw lever (δ sweep gave −17.4% val improvement at fixed sw=10, vs sw=3 → sw=10 giving −8.2% improvement at fixed δ=1.0).

### Commentary & Conclusions

- **Trend is monotone-with-noise:** δ {2.0 → 1.0} flat (within noise), but {1.0 → 0.5 → 0.3} clear monotone descent. Bottom not yet found.
- **Mechanism confirmed:** smaller δ pulls more outlier samples into the L1 regime, where gradient magnitudes are bounded. Most consistent gains are on `val_single_in_dist` and `val_geom_camber_rc` — splits with the heaviest-tailed residual distributions.
- **PR is in conflict with current advisor branch:** askeladd's branch was created before #739 (Huber) merged. Their diff re-adds Huber code that's now on main; combined with #850's sw=3 default change, the merge state is dirty.
- **Stacking question is open:** δ=0.3 (loss-shape lever) and sw=3 (loss-balance lever) attack different mechanisms, so they may stack additively. But both ultimately address outlier-driven instability — partial overlap is plausible.
- **Decision: Send back** for rebase + 2 decisive runs:
  1. `δ=0.3 + sw=3` — test stacking with current baseline.
  2. `δ=0.1 + sw=3` — continue the monotone trend test (askeladd suggested {0.1, 0.2}).
- **If δ=0.3 + sw=3 wins:** merge as new baseline. **If δ=0.1 also wins:** prefer whichever is lower.

---

## 2026-04-29 02:45 — PR #896: Per-sample y-normalization on Huber+sw=3 — closed (redundant with existing baseline)

- **Branch:** `alphonse/per-sample-y-normalization` (closed; ran clean on rebased Huber+sw=3 stack)
- **W&B run:** `ngaailh7` — `per-sample-y-norm-huber-sw3-clip1`
- **Hypothesis:** Normalize each sample's residual by per-sample sigma_per before loss, equalizing Re-regime contributions. This is a TARGET-SPACE fix; Huber is a LOSS-SPACE fix. Hypothesis: they're complementary.

### Results vs current baseline (#850: val_avg=101.563, test_avg=89.918)

| Split | baseline #850 | per-sample-norm + Huber + sw=3 | Δ val | test baseline | test + norm | Δ test |
|-------|-------------:|-------------------------------:|------:|--------------:|------------:|-------:|
| `single_in_dist`    | 120.507 | 127.759 | **+6.0%** | 102.846 | 116.274 | **+13.1%** |
| `geom_camber_rc`    | 107.951 | 120.592 | **+11.7%** | 94.352 | 110.080 | **+16.7%** |
| `geom_camber_cruise`| 82.156  | **63.843** | −22.3% | 70.128 | **53.730** | −23.4% |
| `re_rand`           | 95.636  | **88.545** | −7.4% | 92.346 | **82.658** | −10.5% |
| **avg**             | **101.563** | **100.185** | **−1.36%** | **89.918** | **90.686** | **+0.85%** |

### Commentary & Conclusions

- **Redistributive, not Pareto-improving.** Val/test directions disagree. The per-split flip pattern is exact: low-Re splits (cruise, re_rand) win massively (−22%, −10% test), high-Re splits (sid, rc) regress equally (+13%, +17% test). This confirms per-sample-norm and Huber+sw=3 attack the SAME underlying Re-imbalance problem from different sides, largely substituting for each other.
- **Val avg marginally better (−1.36%) but within seed noise.** Test avg slightly worse (+0.85%) — the paper-facing metric goes the wrong direction.
- **sigma_per stats confirm mechanism:** mean=0.336, min=0.082, max=0.700 in normalized space. Low-Re samples have ~4× smaller sigma than high-Re — confirming the imbalance the normalizer attacks. But Huber already caps high-Re gradient contribution, leaving per-sample-norm with little independent room.
- **Decision: Closed.** The mechanism works but the lever is saturated by the existing baseline. The earlier appearance of a +17%/+19% win (#896 vs #811) was a stale-baseline artifact.
- **Mechanism insight:** Target-space (sigma_per) and loss-space (Huber) approaches to Re-imbalance appear to be nearly equivalent substitutes when both are properly tuned. A hybrid approach (smaller Huber δ AND sigma-rescaling) might still add value if they're not fully equivalent — this is a Wave 3 question.
- **Follow-up assigned: #980 (alphonse)** — boundary-layer-weighted volume loss, a mechanistically distinct lever using dist_to_surface feature.

---

## 2026-04-29 03:00 — PR #923: Vectorize `add_derived_features` — merged (neutral throughput, clean code)

- **Branch:** `willowpai2e5-nezuko/vectorize-add-derived-features` (merged)
- **W&B runs:** A/B runs under group `vectorize-data-prep`; also tested literal B×N×N proposal
- **Hypothesis:** Per-sample CPU `.item()` syncs in `add_derived_features` are the throughput bottleneck. Removing them should give 5-15% wall-clock improvement and unblock batch-size scaling.

### Results

| Metric | A: per-sample loop | B: vectorized | Δ |
|---|---|---|---|
| `epoch_data_prep_ms_mean` | 26.20 ms | 25.78 ms | −1.6% (noise) |
| `epoch_time_s` | 109.53 s | 110.04 s | +0.5% (noise) |
| Total epochs in 30-min budget | 16 | 17 | +1 (likely noise) |
| Max abs numerical diff | — | 0.0 | exact |

Also tested literal B×N×N batched approach: **1.75× slower** (46 ms vs 26 ms per step) due to 50× more pairwise distance computations (N=242K vs s_b≈5K surface nodes).

### Commentary & Conclusions

- **Hypothesis refuted.** GPU pairwise compute dominates (≈22ms/26ms = 85%), not CPU syncs (≈4ms).
- **Full elimination of data_prep would save only ~1.4% of epoch_time** — within measurement noise.
- **Actual bottleneck: model forward+backward = ~91% of epoch time** (~100s/110s).
- **Merged for architectural cleanliness:** bit-exact implementation, removes `.item()` CPU syncs, cleaner code even if not faster currently. Future torch.compile benefits from sync-free data prep.
- **Bottleneck map: model FLOPs dominate.** torch.compile is the correct next throughput attack.
- **Follow-up assigned: #986 (nezuko)** — torch.compile(dynamic=True) targeting 1.2-1.5× model speedup.

---

## 2026-04-29 01:15 — PR #915: Mask padded nodes in PhysicsAttention slice aggregation — closed (mixed result)

- **Branch:** `willowpai2e5-frieren/physics-attention-padding-mask` (closed)
- **W&B run:** `msywsg7o`
- **Hypothesis:** Padded zero-vector nodes contaminate slice tokens via unmasked softmax in PhysicsAttention. Masking them out (post-softmax zero) should improve predictions, especially on cruise geometries with variable mesh sizes / high padding ratio.

### Results vs Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | val/mae_surf_p (base) | val/mae_surf_p (mask) | Δ val | test/mae_surf_p (base) | test/mae_surf_p (mask) | Δ test |
|-------|----------------------:|-----------------------:|------:|------------------------:|------------------------:|-------:|
| `single_in_dist`    | 130.87 | ~130.2 | ~−0.5% | 124.544 | ~127.7 | ~+2.5% |
| `geom_camber_rc`    | 115.14 | ~118.6 | ~+3.0% | 99.385  | ~130.0 | **+30.8%** |
| `geom_camber_cruise`| 92.61  | ~79.6  | **−14.1%** | 80.195  | ~68.7  | **−14.3%** |
| `re_rand`           | 103.76 | ~104.1 | ~+0.3% | 101.070 | ~102.5 | ~+1.4% |
| **avg**             | **110.594** | **~108.1** | **−0.6%** (noise) | **101.299** | **~107.2** | **+3.3%** |

(Approximate per-split numbers reconstructed from PR comment; W&B run `msywsg7o` confirmed.)

### Commentary & Conclusions

- **Mechanism confirmed on cruise** (−14.3% test, −14.1% val) exactly as predicted. Cruise has the most variable mesh sizes and highest padding ratio → most contamination from padded zero-nodes in the slice-softmax.
- **RC split regressed sharply** (+30.8% test). RC geometries (raceCar tandem, M=6-8) have denser, more uniform meshes → lower padding ratio → the hard binary post-softmax mask zeroes real attention weight, disrupting tandem-wake slice tokens.
- **Net aggregate:** val_avg −0.6% (within ~5-unit seed noise), test_avg +3.3% worse. The cruise gain and rc regression approximately cancel; aggregate is negative.
- **Why the binary mask fails on rc:** The fix patches symptom (padded nodes contaminating slices) but breaks mechanism (slice assignment flexibility) on dense-mesh geometries. A soft learnable gate (sigmoid(MLP(x))) would preserve attention on dense meshes while suppressing true padding — this is a Wave 3 idea.
- **Decision: Closed.** Mechanism insight is valuable but the binary mask is not a net improvement. Redirecting frieren to per-channel surface loss weighting (#943).

---

## 2026-04-29 01:20 — PR #896: Per-sample y-normalization — sent back for rebase on Huber baseline

- **Branch:** `willowpai2e5-alphonse/per-sample-y-normalization` (sent back, merge conflict with #739)
- **W&B run:** `5ihd38bk` (winning run: `per-sample-y-norm-clip1`)
- **Hypothesis:** Normalize each sample's residual by its per-sample standard deviation (sigma_per) before computing the loss, equalizing Re-regime contributions. High-Re samples (large sigma_per) are effectively down-weighted; low-Re samples (small sigma_per) get amplified. This is a target-space fix vs Huber's loss-space fix — potentially complementary.

### Results vs current Huber baseline (val_avg=110.594, test_avg=101.299)

| Split | Huber baseline | Per-sample-norm (MSE) | Δ vs Huber |
|-------|---------------:|----------------------:|------------|
| `single_in_dist`    | 130.87 | **131.105** | +0.2% |
| `geom_camber_rc`    | 115.14 | **128.262** | +11.4% |
| `geom_camber_cruise`| 92.61  | **72.292**  | **−21.9%** |
| `re_rand`           | 103.76 | **90.152**  | **−13.1%** |
| **val_avg**         | **110.594** | **105.453** | **−4.7%** |

| Split | Huber baseline | Per-sample-norm (MSE) | Δ vs Huber |
|-------|---------------:|----------------------:|------------|
| `test_single_in_dist`    | 124.544 | **117.075** | −6.0% |
| `test_geom_camber_rc`    | 99.385  | **111.894** | **+12.6%** |
| `test_geom_camber_cruise`| 80.195  | **60.451**  | **−24.6%** |
| `test_re_rand`           | 101.070 | **85.834**  | **−15.1%** |
| **test_avg**             | **101.299** | **93.814** | **−7.4%** |

Note: alphonse's PR compared against the pre-Huber #811 baseline (127.402), not the current Huber best. Even vs current Huber baseline, this is still a clear winner on average.

### Commentary & Conclusions

- **Per-sample-norm is a decisive win on cruise and re_rand** (both OOD Re splits). Cruise −24.6% test, re_rand −15.1% test — the Re-normalization is directly attacking the generalization failure mode.
- **RC regression at +12.6% test** is concerning. RC has dense meshes, tighter Re range — sigma_per may be less variable for RC samples, so per-sample-norm doesn't help and possibly adds noise.
- **Average win is clear** (−4.7% val, −7.4% test) despite the RC regression, because cruise and re_rand dominate by sheer magnitude.
- **Important:** ran WITHOUT Huber loss (huber_delta=None). The per-sample-norm mechanism supersedes Huber for Re-imbalance but may stack with it. Merge conflict with Huber code (different edit points in the loss computation) — sent back for rebase.
- **Grad_clip=1.0** added by student (undocumented in original PR instructions) — correct decision for stability with large per-sample weight variation.
- **Decision: Sent back for rebase on Huber baseline.** When rebased, instruct alphonse to stack both: the sq_err normalization by sigma_per should be applied to the huber_err tensor (not a raw sq_err) — i.e., compute `huber_err = F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)` then divide by `sigma_per.unsqueeze(1)` before weighting by surf/vol masks.
