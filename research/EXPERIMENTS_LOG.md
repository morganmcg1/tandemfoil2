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
