# Round 1 Research Ideas — `icml-appendix-charlie-pai2-r1`

**Date:** 2026-04-27
**Branch base:** vanilla `train.py` (Config: `lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10.0, epochs=50`; model_config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`).
**Primary metric:** `val_avg/mae_surf_p` (lower is better). Vanilla expected ~90–100. Prior `icml-appendix-charlie` round terminal best: ~49 val / ~42 test.

The 8 proposals below are sized so any one student can complete them inside a single 30-min / 50-epoch budget with one 96 GB GPU. Each hypothesis is **single-knob** so the lever can be attributed cleanly. Compounding will happen across rounds via merges.

Mix:
- **1** vanilla calibration baseline (H1)
- **3** re-validation of prior wins as orthogonal levers (H2 surf_weight+L1, H3 Fourier PE, H4 slice_num↓)
- **4** fresh angles not tried in the prior round (H5 EMA, H6 Huber/SmoothL1 with regression label smoothing, H7 per-channel decoder heads, H8 geometric data augmentation: x-axis flip + Galilean shift)

---

## H1. Vanilla calibration baseline

**Hypothesis:** A clean run of the unmodified train.py (vanilla `Config`) on this fresh branch produces `val_avg/mae_surf_p ≈ 90–100` and a matching `test_avg/mae_surf_p`, establishing a verified anchor for every other Round-1 hypothesis to compare against. Without it, every "delta" we cite later is unverified, which is paper-poison.

**Specific change:** None. Run `python train.py --epochs 50 --agent <student> --experiment_name r1_vanilla_baseline`. Do **not** modify train.py. Commit only the `models/.../metrics.{jsonl,yaml}` and `config.yaml` artifacts.

**Rationale:** The prior round table is informational only — those merges are *not* on this branch. Reviewers and the advisor need a verified vanilla anchor in the metrics log on **this** branch before any improvement can be claimed. It also surfaces hardware/environment drift (driver, torch, dataloader) that could otherwise be misread as "improvement."

**Risk:** Anchoring run consumes a GPU-slot that could test a new idea. Mitigated because every later PR's delta is meaningless without it; this is the cheapest insurance we can buy.

---

## H2. surf_weight=1 with L1 loss

**Hypothesis:** Replacing MSE with L1 on the per-node residual and dropping `surf_weight` from 10 to 1 (removing the redundant up-weighting of surface nodes that L1's robustness already provides) drops `val_avg/mae_surf_p` from ~95 to ~88–93 — a re-validation of prior PR #11 as a stand-alone lever before it is stacked.

**Specific change:** In `train.py` lines 447 and 453, replace `sq_err = (pred - y_norm) ** 2` with `abs_err = (pred - y_norm).abs()`, propagate to `vol_loss`/`surf_loss` computations on lines 451–452 (and the matching block in `evaluate_split` lines 243–253). Set `surf_weight: float = 1.0` on the Config dataclass line 354. CLI: `python train.py --epochs 50 --surf_weight 1.0 --agent <student> --experiment_name r1_l1_sw1`.

**Rationale:** MAE is the **evaluation metric** — training MSE is a surrogate that overpenalises high-Re wake outliers and ignores small surface residuals. L1 is the natural Bayes-optimal target for MAE. `surf_weight=10` was a hack to compensate for MSE's bias toward volume nodes (which dominate by count); under L1 that hack becomes harmful.

**Risk:** L1 has no curvature at zero, so AdamW's effective step on near-converged residuals shrinks; if cosine LR ends too low, convergence may stall in last 5 epochs. Easy fallback is SmoothL1 (H6).

---

## H3. Fourier positional encoding on (x, z)

**Hypothesis:** Augmenting the input feature vector with a fixed random Fourier feature embedding of node coordinates `(x, z)` (σ=0.7, m=160, sin+cos → +320 features) drops `val_avg/mae_surf_p` from ~95 to ~78–85 by giving the network high-frequency spatial basis it cannot synthesize from raw `(x, z)` alone.

**Specific change:** Add a `FourierFeatures` module in `train.py` (above the `Transolver` class) that registers a buffer `B ~ N(0, σ²) ∈ R^{2 × 160}` (seeded). In the training and eval forward path, replace `x_norm = ...` with concatenating `[sin(2π x_xz @ B), cos(2π x_xz @ B)]` to the existing 24 features → 344-dim input. Bump `model_config["fun_dim"]` to `X_DIM - 2 + 320` (i.e. 22 + 320 = 342). Add `--fourier_sigma 0.7 --fourier_m 160` flags. CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_fourier_s07_m160`.

**Rationale:** The mesh is irregular and Transolver attention operates on slice tokens, so coordinate sensitivity comes only from the 2-d input — a known low-frequency bottleneck (Tancik et al. 2020). σ=0.7 was the prior round's best (PR #24).

**Risk:** σ too high → spatial overfitting to training meshes and worse OOD camber; σ already validated, but on a different stack.

---

## H4. slice_num = 16 (down from 64)

**Hypothesis:** Reducing Transolver `slice_num` from 64 to 16 drops `val_avg/mae_surf_p` from ~95 to ~75–85 by forcing each slice token to carry more global context, which prior round PR #34 showed compounds especially well with shallower stacks.

**Specific change:** Edit `model_config` dict in `train.py` line 396, change `slice_num=64` to `slice_num=16`. CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_sn16`.

**Rationale:** Physics attention groups N nodes into G slices; smaller G = bigger groups = more receptive-field per token = stronger global flow context. The dataset has long-range pressure coupling between two foils; large G fragments this. The prior round walked sn=64 → 32 → 16 → 8 monotonically downward, suggesting fewer/wider slices is a robust direction.

**Risk:** Too few slices may collapse expressivity for high-Re samples with strong wake structure. Prior data says sn=8 was still improving; 16 is a conservative re-validation point.

---

## H5. Exponential Moving Average of weights (EMA)

**Hypothesis:** Maintaining an EMA of the model parameters with decay 0.999 and using **EMA weights for validation and test** drops `val_avg/mae_surf_p` from ~95 to ~80–88 by averaging out late-training oscillations that cosine LR creates near minimum.

**Specific change:** In `train.py`, add an EMA wrapper after model construction (`ema_model = copy.deepcopy(model); ema_decay = 0.999`). After each `optimizer.step()` (line 457), update EMA: `for p, ep in zip(model.parameters(), ema_model.parameters()): ep.mul_(0.999).add_(p.data, alpha=0.001)`. Replace `model.eval()` with `ema_model.eval()` in the validation path (line 468) and use `ema_model` in `evaluate_split` calls. Save `ema_model.state_dict()` as the checkpoint (line 485). CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_ema_999`.

**Rationale:** EMA is an architecture-agnostic free regularizer ubiquitous in modern training (BYOL, EDM diffusion, ML-pose). The prior round did not test it. It interacts well with cosine annealing because the LR-decay tail is exactly where EMA averaging captures stable minima. No extra compute cost (one extra fwd is not needed because EMA shares forward semantics).

**Risk:** EMA decay of 0.999 with only 50 epochs (~375 steps at bs=4) is borderline — if the warm-up phase is long, EMA lags. Mitigation built-in: only validate with EMA after epoch 5.

---

## H6. SmoothL1 (Huber) loss with β=0.05 and 1% regression label smoothing

**Hypothesis:** Replacing MSE with Huber loss (`F.smooth_l1_loss`, β=0.05 in normalized space ≈ 0.05·y_std physical units) plus adding 1% Gaussian noise on the **target** (not input) during training drops `val_avg/mae_surf_p` from ~95 to ~83–90 by giving large-residual outliers an L1 tail (Bayes-aligned with MAE) while keeping near-zero quadratic curvature for fine convergence — and the label noise softens hard targets near boundaries where the dataset has its highest aleatoric uncertainty.

**Specific change:** In `train.py` line 447, replace `sq_err = (pred - y_norm) ** 2` with `err = F.smooth_l1_loss(pred, y_norm + 0.01 * torch.randn_like(y_norm), beta=0.05, reduction='none')`. Use `err` instead of `sq_err` in vol/surf loss accumulation (lines 451–452). In `evaluate_split` keep `(pred - y_norm)**2` for monitoring loss only, but the training step uses smooth L1. CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_smoothl1_b005_lsm`.

**Rationale:** Huber is the canonical compromise between MSE (smooth gradients, MAE-misaligned) and L1 (MAE-aligned, kink at 0). β=0.05 on normalized targets ≈ 0.05·y_std physical units, so for `val_re_rand` high-Re samples this is meaningful. Regression label smoothing (Gaussian target noise) is well-known in tabular regression but has *not* been tried on this benchmark, and it directly addresses the "near-boundary aleatoric" failure mode of CFD surrogates.

**Risk:** β too small → effectively L1 (which H2 already tests, redundant). β too large → effectively MSE. Mitigated by choosing β=0.05 explicitly between the two regimes.

---

## H7. Per-channel decoder heads with channel-specific normalization

**Hypothesis:** Replacing the single 3-output `mlp2` decoder in the last `TransolverBlock` with three separate per-channel heads (one each for Ux, Uy, p) plus a learned per-channel output scale drops `val_avg/mae_surf_p` from ~95 to ~80–88 by letting the pressure head specialize without competing with velocity gradients in the shared output linear.

**Specific change:** In `train.py` `TransolverBlock` class (lines 139–164), replace `self.mlp2` with `nn.ModuleDict({"Ux": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)), "Uy": ..., "p": ...})`. In `forward`, when `last_layer`, return `torch.cat([self.mlp2["Ux"](h), self.mlp2["Uy"](h), self.mlp2["p"](h)], dim=-1)`. CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_per_channel_heads`.

**Rationale:** Pressure has very different spatial spectrum and dynamic range from velocity (kinematic pressure spans ~30k vs. velocity O(100)). A shared linear forces the encoder to compromise. Per-channel heads is a tiny parameter increase (~3× the 256→3 head, negligible) but unblocks specialization. Untried in prior round, which only manipulated the shared decoder.

**Risk:** With shared trunk and very small (1499-sample) train set, the additional head capacity might overfit. Mitigated because the heads are still very thin (one hidden layer each).

---

## H8. Geometric data augmentation: x-axis mirror + Galilean (x, z)-shift

**Hypothesis:** Augmenting each training sample with a 50% probability x-axis mirror (negate `x`-coord, negate `Ux` target, negate AoA features 14 and 18, leave camber/Re alone) and an independent uniform Galilean shift in `(x, z)` of ±0.5 (subtract from coords only — pressure and velocity are translation-invariant) drops `val_avg/mae_surf_p` from ~95 to ~82–90 by exposing the model to a richer geometric distribution without changing the loss or model.

**Specific change:** Add an `_augment(x, y, is_surface, mask)` function in `train.py` (above the train loop). It takes a batch, with prob 0.5 negates `x[..., 0]` (node x-coord), `x[..., 14]` and `x[..., 18]` (AoAs), `y[..., 0]` (Ux), and adds independent `U(-0.5, 0.5)` shifts to `x[..., 0:2]` (node coords, both `x` and `z`). Apply only in the training loop (line 438), not validation. CLI: `python train.py --epochs 50 --agent <student> --experiment_name r1_aug_mirror_galilean`.

**Rationale:** The 2-d Navier–Stokes equations are exactly equivariant to x-axis reflection (with appropriate sign flips) and to constant Galilean translation in space. So mirrored / shifted samples are physically valid CFD solutions the model is currently missing — pure data augmentation, no architecture change. With only 1499 train samples this should help generalization on the 3 held-out tracks (camber_rc, camber_cruise, re_rand), which are exactly where current OOD MAE is highest.

**Risk:** AoA sign-flip semantics depend on the tandem convention (whether AoA is per-foil absolute or relative). Mitigated by mirroring **both** AoAs simultaneously, which is unambiguously valid. Galilean shift is unambiguously safe because all targets are translation-invariant.

---

## Summary of expected `val_avg/mae_surf_p` deltas

| H | Change | Predicted | Confidence |
|---|--------|-----------|------------|
| 1 | vanilla calibration | 90–100 | anchor |
| 2 | L1 + sw=1 | 88–93 | high (PR #11 prior) |
| 3 | Fourier σ=0.7 m=160 | 78–85 | high (PR #24 prior) |
| 4 | slice_num=16 | 75–85 | high (PR #34 prior) |
| 5 | EMA decay 0.999 | 80–88 | medium-high |
| 6 | SmoothL1 β=0.05 + label smoothing | 83–90 | medium |
| 7 | Per-channel heads | 80–88 | medium |
| 8 | Mirror + Galilean aug | 82–90 | medium-high |

After Round 1 we will have 1 anchor + 3 re-validated levers + 4 novel signals. Whichever subset wins will then compound in Round 2.
