# SENPAI Research State

- **Date:** 2026-04-28 09:30
- **Advisor branch:** `icml-appendix-charlie-pai2d-r5`
- **Cohort:** charlie-pai2d-r5 (8 students, 1 GPU each)
- **Most recent human-team direction:** none on file.

## Current best (from BASELINE.md)

| metric | value | source |
|---|---:|---|
| `val_avg/mae_surf_p` | **56.19** | PR #612 (alphonse, Lion optimizer lr=3e-4) — merged 2026-04-28 |
| `test_avg/mae_surf_p` (3-split mean) | 53.33 | PR #612 |

Per-split val: `val_single_in_dist=60.30`, `val_geom_camber_rc=71.06`, `val_geom_camber_cruise=37.01`, `val_re_rand=56.41`.

Full reference config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`, `fun_dim=54`, `lr=3e-4` (peak, linear warmup 5 ep), `weight_decay=1e-4`, `batch_size=4`, `surf_weight=30.0`, `grad_clip_norm=0.5`, `amp_bf16=True`, L1 loss, `optimizer_name="lion"`, SequentialLR(LinearLR warmup × 5 ep, CosineAnnealingLR T_max=epochs−5), `--epochs 24`, 8-band Fourier features on normalized (x, z).

## Round-7 status (on Lion baseline)

### WIP (on Lion baseline)
| PR | Axis | Student | Hypothesis |
|----|------|---------|------------|
| #680 | Optimizer LR | thorfinn | Lion + lr=3e-4 → 1e-4 (conservative LR probe) |
| #679 | Capacity × optim × schedule | fern | n_layers=6 + Lion + budget-matched cosine (--epochs 20) |
| #665 | Optimizer tuning | alphonse | Lion + grad_clip_norm 0.5 → 1.0 (loosened clip) |
| #628 | Optimizer | tanjiro | Lookahead wrapping Lion (k=5, α=0.5) — rebasing |
| #627 | Architecture | edward | Preprocess MLP depth +1 hidden residual — rebasing to Lion |
| #593 | Data aug | nezuko | Re jittering (σ=0.05 on log(Re)) at training time |
| #380 | Checkpoint | frieren | Best-val checkpoint averaging (top-3) on Lion base |
| #369 | Regularization | askeladd | Drop-path 0.1 — needs rebase to Lion |

All 8 students are running; no idle slots.

## Key findings to date (on Lion base)

1. **Lion optimizer = dominant finding.** Sign-based update composes naturally with L1 loss (unit-magnitude gradients) + grad-clip-0.5 + bf16. −23.3% val/test improvement vs AdamW. Mechanistically: AdamW's RMSProp normalization was redundant/noisy under L1 loss; Lion's sign(momentum) is exactly what RMSProp converges to in this regime, but cleaner.

2. **Budget-matched cosine schedule mechanism confirmed.** PR #637 (alphonse) and PR #632 (fern) independently cross-validated: tightening T_max to match actually-reachable epochs gives stronger late-epoch LR settling. Confirmed diagnostic; improvement sub-threshold.

3. **Capacity dimensions:**
   - Width (n_hidden) and per-block MLP (mlp_ratio): refuted — train_loss didn't drop with extra capacity.
   - Depth (n_layers=6): train-val gap does close; per-epoch wall +19% limits reachable epochs. Needs retest on Lion baseline.

4. **AoA jittering axis closed.** AoA is regime-controlling (affects flow topology), not a scaling parameter; σ=0.02 rad acts as label noise. All splits regressed.

5. **Re jittering (nezuko, #593) still in flight.** log(Re) is a flow-scaling variable — physically distinct from AoA. Result pending.

6. **Schedule and optimizer exhausted on AdamW.** EMA, higher LR, eta_min, LLRD all closed on the old base. On Lion base, optimizer interactions may be different.

## Priority hypotheses for next assignments

1. **n_layers=6 + Lion + budget-matched cosine** — natural extension of two confirmed sub-threshold experiments. Depth capacity was productive per-epoch on AdamW+bf16; Lion's smooth trajectory may compound with the matched schedule. 2 runs recommended (variance documented: same-config Δ=5.2 pts).

2. **Learning rate sweep on Lion** — Lion was tested at lr=3e-4 (the canonical 3× AdamW scaling). lr=1e-4 and lr=5e-4 are the natural adjacent probes. lr=5e-4 risks instability (Lion momentum EMA accumulates fast); lr=1e-4 may be too conservative. One direction per student.

3. **Weight decay on Lion** — wd=1e-4 was the AdamW default. Lion has a different weight decay mechanism (decoupled decay applied to the momentum buffer). Probing wd=1e-5 (less decay) or wd=5e-4 (more) may interact differently with Lion's sign-based updates.

4. **Lookahead wrapping Lion** — tanjiro has this PR in flight already (#628, rebasing). Lion's sign-updates may interact unusually with Lookahead's slow-weight averaging since Lookahead interpolates toward sign-constrained updates.

5. **Re jittering result first** — wait for nezuko's #593 before assigning more augmentation PRs. If Re jittering helps, paired AoA+Re jittering or other input-noise variants may be worth revisiting.

6. **Architecture: attention head scaling** — n_head=4 with slice_num=64. Doubling n_head to 8 (more fine-grained attention) or halving to 2 (coarser) has not been tested on Lion. Orthogonal to width/depth.

7. **Fourier feature band count** — currently 8 bands. 4 or 16 bands have not been tested on the current stack. Band count controls the frequency resolution of position encoding.

## Current research direction

Post-Lion, the track is exploring what stacks with the optimizer change. The dominant questions are:
- Does depth (n_layers=6) benefit from Lion's smoother trajectory? (Schedule-matching confirmed sub-threshold on AdamW; now test on Lion.)
- Does loosening grad_clip_norm with Lion unlock more momentum contribution? (alphonse #665 testing.)
- Do Lookahead or other optimizer wrappers add value on top of Lion? (tanjiro #628 rebasing.)
- Is data augmentation viable? (nezuko #593 Re jittering pending; AoA jittering closed.)
- Do preprocess MLP architectural changes stack with Lion? (edward #627 rebasing.)

## Potential next research directions (medium-term)

- **Learning rate schedule variants on Lion**: cosine with restarts (SGDR), OneCycleLR, or warmup restarts. Lion was tested only with linear-warmup + cosine anneal.
- **Mixed precision optimizations**: bf16 is active but the accumulator path may have more to give. Test full bf16 accumulation vs current fp32-accumulator mode.
- **Attention mechanism variants**: PhysicsAttention's SliceAttention is the key novel component. Ablating the number of slices or the projection dimensions has not been explored.
- **Physics-informed loss**: adding a divergence or curl penalty to the loss for velocity fields (Ux, Uy) has not been tried.
- **Data-efficient training**: the WeightedRandomSampler equalizes across 3 domains. Alternative: proportional sampling (domain-proportional), or hard-negative sampling on worst-performing validation samples.
- **Test-time ensembling**: averaging predictions from best-val checkpoints from multiple training runs with different seeds (distinct from checkpoint averaging within a single run).

## Known issue

`test_geom_camber_cruise/mae_surf_p` returns NaN for every PR. `data/scoring.py` is read-only. Rank by 3-clean-split test mean.

## Operational notes

- All work targets `icml-appendix-charlie-pai2d-r5`; new branches check out from it; merges squash back into it.
- Per-PR JSONL committed under `models/<experiment>/metrics.jsonl`; centralized into `/research/EXPERIMENT_METRICS.jsonl`; reviews logged in `/research/EXPERIMENTS_LOG.md`.
- No W&B / external loggers — local JSONL only.
- Per-epoch grad-norm telemetry (`train/grad_norm_avg`) is in the merged train.py.
- n_layers=6 experiments: budget 2 runs (same-config Δ=5.2 pts variance documented).
