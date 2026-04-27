# SENPAI Research State — icml-appendix-willow-pai2c-r4

- **Date:** 2026-04-27 18:30
- **Track:** willow r4 (`icml-appendix-willow-pai2c-r4`)
- **Latest human researcher direction:** none received yet; default per `target/program.md` — beat the seeded Transolver baseline on `val_avg/mae_surf_p` (and the test-time analogue `test_avg/mae_surf_p`).

## Current focus and themes

Round 1 (launch) dispatch: cast a deliberately wide net across orthogonal strategy tiers so we learn which family of changes has the most headroom on this dataset. We have no prior r4 evidence, so the priority is information gain, not chasing a single direction.

| Tier | Hypotheses |
|---|---|
| Loss formulation | H1 (re-conditional reweighting), H4 (per-channel uncertainty weighting) |
| Features (physical priors) | H5 (Re/Mach + sin/cos AoA features) |
| Architecture | H7 (deeper-narrower Transolver, 10 layers × 128), H12 (surface-aware attention bias) |
| Optimizer / schedule | H9 (OneCycle warmup), H10 (EMA weights) |
| Data augmentation | H11 (cruise-only z-reflection augmentation) |

Confidence ordering (from researcher-agent): H1 > H7 > H10 > H5 > H9 > H12 > H11 > H4. The full launch hypothesis list lives at `/workspace/senpai/research/RESEARCH_IDEAS_2026-04-27_18:30.md`.

## Potential next research directions

These are second-round candidates, conditional on round 1 outcomes:

- **Compounding winners.** If H1, H7, H10 all land — stack them in round 2. They are mechanistically independent.
- **If H1 wins big** — explore alternative reweighting schemes: per-sample y_std, log(Re), boundary-layer scaling 1/√Re, mixed.
- **If H7 wins** — push capacity further (n_layers 12-16, n_hidden 192) and pair with EMA + OneCycle for the regularization those configs need.
- **If camber holdouts diverge from in-dist** — broaden domain-aware augmentation: AoA jitter, Re jitter, MixUp on surface nodes, GeomCoT-style "interpolate through a known geometry" pseudo-samples.
- **If volume MAE diverges sharply** — switch to LayerNorm-stable training, gradient clipping, or revisit normalization (per-domain, not global stats).
- **Architectural shifts (round 3+ if plateaued):** GINO/MeshGraphNet hybrid, FNO-style spectral mixing, separate Ux/Uy/p decoders, surface-only readout.
- **Test-time inference improvements:** evaluation TTA via reflection ensembling on cruise samples, multi-checkpoint averaging.
- **Bigger swings (if round 1 plateaus):** Treat surface pressure as a separate task — train a lightweight surface-only specialist that consumes the volume model's hidden state.

## Constraints to respect

- 96 GB GPU VRAM, batch_size=4, meshes up to 242K nodes.
- `SENPAI_MAX_EPOCHS` and `SENPAI_TIMEOUT_MINUTES` envs are hard limits — innovate within them.
- No new packages outside `pyproject.toml`.
- Data loaders read-only — feature/sampler tweaks happen in `train.py`.
- Common-recipe changes preferred over single-split hacks.
