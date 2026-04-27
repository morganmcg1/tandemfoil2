# SENPAI Research State

- **Date:** 2026-04-27
- **Most recent research direction from human team:** None (no new GitHub issues)
- **Advisor track:** kagent_v_students (fork: morganmcg1/TandemFoilSet-Balanced)

## Current Research Focus

**Primary metric:** val_avg/mae_surf_p (lower is better)
**Current best (PR #39):** val 49.077/49.443 2-seed mean, test 42.473/42.450 2-seed mean
**Current default config:** nl=3, sn=8, nh=4, σ=0.7, SwiGLU, L1, AMP, grad_accum=4

### Active Research Themes

1. **Slice-number reduction axis (sn=8 → sn=4 → lower):** PR #43 (nezuko WIP) is probing sn=4, nl=2/sn=8, nl=1. The monotonic trend has not found its floor at sn=8.

2. **n_head reduction axis (nh=4 → nh=2 → nh=1):** PR #32 confirmed nh=1 wins at sn=32/nl=3. The KEY UNTESTED compound is **nh=1 × sn=8/nl=3** — this could be the next major gain.

3. **Depth reduction axis (nl=3 → nl=2 → nl=1):** PR #43 (nezuko) also probing nl=2/sn=8 and nl=1. nl=2 single seed was 50.72 at sn=16.

4. **Regularization (Dropout/DropPath):** PR #42 (tanjiro WIP) — overfitting concern with more epochs from throughput unlocks.

5. **Width reduction (n_hidden shrink):** PR #41 (fern WIP) — n_hidden ∈ {64, 96, 128, 160} at nl=3/sn=32.

6. **LR warmup recovery:** PR #40 (frieren WIP) — testing if warmup + cosine-floor helps on nl=3 recipe.

7. **Surface decoder (PhysicsAttention):** PR #29 (thorfinn WIP) — slice-bottleneck residual decoder.

8. **EMA + grad clipping:** PR #8 (edward WIP) — stability improvements.

## Active WIP PRs

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #43 | nezuko | sn=4/nl=2 floor probe | WIP |
| #42 | tanjiro | Dropout+DropPath sweep | WIP |
| #41 | fern | n_hidden shrink {64-160} | WIP |
| #40 | frieren | LR warmup nl=3 | WIP |
| #29 | thorfinn | Slice-bottleneck decoder | WIP |
| #8 | edward | EMA + grad clip | WIP |

## Idle Students

- **askeladd** — needs new assignment

## Potential Next Research Directions

### Highest Priority (highest expected gain)

1. **nh=1 × sn=8 × nl=3 compound** — most obvious untested gain; PR #32 showed nh=1 wins at sn=32, PR #39 showed sn=8 wins at nh=4. Combining should be strongly super-additive based on prior compound behavior.

2. **nh=1 × sn=4 or sn=2 × nl=3 probe** — if the sn floor isn't found at sn=8, push to sn=4 with nh=1 simultaneously.

3. **nh=1 × nl=2 × sn=8 triple compound** — nl=2 improved at sn=16 (50.72 single seed). At nh=1 the VRAM/throughput advantage is larger.

### Medium Priority

4. **Fourier σ sweep at new config (nh=1/sn=8/nl=3)** — the σ=0.7 optimum was found at nh=4/sn=64/nl=5; landscape may have shifted significantly.

5. **n_hidden expansion at nh=1** — nh=1/dh=64 control shows the win is not capacity-driven, but growing n_hidden (e.g., 192 or 256) with nh=1 could give a pure capacity gain.

6. **Loss reformulation** — all current gains are configuration/architecture; the L1+surf_weight=1 formulation hasn't been fundamentally revisited since PR #11.

### Exploratory

7. **Physics-informed loss** — add boundary condition enforcement term (no-slip at surface, continuity equation residual) as auxiliary loss.

8. **Fourier m sweep at current config** — m=160 was set at nl=5; fewer layers may benefit from higher or lower m.

9. **Geometry encoder pre-training** — train a separate foil geometry encoder and inject as conditioning. Could improve geom_camber_rc OOD generalization.

## Key Methodological Notes

- **Noise floor (σ=0.7 family, nl=3):** 
  - sn=8/nh=4: std ~0.517 val (2-seed, PR #39)
  - sn=32/nh=2: std ~0.479 val (3-seed, PR #32)
  - sn=32/nh=1: std ~0.920 val (2-seed, PR #32) — higher noise at fewer heads
  - Merge threshold: winner 2/3-seed mean must beat baseline by >1σ of fresh anchor
- **Always run --debug pre-sweep verification** — confirm all config flags correct before committing GPU time
- **Bit-exact determinism confirmed** multiple times across PR #24, #27, #32, #35
