# TandemFoilSet Dataset Analysis

**Compiled:** 2026-04-23 by advisor  
**Source:** program.md, data/SPLITS.md, train.py

---

## Overview

TandemFoilSet is a 2D CFD surrogate benchmark covering three physical domains:
- **RaceCar single** (inverted airfoil, ground effect): 899 samples, ~85K nodes/mesh
- **RaceCar tandem** (dual inverted foils, ground): 1200 samples across Parts 1–3, ~127K nodes
- **Cruise tandem** (dual freestream foils): 900 samples across Parts 1–3, ~210K nodes

Training set: ~1499 samples. Val/test: 4 × 100/200 samples each.

---

## Mesh Structure

Each sample is an **overset mesh** with up to 3 zones:
- Zone 0: coarse background (full domain)
- Zone 1: dense zone around foil 1
- Zone 2: dense zone around foil 2 (0 for single-foil)

Node count ranges 74K–242K per sample. `is_surface` flags foil-surface nodes (True) vs. all others (False). No foil-1/foil-2 distinction in the flag — both foil surfaces are merged.

**Implication:** The model must implicitly learn which surface node belongs to which foil from geometry features (dims 18–23 are zero for foil 2 in single-foil cases).

---

## Input Feature Analysis (24-dim)

| Dims | Feature | Key Properties |
|------|---------|----------------|
| 0–1 | Node (x, z) position | Physical coordinates; range varies by domain |
| 2–3 | Signed arc-length (saf) | Encodes position along surface; 0 for volume nodes |
| 4–11 | Distance-based shape descriptor (dsdf, 8-dim) | Multi-scale distance from surface; rich spatial context |
| 12 | is_surface (0/1) | Binary; ~1% of nodes are surface |
| 13 | log(Re) | Range: log(100K)–log(5M) = 11.5–15.4 |
| 14 | AoA foil 1 (radians) | RaceCar: [-10°, 0°]; Cruise: [-5°, +6°] |
| 15–17 | NACA foil 1 (M, P, T) | Normalized [0,1]; M=0–9 across training |
| 18 | AoA foil 2 (radians) | 0 for single-foil; same range as foil 1 for tandem |
| 19–21 | NACA foil 2 (M, P, T) | 0,0,0 for single-foil |
| 22 | Gap between foils | 0 for single-foil; ~[-0.8, 1.6] for tandem |
| 23 | Stagger between foils | 0 for single-foil; ~[0.0, 2.0] for tandem |

**Key observations:**
- Dims 18–23 are all zero for single-foil samples — the model can infer domain membership from this.
- `log(Re)` is a critical conditioning variable: per-sample y_std varies by 10× within a domain due to Re range.
- NACA M (camber) is the primary OOD generalization axis in the geometry camber splits.
- The dsdf (8-dim) encodes multi-scale shape context — likely the most information-rich geometric feature.

---

## Target Distribution

| Split | Re range | y range (min, max) | Avg y_std | Max y_std |
|-------|----------|--------------------|-----------|-----------|
| val_single_in_dist | 104K–5M | (-29,136, +2,692) | 458 | 2,077 |
| val_geom_camber_rc | 1.0M–5M | (-10,312, +2,228) | 377 | 1,237 |
| val_geom_camber_cruise | 122K–5M | (-7,648, +2,648) | 164 | 506 |

**Key observations:**
1. **Heavy tails:** High-Re samples drive extreme p values (-29K). MSE loss will over-weight these outliers. → Motivates Huber/L1 loss (see frieren's experiment).
2. **Per-sample scale varies 10×:** log(Re) must be used as a conditioning signal, not just a feature. The model needs to learn to scale its predictions with Re.
3. **Cruise has lower magnitude** (~3× smaller y_std than raceCar). Per-domain or per-sample normalization may help.

---

## Generalization Challenges

### 1. Geometry OOD (val_geom_camber_rc / _cruise)
- Training sees raceCar M=2–5 and M=9; val/test requires M=6–8 interpolation.
- Training sees cruise M=0–2 and M=4–6; val/test requires M=2–4 interpolation.
- The model must learn smooth camber interpolation from the NACA M feature (dim 15).
- The rear foil is always one of 30 shared rear foils — only front foil camber varies.

### 2. Reynolds Number OOD (val_re_rand)
- Stratified holdout across full Re range from all tandem domains.
- The tandem P2 files (rc tandem: Re 1–5M only; cruise: Re 110K–5M) are fully held out for the geometry splits, so val_re_rand draws from P1+P3+P4+P6.
- High-Re generalization is likely the hardest axis given the 10× scale variation.

### 3. Domain Transfer (single → tandem)
- The model sees 599 single-foil samples at train time. Tandem-specific tandem flow features (wake interaction, downwash between foils) must be inferred from the geometry features.
- Dims 18–23 (foil-2 features) encode the tandem configuration; the model must "activate" different behavior based on whether they're zero or nonzero.

---

## Training Set Domain Balance

The `WeightedRandomSampler` in train.py assigns equal effective probability mass to each domain (raceCar single ~599 samples, raceCar tandem ~682 samples, cruise tandem ~218 samples). Without this:
- RaceCar single would see 3× more gradient signal.
- Cruise tandem (smallest domain) would be underfit.

**Implication:** Balanced sampling is already implemented. The main lever here is whether the weighting scheme is optimal vs. e.g. inverse-frequency or loss-proportional upsampling.

---

## Surface vs. Volume Node Ratio

Surface nodes are ~1–2% of total nodes (foil surface in a 2D mesh). surf_weight=10 in the loss approximately compensates for this 1:100 volume-to-surface ratio. However:
- The ranking metric is **surface pressure MAE only** — so over-weighting surface could be justified.
- Current surf_weight=10 may be under-weighting relative to the metric importance.
- The Huber/L1 sweep includes surf_weight ∈ {5, 10, 20} to probe this.

---

## Key Questions for Future Experiments

1. Is the dsdf feature fully utilized, or could additional geometric features (e.g., curvature, wall normal) help?
2. Does per-sample (Re-conditional) normalization beat global y_mean/y_std?
3. Can potential flow theory predictions (panel method) serve as a physics-informed prior for initialization?
4. Does EMA of model weights reduce noise in OOD splits at eval time?
5. Is gradient clipping needed for stability at high-Re (large-scale outlier) batches?
6. Can we use the `is_surface` flag more aggressively — e.g., a dual-decoder (surface/volume heads) with different capacity allocation?
