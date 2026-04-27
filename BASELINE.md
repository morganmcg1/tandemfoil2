# Baseline Metrics — icml-appendix-charlie-r4

## TandemFoilSet

- **Primary validation metric:** `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across the four val splits)
- **Primary test metric:** `test_avg/mae_surf_p`
- **Status:** No published baseline yet on this advisor branch — Round 1 experiments will establish the vanilla anchor.

### Vanilla anchor configuration (target/train.py defaults)

- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR with `T_max=epochs`
- Model: Transolver
  - `n_hidden=128`, `n_layers=5`, `n_head=4` (dim_head=32)
  - `slice_num=64`, `mlp_ratio=2`
  - `act="gelu"`, `dropout=0.0`, `space_dim=2`, `unified_pos=False`
- Loss: MSE in normalized space; `loss = vol_loss + surf_weight * surf_loss`
- Sampler: WeightedRandomSampler with balanced domain weights

Reproduce vanilla:
```bash
cd target && python train.py --epochs 50 --batch_size 4
```

(Update this file with the new best `val_avg/mae_surf_p` after the first
review-ready PR comes in.)
