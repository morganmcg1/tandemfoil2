"""Split MAE accumulation matching ``data/score_split`` semantics.

Semantics pinned to what the organizer scoring script (``score.py``) computes
against the hidden test ground truth:

- float64 accumulation for numerical stability over many samples
- per-sample skip for any case whose ground truth contains non-finite values
- global (node-level) aggregation: surface MAE is ``sum|err| / n_surf_nodes``,
  not a per-sample average

The same helpers are used for validation during training and for the end-of-run
test evaluation, so train/val/test metrics are computed identically.
"""

from __future__ import annotations

import torch

CHANNELS = ("Ux", "Uy", "p")


def accumulate_batch(
    pred_orig: torch.Tensor,
    y: torch.Tensor,
    is_surface: torch.Tensor,
    mask: torch.Tensor,
    mae_surf: torch.Tensor,
    mae_vol: torch.Tensor,
) -> tuple[int, int]:
    """Add a padded batch's surface/volume MAE contribution into the accumulators.

    ``pred_orig`` and ``y`` are in the original (denormalized) output space.
    Shapes: all are ``[B, N_max, ...]``; ``mask`` marks valid nodes.
    Samples whose ground-truth is non-finite anywhere are skipped entirely.

    Returns ``(n_surf_added, n_vol_added)``.
    """
    B = y.shape[0]
    y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
    if not y_finite.any():
        return 0, 0

    sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])  # [B, N]
    effective = mask & sample_mask
    surf_mask = effective & is_surface
    vol_mask = effective & ~is_surface

    err = (pred_orig.double() - y.double()).abs()
    # Guard against NaN in GT (e.g. sample 20 of test_geom_camber_cruise has 761
    # non-finite pressure nodes).  IEEE-754 NaN * 0 = NaN, so the masked sum
    # would poison the accumulator even for correctly-excluded samples.
    err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
    mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    return int(surf_mask.sum().item()), int(vol_mask.sum().item())


def finalize_split(
    mae_surf: torch.Tensor,
    mae_vol: torch.Tensor,
    n_surf: int,
    n_vol: int,
) -> dict[str, float]:
    """Divide accumulators by node counts to get per-channel MAE."""
    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    out: dict[str, float] = {}
    for i, ch in enumerate(CHANNELS):
        out[f"mae_surf_{ch}"] = s[i].item()
        out[f"mae_vol_{ch}"] = v[i].item()
    return out


def aggregate_splits(per_split: dict[str, dict[str, float]]) -> dict[str, float]:
    """Equal-weight mean across splits of each per-channel MAE.

    Primary ranking metric: ``avg/mae_surf_p`` (surface pressure MAE
    averaged with equal weight across the four split tracks).
    """
    out: dict[str, float] = {}
    keys = [f"mae_{loc}_{ch}" for loc in ("surf", "vol") for ch in CHANNELS]
    for k in keys:
        vals = [m[k] for m in per_split.values() if k in m]
        if vals:
            out[f"avg/{k}"] = sum(vals) / len(vals)
    return out
