"""Tests for ``data.scoring`` accumulators.

Pinned bug: when the ground-truth ``y`` contains non-finite values (e.g. ``inf``
in the pressure channel of a single test sample), the previous implementation
multiplied ``err * mask_float`` and IEEE-754 ``inf * 0.0 = NaN`` poisoned the
float64 MAE accumulator for the whole split. The fix uses ``torch.where`` so
masked-out positions never participate in the multiplication.
"""

from __future__ import annotations

import torch

from data.scoring import CHANNELS, accumulate_batch, finalize_split


def _make_batch(B: int, N: int, C: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(B, N, C, generator=g, dtype=torch.float32)
    y = torch.randn(B, N, C, generator=g, dtype=torch.float32)
    mask = torch.ones(B, N, dtype=torch.bool)
    is_surface = torch.zeros(B, N, dtype=torch.bool)
    is_surface[:, : N // 2] = True
    return {"pred": pred, "y": y, "mask": mask, "is_surface": is_surface}


def test_accumulators_finite_when_y_has_inf():
    """A single sample with inf in p must not poison the accumulator."""
    B, N, C = 3, 8, 3
    batch = _make_batch(B, N, C)
    p_idx = CHANNELS.index("p")
    # Inject inf into sample 1's p channel — mirrors the test_geom_camber_cruise case.
    batch["y"][1, :, p_idx] = float("inf")

    mae_surf = torch.zeros(C, dtype=torch.float64)
    mae_vol = torch.zeros(C, dtype=torch.float64)

    n_surf, n_vol = accumulate_batch(
        batch["pred"], batch["y"], batch["is_surface"], batch["mask"], mae_surf, mae_vol
    )

    assert torch.isfinite(mae_surf).all(), f"mae_surf has NaN/Inf: {mae_surf}"
    assert torch.isfinite(mae_vol).all(), f"mae_vol has NaN/Inf: {mae_vol}"
    # The inf sample contributes zero nodes (skipped by y_finite); 2 good samples remain.
    assert n_surf == 2 * (N // 2)
    assert n_vol == 2 * (N - N // 2)


def test_accumulators_finite_when_y_has_nan():
    """NaN in y is the same class of bug as inf — guard both."""
    B, N, C = 2, 6, 3
    batch = _make_batch(B, N, C)
    batch["y"][0, 3, 0] = float("nan")

    mae_surf = torch.zeros(C, dtype=torch.float64)
    mae_vol = torch.zeros(C, dtype=torch.float64)

    accumulate_batch(
        batch["pred"], batch["y"], batch["is_surface"], batch["mask"], mae_surf, mae_vol
    )

    assert torch.isfinite(mae_surf).all()
    assert torch.isfinite(mae_vol).all()


def test_finite_data_matches_reference():
    """On all-finite inputs, the masked sum must equal the naive ``err * mask`` sum."""
    B, N, C = 4, 10, 3
    batch = _make_batch(B, N, C)
    # Mark a few positions as padding to exercise the mask.
    batch["mask"][0, 8:] = False
    batch["mask"][2, 9:] = False

    mae_surf = torch.zeros(C, dtype=torch.float64)
    mae_vol = torch.zeros(C, dtype=torch.float64)
    accumulate_batch(
        batch["pred"], batch["y"], batch["is_surface"], batch["mask"], mae_surf, mae_vol
    )

    err = (batch["pred"].double() - batch["y"].double()).abs()
    effective = batch["mask"]
    surf_mask = effective & batch["is_surface"]
    vol_mask = effective & ~batch["is_surface"]
    expected_surf = (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    expected_vol = (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))

    torch.testing.assert_close(mae_surf, expected_surf)
    torch.testing.assert_close(mae_vol, expected_vol)


def test_finalize_split_finite_metrics():
    """End-to-end: inf in a sample yields finite finalized per-channel MAE."""
    B, N, C = 3, 8, 3
    batch = _make_batch(B, N, C)
    batch["y"][1, :, CHANNELS.index("p")] = float("inf")

    mae_surf = torch.zeros(C, dtype=torch.float64)
    mae_vol = torch.zeros(C, dtype=torch.float64)
    n_surf, n_vol = accumulate_batch(
        batch["pred"], batch["y"], batch["is_surface"], batch["mask"], mae_surf, mae_vol
    )
    out = finalize_split(mae_surf, mae_vol, n_surf, n_vol)
    for k, v in out.items():
        assert v == v and v != float("inf") and v != float("-inf"), f"{k} not finite: {v}"
