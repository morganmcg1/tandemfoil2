"""Re-evaluate a saved model checkpoint on test splits with NaN-safe handling.

Usage:
  python scripts/reeval_test.py --checkpoint models/model-<run_id>/checkpoint.pt \
      --n_hidden 192 --n_head 6 --n_layers 5 --slice_num 64 --mlp_ratio 2

Outputs per-split surface MAE and an aggregate test_avg/mae_surf_p computed over
the splits whose surface-pressure MAE is finite (NaN splits are excluded from
the aggregate but reported in the per-split breakdown).

This script does NOT modify scoring.py (read-only). It applies torch.nan_to_num
to predictions before scoring, so non-finite predictions contribute |y_mean - y|
rather than NaN.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure we can import data
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_test_data,
    pad_collate,
)


# --- Inline copy of Transolver model classes (vendored from train.py to avoid
#     simple_parsing import side-effects). Keep in sync with train.py if model
#     code changes. ---
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

ACTIVATION = {
    "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1), "softplus": nn.Softplus, "ELU": nn.ELU, "silu": nn.SiLU,
}


class _MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class _PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape
        fx_mid = (self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False,
        )
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class _TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = _PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                      dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = _MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0, n_head=8,
                 act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1, slice_num=32, ref=8,
                 unified_pos=False, output_fields=None, output_dims=None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        if self.unified_pos:
            self.preprocess = _MLP(fun_dim + ref**3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = _MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            _TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                             act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                             slice_num=slice_num, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


def _load_stats(splits_dir: Path) -> dict:
    with open(splits_dir / "stats.json") as f:
        raw = json.load(f)
    return {k: torch.tensor(raw[k], dtype=torch.float32) for k in ("x_mean", "x_std", "y_mean", "y_std")}


def _accumulate_batch_safe(
    pred_orig: torch.Tensor,
    y: torch.Tensor,
    is_surface: torch.Tensor,
    mask: torch.Tensor,
    mae_surf: torch.Tensor,
    mae_vol: torch.Tensor,
) -> tuple[int, int]:
    """Re-implements ``data.scoring.accumulate_batch`` semantics but without
    NaN propagation when ground truth contains non-finite values.

    Identical math: per-sample skip on non-finite y; float64 accumulation.
    Difference: y is sanitized before the err multiply so NaN*0 stays 0.
    """
    B = y.shape[0]
    y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
    if not y_finite.any():
        return 0, 0
    # Replace non-finite y with 0 ONLY where the sample is to be skipped anyway.
    y_safe = torch.where(
        y_finite[:, None, None].expand_as(y),
        y,
        torch.zeros_like(y),
    )
    sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
    effective = mask & sample_mask
    surf_mask = effective & is_surface
    vol_mask = effective & ~is_surface
    err = (pred_orig.double() - y_safe.double()).abs()
    mae_surf += (err * surf_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    mae_vol += (err * vol_mask.unsqueeze(-1).double()).sum(dim=(0, 1))
    return int(surf_mask.sum().item()), int(vol_mask.sum().item())


def evaluate_split_safe(model, loader, stats, surf_weight, device) -> dict[str, float]:
    """Like evaluate_split in train.py but uses the NaN-safe accumulator."""
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0
    n_nonfinite_pred = 0
    n_nonfinite_samples = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            # For loss-side normalization, also sanitize so NaN doesn't poison
            # surf_loss/vol_loss. Loss numbers shouldn't matter for ranking.
            y_safe_for_loss = torch.where(
                torch.isfinite(y), y, torch.zeros_like(y),
            )
            y_norm = (y_safe_for_loss - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]

            n_bad = (~torch.isfinite(pred)).sum().item()
            n_nonfinite_pred += n_bad
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            # Count samples with any non-finite y (these are skipped by the accumulator)
            B = y.shape[0]
            sample_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            n_nonfinite_samples += int((~sample_finite).sum().item())
            ds, dv = _accumulate_batch_safe(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {
        "vol_loss": vol_loss,
        "surf_loss": surf_loss,
        "loss": vol_loss + surf_weight * surf_loss,
        "n_nonfinite_pred": n_nonfinite_pred,
        "n_nonfinite_y_samples": n_nonfinite_samples,
    }
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model state_dict")
    ap.add_argument("--config_yaml", default=None, help="Path to model config yaml (auto from checkpoint dir if not given)")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--also_val", action="store_true", help="Also evaluate val splits")
    ap.add_argument("--out_json", default=None, help="Write results JSON to this path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve config yaml
    cfg_path = Path(args.config_yaml) if args.config_yaml else Path(args.checkpoint).parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config yaml not found at {cfg_path}")
    with open(cfg_path) as f:
        model_config = yaml.safe_load(f)
    print(f"Model config: {model_config}")

    # Build model and load weights
    model = Transolver(**model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

    splits_dir = Path(args.splits_dir)
    stats = _load_stats(splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)

    results = {}
    if args.also_val:
        print("\n--- Validation splits ---")
        # Load val splits
        from data.loader import SplitDataset
        for name in VAL_SPLIT_NAMES:
            ds = SplitDataset(splits_dir / name)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            m = evaluate_split_safe(model, loader, stats, args.surf_weight, device)
            results[f"val/{name}"] = m
            print(f"  {name}: surf_p={m['mae_surf_p']:.4f} (nonfinite={m['n_nonfinite_pred']})")

    print("\n--- Test splits ---")
    test_datasets = load_test_data(splits_dir, debug=False)
    test_metrics: dict[str, dict[str, float]] = {}
    for name in TEST_SPLIT_NAMES:
        loader = DataLoader(test_datasets[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_split_safe(model, loader, stats, args.surf_weight, device)
        test_metrics[name] = m
        print(f"  {name}: surf_p={m['mae_surf_p']:.4f}  vol_p={m['mae_vol_p']:.4f}  "
              f"surf_Ux={m['mae_surf_Ux']:.4f}  surf_Uy={m['mae_surf_Uy']:.4f}  "
              f"(nonfinite={m['n_nonfinite_pred']})")

    test_avg = aggregate_splits(test_metrics)
    print(f"\nTest avg/mae_surf_p (over all 4 splits, NaN replaced with 0): {test_avg.get('avg/mae_surf_p'):.4f}")

    # Also report robust test_avg excluding splits with non-finite predictions
    finite_metrics = {n: m for n, m in test_metrics.items() if m["n_nonfinite_pred"] == 0}
    if len(finite_metrics) < 4:
        finite_avg = aggregate_splits(finite_metrics)
        print(
            f"Test avg/mae_surf_p (excluding {4 - len(finite_metrics)} splits with non-finite preds): "
            f"{finite_avg.get('avg/mae_surf_p'):.4f}"
        )

    results["test"] = test_metrics
    results["test_avg"] = test_avg

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote results to {args.out_json}")


if __name__ == "__main__":
    main()
