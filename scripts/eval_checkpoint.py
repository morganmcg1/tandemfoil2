"""Re-evaluate a saved checkpoint on val + test splits.

Stand-alone — does NOT import train.py (which has top-level argparse).

Usage:
  python scripts/eval_checkpoint.py \\
    --checkpoint models/model-<run_id>/checkpoint.pt \\
    [--config models/model-<run_id>/config.yaml] \\
    [--bf16] [--skip_test] [--out_json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    TEST_SPLIT_NAMES,
    VAL_SPLIT_NAMES,
    X_DIM,
    accumulate_batch,
    aggregate_splits,
    finalize_split,
    load_data,
    load_test_data,
    pad_collate,
)


# ---- Model definition (mirrors train.py exactly) -------------------------

ACTIVATION = {
    "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus, "ELU": nn.ELU, "silu": nn.SiLU,
}


class MLP(nn.Module):
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


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 ada_temp=False, gumbel_softmax=False):
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

        self.ada_temp = ada_temp
        self.gumbel_softmax = gumbel_softmax
        if ada_temp:
            self.ada_temp_proj = nn.Linear(dim_head, 1)
            nn.init.zeros_(self.ada_temp_proj.weight)
            nn.init.zeros_(self.ada_temp_proj.bias)

    def _slice_weights(self, x_mid):
        logits = self.in_project_slice(x_mid)
        if self.ada_temp:
            tau_offset = self.ada_temp_proj(x_mid)
            tau = F.softplus(self.temperature + tau_offset) + 1e-3
        else:
            tau = self.temperature.clamp(min=1e-3)
        scaled = logits / tau
        if self.gumbel_softmax and self.training:
            eps = torch.rand_like(scaled).clamp(min=1e-8, max=1.0)
            gumbel = -torch.log(-torch.log(eps))
            scaled = scaled + gumbel
        return self.softmax(scaled)

    def forward(self, x):
        B, N, _ = x.shape
        fx_mid = (self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous())
        x_mid = (self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous())
        slice_weights = self._slice_weights(x_mid)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 ada_temp=False, gumbel_softmax=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num,
                                     ada_temp=ada_temp, gumbel_softmax=gumbel_softmax)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, out_dim))

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields=None, output_dims=None,
                 ada_temp=False, gumbel_softmax=False):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
                            last_layer=(i == n_layers - 1),
                            ada_temp=ada_temp, gumbel_softmax=gumbel_softmax)
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


# ---- Eval helpers --------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, autocast_dtype=None):
    """Evaluate on a split. Filters non-finite samples per-batch so the official
    ``data/scoring.accumulate_batch`` doesn't NaN-poison the accumulators
    (its mask-based skip is broken: 0.0 * NaN = NaN). The intent — drop
    non-finite samples entirely — is preserved.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Per-organizer semantics: any sample with non-finite y is dropped
            # from the metric. Filter the batch so 0.0 * NaN doesn't poison sums.
            y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not y_finite.all():
                keep = torch.where(y_finite)[0]
                if keep.numel() == 0:
                    continue
                x = x[keep]; y = y[keep]
                is_surface = is_surface[keep]; mask = mask[keep]

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    pred = model({"x": x_norm})["preds"]
                pred = pred.float()
            else:
                pred = model({"x": x_norm})["preds"]

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    out = {
        "vol_loss": vol_loss_sum / max(n_batches, 1),
        "surf_loss": surf_loss_sum / max(n_batches, 1),
    }
    out["loss"] = out["vol_loss"] + surf_weight * out["surf_loss"]
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def print_split_metrics(split_name, m):
    print(
        f"    {split_name:<26s} loss={m['loss']:.4f}  "
        f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
        f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--surf_weight", type=float, default=10.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = args.config
    if cfg_path is None:
        sibling = Path(args.checkpoint).with_name("config.yaml")
        if sibling.exists():
            cfg_path = str(sibling)
    if cfg_path is None:
        model_config = dict(
            space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
            n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
            output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
        )
        print("[INFO] No config.yaml found; using baseline defaults.")
    else:
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)
        print(f"[INFO] Loaded config from {cfg_path}: {model_config}")

    train_ds, val_splits, stats, _ = load_data(args.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = Transolver(**model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={list(missing)[:3]}... unexpected={list(unexpected)[:3]}...")
    model.eval()
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    print("\n=== Validation ===")
    val_metrics = {
        name: evaluate_split(model, loader, stats, args.surf_weight, device, autocast_dtype=autocast_dtype)
        for name, loader in val_loaders.items()
    }
    val_agg = aggregate_splits(val_metrics)
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, val_metrics[name])
    print(f"  VAL  avg_surf_p={val_agg['avg/mae_surf_p']:.4f}")

    test_metrics = None
    test_agg = None
    if not args.skip_test:
        print("\n=== Test ===")
        test_datasets = load_test_data(args.splits_dir)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(model, loader, stats, args.surf_weight, device, autocast_dtype=autocast_dtype)
            for name, loader in test_loaders.items()
        }
        test_agg = aggregate_splits(test_metrics)
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        print(f"  TEST  avg_surf_p={test_agg['avg/mae_surf_p']:.4f}")

    if args.out_json:
        out = {
            "checkpoint": args.checkpoint, "config": cfg_path,
            "val_per_split": val_metrics, "val_avg": val_agg,
            "test_per_split": test_metrics, "test_avg": test_agg,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
