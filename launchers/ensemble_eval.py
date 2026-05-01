#!/usr/bin/env python3
"""Ensemble evaluation: load multiple Transolver checkpoints, average predictions
in normalized space, and compute val + test MAE matching the organizer scorer.

Usage:
  python launchers/ensemble_eval.py \
    --checkpoints models/model-RUN_ID_1/checkpoint.pt \
                  models/model-RUN_ID_2/checkpoint.pt \
                  ... \
    --config_yaml models/model-RUN_ID_1/config.yaml  # all members must share arch
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
    SPLITS_DIR, TEST_SPLIT_NAMES, VAL_SPLIT_NAMES, X_DIM,
    accumulate_batch, aggregate_splits, finalize_split,
    load_data, load_test_data, pad_collate,
)

# ---------------------------------------------------------------------------
# Model definition (mirrors train.py — kept self-contained to avoid importing
# the train.py top-level argparse logic).
# ---------------------------------------------------------------------------

ACTIVATION = {
    "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1), "softplus": nn.Softplus, "ELU": nn.ELU,
    "silu": nn.SiLU,
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
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token); k = self.to_k(slice_token); v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

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
                 output_fields=None, output_dims=None):
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
            TransolverBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
                            mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num,
                            last_layer=(i == n_layers - 1))
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


def evaluate_ensemble(models, loader, stats, device) -> dict:
    """Per-batch ensemble: average predictions from all models in normalized space,
    then denormalize and accumulate MAE. Matches the organizer scorer."""
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            pred_sum = None
            for m in models:
                p = m({"x": x_norm})["preds"]
                pred_sum = p if pred_sum is None else pred_sum + p
            pred = pred_sum / len(models)
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    return finalize_split(mae_surf, mae_vol, n_surf, n_vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--config_yaml", required=False)
    ap.add_argument("--splits_dir", default=str(SPLITS_DIR))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.config_yaml is None:
        first = Path(args.checkpoints[0]).parent / "config.yaml"
        if not first.exists():
            raise SystemExit("No config_yaml provided and no sibling config.yaml found")
        args.config_yaml = str(first)
    with open(args.config_yaml) as f:
        model_config = yaml.safe_load(f)

    print(f"Model config: {model_config}")
    print(f"Loading {len(args.checkpoints)} checkpoints…")

    models = []
    for ck in args.checkpoints:
        m = Transolver(**model_config).to(device).eval()
        sd = torch.load(ck, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=True)
        models.append(m)
        print(f"  loaded {ck}")

    splits_dir = Path(args.splits_dir)
    train_ds, val_splits, stats, _ = load_data(splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True)

    print("\n=== Ensemble validation ===")
    val_metrics = {}
    for name in VAL_SPLIT_NAMES:
        loader = DataLoader(val_splits[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_ensemble(models, loader, stats, device)
        val_metrics[name] = m
        print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]")

    val_avg = aggregate_splits(val_metrics)
    print(f"\n  VAL  avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

    test_metrics = None
    test_avg = None
    if not args.skip_test:
        print("\n=== Ensemble test ===")
        test_datasets = load_test_data(splits_dir)
        test_metrics = {}
        for name in TEST_SPLIT_NAMES:
            loader = DataLoader(test_datasets[name], batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            m = evaluate_ensemble(models, loader, stats, device)
            test_metrics[name] = m
            print(f"  {name:<26s} surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]")
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")

    out = {
        "n_members": len(models),
        "checkpoints": list(args.checkpoints),
        "model_config": model_config,
        "val_per_split": val_metrics,
        "val_avg": val_avg,
    }
    if test_metrics is not None:
        out["test_per_split"] = test_metrics
        out["test_avg"] = test_avg

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
