"""Post-training per-channel residual loss-magnitude diagnostic.

Computes |residual| in normalized space across all 4 val splits, aggregates
linear vs quadratic fractions per channel (Ux, Uy, p) and per split, at
thresholds 0.05 and 0.10. Writes JSON next to the checkpoint.

Model classes are inlined below (copied from train.py) so importing this
module does not trigger train.py's script-level training loop.

Usage:
    python loss_magnitude_diag.py --model_dir models/<exp> [--compile]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
    VAL_SPLIT_NAMES,
    X_DIM,
    load_data,
    pad_collate,
)


ACTIVATION = {
    "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1), "softplus": nn.Softplus, "ELU": nn.ELU, "silu": nn.SiLU,
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


class SwiGLU_MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, mlp_ratio: int):
        super().__init__()
        intermediate = int(n_hidden * mlp_ratio * 2 / 3)
        intermediate = ((intermediate + 7) // 8) * 8
        self.intermediate = intermediate
        self.w1 = nn.Linear(n_input, intermediate, bias=False)
        self.w2 = nn.Linear(n_input, intermediate, bias=False)
        self.w3 = nn.Linear(intermediate, n_output, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 slice_temp_init: float = 2.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * slice_temp_init)
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
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 drop_path: float = 0.0, slice_temp_init: float = 2.0):
        super().__init__()
        self.last_layer = last_layer
        self.drop_path = drop_path
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, slice_temp_init=slice_temp_init)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = SwiGLU_MLP(hidden_dim, hidden_dim, hidden_dim, mlp_ratio=mlp_ratio)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim))

    def forward(self, fx):
        if self.training and self.drop_path > 0.0 and not self.last_layer:
            if torch.rand(1, device=fx.device).item() < self.drop_path:
                return fx
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False, drop_path_max: float = 0.0,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        drop_rates = torch.linspace(0.0, drop_path_max, n_layers).tolist()
        slice_temp_inits = torch.linspace(1.5, 3.0, n_layers).tolist()
        print(f"Per-block slice_temp_init schedule: {slice_temp_inits}")
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                drop_path=drop_rates[i], slice_temp_init=slice_temp_inits[i])
            for i in range(n_layers)])
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", default=4, type=int)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out_name", default="loss_magnitude.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_dir / "checkpoint.pt"
    out_path = args.model_dir / args.out_name

    _, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}

    model_config = dict(
        space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
        n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2,
        drop_path_max=0.1,
        output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    )
    model = Transolver(**model_config).to(device)
    if args.compile:
        model = torch.compile(model, mode="default")
    load_target = model._orig_mod if args.compile else model
    load_target.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    loader_kwargs = dict(num_workers=4, pin_memory=True, collate_fn=pad_collate,
                         persistent_workers=True, prefetch_factor=2)

    thresholds = [0.05, 0.10]
    channels = ["Ux", "Uy", "p"]

    per_split = {}
    agg_total = torch.zeros(3, dtype=torch.float64, device=device)
    agg_lin = {t: torch.zeros(3, dtype=torch.float64, device=device) for t in thresholds}

    for name in VAL_SPLIT_NAMES:
        loader = DataLoader(val_splits[name], batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs)
        split_total = torch.zeros(3, dtype=torch.float64, device=device)
        split_lin = {t: torch.zeros(3, dtype=torch.float64, device=device)
                     for t in thresholds}

        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]
            res = (pred - y_norm).abs()  # [B, N, 3]
            B = y.shape[0]
            y_finite_per_sample = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            elem_finite = torch.isfinite(res) & y_finite_per_sample[:, None, None]
            valid = mask.unsqueeze(-1) & elem_finite  # [B, N, 3]
            res = torch.where(valid, res, torch.zeros_like(res))
            count = valid.sum(dim=(0, 1)).to(torch.float64)
            split_total += count
            for t in thresholds:
                lin = ((res >= t) & valid).sum(dim=(0, 1)).to(torch.float64)
                split_lin[t] += lin

        agg_total += split_total
        for t in thresholds:
            agg_lin[t] += split_lin[t]

        per_channel_split = {}
        for ci, ch in enumerate(channels):
            tot = float(split_total[ci].item())
            per_channel_split[ch] = {
                "n_points": tot,
                **{f"lin_frac_thr_{t}": float(split_lin[t][ci].item()) / max(tot, 1.0)
                   for t in thresholds},
            }
        tot_all = float(split_total.sum().item())
        per_channel_split["aggregate"] = {
            "n_points": tot_all,
            **{f"lin_frac_thr_{t}": float(split_lin[t].sum().item()) / max(tot_all, 1.0)
               for t in thresholds},
        }
        per_split[name] = per_channel_split

    overall_per_channel = {}
    for ci, ch in enumerate(channels):
        tot = float(agg_total[ci].item())
        overall_per_channel[ch] = {
            "n_points": tot,
            **{f"lin_frac_thr_{t}": float(agg_lin[t][ci].item()) / max(tot, 1.0)
               for t in thresholds},
        }
    tot_all = float(agg_total.sum().item())
    overall_per_channel["aggregate"] = {
        "n_points": tot_all,
        **{f"lin_frac_thr_{t}": float(agg_lin[t].sum().item()) / max(tot_all, 1.0)
           for t in thresholds},
    }

    out = {
        "thresholds": thresholds,
        "per_split": per_split,
        "overall_per_channel": overall_per_channel,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(overall_per_channel, indent=2))


if __name__ == "__main__":
    main()
