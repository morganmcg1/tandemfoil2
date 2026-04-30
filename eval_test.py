"""Standalone test-set evaluator.

Loads a saved checkpoint (``checkpoint.pt`` from a ``models/model-<run_id>``
directory) and the matching ``config.yaml``, then evaluates on the four
held-out val and test splits using the exact ``data.scoring`` semantics
the trainer uses for ``test_avg/mae_surf_p``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader

from data import (
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


# ----- Reproduce the model classes from train.py (must stay in sync) -----

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


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64,
                 ada_temp=False, rep_slice=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.ada_temp = ada_temp
        self.rep_slice = rep_slice
        if ada_temp:
            self.temp_delta = nn.Linear(dim_head, 1, bias=True)
            nn.init.zeros_(self.temp_delta.weight)
            nn.init.zeros_(self.temp_delta.bias)
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
        fx_mid = (
            self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3).contiguous()
        )
        x_mid = (
            self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3).contiguous()
        )
        slice_logits = self.in_project_slice(x_mid)
        temp = self.temperature
        if self.ada_temp:
            delta = self.temp_delta(x_mid)
            temp = (temp + delta).clamp(min=1e-2)
        if self.rep_slice and self.training:
            eps = 1e-9
            u = torch.rand_like(slice_logits).clamp_(eps, 1 - eps)
            gumbel = -torch.log(-torch.log(u))
            slice_logits = slice_logits + gumbel
        slice_weights = self.softmax(slice_logits / temp)
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


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 ada_temp=False, rep_slice=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, ada_temp=ada_temp, rep_slice=rep_slice,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

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
                 ada_temp=False, rep_slice=False):
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
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                ada_temp=ada_temp, rep_slice=rep_slice,
            )
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


# ----- Eval helper -----

def evaluate_split(model, loader, stats, surf_weight, device):
    """Same semantics as train.py evaluate_split."""
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
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]
            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += ((sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)).item()
            surf_loss_sum += ((sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)).item()
            n_batches += 1
            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            # Workaround for NaN * 0 = NaN: replace bad samples' y with 0 and
            # mask them out at the per-sample level so accumulate_batch skips them.
            y_finite_b = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            if not y_finite_b.all():
                y_clean = torch.where(
                    y_finite_b.view(-1, 1, 1).expand_as(y), y, torch.zeros_like(y)
                )
                mask_clean = mask & y_finite_b.unsqueeze(-1)
            else:
                y_clean, mask_clean = y, mask
            ds, dv = accumulate_batch(pred_orig, y_clean, is_surface, mask_clean,
                                       mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv
    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss, "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def print_split_metrics(split_name, m):
    print(
        f"    {split_name:<26s} loss={m['loss']:.4f}  "
        f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
        f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--splits_dir", type=str,
                   default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    p.add_argument("--surf_weight", type=float, default=30.0)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    cfg_path = args.config or args.checkpoint.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Model config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    test_splits = load_test_data(args.splits_dir, debug=False)

    model = Transolver(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transolver ({n_params/1e6:.2f}M params)")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=False, prefetch_factor=2)

    print("\n=== Val ===")
    val_metrics = {}
    for name in VAL_SPLIT_NAMES:
        ds = val_splits[name]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_split(model, loader, stats, args.surf_weight, device)
        val_metrics[name] = m
        print_split_metrics(name, m)
    val_avg = aggregate_splits(val_metrics)
    print(f"\n  VAL  avg_surf_p = {val_avg['avg/mae_surf_p']:.4f}")

    print("\n=== Test ===")
    test_metrics = {}
    for name in TEST_SPLIT_NAMES:
        ds = test_splits[name]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        m = evaluate_split(model, loader, stats, args.surf_weight, device)
        test_metrics[name] = m
        print_split_metrics(name, m)
    test_avg = aggregate_splits(test_metrics)
    print(f"\n  TEST avg_surf_p = {test_avg['avg/mae_surf_p']:.4f}")


if __name__ == "__main__":
    main()
