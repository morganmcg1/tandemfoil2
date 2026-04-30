"""Transolver model classes.

Decoupled from train.py so the module can be imported without launching a run.
The original Transolver baseline (n_hidden=128, n_layers=5, n_head=4,
slice_num=64, mlp_ratio=2) is unchanged. Optional flags toggle a Transolver++
style "eidetic state" slice attention (Gumbel-Softmax sampling + per-point
adaptive temperature) — disabled by default so the baseline path is intact.

Reference for eidetic states / Rep-Slice / Ada-Temp:
  Wu et al., Transolver++: An Accurate Neural Solver for PDEs on
  Million-Scale Geometries (arxiv 2502.02414, Feb 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
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
    """Physics-aware attention for irregular meshes.

    By default uses the original Transolver slice-softmax with one shared
    learnable temperature. With ``use_eidetic=True`` it switches to the
    Transolver++ eidetic-state formulation:

      - per-point learnable temperature (``Ada-Temp``)
      - Gumbel-Softmax sampling of slice weights during training (``Rep-Slice``)

    During eval the eidetic mode falls back to a plain softmax with the
    per-point temperatures so val/test metrics remain deterministic.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_eidetic: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.use_eidetic = use_eidetic
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        # Per-point temperature offset -> Ada-Temp.
        if use_eidetic:
            self.in_project_temp = nn.Linear(dim_head, 1)
            nn.init.zeros_(self.in_project_temp.weight)
            nn.init.zeros_(self.in_project_temp.bias)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _slice_weights(self, x_mid: torch.Tensor) -> torch.Tensor:
        logits = self.in_project_slice(x_mid)
        if not self.use_eidetic:
            return self.softmax(logits / self.temperature)
        # Ada-Temp: per-point temperature = base_temp + softplus(linear(x_mid))
        # Use softplus to keep tau strictly positive.
        tau_extra = F.softplus(self.in_project_temp(x_mid))  # (B, H, N, 1)
        tau = self.temperature + tau_extra
        if self.training:
            # Rep-Slice via Gumbel reparameterization.
            eps = torch.empty_like(logits).uniform_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(eps))
            return F.softmax((logits + gumbel) / tau, dim=-1)
        return F.softmax(logits / tau, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        slice_weights = self._slice_weights(x_mid)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        use_eidetic: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_eidetic=use_eidetic,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
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
    def __init__(
        self,
        space_dim: int = 1,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
        use_eidetic: bool = False,
        output_fields: list[str] | None = None,
        output_dims: list[int] | None = None,
    ):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.use_eidetic = use_eidetic
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
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                use_eidetic=use_eidetic,
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
