"""Train a Transolver surrogate on TandemFoilSet.

Four validation tracks with one pinned test track each:
  val_single_in_dist      / test_single_in_dist      — in-distribution sanity
  val_geom_camber_rc      / test_geom_camber_rc      — unseen front foil (raceCar)
  val_geom_camber_cruise  / test_geom_camber_cruise  — unseen front foil (cruise)
  val_re_rand             / test_re_rand             — stratified Re holdout

Primary ranking metric is ``avg/mae_surf_p`` — the equal-weight mean surface
pressure MAE across the four splits, computed in the original (denormalized)
target space. Train/val/test MAE all flow through ``data.scoring`` so the
numbers are produced identically.

Usage:
  python train.py [--debug] [--epochs 50] [--agent <name>] [--wandb_name <name>]
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

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

# ---------------------------------------------------------------------------
# Transolver model
# ---------------------------------------------------------------------------

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


class FourierEncoder(nn.Module):
    """Random Fourier Features (Tancik et al. 2020) for (x, z) coordinates.

    Produces [sin(2π xy·Bᵀ), cos(2π xy·Bᵀ)] with B ∈ R^{m×2}. Output dim = 2m.
    When ``learnable=True``, B is an ``nn.Parameter``; otherwise a fixed buffer.

    Isotropic case: both columns of B share bandwidth σ (``sigma``).
    Per-coordinate case: when ``sigma_x`` and ``sigma_z`` are both supplied,
    column 0 of B is scaled by σ_x and column 1 by σ_z.
    """

    def __init__(self, m: int = 10, sigma: float = 1.0, learnable: bool = False,
                 sigma_x: float | None = None, sigma_z: float | None = None):
        super().__init__()
        B_init = torch.randn(m, 2)
        if sigma_x is not None and sigma_z is not None:
            B_init[:, 0] *= sigma_x
            B_init[:, 1] *= sigma_z
        else:
            B_init *= sigma
        if learnable:
            self.B = nn.Parameter(B_init)
        else:
            self.register_buffer("B", B_init)
        self.out_dim = 2 * m

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (xy @ self.B.T)  # [B, N, m]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward (Shazeer 2020, LLaMA). Replaces Linear→GELU→Linear.

    ``SwiGLU(x) = (silu(W₁x) ⊙ W₂x) W₃``. With ``gate_hidden = 2/3 × (d·mlp_ratio)``
    the parameter count matches a standard 2-layer GELU MLP of the same
    mlp_ratio (three d×gate_hidden projections instead of two d×hidden).
    """

    def __init__(self, d_model: int, mlp_ratio: int = 2):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        gate_hidden = int(hidden * 2 / 3)
        self.w1 = nn.Linear(d_model, gate_hidden)
        self.w2 = nn.Linear(d_model, gate_hidden)
        self.w3 = nn.Linear(gate_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def apply_fourier_pe(x_norm: torch.Tensor, enc: "FourierEncoder | None") -> torch.Tensor:
    """Append Fourier PE of the leading (x, z) coords to the input features.

    Returns x_norm unchanged when enc is None, else concat(x_norm, pe) so the
    spatial signal survives alongside the original (mean-0, std-1) coordinates.
    """
    if enc is None:
        return x_norm
    pe = enc(x_norm[..., :2])
    return torch.cat([x_norm, pe], dim=-1)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

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
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

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
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 swiglu=False):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if swiglu:
            self.mlp = SwiGLUMLP(hidden_dim, mlp_ratio=mlp_ratio)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        # Always returns the post-residual hidden state (the parent Transolver
        # applies ln_3 + mlp2 of the last block externally so a residual
        # SurfaceDecoder can hook the same pre-projection features).
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class SurfaceDecoder(nn.Module):
    """Zero-init residual cross-attention decoder (ControlNet-style).

    Produces a small [B, N, out_dim] delta on top of the baseline mlp2
    prediction. The final projection head is zero-initialised, so at step 0
    the delta is exactly 0 and the model matches the baseline output. The
    delta learns via gradients routed through the head, which are nonzero
    even when ``head.weight == 0`` (dL/dW = dL/dy ⊗ pre-head features).

    Architecture (DETR/Flamingo evolving-Q pattern):
        h = fx
        for layer in blocks:
            h = h + xattn(LN_q(h), LN_kv(fx), LN_kv(fx))   # K/V stay fixed at encoder output
        h = h + ffn(LN_ffn(h))
        delta = head(LN_out(h))                            # head zero-init
    """

    def __init__(self, dim: int, heads: int = 4, n_layers: int = 2,
                 out_dim: int = 3, mlp_ratio: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.lns_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.lns_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = SwiGLUMLP(dim, mlp_ratio=mlp_ratio)
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)
        # Zero both weight AND bias so surf_delta == 0 exactly at step 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, fx: torch.Tensor, is_surface: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # fx: [B, N, D] encoder hidden states.
        # mask: [B, N] bool, True = valid (not padded).
        # is_surface: [B, N] bool, True = surface node.
        #
        # On TandemFoilSet N≈86k but N_surf≈1.6k (~1.8% of N). Running
        # cross-attention with Q on all N nodes is O(N²·D) which makes the
        # decoder ~50× slower than the trunk — verified empirically. We
        # instead gather surface queries only (O(N_surf·N·D)), and scatter
        # the delta back to a full [B, N, out_dim] tensor with zeros at
        # non-surface nodes. The parent `torch.where(is_surface, ...)` keeps
        # the exact same semantics as a full-N forward.
        B, N, D = fx.shape
        device = fx.device
        out_dim = self.head.out_features

        surf_counts = is_surface.sum(dim=1)  # [B]
        max_surf = int(surf_counts.max().item())
        if max_surf == 0:
            return torch.zeros(B, N, out_dim, device=device, dtype=fx.dtype)

        # Per-sample: indices of surface nodes padded with 0 up to max_surf.
        # query_valid_mask marks real (non-padded) slots so we can zero-out
        # padded outputs before scattering — preventing padded slots (whose
        # gather_idx defaults to 0, a real node position) from overwriting
        # a valid surface delta at position 0.
        gather_idx = torch.zeros(B, max_surf, dtype=torch.long, device=device)
        query_valid_mask = torch.zeros(B, max_surf, dtype=torch.bool, device=device)
        for b in range(B):
            idx = is_surface[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n > 0:
                gather_idx[b, :n] = idx
                query_valid_mask[b, :n] = True

        # Gather surface queries [B, max_surf, D].
        fx_surf = torch.gather(
            fx, dim=1,
            index=gather_idx.unsqueeze(-1).expand(-1, -1, D),
        )

        # Cross-attention: Q = surface nodes, K/V = all valid nodes.
        key_padding_mask = ~mask  # [B, N], True = ignore
        h = fx_surf
        for i in range(self.n_layers):
            q = self.lns_q[i](h)
            kv = self.lns_kv[i](fx)
            attn_out, _ = self.blocks[i](
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            h = h + attn_out
        h = h + self.ffn(self.ln_ffn(h))
        delta_surf = self.head(self.ln_out(h))  # [B, max_surf, out_dim]

        # Zero-out padded slots before scatter so they don't overwrite real
        # values at index 0.
        delta_surf = delta_surf * query_valid_mask.unsqueeze(-1).to(delta_surf.dtype)

        delta_full = torch.zeros(B, N, out_dim, device=device, dtype=delta_surf.dtype)
        delta_full.scatter_(
            dim=1,
            index=gather_idx.unsqueeze(-1).expand(-1, -1, out_dim),
            src=delta_surf,
        )
        return delta_full


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None,
                 swiglu: bool = False,
                 use_surface_decoder: bool = False,
                 decoder_n_layers: int = 2,
                 decoder_n_heads: int = 4):
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
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                swiglu=swiglu,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

        # Instantiate the surface decoder AFTER the trunk is fully initialised
        # so its RNG consumption doesn't shift the trunk's init, keeping the
        # no-decoder and with-decoder runs bit-identical up to the trunk.
        self.use_surface_decoder = use_surface_decoder
        if use_surface_decoder:
            self.surface_decoder = SurfaceDecoder(
                dim=n_hidden,
                heads=decoder_n_heads,
                n_layers=decoder_n_layers,
                out_dim=out_dim,
                mlp_ratio=mlp_ratio,
            )
            # Overwrite default init with the trunk's trunc_normal convention,
            # then re-zero the head (apply() would have undone the zero-init).
            self.surface_decoder.apply(self._init_weights)
            nn.init.zeros_(self.surface_decoder.head.weight)
            nn.init.zeros_(self.surface_decoder.head.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, is_surface=None, mask=None, **kwargs):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)  # now returns hidden state even for the last block

        last = self.blocks[-1]
        vol_preds = last.mlp2(last.ln_3(fx))

        if self.use_surface_decoder:
            surf_delta = self.surface_decoder(fx, is_surface, mask)
            preds = torch.where(is_surface.unsqueeze(-1),
                                vol_preds + surf_delta,
                                vol_preds)
            return {"preds": preds, "surf_delta": surf_delta}

        return {"preds": vol_preds}


# ---------------------------------------------------------------------------
# Loss helper (applied in normalized space)
# ---------------------------------------------------------------------------

LOSS_TYPES = ("mse", "l1", "huber")


def elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str,
                     huber_delta: float) -> torch.Tensor:
    """Per-element loss in normalized space. Caller applies mask + reduction."""
    if loss_type == "mse":
        return (pred - target) ** 2
    if loss_type == "l1":
        return torch.abs(pred - target)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=huber_delta, reduction="none")
    raise ValueError(f"unknown loss_type={loss_type!r}, expected one of {LOSS_TYPES}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device,
                   loss_type: str, huber_delta: float,
                   fourier_enc: "FourierEncoder | None" = None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).
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

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            x_aug = apply_fourier_pe(x_norm, fourier_enc)
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_aug}, is_surface=is_surface, mask=mask)["preds"]

            per_elem = elementwise_loss(pred, y_norm, loss_type, huber_delta)
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (per_elem * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (per_elem * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def _sanitize_artifact_token(s: str) -> str:
    """wandb artifact names allow alnum, '-', '_', '.' — replace everything else."""
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in s)
    return out.strip("-_.") or "run"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_model_artifact(
    run,
    model_path: Path,
    model_dir: Path,
    cfg: "Config",
    best_metrics: dict,
    best_avg_surf_p: float,
    test_metrics: dict | None,
    test_avg: dict | None,
    n_params: int,
    model_config: dict,
) -> None:
    """Log the best checkpoint as a wandb model artifact.

    Name: ``model-<agent-or-wandb_name>-<run.id>`` (run.id guarantees uniqueness).
    Aliases: ``best`` + ``epoch-N`` so the best checkpoint is addressable
    both by role and by the epoch it came from.
    Payload: ``checkpoint.pt`` + ``config.yaml`` (if present).
    Metadata: run context, selected val metric, optional test metrics, git
    commit, model config, and hyperparams — enough to trace and reload.
    """
    if cfg.wandb_name:
        base = _sanitize_artifact_token(cfg.wandb_name)
    elif cfg.agent:
        base = _sanitize_artifact_token(cfg.agent)
    else:
        base = "tandemfoil"
    artifact_name = f"model-{base}-{run.id}"

    metadata: dict = {
        "run_id": run.id,
        "run_name": run.name,
        "agent": cfg.agent,
        "wandb_name": cfg.wandb_name,
        "wandb_group": cfg.wandb_group,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "model_config": model_config,
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "epochs_configured": cfg.epochs,
    }

    description = (
        f"Transolver checkpoint — best val_avg/mae_surf_p = {best_avg_surf_p:.4f} "
        f"at epoch {best_metrics['epoch']}"
    )

    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        metadata["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                metadata[f"test/{split_name}/mae_surf_p"] = m["mae_surf_p"]
        description += f" | test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}"

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description,
        metadata=metadata,
    )
    artifact.add_file(str(model_path), name="checkpoint.pt")
    config_yaml = model_dir / "config.yaml"
    if config_yaml.exists():
        artifact.add_file(str(config_yaml), name="config.yaml")

    aliases = ["best", f"epoch-{best_metrics['epoch']}"]
    run.log_artifact(artifact, aliases=aliases)
    print(f"\nLogged model artifact '{artifact_name}' (aliases: {', '.join(aliases)})")


def print_split_metrics(split_name: str, m: dict[str, float]) -> None:
    print(
        f"    {split_name:<26s} "
        f"loss={m['loss']:.4f}  "
        f"surf[p={m['mae_surf_p']:.4f} Ux={m['mae_surf_Ux']:.4f} Uy={m['mae_surf_Uy']:.4f}]  "
        f"vol[p={m['mae_vol_p']:.4f} Ux={m['mae_vol_Ux']:.4f} Uy={m['mae_vol_Uy']:.4f}]"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_MIN = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 1.0  # matches current baseline (PR #11 sw=1); pre-PR#11 default was 10.0
    epochs: int = 50
    loss_type: str = "mse"  # mse | l1 | huber — applied in normalized space
    huber_delta: float = 1.0  # Huber transition point (normalized units)
    amp: bool = False  # bf16 autocast on training forward/backward (val/test stay fp32)
    grad_accum: int = 1  # accumulate gradients over N micro-batches before optimizer step
    # Fourier positional encoding of (x, z) coords (Tancik 2020).
    fourier_features: str = "none"  # "none" | "fixed" | "learnable"
    fourier_m: int = 10             # number of frequency bands (output dim = 2m)
    fourier_sigma: float = 1.0      # isotropic bandwidth (ignored if per-coord σ set)
    fourier_sigma_x: float | None = None  # per-coord σ for x; both x & z must be set
    fourier_sigma_z: float | None = None  # per-coord σ for z; both x & z must be set
    swiglu: bool = False            # replace GELU-MLP with SwiGLU in each TransolverBlock
    # Zero-init residual cross-attention surface decoder (ControlNet-style).
    use_surface_decoder: bool = False  # add a residual delta head for surface nodes
    decoder_n_layers: int = 2          # cross-attention layers in the decoder
    decoder_n_heads: int = 4           # attention heads in the decoder
    seed: int = 0                   # RNG seed for torch / numpy / python random
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation


cfg = sp.parse(Config)
if cfg.loss_type not in LOSS_TYPES:
    raise ValueError(f"--loss_type must be one of {LOSS_TYPES}, got {cfg.loss_type!r}")
if cfg.grad_accum < 1:
    raise ValueError(f"--grad_accum must be >=1, got {cfg.grad_accum}")
if cfg.fourier_features not in ("none", "fixed", "learnable"):
    raise ValueError(
        f"--fourier_features must be one of none|fixed|learnable, got {cfg.fourier_features!r}"
    )
if (cfg.fourier_sigma_x is None) != (cfg.fourier_sigma_z is None):
    raise ValueError(
        "--fourier_sigma_x and --fourier_sigma_z must be set together "
        f"(got x={cfg.fourier_sigma_x!r}, z={cfg.fourier_sigma_z!r})"
    )
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

# Seed before any model / Fourier-B init so runs with the same seed are bit-exact
# on init (sampler + CUDA kernels are still stochastic).
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))
print(f"Loss: {cfg.loss_type}"
      + (f" (delta={cfg.huber_delta})" if cfg.loss_type == "huber" else "")
      + f"  surf_weight={cfg.surf_weight}")
print(f"AMP (bf16): {cfg.amp}  |  grad_accum: {cfg.grad_accum}  "
      f"|  effective batch: {cfg.batch_size * cfg.grad_accum}")
per_coord = cfg.fourier_sigma_x is not None and cfg.fourier_sigma_z is not None
if cfg.fourier_features == "none":
    fourier_str = "none"
elif per_coord:
    fourier_str = (f"{cfg.fourier_features} (m={cfg.fourier_m}, "
                   f"σ_x={cfg.fourier_sigma_x}, σ_z={cfg.fourier_sigma_z})")
else:
    fourier_str = f"{cfg.fourier_features} (m={cfg.fourier_m}, σ={cfg.fourier_sigma})"
print(f"Fourier: {fourier_str}  swiglu={cfg.swiglu}  seed={cfg.seed}")

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

fourier_enc: FourierEncoder | None = None
if cfg.fourier_features != "none":
    fourier_enc = FourierEncoder(
        m=cfg.fourier_m,
        sigma=cfg.fourier_sigma,
        learnable=(cfg.fourier_features == "learnable"),
        sigma_x=cfg.fourier_sigma_x,
        sigma_z=cfg.fourier_sigma_z,
    ).to(device)

# Fourier PE *appends* 2m features alongside the raw (x, z) coords, so
# space_dim = 2 (raw) + (2*m if fourier enabled else 0).
space_dim = 2 + (fourier_enc.out_dim if fourier_enc is not None else 0)

model_config = dict(
    space_dim=space_dim,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
    swiglu=cfg.swiglu,
    use_surface_decoder=cfg.use_surface_decoder,
    decoder_n_layers=cfg.decoder_n_layers,
    decoder_n_heads=cfg.decoder_n_heads,
)

model = Transolver(**model_config).to(device)

# Include learnable Fourier B in the parameter list so AdamW + cosine schedule
# see it. Fixed Fourier B is a buffer and has no gradient, so nothing to add.
trainable_params = list(model.parameters())
if fourier_enc is not None and cfg.fourier_features == "learnable":
    trainable_params += list(fourier_enc.parameters())
n_params = sum(p.numel() for p in trainable_params)
print(f"Model: Transolver ({n_params/1e6:.2f}M params, "
      f"space_dim={space_dim}, fourier={cfg.fourier_features})")

optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
# Cosine schedule spans the full set of *optimizer* steps (grad-accum aware)
# so LR reaches its floor exactly at the configured epoch budget regardless of
# accumulation depth. Stepped once per optimizer step, not per epoch.
total_optimizer_steps = max(1, (len(train_loader) * MAX_EPOCHS) // cfg.grad_accum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps)

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY"),
    project=os.environ.get("WANDB_PROJECT"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={
        **asdict(cfg),
        "model_config": model_config,
        "n_params": n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
        "total_optimizer_steps": total_optimizer_steps,
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")
wandb.define_metric("lr_step", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_avg_surf_p = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_micro = n_opt = 0
    # Accumulators for one optimizer step (average across micro-batches at log time).
    accum_vol = accum_surf = accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for micro_idx, (x, y, is_surface, mask) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        x_aug = apply_fourier_pe(x_norm, fourier_enc)
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        # bf16 autocast wraps forward + loss; backward inherits the cast through
        # the autograd graph. No GradScaler needed for bf16 (full fp32 exponent
        # range, so no underflow — GradScaler is only required for fp16).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            out = model({"x": x_aug}, is_surface=is_surface, mask=mask)
            pred = out["preds"]
            per_elem = elementwise_loss(pred, y_norm, cfg.loss_type, cfg.huber_delta)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (per_elem * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (per_elem * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        # Zero-init sanity check: on the very first forward pass, the decoder's
        # head is zeroed so surf_delta must be *exactly* 0. If it isn't, init
        # is broken and there's no point running the rest of the sweep.
        if epoch == 0 and micro_idx == 0 and cfg.use_surface_decoder:
            delta_max = out["surf_delta"].detach().float().abs().max().item()
            print(f"[zero-init check] surf_delta.abs().max() at step 0 = {delta_max:.3e}")
            wandb.summary["surf_delta_step0_abs_max"] = delta_max
            if delta_max > 0:
                raise RuntimeError(
                    f"Zero-init residual decoder failed: surf_delta.abs().max() = {delta_max} "
                    "at step 0. Check SurfaceDecoder.head init."
                )

        # Scale by 1/grad_accum so accumulated gradient == mean-of-minibatches
        # gradient (what you'd get at effective batch = batch_size * grad_accum).
        (loss / cfg.grad_accum).backward()

        accum_vol += vol_loss.item()
        accum_surf += surf_loss.item()
        accum_loss += loss.item()
        n_micro += 1

        is_last_micro = (micro_idx + 1) == len(train_loader)
        if ((micro_idx + 1) % cfg.grad_accum == 0) or is_last_micro:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            n_opt += 1

            denom = cfg.grad_accum if not is_last_micro else ((micro_idx % cfg.grad_accum) + 1)
            wandb.log({
                "train/loss": accum_loss / denom,
                "train/vol_loss_step": accum_vol / denom,
                "train/surf_loss_step": accum_surf / denom,
                "lr_step": scheduler.get_last_lr()[0],
                "global_step": global_step,
            })
            epoch_vol += accum_vol
            epoch_surf += accum_surf
            accum_vol = accum_surf = accum_loss = 0.0

    epoch_vol /= max(n_micro, 1)
    epoch_surf /= max(n_micro, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                             cfg.loss_type, cfg.huber_delta, fourier_enc=fourier_enc)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.
    wandb.log(log_metrics)

    tag = ""
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": avg_surf_p,
            "per_split": split_metrics,
        }
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
    })

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_metrics = None
    test_avg = None
    if not cfg.skip_test:
        print("\nEvaluating on held-out test splits...")
        test_datasets = load_test_data(cfg.splits_dir, debug=cfg.debug)
        test_loaders = {
            name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                                 cfg.loss_type, cfg.huber_delta, fourier_enc=fourier_enc)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])

        test_log: dict[str, float] = {}
        for split_name, m in test_metrics.items():
            for k, v in m.items():
                test_log[f"test/{split_name}/{k}"] = v
        for k, v in test_avg.items():
            test_log[f"test_{k}"] = v
        wandb.log(test_log)
        wandb.summary.update(test_log)

    save_model_artifact(
        run=run,
        model_path=model_path,
        model_dir=model_dir,
        cfg=cfg,
        best_metrics=best_metrics,
        best_avg_surf_p=best_avg_surf_p,
        test_metrics=test_metrics,
        test_avg=test_avg,
        n_params=n_params,
        model_config=model_config,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping artifact upload.")

wandb.finish()
