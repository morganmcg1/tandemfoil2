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
  python train.py [--debug] [--epochs 50] [--agent <name>] [--experiment_name <name>]
"""

from __future__ import annotations

import json
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


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FourierFeatures(nn.Module):
    """Random-Fourier-feature positional encoding for 2D coords."""

    def __init__(self, m: int = 160, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(2, m) * sigma)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:  # xy: [B, N, 2]
        proj = 2 * math.pi * (xy @ self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        g = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, g)
        self.w2 = nn.Linear(dim, g)
        self.w3 = nn.Linear(g, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


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
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        if act == "swiglu":
            self.mlp = SwiGLU(hidden_dim, hidden_dim * mlp_ratio)
        else:
            self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                           n_layers=0, res=False, act=act)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.drop_path1(self.attn(self.ln_1(fx))) + fx
        fx = self.drop_path2(self.mlp(self.ln_2(fx))) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 drop_path_rate: float = 0.0,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        # Pre-block has no SwiGLU; keep gelu for MLP preprocess
        pre_act = "gelu" if act == "swiglu" else act
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=pre_act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=pre_act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                drop_path_rate=drop_path_rate,
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_loss(pred, y_norm, vol_mask, surf_mask, loss_type, surf_weight):
    if loss_type == "l1":
        err = torch.abs(pred - y_norm)
    else:
        err = (pred - y_norm) ** 2
    vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
    surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
    return vol_loss, surf_loss, vol_loss + surf_weight * surf_loss


def _apply_features(x_norm, fourier):
    if fourier is None:
        return x_norm
    pe = fourier(x_norm[:, :, :2])
    return torch.cat([pe, x_norm[:, :, 2:]], dim=-1)


def evaluate_split(model, loader, stats, surf_weight, device, fourier=None,
                   loss_type="mse", split_name: str | None = None) -> dict[str, float]:
    """Evaluate a split and return metrics matching the organizer scorer.

    ``loss`` is the (training) loss in normalized space; MAE channels are in
    original target space, accumulated per organizer ``score.py``.

    Defensive: samples whose ground truth or prediction contains non-finite
    values are excluded entirely. ``accumulate_batch`` filters such samples
    via ``sample_mask``, but the masked multiplication still produces NaN
    (``Inf * 0 == NaN``), so we additionally zero non-finite y/pred values
    and zero the bad samples' masks before calling it.
    Known case: ``test_geom_camber_cruise/000020.pt`` has Inf in y[..., p].
    """
    vol_loss_sum = surf_loss_sum = 0.0
    n_loss_batches = 0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf = n_vol = n_dropped = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            x_in = _apply_features(x_norm, fourier)
            pred = model({"x": x_in})["preds"]

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss, surf_loss, _ = _compute_loss(
                pred, y_norm, vol_mask, surf_mask, loss_type, surf_weight
            )
            vl, sl = vol_loss.item(), surf_loss.item()
            if math.isfinite(vl) and math.isfinite(sl):
                vol_loss_sum += vl
                surf_loss_sum += sl
                n_loss_batches += 1

            pred_orig = pred.float() * stats["y_std"] + stats["y_mean"]
            B = pred_orig.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            pred_finite = torch.isfinite(pred_orig.reshape(B, -1)).all(dim=-1)
            both_finite = y_finite & pred_finite

            if not both_finite.all():
                bad = ~both_finite
                n_dropped += int(bad.sum().item())
                # Zero non-finite values *and* the bad samples' masks. Avoids
                # Inf*0=NaN inside accumulate_batch's masked sum.
                pred_orig = torch.nan_to_num(
                    pred_orig, nan=0.0, posinf=0.0, neginf=0.0
                )
                y_clean = torch.nan_to_num(
                    y, nan=0.0, posinf=0.0, neginf=0.0
                )
                mask_clean = mask.clone()
                mask_clean[bad] = False
                is_surface_clean = is_surface.clone()
                is_surface_clean[bad] = False
            else:
                y_clean = y
                mask_clean = mask
                is_surface_clean = is_surface

            ds, dv = accumulate_batch(
                pred_orig, y_clean, is_surface_clean, mask_clean, mae_surf, mae_vol
            )
            n_surf += ds
            n_vol += dv

    if n_dropped > 0:
        tag = f" [{split_name}]" if split_name else ""
        print(f"  WARNING{tag}: dropped {n_dropped} sample(s) with non-finite y or pred")

    vol_loss = vol_loss_sum / max(n_loss_batches, 1)
    surf_loss = surf_loss_sum / max(n_loss_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss,
           "n_dropped": n_dropped}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    return out


def _sanitize_path_token(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "-" for c in s)
    return out.strip("-_.") or "experiment"


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def append_metrics_jsonl(metrics_path: Path, record: dict) -> None:
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_experiment_summary(
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
    """Write a local summary next to the best checkpoint."""
    summary: dict = {
        "agent": cfg.agent,
        "experiment_name": cfg.experiment_name,
        "git_commit": _git_commit_short(),
        "n_params": n_params,
        "model_config": model_config,
        "checkpoint": str(model_path),
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "epochs_configured": cfg.epochs,
        "loss_type": cfg.loss_type,
        "amp": cfg.amp,
        "grad_accum": cfg.grad_accum,
        "fourier_m": cfg.fourier_m,
        "fourier_sigma": cfg.fourier_sigma,
        "swiglu": cfg.swiglu,
        "n_layers": cfg.n_layers,
        "slice_num": cfg.slice_num,
        "n_head": cfg.n_head,
        "n_hidden": cfg.n_hidden,
        "mlp_ratio": cfg.mlp_ratio,
        "drop_path_rate": cfg.drop_path_rate,
        "dropout": cfg.dropout,
        "seed": cfg.seed,
    }

    for split_name, m in best_metrics["per_split"].items():
        for k, v in m.items():
            summary[f"best_val/{split_name}/{k}"] = v
    if test_avg is not None and "avg/mae_surf_p" in test_avg:
        summary["test_avg/mae_surf_p"] = test_avg["avg/mae_surf_p"]
        if test_metrics is not None:
            for split_name, m in test_metrics.items():
                for k, v in m.items():
                    summary[f"test/{split_name}/{k}"] = v

    summary_path = model_dir / "metrics.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"\nSaved experiment summary to {summary_path}")


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
ENV_MAX_EPOCHS = int(os.environ.get("SENPAI_MAX_EPOCHS", "999"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 1.0  # changed from 10.0 to match L1 recipe
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False
    # Winning recipe knobs
    loss_type: str = "l1"
    amp: bool = True
    grad_accum: int = 4
    fourier_m: int = 160
    fourier_sigma: float = 0.7
    swiglu: bool = True
    n_layers: int = 3
    n_head: int = 1
    slice_num: int = 16
    n_hidden: int = 128
    mlp_ratio: int = 2
    # Regularization knobs (this PR)
    drop_path_rate: float = 0.0
    dropout: float = 0.0
    # Reproducibility / logging
    seed: int = 0
    wandb_group: str | None = None
    wandb_name: str | None = None


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else min(cfg.epochs, ENV_MAX_EPOCHS)
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

# Seed everything for reproducibility across seed=0/1 sweeps
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(cfg.seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds),
                                    replacement=True, generator=sampler_gen)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

# Fourier PE (frozen random projection on first 2 dims = node x,z)
fourier = None
if cfg.fourier_m > 0:
    fourier = FourierFeatures(m=cfg.fourier_m, sigma=cfg.fourier_sigma).to(device)

if fourier is not None:
    fun_dim = 2 * cfg.fourier_m + (X_DIM - 2)
    space_dim = 0
else:
    fun_dim = X_DIM - 2
    space_dim = 2

model_config = dict(
    space_dim=space_dim,
    fun_dim=fun_dim,
    out_dim=3,
    n_hidden=cfg.n_hidden,
    n_layers=cfg.n_layers,
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=cfg.mlp_ratio,
    act="swiglu" if cfg.swiglu else "gelu",
    dropout=cfg.dropout,
    drop_path_rate=cfg.drop_path_rate,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")
print(
    f"Recipe: loss={cfg.loss_type} surf_w={cfg.surf_weight} amp={cfg.amp} "
    f"grad_accum={cfg.grad_accum} fourier_m={cfg.fourier_m} swiglu={cfg.swiglu} "
    f"nl={cfg.n_layers} sn={cfg.slice_num} nh={cfg.n_head} "
    f"drop_path={cfg.drop_path_rate} dropout={cfg.dropout} seed={cfg.seed}"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and torch.cuda.is_available())

experiment_label = cfg.experiment_name or cfg.agent or "tandemfoil"
experiment_stamp = time.strftime("%Y%m%d-%H%M%S")
model_dir = Path("models") / f"model-{_sanitize_path_token(experiment_label)}-{experiment_stamp}"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
metrics_jsonl_path = model_dir / "metrics.jsonl"
with open(model_dir / "config.yaml", "w") as f:
    yaml.safe_dump({
        **asdict(cfg),
        "model_config": model_config,
        "n_params": n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    }, f, sort_keys=True)

# W&B (best-effort: log scalar metrics if wandb is available; skip on failure).
wandb_run = None
try:
    import wandb  # type: ignore
    project = os.environ.get("WANDB_PROJECT", "senpai-charlie-pai2c-r5")
    wandb_run = wandb.init(
        project=project,
        group=cfg.wandb_group or (cfg.agent or "default"),
        name=cfg.wandb_name or experiment_label,
        config={**asdict(cfg), "model_config": model_config, "n_params": n_params},
        reinit=True,
    )
    print(f"W&B run: {wandb_run.name} (id={wandb_run.id})")
except Exception as e:  # pragma: no cover - W&B is best-effort
    print(f"W&B disabled ({type(e).__name__}: {e})")
    wandb_run = None


def _flat_split_metrics(prefix: str, split_metrics: dict) -> dict:
    flat = {}
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            flat[f"{prefix}/{split_name}/{k}"] = v
    return flat


best_avg_surf_p = float("inf")
best_metrics: dict = {}
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    optimizer.zero_grad()
    n_steps = len(train_loader)
    for step, (x, y, is_surface, mask) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        x_in = _apply_features(x_norm, fourier)

        with torch.amp.autocast("cuda", enabled=cfg.amp and torch.cuda.is_available(),
                                dtype=torch.bfloat16):
            pred = model({"x": x_in})["preds"]
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss, surf_loss, loss = _compute_loss(
                pred, y_norm, vol_mask, surf_mask, cfg.loss_type, cfg.surf_weight
            )

        scaler.scale(loss / cfg.grad_accum).backward()
        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

        if (step + 1) % cfg.grad_accum == 0 or (step + 1) == n_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device,
                             fourier=fourier, loss_type=cfg.loss_type,
                             split_name=name)
        for name, loader in val_loaders.items()
    }
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    dt = time.time() - t0

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
    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
    })
    if wandb_run is not None:
        wandb_run.log({
            "epoch": epoch + 1,
            "seconds": dt,
            "peak_memory_gb": peak_gb,
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "val_avg/mae_surf_p": avg_surf_p,
            "is_best_so_far": float(tag == " *"),
            "best_val_avg/mae_surf_p": best_avg_surf_p,
            **_flat_split_metrics("val", split_metrics),
        })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, split_metrics[name])

total_time = (time.time() - train_start) / 60.0
print(f"\nTraining done in {total_time:.1f} min")

# --- Test evaluation + local summary ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")

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
                                 fourier=fourier, loss_type=cfg.loss_type,
                                 split_name=name)
            for name, loader in test_loaders.items()
        }
        test_avg = aggregate_splits(test_metrics)
        print(f"\n  TEST  avg_surf_p={test_avg['avg/mae_surf_p']:.4f}")
        for name in TEST_SPLIT_NAMES:
            print_split_metrics(name, test_metrics[name])
        append_metrics_jsonl(metrics_jsonl_path, {
            "event": "test",
            "best_epoch": best_metrics["epoch"],
            "test_avg": test_avg,
            "test_splits": test_metrics,
        })
        if wandb_run is not None:
            wandb_run.log({
                "test_avg/mae_surf_p": test_avg["avg/mae_surf_p"],
                **_flat_split_metrics("test", test_metrics),
            })

    write_experiment_summary(
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
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping test evaluation.")

if wandb_run is not None:
    # SIGALRM-bounded finish: a slow sync should not block the sweep driver.
    import signal as _sig

    def _alarm(*_):
        raise TimeoutError("wandb finish timeout")

    try:
        _sig.signal(_sig.SIGALRM, _alarm)
        _sig.alarm(60)
        wandb_run.finish()
    except Exception as e:
        print(f"W&B finish skipped ({type(e).__name__}: {e})")
    finally:
        _sig.alarm(0)
