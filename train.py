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

import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

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
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
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
            features = self.ln_3(fx)
            return self.mlp2(features), features
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
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
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
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
        for block in self.blocks[:-1]:
            fx = block(fx)
        pred, features = self.blocks[-1](fx)
        return {"preds": pred, "features": features}


class SurfaceAuxHead(nn.Module):
    """Surface-only pressure head over backbone features (post-ln3)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(max(n_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


# ---------------------------------------------------------------------------
# EMA of weights
# ---------------------------------------------------------------------------


class EMA:
    """Exponential moving average of model parameters, evaluated at val/test time."""

    def __init__(self, model: nn.Module, decay: float, warmup_steps: int):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.dtype.is_floating_point
        }

    def update(self, model: nn.Module) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].copy_(p.detach())
            return
        d = self.decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Swap live weights with EMA weights for the duration of the block."""
        backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if n in self.shadow
        }
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        try:
            yield
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, aux_head=None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped).

    When ``aux_head`` is provided, the *primary* surface-pressure prediction is
    swapped in from the aux head while volume p / Ux / Uy stay on the backbone.
    Two extra diagnostics are returned:
      - ``mae_surf_p_aux``     — surface pressure MAE using aux head only
      - ``mae_surf_p_backbone``— surface pressure MAE using backbone only
    so the mechanism check (aux ≪ backbone?) is visible in W&B.
    """
    vol_loss_sum = surf_loss_sum = aux_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    aux_p_abs_sum = torch.zeros((), dtype=torch.float64, device=device)
    backbone_p_abs_sum = torch.zeros((), dtype=torch.float64, device=device)
    surf_p_count = torch.zeros((), dtype=torch.float64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Drop samples whose ground truth has any non-finite values, then
            # replace those values with 0 before any arithmetic. ``data.scoring``
            # documents per-sample skipping, but its ``err * mask`` step would
            # otherwise produce ``0 * inf = NaN`` and poison the accumulators.
            y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
            mask = mask & y_finite.unsqueeze(-1)
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            output = model({"x": x_norm})
            pred = output["preds"]
            features = output["features"]

            sq_err = (pred - y_norm) ** 2
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss_sum += (
                (sq_err * vol_mask.unsqueeze(-1)).sum()
                / vol_mask.sum().clamp(min=1)
            ).item()
            surf_loss_sum += (
                (sq_err * surf_mask.unsqueeze(-1)).sum()
                / surf_mask.sum().clamp(min=1)
            ).item()

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            backbone_p_orig = pred_orig[..., 2]

            if aux_head is not None:
                aux_pred_norm = aux_head(features)
                aux_diff_sq = (aux_pred_norm - y_norm[..., 2]) ** 2
                aux_loss_sum += (
                    (aux_diff_sq * surf_mask).sum() / surf_mask.sum().clamp(min=1)
                ).item()

                aux_pred_p_orig = aux_pred_norm * stats["y_std"][2] + stats["y_mean"][2]
                pred_for_score = pred_orig.clone()
                pred_for_score[..., 2] = torch.where(
                    is_surface, aux_pred_p_orig, pred_orig[..., 2]
                )

                # Diagnostics: aux-only vs backbone-only surface-p MAE.
                surf_pf = surf_mask.to(torch.float64)
                aux_p_abs_sum += ((aux_pred_p_orig - y[..., 2]).abs().to(torch.float64) * surf_pf).sum()
                backbone_p_abs_sum += ((backbone_p_orig - y[..., 2]).abs().to(torch.float64) * surf_pf).sum()
                surf_p_count += surf_pf.sum()
            else:
                pred_for_score = pred_orig

            n_batches += 1

            ds, dv = accumulate_batch(pred_for_score, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    if aux_head is not None:
        out["aux_loss"] = aux_loss_sum / max(n_batches, 1)
        if surf_p_count.item() > 0:
            out["mae_surf_p_aux"] = (aux_p_abs_sum / surf_p_count).item()
            out["mae_surf_p_backbone"] = (backbone_p_abs_sum / surf_p_count).item()
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
        "peak_lr": cfg.peak_lr,
        "warmup_epochs": cfg.warmup_epochs,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "surf_weight": cfg.surf_weight,
        "epochs_configured": cfg.epochs,
        "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay,
        "ema_warmup_steps": cfg.ema_warmup_steps,
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
    peak_lr: float = 1e-3
    warmup_epochs: int = 2
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    n_layers: int = 5
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_warmup_steps: int = 100
    aux_surf_head: bool = True
    aux_surf_head_hidden: int = 128
    aux_surf_head_layers: int = 2
    aux_surf_loss_weight: float = 1.0
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

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
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=cfg.n_layers,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

aux_head: SurfaceAuxHead | None = None
aux_n_params = 0
if cfg.aux_surf_head:
    aux_head = SurfaceAuxHead(
        in_dim=model_config["n_hidden"],
        hidden_dim=cfg.aux_surf_head_hidden,
        n_layers=cfg.aux_surf_head_layers,
    ).to(device)
    aux_n_params = sum(p.numel() for p in aux_head.parameters())
    print(
        f"SurfaceAuxHead enabled: hidden={cfg.aux_surf_head_hidden}, "
        f"layers={cfg.aux_surf_head_layers}, weight={cfg.aux_surf_loss_weight}, "
        f"params={aux_n_params/1e3:.1f}K"
    )

ema = EMA(model, decay=cfg.ema_decay, warmup_steps=cfg.ema_warmup_steps) if cfg.use_ema else None
aux_ema = (
    EMA(aux_head, decay=cfg.ema_decay, warmup_steps=cfg.ema_warmup_steps)
    if cfg.use_ema and aux_head is not None
    else None
)
if ema is not None:
    print(f"EMA enabled: decay={cfg.ema_decay}, warmup_steps={cfg.ema_warmup_steps}")

opt_params = list(model.parameters())
if aux_head is not None:
    opt_params += list(aux_head.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)

warmup_iters = max(cfg.warmup_epochs, 1)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters,
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1),
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs],
)

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
        "aux_head_n_params": aux_n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")

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
    if aux_head is not None:
        aux_head.train()
    epoch_vol = epoch_surf = epoch_aux = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        output = model({"x": x_norm})
        pred = output["preds"]
        features = output["features"]
        sq_err = (pred - y_norm) ** 2

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

        aux_loss = torch.zeros((), device=device)
        if aux_head is not None and surf_mask.any():
            aux_pred_norm = aux_head(features)
            aux_diff_sq = (aux_pred_norm - y_norm[..., 2]) ** 2
            aux_loss = (aux_diff_sq * surf_mask).sum() / surf_mask.sum().clamp(min=1)

        loss = vol_loss + cfg.surf_weight * surf_loss + cfg.aux_surf_loss_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if aux_ema is not None:
            aux_ema.update(aux_head)
        global_step += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/vol_loss_step": vol_loss.item(),
            "train/surf_loss_step": surf_loss.item(),
            "train/aux_loss_step": aux_loss.item() if aux_head is not None else 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        epoch_aux += aux_loss.item() if aux_head is not None else 0.0
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    epoch_aux /= max(n_batches, 1)

    # --- Validate (using EMA weights when enabled) ---
    def _run_val_splits() -> dict:
        return {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, aux_head)
            for name, loader in val_loaders.items()
        }

    model.eval()
    if aux_head is not None:
        aux_head.eval()
    if ema is not None:
        with ema.apply_to(model):
            if aux_ema is not None:
                with aux_ema.apply_to(aux_head):
                    split_metrics = _run_val_splits()
            else:
                split_metrics = _run_val_splits()
    else:
        split_metrics = _run_val_splits()
    val_avg = aggregate_splits(split_metrics)
    avg_surf_p = val_avg["avg/mae_surf_p"]
    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/aux_loss": epoch_aux,
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

        def _save_ckpt() -> None:
            ckpt = {"model": model.state_dict()}
            if aux_head is not None:
                ckpt["aux_head"] = aux_head.state_dict()
            torch.save(ckpt, model_path)

        if ema is not None:
            with ema.apply_to(model):
                if aux_ema is not None:
                    with aux_ema.apply_to(aux_head):
                        _save_ckpt()
                else:
                    _save_ckpt()
        else:
            _save_ckpt()
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

# --- Diagnostic: confirm EMA path is alive (EMA vs live on final-epoch val) ---
if ema is not None:
    print("\nDiagnostic: comparing EMA vs live weights on val splits...")
    model.eval()
    if aux_head is not None:
        aux_head.eval()

    def _final_val() -> dict:
        return {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, aux_head)
            for name, loader in val_loaders.items()
        }

    live_split = _final_val()
    live_avg = aggregate_splits(live_split)
    with ema.apply_to(model):
        if aux_ema is not None:
            with aux_ema.apply_to(aux_head):
                ema_split = _final_val()
        else:
            ema_split = _final_val()
    ema_avg = aggregate_splits(ema_split)
    diag = {
        "diag/val_avg_live_final": live_avg["avg/mae_surf_p"],
        "diag/val_avg_ema_final": ema_avg["avg/mae_surf_p"],
        "diag/val_avg_ema_minus_live": ema_avg["avg/mae_surf_p"] - live_avg["avg/mae_surf_p"],
    }
    wandb.log(diag)
    wandb.summary.update(diag)
    print(
        f"  live={live_avg['avg/mae_surf_p']:.4f}  "
        f"ema={ema_avg['avg/mae_surf_p']:.4f}  "
        f"delta={diag['diag/val_avg_ema_minus_live']:+.4f}"
    )

# --- Test evaluation + artifact upload ---
if best_metrics:
    print(f"\nBest val: epoch {best_metrics['epoch']}, val_avg/mae_surf_p = {best_avg_surf_p:.4f}")
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "total_train_minutes": total_time,
    })

    sd = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        model.load_state_dict(sd["model"])
        if aux_head is not None and "aux_head" in sd:
            aux_head.load_state_dict(sd["aux_head"])
    else:
        model.load_state_dict(sd)
    model.eval()
    if aux_head is not None:
        aux_head.eval()

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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, aux_head)
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
