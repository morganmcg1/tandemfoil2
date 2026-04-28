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

import copy
import math
import os
import subprocess
import time
from contextlib import nullcontext
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


class FiLM(nn.Module):
    """Produce per-layer (gamma, beta) from a single conditioning scalar (log Re).

    Identity-init: gamma starts at 1, beta at 0, so the modulated path matches
    the unconditioned baseline at step 0 and can only deviate as the network
    learns useful Re-dependent modulations.
    """

    def __init__(self, n_layers: int, n_hidden: int, hidden: int = 64):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_layers * n_hidden),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            b = self.net[-1].bias
            b.zero_()
            b[: n_layers * n_hidden].fill_(1.0)

    def forward(self, log_re):
        if log_re.dim() == 1:
            log_re = log_re.unsqueeze(-1)
        out = self.net(log_re)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.n_layers, self.n_hidden)
        beta = beta.view(-1, self.n_layers, self.n_hidden)
        return gamma, beta


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

    def forward(self, fx, gamma=None, beta=None):
        y = self.ln_1(fx)
        if gamma is not None:
            y = gamma.unsqueeze(1) * y + beta.unsqueeze(1)
        fx = self.attn(y) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    # x dim 13 carries log(Re) (already normalized by data/loader stats); we tap
    # it before preprocess() so FiLM sees the same conditioning for every node.
    LOG_RE_DIM = 13

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 film_re: bool = False, film_hidden: int = 64,
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
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.film_re = film_re
        if film_re:
            self.film = FiLM(n_layers=n_layers, n_hidden=n_hidden, hidden=film_hidden)
        self.apply(self._init_weights)
        # Re-apply FiLM identity-init: _init_weights overwrites the
        # last-layer (zero W, gamma-bias=1, beta-bias=0) initialization.
        if film_re:
            with torch.no_grad():
                self.film.net[-1].weight.zero_()
                b = self.film.net[-1].bias
                b.zero_()
                b[: n_layers * n_hidden].fill_(1.0)

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
        if self.film_re:
            log_re = x[:, 0, self.LOG_RE_DIM]
            gamma, beta = self.film(log_re)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        if self.film_re:
            for i, block in enumerate(self.blocks):
                fx = block(fx, gamma=gamma[:, i], beta=beta[:, i])
        else:
            for block in self.blocks:
                fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, amp_ctx=None) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the normalized-space loss used for training monitoring; the MAE
    channels are in the original target space and accumulated per organizer
    ``score.py`` (float64, non-finite samples skipped). ``amp_ctx`` wraps the
    forward only — metric accumulation stays in fp64.
    """
    if amp_ctx is None:
        amp_ctx = nullcontext()
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

            # Skip samples whose ground truth contains any non-finite value
            # (one such case exists in test_geom_camber_cruise sample 20).
            # Without this, `(pred - y)**2 * mask` propagates NaN via inf*0
            # even though the scoring helper zeroes the mask for bad samples.
            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)  # [B]
            mask = mask & y_finite.view(B, 1)
            y_clean = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y_clean - stats["y_mean"]) / stats["y_std"]
            with amp_ctx:
                pred = model({"x": x_norm})["preds"]
            pred = pred.float()
            # bf16 forward can produce non-finite preds on extreme samples.
            # Fall back to an fp32 forward for those batches so test metrics
            # stay valid; checked only on masked-in positions because
            # padding can legitimately be uninitialised.
            if not torch.isfinite(pred[mask]).all():
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    pred = model({"x": x_norm})["preds"].float()
            # Defensive: zero out any remaining non-finite predictions so
            # a stray NaN/inf at any (sample, node, channel) position
            # cannot propagate through `pred * mask = inf * 0 = NaN`
            # into the metric. Masked-out positions are excluded anyway,
            # so this only matters when the model produces non-finite preds.
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

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
            n_batches += 1

            pred_orig = pred * stats["y_std"] + stats["y_mean"]
            ds, dv = accumulate_batch(pred_orig, y_clean, is_surface, mask, mae_surf, mae_vol)
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
    eval_source: str = "raw",
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
        "eval_source": eval_source,
        "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay,
        "ema_warmup_steps": cfg.ema_warmup_steps,
        "ema_eval_every": cfg.ema_eval_every,
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
    surf_weight: float = 10.0
    epochs: int = 50
    warmup_frac: float = 0.05  # fraction of total steps used for linear LR warmup
    grad_accum_steps: int = 1
    amp_dtype: str = "bf16"  # one of: "bf16", "fp32" (no autocast)
    compile: bool = True  # torch.compile the model (auto-disabled in debug)
    # ``reduce-overhead`` records a CUDAGraph per distinct N_max — with the variable
    # mesh sizes of TandemFoilSet that pool blows past 96 GB. ``default`` keeps the
    # inductor kernels but skips CUDAGraphs.
    compile_mode: str = "default"  # passed to torch.compile
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    film_re: bool = True  # FiLM modulation conditioned on log(Re), per-block
    film_hidden: int = 64  # hidden width of the FiLM MLP (1 → hidden → 2*L*H)
    seed: int | None = None  # if set, torch.manual_seed for variance checks
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_warmup_steps: int = 100  # don't update EMA for the first N steps
    ema_eval_every: int = 1      # run EMA validation every N epochs (1 = every epoch)
    re_jitter_sigma: float = 0.0  # std of Gaussian noise added to raw log(Re) per-sample during training only; 0 = disabled


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

if cfg.seed is not None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    print(f"Seed: {cfg.seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
amp_dtype = _AMP_DTYPES[cfg.amp_dtype]
amp_ctx = (
    torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    if (torch.cuda.is_available() and amp_dtype is not None)
    else nullcontext()
)
print(f"AMP: {cfg.amp_dtype}  compile: {cfg.compile and not cfg.debug}  "
      f"compile_mode: {cfg.compile_mode}  grad_accum_steps: {cfg.grad_accum_steps}")

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
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    film_re=cfg.film_re,
    film_hidden=cfg.film_hidden,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

if cfg.use_ema:
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    print(f"EMA: enabled (decay={cfg.ema_decay}, warmup_steps={cfg.ema_warmup_steps})")
else:
    ema_model = None
    print("EMA: disabled")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# Per-step linear warmup + cosine-to-zero schedule. T_max is aligned to the
# total step budget for the configured epoch count (so cosine actually reaches
# zero when the run completes within timeout). Warmup gives the orthogonal-init
# slice projection in PhysicsAttention a stable start at batch_size=4.
optimizer_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
total_steps = max(1, optimizer_steps_per_epoch * MAX_EPOCHS)
warmup_steps = max(1, int(total_steps * cfg.warmup_frac))


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"Schedule: linear warmup {warmup_steps} steps + cosine over {total_steps} total steps")

# torch.compile after optimizer is bound to the raw parameters. ``dynamic=True``
# avoids recompile thrash when each batch's padded N_max varies.
_compile_enabled = cfg.compile and not cfg.debug and torch.cuda.is_available()
if _compile_enabled:
    model = torch.compile(model, mode=cfg.compile_mode, dynamic=True)


def _raw_module(m):
    """Unwrap torch.compile's OptimizedModule so save/load uses prefix-less keys."""
    return m._orig_mod if hasattr(m, "_orig_mod") else m

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
        "schedule": "linear_warmup_cosine_to_zero",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "steps_per_epoch": len(train_loader),
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
wandb.define_metric("val_avg/*", step_metric="global_step")
wandb.define_metric("val_avg_ema/*", step_metric="global_step")
wandb.define_metric("val_avg_active/*", step_metric="global_step")
wandb.define_metric("val_active/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
    wandb.define_metric(f"{_name}_ema/*", step_metric="global_step")
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

total_epochs_completed = 0

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT_MIN:
        print(f"Timeout ({MAX_TIMEOUT_MIN} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Per-sample Gaussian jitter on raw log(Re) (training only). Applied
        # before normalization so FiLM and the rest of the model see a
        # consistent perturbation. Same eps broadcast across all nodes (real
        # and padded) within a sample — log(Re) is a per-sample conditioning
        # variable. evaluate_split never enters this loop, so val/test are
        # untouched.
        if cfg.re_jitter_sigma > 0.0:
            B = x.shape[0]
            eps = torch.randn(B, 1, 1, device=device, dtype=x.dtype) * cfg.re_jitter_sigma
            x = x.clone()
            x[:, :, Transolver.LOG_RE_DIM:Transolver.LOG_RE_DIM + 1] = (
                x[:, :, Transolver.LOG_RE_DIM:Transolver.LOG_RE_DIM + 1] + eps
            )

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with amp_ctx:
            pred = model({"x": x_norm})["preds"]
            sq_err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        # Scale only the backward pass for grad accumulation; logged ``loss`` stays unscaled.
        (loss / cfg.grad_accum_steps).backward()
        n_batches += 1
        if n_batches % cfg.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1

        if cfg.use_ema:
            with torch.no_grad():
                if global_step >= cfg.ema_warmup_steps:
                    d = cfg.ema_decay
                    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                        p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
                else:
                    # During warmup, keep EMA model = model
                    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
                        p_ema.data.copy_(p.data)
                # Update buffers (none in our model, but defensive)
                for b, b_ema in zip(model.buffers(), ema_model.buffers()):
                    b_ema.data.copy_(b.data)

        wandb.log({
            "train/loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()

    # Flush any remaining accumulated gradients at epoch boundary.
    if n_batches % cfg.grad_accum_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    total_epochs_completed = epoch + 1
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx=amp_ctx)
        for name, loader in val_loaders.items()
    }
    val_avg_raw = aggregate_splits(split_metrics)
    raw_avg_surf_p = val_avg_raw["avg/mae_surf_p"]

    do_ema_eval = cfg.use_ema and (
        epoch % cfg.ema_eval_every == 0 or epoch == MAX_EPOCHS - 1
    )
    if do_ema_eval:
        split_metrics_ema = {
            name: evaluate_split(ema_model, loader, stats, cfg.surf_weight, device, amp_ctx=amp_ctx)
            for name, loader in val_loaders.items()
        }
        val_avg_ema = aggregate_splits(split_metrics_ema)
        ema_avg_surf_p = val_avg_ema["avg/mae_surf_p"]
    else:
        split_metrics_ema = None
        val_avg_ema = None
        ema_avg_surf_p = None

    # Pick the better of the two for tracking the best checkpoint and for the test eval.
    # _raw_module() unwraps torch.compile's OptimizedModule so the saved state_dict has
    # prefix-less keys and reloads cleanly into either model or ema_model.
    if do_ema_eval and ema_avg_surf_p < raw_avg_surf_p:
        active_metrics = split_metrics_ema
        active_avg_surf_p = ema_avg_surf_p
        active_state = _raw_module(ema_model).state_dict()
        active_label = "ema"
    else:
        active_metrics = split_metrics
        active_avg_surf_p = raw_avg_surf_p
        active_state = _raw_module(model).state_dict()
        active_label = "raw"

    val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
    dt = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    log_metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": val_loss_mean,
        "epoch_time_s": dt,
        "epoch": epoch + 1,
        "peak_gpu_memory_gb": peak_gb,
        "global_step": global_step,
        "val_active/source_is_ema": int(active_label == "ema"),
        "val_active/ema_eval_done": int(do_ema_eval),
    }
    for split_name, m in split_metrics.items():
        for k, v in m.items():
            log_metrics[f"{split_name}/{k}"] = v
    for k, v in val_avg_raw.items():
        log_metrics[f"val_{k}"] = v  # val_avg/mae_surf_p etc.
    if do_ema_eval:
        for split_name, m in split_metrics_ema.items():
            for k, v in m.items():
                log_metrics[f"{split_name}_ema/{k}"] = v
        for k, v in val_avg_ema.items():
            # k is e.g. "avg/mae_surf_p" -> "val_avg_ema/mae_surf_p"
            log_metrics[f"val_{k.replace('avg/', 'avg_ema/')}"] = v
    log_metrics["val_avg_active/mae_surf_p"] = active_avg_surf_p
    wandb.log(log_metrics)

    tag = ""
    if active_avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = active_avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_avg/mae_surf_p": active_avg_surf_p,
            "val_avg_raw/mae_surf_p": raw_avg_surf_p,
            "val_avg_ema/mae_surf_p": ema_avg_surf_p,
            "per_split": active_metrics,
            "eval_source": active_label,
        }
        torch.save(active_state, model_path)
        tag = f" * [{active_label}]"

    ema_str = f"{ema_avg_surf_p:.4f}" if do_ema_eval else "skipped"
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val_raw={raw_avg_surf_p:.4f}  val_ema={ema_str}  active={active_label}{tag}"
    )
    for name in VAL_SPLIT_NAMES:
        print_split_metrics(name, active_metrics[name])

total_time = (time.time() - train_start) / 60.0
final_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
sec_per_epoch = (total_time * 60.0 / total_epochs_completed) if total_epochs_completed else float("nan")
print(f"\nTraining done in {total_time:.1f} min — epochs completed: {total_epochs_completed}, "
      f"sec/epoch: {sec_per_epoch:.1f}, peak GPU: {final_peak_gb:.1f} GB")

# --- Test evaluation + artifact upload ---
if best_metrics:
    eval_source = best_metrics.get("eval_source", "raw")
    print(
        f"\nBest val: epoch {best_metrics['epoch']}, "
        f"val_avg/mae_surf_p = {best_avg_surf_p:.4f} (source: {eval_source})"
    )
    wandb.summary.update({
        "best_epoch": best_metrics["epoch"],
        "best_val_avg/mae_surf_p": best_avg_surf_p,
        "best_val_avg_raw/mae_surf_p": best_metrics.get("val_avg_raw/mae_surf_p"),
        "best_val_avg_ema/mae_surf_p": best_metrics.get("val_avg_ema/mae_surf_p"),
        "best_eval_source": eval_source,
        "total_train_minutes": total_time,
        "total_epochs_completed": total_epochs_completed,
        "sec_per_epoch": sec_per_epoch,
        "peak_gpu_memory_gb": final_peak_gb,
    })

    _raw_module(model).load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, amp_ctx=amp_ctx)
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
        eval_source=eval_source,
    )
else:
    print("\nNo checkpoint was saved (no epoch improved on val_avg/mae_surf_p). Skipping artifact upload.")

wandb.finish()
