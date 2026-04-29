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
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.cuda.amp import GradScaler, autocast
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


class RFFEncoder(nn.Module):
    """Random Fourier Features for spatial coordinates.

    Replaces input dims 0:2 (spatial position) with [sin(2π·B·pos), cos(2π·B·pos)]
    where B ~ N(0, σ²). Fixed (non-learned) projection. Output dim is 2 * n_freq.
    """
    def __init__(self, in_dim: int = 2, n_freq: int = 32, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * sigma)
        self.n_freq = n_freq

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (pos @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU FFN: gated FFN as in LLaMA / PaLM (Shazeer 2020, arxiv:2002.05202).

    out = W_down(SiLU(W_gate(x)) * W_up(x))

    Inner dim sized to match the parameter count of a vanilla 2-matmul FFN at
    the same ``mlp_ratio``: inner = int(hidden * mlp_ratio * 2 / 3), rounded
    up to a multiple of 8 for kernel efficiency.
    """

    def __init__(self, hidden_dim: int, mlp_ratio: int = 2):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio * 2 / 3)
        inner = ((inner + 7) // 8) * 8
        self.w_gate = nn.Linear(hidden_dim, inner, bias=True)
        self.w_up = nn.Linear(hidden_dim, inner, bias=True)
        self.w_down = nn.Linear(inner, hidden_dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


class FiLMNet(nn.Module):
    """FiLM (Feature-wise Linear Modulation) generator.

    Maps per-sample global features (Re, AoA, NACA, gap, stagger) to (γ, β)
    shift/scale parameters for each Transolver block's LayerNorms.
    Final linear is zero-initialised so the modulation starts at identity
    (γ=0, β=0; apply-time uses (1.0 + γ) so γ=0 → multiplier 1).
    """

    def __init__(self, cond_dim: int = 11, n_hidden: int = 128, n_layers: int = 5,
                 n_norms_per_block: int = 2, hidden_mult: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.n_norms_per_block = n_norms_per_block
        self.n_hidden = n_hidden
        out_dim = n_layers * n_norms_per_block * 2 * n_hidden
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        out = self.net(cond)
        return out.view(
            -1, self.n_layers, self.n_norms_per_block, 2, self.n_hidden
        )


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
        self.mlp = SwiGLU(hidden_dim, mlp_ratio=mlp_ratio)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, film=None):
        h_attn = self.ln_1(fx)
        if film is not None:
            gamma, beta = film[:, 0, 0, :], film[:, 0, 1, :]
            h_attn = (1.0 + gamma).unsqueeze(1) * h_attn + beta.unsqueeze(1)
        fx = self.attn(h_attn) + fx
        h_mlp = self.ln_2(fx)
        if film is not None:
            gamma, beta = film[:, 1, 0, :], film[:, 1, 1, :]
            h_mlp = (1.0 + gamma).unsqueeze(1) * h_mlp + beta.unsqueeze(1)
        fx = self.mlp(h_mlp) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 rff_n_freq: int = 32, rff_sigma: float = 1.0,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        self.rff = RFFEncoder(in_dim=2, n_freq=rff_n_freq, sigma=rff_sigma)
        rff_out_dim = 2 * rff_n_freq

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                  n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + rff_out_dim, n_hidden * 2, n_hidden,
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
        self.film_net = FiLMNet(
            cond_dim=11, n_hidden=n_hidden, n_layers=n_layers,
            n_norms_per_block=2, hidden_mult=2,
        )
        self.apply(self._init_weights)
        nn.init.zeros_(self.film_net.net[-1].weight)
        nn.init.zeros_(self.film_net.net[-1].bias)

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
        cond = x[:, 0, 13:24]
        film_all = self.film_net(cond)
        pos = x[..., :2]
        feat = x[..., 2:]
        rff = self.rff(pos)
        x = torch.cat([rff, feat], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for i, block in enumerate(self.blocks):
            fx = block(fx, film=film_all[:, i])
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device) -> dict[str, float]:
    """Evaluate a split and return metrics matching the organizer scorer.

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
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            pred = model({"x": x_norm})["preds"]

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
            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
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


class IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset so __getitem__ returns ((sample), idx)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


def indexed_pad_collate(batch):
    """Pad collate variant that also returns per-sample dataset indices."""
    samples, indices = zip(*batch)
    x, y, is_surface, mask = pad_collate(samples)
    return x, y, is_surface, mask, torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_MIN = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "30"))


class CautiousAdamW(torch.optim.AdamW):
    """AdamW with the 'cautious' mask from Liang et al. 2024 (arxiv:2411.16085).

    Masks updates whose direction disagrees with the current minibatch gradient
    sign. Standard AdamW step computes u = m_hat / (sqrt(v_hat)+eps); cautious
    sets u to 0 where sign(u) != sign(g), then rescales by 1/mask.mean() to
    preserve aggregate update magnitude. Tracks mean mask agreement per step
    for diagnostic logging.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask_running_sum = 0.0
        self._mask_running_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        per_step_mask_sum = 0.0
        per_step_mask_count = 0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse grads")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if wd != 0:
                    p.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                m_hat = exp_avg / bias1
                v_hat = exp_avg_sq / bias2

                u = m_hat / (v_hat.sqrt() + eps)

                mask = (u * grad > 0).to(u.dtype)
                mask_mean = mask.mean()
                per_step_mask_sum += float(mask_mean)
                per_step_mask_count += 1

                scale = mask_mean.clamp(min=1e-8)
                u = u * mask / scale

                p.add_(u, alpha=-lr)

        if per_step_mask_count > 0:
            self._mask_running_sum += per_step_mask_sum / per_step_mask_count
            self._mask_running_count += 1

        return loss

    def pop_avg_mask(self) -> float:
        if self._mask_running_count == 0:
            return float("nan")
        avg = self._mask_running_sum / self._mask_running_count
        self._mask_running_sum = 0.0
        self._mask_running_count = 0
        return avg


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    experiment_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip final test evaluation
    cosine_t_max: int = 13  # T_max for the post-warmup CosineAnnealingLR


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

# Online importance sampling state.
# Per-sample EMA of surface MAE in normalized space; used to derive temperature-
# scaled per-sample loss weights (NOT a sampler) once the curriculum activates.
# Domain-balanced WeightedRandomSampler is preserved.
train_ds_indexed = IndexedDataset(train_ds)
sample_loss_ema = torch.ones(len(train_ds), dtype=torch.float32)
ema_alpha = 0.3              # new = alpha*old + (1-alpha)*current — fast adapt
curriculum_warmup_epochs = 3 # epochs 1..N use unweighted loss (uniform curriculum)
weight_temperature = 0.3     # weights = (ema/ema.mean()).pow(temperature)

# Per-sample domain id for diagnostics.
with open(Path(cfg.splits_dir) / "meta.json") as _f:
    _meta = json.load(_f)
_idx_to_group = {i: name for name, idxs in _meta["domain_groups"].items() for i in idxs}
domain_names = sorted(set(_idx_to_group.values()))
sample_domain = torch.tensor(
    [domain_names.index(_idx_to_group[i]) for i in range(len(train_ds))],
    dtype=torch.long,
)

indexed_loader_kwargs = {**loader_kwargs, "collate_fn": indexed_pad_collate}

if cfg.debug:
    train_loader = DataLoader(train_ds_indexed, batch_size=cfg.batch_size,
                              shuffle=True, **indexed_loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds_indexed, batch_size=cfg.batch_size,
                              sampler=sampler, **indexed_loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=192,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

optimizer = CautiousAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = GradScaler()
warmup_epochs = 1
min_lr = cfg.lr / 100.0
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, cfg.cosine_t_max), eta_min=min_lr
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
)

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

    n_skipped_steps = 0
    curriculum_active = epoch >= curriculum_warmup_epochs
    for x, y, is_surface, mask, batch_indices in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        with autocast(dtype=torch.bfloat16):
            pred = model({"x": x_norm})["preds"]
            sq_err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)

            if curriculum_active:
                # Per-sample loss with EMA-derived importance weights.
                vol_count = vol_mask.sum(dim=1).clamp(min=1).to(sq_err.dtype) * sq_err.shape[-1]
                surf_count = surf_mask.sum(dim=1).clamp(min=1).to(sq_err.dtype) * sq_err.shape[-1]
                ps_vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum(dim=(1, 2)) / vol_count
                ps_surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum(dim=(1, 2)) / surf_count
                ps_loss = ps_vol_loss + cfg.surf_weight * ps_surf_loss

                ema_batch = sample_loss_ema[batch_indices].to(device).to(ps_loss.dtype)
                ema_global_mean = sample_loss_ema.mean().to(device).to(ps_loss.dtype).clamp(min=1e-8)
                ema_weights = (ema_batch / ema_global_mean).pow(weight_temperature)
                loss = (ema_weights * ps_loss).mean()
            else:
                loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < prev_scale:
            n_skipped_steps += 1

        # Update per-sample surface MAE EMA (in normalized space).
        with torch.no_grad():
            abs_err = (pred - y_norm).abs()
            n_surf_per_sample = surf_mask.sum(dim=1).clamp(min=1).to(abs_err.dtype)
            per_sample_surf_mae = (
                (abs_err * surf_mask.unsqueeze(-1).to(abs_err.dtype)).sum(dim=(1, 2))
                / (n_surf_per_sample * abs_err.shape[-1])
            ).cpu().to(torch.float32)
        for b in range(per_sample_surf_mae.shape[0]):
            sample_idx = int(batch_indices[b].item())
            sample_loss_ema[sample_idx] = (
                ema_alpha * sample_loss_ema[sample_idx]
                + (1.0 - ema_alpha) * per_sample_surf_mae[b]
            )

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)
    avg_mask = optimizer.pop_avg_mask()

    # --- Validate ---
    model.eval()
    split_metrics = {
        name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
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

    # EMA stats (overall + per-domain) for online importance weighting diagnostics.
    ema_min = float(sample_loss_ema.min().item())
    ema_max = float(sample_loss_ema.max().item())
    ema_mean = float(sample_loss_ema.mean().item())
    ema_per_domain = {
        domain_names[d]: float(sample_loss_ema[sample_domain == d].mean().item())
        for d in range(len(domain_names))
    }

    append_metrics_jsonl(metrics_jsonl_path, {
        "event": "epoch",
        "epoch": epoch + 1,
        "seconds": dt,
        "peak_memory_gb": peak_gb,
        "lr": current_lr,
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/avg_mask": avg_mask,
        "val_avg/mae_surf_p": avg_surf_p,
        "val_splits": split_metrics,
        "is_best": tag == " *",
        "amp_skipped_steps": n_skipped_steps,
        "amp_scale": scaler.get_scale(),
        "ema/min": ema_min,
        "ema/max": ema_max,
        "ema/mean": ema_mean,
        "ema/per_domain_mean": ema_per_domain,
        "curriculum/active": curriculum_active,
        "curriculum/warmup_epochs": curriculum_warmup_epochs,
        "curriculum/ema_alpha": ema_alpha,
        "curriculum/weight_temperature": weight_temperature,
    })
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  lr={current_lr:.2e}  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f} mask={avg_mask:.3f}]  "
        f"val_avg_surf_p={avg_surf_p:.4f}{tag}  "
        f"amp[skip={n_skipped_steps} scale={scaler.get_scale():.0f}]  "
        f"curriculum={'on' if curriculum_active else 'off'}"
    )
    print(
        f"    ema[min={ema_min:.4f} mean={ema_mean:.4f} max={ema_max:.4f}]  "
        + "  ".join(f"{name}={v:.4f}" for name, v in ema_per_domain.items())
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
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device)
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
