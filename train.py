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
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
            return self.mlp2(self.ln_3(fx))
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
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# Loss helper (applied in transformed space)
# ---------------------------------------------------------------------------

LOSS_TYPES = ("mse", "l1", "huber")


def elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str,
                     huber_delta: float) -> torch.Tensor:
    """Per-element loss in transformed space. Caller applies mask + reduction."""
    if loss_type == "mse":
        return (pred - target) ** 2
    if loss_type == "l1":
        return torch.abs(pred - target)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=huber_delta, reduction="none")
    raise ValueError(f"unknown loss_type={loss_type!r}, expected one of {LOSS_TYPES}")


# ---------------------------------------------------------------------------
# Pressure target reparameterization (PR #9)
# ---------------------------------------------------------------------------
#
# Per-sample domain routing for ``--y_norm per_domain``.
#
# Rule:
#   - If x dims 18–23 are all zero  ⇒ racecar_single (id 0)
#   - Else, racecar_tandem vs cruise is inferred via the split manifest
#     (which file a sample's global index lives in). These two tandem
#     domains have overlapping AoA / Re / NACA ranges, so pure feature
#     inference is unreliable — the manifest is the authoritative source.
#
# The lookup is built once per run from ``data/split_manifest.json`` and
# keyed by per-split local index (the ``{i:06d}.pt`` filename number),
# which is the order samples appear in ``manifest['splits'][<split>]``.
#
DOMAINS = ("racecar_single", "racecar_tandem", "cruise")
FILE_TO_DOMAIN = {
    "raceCar_single_randomFields.pickle": 0,
    "raceCar_randomFields_mgn_Part1.pickle": 1,
    "raceCar_randomFields_mgn_Part2.pickle": 1,
    "raceCar_randomFields_mgn_Part3.pickle": 1,
    "cruise_randomFields_mgn_Part1.pickle": 2,
    "cruise_randomFields_mgn_Part2.pickle": 2,
    "cruise_randomFields_mgn_Part3.pickle": 2,
}


def _load_domain_lookups() -> dict[str, list[int]]:
    """Return {split_name: [domain_id, …]} aligned with per-split local index."""
    manifest_path = Path(__file__).parent / "data" / "split_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    file_sizes = manifest["file_sizes"]
    cum = [0]
    for sz in file_sizes:
        cum.append(cum[-1] + sz)

    def gidx_to_domain(gidx: int) -> int:
        for i, pfile in enumerate(manifest["pickle_files"]):
            if cum[i] <= gidx < cum[i + 1]:
                return FILE_TO_DOMAIN[pfile]
        raise ValueError(f"global idx {gidx} out of range")

    return {
        split: [gidx_to_domain(g) for g in gidxs]
        for split, gidxs in manifest["splits"].items()
    }


class IndexedDataset(Dataset):
    """Wrap a SplitDataset/TestDataset to also yield a per-sample domain id.

    Keeps the base dataset's on-disk layout unchanged; the domain tag is
    looked up from an in-memory list aligned with the base's own index.
    """

    def __init__(self, base: Dataset, domain_ids: list[int]):
        self.base = base
        self.domain_ids = domain_ids
        assert len(self.base) <= len(self.domain_ids), (
            f"Not enough domain ids ({len(self.domain_ids)}) for base "
            f"({len(self.base)})"
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, is_surface = self.base[idx]
        return x, y, is_surface, int(self.domain_ids[idx])


def pad_collate_with_domain(batch):
    """Pad variable-length mesh samples; also emit per-sample domain ids.

    Returns (x, y, is_surface, mask, domain_ids). Matches ``data.pad_collate``
    for the first four elements; adds ``domain_ids: [B] int64``.
    """
    xs, ys, surfs, dids = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    domain_ids = torch.tensor(dids, dtype=torch.long)
    return x_pad, y_pad, surf_pad, mask, domain_ids


class YNormContext:
    """Forward/inverse y normalization for the 4 reparameterization modes.

    All tensors are held on ``device``. Shapes:
      - zscore:     y_mean, y_std  [3]
      - asinh:      y_mean, y_std  [3] (only :2 used — Ux,Uy via z-score);
                    asinh_scale (scalar) for the p channel
      - robust:     y_median, y_mad  [3]
      - per_domain: y_mean, y_std  [3, 3]  (D domains × C channels)

    For ``asinh``, Ux/Uy are z-scored as in baseline and the p channel is
    transformed as ``t_p = asinh(y_p / s)`` with inverse ``y_p = s sinh(t_p)``.
    No division by ``y_std_p`` on the p channel — the asinh already compresses
    the heavy tail into a reasonable O(1–5) range.
    """

    def __init__(
        self,
        mode: str,
        global_stats: dict[str, torch.Tensor],
        train_ds: Dataset,
        train_domain_ids: list[int],
        asinh_scale: float,
        device: torch.device,
    ):
        self.mode = mode
        self.device = device
        self.asinh_scale = float(asinh_scale)

        if mode == "zscore":
            self.y_mean = global_stats["y_mean"].to(device)
            self.y_std = global_stats["y_std"].to(device)

        elif mode == "asinh":
            # Ux, Uy keep z-score via the global stats; p uses asinh(y/s).
            self.y_mean = global_stats["y_mean"].to(device)
            self.y_std = global_stats["y_std"].to(device)

        elif mode == "robust":
            median, mad = self._compute_robust_stats(train_ds)
            self.y_median = median.to(device)
            self.y_mad = mad.to(device).clamp(min=1e-6)

        elif mode == "per_domain":
            mean_d, std_d = self._compute_per_domain_stats(
                train_ds, train_domain_ids
            )
            self.y_mean = mean_d.to(device)
            self.y_std = std_d.to(device).clamp(min=1e-6)

        else:
            raise ValueError(f"Unknown y_norm mode: {mode}")

    @staticmethod
    def _compute_robust_stats(
        train_ds: Dataset, nodes_per_sample: int = 2000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate per-channel median and MAD by subsampling train nodes.

        Holding all ~200M training nodes in RAM is avoidable; 2K-node
        random subsamples across 1499 training samples ≈ 3M points, which
        gives a very stable median while keeping the estimator cheap.
        """
        g = torch.Generator().manual_seed(0)
        pools = []
        for i in range(len(train_ds)):
            x, y, is_surface = train_ds[i]
            n = y.shape[0]
            k = min(nodes_per_sample, n)
            perm = torch.randperm(n, generator=g)[:k]
            pools.append(y[perm].float())
        Y = torch.cat(pools, dim=0)  # [~3M, 3]
        median = Y.median(dim=0).values.float()
        mad = 1.4826 * (Y - median).abs().median(dim=0).values.float()
        return median, mad

    @staticmethod
    def _compute_per_domain_stats(
        train_ds: Dataset, train_domain_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming per-domain mean/std across training nodes (float64)."""
        sums = torch.zeros(3, 3, dtype=torch.float64)
        sq = torch.zeros(3, 3, dtype=torch.float64)
        counts = torch.zeros(3, dtype=torch.int64)
        for i in range(len(train_ds)):
            x, y, is_surface = train_ds[i]
            d = train_domain_ids[i]
            yd = y.double()
            sums[d] += yd.sum(0)
            counts[d] += yd.shape[0]
        means = sums / counts.unsqueeze(-1).clamp(min=1).double()
        for i in range(len(train_ds)):
            x, y, is_surface = train_ds[i]
            d = train_domain_ids[i]
            sq[d] += ((y.double() - means[d]) ** 2).sum(0)
        stds = (sq / (counts - 1).unsqueeze(-1).clamp(min=1).double()).sqrt()
        return means.float(), stds.float()

    def forward(self, y: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """y: [B, N, 3] (original units) -> transformed [B, N, 3]."""
        if self.mode == "zscore":
            return (y - self.y_mean) / self.y_std
        if self.mode == "asinh":
            uv = (y[..., :2] - self.y_mean[:2]) / self.y_std[:2]
            p = torch.asinh(y[..., 2:] / self.asinh_scale)
            return torch.cat([uv, p], dim=-1)
        if self.mode == "robust":
            return (y - self.y_median) / self.y_mad
        if self.mode == "per_domain":
            mean = self.y_mean[domain_ids].unsqueeze(1)  # [B, 1, 3]
            std = self.y_std[domain_ids].unsqueeze(1)
            return (y - mean) / std
        raise RuntimeError(self.mode)

    def inverse(self, t: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """t: [B, N, 3] (transformed) -> [B, N, 3] original units."""
        if self.mode == "zscore":
            return t * self.y_std + self.y_mean
        if self.mode == "asinh":
            uv = t[..., :2] * self.y_std[:2] + self.y_mean[:2]
            p = self.asinh_scale * torch.sinh(t[..., 2:])
            return torch.cat([uv, p], dim=-1)
        if self.mode == "robust":
            return t * self.y_mad + self.y_median
        if self.mode == "per_domain":
            mean = self.y_mean[domain_ids].unsqueeze(1)
            std = self.y_std[domain_ids].unsqueeze(1)
            return t * std + mean
        raise RuntimeError(self.mode)

    def summary(self) -> dict:
        """Compact summary for logging to W&B config."""
        out: dict = {"y_norm": self.mode, "asinh_scale": self.asinh_scale}
        if self.mode == "robust":
            out["y_median"] = self.y_median.cpu().tolist()
            out["y_mad"] = self.y_mad.cpu().tolist()
        elif self.mode == "per_domain":
            out["y_mean_per_domain"] = self.y_mean.cpu().tolist()
            out["y_std_per_domain"] = self.y_std.cpu().tolist()
        return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(
    model,
    loader,
    stats,
    ynorm: "YNormContext",
    surf_weight: float,
    device,
    loss_type: str,
    huber_delta: float,
) -> dict[str, float]:
    """Run inference over a split and return metrics matching the organizer scorer.

    ``loss`` is the transformed-space loss used for training monitoring; the
    MAE channels are in the original target space (after inverting the y-norm)
    and accumulated per organizer ``score.py`` (float64, non-finite samples
    skipped).

    Also accumulates a per-domain surface-p MAE breakdown so mixed splits
    (val_re_rand / test_re_rand) can be analyzed by domain.
    """
    vol_loss_sum = surf_loss_sum = 0.0
    mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
    mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
    # Per-domain surface-p accumulators (3 domains).
    p_abs_per_domain = torch.zeros(3, dtype=torch.float64, device=device)
    n_surf_per_domain = torch.zeros(3, dtype=torch.int64, device=device)
    n_surf = n_vol = n_batches = 0

    with torch.no_grad():
        for x, y, is_surface, mask, domain_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = ynorm.forward(y, domain_ids)
            pred = model({"x": x_norm})["preds"]

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

            pred_orig = ynorm.inverse(pred, domain_ids)
            # Per-domain surf-p accumulation (respects sample skipping: we
            # compute over the same surf_mask used below, but only for
            # samples whose y is finite). Zero out non-finite samples' y/pred
            # before the subtraction so inf-finite never produces inf*0=NaN.
            B = y.shape[0]
            y_finite = torch.isfinite(y.reshape(B, -1)).all(dim=-1)
            if y_finite.any():
                sample_mask = y_finite.unsqueeze(-1).expand(-1, mask.shape[-1])
                eff_surf = surf_mask & sample_mask
                zeros_y = torch.zeros_like(y[..., 2])
                zeros_p = torch.zeros_like(pred_orig[..., 2])
                y_p_safe = torch.where(sample_mask, y[..., 2], zeros_y)
                pred_p_safe = torch.where(sample_mask, pred_orig[..., 2], zeros_p)
                err_p = (pred_p_safe.double() - y_p_safe.double()).abs()
                for d in range(3):
                    d_mask = (domain_ids == d).view(B, 1).expand_as(eff_surf)
                    combined = eff_surf & d_mask
                    p_abs_per_domain[d] += (err_p * combined.double()).sum()
                    n_surf_per_domain[d] += combined.sum()

            ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
            n_surf += ds
            n_vol += dv

    vol_loss = vol_loss_sum / max(n_batches, 1)
    surf_loss = surf_loss_sum / max(n_batches, 1)
    out = {"vol_loss": vol_loss, "surf_loss": surf_loss,
           "loss": vol_loss + surf_weight * surf_loss}
    out.update(finalize_split(mae_surf, mae_vol, n_surf, n_vol))
    # Per-domain surf-p MAE (NaN where no samples from that domain).
    for d, name in enumerate(DOMAINS):
        n = int(n_surf_per_domain[d].item())
        out[f"mae_surf_p_{name}"] = (
            float((p_abs_per_domain[d] / max(n, 1)).item()) if n > 0 else float("nan")
        )
        out[f"n_surf_nodes_{name}"] = n
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
        "y_norm": cfg.y_norm,
        "asinh_scale": cfg.asinh_scale,
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
    loss_type: str = "mse"  # mse | l1 | huber — applied in normalized space
    huber_delta: float = 1.0  # Huber transition point (normalized units)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation
    # Pressure target reparameterization (PR #9). Loss is computed in the
    # transformed space on all 3 channels; MAE metrics are computed in the
    # original physical space after inversion.
    y_norm: str = "zscore"  # zscore | asinh | robust | per_domain
    asinh_scale: float = 500.0  # asinh scale s for the p channel (orig units)


def _run_training() -> None:
    """Training/val/test entrypoint. Gated by ``if __name__ == "__main__"``
    so importing train.py for helpers (evaluate_split, YNormContext, …)
    doesn't trigger CLI parsing or W&B init.
    """
    cfg = sp.parse(Config)
    if cfg.loss_type not in LOSS_TYPES:
        raise ValueError(f"--loss_type must be one of {LOSS_TYPES}, got {cfg.loss_type!r}")
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
    MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))
    print(f"Loss: {cfg.loss_type}"
          + (f" (delta={cfg.huber_delta})" if cfg.loss_type == "huber" else "")
          + f"  surf_weight={cfg.surf_weight}")

    train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
    stats = {k: v.to(device) for k, v in stats.items()}

    # Per-sample domain ids (0=racecar_single, 1=racecar_tandem, 2=cruise).
    # Sourced from the split manifest — features alone don't separate the two
    # tandem domains (overlapping Re/AoA/NACA ranges).
    _domain_lookups = _load_domain_lookups()
    train_domain_ids = _domain_lookups["train"][: len(train_ds)]
    train_ds = IndexedDataset(train_ds, train_domain_ids)
    val_splits = {
        name: IndexedDataset(ds, _domain_lookups[name][: len(ds)])
        for name, ds in val_splits.items()
    }

    # Build y-normalization context once (robust/per_domain iterate train set once each).
    print(f"Building YNormContext (y_norm={cfg.y_norm}, asinh_scale={cfg.asinh_scale})...")
    ynorm = YNormContext(
        mode=cfg.y_norm,
        global_stats=stats,
        train_ds=train_ds.base,  # feed raw base; we only need y tensors here
        train_domain_ids=train_domain_ids,
        asinh_scale=cfg.asinh_scale,
        device=device,
    )

    loader_kwargs = dict(collate_fn=pad_collate_with_domain, num_workers=4, pin_memory=True,
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
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )

    model = Transolver(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transolver ({n_params/1e6:.2f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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
            "ynorm": ynorm.summary(),
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
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        for x, y, is_surface, mask, domain_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = ynorm.forward(y, domain_ids)
            pred = model({"x": x_norm})["preds"]
            per_elem = elementwise_loss(pred, y_norm, cfg.loss_type, cfg.huber_delta)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (per_elem * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (per_elem * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_vol /= max(n_batches, 1)
        epoch_surf /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        split_metrics = {
            name: evaluate_split(model, loader, stats, ynorm, cfg.surf_weight, device,
                                 cfg.loss_type, cfg.huber_delta)
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
            test_datasets = {
                name: IndexedDataset(ds, _domain_lookups[name][: len(ds)])
                for name, ds in test_datasets.items()
            }
            test_loaders = {
                name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
                for name, ds in test_datasets.items()
            }
            test_metrics = {
                name: evaluate_split(model, loader, stats, ynorm, cfg.surf_weight, device,
                                     cfg.loss_type, cfg.huber_delta)
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


if __name__ == "__main__":
    _run_training()
