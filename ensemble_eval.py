"""Ensemble evaluation: load multiple best checkpoints and average predictions on val/test.

Usage:
  python ensemble_eval.py \
    --checkpoint_paths model_dir1/checkpoint.pt,model_dir2/checkpoint.pt,... \
    --config_paths model_dir1/config.yaml,model_dir2/config.yaml,... \
    [--splits_dir /mnt/new-pvc/datasets/tandemfoil/splits_v2] \
    [--batch_size 4]

The script loads each model from its config + checkpoint, runs inference on the val
and test splits in normalized space, averages predictions across the ensemble, and
reports the same metrics as train.py for comparison.
"""

from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# Re-import from train.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Note: import the model class from train.py without re-running its training script
import importlib.util
_train_path = Path(__file__).resolve().parent / "train.py"
_spec = importlib.util.spec_from_file_location("_train_module", _train_path)

# We'll import only the model class — but train.py runs at import time.
# So instead, recreate Transolver class here (verbatim from train.py).
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
            self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )
        x_mid = (
            self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False,
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
                 slice_nums=None,
                 output_fields=None, output_dims=None):
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
        if slice_nums is None:
            slice_nums = [slice_num] * n_layers
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_nums[i], last_layer=(i == n_layers - 1),
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


def main():
    from data import (
        VAL_SPLIT_NAMES,
        TEST_SPLIT_NAMES,
        load_data,
        load_test_data,
        pad_collate,
        accumulate_batch,
        finalize_split,
        aggregate_splits,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_paths", required=True, type=str,
                        help="Comma-separated checkpoint paths")
    parser.add_argument("--config_paths", required=True, type=str,
                        help="Comma-separated config paths")
    parser.add_argument("--weights", default=None, type=str,
                        help="Comma-separated per-model weights (e.g. inverse val MAE). "
                             "If omitted, uniform.")
    parser.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    ckpts = args.checkpoint_paths.split(",")
    cfgs = args.config_paths.split(",")
    assert len(ckpts) == len(cfgs), "checkpoint_paths and config_paths must have same length"

    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        assert len(weights) == len(ckpts), "weights must have same length as checkpoints"
        # Normalize weights to sum to 1
        s = sum(weights)
        weights = [w / s for w in weights]
    else:
        weights = [1.0 / len(ckpts)] * len(ckpts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, ensemble size: {len(ckpts)}, weights: {[round(w, 3) for w in weights]}")

    # Load data
    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    # Load all models
    models = []
    for ckpt_path, cfg_path in zip(ckpts, cfgs):
        with open(cfg_path) as f:
            mc = yaml.safe_load(f)
        # Convert known fields to matching names if needed
        m = Transolver(**mc).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
        print(f"  Loaded {Path(ckpt_path).parent.name}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")

    def evaluate_split_ensemble(loader):
        mae_surf = torch.zeros(3, dtype=torch.float64, device=device)
        mae_vol = torch.zeros(3, dtype=torch.float64, device=device)
        n_surf = n_vol = 0
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # Filter non-finite-y samples (e.g. test_geom_camber_cruise sample 000020)
                # before passing to scoring; scoring.py's masked-sum has a NaN propagation
                # bug for inf*0=NaN, so we keep the bad sample out of the batch entirely.
                y_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)
                if not y_finite.any():
                    continue
                if not y_finite.all():
                    good = torch.where(y_finite)[0]
                    x = x[good]
                    y = y[good]
                    is_surface = is_surface[good]
                    mask = mask[good]

                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                # Weighted average of predictions in normalized space
                pred = None
                for m, w in zip(models, weights):
                    p = m({"x": x_norm})["preds"] * w
                    pred = p if pred is None else pred + p
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                ds, dv = accumulate_batch(pred_orig, y, is_surface, mask, mae_surf, mae_vol)
                n_surf += ds
                n_vol += dv
        return finalize_split(mae_surf, mae_vol, n_surf, n_vol)

    print("\n=== Validation ensemble ===")
    val_metrics = {name: evaluate_split_ensemble(loader) for name, loader in val_loaders.items()}
    val_avg = aggregate_splits(val_metrics)
    print(f"  val_avg/mae_surf_p = {val_avg['avg/mae_surf_p']:.4f}")
    for name, m in val_metrics.items():
        print(f"  {name:30s} surf_p={m['mae_surf_p']:.2f} vol_p={m['mae_vol_p']:.2f} "
              f"surf_Ux={m['mae_surf_Ux']:.2f} surf_Uy={m['mae_surf_Uy']:.2f}")

    if not args.skip_test:
        print("\n=== Test ensemble ===")
        test_datasets = load_test_data(args.splits_dir, debug=False)
        test_loaders = {
            name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            for name, ds in test_datasets.items()
        }
        test_metrics = {name: evaluate_split_ensemble(loader) for name, loader in test_loaders.items()}
        test_avg = aggregate_splits(test_metrics)
        print(f"  test_avg/mae_surf_p = {test_avg['avg/mae_surf_p']:.4f}")
        for name, m in test_metrics.items():
            print(f"  {name:30s} surf_p={m['mae_surf_p']:.2f} vol_p={m['mae_vol_p']:.2f}")
        return val_avg, test_avg
    return val_avg, None


if __name__ == "__main__":
    main()
