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
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn.functional as F
import wandb
import yaml
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
from models import Transolver

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(model, loader, stats, surf_weight, device, autocast_dtype=None) -> dict[str, float]:
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
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    pred = model({"x": x_norm})["preds"]
                pred = pred.float()
            else:
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
    """Log the best checkpoint as a wandb model artifact."""
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
    # --- training ---
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    grad_clip: float = 0.0  # 0 disables; >0 clips by global norm
    # --- model ---
    n_hidden: int = 128
    n_layers: int = 5
    n_head: int = 4
    slice_num: int = 64
    mlp_ratio: int = 2
    dropout: float = 0.0
    use_eidetic: bool = False  # Transolver++ Rep-Slice + Ada-Temp
    # --- options ---
    bf16: bool = False  # autocast bf16 forward/backward
    scheduler: str = "cosine"  # cosine | onecycle | none
    loss: str = "mse"  # mse | huber
    huber_delta: float = 1.0
    # --- bookkeeping ---
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    skip_test: bool = False  # skip end-of-run test evaluation


def main() -> None:
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs
    MAX_TIMEOUT_MIN = DEFAULT_TIMEOUT_MIN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))
    autocast_dtype = torch.bfloat16 if cfg.bf16 else None

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
        n_hidden=cfg.n_hidden,
        n_layers=cfg.n_layers,
        n_head=cfg.n_head,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        use_eidetic=cfg.use_eidetic,
        output_fields=["Ux", "Uy", "p"],
        output_dims=[1, 1, 1],
    )

    model = Transolver(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Transolver ({n_params/1e6:.2f}M params, eidetic={cfg.use_eidetic})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    elif cfg.scheduler == "onecycle":
        steps_per_epoch = max(1, (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=MAX_EPOCHS * steps_per_epoch,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e3,
        )
    else:
        scheduler = None

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

        for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    pred = model({"x": x_norm})["preds"]
                pred = pred.float()
            else:
                pred = model({"x": x_norm})["preds"]

            if cfg.loss == "huber":
                # Huber on per-element residual; mask-weighted reductions below.
                err = F.huber_loss(pred, y_norm, reduction="none", delta=cfg.huber_delta)
            else:
                err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if cfg.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        if cfg.scheduler == "cosine" and scheduler is not None:
            scheduler.step()
        epoch_vol /= max(n_batches, 1)
        epoch_surf /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        split_metrics = {
            name: evaluate_split(model, loader, stats, cfg.surf_weight, device, autocast_dtype)
            for name, loader in val_loaders.items()
        }
        val_avg = aggregate_splits(split_metrics)
        avg_surf_p = val_avg["avg/mae_surf_p"]
        val_loss_mean = sum(m["loss"] for m in split_metrics.values()) / len(split_metrics)
        dt = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        log_metrics = {
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "val/loss": val_loss_mean,
            "lr": cur_lr,
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
                name: evaluate_split(model, loader, stats, cfg.surf_weight, device, autocast_dtype)
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
    main()
