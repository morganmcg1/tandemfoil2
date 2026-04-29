#!/usr/bin/env bash
# Round-3 sweep: 90-min runs centered on the Round-2 winner: paper recipe
# (Adam + OneCycleLR + grad_clip=1.0 + lr=1e-3, properly-scoped epochs).
# Variations: longer training, bigger arch, warmup_pct, lr, surf_weight.

set -u
cd "$(dirname "$0")"
GROUP="mlintern-pai2-r4"
AGENT="ml-intern-r4"
TIMEOUT=90
mkdir -p logs

launch() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="logs/r3-gpu${gpu}-${name}.log"
    echo "[gpu${gpu}] launching ${name} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" SENPAI_TIMEOUT_MINUTES="${TIMEOUT}" nohup \
        python train.py --skip_test \
            --agent "${AGENT}" --wandb_group "${GROUP}" \
            --wandb_name "${GROUP}/r3-${name}" "$@" \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

# 90 min @ ~130s/ep ≈ 41 epochs default arch; 25 epochs arch_m; 12 epochs arch_l.

# GPU 0 — recipe over 40 epochs (longer training of winner).
launch 0 recipe40 \
    --epochs 40 --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 1 — recipe over 60 epochs (slower schedule, more cooldown).
launch 1 recipe60 \
    --epochs 60 --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 2 — recipe + arch_m (192/6/8/32), 25 epochs.
launch 2 recipe-arch-m \
    --epochs 25 --n_hidden 192 --n_layers 6 --n_head 8 --slice_num 32 --mlp_ratio 2 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 3 — recipe + arch_l (256/8/8/32 + grad_ckpt), 12 epochs.
launch 3 recipe-arch-l \
    --epochs 12 --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 --grad_checkpoint True \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 4 — recipe with longer warmup (0.3 = paper default).
launch 4 recipe-warmup30 \
    --epochs 40 --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.3

# GPU 5 — recipe with higher peak lr.
launch 5 recipe-lr2e-3 \
    --epochs 40 --optimizer adam --scheduler onecycle --lr 2e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 6 — recipe + surf_weight=5.
launch 6 recipe-sw5 \
    --epochs 40 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 7 — recipe + surf_weight=20.
launch 7 recipe-sw20 \
    --epochs 40 --surf_weight 20.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

echo "All round-3 jobs launched."
