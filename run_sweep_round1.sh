#!/usr/bin/env bash
# Round-1 ablation sweep on TandemFoilSet-Balanced.
# Each job runs ~30 min on a single GPU (1-7). GPU 0 is occupied by the baseline.
# All runs use --epochs 999 (no epoch limit); SENPAI_TIMEOUT_MINUTES=30 caps wall clock.

set -u

cd "$(dirname "$0")"

GROUP="mlintern-pai2-r4"
AGENT="ml-intern-r4"
TIMEOUT=30

mkdir -p logs

launch() {
    local gpu="$1"
    local name="$2"
    shift 2
    local logfile="logs/r1-gpu${gpu}-${name}.log"
    echo "[gpu${gpu}] launching ${name} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" SENPAI_TIMEOUT_MINUTES="${TIMEOUT}" nohup \
        python train.py --epochs 999 --skip_test \
            --agent "${AGENT}" --wandb_group "${GROUP}" \
            --wandb_name "${GROUP}/r1-${name}" "$@" \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

# GPU 1 — arch_m: bump hidden/depth/heads/slice_num to a mid-paper-aligned config.
launch 1 arch-m \
    --n_hidden 192 --n_layers 6 --n_head 8 --slice_num 32 --mlp_ratio 2

# GPU 2 — arch_l: paper-aligned arch with grad_checkpoint to fit memory.
launch 2 arch-l-gc \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 \
    --grad_checkpoint True

# GPU 3 — recipe: just swap optim/scheduler to paper recipe (Adam + OneCycleLR + clip).
launch 3 recipe \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 4 — arch_l + recipe (combined).
launch 4 arch-l-gc-recipe \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 \
    --grad_checkpoint True \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 5 — surf_weight=2 (lower surf weight, paper-aligned).
launch 5 sw2 --surf_weight 2.0

# GPU 6 — surf_weight=20 (higher surf weight, more aggressive).
launch 6 sw20 --surf_weight 20.0

# GPU 7 — bigger batch size + scaled lr.
launch 7 bs8-lr1e3 --batch_size 8 --lr 1e-3

echo "All round-1 jobs launched. Tail logs in logs/r1-gpu*.log"
