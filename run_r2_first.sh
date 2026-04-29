#!/usr/bin/env bash
# Round 2 first wave: launch on free GPUs (0, 3, 5, 6, 7).
# 60-min budget, --epochs set so cosine LR actually decays in budget.
set -u
cd "$(dirname "$0")"
GROUP="mlintern-pai2-r4"
AGENT="ml-intern-r4"
TIMEOUT=60
mkdir -p logs

launch() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="logs/r2-gpu${gpu}-${name}.log"
    echo "[gpu${gpu}] launching ${name} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" SENPAI_TIMEOUT_MINUTES="${TIMEOUT}" nohup \
        python train.py --skip_test \
            --agent "${AGENT}" --wandb_group "${GROUP}" \
            --wandb_name "${GROUP}/r2-${name}" "$@" \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

# GPU 0 — baseline cosine over 27 epochs (proper schedule).
launch 0 bl-cos27 --epochs 27

# GPU 3 — paper recipe, properly scoped (Adam+OneCycle, lr=1e-3, clip=1).
launch 3 recipe-cos27 --epochs 27 --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0 --warmup_pct 0.05

# GPU 5 — surf_weight=5 (between 2 and 10), cosine 27.
launch 5 sw5-cos27 --surf_weight 5.0 --epochs 27

# GPU 6 — baseline cosine over 50 (longer schedule).
launch 6 bl-cos50 --epochs 50

# GPU 7 — lr=1e-3 cosine 27 ep.
launch 7 lr1e3-cos27 --lr 1e-3 --epochs 27

echo "First wave launched."
