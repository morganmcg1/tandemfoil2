#!/usr/bin/env bash
# Round-4 final long sweep: 3-hour runs centered on the R3 winner
# (recipe + warmup_pct=0.3) and its combinations. Covers:
#  - Extend leader with more epochs
#  - Combine warmup30 with surf_weight=5
#  - Combine warmup30 with bigger arch (arch_m)
#  - Paper arch (arch_l_gc) with warmup30
#  - Variance check (multiple seeds)

set -u
cd "$(dirname "$0")"
GROUP="mlintern-pai2-r4"
AGENT="ml-intern-r4"
TIMEOUT=180   # 3 hours
mkdir -p logs

launch() {
    local gpu="$1"; local name="$2"; shift 2
    local logfile="logs/r4-gpu${gpu}-${name}.log"
    echo "[gpu${gpu}] launching ${name} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" SENPAI_TIMEOUT_MINUTES="${TIMEOUT}" nohup \
        python train.py --skip_test \
            --agent "${AGENT}" --wandb_group "${GROUP}" \
            --wandb_name "${GROUP}/r4-${name}" "$@" \
        > "${logfile}" 2>&1 &
    echo "  pid=$!"
}

# 3 hr budget. Per-epoch times:
#   default (0.66M): ~132 s   →  80 epochs / 3 hr
#   arch_m  (1.70M): ~220 s   →  49 epochs / 3 hr
#   arch_l  (3.94M): ~425 s   →  25 epochs / 3 hr

# GPU 0 — leader: warmup30 + ep=70 (default arch).
launch 0 w30-70 \
    --epochs 70 --warmup_pct 0.3 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 1 — leader + sw5 (combine top 2 levers).
launch 1 w30-sw5-70 \
    --epochs 70 --warmup_pct 0.3 --surf_weight 5.0 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 2 — leader + arch_m, 40 epochs.
launch 2 w30-am-40 \
    --epochs 40 --warmup_pct 0.3 \
    --n_hidden 192 --n_layers 6 --n_head 8 --slice_num 32 --mlp_ratio 2 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 3 — leader + arch_m + sw5 (combine all winners).
launch 3 w30-am-sw5-40 \
    --epochs 40 --warmup_pct 0.3 --surf_weight 5.0 \
    --n_hidden 192 --n_layers 6 --n_head 8 --slice_num 32 --mlp_ratio 2 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 4 — leader + arch_l_gc, 22 epochs.
launch 4 w30-al-22 \
    --epochs 22 --warmup_pct 0.3 \
    --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 2 --grad_checkpoint True \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 5 — leader replicated with seed=1.
launch 5 w30-70-s1 \
    --epochs 70 --warmup_pct 0.3 --seed 1 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 6 — leader replicated with seed=2.
launch 6 w30-70-s2 \
    --epochs 70 --warmup_pct 0.3 --seed 2 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

# GPU 7 — explore even longer warmup (50%) for default arch.
launch 7 w50-70 \
    --epochs 70 --warmup_pct 0.5 \
    --optimizer adam --scheduler onecycle --lr 1e-3 --grad_clip 1.0

echo "All round-4 jobs launched."
