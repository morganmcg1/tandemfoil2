#!/usr/bin/env bash
# Phase 1: 8 parallel one-GPU runs, 60 epochs each.
# Each pinned with CUDA_VISIBLE_DEVICES so two jobs don't share a GPU.
set -u

cd "$(dirname "$0")"
mkdir -p logs
GROUP="mlintern-pai2-72h-v4-r5"

launch() {
  local gpu="$1" name="$2"; shift 2
  local logfile="logs/p1_g${gpu}_${name}.log"
  CUDA_VISIBLE_DEVICES="$gpu" SENPAI_TIMEOUT_MINUTES=720 \
    nohup python ./train.py --epochs 60 --agent ml-intern-r5 \
    --wandb_group "$GROUP" --wandb_name "$GROUP/p1-$name" \
    "$@" > "$logfile" 2>&1 &
  local pid=$!
  echo "$gpu p1-$name PID=$pid log=$logfile"
}

# E1: baseline reference
launch 0 baseline

# E2: bf16 - same arch, mixed precision (faster + maybe better generalization)
launch 1 bf16 --bf16 True

# E3: Transolver++ eidetic states (Rep-Slice + Ada-Temp), same size
launch 2 eidetic --use_eidetic True

# E4: scaled-up model — AirfRANS Transolver config
launch 3 h256-nl8 --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 128 --mlp_ratio 2

# E5: Huber loss (robust to pressure outliers)
launch 4 huber --loss huber --huber_delta 1.0

# E6: more surface emphasis
launch 5 surf20 --surf_weight 20

# E7: lower learning rate
launch 6 lr1e-4 --lr 1e-4

# E8: medium scale-up + slightly higher lr + grad clip
launch 7 h192-nl6 --n_hidden 192 --n_layers 6 --n_head 6 --slice_num 96 --mlp_ratio 2 --grad_clip 1.0

echo "All jobs launched."
