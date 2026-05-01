#!/usr/bin/env bash
# Phase 1: 8 parallel candidates, each on its own GPU, capped at 6h wall.
#
# Anchor architecture: n_hidden=192, n_layers=8, n_head=8, slice_num=32, mlp_ratio=4
# (~3M params; between baseline 1M and Transolver paper 6M; ~5 min/epoch at bs=2).
# Common training: lr=1e-3, weight_decay=1e-5, AdamW, warmup_epochs=5, grad_clip=0.5, bs=2.
# All variations isolate one variable from the anchor.
#
# Each experiment uses --epochs 999 + SENPAI_TIMEOUT_MINUTES=360 (6 hours)
# so they run for as many epochs as fit, with cosine schedule over T_max=999.
#
# Usage: bash launchers/phase1.sh

set -u
cd /workspace/ml-intern-benchmark/target

GROUP=mlintern-pai2-24h-v3-r4
AGENT=ml-intern-r4
TIMEOUT_MIN=360            # 6 hours per experiment
EPOCHS=999

LOGDIR=/workspace/ml-intern-benchmark/target/logs/phase1
mkdir -p $LOGDIR

PIDFILE=/workspace/ml-intern-benchmark/target/logs/phase1.pids
> $PIDFILE

# Common architecture / training (medium-Transolver anchor)
COMMON_ARCH="--n_hidden 192 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 4"
COMMON_TRAIN="--lr 1e-3 --weight_decay 1e-5 --batch_size 2 --warmup_epochs 5 --grad_clip 0.5 --epochs $EPOCHS"

launch() {
  local gpu=$1
  local name=$2
  shift 2
  local extra="$*"
  local logfile=$LOGDIR/${name}.log
  echo "[GPU $gpu] $name -> $logfile" >&2
  CUDA_VISIBLE_DEVICES=$gpu SENPAI_TIMEOUT_MINUTES=$TIMEOUT_MIN nohup \
    python train.py \
      --agent $AGENT \
      --wandb_group $GROUP \
      --wandb_name "$GROUP/$name" \
      $COMMON_ARCH $COMMON_TRAIN $extra \
      > $logfile 2>&1 &
  local pid=$!
  echo "$gpu $name $pid" >> $PIDFILE
  echo "[GPU $gpu] PID=$pid" >&2
}

# 1. Anchor: medium L1
launch 0 medium-l1 \
  --loss_type l1 --surf_weight 10

# 2. Anchor + extra pressure surface weight
launch 1 medium-l1-pextra10 \
  --loss_type l1 --surf_weight 10 --p_extra_weight 10

# 3. Anchor + higher surf_weight (boost surface across all channels)
launch 2 medium-l1-surfw30 \
  --loss_type l1 --surf_weight 30

# 4. Anchor with Huber loss
launch 3 medium-huber \
  --loss_type huber --huber_beta 0.1 --surf_weight 10

# 5. Anchor with MSE (control vs L1)
launch 4 medium-mse \
  --loss_type mse --surf_weight 10

# 6. Higher LR (paper Adam ran lr=1e-3 effectively, try 2e-3)
launch 5 medium-l1-lr2e3 \
  --loss_type l1 --surf_weight 10 --lr 2e-3 --warmup_epochs 10

# 7. Deeper, narrower (depth scaling per Transolver Appx E.1: +30% from 8→40 layers)
launch 6 deep-l1 \
  --loss_type l1 --surf_weight 10 \
  --n_hidden 160 --n_layers 12

# 8. Long-baseline control (original arch, longer training, L1 loss)
launch 7 baseline-l1 \
  --loss_type l1 --surf_weight 10 \
  --n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2

echo "All launched. PID file: $PIDFILE"
cat $PIDFILE
