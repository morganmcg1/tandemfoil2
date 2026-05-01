#!/usr/bin/env bash
# Phase 2: longer training of variations of phase 1 winners.
#
# Phase 1 final rankings (best val_avg/mae_surf_p):
#  1. baseline-l1               37.80  (128/5/4/64/2 baseline arch, l1, lr=5e-4) -- WINNER
#  2. medium-l1                 48.43  (192/8/8/32/4, l1, lr=1e-3)
#  3. medium-huber              49.97  (same arch, huber)
#  4. medium-mse                50.32  (same arch, mse)
#  5. medium-l1-surfw30         51.14  (l1, surf_weight=30)  - surf-weight tweak no help
#  6. medium-l1-pextra10        52.52  (l1, p_extra_weight=10) - no help either
#  7. medium-l1-lr2e3           53.86  (lr=2e-3)            - too high LR
#  8. deep-l1                   57.75  (160/12)             - deeper made it worse
#
# Phase 2 lessons:
#  - small (1M) baseline arch wins by 10+ points
#  - p_extra_weight, surf_weight tuning did NOT help
#  - lr=1e-3 OK for medium; lr=2e-3 too high
#  - L1 ~ MSE ~ Huber within 2 points; L1 slight edge
#
# Phase 2 setup:
#  - 6 baseline-arch variations including 3 seeds for variance / ensemble
#  - 1 medium variant with proper LR schedule (does it catch up?)
#  - 1 baseline with longer training (300 ep)
#
# All runs use full cosine decay (T_max = epochs). SENPAI_TIMEOUT_MINUTES=540 (9h) caps wall.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP=mlintern-pai2-24h-v3-r4
AGENT=ml-intern-r4
TIMEOUT_MIN=540  # 9h cap per run

LOGDIR=/workspace/ml-intern-benchmark/target/logs/phase2
mkdir -p $LOGDIR

PIDFILE=/workspace/ml-intern-benchmark/target/logs/phase2.pids
> $PIDFILE

ARCH_BASE="--n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2"

launch() {
  local gpu=$1
  local name=$2
  shift 2
  local logfile=$LOGDIR/${name}.log
  echo "[GPU $gpu] $name -> $logfile" >&2
  CUDA_VISIBLE_DEVICES=$gpu SENPAI_TIMEOUT_MINUTES=$TIMEOUT_MIN nohup \
    python train.py \
      --agent $AGENT \
      --wandb_group $GROUP \
      --wandb_name "$GROUP/$name" \
      "$@" \
      > $logfile 2>&1 &
  local pid=$!
  echo "$gpu $name $pid" >> $PIDFILE
  echo "[GPU $gpu] PID=$pid" >&2
}

# 1. baseline-l1, 200 epochs proper schedule, seed 0 (control)
launch 0 baseline-l1-200 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs 200 \
  --loss_type l1 --surf_weight 10 --seed 0

# 2. baseline-l1 seed 1 (variance)
launch 1 baseline-l1-200-s1 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs 200 \
  --loss_type l1 --surf_weight 10 --seed 1

# 3. baseline-l1 seed 42 (variance)
launch 2 baseline-l1-200-s42 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs 200 \
  --loss_type l1 --surf_weight 10 --seed 42

# 4. baseline-huber-200 (test if Huber > L1 with small arch)
launch 3 baseline-huber-200 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs 200 \
  --loss_type huber --huber_beta 0.1 --surf_weight 10

# 5. baseline-l1 + higher LR (1e-3) and longer warmup (10)
launch 4 baseline-l1-200-lr1e3 \
  $ARCH_BASE --lr 1e-3 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 10 --grad_clip 0.5 --epochs 200 \
  --loss_type l1 --surf_weight 10

# 6. baseline-l1 longer training (300 epochs)
launch 5 baseline-l1-300 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs 300 \
  --loss_type l1 --surf_weight 10

# 7. medium-l1-80: best medium variant with proper LR schedule (T_max=80, ~7.3h)
launch 6 medium-l1-80 \
  --n_hidden 192 --n_layers 8 --n_head 8 --slice_num 32 --mlp_ratio 4 \
  --lr 1e-3 --weight_decay 1e-5 --batch_size 2 --warmup_epochs 5 --grad_clip 0.5 --epochs 80 \
  --loss_type l1 --surf_weight 10

# 8. baseline-l1 with bs=8 (larger batch, 1M model can handle it)
#    More accurate gradients; tests if bs=4 was suboptimal
launch 7 baseline-l1-200-bs8 \
  $ARCH_BASE --lr 5e-4 --weight_decay 1e-4 --batch_size 8 --warmup_epochs 5 --grad_clip 0.5 --epochs 200 \
  --loss_type l1 --surf_weight 10

echo "Phase 2 launched. PID file: $PIDFILE"
cat $PIDFILE
