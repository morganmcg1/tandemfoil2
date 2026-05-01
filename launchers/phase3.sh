#!/usr/bin/env bash
# Phase 3: more seeds and a couple of variations to enable ensembling.
#
# Phase 2 cluster is tight: best=baseline-l1-200-s42=34.55 across seeds 0/1/42.
# Phase 3 grows the ensemble: 5 more L1 seeds + 2 huber + 1 lr1e3 alt seed.
#
# All runs use --epochs 200 with cosine T_max=200 + warmup=5.
# Per-run timeout 7h (420 min) to leave safety margin for ensemble eval.

set -u
cd /workspace/ml-intern-benchmark/target

GROUP=mlintern-pai2-24h-v3-r4
AGENT=ml-intern-r4
TIMEOUT_MIN=420  # 7h cap per run -- leaves ~2h for ensemble eval / cleanup
EPOCHS=200

LOGDIR=/workspace/ml-intern-benchmark/target/logs/phase3
mkdir -p $LOGDIR

PIDFILE=/workspace/ml-intern-benchmark/target/logs/phase3.pids
> $PIDFILE

ARCH_BASE="--n_hidden 128 --n_layers 5 --n_head 4 --slice_num 64 --mlp_ratio 2"
TRAIN_BASE="--lr 5e-4 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 5 --grad_clip 0.5 --epochs $EPOCHS"

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

# 5 more baseline-l1 seeds for ensemble diversity
launch 0 baseline-l1-200-s2 \
  $ARCH_BASE $TRAIN_BASE --loss_type l1 --surf_weight 10 --seed 2

launch 1 baseline-l1-200-s3 \
  $ARCH_BASE $TRAIN_BASE --loss_type l1 --surf_weight 10 --seed 3

launch 2 baseline-l1-200-s4 \
  $ARCH_BASE $TRAIN_BASE --loss_type l1 --surf_weight 10 --seed 4

launch 3 baseline-l1-200-s5 \
  $ARCH_BASE $TRAIN_BASE --loss_type l1 --surf_weight 10 --seed 5

launch 4 baseline-l1-200-s7 \
  $ARCH_BASE $TRAIN_BASE --loss_type l1 --surf_weight 10 --seed 7

# 2 more huber seeds (for cross-loss ensembling)
launch 5 baseline-huber-200-s1 \
  $ARCH_BASE $TRAIN_BASE --loss_type huber --huber_beta 0.1 --surf_weight 10 --seed 1

launch 6 baseline-huber-200-s42 \
  $ARCH_BASE $TRAIN_BASE --loss_type huber --huber_beta 0.1 --surf_weight 10 --seed 42

# 1 more lr1e3 seed
launch 7 baseline-l1-200-lr1e3-s1 \
  $ARCH_BASE --lr 1e-3 --weight_decay 1e-4 --batch_size 4 --warmup_epochs 10 --grad_clip 0.5 --epochs $EPOCHS \
  --loss_type l1 --surf_weight 10 --seed 1

echo "Phase 3 launched. PID file: $PIDFILE"
cat $PIDFILE
