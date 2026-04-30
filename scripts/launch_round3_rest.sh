#!/usr/bin/env bash
# Round 3 rest: launch as round 2 GPUs free up. r3-l1-onecycle-ema-150 already
# on GPU 6.
#
# Usage: bash scripts/launch_round3_rest.sh <gpu> <name> [extra args...]
set -u

cd "$(dirname "$0")/.."
mkdir -p logs/round3

GROUP="mlintern-pai2-72h-v4-r1"
AGENT="ml-intern-r1"
COMMON=(--epochs 150 --agent "$AGENT" --wandb_group "$GROUP" --skip_test true --no_progress true --bf16 true --grad_clip 1.0)

GPU="$1"; shift
NAME="$1"; shift
LOGFILE="logs/round3/${NAME}.log"
echo "[GPU $GPU] $NAME → $LOGFILE"
CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=290 \
  nohup python ./train.py "${COMMON[@]}" --wandb_name "${GROUP}/r3-${NAME}" "$@" \
  > "$LOGFILE" 2>&1 &
sleep 5
PID=$(ps -eo pid,ppid,cmd --no-headers | grep "wandb_name ${GROUP}/r3-${NAME}" | grep -v grep | awk '$2==1 {print $1}' | head -1)
echo "  main PID=$PID"
echo "$PID" > "logs/round3/${NAME}.pid"
