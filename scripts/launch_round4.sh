#!/usr/bin/env bash
# Round 4: launch as round-3 GPUs free up. Each invocation launches one job.
# Usage: bash scripts/launch_round4.sh <gpu> <name> [extra args...]
set -u

cd "$(dirname "$0")/.."
mkdir -p logs/round4

GROUP="mlintern-pai2-72h-v4-r1"
AGENT="ml-intern-r1"
EPOCHS="${EPOCHS:-200}"
TIMEOUT="${TIMEOUT_MIN:-360}"
COMMON=(--epochs "$EPOCHS" --agent "$AGENT" --wandb_group "$GROUP" --skip_test true --no_progress true --bf16 true --grad_clip 1.0)

GPU="$1"; shift
NAME="$1"; shift
LOGFILE="logs/round4/${NAME}.log"
echo "[GPU $GPU] $NAME (epochs=$EPOCHS, timeout=${TIMEOUT}min) → $LOGFILE"
CUDA_VISIBLE_DEVICES=$GPU SENPAI_TIMEOUT_MINUTES=$TIMEOUT \
  nohup python ./train.py "${COMMON[@]}" --wandb_name "${GROUP}/r4-${NAME}" "$@" \
  > "$LOGFILE" 2>&1 &
sleep 5
PID=$(ps -eo pid,ppid,cmd --no-headers | grep "wandb_name ${GROUP}/r4-${NAME}" | grep -v grep | awk '$2==1 {print $1}' | head -1)
echo "  main PID=$PID"
echo "$PID" > "logs/round4/${NAME}.pid"
