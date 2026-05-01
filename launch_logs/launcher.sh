#!/bin/bash
# Helper: launch a training job on a specific GPU, record PID for clean shutdown.
#
# Usage:  bash launch_logs/launcher.sh <gpu_id> <wandb_short_name> <timeout_min> <flags...>
# Records PID and command to launch_logs/runs.tsv.

set -euo pipefail
GPU=$1; shift
NAME=$1; shift
TIMEOUT_MIN=$1; shift
FLAGS=( "$@" )

LOG_DIR="/workspace/ml-intern-benchmark/target/launch_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${NAME}.log"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cd /workspace/ml-intern-benchmark/target

# Train.py is the single entrypoint; use the canonical command shape from prompt.md.
nohup env CUDA_VISIBLE_DEVICES="$GPU" SENPAI_TIMEOUT_MINUTES="$TIMEOUT_MIN" \
  python ./train.py --epochs 999 --agent ml-intern-r4 \
    --wandb_group mlintern-pai2-72h-v4-r4 \
    --wandb_name "mlintern-pai2-72h-v4-r4/$NAME" \
    "${FLAGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!
printf '%s\tgpu=%s\tpid=%s\tname=%s\ttimeout_min=%s\tflags=%s\n' \
  "$TS" "$GPU" "$PID" "$NAME" "$TIMEOUT_MIN" "${FLAGS[*]}" \
  >> "$LOG_DIR/runs.tsv"
echo "$PID $LOG_FILE"
