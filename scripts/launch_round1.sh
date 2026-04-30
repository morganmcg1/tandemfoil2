#!/usr/bin/env bash
# Round 1: 8 parallel single-GPU experiments, 60 epochs each, ~2h cap.
# All defaults match baseline; each run perturbs ONE thing at a time so we
# can attribute the win to a specific change.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs/round1

# Common flags
GROUP="mlintern-pai2-72h-v4-r1"
AGENT="ml-intern-r1"
COMMON=(--epochs 60 --agent "$AGENT" --wandb_group "$GROUP" --skip_test true --no_progress true)

# Helper: launch one experiment on a specific GPU
launch() {
  local gpu=$1; shift
  local name=$1; shift
  local logfile="logs/round1/${name}.log"
  echo "[GPU $gpu] $name → $logfile"
  CUDA_VISIBLE_DEVICES=$gpu SENPAI_TIMEOUT_MINUTES=125 \
    nohup python ./train.py "${COMMON[@]}" --wandb_name "${GROUP}/r1-${name}" "$@" \
    > "$logfile" 2>&1 &
  echo "  PID=$!"
  echo "$!" > "logs/round1/${name}.pid"
}

# 0. Pure baseline anchor (default config, longer training)
launch 0 "baseline"

# 1. L1 loss (replace MSE)
launch 1 "l1" --loss_type l1

# 2. Huber loss + extra surface-pressure weight (most direct optimization of metric)
launch 2 "huber-pw" --loss_type huber --huber_delta 0.5 --p_surf_weight 2.0

# 3. Wider model: hidden 128->256, head 4->8 (capacity)
launch 3 "wider256" --n_hidden 256 --n_head 8

# 4. Deeper model: 5->8 layers (Transolver++ default depth)
launch 4 "deep8" --n_layers 8

# 5. Transolver++ slice routing: ada-temp + gumbel
launch 5 "adagumbel" --ada_temp true --gumbel_softmax true

# 6. bf16 mixed-precision (throughput; check accuracy not hurt)
launch 6 "bf16" --bf16 true --grad_clip 1.0

# 7. OneCycle LR + L1 (better optimizer schedule + better loss)
launch 7 "onecycle-l1" --loss_type l1 --lr_schedule onecycle --lr 1e-3 --warmup_pct 0.05 --grad_clip 1.0

echo
echo "All 8 launched. Watch with: tail -f logs/round1/*.log"
