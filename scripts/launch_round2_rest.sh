#!/usr/bin/env bash
# Round 2 (rest): l1-ema is already running on GPU 6. Launch the other 7 on
# GPUs 0-5 and 7.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs/round2

GROUP="mlintern-pai2-72h-v4-r1"
AGENT="ml-intern-r1"
COMMON=(--epochs 80 --agent "$AGENT" --wandb_group "$GROUP" --skip_test true --no_progress true --bf16 true --grad_clip 1.0)

launch() {
  local gpu=$1; shift
  local name=$1; shift
  local logfile="logs/round2/${name}.log"
  echo "[GPU $gpu] $name → $logfile"
  CUDA_VISIBLE_DEVICES=$gpu SENPAI_TIMEOUT_MINUTES=200 \
    nohup python ./train.py "${COMMON[@]}" --wandb_name "${GROUP}/r2-${name}" "$@" \
    > "$logfile" 2>&1 &
  echo "  PID=$!"
  echo "$!" > "logs/round2/${name}.pid"
}

# (GPU 6 already has l1-ema running)

# 0. L1 + EMA + p_surf_weight
launch 0 "l1-ema-pw" --loss_type l1 --ema_decay 0.999 --p_surf_weight 2.0

# 1. L1 + OneCycle + EMA (round-1 leader + EMA)
launch 1 "l1-onecycle-ema" --loss_type l1 --lr_schedule onecycle --lr 1e-3 --warmup_pct 0.05 --ema_decay 0.999

# 2. L1 + EMA + width 192
launch 2 "l1-ema-w192" --loss_type l1 --ema_decay 0.999 --n_hidden 192 --n_head 6

# 3. Huber + p_surf_weight + EMA
launch 3 "huber-pw-ema" --loss_type huber --huber_delta 0.5 --p_surf_weight 2.0 --ema_decay 0.999

# 4. L1 + EMA + input noise σ=0.005
launch 4 "l1-ema-noise" --loss_type l1 --ema_decay 0.999 --input_noise_std 0.005

# 5. L1 + EMA + depth=7
launch 5 "l1-ema-deep7" --loss_type l1 --ema_decay 0.999 --n_layers 7

# 7. L1 + EMA + ada_temp + gumbel
launch 7 "l1-ema-ada" --loss_type l1 --ema_decay 0.999 --ada_temp true --gumbel_softmax true

echo
echo "Round 2 (7 more) launched. Watch with: tail -f logs/round2/*.log"
