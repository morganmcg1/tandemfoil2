#!/usr/bin/env bash
# Sequential 8-config sweep for tanjiro DropPath / attention-dropout PR #253
# Single-GPU node — runs are dispatched one after another, each capped by
# SENPAI_TIMEOUT_MINUTES (default 30 min).
set -u

cd "$(dirname "$0")"

LOG_DIR="$PWD/sweep_logs"
mkdir -p "$LOG_DIR"

# (drop_path_rate, dropout, seed, name)
configs=(
  "0.0|0.0|0|tanjiro/no-reg-s0"
  "0.0|0.0|1|tanjiro/no-reg-s1"
  "0.05|0.0|0|tanjiro/dp005-s0"
  "0.05|0.0|1|tanjiro/dp005-s1"
  "0.1|0.0|0|tanjiro/dp010-s0"
  "0.1|0.0|1|tanjiro/dp010-s1"
  "0.0|0.1|0|tanjiro/attn-drop-s0"
  "0.05|0.05|0|tanjiro/combo-drop-s0"
)

base_args=(
  --agent charliepai2c5-tanjiro
  --loss_type l1
  --surf_weight 1
  --amp true
  --grad_accum 4
  --batch_size 4
  --fourier_m 160
  --fourier_sigma 0.7
  --swiglu true
  --n_layers 3
  --slice_num 16
  --n_head 1
  --n_hidden 128
  --mlp_ratio 2
  --epochs 50
)

idx=0
for entry in "${configs[@]}"; do
  IFS='|' read -r dp dr seed name <<<"$entry"
  idx=$((idx + 1))
  log_file="$LOG_DIR/${idx}_${name//\//_}.log"
  echo "==== [$(date -u +%FT%TZ)] ($idx/${#configs[@]}) start: $name dp=$dp dropout=$dr seed=$seed ====" | tee -a "$log_file"
  python train.py \
    "${base_args[@]}" \
    --drop_path_rate "$dp" \
    --dropout "$dr" \
    --seed "$seed" \
    --experiment_name "$name" \
    --wandb_group tanjiro \
    --wandb_name "$name" \
    >>"$log_file" 2>&1
  rc=$?
  echo "==== [$(date -u +%FT%TZ)] ($idx/${#configs[@]}) end: $name rc=$rc ====" | tee -a "$log_file"
  echo "$(date -u +%FT%TZ) ($idx/${#configs[@]}) $name rc=$rc" >> "$LOG_DIR/sweep_status.log"
done

echo "==== [$(date -u +%FT%TZ)] ALL DONE ====" >> "$LOG_DIR/sweep_status.log"
