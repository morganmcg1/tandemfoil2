#!/bin/bash
# Sweep 8: 12h × 8 GPUs. Architectural diversity for ensemble.
# Mix: 2 multi-seed of best config + 6 architectural variants
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

# All use AMP + L1 + warmup5 + EMA(0.999); arch differs
declare -a runs
runs=(
  # 2 multi-seed of best (seeds 8, 9)
  "0|s8-12h-seed8|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 8"
  "1|s8-12h-seed9|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 9"
  # 6 architectural variants
  "2|s8-12h-mlp4|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 4 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
  "3|s8-12h-2head|--n_layers 3 --n_hidden 128 --n_head 2 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
  "4|s8-12h-slice8|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 8 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
  "5|s8-12h-huber0.05|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type huber --huber_delta 0.05 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
  "6|s8-12h-dropout|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --dropout 0.05 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
  "7|s8-12h-clip0.5-warmup15|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 15 --use_amp True --ema_decay 0.999 --grad_clip 0.5 --seed 0"
)

> $LOGDIR/sweep8_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 1200 --timeout_min 720 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep8_pids.txt
  sleep 2
done

cat $LOGDIR/sweep8_pids.txt
