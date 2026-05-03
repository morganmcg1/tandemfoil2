#!/bin/bash
# Sweep 10: 12h × 8 GPUs. Final round for ensemble diversity.
# 4 multi-seed of 2head best + 4 explicit diversity variants.
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

CW2H="--mlp_ratio 2 --slice_num 16 --n_hidden 128 --n_layers 3 --n_head 2 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.999"

declare -a runs
runs=(
  # 4 multi-seed of 2head best (seeds 14-17)
  "0|s10-2head-cw-seed14|$CW2H --seed 14"
  "1|s10-2head-cw-seed15|$CW2H --seed 15"
  "2|s10-2head-cw-seed16|$CW2H --seed 16"
  "3|s10-2head-cw-seed17|$CW2H --seed 17"
  # 4 architectural diversity for ensemble
  "4|s10-2head-slice32|--mlp_ratio 2 --slice_num 32 --n_hidden 128 --n_layers 3 --n_head 2 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.999 --seed 0"
  "5|s10-2head-slice8|--mlp_ratio 2 --slice_num 8 --n_hidden 128 --n_layers 3 --n_head 2 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.999 --seed 0"
  "6|s10-2head-mlp3|--mlp_ratio 3 --slice_num 16 --n_hidden 128 --n_layers 3 --n_head 2 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.999 --seed 0"
  "7|s10-2head-decay9995|--mlp_ratio 2 --slice_num 16 --n_hidden 128 --n_layers 3 --n_head 2 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.9995 --seed 0"
)

> $LOGDIR/sweep10_pids.txt
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
  echo "$pid $gpu $name" >> $LOGDIR/sweep10_pids.txt
  sleep 2
done

cat $LOGDIR/sweep10_pids.txt
