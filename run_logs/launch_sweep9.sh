#!/bin/bash
# Sweep 9: 12h × 8 GPUs. Multi-seed of two best configs (1head, 2head)
# with the strongest regularization combo (clip1.0 + warmup10).
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

CW="--mlp_ratio 2 --slice_num 16 --n_hidden 128 --n_layers 3 --loss_type l1 --warmup_epochs 10 --grad_clip 1.0 --use_amp True --ema_decay 0.999"

declare -a runs
runs=(
  # 1-head x4 seeds
  "0|s9-1head-cw-seed10|$CW --n_head 1 --seed 10"
  "1|s9-1head-cw-seed11|$CW --n_head 1 --seed 11"
  "2|s9-1head-cw-seed12|$CW --n_head 1 --seed 12"
  "3|s9-1head-cw-seed13|$CW --n_head 1 --seed 13"
  # 2-head x4 seeds
  "4|s9-2head-cw-seed10|$CW --n_head 2 --seed 10"
  "5|s9-2head-cw-seed11|$CW --n_head 2 --seed 11"
  "6|s9-2head-cw-seed12|$CW --n_head 2 --seed 12"
  "7|s9-2head-cw-seed13|$CW --n_head 2 --seed 13"
)

> $LOGDIR/sweep9_pids.txt
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
  echo "$pid $gpu $name" >> $LOGDIR/sweep9_pids.txt
  sleep 2
done

cat $LOGDIR/sweep9_pids.txt
