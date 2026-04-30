#!/bin/bash
# Sweep 3: refine best (L1 + warmup) with 90-min runs and combinations.
# Goal: confirm best config + explore AMP/slice_num/depth at longer budget
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

COMMON="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5"

declare -a runs
runs=(
  "0|s3-l1-warmup-clip1-90|$COMMON --grad_clip 1.0"
  "1|s3-l1-warmup-90|$COMMON"
  "2|s3-l1-warmup-amp-90|$COMMON --use_amp True"
  "3|s3-huber0.05-warmup-90|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type huber --huber_delta 0.05 --warmup_epochs 5"
  "4|s3-l1-warmup-slice8-90|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 8 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5"
  "5|s3-l1-warmup-slice32-90|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 32 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5"
  "6|s3-l1-warmup-deeper4-90|--n_layers 4 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5"
  "7|s3-l1-warmup-h192-90|--n_layers 3 --n_hidden 192 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5"
)

> $LOGDIR/sweep3_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 130 --timeout_min 90 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep3_pids.txt
  sleep 2
done

cat $LOGDIR/sweep3_pids.txt
