#!/bin/bash
# Sweep 2: refine winner (small3l + L1), 8 parallel ~55-min runs
# Budget: ~1h, then sweep 3
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs
mkdir -p $LOGDIR

# Common: small3l small3l = n_layers 3, n_head 1, slice_num 16, hidden 128, mlp_ratio 2
COMMON_SMALL="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2"

declare -a runs
runs=(
  "0|s2-small3l-l1-60min|$COMMON_SMALL --loss_type l1"
  "1|s2-small3l-huber0.1|$COMMON_SMALL --loss_type huber --huber_delta 0.1"
  "2|s2-small3l-huber0.3|$COMMON_SMALL --loss_type huber --huber_delta 0.3"
  "3|s2-small3l-l1-bs8|$COMMON_SMALL --loss_type l1 --batch_size 8"
  "4|s2-small3l-l1-sw20|$COMMON_SMALL --loss_type l1 --surf_weight 20"
  "5|s2-small4l-l1|--n_layers 4 --n_hidden 128 --n_head 1 --slice_num 32 --mlp_ratio 2 --loss_type l1"
  "6|s2-small3l-l1-warmup|$COMMON_SMALL --loss_type l1 --warmup_epochs 5"
  "7|s2-small3l-l1-clip1|$COMMON_SMALL --loss_type l1 --grad_clip 1.0"
)

> $LOGDIR/sweep2_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 80 --timeout_min 55 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep2_pids.txt
  sleep 2
done

cat $LOGDIR/sweep2_pids.txt
