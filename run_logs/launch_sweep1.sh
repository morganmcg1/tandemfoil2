#!/bin/bash
# Sweep 1: probe design space with 8 parallel ~25-min runs
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs
mkdir -p $LOGDIR

declare -a runs
runs=(
  # (gpu) (run_name) (extra_flags)
  "0|s1-baseline|--n_layers 5 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --loss_type mse"
  "1|s1-small3l-mse|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type mse"
  "2|s1-small3l-l1|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1"
  "3|s1-mid4l-mse|--n_layers 4 --n_hidden 128 --n_head 2 --slice_num 32 --mlp_ratio 2 --loss_type mse"
  "4|s1-small3l-pw5|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type mse --p_weight 5"
  "5|s1-small3l-sw20|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type mse --surf_weight 20"
  "6|s1-small3l-lowlr|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type mse --lr 2e-4"
  "7|s1-medium-reg|--n_layers 5 --n_hidden 128 --n_head 4 --slice_num 32 --mlp_ratio 2 --loss_type mse --dropout 0.1 --weight_decay 1e-3"
)

> $LOGDIR/sweep1_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 50 --timeout_min 25 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep1_pids.txt
  sleep 2
done

cat $LOGDIR/sweep1_pids.txt
