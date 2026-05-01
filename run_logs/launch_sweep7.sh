#!/bin/bash
# Sweep 7: 12h × 8 GPUs. 4 multi-seed + 4 diversity variants for ensemble.
# All AMP + L1 + warmup5 + EMA(0.999) + small3l/16slice/1head as base
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

BEST="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999"

declare -a runs
runs=(
  # 4 multi-seed of best (seeds 4, 5, 6, 7)
  "0|s7-12h-seed4|$BEST --seed 4"
  "1|s7-12h-seed5|$BEST --seed 5"
  "2|s7-12h-seed6|$BEST --seed 6"
  "3|s7-12h-seed7|$BEST --seed 7"
  # 4 variations for diversity
  "4|s7-12h-warmup15|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 15 --use_amp True --ema_decay 0.999 --seed 0"
  "5|s7-12h-lr7e4|$BEST --lr 7e-4 --seed 0"
  "6|s7-12h-lr3e4|$BEST --lr 3e-4 --seed 0"
  "7|s7-12h-clip1-warmup10|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 10 --use_amp True --ema_decay 0.999 --grad_clip 1.0 --seed 0"
)

> $LOGDIR/sweep7_pids.txt
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
  echo "$pid $gpu $name" >> $LOGDIR/sweep7_pids.txt
  sleep 2
done

cat $LOGDIR/sweep7_pids.txt
