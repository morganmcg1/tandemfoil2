#!/bin/bash
# Sweep 5: AMP + L1 + warmup + EMA confirmed best. Test variations at 3h.
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

# Common: AMP + L1 + 5-epoch warmup + small3l + EMA(0.999)
BEST="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999"

declare -a runs
runs=(
  "0|s5-best-3h|$BEST"
  "1|s5-decay9995-3h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.9995"
  "2|s5-clip1-3h|$BEST --grad_clip 1.0"
  "3|s5-warmup10-3h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 10 --use_amp True --ema_decay 0.999"
  "4|s5-bs8-3h|$BEST --batch_size 8"
  "5|s5-h160-3h|--n_layers 3 --n_hidden 160 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999"
  "6|s5-slice24-3h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 24 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999"
  "7|s5-decay99-3h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.99"
)

> $LOGDIR/sweep5_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 600 --timeout_min 180 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep5_pids.txt
  sleep 2
done

cat $LOGDIR/sweep5_pids.txt
