#!/bin/bash
# Sweep 6: 6h × 8 GPUs. Multi-seed ensemble + variations on best config.
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

# Best confirmed: AMP + L1 + warmup5 + EMA(0.999) + small3l/16slice/1head
BEST="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999"

declare -a runs
runs=(
  # Multi-seed of best config (4 runs)
  "0|s6-best-6h-seed0|$BEST --seed 0"
  "1|s6-best-6h-seed1|$BEST --seed 1"
  "2|s6-best-6h-seed2|$BEST --seed 2"
  "3|s6-best-6h-seed3|$BEST --seed 3"
  # Variants (4 runs)
  "4|s6-clip0.5-6h|$BEST --grad_clip 0.5"
  "5|s6-decay9995-6h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.9995 --seed 0"
  "6|s6-bs8-lr8e4-6h|$BEST --batch_size 8 --lr 8e-4"
  "7|s6-slice32-6h|--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 32 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True --ema_decay 0.999 --seed 0"
)

> $LOGDIR/sweep6_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 600 --timeout_min 360 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep6_pids.txt
  sleep 2
done

cat $LOGDIR/sweep6_pids.txt
