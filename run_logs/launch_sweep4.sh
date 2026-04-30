#!/bin/bash
# Sweep 4: All runs use AMP + L1 + warmup. Test RFF, EMA, multiscale.
# Each 90 min. AMP enables ~2x more epochs in same time.
set -u
cd /workspace/ml-intern-benchmark/target
GROUP=mlintern-pai2-72h-v4-r3
AGENT=ml-intern-r3
LOGDIR=run_logs

# Common: AMP + L1 + 5-epoch warmup + small3l
COMMON="--n_layers 3 --n_hidden 128 --n_head 1 --slice_num 16 --mlp_ratio 2 --loss_type l1 --warmup_epochs 5 --use_amp True"

declare -a runs
runs=(
  "0|s4-amp-baseline-90|$COMMON"
  "1|s4-amp-rff1-90|$COMMON --rff_sigma 1.0 --rff_B_size 16"
  "2|s4-amp-rff2-90|$COMMON --rff_sigma 2.0 --rff_B_size 16"
  "3|s4-amp-rff4-90|$COMMON --rff_sigma 4.0 --rff_B_size 16"
  "4|s4-amp-ema-90|$COMMON --ema_decay 0.999"
  "5|s4-amp-multi-32-16-8|$COMMON --slice_nums 32,16,8"
  "6|s4-amp-rff2-ema-90|$COMMON --rff_sigma 2.0 --rff_B_size 16 --ema_decay 0.999"
  "7|s4-amp-rff2-ema-multi-90|$COMMON --rff_sigma 2.0 --rff_B_size 16 --ema_decay 0.999 --slice_nums 32,16,8"
)

> $LOGDIR/sweep4_pids.txt
for run in "${runs[@]}"; do
  IFS='|' read -r gpu name flags <<< "$run"
  log_file=$LOGDIR/${name}.log
  echo "Launching $name on GPU $gpu → $log_file"
  CUDA_VISIBLE_DEVICES=$gpu nohup python ./train.py \
    $flags \
    --epochs 200 --timeout_min 90 \
    --agent $AGENT \
    --wandb_group $GROUP \
    --wandb_name "$GROUP/$name" \
    > $log_file 2>&1 &
  pid=$!
  echo "$pid $gpu $name" >> $LOGDIR/sweep4_pids.txt
  sleep 2
done

cat $LOGDIR/sweep4_pids.txt
