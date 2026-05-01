#!/bin/bash
# Phase 6: 8 parallel ~2.5-hour runs of additional seeds for ensemble diversity.
# Use the same proven recipe as Phase 5 winner (slice_num=16 + EMA + lr=2e-3 + gcond + gradclip + OneCycle).

cd /workspace/ml-intern-benchmark/target
GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# A-H: 8 more seeds for ensemble (3,4,5,6,7,8,9,10)
for i in 0 1 2 3 4 5 6 7; do
    SEED=$((i+3))
    GPU=$i
    LETTER=$(echo "A B C D E F G H" | cut -d' ' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$GPU nohup python train.py \
        --epochs 999 --max_minutes 150 --lr_schedule_epochs 90 \
        --use_global_cond --max_grad_norm 0.1 \
        --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
        --slice_num 16 --use_ema --ema_decay 0.999 \
        --seed $SEED \
        --agent $AGENT --wandb_group $GROUP \
        --wandb_name "$GROUP/p6$LETTER-slice16-ema-2.5h-s$SEED" \
        > logs/p6$LETTER-slice16-ema-2.5h-s$SEED.log 2>&1 &
    echo "p6$LETTER seed=$SEED PID=$!"
done

date +"%H:%M:%S"
