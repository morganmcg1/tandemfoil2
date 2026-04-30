#!/bin/bash
# Phase 4: 8 parallel 6-hour final candidate runs
# Phase 3 winner was p3H (lr=2e-3, slice_num=64, no eidetic, gcond, gradclip 0.1, OneCycle)
# with test_avg/mae_surf_p = 37.78. Phase 4 explores variations on top of this winner.
# Schedule_epochs sized to actual expected epoch count for each config.

cd /workspace/ml-intern-benchmark/target
GROUP="mlintern-pai2-24h-v3-r5"
AGENT="ml-intern-r5"

# A: BEST p3H winner exact recipe at 6h (control)
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 140 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --seed 0 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4A-lr2e3-6h-s0" \
    > logs/p4A-lr2e3-6h-s0.log 2>&1 &
echo "p4A PID=$!"

# B: BEST + slice_num=32 + EMA (combine top winners)
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 180 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 32 --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4B-lr2e3-slice32-ema" \
    > logs/p4B-lr2e3-slice32-ema.log 2>&1 &
echo "p4B PID=$!"

# C: BEST + EMA only (separate EMA effect from slice_num)
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 140 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4C-lr2e3-ema" \
    > logs/p4C-lr2e3-ema.log 2>&1 &
echo "p4C PID=$!"

# D: BEST + n_hidden=160 + slice_num=32 (medium bigger model)
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 120 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 32 --n_hidden 160 --n_head 8 \
    --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4D-lr2e3-h160-ema" \
    > logs/p4D-lr2e3-h160-ema.log 2>&1 &
echo "p4D PID=$!"

# E: BEST + n_layers=6 + slice_num=32 (deeper)
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 120 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 32 --n_layers 6 \
    --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4E-lr2e3-l6-ema" \
    > logs/p4E-lr2e3-l6-ema.log 2>&1 &
echo "p4E PID=$!"

# F: BEST + slice_num=16 (smallest slices, fastest)
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 200 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --slice_num 16 --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4F-lr2e3-slice16-ema" \
    > logs/p4F-lr2e3-slice16-ema.log 2>&1 &
echo "p4F PID=$!"

# G: BEST + lr=3e-3 (even higher peak)
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 140 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 3e-3 --pct_start 0.05 \
    --use_ema --ema_decay 0.999 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4G-lr3e3-ema" \
    > logs/p4G-lr3e3-ema.log 2>&1 &
echo "p4G PID=$!"

# H: BEST seed 1 (variance check)
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --epochs 999 --max_minutes 360 --lr_schedule_epochs 140 \
    --use_global_cond --max_grad_norm 0.1 \
    --scheduler onecycle --lr 2e-3 --pct_start 0.05 \
    --seed 1 \
    --agent $AGENT --wandb_group $GROUP \
    --wandb_name "$GROUP/p4H-lr2e3-6h-s1" \
    > logs/p4H-lr2e3-6h-s1.log 2>&1 &
echo "p4H PID=$!"

date +"%H:%M:%S"
